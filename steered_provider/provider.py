"""Inspect model provider for a vLLM server with per-request activation steering.

Models are addressed as ``steered/<served-model-name>`` and both halves of the
condition are model args::

    inspect eval task.py --model steered/qwen -M steer_vector=0007 -M steer_strength=1.0

which travel to the server as per-request vLLM extra args::

    {"vllm_xargs": {"steer_vector": "0007", "steer_strength": 1.0}}

The vectors themselves live server side, one directory each, and the server
lists what it holds at ``GET /steering/vectors``. The client sends an id and a
scalar, so a sweep over vectors and a sweep over strengths are the same kind of
loop and both are recorded in the eval log.

The two arguments go together: the server rejects one without the other with a
400, because a strength with no vector is a request that means to steer and
would generate from the base model instead. That pairing is checked here too, at
construction, so a mistyped sweep fails before the first sample rather than on
every one of them.

Strength is in the unit the vectors' metadata calls the relative perturbation:
``1.0`` adds a delta the size of a typical residual-stream row at the vector's
layer, whichever vector was named. A request with neither arg is unsteered.

A steered request also carries ``cache_salt``, naming the condition, so that
vLLM's prefix cache cannot serve one condition's KV blocks to another. See the
comment in :meth:`SteeredAPI.completion_params`.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelOutput,
    modelapi,
)
from inspect_ai.model._model_call import ModelCall
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.tool import ToolCall, ToolChoice, ToolInfo
from inspect_ai.tool._tool_call import ToolCallContent  # noqa: F401  (typing only)
from typing_extensions import override

from steering_vectors import vectorfmt

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"

XARGS_FIELD = "vllm_xargs"
VECTOR_ARG = "steer_vector"
STRENGTH_ARG = "steer_strength"
TEMPLATE_KWARGS_FIELD = "chat_template_kwargs"
CACHE_SALT_FIELD = "cache_salt"


def _as_bool(value: bool | str) -> bool:
    """Coerce a CLI-supplied (``-M key=value``) or Python value to a bool."""
    if isinstance(value, bool):
        return value
    low = str(value).strip().lower()
    if low in ("true", "1", "yes", "on"):
        return True
    if low in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"cannot interpret {value!r} as a boolean")


# --------------------------------------------------------- Kimi tool calling
#
# The Modal server this talks to is a plain `vllm serve` WITHOUT
# `--enable-auto-tool-choice` / `--tool-call-parser`, because the two evals it
# was brought up for (ctfish, agentic-misalignment) drive the model through a
# text protocol and never send `tools`. This eval does send them, and vLLM then
# answers HTTP 400 to every tool_choice it can be given:
#
#     "auto"      -> requires --enable-auto-tool-choice and --tool-call-parser
#     "required"  -> requires --tool-call-parser
#     {"function"}-> requires --tool-call-parser
#     "none"      -> 200 OK
#
# `tool_choice: "none"` is accepted and -- this is the part that makes the
# whole thing work -- vLLM still passes `tools` to the chat template, so the
# model is prompted with its NATIVE tool-calling format and answers in it. The
# only missing piece is server-side parsing, which is what this does, in the
# same format vLLM's own `kimi_k2` tool parser reads:
#
#     <|tool_calls_section_begin|>
#       <|tool_call_begin|> functions.NAME:IDX
#       <|tool_call_argument_begin|> {json arguments} <|tool_call_end|>
#     <|tool_calls_section_end|>
#
# (the detokenizer puts spaces around the special tokens, hence the \s* here.)
#
# The alternative, inspect's own `emulate_tools=True`, was NOT taken: it drops
# `tools` from the request and describes them in prose inside the prompt
# instead, which is a format the model was not trained on. Parsing the native
# format keeps the model on the distribution it was post-trained for, and it is
# applied identically in both conditions, so it cannot be what separates them.

_KIMI_SECTION = re.compile(
    r"<\|tool_calls_section_begin\|>(?P<body>.*?)"
    r"(?:<\|tool_calls_section_end\|>|\Z)",
    re.S,
)
_KIMI_CALL = re.compile(
    r"<\|tool_call_begin\|>\s*(?P<id>[^\s<|]+)\s*"
    r"<\|tool_call_argument_begin\|>\s*(?P<args>.*?)\s*"
    r"(?:<\|tool_call_end\|>|\Z)",
    re.S,
)
# `functions.bash:0` -> bash. The whole token stays as the call id: it carries
# an index, so two calls in one message keep distinct ids, which is what the
# tool-result messages are matched back on.
_KIMI_NAME = re.compile(r"^(?:functions?\.)?(?P<name>[A-Za-z0-9_.-]+?)(?::\d+)?$")


def parse_kimi_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Split Kimi's native tool-call markup out of a completion.

    Returns the text with every tool-call section removed, and the calls.
    Malformed JSON is reported through `ToolCall.parse_error` rather than
    raised: inspect surfaces that to the model as a tool error and the episode
    continues, which is the same thing a server-side parser would do, and much
    better than losing the rollout.
    """
    calls: list[ToolCall] = []
    for section in _KIMI_SECTION.finditer(text):
        for m in _KIMI_CALL.finditer(section.group("body")):
            raw_id = m.group("id")
            name_m = _KIMI_NAME.match(raw_id)
            name = name_m.group("name") if name_m else raw_id
            raw_args = m.group("args").strip()
            arguments: dict[str, Any] = {}
            parse_error: str | None = None
            try:
                parsed = json.loads(raw_args) if raw_args else {}
                if isinstance(parsed, dict):
                    arguments = parsed
                else:
                    parse_error = (
                        f"tool arguments must be a JSON object, got "
                        f"{type(parsed).__name__}: {raw_args[:200]}"
                    )
            except json.JSONDecodeError as ex:
                parse_error = f"could not parse tool arguments as JSON: {ex}"
            calls.append(
                ToolCall(
                    id=raw_id,
                    function=name,
                    arguments=arguments,
                    parse_error=parse_error,
                    type="function",
                )
            )
    cleaned = _KIMI_SECTION.sub("", text).strip()
    return cleaned, calls


class SteeredAPI(OpenAICompatibleAPI):
    """OpenAI-compatible provider carrying a per-request vector and strength."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        steer_vector: str | int | None = None,
        steer_strength: float | str | None = None,
        enable_thinking: bool | str | None = None,
        kimi_tool_calls: bool | str = False,
        **model_args: Any,
    ) -> None:
        # All three are consumed here rather than forwarded: whatever remains in
        # model_args reaches AsyncOpenAI(), which raises on unexpected keywords.
        if steer_vector is None:
            self.steer_vector: str | None = None
        else:
            # Normalised to the four-digit form the server addresses vectors by,
            # so `-M steer_vector=7` and `-M steer_vector=0007` are one
            # condition in the eval log rather than two.
            self.steer_vector = vectorfmt.vector_id(steer_vector)

        if steer_strength is None:
            self.steer_strength: float | None = None
        else:
            try:
                self.steer_strength = float(steer_strength)
            except (TypeError, ValueError) as ex:
                raise ValueError(
                    f"steer_strength must be a number, got {steer_strength!r}"
                ) from ex

        if (self.steer_vector is None) != (self.steer_strength is None):
            raise ValueError(
                f"steer_vector and steer_strength go together: got "
                f"steer_vector={self.steer_vector!r}, "
                f"steer_strength={self.steer_strength!r}. A strength with no "
                f"vector has nothing to apply and a vector with no strength "
                f"applies it at 0; either way the eval would run against the "
                f"unsteered model under a name that says otherwise. Pass both, "
                f"or neither for the base model."
            )

        self.enable_thinking = (
            None if enable_thinking is None else _as_bool(enable_thinking)
        )

        # Client-side parsing of the model's native tool-call markup, for a
        # server with no --tool-call-parser. See parse_kimi_tool_calls().
        self.kimi_tool_calls = _as_bool(kimi_tool_calls)

        super().__init__(
            model_name=model_name,
            base_url=base_url,
            # The server does not authenticate, but AsyncOpenAI requires a key.
            api_key=api_key or os.environ.get("STEERED_API_KEY") or "EMPTY",
            config=config,
            service="steered",
            service_base_url=DEFAULT_BASE_URL,
            api_key_var="STEERED_API_KEY",
            **model_args,
        )

    @override
    def completion_params(self, config: GenerateConfig, tools: bool) -> dict[str, Any]:
        params = super().completion_params(config, tools)
        if self.steer_vector is None and self.enable_thinking is None:
            return params

        # Merge rather than replace: the base class may already have populated
        # extra_body from config.extra_body and prompt_logprobs.
        extra_body: dict[str, Any] = dict(params.get("extra_body") or {})

        if self.steer_vector is not None:
            xargs: dict[str, Any] = dict(extra_body.get(XARGS_FIELD) or {})
            xargs[VECTOR_ARG] = self.steer_vector
            xargs[STRENGTH_ARG] = self.steer_strength
            extra_body[XARGS_FIELD] = xargs
            # KIMI CHANGE (three lines, and the reason is a correctness bug).
            #
            # vLLM's prefix-cache block hash is built from the token ids plus
            # `generate_block_hash_extra_keys`, which contributes the LoRA name,
            # the multimodal inputs, `cache_salt` and prompt-embeds keys -- and
            # NOT `SamplingParams.extra_args`, which is where the two steering
            # arguments live (vllm/v1/core/kv_cache_utils.py:583, read against
            # vLLM 0.29.0). So on a server with prefix caching on, the KV blocks
            # a strength-0 run leaves behind are HIT by a strength-1.0 run of the
            # same prompt, and the steering then applies only to the tokens that
            # missed the cache. The effect grows with the shared prefix and is
            # invisible in the output: the steering simply gets weaker.
            #
            # `cache_salt` IS in that hash, so naming the condition in it makes
            # each condition's cache disjoint while keeping the reuse WITHIN a
            # condition -- which is what makes a long-context eval affordable.
            # The alternative is `--no-enable-prefix-caching`, which is correct
            # and much slower; see STEERING_NOTES.md section 9.
            extra_body.setdefault(
                CACHE_SALT_FIELD,
                f"steer:{self.steer_vector}:{self.steer_strength!r}",
            )

        if self.enable_thinking is not None:
            template_kwargs: dict[str, Any] = dict(
                extra_body.get(TEMPLATE_KWARGS_FIELD) or {}
            )
            template_kwargs["enable_thinking"] = self.enable_thinking
            extra_body[TEMPLATE_KWARGS_FIELD] = template_kwargs

        params["extra_body"] = extra_body
        return params

    @override
    def resolve_tools(
        self, tools: list[ToolInfo], tool_choice: ToolChoice, config: GenerateConfig
    ) -> tuple[list[ToolInfo], ToolChoice, GenerateConfig]:
        """Force `tool_choice: "none"` when the client is doing the parsing.

        `tools` is deliberately left ALONE -- it still goes on the wire, which
        is what makes vLLM render the native tool-calling template. Only the
        tool_choice is overridden, because every other value is a 400 on a
        server without --tool-call-parser.

        `none` is a lie to the server and the truth about the server: it does
        not mean "no tools available" to the chat template, it means "do not
        try to parse or constrain the output", which is exactly the division of
        labour here.
        """
        if self.kimi_tool_calls and tools:
            tool_choice = "none"
        return tools, tool_choice, config

    @override
    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput | tuple[ModelOutput | Exception, ModelCall]:
        result = await super().generate(input, tools, tool_choice, config)
        if not (self.kimi_tool_calls and tools):
            return result

        output = result[0] if isinstance(result, tuple) else result
        if isinstance(output, ModelOutput):
            self._extract_tool_calls(output)
        return result

    def _extract_tool_calls(self, output: ModelOutput) -> None:
        """Move Kimi's tool-call markup out of the text and into tool_calls.

        Mutates in place, because this runs between the base class building the
        ModelOutput and inspect logging it -- so the eval log shows the parsed
        call rather than the raw markup, exactly as a server-side parser would.
        """
        from inspect_ai._util.content import ContentText

        for choice in output.choices:
            msg = choice.message
            calls: list[ToolCall] = []

            if isinstance(msg.content, str):
                cleaned, found = parse_kimi_tool_calls(msg.content)
                if found:
                    msg.content = cleaned
                calls += found
            elif isinstance(msg.content, list):
                for i, part in enumerate(msg.content):
                    # Only the TEXT parts. Reasoning is a separate content type
                    # and must be left untouched: the markup the model emits in
                    # its thinking is the model talking about a call, not making
                    # one, and lifting it out would invent tool calls.
                    if isinstance(part, ContentText):
                        cleaned, found = parse_kimi_tool_calls(part.text)
                        if found:
                            msg.content[i] = ContentText(text=cleaned)
                        calls += found

            if calls:
                msg.tool_calls = (msg.tool_calls or []) + calls
                # Without this, inspect reads the choice as a normal stop and
                # the react loop treats a tool call as the final answer.
                choice.stop_reason = "tool_calls"

    @override
    def connection_key(self) -> str:
        # Scope adaptive concurrency per condition. The base class keys on
        # (api_key, model), which is identical across a sweep, so one pool would
        # be tuned by whichever condition happens to run fastest. The vector is
        # in the key as well as the strength: two vectors at one strength are
        # two different interventions and need not generate at the same rate.
        return f"steered:{self.model_name}:{self.steer_vector}:{self.steer_strength}"


@modelapi(name="steered")
def steered() -> type[SteeredAPI]:
    """Register the `steered` provider."""
    return SteeredAPI
