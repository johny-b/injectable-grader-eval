"""The frontier models this eval is run on, and exactly how each is called.

One place, so the runner, the probe and the summary agree. All six are
reasoning models; none takes temperature/top_p, so nothing is pinned beyond
effort and the output cap.

Anthropic (claude-fable-5-1, claude-opus-5, claude-opus-5-5): Claude 5 is
always in adaptive thinking. inspect_ai 0.3.259 maps reasoning_effort="high"
to `thinking={"type": "adaptive", "display": "summarized"}` +
`output_config={"effort": "high"}`, drops temperature/top_p/top_k, never
sends budget_tokens, and streams (auto_streaming is on under thinking).
`fallback_models` is left unset, so no server-side refusal fallback is sent.

OpenAI (gpt-5.6-sol, gpt-6-sol, gpt-6-astra): function tools + reasoning are
only accepted on the Responses API, so `responses_api=True` is forced as a
model arg. reasoning_effort="high" -> `reasoning.effort`; max_tokens ->
`max_output_tokens`; reasoning_summary="auto" so a summary of the reasoning
is kept in the log.
"""

from __future__ import annotations

EFFORT = "high"
MAX_TOKENS = 32768

MODELS: dict[str, dict] = {
    "anthropic/claude-fable-5-1": {},
    "anthropic/claude-opus-5": {},
    "anthropic/claude-opus-5-5": {},
    "openai/gpt-5.6-sol": {"responses_api": True},
    "openai/gpt-6-sol": {"responses_api": True},
    "openai/gpt-6-astra": {"responses_api": True},
}


def generate_config_kwargs(model: str) -> dict:
    kw: dict = {"reasoning_effort": EFFORT, "max_tokens": MAX_TOKENS}
    if model.startswith("openai/"):
        kw["reasoning_summary"] = "auto"
    return kw


def model_args(model: str) -> dict:
    return dict(MODELS.get(model, {}))
