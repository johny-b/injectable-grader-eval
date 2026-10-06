"""Tool-call round trip through ige's `steered` provider, both conditions."""
import asyncio, json, sys, os
sys.path.insert(0, "/work/workspace/ige")
os.environ["STEERED_BASE_URL"] = "http://127.0.0.1:18001/v1"
from inspect_ai.model import get_model, GenerateConfig, ChatMessageUser, ChatMessageTool, ChatMessageSystem
from inspect_ai.tool import ToolInfo, ToolParams
from inspect_ai.util._json import JSONSchema
import steered_provider.provider  # registers `steered`
sys.path.insert(0, "/work/workspace/ige/scripts")
from run_qwen import generate_kwargs, CONTEXT_WINDOW, CONTEXT_MARGIN

bash = ToolInfo(name="bash", description="Use this function to execute bash commands.",
                parameters=ToolParams(properties={"command": JSONSchema(type="string", description="The bash command to execute.")}, required=["command"]))

async def run(steer):
    margs = {"kimi_tool_calls": False, "enable_thinking": True, "context_window": CONTEXT_WINDOW,
             "context_margin": CONTEXT_MARGIN, "cap_log": "/work/workspace/qwen_serve/verify_cap_log.jsonl"}
    if steer: margs["steer_vector"], margs["steer_strength"] = steer
    gen = generate_kwargs(); gen["max_tokens"] = 4096
    m = get_model("steered/Qwen/Qwen3.6-27B", config=GenerateConfig(**gen), memoize=False, **margs)
    msgs = [ChatMessageUser(content="What is the hostname of this machine? Use the bash tool to find out, then tell me the answer.")]
    out = await m.generate(msgs, tools=[bash])
    msg = out.message
    print(f"[{steer}] turn1 stop={out.stop_reason} tool_calls={[ (t.function, t.arguments, t.parse_error) for t in (msg.tool_calls or [])]}")
    print(f"   reasoning chars={sum(len(c.reasoning) for c in msg.content if getattr(c,'type','')=='reasoning') if isinstance(msg.content, list) else 0} text={msg.text[:120]!r}")
    assert msg.tool_calls, "no structured tool call"
    msgs += [msg, ChatMessageTool(content="sandbox-7f3a\n", tool_call_id=msg.tool_calls[0].id, function="bash")]
    out2 = await m.generate(msgs, tools=[bash])
    print(f"[{steer}] turn2 stop={out2.stop_reason} text={out2.message.text[:200]!r} tool_calls={out2.message.tool_calls}")
    # what went on the wire
    print("   request extra keys:", {k: out2_req for k, out2_req in []} or "")
    print("   cap_stats:", m.api.cap_stats)
    return out, out2

async def main():
    for steer in (None, ("0007", 0.3)):
        await run(steer)
asyncio.run(main())
