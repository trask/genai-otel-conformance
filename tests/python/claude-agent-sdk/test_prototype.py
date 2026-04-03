"""Conformance test: prototype instrumentation for Claude Agent SDK."""

import os

from opentelemetry import trace

from otel_setup import flush_and_shutdown, setup_otel

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MOCK_CLI_PATH = os.path.join(
    _SCRIPT_DIR,
    "mock_cli.cmd" if os.name == "nt" else "mock_cli.py",
)

_prototype_tracer = trace.get_tracer("gen_ai.prototype")


async def run_agent_query_prototype():
    """Scenario: basic agent query via mock CLI with prototype instrumentation."""
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    print("  [agent_query] basic query via mock CLI (prototype)")

    if os.name != "nt":
        os.chmod(MOCK_CLI_PATH, os.stat(MOCK_CLI_PATH).st_mode | 0o111)
    os.environ["CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK"] = "1"

    options = ClaudeAgentOptions(
        cli_path=MOCK_CLI_PATH,
        max_turns=1,
        permission_mode="bypassPermissions",
    )

    with _prototype_tracer.start_as_current_span("chat claude") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "anthropic")
        async for message in query(prompt="Say hello.", options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"    -> {block.text[:60]}")
            elif isinstance(message, ResultMessage):
                print(f"    -> result: turns={message.num_turns}")


def main():
    import anyio

    print("=== Prototype: Claude Agent SDK Conformance Test ===")

    tp, lp, mp = setup_otel()

    anyio.run(run_agent_query_prototype)

    flush_and_shutdown(tp, lp, mp)


if __name__ == "__main__":
    main()
