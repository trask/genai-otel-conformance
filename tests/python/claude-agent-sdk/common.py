"""Shared test infrastructure for Claude Agent SDK conformance tests."""

import os

from otel_setup import setup_otel, flush_and_shutdown

# Path to the mock CLI launcher bundled alongside this file.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MOCK_CLI_PATH = os.path.join(
    _SCRIPT_DIR,
    "mock_cli.cmd" if os.name == "nt" else "mock_cli.py",
)


async def run_agent_query():
    """Scenario: basic agent query via mock CLI."""
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        AssistantMessage,
        ResultMessage,
        TextBlock,
        query,
    )

    print("  [agent_query] basic query via mock CLI")

    # Git checkouts in CI may drop the executable bit on the Unix helper script.
    if os.name != "nt":
        os.chmod(MOCK_CLI_PATH, os.stat(MOCK_CLI_PATH).st_mode | 0o111)

    # Skip version check since mock_cli.py doesn't support -v
    os.environ["CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK"] = "1"

    options = ClaudeAgentOptions(
        cli_path=MOCK_CLI_PATH,
        max_turns=1,
        permission_mode="bypassPermissions",
    )

    async for message in query(prompt="Say hello.", options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"    -> {block.text[:60]}")
        elif isinstance(message, ResultMessage):
            print(f"    -> result: turns={message.num_turns}")


def run(title, instrument_fn):
    """Run conformance test scenarios."""
    import anyio

    print(f"=== {title} ===")

    tp, lp, mp = setup_otel()
    instrument_fn()

    anyio.run(run_agent_query)

    flush_and_shutdown(tp, lp, mp)
