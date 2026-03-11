#!/usr/bin/env node
/**
 * Mock Claude Code CLI for conformance testing.
 *
 * Implements the JSON-line protocol expected by @anthropic-ai/claude-agent-sdk:
 *   stdin  ← SDK sends control_request and user messages (JSON lines)
 *   stdout → CLI responds with control_response, assistant, and result messages
 */

import { createInterface } from "readline";

const rl = createInterface({ input: process.stdin });

for await (const line of rl) {
  if (!line.trim()) continue;

  let msg;
  try {
    msg = JSON.parse(line);
  } catch {
    continue;
  }

  if (msg.type === "control_request") {
    // Respond to every control request with success
    const resp = {
      type: "control_response",
      response: {
        subtype: "success",
        request_id: msg.request_id,
        response: {},
      },
    };
    process.stdout.write(JSON.stringify(resp) + "\n");
  } else if (msg.type === "user") {
    // Emit an assistant message followed by a result
    const assistant = {
      type: "assistant",
      message: {
        id: "msg_mock_001",
        type: "message",
        role: "assistant",
        model: "claude-sonnet-4-20250514",
        content: [{ type: "text", text: "Hello! I'm a mock Claude response." }],
        stop_reason: "end_turn",
        usage: { input_tokens: 10, output_tokens: 8 },
      },
      parent_tool_use_id: null,
      uuid: "00000000-0000-0000-0000-000000000001",
      session_id: "mock-session-001",
    };
    process.stdout.write(JSON.stringify(assistant) + "\n");

    const result = {
      type: "result",
      subtype: "success",
      duration_ms: 100,
      duration_api_ms: 50,
      is_error: false,
      num_turns: 1,
      result: "Hello! I'm a mock Claude response.",
      stop_reason: "end_turn",
      total_cost_usd: 0.001,
      usage: { input_tokens: 10, output_tokens: 8 },
      modelUsage: {},
      permission_denials: [],
    };
    process.stdout.write(JSON.stringify(result) + "\n");
  }
}
