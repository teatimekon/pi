import { chmodSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	type Api,
	type AssistantMessage,
	type AssistantMessageEvent,
	type Model,
	normalizeContext,
	type SimpleStreamOptions,
	type TranscriptContext,
} from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { ExtensionAPI, ProviderConfig } from "../src/core/extensions/types.ts";
import claudeCodeExtension from "../src/extensions/claude-code/index.ts";

// Emits the stream-json lines `claude -p --verbose --include-partial-messages` prints for one
// turn that runs a Bash tool and then answers, and records its argv for the test.
const FAKE_CLAUDE = `
import { writeFileSync } from "node:fs";
const args = process.argv.slice(2);
writeFileSync(process.env.FAKE_CLAUDE_ARGS, JSON.stringify(args));
if (process.env.FAKE_CLAUDE_FAIL) {
	process.stderr.write("not logged in");
	process.exit(2);
}
const flag = args.includes("--resume") ? "--resume" : "--session-id";
const session_id = args[args.indexOf(flag) + 1];
const emit = (value) => process.stdout.write(JSON.stringify({ session_id, ...value }) + "\\n");
const inner = (event) => emit({ type: "stream_event", event });
emit({ type: "system", subtype: "init" });
inner({ type: "message_start", message: { model: "claude-sonnet-5", usage: { input_tokens: 3, output_tokens: 0 } } });
inner({ type: "content_block_start", index: 0, content_block: { type: "thinking", thinking: "" } });
inner({ type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "think" } });
inner({ type: "content_block_stop", index: 0 });
inner({ type: "content_block_start", index: 1, content_block: { type: "tool_use", id: "tool-1", name: "Bash", input: {} } });
inner({ type: "content_block_delta", index: 1, delta: { type: "input_json_delta", partial_json: '{"command":"echo hi"}' } });
inner({ type: "content_block_stop", index: 1 });
inner({ type: "message_delta", delta: { stop_reason: "tool_use" }, usage: { input_tokens: 3, output_tokens: 8 } });
emit({ type: "user", message: { role: "user", content: [{ type: "tool_result", tool_use_id: "tool-1", content: "hi", is_error: false }] } });
inner({ type: "message_start", message: { model: "claude-sonnet-5", usage: { input_tokens: 4, output_tokens: 0 } } });
inner({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
inner({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "done" } });
inner({ type: "content_block_stop", index: 0 });
inner({ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { input_tokens: 4, output_tokens: 4 } });
emit({ type: "result", result: "done", terminal_reason: "completed", usage: { input_tokens: 4, output_tokens: 4 } });
`;

describe.skipIf(process.platform === "win32")("claude-code provider", () => {
	let tempDir: string;
	let argsFile: string;
	let provider: ProviderConfig;
	let model: Model<Api>;
	const savedEnv = { ...process.env };

	beforeEach(() => {
		tempDir = mkdtempSync(join(tmpdir(), "pi-claude-code-"));
		argsFile = join(tempDir, "args.json");
		const script = join(tempDir, "fake-claude.mjs");
		const bin = join(tempDir, "claude");
		writeFileSync(script, FAKE_CLAUDE);
		writeFileSync(bin, `#!/bin/sh\nexec "${process.execPath}" "${script}" "$@"\n`);
		chmodSync(bin, 0o755);
		process.env.PI_CLAUDE_CODE_BIN = bin;
		process.env.FAKE_CLAUDE_ARGS = argsFile;
		delete process.env.FAKE_CLAUDE_FAIL;
		delete process.env.PI_CLAUDE_CODE_PERMISSION_MODE;
		delete process.env.PI_CLAUDE_CODE_TOOLS;

		const registered: ProviderConfig[] = [];
		const pi = {
			on: () => () => {},
			registerProvider: (_name: string, config: ProviderConfig) => registered.push(config),
		} as unknown as ExtensionAPI;
		claudeCodeExtension(pi);
		provider = registered[0]!;
		const config = provider.models!.find((entry) => entry.id === "claude-sonnet-5")!;
		model = { ...config, provider: "claude-code", api: provider.api!, baseUrl: provider.baseUrl! } as Model<Api>;
	});

	afterEach(() => {
		process.env = { ...savedEnv };
		rmSync(tempDir, { recursive: true, force: true });
	});

	async function run(context: TranscriptContext, options: SimpleStreamOptions) {
		const stream = provider.streamSimple!(model, context, { cwd: tempDir, ...options });
		const events: AssistantMessageEvent[] = [];
		for await (const event of stream) events.push(event);
		return { events, message: await stream.result(), args: JSON.parse(readFileSync(argsFile, "utf8")) as string[] };
	}

	const userContext = (...messages: TranscriptContext["messages"]) => normalizeContext({ messages });
	const user = (content: string) => ({ role: "user" as const, content, timestamp: Date.now() });

	it("starts a Claude session with the raw prompt, model, and effort", async () => {
		const { args } = await run(userContext(user('say "hi" $HOME')), { sessionId: "pi-session", reasoning: "high" });

		expect(args).toEqual([
			"-p",
			"--verbose",
			"--output-format",
			"stream-json",
			"--include-partial-messages",
			"--session-id",
			"pi-session",
			"--model",
			"claude-sonnet-5",
			"--effort",
			"high",
			'say "hi" $HOME',
		]);
	});

	it("streams thinking, tool calls, external tool results, and text as one assistant message", async () => {
		const { events, message } = await run(userContext(user("run it")), { sessionId: "pi-session" });

		expect(events.map((event) => event.type)).toEqual([
			"start",
			"thinking_start",
			"thinking_delta",
			"thinking_end",
			"toolcall_start",
			"toolcall_delta",
			"toolcall_end",
			"tool_result",
			"text_start",
			"text_delta",
			"text_end",
			"done",
		]);
		const toolResult = events.find((event) => event.type === "tool_result");
		expect(toolResult?.type === "tool_result" && toolResult.message).toMatchObject({
			toolCallId: "tool-1",
			toolName: "Bash",
			content: [{ type: "text", text: "hi" }],
			isError: false,
		});
		expect(message.content).toEqual([
			{ type: "thinking", thinking: "think" },
			{ type: "toolCall", id: "tool-1", name: "Bash", arguments: { command: "echo hi" } },
			{ type: "text", text: "done" },
		]);
		expect(message.stopReason).toBe("stop");
		expect(message.responseId).toBe("pi-session");
	});

	it("resumes the Claude session recorded on the previous Claude reply", async () => {
		const previous: AssistantMessage = {
			role: "assistant",
			content: [{ type: "text", text: "earlier" }],
			api: provider.api!,
			provider: "claude-code",
			model: model.id,
			responseId: "claude-session",
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: Date.now(),
		};
		const { args, message } = await run(userContext(user("first"), previous, user("continue")), {
			sessionId: "other-pi-session",
		});

		expect(args.slice(5, 7)).toEqual(["--resume", "claude-session"]);
		expect(args.at(-1)).toBe("continue");
		expect(message.responseId).toBe("claude-session");
	});

	it("reports a failed Claude process as an error with its stderr", async () => {
		process.env.FAKE_CLAUDE_FAIL = "1";
		const { message } = await run(userContext(user("hi")), { sessionId: "pi-session" });

		expect(message.stopReason).toBe("error");
		expect(message.errorMessage).toBe("not logged in");
	});
});
