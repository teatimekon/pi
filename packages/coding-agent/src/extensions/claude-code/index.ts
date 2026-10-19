import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import type {
	Api,
	AssistantMessage,
	AssistantMessageEventStream,
	JsonObject,
	Model,
	SimpleStreamOptions,
	ThinkingContent,
	ToolCall,
	ToolResultMessage,
	TranscriptContext,
	Usage,
} from "@earendil-works/pi-ai";
import { createAssistantMessageEventStream, parseStreamingJson } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "../../core/extensions/types.ts";

const PROVIDER = "claude-code";
const API = "claude-code-cli";
const DEFAULT_CONTEXT_WINDOW = 200_000;
const DEFAULT_MAX_TOKENS = 32_000;
const CLAUDE_MODELS = [
	{ id: "claude-opus-4-6", name: "Claude Opus 4.6" },
	{ id: "claude-sonnet-4-6", name: "Claude Sonnet 4.6" },
	{ id: "claude-haiku-4-5", name: "Claude Haiku 4.5" },
];

type ClaudeUsage = {
	input_tokens?: number;
	output_tokens?: number;
	cache_read_input_tokens?: number;
	cache_creation_input_tokens?: number;
	output_tokens_details?: { thinking_tokens?: number };
};

type ClaudeStreamEvent = {
	type?: string;
	subtype?: string;
	session_id?: string;
	event?: {
		type?: string;
		index?: number;
		content_block?: {
			type?: string;
			id?: string;
			name?: string;
			input?: JsonObject;
			thinking?: string;
			signature?: string;
		};
		delta?: {
			type?: string;
			text?: string;
			thinking?: string;
			signature?: string;
			partial_json?: string;
			stop_reason?: string;
		};
		message?: {
			model?: string;
			usage?: {
				input_tokens?: number;
				output_tokens?: number;
				cache_read_input_tokens?: number;
				cache_creation_input_tokens?: number;
				output_tokens_details?: { thinking_tokens?: number };
			};
		};
	};
	message?: {
		model?: string;
		content?: Array<{
			type?: string;
			id?: string;
			name?: string;
			input?: JsonObject;
			thinking?: string;
			signature?: string;
		}>;
		usage?: {
			input_tokens?: number;
			output_tokens?: number;
			cache_read_input_tokens?: number;
			cache_creation_input_tokens?: number;
		};
	};
	result?: string;
	model?: string;
	usage?: {
		input_tokens?: number;
		output_tokens?: number;
		cache_read_input_tokens?: number;
		cache_creation_input_tokens?: number;
		output_tokens_details?: { thinking_tokens?: number };
	};
	stop_reason?: string;
	terminal_reason?: string;
	is_error?: boolean;
	error?: string;
};

type ActiveBlock = {
	kind: "text" | "thinking" | "tool";
	contentIndex: number;
	toolCall?: ToolCall;
	partialJson?: string;
};

function textFromContext(context: TranscriptContext): string {
	for (let index = context.messages.length - 1; index >= 0; index--) {
		const message = context.messages[index];
		if (message.role !== "user") continue;
		if (typeof message.content === "string") return message.content;
		return message.content
			.filter((block): block is { type: "text"; text: string } => block.type === "text")
			.map((block) => block.text)
			.join("\n");
	}
	return "";
}

function emptyUsage(): Usage {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function usageFromEvent(event: ClaudeStreamEvent, previous: Usage): Usage {
	const usage = (event.usage ?? event.message?.usage ?? event.event?.message?.usage) as ClaudeUsage | undefined;
	if (!usage) return previous;
	const input = usage.input_tokens ?? previous.input;
	const output = usage.output_tokens ?? previous.output;
	const cacheRead = usage.cache_read_input_tokens ?? previous.cacheRead;
	const cacheWrite = usage.cache_creation_input_tokens ?? previous.cacheWrite;
	const reasoning = usage.output_tokens_details?.thinking_tokens;
	return {
		...previous,
		input,
		output,
		cacheRead,
		cacheWrite,
		reasoning,
		totalTokens: input + output + cacheRead + cacheWrite,
	};
}

function stopReason(reason: string | undefined): "stop" | "length" | "toolUse" | "error" {
	if (reason === "max_tokens") return "length";
	if (reason === "tool_use") return "toolUse";
	if (reason === "error") return "error";
	return "stop";
}

function shellQuote(value: string): string {
	return `'${value.replaceAll("'", "'\\''")}'`;
}

function resultText(content: unknown): string {
	if (typeof content === "string") return content;
	if (Array.isArray(content)) {
		const text = content
			.filter(
				(block): block is { type: "text"; text: string } =>
					typeof block === "object" &&
					block !== null &&
					(block as { type?: unknown }).type === "text" &&
					typeof (block as { text?: unknown }).text === "string",
			)
			.map((block) => block.text)
			.join("\n");
		if (text) return text;
	}
	return JSON.stringify(content ?? "");
}

function toolResultMessage(toolCallId: string, toolName: string, content: string, isError: boolean): ToolResultMessage {
	return {
		role: "toolResult",
		toolCallId,
		toolName,
		content: [{ type: "text", text: content }],
		isError,
		timestamp: Date.now(),
	};
}

function createStream(
	model: Model<Api>,
	context: TranscriptContext,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream {
	const stream = createAssistantMessageEventStream();
	void runClaude(model, context, options, stream);
	return stream;
}

async function runClaude(
	model: Model<Api>,
	context: TranscriptContext,
	options: SimpleStreamOptions | undefined,
	stream: AssistantMessageEventStream,
): Promise<void> {
	const previousClaudeAssistant = [...context.messages]
		.reverse()
		.find(
			(message): message is AssistantMessage =>
				message.role === "assistant" && message.provider === PROVIDER && message.responseId !== undefined,
		);
	const previousClaudeSessionId = previousClaudeAssistant?.responseId;
	const sessionId = previousClaudeSessionId || options?.sessionId || randomUUID();
	let claudeSessionId = sessionId;
	const hasPreviousClaudeTurn = previousClaudeSessionId !== undefined;
	const args = ["-p", "--verbose", "--output-format", "stream-json", "--include-partial-messages"];
	const configuredTools = process.env.PI_CLAUDE_CODE_TOOLS;
	const permissionMode = process.env.PI_CLAUDE_CODE_PERMISSION_MODE;
	if (configuredTools !== undefined) args.push("--tools", configuredTools);
	if (permissionMode) args.push("--permission-mode", permissionMode);
	args.push(hasPreviousClaudeTurn ? "--resume" : "--session-id", sessionId, "--model", model.id);
	const effort = model.thinkingLevelMap?.[options?.reasoning ?? "off"] ?? options?.reasoning;
	if (effort && effort !== "off") args.push("--effort", effort);
	args.push(textFromContext(context));

	const abort = new AbortController();
	const onAbort = () => abort.abort();
	options?.signal?.addEventListener("abort", onAbort, { once: true });
	let partial: AssistantMessage | undefined;
	let usage = emptyUsage();
	let activeBlock: ActiveBlock | undefined;
	let contentIndexOffset = 0;
	let buffer = "";
	let sawStart = false;
	let terminalReason: "stop" | "length" | "toolUse" | "error" = "stop";

	const ensureStart = (reportedModel?: string) => {
		if (sawStart) return;
		partial = {
			role: "assistant",
			content: [],
			api: API,
			provider: PROVIDER,
			model: model.id,
			responseModel: reportedModel,
			usage,
			stopReason: "pending",
			timestamp: Date.now(),
		};
		stream.push({ type: "start", partial });
		sawStart = true;
	};

	const update = () => {
		if (partial) {
			partial.usage = usage;
			partial.stopReason = "pending";
		}
	};

	const processEvent = (event: ClaudeStreamEvent) => {
		if (event.session_id) {
			claudeSessionId = event.session_id;
			ensureStart(event.model);
		}
		if (event.type === "system" && event.subtype === "init") {
			ensureStart(event.model);
			return;
		}
		if (event.type === "stream_event" && event.event) {
			const inner = event.event;
			usage = usageFromEvent(inner, usage);
			if (inner.type === "message_start") {
				ensureStart(inner.message?.model);
				contentIndexOffset = partial?.content.length ?? 0;
				activeBlock = undefined;
				return;
			}
			if (inner.type === "content_block_start" && inner.content_block) {
				ensureStart();
				const index = contentIndexOffset + (inner.index ?? 0);
				const block = inner.content_block;
				if (block.type === "text") {
					activeBlock = { kind: "text", contentIndex: index };
					partial!.content[index] = { type: "text", text: "" };
					stream.push({ type: "text_start", contentIndex: index, partial: partial! });
				} else if (block.type === "thinking") {
					activeBlock = { kind: "thinking", contentIndex: index };
					partial!.content[index] = { type: "thinking", thinking: "", thinkingSignature: block.signature };
					stream.push({ type: "thinking_start", contentIndex: index, partial: partial! });
				} else if (block.type === "tool_use") {
					const toolCall: ToolCall = {
						type: "toolCall",
						id: block.id ?? `claude-tool-${index}`,
						name: block.name ?? "unknown",
						arguments: {},
					};
					activeBlock = { kind: "tool", contentIndex: index, toolCall, partialJson: "" };
					partial!.content[index] = toolCall;
					stream.push({ type: "toolcall_start", contentIndex: index, partial: partial! });
				}
				return;
			}
			if (inner.type === "content_block_delta" && inner.delta && activeBlock) {
				if (inner.delta.type === "text_delta" && activeBlock.kind === "text") {
					const delta = inner.delta.text ?? "";
					const block = partial!.content[activeBlock.contentIndex];
					if (block.type === "text") block.text += delta;
					stream.push({ type: "text_delta", contentIndex: activeBlock.contentIndex, delta, partial: partial! });
				} else if (inner.delta.type === "thinking_delta" && activeBlock.kind === "thinking") {
					const delta = inner.delta.thinking ?? "";
					const block = partial!.content[activeBlock.contentIndex];
					if (block.type === "thinking") block.thinking += delta;
					stream.push({
						type: "thinking_delta",
						contentIndex: activeBlock.contentIndex,
						delta,
						partial: partial!,
					});
				} else if (inner.delta.type === "signature_delta" && activeBlock.kind === "thinking") {
					const block = partial!.content[activeBlock.contentIndex];
					if (block.type === "thinking")
						block.thinkingSignature = `${block.thinkingSignature ?? ""}${inner.delta.signature ?? ""}`;
				} else if (inner.delta.type === "input_json_delta" && activeBlock.kind === "tool") {
					const delta = inner.delta.partial_json ?? "";
					activeBlock.partialJson = `${activeBlock.partialJson ?? ""}${delta}`;
					activeBlock.toolCall!.arguments = parseStreamingJson<JsonObject>(activeBlock.partialJson);
					stream.push({
						type: "toolcall_delta",
						contentIndex: activeBlock.contentIndex,
						delta,
						partial: partial!,
					});
				}
				update();
				return;
			}
			if (inner.type === "content_block_stop" && activeBlock) {
				if (activeBlock.kind === "text") {
					const block = partial!.content[activeBlock.contentIndex] as { type: "text"; text: string };
					stream.push({
						type: "text_end",
						contentIndex: activeBlock.contentIndex,
						content: block.text,
						partial: partial!,
					});
				} else if (activeBlock.kind === "thinking") {
					const block = partial!.content[activeBlock.contentIndex] as ThinkingContent;
					stream.push({
						type: "thinking_end",
						contentIndex: activeBlock.contentIndex,
						content: block.thinking,
						partial: partial!,
					});
				} else {
					stream.push({
						type: "toolcall_end",
						contentIndex: activeBlock.contentIndex,
						toolCall: activeBlock.toolCall!,
						partial: partial!,
					});
				}
				activeBlock = undefined;
				return;
			}
			if (inner.type === "message_delta") {
				terminalReason = stopReason(inner.delta?.stop_reason);
				usage = usageFromEvent(inner, usage);
				update();
			}
			return;
		}
		if (event.type === "user" && Array.isArray(event.message?.content)) {
			for (const block of event.message.content) {
				if (block.type !== "tool_result") continue;
				const result = block as { type: "tool_result"; tool_use_id?: string; content?: string; is_error?: boolean };
				const call = partial?.content.find((item) => item.type === "toolCall" && item.id === result.tool_use_id) as
					| ToolCall
					| undefined;
				if (!partial) return;
				const toolResult = toolResultMessage(
					result.tool_use_id ?? "unknown",
					call?.name ?? "Claude tool",
					resultText(result.content),
					result.is_error === true,
				);
				stream.push({ type: "tool_result", message: toolResult, partial });
			}
			return;
		}
		if (event.type === "result") {
			ensureStart(event.model);
			usage = usageFromEvent(event, usage);
			terminalReason = event.is_error ? "error" : stopReason(event.stop_reason ?? event.terminal_reason);
			if (partial && partial.content.length === 0 && event.result) {
				partial.content.push({ type: "text", text: event.result });
				stream.push({ type: "text_start", contentIndex: 0, partial });
				stream.push({ type: "text_delta", contentIndex: 0, delta: event.result, partial });
				stream.push({ type: "text_end", contentIndex: 0, content: event.result, partial });
			}
		}
	};

	try {
		const child = spawn(process.env.PI_CLAUDE_CODE_BIN || "claude", args, {
			cwd: options?.cwd ?? process.cwd(),
			stdio: ["ignore", "pipe", "pipe"],
			signal: abort.signal,
		});
		let stderr = "";
		child.stderr.on("data", (chunk: Buffer) => {
			stderr += chunk.toString();
		});
		child.stdout.on("data", (chunk: Buffer) => {
			buffer += chunk.toString();
			const lines = buffer.split("\n");
			buffer = lines.pop() ?? "";
			for (const line of lines) {
				if (!line.trim()) continue;
				try {
					processEvent(JSON.parse(line) as ClaudeStreamEvent);
				} catch {
					// Ignore non-JSON diagnostics; Claude's stderr remains available below.
				}
			}
		});
		const exitCode = await new Promise<number>((resolve, reject) => {
			child.once("error", reject);
			child.once("close", (code) => resolve(code ?? 1));
		});
		if (buffer.trim()) {
			try {
				processEvent(JSON.parse(buffer) as ClaudeStreamEvent);
			} catch {
				// Ignore an incomplete trailing diagnostic.
			}
		}
		ensureStart();
		if (exitCode !== 0 || abort.signal.aborted) {
			const error = abort.signal.aborted
				? "Claude Code request aborted"
				: stderr.trim() || `claude exited with code ${exitCode}`;
			partial!.stopReason = abort.signal.aborted ? "aborted" : "error";
			partial!.responseId = claudeSessionId;
			partial!.errorMessage = error;
			stream.push({ type: "error", reason: partial!.stopReason, error: partial! });
		} else {
			partial!.usage = usage;
			partial!.stopReason = terminalReason;
			partial!.responseId = claudeSessionId;
			partial!.endTurn = terminalReason === "stop";
			stream.push({ type: "done", reason: terminalReason, message: partial! });
		}
	} catch (error) {
		ensureStart();
		partial!.stopReason = abort.signal.aborted ? "aborted" : "error";
		partial!.responseId = claudeSessionId;
		partial!.errorMessage = error instanceof Error ? error.message : String(error);
		stream.push({ type: "error", reason: partial!.stopReason, error: partial! });
	} finally {
		options?.signal?.removeEventListener("abort", onAbort);
		stream.end();
	}
}

export default function (pi: ExtensionAPI) {
	pi.on("agent_end", (event, ctx) => {
		const lastAssistant = [...event.messages]
			.reverse()
			.find((message) => message.role === "assistant" && message.provider === PROVIDER && message.responseId);
		if (lastAssistant?.role === "assistant" && lastAssistant.responseId) {
			ctx.ui.setStatus(
				"claude-code",
				`resume: cd ${shellQuote(ctx.cwd)} && claude --resume ${shellQuote(lastAssistant.responseId)}`,
			);
		}
	});
	pi.registerProvider(PROVIDER, {
		name: "Claude Code",
		api: API,
		baseUrl: "claude-code://local",
		apiKey: "claude-code",
		models: CLAUDE_MODELS.map(({ id, name }) => ({
			id,
			name,
			api: API,
			input: ["text"],
			reasoning: true,
			thinkingLevelMap: {
				off: null,
				minimal: "low",
				low: "low",
				medium: "medium",
				high: "high",
				xhigh: "max",
				max: "max",
			},
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: DEFAULT_CONTEXT_WINDOW,
			maxTokens: DEFAULT_MAX_TOKENS,
		})),
		streamSimple: createStream,
	});
}
