// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { clampThinkingLevel, Type, type Message } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { RoutingContent, RoutingMessage } from "@switchyard/libsy";

interface NativeRoute {
  select(messagesJson: string, localOnly: boolean, taskId: string): Promise<string>;
}
const { NativeConfiguredRoute }: {
  NativeConfiguredRoute: new (toml: string, routeId: string) => NativeRoute;
} = createRequire(import.meta.url)("./native/pi_runner_spike.node");
const LOCAL_ENTRY = "switchyard.runner-spike.local-only";

// Pi keeps the original media and reasoning for its final provider request.
function routingMessages(messages: readonly Message[]): RoutingMessage[] {
  return messages.map((message) => {
    const blocks = typeof message.content === "string"
      ? [{ type: "text" as const, text: message.content }] : message.content;
    const content = blocks.flatMap<RoutingContent>((block) => {
      if (block.type === "text") return [{ type: "text", text: block.text }];
      if (block.type === "toolCall") {
        return [{ type: "tool_call", id: block.id, name: block.name, arguments: block.arguments }];
      }
      return [];
    });
    if (message.role === "toolResult") {
      return { role: "tool", content: [{
        type: "tool_result", tool_call_id: message.toolCallId, content, is_error: message.isError,
      }] };
    }
    if (message.role === "system") {
      for (const value of Object.values(message.sections ?? {})) {
        if (value) content.push({ type: "text", text: value });
      }
    }
    return { role: message.role, content };
  });
}

export default function (pi: ExtensionAPI) {
  const router = new NativeConfiguredRoute(readFileSync(
    process.env.PI_RUNNER_SPIKE_CONFIG ?? new URL("./spike.toml", import.meta.url), "utf8",
  ), "switchyard/runner-spike");
  pi.registerFlag("local-only", {
    description: "Keep this spike session on the approved local model pool",
    type: "boolean", default: false,
  });
  function localOnly(ctx: ExtensionContext): boolean {
    const retained = ctx.sessionManager.getEntries().some((entry) =>
      entry.type === "custom" && entry.customType === LOCAL_ENTRY && entry.data === true);
    const enabled = retained || pi.getFlag("local-only") === true;
    if (enabled && !retained) pi.appendEntry(LOCAL_ENTRY, true);
    ctx.ui.setStatus("runner-local-only", enabled ? "local only" : undefined);
    return enabled;
  }
  pi.on("session_start", (_event, ctx) => { localOnly(ctx); });
  pi.registerCommand("local", {
    description: "Keep this session local; a new blank session without the flag resets it",
    async handler(_args, ctx) {
      if (!ctx.isIdle()) {
        ctx.ui.notify("Wait until the agent is idle.", "warning");
        return;
      }
      if (!localOnly(ctx)) pi.appendEntry(LOCAL_ENTRY, true);
      localOnly(ctx);
    },
  });
  pi.registerCommand("direct-smoke", {
    description: "Exercise Pi's direct-call path without an agent turn",
    async handler(_args, ctx) {
      const model = ctx.modelRegistry.find("switchyard", "runner-spike");
      if (!model) throw new Error("Spike virtual model is missing");
      const response = await ctx.modelRegistry.streamSimple(model, {
        messages: [{ role: "user", content: "Reply exactly DIRECT_LOCAL_OK.", timestamp: Date.now() }],
      }, { maxTokens: 128 }).result();
      console.error("SPIKE_DIRECT " + JSON.stringify({
        provider: response.provider, model: response.model, stop_reason: response.stopReason,
        text: response.content.filter((block) => block.type === "text").map((block) => block.text).join(""),
      }));
    },
  });
  pi.registerTool({
    name: "smoke_echo", label: "Smoke echo", description: "Return TOOL_SMOKE_OK. Call only once.",
    parameters: Type.Object({}),
    async execute() {
      return { content: [{ type: "text", text: "TOOL_SMOKE_OK" }], details: { synthetic: true } };
    },
  });
  pi.registerVirtualModel({
    provider: "switchyard", id: "runner-spike", name: "Runner admission spike", thinkingLevels: ["off"],
    async route(request, ctx) {
      if (request.signal?.aborted) throw Object.assign(new Error("Routing cancelled"), { name: "AbortError" });
      const local = localOnly(ctx);
      const selected = await router.select(
        JSON.stringify(routingMessages(request.messages)), local, ctx.sessionManager.getSessionId(),
      );
      if (request.signal?.aborted) throw Object.assign(new Error("Routing cancelled"), { name: "AbortError" });
      const slash = selected.indexOf("/");
      const model = ctx.modelRegistry.find(selected.slice(0, slash), selected.slice(slash + 1));
      if (!model || model.api === "pi-virtual") throw new Error("Runner selected an unavailable physical model");
      console.error("SPIKE_ROUTE " + JSON.stringify({ reason: request.reason, local_only: local, selected }));
      return { model, thinkingLevel: clampThinkingLevel(model, request.thinkingLevel) };
    },
  });
}
