// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";

const { NativeConfiguredRoute } = createRequire(import.meta.url)("./native/pi_runner_spike.node");
const config = readFileSync(new URL("./spike.toml", import.meta.url), "utf8");
const messages = JSON.stringify([{ role: "user", content: [{ type: "text", text: "Say hello." }] }]);

test("Runner admits explicit signals and keeps local-only tasks restricted", async () => {
  const route = new NativeConfiguredRoute(config, "switchyard/runner-spike");
  assert.equal(await route.select(messages, false, "public-task"), "standard/efficient");
  assert.equal(await route.select(messages, true, "sensitive-task"), "local/efficient");
  assert.equal(await route.select(messages, false, "sensitive-task"), "local/efficient");
  assert.equal(await route.select(messages, false, "another-task"), "standard/efficient");
  const failedBuild = JSON.stringify([
    ...JSON.parse(messages),
    { role: "assistant", content: [{ type: "tool_call", id: "build", name: "Bash", arguments: { command: "cargo test" } }] },
    { role: "tool", content: [{ type: "tool_result", tool_call_id: "build", is_error: true,
      content: [{ type: "text", text: "fatal runtime error: out of memory" }] }] },
  ]);
  assert.equal(await route.select(failedBuild, false, "public-failure"), "standard/capable");
  assert.equal(await route.select(failedBuild, true, "sensitive-failure"), "local/capable");
  assert.equal(await route.select(messages, false, "sensitive-failure"), "local/efficient");
  for (const invalid of [undefined, null, "false"]) {
    await assert.rejects(async () => route.select(messages, invalid, "invalid-boolean"), { code: "BooleanExpected" });
  }
  await assert.rejects(async () => route.select(messages), { code: "BooleanExpected" });
  await assert.rejects(async () => route.select(messages, false), { code: "StringExpected" });
  await assert.rejects(route.select("not json", true, "bad-json"), /ERR_INVALID_REQUEST/);
  await assert.rejects(route.select(messages, true, ""), /valid task id/);
  assert.throws(() => new NativeConfiguredRoute(config, "unknown"), /ERR_CONFIG: unknown route/);
});
