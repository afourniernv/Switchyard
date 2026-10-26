// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { execFileSync } from "node:child_process";
import { copyFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const native = fileURLToPath(new URL("./native/", import.meta.url));
const manifest = join(native, "Cargo.toml");
execFileSync("cargo", ["build", "--manifest-path", manifest, ...process.argv.slice(2)], {
  stdio: "inherit",
});
const metadata = JSON.parse(execFileSync("cargo", ["metadata", "--manifest-path", manifest,
  "--format-version", "1", "--no-deps", "--offline"], { encoding: "utf8" }));
const library = process.platform === "win32" ? "pi_runner_spike.dll"
  : process.platform === "darwin" ? "libpi_runner_spike.dylib" : "libpi_runner_spike.so";
copyFileSync(join(metadata.target_directory, "debug", library), join(native, "pi_runner_spike.node"));
