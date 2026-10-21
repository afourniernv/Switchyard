#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
wasm_dir="$repo_root/bindings/go/internal/wasm"
wasm_path="$wasm_dir/switchyard.wasm"
toolchain="$(sed -n 's/^channel = "\(.*\)"$/\1/p' "$repo_root/rust-toolchain.toml")"
[[ -n "$toolchain" ]] || {
	printf 'rust-toolchain.toml has no channel\n' >&2
	exit 1
}
cargo_home="${CARGO_HOME:-${HOME}/.cargo}"

mkdir -p "$wasm_dir"
cd "$repo_root"
rustup target add wasm32-wasip1 --toolchain "$toolchain"
# With rust-src installed, rustc embeds local std paths. Map them to the /rustc/<commit>
# prefix that toolchains without rust-src use, so the blob holds no local paths.
rust_src="$(rustc "+$toolchain" --print sysroot)/lib/rustlib/src/rust"
rust_commit="$(rustc "+$toolchain" -vV | sed -n 's/^commit-hash: //p')"
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }--remap-path-prefix=$repo_root=/switchyard --remap-path-prefix=$cargo_home=/cargo --remap-path-prefix=$rust_src=/rustc/$rust_commit"
cargo "+$toolchain" build --locked --release --target wasm32-wasip1 -p switchyard-ffi --lib
cp "$repo_root/target/wasm32-wasip1/release/switchyard_ffi.wasm" "$wasm_path"
if command -v sha256sum >/dev/null; then
	sha256sum "$wasm_path" | awk '{ print $1 "  switchyard.wasm" }' > "$wasm_path.sha256"
else
	shasum -a 256 "$wasm_path" | awk '{ print $1 "  switchyard.wasm" }' > "$wasm_path.sha256"
fi
"$script_dir/go-wasm-inputs.sh" > "$wasm_path.inputs"

if [[ "${WASM_OPT:-0}" == "1" ]]; then
	command -v wasm-opt >/dev/null || {
		printf 'WASM_OPT=1 requires wasm-opt in PATH\n' >&2
		exit 1
	}
	wasm-opt -Oz "$wasm_path" -o "$wasm_dir/switchyard.opt.wasm"
fi
