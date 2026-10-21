#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Prints a hash of every tracked input to the Go wasm blob. Blob bytes differ by build
# host, so CI compares this hash with switchyard.wasm.inputs to find a stale blob.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo_root"

crates=()
while IFS= read -r crate; do
	crates+=("$crate")
done < <(
	cargo tree --locked -p switchyard-ffi --target wasm32-wasip1 -e normal,build --prefix none --format '{p}' |
		sed -n "s|.*(${repo_root}/\(crates/[^)]*\)).*|\1|p" | sort -u
)
[[ ${#crates[@]} -gt 0 ]] || {
	printf 'found no workspace crates for switchyard-ffi\n' >&2
	exit 1
}

hash() {
	if command -v sha256sum >/dev/null; then sha256sum; else shasum -a 256; fi
}

git ls-files -- Cargo.toml Cargo.lock rust-toolchain.toml scripts/build-go-wasm.sh scripts/go-wasm-inputs.sh "${crates[@]}" |
	LC_ALL=C sort |
	while IFS= read -r file; do
		printf '%s  %s\n' "$(hash < "$file" | cut -d' ' -f1)" "$file"
	done |
	hash | cut -d' ' -f1
