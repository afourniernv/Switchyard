// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package switchyard

import (
	"encoding/json"
	"testing"
)

func testConfig(algorithm string) json.RawMessage {
	return json.RawMessage(`{"algorithm":` + algorithm + `,"targets":{"weak":"weak-model","strong":"strong-model","judge":"judge-model"}}`)
}

func testEngines() []Engine {
	engines := []Engine{WASM}
	if nativeAvailable() {
		engines = append(engines, Native)
	}
	return engines
}

func engineName(engine Engine) string {
	if engine == Native {
		return "native"
	}
	return "wasm"
}

func forEachEngine(t *testing.T, test func(*testing.T, Engine)) {
	t.Helper()
	for _, engine := range testEngines() {
		t.Run(engineName(engine), func(t *testing.T) { test(t, engine) })
	}
}
