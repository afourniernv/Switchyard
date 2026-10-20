// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package switchyard

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
	"time"
)

func TestEmbeddedWasmChecksum(t *testing.T) {
	wire, err := os.ReadFile("internal/wasm/switchyard.wasm.sha256")
	if err != nil {
		t.Fatal(err)
	}
	fields := strings.Fields(string(wire))
	if len(fields) != 2 || fields[1] != "switchyard.wasm" {
		t.Fatalf("invalid wasm checksum file: %q", wire)
	}
	want, err := hex.DecodeString(fields[0])
	if err != nil || len(want) != sha256.Size {
		t.Fatalf("invalid SHA-256 value in checksum file: %q", fields[0])
	}
	actual := sha256.Sum256(switchyardWasm)
	if !strings.EqualFold(fields[0], hex.EncodeToString(actual[:])) {
		t.Fatalf("embedded wasm SHA-256=%x, checksum file=%s", actual, fields[0])
	}
}

func TestWasmTrapIsolatesAndReplacesInstance(t *testing.T) {
	config := readRecipe(t, "stage-signal-only.json")
	backend, err := newWasmBackendWithPoolSize(config, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer backend.close()
	input, err := json.Marshal(recipeInput())
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	first, err := backend.slots[0].startRun(ctx, input)
	if err != nil {
		t.Fatal(err)
	}
	defer first.close()
	second, err := backend.slots[0].startRun(ctx, input)
	if err != nil {
		t.Fatal(err)
	}
	defer second.close()
	other, err := backend.slots[1].startRun(ctx, input)
	if err != nil {
		t.Fatal(err)
	}
	defer other.close()

	backend.slots[0].mu.Lock()
	_, allocationErr := first.gen.guest.call(ctx, "sy_alloc", uint64(^uint32(0)))
	backend.slots[0].mu.Unlock()
	if !isWasmTrap(allocationErr) {
		t.Fatalf("forced guest allocation panic returned %v", allocationErr)
	}
	first.ptr = ^uint32(0)
	if _, err := first.next(ctx); err == nil || !strings.Contains(err.Error(), "trapped") {
		t.Fatalf("forced guest trap returned %v", err)
	}
	if _, err := second.next(ctx); err == nil || !strings.Contains(err.Error(), "trapped") {
		t.Fatalf("run sharing the trapped instance returned %v", err)
	}
	step, err := other.next(ctx)
	if err != nil {
		t.Fatalf("unrelated instance failed: %v", err)
	}
	if !strings.Contains(string(step), `"type":"done"`) {
		t.Fatalf("unrelated run returned %s", step)
	}

	replacement, err := backend.slots[0].startRun(ctx, input)
	if err != nil {
		t.Fatalf("pool did not replace the trapped instance: %v", err)
	}
	defer replacement.close()
	step, err = replacement.next(ctx)
	if err != nil || !strings.Contains(string(step), `"type":"done"`) {
		t.Fatalf("replacement returned step=%s err=%v", step, err)
	}
}

func TestWasmClosedInstanceIsReplaced(t *testing.T) {
	config := readRecipe(t, "stage-signal-only.json")
	backend, err := newWasmBackendWithPoolSize(config, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer backend.close()
	input, err := json.Marshal(recipeInput())
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	closedRun, err := backend.slots[0].startRun(ctx, input)
	if err != nil {
		t.Fatal(err)
	}
	defer closedRun.close()
	otherRun, err := backend.slots[1].startRun(ctx, input)
	if err != nil {
		t.Fatal(err)
	}
	defer otherRun.close()
	if err := closedRun.gen.guest.module.Close(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := closedRun.next(ctx); err == nil {
		t.Fatal("call on a closed instance succeeded")
	}
	step, err := otherRun.next(ctx)
	if err != nil || !strings.Contains(string(step), `"type":"done"`) {
		t.Fatalf("unrelated instance returned step=%s err=%v", step, err)
	}
	replacement, err := backend.slots[0].startRun(ctx, input)
	if err != nil {
		t.Fatalf("closed instance was not replaced: %v", err)
	}
	defer replacement.close()
	step, err = replacement.next(ctx)
	if err != nil || !strings.Contains(string(step), `"type":"done"`) {
		t.Fatalf("replacement returned step=%s err=%v", step, err)
	}
}

// A caller's expired context must not close an instance that other runs share.
func TestWasmCancelledRunKeepsSharedInstance(t *testing.T) {
	backend, err := newWasmBackendWithPoolSize(json.RawMessage(`{"algorithm":`+composite+`,"targets":{"weak":"weak-model","strong":"strong-model","judge":"judge-model"}}`), 1)
	if err != nil {
		t.Fatal(err)
	}
	defer backend.close()

	entered := make(chan struct{})
	release := make(chan struct{})
	result := make(chan error, 1)
	go func() {
		_, err := backend.route(context.Background(), input(), func(ctx context.Context, _ Work) (Reply, error) {
			close(entered)
			select {
			case <-release:
				return Reply{Response: json.RawMessage(judgeAnswer)}, nil
			case <-ctx.Done():
				return Reply{}, ctx.Err()
			}
		})
		result <- err
	}()
	<-entered

	large := input()
	large.Request = json.RawMessage(`{"messages":[{"role":"user","content":[{"type":"text","text":"` + strings.Repeat("x", 32<<20) + `"}]}]}`)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Millisecond)
	defer cancel()
	if _, err := backend.route(ctx, large, nil); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expired run returned %v", err)
	}

	close(release)
	if err := <-result; err != nil {
		t.Fatalf("run sharing the instance failed: %v", err)
	}
}

func TestWasmTrapIncludesGuestPanicMessage(t *testing.T) {
	backend, err := newWasmBackendWithPoolSize(readRecipe(t, "stage-signal-only.json"), 1)
	if err != nil {
		t.Fatal(err)
	}
	defer backend.close()
	slot := backend.slots[0]
	slot.mu.Lock()
	_, err = slot.gen.guest.call(context.Background(), "sy_alloc", uint64(^uint32(0)))
	slot.mu.Unlock()
	if !isWasmTrap(err) || !strings.Contains(err.Error(), "panicked") {
		t.Fatalf("guest panic returned %v", err)
	}
}

// Reading while a host call is unanswered must report a pending step, not block or trap.
func TestWasmNextWhileHostCallPending(t *testing.T) {
	backend, err := newWasmBackendWithPoolSize(json.RawMessage(`{"algorithm":`+composite+`,"targets":{"weak":"weak-model","strong":"strong-model","judge":"judge-model"}}`), 1)
	if err != nil {
		t.Fatal(err)
	}
	defer backend.close()
	data, err := json.Marshal(input())
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	run, err := backend.slots[0].startRun(ctx, data)
	if err != nil {
		t.Fatal(err)
	}
	defer run.close()
	step, err := run.next(ctx)
	if err != nil || !strings.Contains(string(step), `"type":"call_model"`) {
		t.Fatalf("first step=%s err=%v", step, err)
	}
	if step, err := run.next(ctx); err != nil || step != nil {
		t.Fatalf("read with an unanswered call returned step=%s err=%v", step, err)
	}
}

func TestWasmTranslatorIsReplacedAfterFailure(t *testing.T) {
	input := []byte(`{"format":"openai_chat","body":{"messages":[{"role":"user","content":"hi"}]}}`)
	if _, err := wasmTranslate("sy_decode_request", input); err != nil {
		t.Fatal(err)
	}
	translator := defaultWasmTranslator.value
	translator.mu.Lock()
	err := translator.guest.module.Close(context.Background())
	translator.mu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := wasmTranslate("sy_decode_request", input); err == nil {
		t.Fatal("translation on a closed instance succeeded")
	}
	if _, err := wasmTranslate("sy_decode_request", input); err != nil {
		t.Fatalf("translator was not replaced: %v", err)
	}
}

func TestWasmInstanceMemory(t *testing.T) {
	config := readRecipe(t, "stage-signal-only.json")
	backend, err := newWasmBackendWithPoolSize(config, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer backend.close()
	first := backend.slots[0].gen.guest.module.Memory().Size()
	second := backend.slots[1].gen.guest.module.Memory().Size()
	if first == 0 || second != first {
		t.Fatalf("guest memory sizes are %d and %d", first, second)
	}
	t.Logf("guest_linear_memory_bytes_per_instance=%d", first)
}
