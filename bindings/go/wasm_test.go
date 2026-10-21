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
	"reflect"
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
	config := testConfig(`{"type":"stage_router","picker":"efficient_first",` + stage + `}`)
	backend, err := newWasmBackendWithPoolSize(config, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer backend.close()
	input, err := json.Marshal(input())
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
	config := testConfig(`{"type":"stage_router","picker":"efficient_first",` + stage + `}`)
	backend, err := newWasmBackendWithPoolSize(config, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer backend.close()
	input, err := json.Marshal(input())
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
	backend, err := newWasmBackendWithPoolSize(testConfig(`{"type":"stage_router","picker":"efficient_first",`+stage+`}`), 1)
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
	config := testConfig(`{"type":"stage_router","picker":"efficient_first",` + stage + `}`)
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

// Host failures must retain the error policy configured in libsy on both engines.
func TestJudgeFailOpen(t *testing.T) {
	decision := `{"type":"composite","classifier":{"target":"judge","classify_trigger":"user_turn","fail_open":false,"decision":{"cutoff":0.5,"candidates":{"a":"strong","b":"weak"},"evidence":{}}},"stage":{` + stage + `}}`
	forEachEngine(t, func(t *testing.T, engine Engine) {
		for _, algorithm := range []string{composite, decision} {
			for _, failOpen := range []bool{false, true} {
				configured := algorithm
				if failOpen {
					configured = strings.Replace(algorithm, `"fail_open":false`, `"fail_open":true`, 1)
				}
				router := newRouter(t, engine, configured)
				calls := 0
				outcome, err := router.Route(context.Background(), input(), func(context.Context, Work) (Reply, error) {
					calls++
					return Reply{}, errors.New("host unavailable")
				})
				if calls != 1 {
					t.Fatalf("fail_open=%t made %d host calls", failOpen, calls)
				}
				if failOpen {
					if err != nil || len(outcome.SelectedModelIDs) == 0 {
						t.Fatalf("fail_open=true outcome=%+v err=%v", outcome, err)
					}
				} else if err == nil || !strings.Contains(err.Error(), "host unavailable") {
					t.Fatalf("fail_open=false returned %v", err)
				}
			}
		}
	})
}

// Native and WASM produce the same candidates, prepared request, and answer.
func TestBackendParity(t *testing.T) {
	if !nativeAvailable() {
		t.Skip("native parity runs with -tags switchyard_native")
	}
	for _, algorithm := range []string{
		`{"type":"random","targets":["weak"],"seed":5}`,
		`{"type":"stage_router","picker":"efficient_first",` + stage + `}`,
		composite,
	} {
		var native Outcome
		for _, engine := range []Engine{Native, WASM} {
			router := newRouter(t, engine, algorithm)
			outcome, err := router.Route(context.Background(), input(), func(context.Context, Work) (Reply, error) {
				return Reply{Response: json.RawMessage(judgeAnswer)}, nil
			})
			if err != nil {
				t.Fatal(err)
			}
			if engine == Native {
				native = outcome
				continue
			}
			if !reflect.DeepEqual(outcome.SelectedModelIDs, native.SelectedModelIDs) ||
				string(outcome.Request) != string(native.Request) || !reflect.DeepEqual(outcome.Response, native.Response) {
				t.Fatalf("backend outcomes differ: native=%+v wasm=%+v", native, outcome)
			}
		}
	}
}

// Sessionless requests sharing libsy's message key must retain one judge verdict.
func TestSessionlessRetentionParity(t *testing.T) {
	if !nativeAvailable() {
		t.Skip("native parity runs with -tags switchyard_native")
	}
	algorithm := strings.Replace(composite, `"classify_trigger":"user_turn"`, `"classify_trigger":"new_session","message_hash_fallback":true`, 1)
	config := testConfig(algorithm)
	judgeCalls := func(backend engineBackend) int {
		defer backend.close()
		calls := 0
		for range 8 {
			_, err := backend.route(context.Background(), input(), func(context.Context, Work) (Reply, error) {
				calls++
				return Reply{Response: json.RawMessage(judgeAnswer)}, nil
			})
			if err != nil {
				t.Fatal(err)
			}
		}
		return calls
	}
	native, err := newNativeBackend(config)
	if err != nil {
		t.Fatal(err)
	}
	want := judgeCalls(native)
	for _, size := range []int{1, 2, 4} {
		wasm, err := newWasmBackendWithPoolSize(config, size)
		if err != nil {
			t.Fatal(err)
		}
		if got := judgeCalls(wasm); got != want {
			t.Errorf("pool size %d made %d judge calls, native made %d", size, got, want)
		}
	}
}

// Composite's message fallback retains a verdict when a child lacks an agent ID.
func TestWasmSubagentMessageRetention(t *testing.T) {
	algorithm := strings.Replace(composite, `"classify_trigger":"user_turn"`, `"classify_trigger":"new_session","message_hash_fallback":true`, 1)
	for _, size := range []int{1, 2} {
		backend, err := newWasmBackendWithPoolSize(testConfig(algorithm), size)
		if err != nil {
			t.Fatal(err)
		}
		request := input()
		request.Metadata = &RequestMetadata{SessionID: "parent", IsSubagent: true}
		calls := 0
		for range 4 {
			_, err := backend.route(context.Background(), request, func(context.Context, Work) (Reply, error) {
				calls++
				return Reply{Response: json.RawMessage(judgeAnswer)}, nil
			})
			if err != nil {
				backend.close()
				t.Fatal(err)
			}
		}
		backend.close()
		if calls != 1 {
			t.Errorf("pool size %d made %d judge calls, want 1", size, calls)
		}
	}
}

// A routing deadline must expire while its host call is still pending.
func TestRoutingDeadline(t *testing.T) {
	if !nativeAvailable() {
		t.Skip("deadline parity runs with -tags switchyard_native")
	}
	for _, engine := range []Engine{Native, WASM} {
		t.Run(engineName(engine), func(t *testing.T) {
			router := newRouter(t, engine, `{"type":"vgr","local_target":"weak","cloud_target":"strong","mode":"evaluate","deadline_seconds":0.02,"task_typing":false}`)
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			hostFinished := make(chan struct{})
			started := time.Now()
			outcome, err := router.Route(ctx, input(), func(ctx context.Context, _ Work) (Reply, error) {
				defer close(hostFinished)
				delay := time.NewTimer(400 * time.Millisecond)
				defer delay.Stop()
				select {
				case <-delay.C:
					return Reply{Response: json.RawMessage(judgeAnswer)}, nil
				case <-ctx.Done():
					return Reply{}, ctx.Err()
				}
			})
			elapsed := time.Since(started)
			t.Logf("routing deadline=20ms, host delay=400ms, route duration=%v", elapsed)
			select {
			case <-hostFinished:
			case <-time.After(time.Second):
				t.Fatal("host callback did not stop after routing finished")
			}
			if err != nil || len(outcome.SelectedModelIDs) == 0 || outcome.SelectedModelIDs[0] != "strong-model" {
				t.Fatalf("outcome=%+v err=%v", outcome, err)
			}
			if elapsed >= 200*time.Millisecond {
				t.Fatal("routing waited for the host instead of its configured deadline")
			}
		})
	}
}

func TestDefaultEngineSelection(t *testing.T) {
	config := testConfig(`{"type":"stage_router","picker":"efficient_first",` + stage + `}`)
	auto, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	defer auto.Close()
	if nativeAvailable() {
		if !backendUsesNative(auto.state.backend) {
			t.Fatal("Auto did not select native")
		}
	} else if _, ok := auto.state.backend.(*wasmBackend); !ok {
		t.Fatalf("Auto selected %T, want wasm", auto.state.backend)
	}
	forced, err := New(config, WithEngine(WASM))
	if err != nil {
		t.Fatal(err)
	}
	defer forced.Close()
	if _, ok := forced.state.backend.(*wasmBackend); !ok {
		t.Fatalf("WASM selected %T", forced.state.backend)
	}
	forcedNative, err := New(config, WithEngine(Native))
	if !nativeAvailable() && err == nil {
		t.Fatal("Native succeeded without cgo and switchyard_native")
	}
	if forcedNative != nil {
		forcedNative.Close()
	}
}
