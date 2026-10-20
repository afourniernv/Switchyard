// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package switchyard

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

const decisionAnswer = `{"answers":{"route":{"value":{"type":"choice","data":{"selected":"advantage","probabilities":{"advantage":0.9,"no_advantage":0.1}}}}}}`

var recipeFixtures = []string{
	"stage-signal-only.json",
	"stage-llm-judge.json",
	"stage-decision-judge.json",
}

func readRecipe(t testing.TB, name string) json.RawMessage {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "recipes", name))
	if err != nil {
		t.Fatal(err)
	}
	if !json.Valid(data) {
		t.Fatalf("%s is not valid JSON", name)
	}
	return data
}

func recipeInput() Input {
	return Input{
		Request: json.RawMessage(`{"messages":[{"role":"user","content":[{"type":"text","text":"Summarize the implementation plan."}]}]}`),
		Models: map[string][]string{
			"any":       {"deployment-a", "deployment-b"},
			"capable":   {"deployment-a"},
			"efficient": {"deployment-b"},
			"judge":     {"judge-model"},
		},
	}
}

func recipeHost(calls *[]Work) HostCall {
	return func(_ context.Context, work Work) (Reply, error) {
		*calls = append(*calls, work)
		switch work.Type {
		case "call_model":
			return Reply{Response: json.RawMessage(judgeAnswer)}, nil
		case "call_decision":
			return Reply{Response: json.RawMessage(decisionAnswer)}, nil
		default:
			return Reply{}, fmt.Errorf("unexpected host work %q", work.Type)
		}
	}
}

func TestRecipeFixturesRoute(t *testing.T) {
	forEachEngine(t, func(t *testing.T, engine Engine) {
		for _, fixture := range recipeFixtures {
			t.Run(fixture, func(t *testing.T) {
				router, err := New(readRecipe(t, fixture), WithEngine(engine))
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(router.Close)
				var calls []Work
				var host HostCall
				if fixture != "stage-signal-only.json" {
					host = recipeHost(&calls)
				}
				outcome, err := router.Route(context.Background(), recipeInput(), host)
				if err != nil {
					t.Fatal(err)
				}
				if len(outcome.SelectedModelIDs) == 0 {
					t.Fatal("route selected no models")
				}
				switch fixture {
				case "stage-signal-only.json":
					if len(calls) != 0 {
						t.Fatalf("signal-only recipe made host calls: %+v", calls)
					}
				case "stage-llm-judge.json":
					if len(calls) != 1 || calls[0].Type != "call_model" {
						t.Fatalf("LLM judge work=%+v", calls)
					}
				case "stage-decision-judge.json":
					if len(calls) != 1 || calls[0].Type != "call_decision" {
						t.Fatalf("decision judge work=%+v", calls)
					}
				}
			})
		}
	})
}

func TestDecisionRecipeCandidateLabels(t *testing.T) {
	var config struct {
		Algorithm struct {
			Classifier struct {
				Decision struct {
					Candidates map[string]string `json:"candidates"`
				} `json:"decision"`
			} `json:"classifier"`
		} `json:"algorithm"`
	}
	if err := json.Unmarshal(readRecipe(t, "stage-decision-judge.json"), &config); err != nil {
		t.Fatal(err)
	}
	labels := make([]string, 28)
	for index := range labels {
		if index < 26 {
			labels[index] = string(rune('a' + index))
		} else {
			labels[index] = fmt.Sprintf("m%d", index)
		}
	}
	if len(config.Algorithm.Classifier.Decision.Candidates) != len(labels) {
		t.Fatalf("candidate count=%d, want %d", len(config.Algorithm.Classifier.Decision.Candidates), len(labels))
	}
	for index, label := range labels {
		alias, ok := config.Algorithm.Classifier.Decision.Candidates[label]
		if !ok {
			t.Fatalf("candidate label %q is missing", label)
		}
		want := []string{"capable", "efficient"}
		if index >= 2 {
			want = []string{fmt.Sprintf("answer_%02d", index)}
		}
		if !containsString(want, alias) {
			t.Fatalf("candidate %s maps to %s, want one of %v", label, alias, want)
		}
	}
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func TestJudgeFailOpenDoesNotHideHostErrors(t *testing.T) {
	forEachEngine(t, func(t *testing.T, engine Engine) {
		for _, fixture := range []string{"stage-llm-judge.json", "stage-decision-judge.json"} {
			for _, failOpen := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/fail_open_%t", fixture, failOpen), func(t *testing.T) {
					config := setFixtureFailOpen(t, readRecipe(t, fixture), failOpen)
					router, err := New(config, WithEngine(engine))
					if err != nil {
						t.Fatal(err)
					}
					t.Cleanup(router.Close)
					_, err = router.Route(context.Background(), recipeInput(), func(context.Context, Work) (Reply, error) {
						return Reply{}, errors.New("injected host failure")
					})
					if err == nil || !strings.Contains(err.Error(), "injected host failure") {
						t.Fatalf("HostCall failure returned %v", err)
					}
				})
			}
		}
	})
}

func setFixtureFailOpen(t *testing.T, data json.RawMessage, value bool) json.RawMessage {
	t.Helper()
	var config map[string]any
	if err := json.Unmarshal(data, &config); err != nil {
		t.Fatal(err)
	}
	algorithm := config["algorithm"].(map[string]any)
	classifier := algorithm["classifier"].(map[string]any)
	classifier["fail_open"] = value
	encoded, err := json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}

func TestBackendParity(t *testing.T) {
	if !nativeAvailable() {
		t.Skip("native parity runs with -tags switchyard_native")
	}
	parityCases := []struct {
		name      string
		config    json.RawMessage
		input     Input
		needsHost bool
	}{
		{
			name:   "random",
			config: json.RawMessage(`{"algorithm":{"type":"random","targets":["only"],"seed":5},"targets":{"only":"deployment-a"}}`),
			input:  recipeInput(),
		},
	}
	parityCases[0].input.Models["any"] = []string{"deployment-a"}
	for _, fixture := range recipeFixtures {
		parityCases = append(parityCases, struct {
			name      string
			config    json.RawMessage
			input     Input
			needsHost bool
		}{
			name:      fixture,
			config:    readRecipe(t, fixture),
			input:     recipeInput(),
			needsHost: fixture != "stage-signal-only.json",
		})
	}
	for _, testCase := range parityCases {
		t.Run(testCase.name, func(t *testing.T) {
			outcomes := make(map[Engine][]byte)
			for _, engine := range []Engine{WASM, Native} {
				router, err := New(testCase.config, WithEngine(engine))
				if err != nil {
					t.Fatal(err)
				}
				var calls []Work
				var host HostCall
				if testCase.needsHost {
					host = recipeHost(&calls)
				}
				outcome, err := router.Route(context.Background(), testCase.input, host)
				router.Close()
				if err != nil {
					t.Fatal(err)
				}
				wire, err := json.Marshal(outcome)
				if err != nil {
					t.Fatal(err)
				}
				outcomes[engine] = normalizedOutcome(t, wire)
			}
			if !reflect.DeepEqual(outcomes[WASM], outcomes[Native]) {
				t.Fatalf("backend outcomes differ:\nwasm: %s\nnative: %s", outcomes[WASM], outcomes[Native])
			}
		})
	}
}

// Requests without a session id share libsy's first-user-message key, so the wasm pool
// must route them to one instance to keep its retained state.
func TestSessionlessRetentionParity(t *testing.T) {
	if !nativeAvailable() {
		t.Skip("native parity runs with -tags switchyard_native")
	}
	algorithm := strings.Replace(composite, `"classify_trigger":"user_turn"`, `"classify_trigger":"new_session","message_hash_fallback":true`, 1)
	config := json.RawMessage(`{"algorithm":` + algorithm + `,"targets":{"weak":"weak-model","strong":"strong-model","judge":"judge-model"}}`)
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

func normalizedOutcome(t *testing.T, wire []byte) []byte {
	t.Helper()
	var value any
	if err := json.Unmarshal(wire, &value); err != nil {
		t.Fatal(err)
	}
	stripVolatileFields(value)
	normalized, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return normalized
}

func stripVolatileFields(value any) {
	switch item := value.(type) {
	case map[string]any:
		for key, child := range item {
			lower := strings.ToLower(key)
			volatileTime := lower == "time" || lower == "timestamp" || lower == "created_at" || lower == "updated_at" || strings.HasSuffix(lower, "_time") || strings.HasSuffix(lower, "_timestamp")
			if key == "outcome_id" || volatileTime {
				delete(item, key)
				continue
			}
			stripVolatileFields(child)
		}
	case []any:
		for _, child := range item {
			stripVolatileFields(child)
		}
	}
}

func TestDefaultEngineSelection(t *testing.T) {
	config := readRecipe(t, "stage-signal-only.json")
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
	_, err = New(config, WithEngine(Native))
	if !nativeAvailable() && err == nil {
		t.Fatal("Native succeeded without cgo and switchyard_native")
	}
}
