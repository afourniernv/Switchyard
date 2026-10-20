// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package switchyard

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

type tarsCorpus struct {
	Scopes   map[string]map[string][]string `json:"scopes"`
	Requests map[string]json.RawMessage     `json:"requests"`
	Cases    []tarsCorpusCase               `json:"cases"`
}

type tarsCorpusCase struct {
	ID       string `json:"id"`
	Scope    string `json:"scope"`
	Request  string `json:"request"`
	Settings struct {
		Mode string `json:"mode"`
	} `json:"settings"`
	Expect struct {
		Selected       string   `json:"selected"`
		Fallbacks      []string `json:"fallbacks"`
		DecisionSource string   `json:"decisionSource"`
	} `json:"expect"`
}

func TestTarsCorpus(t *testing.T) {
	path := os.Getenv("SWITCHYARD_TARS_CORPUS")
	if path == "" {
		t.Skip("set SWITCHYARD_TARS_CORPUS to replay the local TARS corpus")
	}
	if strings.HasPrefix(path, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			t.Fatal(err)
		}
		path = filepath.Join(home, path[2:])
	}
	wire, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var corpus tarsCorpus
	if err := json.Unmarshal(wire, &corpus); err != nil {
		t.Fatal(err)
	}
	fixture := readRecipe(t, "stage-signal-only.json")
	var mismatches []string
	forEachEngine(t, func(t *testing.T, engine Engine) {
		modes := make(map[string]struct{})
		orderedModes := make([]string, 0, 2)
		for _, testCase := range corpus.Cases {
			if _, exists := modes[testCase.Settings.Mode]; !exists {
				modes[testCase.Settings.Mode] = struct{}{}
				orderedModes = append(orderedModes, testCase.Settings.Mode)
			}
		}
		for _, mode := range orderedModes {
			config := setFixturePicker(t, fixture, mode)
			router, err := New(config, WithEngine(engine))
			if err != nil {
				t.Fatalf("mode %s: %v", mode, err)
			}
			t.Cleanup(router.Close)
			for _, testCase := range corpus.Cases {
				if testCase.Settings.Mode != mode {
					continue
				}
				scope, ok := corpus.Scopes[testCase.Scope]
				if !ok {
					mismatches = append(mismatches, fmt.Sprintf("%s %s: missing scope %q", engineName(engine), testCase.ID, testCase.Scope))
					continue
				}
				request, ok := corpus.Requests[testCase.Request]
				if !ok {
					mismatches = append(mismatches, fmt.Sprintf("%s %s: missing request %q", engineName(engine), testCase.ID, testCase.Request))
					continue
				}
				normalizedRequest, err := DecodeRequest("openai_chat", request)
				if err != nil {
					mismatches = append(mismatches, fmt.Sprintf("%s %s: request decode error %v", engineName(engine), testCase.ID, err))
					continue
				}
				outcome, err := router.Route(context.Background(), Input{Request: normalizedRequest, Models: scope}, nil)
				if err != nil {
					mismatches = append(mismatches, fmt.Sprintf("%s %s: route error %v", engineName(engine), testCase.ID, err))
					continue
				}
				actual := corpusOutcome(outcome)
				want := struct {
					Selected       string
					Fallbacks      []string
					DecisionSource string
				}{testCase.Expect.Selected, testCase.Expect.Fallbacks, testCase.Expect.DecisionSource}
				if !reflect.DeepEqual(actual, want) {
					mismatches = append(mismatches, fmt.Sprintf("%s %s scope=%s mode=%s: expected selected=%s fallbacks=%v source=%s; got selected=%s fallbacks=%v source=%s", engineName(engine), testCase.ID, testCase.Scope, mode, want.Selected, want.Fallbacks, want.DecisionSource, actual.Selected, actual.Fallbacks, actual.DecisionSource))
				}
			}
		}
	})
	for _, mismatch := range mismatches {
		t.Error(mismatch)
	}
}

func setFixturePicker(t *testing.T, fixture json.RawMessage, mode string) json.RawMessage {
	t.Helper()
	var config map[string]any
	if err := json.Unmarshal(fixture, &config); err != nil {
		t.Fatal(err)
	}
	algorithm := config["algorithm"].(map[string]any)
	algorithm["picker"] = mode
	encoded, err := json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}

func corpusOutcome(outcome Outcome) struct {
	Selected       string
	Fallbacks      []string
	DecisionSource string
} {
	result := struct {
		Selected       string
		Fallbacks      []string
		DecisionSource string
	}{}
	if len(outcome.SelectedModelIDs) > 0 {
		result.Selected = outcome.SelectedModelIDs[0]
		result.Fallbacks = outcome.SelectedModelIDs[1:]
	}
	var metadata struct {
		Evidence struct {
			Source string `json:"source"`
		} `json:"evidence"`
	}
	_ = json.Unmarshal(outcome.Metadata, &metadata)
	result.DecisionSource = metadata.Evidence.Source
	return result
}
