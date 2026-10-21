// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package switchyard

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func BenchmarkDecodeRequestParallel(b *testing.B) {
	body := json.RawMessage(fmt.Sprintf(`{"model":"route","messages":[{"role":"user","content":%q}]}`, strings.Repeat("x", 24*1024)))
	if _, err := DecodeRequest("openai_chat", body); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if _, err := DecodeRequest("openai_chat", body); err != nil {
				b.Error(err)
				return
			}
		}
	})
}

func BenchmarkRouteLatency(b *testing.B) {
	for _, engine := range testEngines() {
		for _, fixture := range []string{"stage-signal-only.json", "stage-llm-judge.json", "stage-decision-judge.json"} {
			for _, size := range []int{10 * 1024, 500 * 1024} {
				name := fmt.Sprintf("%s/%s/%dkb", engineName(engine), strings.TrimSuffix(fixture, ".json"), size/1024)
				b.Run(name, func(b *testing.B) {
					router, err := New(readRecipe(b, fixture), WithEngine(engine))
					if err != nil {
						b.Fatal(err)
					}
					defer router.Close()
					request := recipeInput()
					request.Request = json.RawMessage(fmt.Sprintf(`{"messages":[{"role":"user","content":[{"type":"text","text":%q}]}]}`, strings.Repeat("x", size)))
					host := HostCall(func(_ context.Context, work Work) (Reply, error) {
						switch work.Type {
						case "call_model":
							return Reply{Response: json.RawMessage(judgeAnswer)}, nil
						case "call_decision":
							return Reply{Response: json.RawMessage(decisionAnswer)}, nil
						default:
							return Reply{}, fmt.Errorf("unexpected host work %q", work.Type)
						}
					})
					if fixture == "stage-signal-only.json" {
						host = nil
					}
					b.ReportAllocs()
					b.ResetTimer()
					for index := 0; index < b.N; index++ {
						if _, err := router.Route(context.Background(), request, host); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}
