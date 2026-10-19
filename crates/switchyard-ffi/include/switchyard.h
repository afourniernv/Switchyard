// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef SWITCHYARD_H
#define SWITCHYARD_H

#include <stdint.h>

// Covers the C interface and JSON contract; bump for incompatible changes only.
#define SY_ABI_VERSION 1
uint32_t sy_abi_version(void);

typedef struct SyRouter SyRouter;
typedef struct SyRun SyRun;
typedef void (*SyNextCallback)(uintptr_t user_data, char *error, char *step_json);

// Fallible functions return NULL on success, otherwise an owned error string.
// Inputs are borrowed for the call. Free returned strings with sy_string_free.
char *sy_router_new(const char *config_json, SyRouter **out);
// Run JSON: {request, models, subagent_models?, metadata?}.
// Metadata: session_id, agent_id, is_subagent, is_delegated_work, session_final.
char *sy_run_new(const SyRouter *router, const char *input_json, SyRun **out);
// Only one routing read may be active. Respond and cancel may overlap it.
// NULL accepts one read and promises exactly one callback, including on cancellation.
// An immediate error means no callback. Delivery may precede return; callbacks may overlap.
// Callback owns its strings and must return promptly; do not call sy_run_respond or handle free.
char *sy_run_next_async(const SyRun *run, SyNextCallback callback, uintptr_t user_data);
// Replies: {response, served_model?} or {error}. served_model is a target ID.
char *sy_run_respond(const SyRun *run, uint64_t call_id, const char *reply_json);
// Decode envelopes contain {format, body}; encode contains {format, request, model}.
char *sy_decode_request(const char *input_json, char **out_json);
char *sy_encode_request(const char *input_json, char **out_json);
char *sy_decode_response(const char *input_json, char **out_json);
// Signals cancellation. Accepted callbacks still finish; run free is the cleanup barrier.
void sy_run_cancel(const SyRun *run);
// Free only after active bridge calls finish. Run free cancels and waits for callbacks;
// do not call it from a callback. Keep callback state live until that callback returns.
void sy_run_free(SyRun *run);
void sy_router_free(SyRouter *router);
void sy_string_free(char *string);

#endif
