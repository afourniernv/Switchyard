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

// Fallible functions return NULL on success, otherwise an owned error string.
// Inputs are borrowed for the call. Free returned strings with sy_string_free.
char *sy_router_new(const char *config_json, SyRouter **out);
// Run JSON: {request, models, subagent_models?, metadata?}.
// Metadata: session_id, agent_id, is_subagent, is_delegated_work, session_final.
char *sy_run_new(const SyRouter *router, const char *input_json, SyRun **out);
// Only one next call may be active per run. Respond and cancel may overlap it.
char *sy_run_next(const SyRun *run, char **out_json);
// Replies: {response, served_model?} or {error}. served_model is a target ID.
char *sy_run_respond(const SyRun *run, uint64_t call_id, const char *reply_json);
// Decode envelopes contain {format, body}; encode contains {format, request, model}.
char *sy_decode_request(const char *input_json, char **out_json);
char *sy_encode_request(const char *input_json, char **out_json);
char *sy_decode_response(const char *input_json, char **out_json);
void sy_run_cancel(const SyRun *run);
// Free handles only after active calls finish; runs retain the engine.
void sy_run_free(SyRun *run);
void sy_router_free(SyRouter *router);
void sy_string_free(char *string);

#endif
