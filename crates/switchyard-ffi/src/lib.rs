// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pull-based C bridge to libsy.

use std::collections::{BTreeMap, HashMap};
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use futures::StreamExt;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use switchyard_libsy::{Algorithm, Call, LibsyError, RuntimeModels, Step, StepStream};
use switchyard_protocol::{
    AggLlmResponse, Category, DecisionRequest, LlmRequest, LlmResponse, Metadata, ModelId, Request,
    Response, WireFormat,
};
use switchyard_runner::AlgorithmSpec;
use tokio::runtime::Runtime;
use tokio_util::sync::CancellationToken;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type ModelGroups = HashMap<String, Option<Vec<ModelId>>>;

/// Returns the compatibility generation of the C interface and its JSON payloads.
#[unsafe(no_mangle)]
pub extern "C" fn sy_abi_version() -> u32 {
    1
}

#[derive(Deserialize)]
struct RouterConfig {
    algorithm: AlgorithmSpec,
    targets: BTreeMap<String, ModelId>,
}

#[derive(Deserialize)]
struct RunInput {
    request: LlmRequest,
    models: ModelGroups,
    subagent_models: Option<ModelGroups>,
    metadata: Option<RequestMetadata>,
}

#[derive(Deserialize)]
struct RequestMetadata {
    session_id: Option<String>,
    agent_id: Option<String>,
    #[serde(default)]
    is_subagent: bool,
    #[serde(default)]
    is_delegated_work: bool,
    session_final: Option<bool>,
}

#[derive(Deserialize)]
struct DecodeInput {
    format: WireFormat,
    body: serde_json::Value,
}

#[derive(Deserialize)]
struct EncodeInput {
    format: WireFormat,
    request: LlmRequest,
    model: ModelId,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum CallReply {
    Error {
        error: String,
    },
    Response {
        response: serde_json::Value,
        served_model: Option<ModelId>,
    },
}

#[derive(Serialize)]
struct OutcomeMetadataRef<'a> {
    outcome_id: &'a str,
    algorithm: &'a str,
    evidence: Option<&'a serde_json::Value>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RunStep<'a> {
    CallModel {
        id: u64,
        models: &'a [ModelId],
        request: &'a LlmRequest,
    },
    CallDecision {
        id: u64,
        model: &'a ModelId,
        request: &'a DecisionRequest,
    },
    Done {
        selected_model_ids: &'a [ModelId],
        request: &'a LlmRequest,
        response: Option<&'a AggLlmResponse>,
        metadata: Option<OutcomeMetadataRef<'a>>,
    },
}

struct Engine {
    algorithm: Arc<dyn Algorithm>,
    runtime: Runtime,
}

/// Algorithm handle; routing runs retain its engine.
pub struct SyRouter {
    engine: Arc<Engine>,
}

struct Reader {
    stream: StepStream,
    next_id: u64,
}

/// Routing run with independent reader, reply, and cancellation access.
pub struct SyRun {
    reader: Mutex<Reader>,
    pending: Mutex<HashMap<u64, Call>>,
    cancellation: CancellationToken,
    // Drop streams and pending calls before shutting down the last runtime owner.
    engine: Arc<Engine>,
}

// Release the returned error through sy_string_free.
fn error_string(message: String) -> *mut c_char {
    match CString::new(message) {
        Ok(message) => message.into_raw(),
        Err(_) => c"Switchyard error contained a NUL byte"
            .to_owned()
            .into_raw(),
    }
}

// Input must be valid, NUL-terminated UTF-8; parsed values do not borrow it.
unsafe fn parse<T: serde::de::DeserializeOwned>(input: *const c_char) -> Result<T> {
    if input.is_null() {
        return Err("missing JSON input".into());
    }
    let input = unsafe { CStr::from_ptr(input) };
    Ok(serde_json::from_slice(input.to_bytes())?)
}

fn encoded(output: &impl Serialize) -> Result<*mut c_char> {
    Ok(CString::new(serde_json::to_vec(output)?)?.into_raw())
}

/// Decodes provider JSON into a normalized request.
/// # Safety
/// JSON must be NUL-terminated and `out_json` writable. Free the output with `sy_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_decode_request(
    input_json: *const c_char,
    out_json: *mut *mut c_char,
) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let out = unsafe { out_json.as_mut() }.ok_or("missing request output")?;
        *out = std::ptr::null_mut();
        let input: DecodeInput = unsafe { parse(input_json) }?;
        *out = encoded(&switchyard_translation::decode_request(
            input.format,
            &input.body,
        )?)?;
        Ok(())
    }));
    match result {
        Ok(Ok(())) => std::ptr::null_mut(),
        Ok(Err(error)) => error_string(error.to_string()),
        Err(_) => error_string("Switchyard binding panicked".to_string()),
    }
}

/// Applies the selected model and encodes the request.
/// # Safety
/// JSON must be NUL-terminated and `out_json` writable. Free the output with `sy_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_encode_request(
    input_json: *const c_char,
    out_json: *mut *mut c_char,
) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let out = unsafe { out_json.as_mut() }.ok_or("missing request output")?;
        *out = std::ptr::null_mut();
        let mut input: EncodeInput = unsafe { parse(input_json) }?;
        switchyard_translation::prepare_request_for_target(&mut input.request, &input.model, None);
        *out = encoded(&switchyard_translation::encode_request(
            &input.request,
            input.format,
        )?)?;
        Ok(())
    }));
    match result {
        Ok(Ok(())) => std::ptr::null_mut(),
        Ok(Err(error)) => error_string(error.to_string()),
        Err(_) => error_string("Switchyard binding panicked".to_string()),
    }
}

/// Decodes a buffered provider response.
/// # Safety
/// JSON must be NUL-terminated and `out_json` writable. Free the output with `sy_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_decode_response(
    input_json: *const c_char,
    out_json: *mut *mut c_char,
) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let out = unsafe { out_json.as_mut() }.ok_or("missing response output")?;
        *out = std::ptr::null_mut();
        let input: DecodeInput = unsafe { parse(input_json) }?;
        *out = encoded(&switchyard_translation::decode_aggregated_response(
            &input.body,
            input.format,
        )?)?;
        Ok(())
    }));
    match result {
        Ok(Ok(())) => std::ptr::null_mut(),
        Ok(Err(error)) => error_string(error.to_string()),
        Err(_) => error_string("Switchyard binding panicked".to_string()),
    }
}

fn model_groups(groups: ModelGroups) -> Result<HashMap<Category, Vec<ModelId>>> {
    groups
        .into_iter()
        .map(|(name, models)| Ok((name.parse()?, models.unwrap_or_default())))
        .collect()
}

/// Builds an algorithm without provider clients.
/// # Safety
/// JSON must be NUL-terminated and `out` writable. Free the returned handle with `sy_router_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_router_new(
    config_json: *const c_char,
    out: *mut *mut SyRouter,
) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let out = unsafe { out.as_mut() }.ok_or("missing router output")?;
        *out = std::ptr::null_mut();
        let config: RouterConfig = unsafe { parse(config_json) }?;
        let algorithm = config.algorithm.build("embedded", &config.targets)?;
        let runtime = Runtime::new()?;
        *out = Box::into_raw(Box::new(SyRouter {
            engine: Arc::new(Engine { algorithm, runtime }),
        }));
        Ok(())
    }));
    match result {
        Ok(Ok(())) => std::ptr::null_mut(),
        Ok(Err(error)) => error_string(error.to_string()),
        Err(_) => error_string("Switchyard binding panicked".to_string()),
    }
}

/// Starts a run with host-approved model groups.
/// # Safety
/// Handles must be live, JSON NUL-terminated, and `out` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_run_new(
    router: *const SyRouter,
    input_json: *const c_char,
    out: *mut *mut SyRun,
) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let router = unsafe { router.as_ref() }.ok_or("missing router")?;
        let out = unsafe { out.as_mut() }.ok_or("missing run output")?;
        *out = std::ptr::null_mut();
        let input: RunInput = unsafe { parse(input_json) }?;
        let mut models = RuntimeModels::new(model_groups(input.models)?);
        if let Some(subagent_models) = input.subagent_models {
            models = models.with_subagent(model_groups(subagent_models)?);
        }
        let request = Request {
            llm_request: input.request,
            metadata: input.metadata.map(|metadata| Metadata {
                session_id: metadata.session_id,
                agent_id: metadata.agent_id,
                is_subagent: metadata.is_subagent,
                is_delegated_work: metadata.is_delegated_work,
                session_final: metadata.session_final,
                ..Default::default()
            }),
            ..Default::default()
        };
        let _entered = router.engine.runtime.enter();
        let stream = Arc::clone(&router.engine.algorithm).run_stream(request, Arc::new(models));
        *out = Box::into_raw(Box::new(SyRun {
            reader: Mutex::new(Reader { stream, next_id: 1 }),
            pending: Mutex::new(HashMap::new()),
            cancellation: CancellationToken::new(),
            engine: Arc::clone(&router.engine),
        }));
        Ok(())
    }));
    match result {
        Ok(Ok(())) => std::ptr::null_mut(),
        Ok(Err(error)) => error_string(error.to_string()),
        Err(_) => error_string("Switchyard binding panicked".to_string()),
    }
}

/// Returns the next host call or routing outcome.
/// # Safety
/// The run must remain live. Only one reader may call this at a time; `out_json` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_run_next(run: *const SyRun, out_json: *mut *mut c_char) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let run = unsafe { run.as_ref() }.ok_or("missing run")?;
        let out = unsafe { out_json.as_mut() }.ok_or("missing step output")?;
        *out = std::ptr::null_mut();
        let mut reader = run.reader.lock();
        let step: Result<Step> = run.engine.runtime.block_on(async {
            tokio::select! {
                biased;
                _ = run.cancellation.cancelled() => Err("routing cancelled".into()),
                step = reader.stream.next() => Ok(step.ok_or("routing stream ended without an outcome")??),
            }
        });
        let step = step?;
        let id = reader.next_id;
        reader.next_id += 1;
        *out = match step {
            Step::CallModel(call) => {
                let output = encoded(&RunStep::CallModel {
                    id,
                    models: &call.models,
                    request: &call.request.llm_request,
                })?;
                run.pending.lock().insert(id, Call::Model(call));
                output
            }
            Step::CallDecision(call) => {
                let output = encoded(&RunStep::CallDecision {
                    id,
                    model: &call.model,
                    request: &call.request,
                })?;
                run.pending.lock().insert(id, Call::Decision(call));
                output
            }
            Step::Done(outcome) => {
                let response = match outcome.response.as_ref() {
                    None => None,
                    Some(Response {
                        llm_response: LlmResponse::Agg(response),
                        ..
                    }) => Some(response),
                    Some(_) => {
                        return Err(
                            "streaming routing outcomes are not supported by this spike".into()
                        );
                    }
                };
                encoded(&RunStep::Done {
                    selected_model_ids: &outcome.selected_model_ids,
                    request: &outcome.request.llm_request,
                    response,
                    metadata: outcome
                        .metadata
                        .as_ref()
                        .map(|metadata| OutcomeMetadataRef {
                            outcome_id: metadata.outcome_id(),
                            algorithm: &metadata.algorithm,
                            evidence: metadata.evidence.as_ref(),
                        }),
                })?
            }
        };
        Ok(())
    }));
    match result {
        Ok(Ok(())) => std::ptr::null_mut(),
        Ok(Err(error)) => error_string(error.to_string()),
        Err(_) => error_string("Switchyard binding panicked".to_string()),
    }
}

/// Delivers a buffered host reply to its pending call.
/// # Safety
/// The run must remain live and JSON must be NUL-terminated. May overlap `sy_run_next`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_run_respond(
    run: *const SyRun,
    call_id: u64,
    reply_json: *const c_char,
) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let run = unsafe { run.as_ref() }.ok_or("missing run")?;
        let reply: CallReply = unsafe { parse(reply_json) }?;
        let call = run
            .pending
            .lock()
            .remove(&call_id)
            .ok_or("unknown call ID")?;
        let result = match call {
            Call::Model(call) => {
                let response = match reply {
                    CallReply::Error { error } => Err(LibsyError::external(
                        "host model call",
                        std::io::Error::other(error),
                    )),
                    CallReply::Response {
                        response,
                        served_model,
                    } => Ok(Response {
                        llm_response: LlmResponse::Agg(serde_json::from_value(response)?),
                        metadata: served_model.map(|model| Metadata {
                            served_model: Some(model),
                            ..Default::default()
                        }),
                        upstream_headers: Default::default(),
                    }),
                };
                run.engine
                    .runtime
                    .block_on(call.respond(std::future::ready(response)))
            }
            Call::Decision(call) => {
                let response = match reply {
                    CallReply::Error { error } => Err(LibsyError::external(
                        "host decision call",
                        std::io::Error::other(error),
                    )),
                    CallReply::Response { response, .. } => Ok(serde_json::from_value(response)?),
                };
                run.engine
                    .runtime
                    .block_on(call.respond(std::future::ready(response)))
            }
        };
        if result.is_err() {
            run.cancellation.cancel();
        }
        Ok(result?)
    }));
    match result {
        Ok(Ok(())) => std::ptr::null_mut(),
        Ok(Err(error)) => error_string(error.to_string()),
        Err(_) => error_string("Switchyard binding panicked".to_string()),
    }
}

/// Signals cancellation without waiting for active calls.
/// # Safety
/// The run must remain live for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_run_cancel(run: *const SyRun) {
    if let Some(run) = unsafe { run.as_ref() } {
        run.cancellation.cancel();
    }
}

/// Frees a routing run.
/// # Safety
/// Pass a run returned by `sy_run_new` once, with no concurrent users, or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_run_free(run: *mut SyRun) {
    if !run.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(run) })));
    }
}

/// Frees the router handle; existing runs retain its engine.
/// # Safety
/// Pass a router returned by `sy_router_new` once, with no concurrent users, or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_router_free(router: *mut SyRouter) {
    if !router.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(router) })));
    }
}

/// Frees a library-owned error or JSON string.
/// # Safety
/// Pass a returned string once, with no remaining readers, or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sy_string_free(string: *mut c_char) {
    if !string.is_null() {
        drop(unsafe { CString::from_raw(string) });
    }
}

#[cfg(test)]
mod tests;
