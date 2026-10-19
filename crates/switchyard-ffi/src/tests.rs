// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the C bridge.

use super::*;
use serde_json::{Value, json};

// Version 1 identifies the bridge contract, independently of the package release.
#[test]
fn abi_version_identifies_v1_contract() {
    assert_eq!(sy_abi_version(), 1);
}

fn check(error: *mut c_char) {
    if !error.is_null() {
        panic!("{}", unsafe { CString::from_raw(error) }.to_string_lossy());
    }
}

fn router(algorithm: Value) -> Box<SyRouter> {
    let config = json!({
        "algorithm": algorithm,
        "targets": {"capable": "capable", "efficient": "efficient", "judge": "judge"},
    });
    let config = CString::new(config.to_string()).unwrap();
    let mut router = std::ptr::null_mut();
    check(unsafe { sy_router_new(config.as_ptr(), &mut router) });
    assert!(!router.is_null());
    unsafe { Box::from_raw(router) }
}

fn run(router: &SyRouter, input: Value) -> Box<SyRun> {
    let input = CString::new(input.to_string()).unwrap();
    let mut run = std::ptr::null_mut();
    check(unsafe { sy_run_new(router, input.as_ptr(), &mut run) });
    assert!(!run.is_null());
    unsafe { Box::from_raw(run) }
}

fn next(run: &SyRun) -> Value {
    let mut output = std::ptr::null_mut();
    check(unsafe { sy_run_next(run, &mut output) });
    assert!(!output.is_null());
    let output = unsafe { CString::from_raw(output) };
    serde_json::from_slice(output.as_bytes()).unwrap()
}

// The serving target can differ from both the first candidate and the provider's model name.
#[test]
fn advisor_attributes_host_fallback() {
    let router = router(json!({
        "type": "advisor", "executor_target": "capable", "advisor_target": "judge",
        "gate_trigger": "pattern", "gate_trigger_pattern": "NEVER",
    }));
    for served_model in [None, Some("fallback")] {
        let input = json!({
            "request": {}, "models": {"efficient": ["capable", "fallback"], "judge": ["judge"]},
        });
        let run = run(&router, input);
        let call = next(&run);
        assert_eq!(call["models"], json!(["capable", "fallback"]));
        let mut reply = json!({
            "response": switchyard_protocol::text_response(Some("provider-model".into()), "answer"),
        });
        if let Some(model) = served_model {
            reply["served_model"] = json!(model);
        }
        let reply = CString::new(reply.to_string()).unwrap();
        check(unsafe { sy_run_respond(&*run, call["id"].as_u64().unwrap(), reply.as_ptr()) });
        let done = next(&run);
        let expected = served_model.unwrap_or("capable");
        assert_eq!(done["type"], "done");
        assert_eq!(done["selected_model_ids"], json!([expected]));
        assert_eq!(done["request"]["model"], expected);
        assert_eq!(done["response"]["model"], "provider-model");
    }
}

// Child sessions retain independent phases and release their state on the final request.
#[test]
fn plan_execute_isolates_and_finishes_child_sessions() {
    let router = router(json!({
        "type": "plan_execute", "capable_target": "capable", "efficient_target": "efficient",
    }));
    for (agent, is_subagent, is_final, edited, expected) in [
        ("a", true, false, true, "efficient"),
        ("b", true, false, false, "capable"),
        ("root", false, false, false, "capable"),
        ("a", true, true, false, "efficient"),
        ("a", true, false, false, "capable"),
    ] {
        let request = if edited {
            json!({"messages": [{"role": "assistant", "content": [{
                "type": "tool_call", "id": "edit", "name": "apply_patch", "arguments": {},
            }]}]})
        } else {
            json!({})
        };
        let input = json!({
            "request": request, "models": {"capable": ["capable"], "efficient": ["efficient"]},
            "metadata": {"session_id": "s", "agent_id": agent, "is_subagent": is_subagent,
                "session_final": is_final},
        });
        let run = run(&router, input);
        assert_eq!(
            next(&run)["selected_model_ids"],
            json!([expected]),
            "{agent}"
        );
    }
}

// Child lineage alone does not send maintenance requests through the delegated-work policy.
#[test]
fn delegated_work_uses_subagent_model_groups() {
    let router = router(json!({
        "type": "passthrough", "target": "capable",
        "subagents": {"type": "passthrough", "target": "efficient"},
    }));
    for (metadata, expected) in [
        (Value::Null, "capable"),
        (json!({"is_subagent": true}), "capable"),
        (
            json!({"is_subagent": true, "is_delegated_work": true}),
            "efficient",
        ),
    ] {
        let input = json!({
            "request": {}, "models": {"any": ["capable"]},
            "subagent_models": {"any": ["efficient"]}, "metadata": metadata,
        });
        let run = run(&router, input);
        assert_eq!(next(&run)["selected_model_ids"], json!([expected]));
    }
}

#[test]
fn invalid_router_input_returns_owned_errors() {
    for input in [std::ptr::null(), c"{".as_ptr()] {
        let mut router = std::ptr::dangling_mut::<SyRouter>();
        let error = unsafe { sy_router_new(input, &mut router) };
        assert!(!error.is_null());
        assert!(router.is_null());
        let message = unsafe { CStr::from_ptr(error) }.to_str().unwrap();
        assert!(!message.is_empty());
        unsafe { sy_string_free(error) };
    }
    let error = unsafe { sy_router_new(c"{".as_ptr(), std::ptr::null_mut()) };
    assert!(!error.is_null());
    unsafe { sy_string_free(error) };
}

// An unwinding panic becomes an owned error at the exported C boundary.
#[test]
fn panics_return_owned_errors() {
    let run = SyRun {
        reader: Mutex::new(Reader {
            stream: Box::pin(futures::stream::poll_fn(|_| panic!("test panic"))),
            next_id: 1,
        }),
        pending: Mutex::new(HashMap::new()),
        cancellation: CancellationToken::new(),
        engine: Arc::new(Engine {
            algorithm: Arc::new(switchyard_libsy::Noop {}),
            runtime: Runtime::new().unwrap(),
        }),
    };
    let mut output = std::ptr::dangling_mut::<c_char>();
    let error = unsafe { sy_run_next(&run, &mut output) };
    assert!(!error.is_null());
    assert!(output.is_null());
    assert_eq!(
        unsafe { CStr::from_ptr(error) }.to_str().unwrap(),
        "Switchyard binding panicked"
    );
    unsafe { sy_string_free(error) };
}
