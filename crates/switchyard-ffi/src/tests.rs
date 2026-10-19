// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the C bridge.

use super::*;
use serde_json::{Value, json};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

type Completion = std::result::Result<CString, CString>;

unsafe extern "C" fn complete(user_data: usize, error: *mut c_char, output: *mut c_char) {
    let sender = unsafe { &*(user_data as *const mpsc::Sender<Completion>) };
    let output = (!output.is_null()).then(|| unsafe { CString::from_raw(output) });
    let result = if error.is_null() {
        Ok(output.unwrap_or_default())
    } else {
        Err(unsafe { CString::from_raw(error) })
    };
    let _ = sender.send(result);
}

unsafe extern "C" fn complete_once(user_data: usize, error: *mut c_char, output: *mut c_char) {
    let sender = unsafe { Box::from_raw(user_data as *mut mpsc::Sender<Completion>) };
    unsafe { complete(&*sender as *const _ as usize, error, output) };
}

fn next_async(run: &SyRun, sender: &mpsc::Sender<Completion>) -> std::result::Result<(), String> {
    let sender = Box::into_raw(Box::new(sender.clone()));
    let error = unsafe { sy_run_next_async(run, Some(complete_once), sender as usize) };
    if error.is_null() {
        Ok(())
    } else {
        drop(unsafe { Box::from_raw(sender) });
        Err(error_message(error))
    }
}

fn received_step(receiver: &mpsc::Receiver<Completion>) -> Value {
    let output = receiver
        .recv_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap();
    serde_json::from_slice(output.as_bytes()).unwrap()
}

// The ABI generation is independent of the package release.
#[test]
fn abi_version_identifies_v1_contract() {
    assert_eq!(sy_abi_version(), 1);
}

fn check(error: *mut c_char) {
    if !error.is_null() {
        panic!("{}", unsafe { CString::from_raw(error) }.to_string_lossy());
    }
}

fn error_message(error: *mut c_char) -> String {
    assert!(!error.is_null(), "expected an error");
    unsafe { CString::from_raw(error) }.into_string().unwrap()
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
    let (sender, receiver) = mpsc::channel();
    next_async(run, &sender).unwrap();
    received_step(&receiver)
}

fn advisor_router() -> Box<SyRouter> {
    router(json!({
        "type": "advisor", "executor_target": "capable", "advisor_target": "judge",
        "gate_trigger": "pattern", "gate_trigger_pattern": "NEVER",
    }))
}

fn pending_advisor() -> (Box<SyRun>, u64) {
    let router = advisor_router();
    let run = run(
        &router,
        json!({"request": {}, "models": {"efficient": ["capable"], "judge": ["judge"]}}),
    );
    let call = next(&run);
    assert_eq!(call["type"], "call_model");
    (run, call["id"].as_u64().unwrap())
}

// A successful outcome remains terminal on every later read.
#[test]
fn completed_runs_reject_further_reads() {
    let router = router(json!({"type": "passthrough", "target": "capable"}));
    let run = run(
        &router,
        json!({"request": {}, "models": {"any": ["capable"]}}),
    );
    assert_eq!(next(&run)["type"], "done");
    let (sender, receiver) = mpsc::channel();
    for _ in 0..2 {
        assert_eq!(
            next_async(&run, &sender).unwrap_err(),
            "routing already completed"
        );
        assert!(matches!(
            receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
    }
}

// Cancellation prevents a host from fulfilling work that the run no longer needs.
#[test]
fn cancelled_runs_reject_pending_replies() {
    let (run, id) = pending_advisor();
    unsafe { sy_run_cancel(&*run) };
    let error = unsafe { sy_run_respond(&*run, id, c"{\"response\":{}}".as_ptr()) };
    assert_eq!(error_message(error), "routing cancelled");
}

// A typed decoding failure must leave the call available for a corrected reply.
#[test]
fn invalid_model_replies_preserve_pending_calls() {
    let (run, id) = pending_advisor();
    let error = unsafe {
        sy_run_respond(
            &*run,
            id,
            c"{\"response\":{\"outputs\":\"wrong\"}}".as_ptr(),
        )
    };
    assert!(!error_message(error).is_empty());
    check(unsafe { sy_run_respond(&*run, id, c"{\"response\":{}}".as_ptr()) });
    assert_eq!(next(&run)["type"], "done");
}

// Pending host work yields the only runtime worker without creating blocking threads.
#[test]
fn async_reads_wait_without_native_threads() {
    let thread_starts = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&thread_starts);
    let mut router = advisor_router();
    Arc::get_mut(&mut router.engine).unwrap().runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .on_thread_start(move || {
            observed.fetch_add(1, Ordering::SeqCst);
        })
        .build()
        .unwrap();
    let (sender, receiver) = mpsc::channel();
    let mut runs = Vec::new();
    for _ in 0..32 {
        let run = run(
            &router,
            json!({"request": {}, "models": {"efficient": ["capable"], "judge": ["judge"]}}),
        );
        next_async(&run, &sender).unwrap();
        assert_eq!(received_step(&receiver)["type"], "call_model");
        next_async(&run, &sender).unwrap();
        runs.push(run);
    }
    let (scheduled, sentinel) = mpsc::channel();
    router.engine.runtime.spawn(async move {
        let _ = scheduled.send(());
    });
    sentinel.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(thread_starts.load(Ordering::SeqCst), 1);
    drop(runs);
    for _ in 0..32 {
        let error = receiver
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
            .unwrap_err();
        assert_eq!(error.to_str().unwrap(), "routing cancelled");
    }
}

struct BlockedCompletion {
    sender: mpsc::Sender<Completion>,
    resume: Mutex<mpsc::Receiver<()>>,
    finished: mpsc::Sender<()>,
}

unsafe extern "C" fn complete_after_release(
    user_data: usize,
    error: *mut c_char,
    output: *mut c_char,
) {
    let state = unsafe { &*(user_data as *const BlockedCompletion) };
    unsafe { complete(&state.sender as *const _ as usize, error, output) };
    let _ = state.resume.lock().recv();
    let _ = state.finished.send(());
}

// Free is the userdata lifetime barrier even when another owner keeps the runtime alive.
#[test]
fn free_waits_for_cancelled_callback_to_return() {
    let engine = Arc::new(Engine {
        algorithm: Arc::new(switchyard_libsy::Noop {}),
        runtime: tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap(),
    });
    let (sender, receiver) = mpsc::channel();
    let (resume, paused) = mpsc::channel();
    let (finished, returned) = mpsc::channel();
    let state = Box::into_raw(Box::new(BlockedCompletion {
        sender,
        resume: Mutex::new(paused),
        finished,
    }));
    let run = Box::new(SyRun::new(
        Box::pin(futures::stream::pending()),
        Arc::clone(&engine),
    ));
    check(unsafe { sy_run_next_async(&*run, Some(complete_after_release), state as usize) });
    let run = Box::into_raw(run) as usize;
    let (freed, closed) = mpsc::channel();
    let freeing = std::thread::spawn(move || {
        unsafe { sy_run_free(run as *mut SyRun) };
        let _ = freed.send(());
    });
    let cancellation = receiver.recv_timeout(Duration::from_secs(5));
    let premature_free = closed.recv_timeout(Duration::from_millis(50));
    let _ = resume.send(());
    returned.recv_timeout(Duration::from_secs(5)).unwrap();
    if premature_free.is_err() {
        closed.recv_timeout(Duration::from_secs(5)).unwrap();
    }
    freeing.join().unwrap();
    drop(unsafe { Box::from_raw(state) });
    assert!(matches!(
        premature_free,
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    assert_eq!(
        cancellation.unwrap().unwrap_err().to_str().unwrap(),
        "routing cancelled"
    );
}

// Reply order and cancellation belong to each run, including when call IDs are equal.
#[test]
fn async_runs_complete_independently() {
    let router = advisor_router();
    let (sender, receiver) = mpsc::channel();
    let runs: Vec<_> = (0..3)
        .map(|_| {
            run(
                &router,
                json!({"request": {}, "models": {"efficient": ["capable"], "judge": ["judge"]}}),
            )
        })
        .collect();
    let mut ids = Vec::new();
    for run in &runs {
        next_async(run, &sender).unwrap();
        ids.push(received_step(&receiver)["id"].as_u64().unwrap());
        next_async(run, &sender).unwrap();
    }
    for (index, text) in [(1, "second"), (0, "first")] {
        if index == 0 {
            unsafe { sy_run_cancel(&*runs[2]) };
            let error = receiver
                .recv_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap_err();
            assert_eq!(error.to_str().unwrap(), "routing cancelled");
        }
        let reply = CString::new(
            json!({
                "response": switchyard_protocol::text_response(None, text),
            })
            .to_string(),
        )
        .unwrap();
        check(unsafe { sy_run_respond(&*runs[index], ids[index], reply.as_ptr()) });
        let done = received_step(&receiver);
        assert_eq!(done["type"], "done");
        assert_eq!(done["response"]["outputs"][0]["content"][0]["text"], text);
    }
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
    let run = SyRun::new(
        Box::pin(futures::stream::poll_fn(|_| panic!("test panic"))),
        Arc::new(Engine {
            algorithm: Arc::new(switchyard_libsy::Noop {}),
            runtime: Runtime::new().unwrap(),
        }),
    );
    let (sender, receiver) = mpsc::channel();
    next_async(&run, &sender).unwrap();
    let error = receiver
        .recv_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap_err();
    assert_eq!(error.to_str().unwrap(), "Switchyard binding panicked");
}
