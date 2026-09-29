// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay_plugin::{Json, PluginRuntime, ScopeType};
use serde_json::{Map, Value, json};
use switchyard_llm_client::{LlmCallPhase, LlmCallTrace, LlmCallTraceOutcome};
use switchyard_protocol::{UpstreamAttemptObservation, UpstreamAttemptOutcome};
use switchyard_runner::ProviderKeyRedactor;

const CLIENT_CALL: &str = "switchyard.client_call";
const UPSTREAM_ATTEMPT: &str = "switchyard.upstream_attempt";

/// A completed Switchyard model call ready for Relay scope projection.
#[derive(Debug)]
pub(crate) struct RelayTrace(LlmCallTrace);

impl From<LlmCallTrace> for RelayTrace {
    fn from(trace: LlmCallTrace) -> Self {
        Self(trace)
    }
}

impl RelayTrace {
    pub(crate) fn sanitize(&mut self, redactor: &ProviderKeyRedactor) {
        self.0.selected_model = redactor.text(self.0.selected_model.to_string()).into();
        for attempt in &mut self.0.attempts {
            attempt.model = redactor.text(attempt.model.to_string()).into();
        }
    }

    pub(crate) fn emit(self, runtime: &PluginRuntime) -> Result<(), String> {
        let LlmCallTrace {
            algorithm,
            phase,
            candidate,
            candidate_count,
            selected_model,
            started_at,
            ended_at,
            outcome,
            attempts,
        } = self.0;
        let metadata = client_metadata(
            algorithm,
            phase,
            candidate,
            candidate_count,
            selected_model.as_str(),
            &outcome,
        );
        let status = status_metadata(call_outcome_name(&outcome));
        let mut scope = runtime.scope_at(
            CLIENT_CALL,
            ScopeType::Llm,
            None,
            Some(&metadata),
            None,
            started_at,
        )?;
        let attempts_result = attempts
            .into_iter()
            .try_for_each(|attempt| emit_attempt(runtime, attempt));
        let close_result = scope.close_at(None, Some(&status), ended_at);
        attempts_result.and(close_result)
    }
}

fn emit_attempt(
    runtime: &PluginRuntime,
    attempt: UpstreamAttemptObservation,
) -> Result<(), String> {
    let metadata = attempt_metadata(&attempt);
    let status = status_metadata(attempt_outcome_name(attempt.outcome));
    let mut scope = runtime.scope_at(
        UPSTREAM_ATTEMPT,
        ScopeType::Custom,
        None,
        Some(&metadata),
        None,
        attempt.started_at,
    )?;
    scope.close_at(None, Some(&status), attempt.ended_at)
}

fn client_metadata(
    algorithm: String,
    phase: LlmCallPhase,
    candidate: usize,
    candidate_count: usize,
    selected_model: &str,
    outcome: &LlmCallTraceOutcome,
) -> Json {
    let mut metadata = Map::from_iter([
        ("algorithm".into(), Value::String(algorithm)),
        (
            "switchyard.call_phase".into(),
            Value::String(
                match phase {
                    LlmCallPhase::Routing => "routing",
                    LlmCallPhase::Completion => "completion",
                }
                .into(),
            ),
        ),
        ("switchyard.candidate".into(), Value::from(candidate)),
        (
            "switchyard.candidate_count".into(),
            Value::from(candidate_count),
        ),
        (
            "selected_model".into(),
            Value::String(selected_model.into()),
        ),
        (
            "outcome".into(),
            Value::String(call_outcome_name(outcome).into()),
        ),
    ]);
    if let LlmCallTraceOutcome::Error { error_type } = outcome {
        metadata.insert("error.type".into(), Value::String(error_type.clone()));
    }
    Value::Object(metadata)
}

fn attempt_metadata(attempt: &UpstreamAttemptObservation) -> Json {
    let mut fields = Map::from_iter([
        (
            "selected_model".into(),
            Value::String(attempt.model.to_string()),
        ),
        (
            "wire_format".into(),
            Value::String(attempt.wire_format.to_string()),
        ),
        ("attempt".into(), Value::from(attempt.attempt)),
        ("max_attempts".into(), Value::from(attempt.max_attempts)),
        ("retry".into(), Value::from(attempt.attempt > 1)),
        (
            "outcome".into(),
            Value::String(attempt_outcome_name(attempt.outcome).into()),
        ),
        ("will_retry".into(), Value::from(attempt.will_retry)),
    ]);
    if let Some(status_code) = attempt.status_code {
        fields.insert("status_code".into(), Value::from(status_code));
    }
    if let Some(delay) = attempt.retry_delay {
        fields.insert("retry_delay_ms".into(), Value::from(duration_millis(delay)));
    }
    Value::Object(fields)
}

fn call_outcome_name(outcome: &LlmCallTraceOutcome) -> &'static str {
    match outcome {
        LlmCallTraceOutcome::Ok => "ok",
        LlmCallTraceOutcome::Error { .. } => "error",
        LlmCallTraceOutcome::Cancelled => "cancelled",
    }
}

fn attempt_outcome_name(outcome: UpstreamAttemptOutcome) -> &'static str {
    match outcome {
        UpstreamAttemptOutcome::Ok => "ok",
        UpstreamAttemptOutcome::Error => "error",
        UpstreamAttemptOutcome::Cancelled => "cancelled",
    }
}

fn status_metadata(outcome: &str) -> Json {
    json!({
        "otel.status_code": if outcome == "ok" { "OK" } else { "ERROR" },
    })
}

fn duration_millis(duration: std::time::Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, SystemTime};

    use switchyard_protocol::{ModelId, WireFormat};

    use super::*;

    #[test]
    fn projects_only_bounded_fields_and_redacts_models() {
        let started_at = SystemTime::UNIX_EPOCH + Duration::from_secs(10);
        let mut trace = RelayTrace(LlmCallTrace {
            algorithm: "first_available".into(),
            phase: LlmCallPhase::Completion,
            candidate: 2,
            candidate_count: 3,
            selected_model: ModelId::from("secret-model"),
            started_at,
            ended_at: started_at + Duration::from_secs(1),
            outcome: LlmCallTraceOutcome::Error {
                error_type: "503".into(),
            },
            attempts: vec![UpstreamAttemptObservation {
                model: ModelId::from("secret-model"),
                wire_format: WireFormat::OpenAiChat,
                attempt: 2,
                max_attempts: 3,
                started_at,
                ended_at: started_at + Duration::from_millis(5),
                outcome: UpstreamAttemptOutcome::Error,
                status_code: Some(503),
                will_retry: true,
                retry_delay: Some(Duration::from_millis(250)),
            }],
        });

        trace.sanitize(&ProviderKeyRedactor::new(&["secret-model".into()]));
        let metadata = client_metadata(
            trace.0.algorithm.clone(),
            trace.0.phase,
            trace.0.candidate,
            trace.0.candidate_count,
            trace.0.selected_model.as_str(),
            &trace.0.outcome,
        );
        assert_eq!(metadata["selected_model"], "[REDACTED]");
        assert_eq!(metadata["error.type"], "503");
        let attempt = attempt_metadata(&trace.0.attempts[0]);
        assert_eq!(attempt["selected_model"], "[REDACTED]");
        assert_eq!(attempt["retry_delay_ms"], 250);
        assert!(attempt.get("error").is_none());
    }
}
