// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::future::Future;
use std::time::SystemTime;

use nemo_relay_plugin::{Json, PluginRuntime, ScopeType};
use serde_json::{Map, Value};
use switchyard_runner::ProviderKeyRedactor;
use tracing::field::{Field, Visit};
use tracing::instrument::WithSubscriber;
use tracing::metadata::LevelFilter;
use tracing::span::{Attributes, Id, Record};
use tracing::subscriber::Interest;
use tracing::{Metadata, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;

const CLIENT_CALL: &str = "libsy.client_call";
const UPSTREAM_ATTEMPT: &str = "libsy.upstream_attempt";

#[derive(Debug)]
pub(crate) struct CapturedSpan {
    name: &'static str,
    fields: Map<String, Value>,
    started_at: SystemTime,
    ended_at: SystemTime,
    children: Vec<CapturedSpan>,
}

struct OpenSpan {
    name: &'static str,
    fields: Map<String, Value>,
    started_at: SystemTime,
    children: Vec<CapturedSpan>,
}

struct CaptureLayer<F> {
    emit: F,
}

impl<F> CaptureLayer<F> {
    fn new(emit: F) -> Self {
        Self { emit }
    }
}

impl<S, F> Layer<S> for CaptureLayer<F>
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
    F: Fn(CapturedSpan) + Send + Sync + 'static,
{
    fn register_callsite(&self, metadata: &'static Metadata<'static>) -> Interest {
        if captures(metadata) {
            Interest::always()
        } else {
            Interest::never()
        }
    }

    fn enabled(&self, metadata: &Metadata<'_>, _ctx: Context<'_, S>) -> bool {
        captures(metadata)
    }

    fn max_level_hint(&self) -> Option<LevelFilter> {
        Some(LevelFilter::DEBUG)
    }

    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let name = attrs.metadata().name();
        let mut fields = Map::new();
        attrs.record(&mut FieldVisitor(&mut fields));
        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(OpenSpan {
                name,
                fields,
                started_at: SystemTime::now(),
                children: Vec::new(),
            });
        }
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else {
            return;
        };
        let mut extensions = span.extensions_mut();
        if let Some(open) = extensions.get_mut::<OpenSpan>() {
            values.record(&mut FieldVisitor(&mut open.fields));
        }
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else {
            return;
        };
        let parent_id = span.parent().map(|parent| parent.id().clone());
        let Some(open) = span.extensions_mut().remove::<OpenSpan>() else {
            return;
        };
        let captured = CapturedSpan {
            name: open.name,
            fields: open.fields,
            started_at: open.started_at,
            ended_at: SystemTime::now(),
            children: open.children,
        };

        if captured.name == UPSTREAM_ATTEMPT
            && let Some(parent) = parent_id.and_then(|parent| ctx.span(&parent))
            && let Some(open) = parent.extensions_mut().get_mut::<OpenSpan>()
        {
            open.children.push(captured);
            return;
        }
        (self.emit)(captured);
    }
}

impl CapturedSpan {
    pub(crate) fn sanitize(&mut self, redactor: &ProviderKeyRedactor) {
        if let Json::Object(fields) = redactor.value(Json::Object(std::mem::take(&mut self.fields)))
        {
            self.fields = fields;
        }
        for child in &mut self.children {
            child.sanitize(redactor);
        }
    }

    pub(crate) fn emit(self, runtime: &PluginRuntime) -> Result<(), String> {
        let metadata = Json::Object(self.fields);
        let status = span_status(&metadata);
        let mut scope = runtime.scope_at(
            self.name,
            ScopeType::Custom,
            None,
            Some(&metadata),
            None,
            self.started_at,
        )?;
        let child_result = self
            .children
            .into_iter()
            .try_for_each(|child| child.emit(runtime));
        let close_result = scope.close_at(None, status.as_ref(), self.ended_at);
        child_result.and(close_result)
    }
}

/// Captures Switchyard client spans while polling one route execution.
///
/// This scoped registry replaces any tracing subscriber configured inside the plugin cdylib.
pub(crate) async fn capture_client_spans<T>(
    future: impl Future<Output = T>,
    emit: impl Fn(CapturedSpan) + Send + Sync + 'static,
) -> T {
    future
        .with_subscriber(tracing_subscriber::registry().with(CaptureLayer::new(emit)))
        .await
}

struct FieldVisitor<'a>(&'a mut Map<String, Value>);

impl FieldVisitor<'_> {
    fn insert(&mut self, field: &Field, value: impl FnOnce() -> Value) {
        if is_safe_field(field.name()) {
            self.0.insert(field.name().into(), value());
        }
    }
}

impl Visit for FieldVisitor<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.insert(field, || Value::String(format!("{value:?}")));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.insert(field, || Value::String(value.into()));
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.insert(field, || Value::Bool(value));
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.insert(field, || Value::from(value));
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.insert(field, || Value::from(value));
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        self.insert(field, || Value::from(value));
    }
}

fn captures(metadata: &Metadata<'_>) -> bool {
    metadata.target() == "libsy" && matches!(metadata.name(), CLIENT_CALL | UPSTREAM_ATTEMPT)
}

fn is_safe_field(name: &str) -> bool {
    matches!(
        name,
        "algorithm"
            | "switchyard.candidate"
            | "switchyard.candidate_count"
            | "switchyard.call_phase"
            | "selected_model"
            | "wire_format"
            | "attempt"
            | "max_attempts"
            | "retry"
            | "outcome"
            | "status_code"
            | "will_retry"
            | "retry_delay_ms"
            | "otel.status_code"
            | "error.type"
    )
}

fn span_status(metadata: &Json) -> Option<Json> {
    let outcome = metadata.get("outcome").and_then(Json::as_str);
    let status = metadata.get("otel.status_code").and_then(Json::as_str);
    match (status, outcome) {
        (Some("ERROR"), _) | (_, Some("error" | "cancelled")) => {
            Some(serde_json::json!({"otel.status_code": "ERROR"}))
        }
        (Some("OK"), _) | (_, Some("ok")) => Some(serde_json::json!({"otel.status_code": "OK"})),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::fmt;
    use std::sync::{Arc, Mutex};

    use tracing::Instrument;

    use super::*;

    struct PanicOnDebug;

    impl fmt::Debug for PanicOnDebug {
        fn fmt(&self, _formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            panic!("disallowed fields must not be formatted");
        }
    }

    #[tokio::test]
    async fn captures_only_bounded_client_and_attempt_fields() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let observed = Arc::clone(&captured);
        let result = capture_client_spans(
            async {
                async {}
                    .instrument(tracing::info_span!(
                        target: "other",
                        CLIENT_CALL,
                        outcome = "ignored",
                    ))
                    .await;
                async {
                    async {}
                        .instrument(tracing::debug_span!(
                            target: "libsy",
                            UPSTREAM_ATTEMPT,
                            attempt = 1_u64,
                            outcome = "error",
                            error = "provider body",
                        ))
                        .await;
                    tracing::Span::current().record("outcome", "ok");
                }
                .instrument(tracing::info_span!(
                    target: "libsy",
                    CLIENT_CALL,
                    selected_model = "target/model",
                    outcome = tracing::field::Empty,
                    error = ?PanicOnDebug,
                ))
                .await;
                7
            },
            move |span| {
                observed
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(span);
            },
        )
        .await;

        assert_eq!(result, 7);
        let mut spans = captured
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let [call] = spans.as_mut_slice() else {
            panic!("expected one client call: {spans:?}");
        };
        assert_eq!(call.name, CLIENT_CALL);
        assert_eq!(call.fields["selected_model"], "target/model");
        assert_eq!(call.fields["outcome"], "ok");
        assert!(!call.fields.contains_key("error"));
        let [attempt] = call.children.as_slice() else {
            panic!("expected one upstream attempt: {call:?}");
        };
        assert_eq!(attempt.name, UPSTREAM_ATTEMPT);
        assert_eq!(attempt.fields["attempt"], 1);
        assert_eq!(attempt.fields["outcome"], "error");
        assert!(!attempt.fields.contains_key("error"));
        assert!(call.started_at <= attempt.started_at);
        assert!(attempt.ended_at <= call.ended_at);

        call.sanitize(&ProviderKeyRedactor::new(&["target/model".into()]));
        assert_eq!(call.fields["selected_model"], "[REDACTED]");
    }
}
