// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The routed-call server trait and its shared error types.
//!
//! [`RoutedLlmClient`] is the one piece of I/O the protocol does not own: a host
//! implements it to actually perform a model call. It lives here — rather than in
//! libsy's orchestration crate — so a client crate that depends only on the protocol
//! can serve routed calls without pulling in the orchestrator.

use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use thiserror::Error;

use crate::{ModelId, Request, Response, WireFormat};

/// Terminal outcome of one physical upstream request attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UpstreamAttemptOutcome {
    /// The upstream accepted the request and returned a usable response.
    Ok,
    /// The attempt failed before a usable response was returned.
    Error,
    /// The caller cancelled the request while the attempt was in flight.
    Cancelled,
}

/// Bounded, payload-free telemetry for one physical upstream request attempt.
#[derive(Clone, Debug)]
pub struct UpstreamAttemptObservation {
    /// Target model selected for the request.
    pub model: ModelId,
    /// Provider wire format used for the request.
    pub wire_format: WireFormat,
    /// One-based attempt number.
    pub attempt: u64,
    /// Maximum attempts allowed by the backend retry policy.
    pub max_attempts: u64,
    /// Wall-clock start time paired with a monotonic duration.
    pub started_at: SystemTime,
    /// Derived wall-clock end time.
    pub ended_at: SystemTime,
    /// Terminal attempt outcome.
    pub outcome: UpstreamAttemptOutcome,
    /// Upstream HTTP status when one was received.
    pub status_code: Option<u16>,
    /// Whether another attempt follows this one.
    pub will_retry: bool,
    /// Retry delay selected after this attempt.
    pub retry_delay: Option<Duration>,
}

/// Thread-safe callback for completed upstream attempts.
pub type UpstreamAttemptObserver = dyn Fn(UpstreamAttemptObservation) + Send + Sync;

/// A boxed client-specific error preserved as the source of a routed call failure.
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Failures a routed LLM client can surface to its caller.
///
/// The variants classify failures that routing hosts commonly need to handle,
/// while boxed sources preserve implementation-specific detail. `General` is the
/// escape hatch for failures that do not fit a shared category.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum LlmClientError {
    /// The request cannot be served as supplied.
    #[error("invalid request: {message}")]
    InvalidRequest {
        /// Human-readable request validation failure.
        message: String,
    },

    /// Decoding the inbound request failed in the translation engine.
    #[error("request translation failed: {0}")]
    RequestTranslation(String),

    /// Encoding the request for the upstream failed in the translation engine.
    #[error("outbound request encoding failed: {0}")]
    RequestEncoding(String),

    /// Decoding or encoding the response failed in the translation engine.
    #[error("response translation failed: {0}")]
    ResponseTranslation(String),

    /// The client is not configured to serve the selected target.
    #[error("client configuration error: {message}")]
    Configuration {
        /// Human-readable configuration failure.
        message: String,
    },

    /// The route cannot record another provider-owned response or conversation ID.
    #[error(
        "Responses state tracking reached its limit of {limit} IDs; no existing records were removed"
    )]
    ResponseStateLimitExceeded {
        /// Maximum number of IDs retained by this route.
        limit: usize,
    },

    /// Two configured models reported the same saved response or conversation ID.
    #[error("Responses state ID is already recorded for another model; its owner was not changed")]
    ResponseStateConflict,

    /// The selected backend is temporarily unavailable. Another candidate may serve the request.
    #[error("backend temporarily unavailable")]
    TemporarilyUnavailable,

    /// The upstream could not be reached or the request could not be sent.
    #[error("upstream transport error: {source}")]
    Transport {
        /// Client-specific transport failure.
        #[source]
        source: BoxError,
    },

    /// The upstream request exceeded its timeout.
    #[error("upstream request timed out: {source}")]
    Timeout {
        /// Client-specific timeout failure.
        #[source]
        source: BoxError,
    },

    /// The upstream rejected the request because it exceeds the model's context window.
    #[error("context window exceeded for model {model}: {message}")]
    ContextWindowExceeded {
        /// Model whose context window was exceeded.
        model: ModelId,
        /// Upstream error message.
        message: String,
    },

    /// The upstream returned a non-success HTTP response.
    #[error("upstream returned HTTP {status}: {body}")]
    UpstreamHttp {
        /// Upstream HTTP status code.
        status: http::StatusCode,
        /// Raw upstream error body.
        body: String,
    },

    /// The upstream returned a response the client could not decode.
    #[error("invalid upstream response: {source}")]
    InvalidResponse {
        /// Client-specific decoding or validation failure.
        #[source]
        source: BoxError,
    },

    /// A call across a foreign-function boundary (e.g. a Python-implemented client)
    /// failed. The boxed source is the foreign error itself.
    #[error("foreign function interface error: {source}")]
    Ffi {
        /// Foreign-language failure, preserved verbatim.
        #[source]
        source: BoxError,
    },

    /// A string message. Useful in testing, but prefer adding variants over using this.
    #[error("{0}")]
    General(String),
}

/// Why routing replaced a selected target with another eligible target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoutingFallbackReason {
    /// The selected target rejected the request because its context window was too small.
    ContextWindow,
    /// The selected target was unavailable after its client retries finished.
    Unavailable,
}

impl RoutingFallbackReason {
    /// Stable value used when logging a routing fallback.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ContextWindow => "context_window",
            Self::Unavailable => "unavailable",
        }
    }
}

/// Performs the actual model call for a target. This is the one piece of I/O the
/// library does not own — a host implements it over its own transport (HTTP SDK,
/// in-process model, mock). It serves a call the stream consumer chose not to
/// override, reached as a routed request's `default_client`.
///
/// # Concurrency
///
/// A client may be shared by many targets and concurrent algorithm runs. Calls may
/// overlap, so implementations must synchronize mutable state internally and should
/// not serialize requests unless their transport requires it.
#[async_trait]
pub trait RoutedLlmClient: Send + Sync {
    /// Make a request
    async fn call(&self, request: Request) -> Result<Response, LlmClientError>;

    /// Make a request and report completed physical upstream attempts when supported.
    ///
    /// Implementations without an attempt boundary may rely on this default and
    /// still participate in routing. The observer never receives request or
    /// response payloads.
    async fn call_with_attempt_observer(
        &self,
        request: Request,
        observer: &UpstreamAttemptObserver,
    ) -> Result<Response, LlmClientError> {
        let _ = observer;
        self.call(request).await
    }
}
