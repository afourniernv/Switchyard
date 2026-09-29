// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-scoped observations emitted while serving an algorithm run.

use std::sync::Arc;
use std::time::{Duration, SystemTime};

use switchyard_libsy::OutcomeMetadata;
use switchyard_protocol::{ModelId, UpstreamAttemptObservation, Usage};

/// Role of a model call within one Switchyard route execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LlmCallPhase {
    /// A call made by the routing algorithm.
    Routing,
    /// A call made to produce the final answer.
    Completion,
}

/// Terminal outcome of a completed model call trace.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LlmCallTraceOutcome {
    /// The call completed successfully.
    Ok,
    /// The call failed with a bounded error classification.
    Error {
        /// Stable error classification without provider response content.
        error_type: String,
    },
    /// The caller abandoned an open response stream.
    Cancelled,
}

/// Completed payload-free trace for one Switchyard model call and its attempts.
#[derive(Clone, Debug)]
pub struct LlmCallTrace {
    /// Routing algorithm that initiated the call.
    pub algorithm: String,
    /// Whether the call served routing or final completion work.
    pub phase: LlmCallPhase,
    /// One-based candidate position for ordered completion fallback.
    pub candidate: usize,
    /// Total candidates available to this call site.
    pub candidate_count: usize,
    /// Model selected for the call.
    pub selected_model: ModelId,
    /// Wall-clock start time paired with a monotonic duration.
    pub started_at: SystemTime,
    /// Derived wall-clock end time.
    pub ended_at: SystemTime,
    /// Terminal call outcome.
    pub outcome: LlmCallTraceOutcome,
    /// Physical upstream attempts completed beneath this call.
    pub attempts: Vec<UpstreamAttemptObservation>,
}

/// Request-scoped callback for completed model call traces.
pub type RunTraceObserver = Arc<dyn Fn(LlmCallTrace) + Send + Sync>;

/// One completed model call observed while serving an algorithm run.
#[derive(Clone, Debug)]
pub struct LlmCallObservation {
    /// Model selected for the completed call.
    pub selected_model: ModelId,
    /// Whether the call completed successfully.
    pub is_success: bool,
    /// Time spent waiting for the model call to resolve.
    pub duration: Duration,
    /// Normalized usage for a buffered successful response.
    pub usage: Option<Usage>,
}

/// Events emitted inline while [`crate::run`] serves a routing request.
#[derive(Clone, Debug)]
pub enum RunObservation {
    /// Metadata attached to the completed routing outcome.
    Outcome(OutcomeMetadata),
    /// A completed model call requested by the algorithm for routing work.
    LlmCall(LlmCallObservation),
    /// A completed terminal model call made from the routing outcome.
    AnswerCall(LlmCallObservation),
    /// Routing time recorded by the `switchyard.routing_overhead_ms` metric.
    RoutingOverhead(Duration),
}

/// Request-scoped callback for algorithm-run observations.
///
/// The runner invokes this callback inline while resolving observations. Several
/// runs may invoke the same observer concurrently, so implementations must be
/// thread-safe, fast, and non-blocking.
pub type RunObserver = Arc<dyn Fn(RunObservation) + Send + Sync>;
