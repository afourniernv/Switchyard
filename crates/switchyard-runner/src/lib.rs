// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared configured routing for Switchyard serving surfaces.

mod algorithm;
#[cfg(feature = "client")]
mod config;
#[cfg(feature = "client")]
mod failure;
mod provider_key_redactor;
#[cfg(feature = "client")]
mod route;
#[cfg(feature = "client")]
mod runner;

#[cfg(feature = "client")]
pub use config::api_key_variables;

pub use algorithm::{
    AdvisorTriggerConfig, AlgorithmConfigError, AlgorithmSpec, CategoryModelConfig, ClassifierMode,
    ClassifierPolicyConfig, DecisionJudgeRouteConfig, LlmClassifierRouteConfig,
    StageClassifierConfig, SubagentRouteConfig, VgrModeConfig, VgrRouteConfig,
};
#[cfg(feature = "client")]
pub use failure::{RouteErrorKind, RouteErrorPhase, RouteErrorSummary, stream_error_summary};
// Re-exported because `Route::new` takes it, so a host wiring routes does not need a libsy dep.
pub use libsy::RuntimeModels;
pub use provider_key_redactor::ProviderKeyRedactor;
#[cfg(feature = "client")]
pub use route::{
    AuxiliaryTarget, CallerAuthKind, ModelCapabilities, Route, RunOutput, RunnerError,
};
#[cfg(feature = "client")]
pub use runner::{DecisionDescription, DecisionTarget, ModelInfo, Runner};
