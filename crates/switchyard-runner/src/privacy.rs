// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Privacy assessment contributing a fixed private-target requirement.

mod input;
mod semantic;

use strum_macros::IntoStaticStr;
use switchyard_protocol::{LlmClientError, Request};

use crate::eligibility::{Requirements, external_requirements};

const EXTERNAL_RESTRICTION_KEY: &str = "switchyard.internal.external_privacy_restriction";
use input::validate_input;
pub(crate) use semantic::SemanticPrivacyClassifier;

pub(crate) struct PrivacyPolicy {
    classifier: Option<SemanticPrivacyClassifier>,
    classifier_labels: Requirements,
}

/// Target set allowed to serve one request.
#[derive(Clone, Copy, IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum PrivacyLane {
    Standard,
    Restricted,
}

impl PrivacyLane {
    pub(crate) fn as_str(self) -> &'static str {
        self.into()
    }
}

/// Bounded privacy assessment, separate from other target requirements.
pub struct PrivacyDecision {
    lane: PrivacyLane,
    source: PrivacySource,
    reason_code: &'static str,
    clear_score: Option<f64>,
    clear_threshold: Option<f64>,
    retain_for_task: bool,
}

#[derive(Clone, Copy, IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
enum PrivacySource {
    Policy,
    ExternalSignal,
    SemanticClassifier,
    TaskRetention,
}

impl PrivacySource {
    fn as_str(self) -> &'static str {
        self.into()
    }
}

impl PrivacyDecision {
    /// Returns whether privacy permits standard targets or requires private targets.
    pub fn lane(&self) -> &'static str {
        self.lane.as_str()
    }

    /// Returns the policy component that selected the lane.
    pub fn source(&self) -> &'static str {
        self.source.as_str()
    }

    /// Returns a bounded reason for the decision.
    pub const fn reason_code(&self) -> &'static str {
        self.reason_code
    }

    /// Returns the classifier probability assigned to clear content, when available.
    pub const fn clear_score(&self) -> Option<f64> {
        self.clear_score
    }

    /// Returns the minimum clear score required for the standard lane, when applicable.
    pub const fn clear_threshold(&self) -> Option<f64> {
        self.clear_threshold
    }

    pub(crate) const fn selected_lane(&self) -> PrivacyLane {
        self.lane
    }

    pub(crate) const fn retain_for_task(&self) -> bool {
        self.retain_for_task
    }

    const fn new(lane: PrivacyLane, source: PrivacySource, reason_code: &'static str) -> Self {
        Self {
            lane,
            source,
            reason_code,
            clear_score: None,
            clear_threshold: None,
            retain_for_task: false,
        }
    }

    const fn semantic(
        lane: PrivacyLane,
        reason_code: &'static str,
        clear_score: Option<f64>,
        clear_threshold: f64,
    ) -> Self {
        Self {
            lane,
            source: PrivacySource::SemanticClassifier,
            reason_code,
            clear_score,
            clear_threshold: Some(clear_threshold),
            retain_for_task: false,
        }
    }

    const fn retained_for_task(mut self) -> Self {
        self.retain_for_task = true;
        self
    }

    const fn all_clear() -> Self {
        Self::new(PrivacyLane::Standard, PrivacySource::Policy, "all_clear")
    }

    pub(crate) const fn task_restriction(reason_code: &'static str) -> Self {
        Self::new(
            PrivacyLane::Restricted,
            PrivacySource::TaskRetention,
            reason_code,
        )
    }
}

/// Marks a request as requiring a route that accepts external privacy signals.
/// Execution fails if the selected route has not enabled `accept_external_signal`.
/// This marker is available only to trusted in-process hosts, not HTTP clients.
pub fn mark_privacy_restricted(request: &mut Request) {
    request
        .metadata
        .get_or_insert_default()
        .extra_metadata
        .get_or_insert_default()
        .insert(EXTERNAL_RESTRICTION_KEY.to_string(), String::new());
}

pub(crate) fn has_external_restriction(request: &Request) -> bool {
    request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.extra_metadata.as_ref())
        .is_some_and(|metadata| metadata.contains_key(EXTERNAL_RESTRICTION_KEY))
}

impl PrivacyPolicy {
    pub(crate) fn new(
        classifier: Option<SemanticPrivacyClassifier>,
        classifier_labels: Requirements,
    ) -> Self {
        Self {
            classifier,
            classifier_labels,
        }
    }

    pub(crate) async fn assess(
        &self,
        request: &Request,
        requirements: &Requirements,
        route: &str,
        algorithm: &str,
    ) -> Result<PrivacyDecision, LlmClientError> {
        if requirements.contains("private") {
            return Ok(if external_requirements(request)?.contains("private") {
                PrivacyDecision::new(
                    PrivacyLane::Restricted,
                    PrivacySource::ExternalSignal,
                    "restricted",
                )
                .retained_for_task()
            } else {
                PrivacyDecision::task_restriction("retained_restriction")
            });
        }
        let Some(classifier) = &self.classifier else {
            return Ok(PrivacyDecision::all_clear());
        };
        // Approval covers known constraints before any request content leaves the host.
        if !self.classifier_labels.is_superset(requirements) {
            return Err(LlmClientError::InvalidRequest {
                message: "privacy classifier target does not satisfy target requirements".into(),
            });
        }
        if let Err(reason_code) = validate_input(request) {
            return Ok(PrivacyDecision::new(
                PrivacyLane::Restricted,
                PrivacySource::Policy,
                reason_code,
            ));
        }
        Ok(classifier.assess(request, route, algorithm).await)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn known_private_requirements_skip_assessment() {
        let policy = PrivacyPolicy::new(None, Requirements::new());
        let requirements = Requirements::from(["private".into()]);
        for external in [false, true] {
            let mut request = Request::default();
            if external {
                crate::eligibility::require_target_labels(&mut request, ["private"])
                    .expect("trusted requirement");
            }
            let decision = policy
                .assess(&request, &requirements, "test/route", "test_algorithm")
                .await
                .expect("private requirement");
            assert_eq!(decision.lane(), "restricted");
            assert_eq!(
                decision.source(),
                if external {
                    "external_signal"
                } else {
                    "task_retention"
                }
            );
            assert_eq!(decision.retain_for_task(), external);
        }
    }
}
