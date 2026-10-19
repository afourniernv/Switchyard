// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configured route execution and the clients that serve its targets.

use std::collections::BTreeMap;
use std::error::Error;
use std::sync::Arc;

use libsy::{Algorithm, LibsyError, RoutingOutcome, RuntimeModels};
use serde_json::Value;
use switchyard_llm_client::{AuxiliaryOperation, ClientRouter, RunObserver, TranslatingLlmClient};
use switchyard_protocol::{LlmClientError, ModelId, Request, Response, WireFormat};
use thiserror::Error;

use crate::DecisionTarget;
use crate::eligibility::{EligibilityPolicy, Requirements, has_external_requirements};
use crate::privacy::{PrivacyDecision, PrivacyLane, PrivacyPolicy, has_external_restriction};

/// Capabilities declared for one route.
///
/// `GET /v1/models` includes `context_window`, `tool_calling`, and `vision` in each
/// standard `data` entry, using `null` for unset values. It omits `reasoning`.
/// The server rejects tool inputs, reasoning controls, or images when the
/// corresponding declaration is explicitly `false` for the selected route.
#[derive(Clone, Copy, Default)]
pub struct ModelCapabilities {
    pub context_window: Option<u32>,
    pub tool_calling: Option<bool>,
    /// Whether the routed model accepts reasoning controls, as declared in config.
    pub reasoning: Option<bool>,
    /// Whether the routed model accepts image input, as declared in config.
    pub vision: Option<bool>,
}

/// Caller credential family required by a forwarded-auth route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CallerAuthKind {
    Anthropic,
    OpenAi,
}

impl CallerAuthKind {
    /// Stable provider name used by the server compatibility API.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAi => "openai",
        }
    }

    const fn accepts(self, wire_format: WireFormat) -> bool {
        matches!(
            (self, wire_format),
            (Self::Anthropic, WireFormat::AnthropicMessages)
                | (
                    Self::OpenAi,
                    WireFormat::OpenAiChat | WireFormat::OpenAiResponses
                )
        )
    }
}

/// Exact upstream model and client used for an auxiliary provider operation.
#[derive(Clone)]
pub struct AuxiliaryTarget {
    pub model: ModelId,
    pub client: Arc<TranslatingLlmClient>,
}

/// Error returned while loading or executing configured routes.
#[derive(Debug, Error)]
pub enum RunnerError {
    #[error("{message}")]
    Configuration {
        message: String,
        #[source]
        source: Option<Box<dyn Error + Send + Sync>>,
    },
    #[error("unknown route model {0:?}")]
    UnknownRouteModel(String),
    #[error("caller format is incompatible with {} credentials", .0.as_str())]
    IncompatibleCallerFormat(CallerAuthKind),
    #[error("route has no compatible target for the auxiliary operation")]
    AuxiliaryUnsupported,
    #[error(transparent)]
    Algorithm(#[from] LibsyError),
    #[error(transparent)]
    Client(#[from] LlmClientError),
}

impl RunnerError {
    pub(crate) fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration {
            message: message.into(),
            source: None,
        }
    }

    pub(crate) fn configuration_source(
        message: impl Into<String>,
        source: impl Error + Send + Sync + 'static,
    ) -> Self {
        Self::Configuration {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }
}

// Configured algorithm and targets for one execution lane.
pub(crate) struct ExecutionLane {
    algorithm: Arc<dyn Algorithm>,
    // Resolves each offloaded call to the client configured for the target the algorithm
    // selected. A route is a synthetic model with no upstream of its own, so this is a
    // per-target lookup, never one client serving the whole route.
    clients: ClientRouter,
    anthropic_auxiliary_target: Option<AuxiliaryTarget>,
    responses_auxiliary_target: Option<AuxiliaryTarget>,
    models: Arc<RuntimeModels>,
}

impl ExecutionLane {
    pub(crate) fn new(
        algorithm: Arc<dyn Algorithm>,
        clients: ClientRouter,
        anthropic_auxiliary_target: Option<AuxiliaryTarget>,
        responses_auxiliary_target: Option<AuxiliaryTarget>,
        models: RuntimeModels,
    ) -> Self {
        Self {
            algorithm,
            clients,
            anthropic_auxiliary_target,
            responses_auxiliary_target,
            models: Arc::new(models),
        }
    }
}

struct EligibilityExecution {
    policy: EligibilityPolicy,
    groups: BTreeMap<Requirements, ExecutionLane>,
}

struct LaneSelection<'a> {
    lane: &'a ExecutionLane,
    privacy: Option<PrivacyDecision>,
}

/// One or more execution lanes behind a configured route.
pub struct Route {
    standard: ExecutionLane,
    privacy: Option<PrivacyPolicy>,
    eligibility: Option<EligibilityExecution>,
    decision_targets: Vec<DecisionTarget>,
    caller_auth: Option<CallerAuthKind>,
    capabilities: ModelCapabilities,
}

/// The selected model and untouched response produced by a route execution.
pub struct RunOutput {
    pub selected_model: ModelId,
    pub response: Response,
}

impl Route {
    /// Creates a fully configured execution route.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        algorithm: Arc<dyn Algorithm>,
        clients: ClientRouter,
        caller_auth: Option<CallerAuthKind>,
        capabilities: ModelCapabilities,
        anthropic_auxiliary_target: Option<AuxiliaryTarget>,
        responses_auxiliary_target: Option<AuxiliaryTarget>,
        decision_targets: Vec<DecisionTarget>,
        models: RuntimeModels,
    ) -> Self {
        Self::from_lane(
            ExecutionLane::new(
                algorithm,
                clients,
                anthropic_auxiliary_target,
                responses_auxiliary_target,
                models,
            ),
            caller_auth,
            capabilities,
            decision_targets,
        )
    }

    pub(crate) fn from_lane(
        lane: ExecutionLane,
        caller_auth: Option<CallerAuthKind>,
        capabilities: ModelCapabilities,
        decision_targets: Vec<DecisionTarget>,
    ) -> Self {
        Self {
            standard: lane,
            privacy: None,
            eligibility: None,
            decision_targets,
            caller_auth,
            capabilities,
        }
    }

    pub(crate) fn with_privacy_policy(mut self, policy: PrivacyPolicy) -> Self {
        self.privacy = Some(policy);
        self
    }

    pub(crate) fn with_eligibility(
        mut self,
        policy: EligibilityPolicy,
        groups: BTreeMap<Requirements, ExecutionLane>,
    ) -> Self {
        self.eligibility = Some(EligibilityExecution { policy, groups });
        self
    }

    async fn select_lane(&self, request: &Request) -> Result<LaneSelection<'_>, RunnerError> {
        if let Some(eligibility) = &self.eligibility {
            let mut requirements = eligibility.policy.admit(request)?;
            let mut privacy = match &self.privacy {
                Some(policy) => Some(
                    policy
                        .assess(
                            request,
                            &requirements,
                            request.llm_request.model.as_deref().unwrap_or_default(),
                            self.algorithm_name(),
                        )
                        .await?,
                ),
                None => None,
            };
            let retain = privacy
                .as_ref()
                .is_some_and(PrivacyDecision::retain_for_task);
            if privacy
                .as_ref()
                .is_some_and(|decision| matches!(decision.selected_lane(), PrivacyLane::Restricted))
            {
                requirements.insert("private".into());
            }
            if privacy.is_some() {
                requirements = eligibility.policy.finish(request, requirements, retain)?;
            }
            if let Some(decision) = &mut privacy
                && requirements.contains("private")
                && matches!(decision.selected_lane(), PrivacyLane::Standard)
            {
                *decision = PrivacyDecision::task_restriction("retained_restriction");
            }
            let lane = if requirements.is_empty() {
                &self.standard
            } else {
                eligibility.groups.get(&requirements).ok_or_else(|| {
                    RunnerError::configuration("eligible execution group was not constructed")
                })?
            };
            tracing::debug!(
                switchyard.route = request.llm_request.model.as_deref().unwrap_or_default(),
                switchyard.algorithm = self.algorithm_name(),
                eligibility.required_labels = ?requirements,
                privacy.lane = privacy.as_ref().map(PrivacyDecision::lane),
                privacy.source = privacy.as_ref().map(PrivacyDecision::source),
                privacy.reason_code = privacy.as_ref().map(PrivacyDecision::reason_code),
                privacy.clear_score = privacy.as_ref().and_then(PrivacyDecision::clear_score),
                privacy.clear_threshold = privacy.as_ref().and_then(PrivacyDecision::clear_threshold),
                "eligible targets selected"
            );
            return Ok(LaneSelection { lane, privacy });
        }
        if has_external_requirements(request) || has_external_restriction(request) {
            return Err(LlmClientError::InvalidRequest {
                message: "route does not accept target requirements".to_string(),
            }
            .into());
        }
        Ok(LaneSelection {
            lane: &self.standard,
            privacy: None,
        })
    }

    /// Returns the configured libsy algorithm name.
    pub fn algorithm_name(&self) -> &str {
        self.standard.algorithm.name()
    }

    /// Returns model-list capability metadata.
    pub fn capabilities(&self) -> ModelCapabilities {
        self.capabilities
    }

    /// Returns the forwarded caller credential family.
    pub fn caller_auth(&self) -> Option<CallerAuthKind> {
        self.caller_auth
    }

    /// Resolves a selected model to this route's non-secret target metadata.
    pub(crate) fn decision_target(&self, model: &ModelId) -> Option<DecisionTarget> {
        self.decision_targets
            .iter()
            .find(|target| target.model == *model)
            .cloned()
    }

    /// Returns the models grouped for the standard execution lane.
    pub fn models(&self) -> &RuntimeModels {
        &self.standard.models
    }

    /// Rejects a caller format incompatible with forwarded credentials.
    pub fn check_caller_format(&self, input_format: WireFormat) -> Result<(), RunnerError> {
        if let Some(kind) = self.caller_auth
            && !kind.accepts(input_format)
        {
            return Err(RunnerError::IncompatibleCallerFormat(kind));
        }
        Ok(())
    }

    /// Executes the configured route without consuming or proxying streamed responses.
    pub async fn execute(
        &self,
        request: Request,
        observer: Option<RunObserver>,
    ) -> Result<RunOutput, RunnerError> {
        self.execute_with_privacy_observer(request, observer, |_| {})
            .await
    }

    /// Executes the route and reports a configured privacy decision before its lane runs.
    pub async fn execute_with_privacy_observer(
        &self,
        request: Request,
        observer: Option<RunObserver>,
        on_privacy_decision: impl FnOnce(&PrivacyDecision) + Send,
    ) -> Result<RunOutput, RunnerError> {
        let selection = self.select_lane(&request).await?;
        if let Some(decision) = &selection.privacy {
            on_privacy_decision(decision);
        }
        let lane = selection.lane;
        let (selected_model, response) = switchyard_llm_client::run(
            Arc::clone(&lane.algorithm),
            lane.clients.clone(),
            request,
            Arc::clone(&lane.models),
            observer,
        )
        .await?;
        Ok(RunOutput {
            selected_model,
            response,
        })
    }

    /// Completes routing-time calls without serving a post-routing completion.
    pub async fn decide(&self, request: Request) -> Result<RoutingOutcome, RunnerError> {
        let selection = self.select_lane(&request).await?;
        let lane = selection.lane;
        switchyard_llm_client::decide(
            Arc::clone(&lane.algorithm),
            lane.clients.clone(),
            request,
            Arc::clone(&lane.models),
        )
        .await
        .map_err(RunnerError::from)
    }

    /// Executes a model-bearing provider operation through a compatible target.
    pub async fn call_auxiliary(
        &self,
        request: Request,
        operation: AuxiliaryOperation,
    ) -> Result<Value, RunnerError> {
        let selection = self.select_lane(&request).await?;
        let lane = selection.lane;
        let target = match operation {
            AuxiliaryOperation::AnthropicCountTokens => &lane.anthropic_auxiliary_target,
            AuxiliaryOperation::ResponsesInputTokens | AuxiliaryOperation::ResponsesCompact => {
                &lane.responses_auxiliary_target
            }
        }
        .as_ref()
        .ok_or(RunnerError::AuxiliaryUnsupported)?;
        target
            .client
            .call_auxiliary(&target.model, request, operation)
            .await
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use libsy::{Driver, Passthrough};
    use reqwest::StatusCode;
    use switchyard_protocol::{
        Category, ContentBlock, DecisionAnswer, DecisionRequest, DecisionResponse, DecisionValue,
        LlmResponse, Message, Probability, Role, RoutedDecisionClient, RoutedLlmClient, Usage,
        text_response,
    };

    use super::*;
    use crate::privacy::{SemanticPrivacyClassifier, mark_privacy_restricted};

    impl Route {
        fn with_privacy(self, policy: PrivacyPolicy, restricted: ExecutionLane) -> Self {
            let requirements = Requirements::from(["private".into()]);
            self.with_eligibility(
                EligibilityPolicy::new(true, vec![vec!["private".into()]], false)
                    .expect("privacy requirement set"),
                BTreeMap::from([(requirements, restricted)]),
            )
            .with_privacy_policy(policy)
        }
    }

    enum RoutingCall {
        Llm,
        Decision,
    }

    struct CallThenRoute(RoutingCall);

    #[async_trait::async_trait]
    impl Algorithm for CallThenRoute {
        fn name(&self) -> &str {
            "call_then_route"
        }

        async fn route(
            self: Arc<Self>,
            driver: Driver,
            request: Request,
        ) -> libsy::Result<RoutingOutcome> {
            match &self.0 {
                RoutingCall::Llm => {
                    driver
                        .call_model(
                            request.clone(),
                            driver.models_for(&Category::Judge).to_vec(),
                        )
                        .await?;
                }
                RoutingCall::Decision => {
                    let judge = driver.first_model_for(&Category::Judge)?.clone();
                    driver
                        .call_decision(
                            DecisionRequest {
                                model: None,
                                context: Value::Null,
                                questions: BTreeMap::new(),
                            },
                            judge,
                        )
                        .await?;
                }
            }
            let (selected, fallbacks) = driver
                .models_for(&Category::Any)
                .split_first()
                .ok_or(LibsyError::NoTargets)?;
            Ok(RoutingOutcome::route_to(
                selected.clone(),
                fallbacks.to_vec(),
                request,
            ))
        }
    }

    struct RecordingClient {
        calls: Arc<Mutex<Vec<ModelId>>>,
    }

    struct VerdictClient {
        calls: Arc<AtomicUsize>,
        gate: Option<(Arc<tokio::sync::Notify>, Arc<tokio::sync::Notify>)>,
    }

    #[async_trait::async_trait]
    impl RoutedLlmClient for RecordingClient {
        async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
            let model = request
                .llm_request
                .model
                .map(ModelId::from)
                .expect("driver should stamp a model");
            self.calls.lock().expect("calls lock").push(model.clone());
            if model == "restricted-primary" {
                return Err(LlmClientError::UpstreamHttp {
                    status: StatusCode::SERVICE_UNAVAILABLE,
                    body: "unavailable".to_string(),
                });
            }
            Ok(Response {
                llm_response: LlmResponse::Agg(text_response(Some(model.to_string()), "ok")),
                metadata: None,
                upstream_headers: Default::default(),
            })
        }
    }

    #[async_trait::async_trait]
    impl RoutedDecisionClient for VerdictClient {
        async fn call(
            &self,
            _request: DecisionRequest,
        ) -> Result<DecisionResponse, LlmClientError> {
            if self.calls.fetch_add(1, Ordering::Relaxed) == 0
                && let Some((started, release)) = &self.gate
            {
                started.notify_one();
                release.notified().await;
            }
            let mut probabilities = [
                "no_sensitive_content",
                "personal_data",
                "credentials",
                "confidential_data",
                "regulated_data",
                "uncertain",
            ]
            .into_iter()
            .map(|id| (id.into(), Probability(0.0)))
            .collect::<BTreeMap<_, _>>();
            probabilities.insert("no_sensitive_content".into(), Probability(0.95));
            probabilities.insert("uncertain".into(), Probability(0.05));
            Ok(DecisionResponse {
                id: None,
                model: None,
                answers: BTreeMap::from([(
                    "privacy".into(),
                    DecisionAnswer {
                        value: DecisionValue::Choice {
                            selected: "no_sensitive_content".into(),
                            probabilities: Some(probabilities),
                        },
                        provider_confidence: None,
                    },
                )]),
                usage: Usage::default(),
            })
        }
    }

    fn lane(
        algorithm: Arc<dyn Algorithm>,
        client: Arc<dyn RoutedLlmClient>,
        models: RuntimeModels,
        auxiliary_model: &str,
    ) -> ExecutionLane {
        let routed_models = models
            .models_for(&Category::Any)
            .iter()
            .chain(models.models_for(&Category::Judge))
            .cloned()
            .map(|model| (model, Arc::clone(&client)))
            .collect();
        let auxiliary_client =
            Arc::new(TranslatingLlmClient::new(&[]).expect("empty test client should be valid"));
        ExecutionLane::new(
            algorithm,
            ClientRouter::new(routed_models),
            Some(AuxiliaryTarget {
                model: auxiliary_model.into(),
                client: auxiliary_client,
            }),
            None,
            models,
        )
    }

    fn restricted_request() -> Request {
        let mut request = Request::default();
        mark_privacy_restricted(&mut request);
        request
    }

    fn decision_target(target: &str, model: &ModelId) -> DecisionTarget {
        DecisionTarget {
            target: target.to_string(),
            model: model.clone(),
            format: WireFormat::OpenAiChat,
            base_url: format!("https://{target}.example/v1"),
            extra_body: BTreeMap::new(),
        }
    }

    // Rejected content must not erase authoritative requirements supplied with that turn.
    #[tokio::test]
    async fn rejected_turn_retains_requirements() {
        let client: Arc<dyn RoutedLlmClient> = Arc::new(RecordingClient {
            calls: Arc::new(Mutex::new(Vec::new())),
        });
        let build_lane = |name: &str| {
            lane(
                Arc::new(Passthrough),
                Arc::clone(&client),
                RuntimeModels::new(HashMap::from([(Category::Any, vec![name.into()])])),
                name,
            )
        };
        let groups = BTreeMap::from([
            (Requirements::from(["eu".into()]), build_lane("eu")),
            (
                Requirements::from(["eu".into(), "private".into()]),
                build_lane("private-eu"),
            ),
        ]);
        let policy = EligibilityPolicy::new(
            true,
            vec![vec!["eu".into()], vec!["eu".into(), "private".into()]],
            true,
        )
        .expect("configured requirement sets");
        let route = Route::from_lane(
            build_lane("standard"),
            None,
            ModelCapabilities::default(),
            Vec::new(),
        )
        .with_eligibility(policy, groups);
        let mut request = Request::default();
        request.metadata.get_or_insert_default().task_id = Some("task".into());
        crate::require_target_labels(&mut request, ["eu"]).expect("trusted labels");
        let outcome = route.decide(request.clone()).await.expect("EU decision");
        assert_eq!(outcome.selected_model_id().expect("model"), "eu");

        crate::require_target_labels(&mut request, ["private"]).expect("trusted labels");
        request.llm_request.extensions.fields.insert(
            "previous_response_id".into(),
            Value::String("hidden".into()),
        );
        assert!(route.decide(request).await.is_err());
        let mut next = Request::default();
        next.metadata.get_or_insert_default().task_id = Some("task".into());
        let outcome = route.decide(next).await.expect("retained decision");
        assert_eq!(outcome.selected_model_id().expect("model"), "private-eu");
    }

    #[tokio::test]
    async fn requirements_cover_judges_fallbacks_and_auxiliary_calls() {
        for generic in [false, true] {
            let calls = Arc::new(Mutex::new(Vec::new()));
            let client: Arc<dyn RoutedLlmClient> = Arc::new(RecordingClient {
                calls: Arc::clone(&calls),
            });
            let standard = lane(
                Arc::new(Passthrough),
                Arc::clone(&client),
                RuntimeModels::new(HashMap::from([(
                    Category::Any,
                    vec!["standard-primary".into()],
                )])),
                "standard-auxiliary",
            );
            let restricted = lane(
                Arc::new(CallThenRoute(RoutingCall::Llm)),
                client,
                RuntimeModels::new(HashMap::from([
                    (Category::Judge, vec!["restricted-judge".into()]),
                    (
                        Category::Any,
                        vec!["restricted-primary".into(), "restricted-fallback".into()],
                    ),
                ])),
                "restricted-auxiliary",
            );
            let route = Route::from_lane(standard, None, ModelCapabilities::default(), Vec::new());
            let route = if generic {
                let requirements = Requirements::from(["eu".into(), "private".into()]);
                route.with_eligibility(
                    EligibilityPolicy::new(
                        true,
                        vec![requirements.iter().cloned().collect()],
                        false,
                    )
                    .expect("configured labels"),
                    BTreeMap::from([(requirements, restricted)]),
                )
            } else {
                route.with_privacy(PrivacyPolicy::new(None, Requirements::new()), restricted)
            };
            let request = || {
                if generic {
                    let mut request = Request::default();
                    crate::require_target_labels(&mut request, ["private", "eu"])
                        .expect("trusted labels");
                    request
                } else {
                    restricted_request()
                }
            };

            let mut observed = None;
            let output = route
                .execute_with_privacy_observer(request(), None, |decision| {
                    observed = Some((decision.lane(), decision.source(), decision.reason_code()));
                })
                .await
                .expect("restricted fallback should answer");
            assert_eq!(
                observed,
                (!generic).then_some(("restricted", "external_signal", "restricted"))
            );
            assert_eq!(output.selected_model, "restricted-primary");
            assert_eq!(
                output.response.served_model().map(ModelId::as_str),
                Some("restricted-fallback")
            );
            assert_eq!(
                *calls.lock().expect("calls lock"),
                [
                    ModelId::from("restricted-judge"),
                    ModelId::from("restricted-primary"),
                    ModelId::from("restricted-fallback"),
                ]
            );

            let error = route
                .call_auxiliary(request(), AuxiliaryOperation::AnthropicCountTokens)
                .await
                .expect_err("empty auxiliary client should reject the model");
            assert!(
                error.to_string().contains("restricted-auxiliary"),
                "{error}"
            );
        }
    }

    #[tokio::test]
    async fn restricted_typed_decision_resolves_configured_target_metadata() {
        let answer: ModelId = "restricted-answer".into();
        let llm: Arc<dyn RoutedLlmClient> = Arc::new(RecordingClient {
            calls: Arc::new(Mutex::new(Vec::new())),
        });
        let standard = lane(
            Arc::new(Passthrough),
            Arc::clone(&llm),
            RuntimeModels::new(HashMap::from([(
                Category::Any,
                vec!["standard-answer".into(), answer.clone()],
            )])),
            "standard-auxiliary",
        );

        let judge: ModelId = "restricted-decision".into();
        let decision_calls = Arc::new(AtomicUsize::new(0));
        let restricted = ExecutionLane::new(
            Arc::new(CallThenRoute(RoutingCall::Decision)),
            ClientRouter::single_with_decision_clients(
                llm,
                HashMap::from([(
                    judge.clone(),
                    Arc::new(VerdictClient {
                        calls: Arc::clone(&decision_calls),
                        gate: None,
                    }) as Arc<dyn RoutedDecisionClient>,
                )]),
            ),
            None,
            None,
            RuntimeModels::new(HashMap::from([
                (Category::Judge, vec![judge]),
                (Category::Any, vec![answer]),
            ])),
        );
        let route = Route::from_lane(
            standard,
            None,
            ModelCapabilities::default(),
            vec![
                decision_target("standard", &ModelId::from("standard-answer")),
                decision_target("restricted", &ModelId::from("restricted-answer")),
            ],
        )
        .with_privacy(PrivacyPolicy::new(None, Requirements::new()), restricted);
        let route_id: ModelId = "switchyard/private".into();
        let runner = crate::Runner::new(vec![(route_id.clone(), route)]);

        let outcome = runner
            .route(route_id.as_str())
            .expect("route")
            .decide(restricted_request())
            .await
            .expect("restricted decision");
        assert_eq!(decision_calls.load(Ordering::Relaxed), 1);
        let description = runner
            .describe_decision(&route_id, &outcome)
            .expect("restricted target metadata");
        assert_eq!(description.selected.target, "restricted");
        assert_eq!(
            description.selected.base_url,
            "https://restricted.example/v1"
        );
    }

    #[tokio::test]
    async fn external_and_input_restrictions_skip_semantic_assessment() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let client: Arc<dyn RoutedLlmClient> = Arc::new(RecordingClient {
            calls: Arc::clone(&calls),
        });
        let standard = lane(
            Arc::new(Passthrough),
            Arc::clone(&client),
            RuntimeModels::new(HashMap::from([(
                Category::Any,
                vec!["standard-primary".into()],
            )])),
            "standard-auxiliary",
        );
        let restricted = lane(
            Arc::new(Passthrough),
            client,
            RuntimeModels::new(HashMap::from([(
                Category::Any,
                vec!["restricted-primary".into()],
            )])),
            "restricted-auxiliary",
        );
        let classifier_calls = Arc::new(AtomicUsize::new(0));
        let classifier = SemanticPrivacyClassifier::decision(
            "privacy-judge".into(),
            Arc::new(VerdictClient {
                calls: Arc::clone(&classifier_calls),
                gate: None,
            }),
            None,
            0.9,
        );
        let route = Route::from_lane(standard, None, ModelCapabilities::default(), Vec::new())
            .with_privacy(
                PrivacyPolicy::new(Some(classifier), Requirements::from(["private".into()])),
                restricted,
            );

        assert_eq!(
            route
                .decide(Request::default())
                .await
                .expect("classifier verdict should select a lane")
                .selected_model_id()
                .expect("selected model"),
            "standard-primary"
        );

        assert_eq!(
            route
                .decide(restricted_request())
                .await
                .expect("external restriction")
                .selected_model_id()
                .expect("selected model"),
            "restricted-primary"
        );
        let mut opaque = Request::default();
        opaque.llm_request.messages.push(Message {
            role: Role::User,
            content: vec![ContentBlock::Unknown {
                provider: "custom".into(),
                raw: Value::Null,
            }],
        });
        assert_eq!(
            route
                .decide(opaque)
                .await
                .expect("opaque input restriction")
                .selected_model_id()
                .expect("selected model"),
            "restricted-primary"
        );
        assert_eq!(classifier_calls.load(Ordering::Relaxed), 1);
    }

    // An assessment cannot erase a concurrent restriction or inspect an ineligible request.
    #[tokio::test]
    async fn classification_rechecks_task_requirements_and_its_own_eligibility() {
        let calls = Arc::new(AtomicUsize::new(0));
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let classifier = SemanticPrivacyClassifier::decision(
            "private-eu-judge".into(),
            Arc::new(VerdictClient {
                calls: Arc::clone(&calls),
                gate: Some((Arc::clone(&started), Arc::clone(&release))),
            }),
            None,
            0.9,
        );
        let client: Arc<dyn RoutedLlmClient> = Arc::new(RecordingClient {
            calls: Arc::new(Mutex::new(Vec::new())),
        });
        let build_lane = |name: &str| {
            lane(
                Arc::new(Passthrough),
                Arc::clone(&client),
                RuntimeModels::new(HashMap::from([(Category::Any, vec![name.into()])])),
                name,
            )
        };
        let requirement_sets = [
            vec!["eu"],
            vec!["us"],
            vec!["private"],
            vec!["eu", "private"],
        ]
        .map(|labels| labels.into_iter().map(str::to_string).collect::<Vec<_>>());
        let groups = requirement_sets
            .iter()
            .map(|labels| {
                (
                    labels.iter().cloned().collect(),
                    build_lane(&labels.join("-")),
                )
            })
            .collect();
        let route = Route::from_lane(
            build_lane("standard"),
            None,
            ModelCapabilities::default(),
            Vec::new(),
        )
        .with_eligibility(
            EligibilityPolicy::new(true, requirement_sets.into(), true).expect("requirement sets"),
            groups,
        )
        .with_privacy_policy(PrivacyPolicy::new(
            Some(classifier),
            Requirements::from(["private".into(), "eu".into()]),
        ));
        let request = |task: &str, labels: &[&str]| {
            let mut request = Request::default();
            request.metadata.get_or_insert_default().task_id = Some(task.into());
            crate::require_target_labels(&mut request, labels).expect("trusted labels");
            request
        };
        let slow = route.decide(request("shared", &[]));
        let tighten = async {
            started.notified().await;
            let outcome = route.decide(request("shared", &["eu"])).await;
            release.notify_one();
            outcome
        };
        let outcomes = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            tokio::join!(slow, tighten)
        })
        .await
        .expect("no lock may be held across classification");
        for outcome in [outcomes.0, outcomes.1] {
            let outcome = outcome.expect("EU decision");
            assert_eq!(outcome.selected_model_id().expect("model"), "eu");
        }
        assert!(route.decide(request("other", &["us"])).await.is_err());
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }
}
