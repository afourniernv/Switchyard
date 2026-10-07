// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed semantic preflight for contextual privacy decisions.

use std::collections::BTreeMap;
use std::io::{self, Write};
use std::sync::Arc;

use serde::Serialize;
use serde_json::{Map, Value};
use strum_macros::{EnumString, IntoStaticStr};
use switchyard_protocol::{
    ChoiceOption, DecisionKind, DecisionQuestion, DecisionRequest, DecisionResponse, DecisionValue,
    FormatId, InstructionBlock, LlmRequest, Message, ModelId, OutputParams, PreservationMetadata,
    ProviderExtensions, ReasoningParams, Request, RoutedDecisionClient, ToolChoice, ToolDefinition,
};

use super::{PrivacyDecision, PrivacyLane};

const MAX_PRIVACY_CLASSIFIER_CONTEXT_BYTES: usize = 64 * 1024;
const PROBABILITY_SUM_TOLERANCE: f64 = 1e-6;
const QUESTION_ID: &str = "privacy";

pub(crate) struct SemanticPrivacyClassifier {
    target: ModelId,
    client: Arc<dyn RoutedDecisionClient>,
    clear_threshold: f64,
    question: DecisionQuestion,
}

impl SemanticPrivacyClassifier {
    pub(crate) fn new(
        target: ModelId,
        client: Arc<dyn RoutedDecisionClient>,
        instructions: Option<&Value>,
        clear_threshold: f64,
    ) -> Self {
        Self {
            target,
            client,
            clear_threshold,
            question: privacy_question(instructions),
        }
    }

    pub(crate) async fn assess(&self, request: &Request) -> PrivacyDecision {
        let Ok(request) = self.request(request) else {
            return self.restricted("input_unavailable", None);
        };
        let Ok(response) = self.client.call(request).await else {
            return self.restricted("classifier_failed", None);
        };
        self.decision(&response)
    }

    fn request(&self, request: &Request) -> Result<DecisionRequest, ()> {
        let context = serialize_classifier_context(&request.llm_request)?;
        Ok(DecisionRequest {
            model: Some(self.target.clone()),
            context,
            questions: BTreeMap::from([(QUESTION_ID.into(), self.question.clone())]),
        })
    }

    fn decision(&self, response: &DecisionResponse) -> PrivacyDecision {
        let Some(DecisionValue::Choice {
            selected,
            probabilities: Some(probabilities),
        }) = response
            .answers
            .get(QUESTION_ID)
            .map(|answer| &answer.value)
        else {
            return self.restricted("invalid_verdict", None);
        };
        let Ok(verdict) = selected.parse::<PrivacyVerdict>() else {
            return self.restricted("invalid_verdict", None);
        };
        let valid_distribution = probabilities.len() == PrivacyVerdict::ALL.len()
            && PrivacyVerdict::ALL.iter().all(|verdict| {
                probabilities
                    .get(verdict.as_str())
                    .is_some_and(|probability| {
                        probability.0.is_finite() && (0.0..=1.0).contains(&probability.0)
                    })
            })
            && (probabilities
                .values()
                .map(|probability| probability.0)
                .sum::<f64>()
                - 1.0)
                .abs()
                <= PROBABILITY_SUM_TOLERANCE;
        if !valid_distribution {
            return self.restricted("invalid_verdict", None);
        }
        let clear_score = probabilities
            .get(PrivacyVerdict::NoSensitiveContent.as_str())
            .map(|probability| probability.0);
        let Some(clear_score) = clear_score else {
            return self.restricted("invalid_verdict", None);
        };
        if verdict == PrivacyVerdict::NoSensitiveContent && clear_score >= self.clear_threshold {
            return PrivacyDecision::semantic(
                PrivacyLane::Standard,
                verdict.as_str(),
                Some(clear_score),
                self.clear_threshold,
            );
        }
        let reason = if verdict == PrivacyVerdict::NoSensitiveContent {
            "below_clear_threshold"
        } else {
            verdict.as_str()
        };
        self.restricted(reason, Some(clear_score))
    }

    fn restricted(&self, reason_code: &'static str, clear_score: Option<f64>) -> PrivacyDecision {
        PrivacyDecision::semantic(
            PrivacyLane::Restricted,
            reason_code,
            clear_score,
            self.clear_threshold,
        )
    }
}

fn serialize_classifier_context(request: &LlmRequest) -> Result<Value, ()> {
    let mut json = BoundedJson::default();
    serde_json::to_writer(&mut json, &PrivacyClassifierContext::new(request)).map_err(|_| ())?;
    json.into_value()
}

#[derive(Serialize)]
struct PrivacyClassifierContext<'a> {
    instructions: &'a [InstructionBlock],
    messages: &'a [Message],
    tools: &'a [ToolDefinition],
    tool_choice: Option<&'a ToolChoice>,
    response_format: Option<&'a Value>,
    reasoning_effort: Option<&'a str>,
    reasoning_raw: Option<&'a Value>,
    extensions: &'a Map<String, Value>,
    preserved_requests: &'a BTreeMap<FormatId, Value>,
}

impl<'a> PrivacyClassifierContext<'a> {
    fn new(request: &'a LlmRequest) -> Self {
        // Keep this exhaustive so each new protocol field gets an explicit privacy decision.
        let LlmRequest {
            model: _,
            instructions,
            messages,
            tools,
            tool_choice,
            sampling: _,
            output,
            reasoning,
            stream: _,
            extensions,
            preservation,
        } = request;
        let OutputParams {
            max_output_tokens: _,
            response_format,
            is_schema_enforced: _,
        } = output;
        let ReasoningParams { effort, raw } = reasoning;
        let ProviderExtensions { fields: extensions } = extensions;
        let PreservationMetadata {
            requests: preserved_requests,
            responses: _,
        } = preservation;
        Self {
            instructions,
            messages,
            tools,
            tool_choice: tool_choice.as_ref(),
            response_format: response_format.as_ref(),
            reasoning_effort: effort.as_deref(),
            reasoning_raw: raw.as_ref(),
            extensions,
            preserved_requests,
        }
    }
}

#[derive(Default)]
struct BoundedJson(Vec<u8>);

impl BoundedJson {
    fn into_value(self) -> Result<Value, ()> {
        serde_json::from_slice(&self.0).map_err(|_| ())
    }
}

impl Write for BoundedJson {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > MAX_PRIVACY_CLASSIFIER_CONTEXT_BYTES.saturating_sub(self.0.len()) {
            return Err(io::Error::other("privacy classifier context exceeds limit"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[derive(Clone, Copy, EnumString, Eq, IntoStaticStr, PartialEq)]
#[strum(serialize_all = "snake_case")]
enum PrivacyVerdict {
    NoSensitiveContent,
    PersonalData,
    Credentials,
    ConfidentialData,
    RegulatedData,
    Uncertain,
}

impl PrivacyVerdict {
    const ALL: [Self; 6] = [
        Self::NoSensitiveContent,
        Self::PersonalData,
        Self::Credentials,
        Self::ConfidentialData,
        Self::RegulatedData,
        Self::Uncertain,
    ];

    fn as_str(self) -> &'static str {
        self.into()
    }

    const fn description(self) -> &'static str {
        match self {
            Self::NoSensitiveContent => "No personal, secret, confidential, or regulated content.",
            Self::PersonalData => "Personal or identifying data is present.",
            Self::Credentials => "Credentials, secrets, or authentication material are present.",
            Self::ConfidentialData => "Confidential organizational data is present.",
            Self::RegulatedData => "Regulated data is present.",
            Self::Uncertain => "The privacy boundary cannot be determined confidently.",
        }
    }
}

fn privacy_question(instructions: Option<&Value>) -> DecisionQuestion {
    let options = PrivacyVerdict::ALL
        .into_iter()
        .map(|verdict| ChoiceOption {
            id: verdict.as_str().into(),
            description: Some(Value::String(verdict.description().into())),
        })
        .collect();
    DecisionQuestion {
        instructions: instructions
            .cloned()
            .unwrap_or_else(|| Value::String(include_str!("semantic_prompt.txt").trim().into())),
        kind: DecisionKind::Choice { options },
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use async_trait::async_trait;
    use serde_json::json;
    use switchyard_protocol::{
        ContentBlock, DecisionAnswer, InstructionBlock, LlmClientError, Message, Probability, Role,
        ToolCall, ToolDefinition, ToolResult, Usage,
    };

    use super::*;

    struct ReplyClient(Mutex<Option<Result<DecisionResponse, LlmClientError>>>);

    #[async_trait]
    impl RoutedDecisionClient for ReplyClient {
        async fn call(
            &self,
            _request: DecisionRequest,
        ) -> Result<DecisionResponse, LlmClientError> {
            self.0
                .lock()
                .expect("reply lock")
                .take()
                .expect("one classifier call")
        }
    }

    fn response(selected: &str, clear_score: Option<f64>) -> DecisionResponse {
        DecisionResponse {
            id: None,
            model: None,
            answers: BTreeMap::from([(
                QUESTION_ID.into(),
                DecisionAnswer {
                    value: DecisionValue::Choice {
                        selected: selected.into(),
                        probabilities: clear_score.map(|score| {
                            let mut probabilities = PrivacyVerdict::ALL
                                .into_iter()
                                .map(|verdict| (verdict.as_str().into(), Probability(0.0)))
                                .collect::<BTreeMap<_, _>>();
                            probabilities.insert("no_sensitive_content".into(), Probability(score));
                            if selected != "no_sensitive_content" {
                                probabilities.insert(selected.into(), Probability(1.0 - score));
                            } else {
                                probabilities.insert("uncertain".into(), Probability(1.0 - score));
                            }
                            probabilities
                        }),
                    },
                    provider_confidence: None,
                },
            )]),
            usage: Usage::default(),
        }
    }

    fn probabilities(response: &mut DecisionResponse) -> &mut BTreeMap<String, Probability> {
        let DecisionValue::Choice {
            probabilities: Some(probabilities),
            ..
        } = &mut response.answers.get_mut(QUESTION_ID).expect("answer").value
        else {
            panic!("choice answer")
        };
        probabilities
    }

    fn classifier(reply: Result<DecisionResponse, LlmClientError>) -> SemanticPrivacyClassifier {
        SemanticPrivacyClassifier::new(
            "privacy/model".into(),
            Arc::new(ReplyClient(Mutex::new(Some(reply)))),
            None,
            0.9,
        )
    }

    #[test]
    fn context_is_bounded_and_covers_privacy_relevant_request_state() {
        let instructions = json!({"policy": "custom"});
        assert_eq!(
            privacy_question(Some(&instructions)).instructions,
            instructions
        );
        let classifier = classifier(Ok(response("no_sensitive_content", Some(1.0))));
        let mut request = Request::default();
        request.llm_request.messages.push(Message::text(
            switchyard_protocol::Role::User,
            "normalized_marker",
        ));
        request.llm_request.instructions.push(InstructionBlock {
            role: Role::System,
            content: vec![ContentBlock::Reasoning {
                text: "reasoning_marker".into(),
                signature: Some("signature_marker".into()),
                details: vec![json!({"provider_detail_marker": true})],
            }],
        });
        request.llm_request.messages.push(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "call_marker".into(),
                name: "tool_call_marker".into(),
                arguments: json!({"argument_marker": true}),
            })],
        });
        request.llm_request.messages.push(Message {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: "call_marker".into(),
                content: vec![ContentBlock::Text {
                    text: "result_marker".into(),
                }],
                is_error: Some(false),
            })],
        });
        request.llm_request.tools.push(ToolDefinition {
            name: "tool_definition_marker".into(),
            description: None,
            parameters: json!({}),
            strict: None,
        });
        request.llm_request.tool_choice = Some(ToolChoice::Raw(json!({"choice_marker": true})));
        request.llm_request.output.response_format = Some(json!({"format_marker": true}));
        request.llm_request.reasoning.effort = Some("effort_marker".into());
        request.llm_request.reasoning.raw = Some(json!({"reasoning_raw_marker": true}));
        request
            .llm_request
            .extensions
            .fields
            .insert("extension_marker".into(), Value::Bool(true));
        request
            .llm_request
            .preservation
            .requests
            .insert("openai_chat".into(), json!({"preserved_marker": true}));
        request
            .llm_request
            .preservation
            .responses
            .insert("openai_chat".into(), json!({"response_marker": true}));
        request.llm_request.model = Some("model_marker".into());
        request.raw_request = Some(json!({"request_envelope_marker": true}));
        request
            .metadata
            .get_or_insert_default()
            .extra_metadata
            .get_or_insert_default()
            .insert("metadata_marker".into(), String::new());

        let decision = classifier.request(&request).expect("bounded context");
        let encoded = decision.context.to_string();
        for marker in [
            "normalized_marker",
            "reasoning_marker",
            "tool_call_marker",
            "argument_marker",
            "result_marker",
            "signature_marker",
            "provider_detail_marker",
            "tool_definition_marker",
            "choice_marker",
            "format_marker",
            "effort_marker",
            "reasoning_raw_marker",
            "extension_marker",
            "preserved_marker",
        ] {
            assert!(encoded.contains(marker), "missing {marker}");
        }
        for marker in [
            "response_marker",
            "model_marker",
            "request_envelope_marker",
            "metadata_marker",
        ] {
            assert!(!encoded.contains(marker), "unexpected {marker}");
        }

        request.llm_request.messages = vec![Message::text(
            switchyard_protocol::Role::User,
            "x".repeat(MAX_PRIVACY_CLASSIFIER_CONTEXT_BYTES),
        )];
        assert!(classifier.request(&request).is_err());
    }

    #[tokio::test]
    async fn typed_verdicts_and_failures_fail_closed() {
        let failure = LlmClientError::Configuration {
            message: "unavailable".into(),
        };
        let mut incomplete = response("no_sensitive_content", Some(0.95));
        probabilities(&mut incomplete).remove("uncertain");
        let mut unnormalized = response("no_sensitive_content", Some(0.95));
        probabilities(&mut unnormalized).insert("personal_data".into(), Probability(0.2));
        for (reply, lane, reason, score) in [
            (
                Ok(response("no_sensitive_content", Some(0.95))),
                PrivacyLane::Standard,
                "no_sensitive_content",
                Some(0.95),
            ),
            (
                Ok(response("no_sensitive_content", Some(0.8))),
                PrivacyLane::Restricted,
                "below_clear_threshold",
                Some(0.8),
            ),
            (
                Ok(response("confidential_data", Some(0.1))),
                PrivacyLane::Restricted,
                "confidential_data",
                Some(0.1),
            ),
            (
                Ok(response("no_sensitive_content", None)),
                PrivacyLane::Restricted,
                "invalid_verdict",
                None,
            ),
            (
                Ok(incomplete),
                PrivacyLane::Restricted,
                "invalid_verdict",
                None,
            ),
            (
                Ok(unnormalized),
                PrivacyLane::Restricted,
                "invalid_verdict",
                None,
            ),
            (
                Err(failure),
                PrivacyLane::Restricted,
                "classifier_failed",
                None,
            ),
        ] {
            let decision = classifier(reply).assess(&Request::default()).await;
            assert_eq!(decision.lane.as_str(), lane.as_str());
            assert_eq!(decision.reason_code, reason);
            assert_eq!(decision.clear_score, score);
        }
    }
}
