// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental Node bridge proving trusted admission through the real Runner API.

use napi_derive::napi;
use switchyard_protocol::{Category, LlmRequest, Metadata, Request};
use switchyard_runner::{Runner, require_local_targets};

fn error(code: &str, message: impl std::fmt::Display) -> napi::Error {
    napi::Error::from_reason(format!("{code}: {message}"))
}

/// Experimental Pi model selector backed by one configured Runner route.
#[napi]
pub struct NativeConfiguredRoute {
    runner: Runner,
    route_id: String,
}

#[napi]
impl NativeConfiguredRoute {
    /// Loads a routing-only Stage or passthrough route without judges.
    #[napi(constructor)]
    pub fn new(config_toml: String, route_id: String) -> napi::Result<Self> {
        let runner = Runner::from_toml(&config_toml).map_err(|err| error("ERR_CONFIG", err))?;
        let route = runner
            .route(&route_id)
            .ok_or_else(|| error("ERR_CONFIG", "unknown route"))?;
        if !matches!(route.algorithm_name(), "stage_router" | "passthrough") {
            return Err(error("ERR_CONFIG", "spike supports Stage or passthrough"));
        }
        if !route.models().models_for(&Category::Judge).is_empty() {
            return Err(error("ERR_CONFIG", "spike does not serve routing judges"));
        }
        Ok(Self { runner, route_id })
    }

    /// Routes normalized messages; Pi owns the final provider request.
    #[napi]
    pub async fn select(
        &self,
        messages_json: String,
        local_only: bool,
        task_id: String,
    ) -> napi::Result<String> {
        let mut request = Request {
            llm_request: LlmRequest {
                model: Some(self.route_id.clone()),
                messages: serde_json::from_str(&messages_json)
                    .map_err(|err| error("ERR_INVALID_REQUEST", err))?,
                ..Default::default()
            },
            metadata: Some(Metadata {
                task_id: Some(task_id),
                ..Default::default()
            }),
            ..Default::default()
        };
        require_local_targets(&mut request, local_only)
            .map_err(|err| error("ERR_INVALID_REQUEST", err))?;
        let route = self
            .runner
            .route(&self.route_id)
            .ok_or_else(|| error("ERR_CONFIG", "unknown route"))?;
        let outcome = route
            .decide(request)
            .await
            .map_err(|err| error("ERR_ROUTING", err))?;
        outcome
            .selected_model_id()
            .map(ToString::to_string)
            .map_err(|err| error("ERR_ROUTING", err))
    }
}
