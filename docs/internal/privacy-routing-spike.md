# Target eligibility and privacy routing spike

This is the agreed prototype contract, not a released feature. Eligibility limits which
configured targets a router may call; it does not replace the routing algorithm or classify
request content. Labels such as `private` and `eu` are deployment-owned approvals, not proof
of endpoint location, security or provider retention policy.

This note describes the current uncommitted changes in the canonical local worktree, not a
published release or restacked branch state. Classification and trusted requirements now use
one eligibility engine. Composition is implemented and validated locally; individual branches
have not been restacked.

## Labels and trusted requirements

Targets declare flat string labels. The route names the finite requirement sets
it supports; the empty set is implicit. Client definitions are omitted here.

```toml
[targets.fast]
id = "public-model"
llm_client = "public_endpoint"
labels = ["us"]

[targets.private_us]
id = "internal-us-model"
llm_client = "us_endpoint"
labels = ["private", "us"]

[targets.private_eu]
id = "internal-eu-model"
llm_client = "eu_endpoint"
labels = ["private", "eu"]

[routes.coding]
id = "switchyard/coding"
type = "random"
targets = ["fast", "private_us", "private_eu"]
weights = [3, 1, 1]

[routes.coding.eligibility]
accept_external_signal = true
restriction_scope = "request"

[routes.coding.eligibility.requirement_sets]
private = ["private"]
eu = ["eu"]
private_eu = ["private", "eu"]
```

The embedded application supplies requirements through the additive Rust helper:

```rust
use switchyard_runner::require_target_labels;

require_target_labels(&mut request, ["private", "eu"])?;
let output = route.execute(request, None).await?;
```

`require_target_labels(&mut Request, impl IntoIterator<Item = impl AsRef<str>>)` returns
`Result<(), LlmClientError>` and merges requirements into trusted internal metadata. It adds
no HTTP header, request-body field or Python binding. Routes must opt into external signals.

For routes that require a signal on every request, use:

```toml
[routes.coding.eligibility]
require_external_signal = true
```

Requiring a signal also enables accepting it; `accept_external_signal` can be omitted.
A missing signal is rejected before classification or other model calls, even when the task
already has retained requirements. Legacy `mark_privacy_restricted()` signals also qualify.

The application can explicitly supply no additional requirements:

```rust
require_target_labels(&mut request, std::iter::empty::<&str>())?;
```

An empty signal does not clear retained task requirements. Presence does not prove the
application performed an assessment, and this option does not synchronize task state across
replicas. The default remains optional; existing routes need no changes.

Every required label must exactly match a target label; a target may have extra labels.
Each `requirement_sets` entry names a supported label set. The names also identify optional
target overrides; applications still supply labels, not those names. Order and duplicate labels
do not change a set.
In the example, `["private"]` admits both internal targets, while `["private", "eu"]` admits only
`private_eu`. No requirements admits all configured candidates. Only declared requirement sets
are supported; Runner does not generate other combinations.

Unknown or malformed labels fail closed. A valid but unsupported requirement set also returns an
error; Switchyard never drops a requirement or chooses a nearby requirement set to keep serving.
Completion and decision-target aliases sharing a client and model must declare the same labels.
Labels are limited to 128 bytes each, with at most 32 distinct labels per route and per target.
Routes support at most 64 canonical requirement sets, including the implicit empty set.

## Prebuilt execution groups

Runner builds a separate execution lane for each declared requirement set, using only targets
that satisfy it. The existing router still chooses among that pool, including its judges,
answer candidates, fallbacks and auxiliaries. Random keeps weights paired with their targets.
Failures never restore an ineligible target or import one from another deployment.

Fixed-role routers keep their efficient, capable and judge roles. A requirement set can bind
those existing names to approved replacements without changing the router settings. For example,
assuming the normal targets and their EU-approved replacements are defined:

```toml
[routes.coding_stage]
id = "switchyard/coding-stage"
type = "stage_router"
efficient_target = "fast"
capable_target = "capable"
picker = "efficient_first"
confidence_threshold = 0.5
classifier = { target = "judge", base_threshold = 0.5 }

[routes.coding_stage.eligibility]
require_external_signal = true

[routes.coding_stage.eligibility.requirement_sets]
private_eu = ["private", "eu"]

[routes.coding_stage.eligibility.target_overrides.private_eu]
fast = "private_fast_eu"
capable = "private_capable_eu"
judge = "private_judge_eu"
```

An explicit empty signal uses the original targets. The `private` plus `eu` requirements use
the replacements, each of which must declare both labels. This binding is built once at startup;
model calls, prompts, fallbacks, subagent policies and auxiliary operations use that lane's
resolved targets. Decision metadata reports the replacement targets.

Mappings apply once: `fast = "private_fast_eu"` does not follow another mapping for
`private_fast_eu`. Unmapped roles must already meet the requirements. Missing replacements,
unused names, changed LLM/decision target kinds and conflicting mappings for equivalent label
sets are configuration errors. Empty-set overrides are rejected so unrestricted routing stays
unchanged. Random overrides cannot collapse distinct models onto one weighted entry.

The same binding supports Auto, PlanExecute, Advisor, passthrough, Composite and classifier
routes, including typed judges and subagent targets. Judge prompts and decision evidence are
unchanged; deployments must ensure they remain appropriate for replacement models. Completion
model IDs must identify one client across the route's groups so decision metadata is unambiguous.

Without overrides, a declared group missing a required role remains a configuration error.
Workloads whose configured targets already meet every fixed requirement can use an ordinary
route without classification or extra execution groups.

Legacy privacy-only configuration also uses this engine. Targets declare privacy approval through
`labels = ["private"]`, and `mark_privacy_restricted()` contributes the `private` requirement.
A legacy route can use the existing privacy table instead of an eligibility table:

```toml
[routes.legacy]
id = "switchyard/legacy"
type = "passthrough"
target = "private_eu"

[routes.legacy.privacy]
accept_external_signal = true
restriction_scope = "request"
```

## Task retention

Set `restriction_scope = "task"` in the eligibility table and supply
`request.metadata.task_id`. Each request adds valid requirements to the task's retained union;
an already-constrained task must still process new requirements. A later empty or clear request
cannot remove earlier host requirements.

Declare the unions a task may accumulate. The example includes `["private", "eu"]` so a task
can acquire `private` on one turn and `eu` on a later turn.

If the union is not a configured requirement set, the request fails and the valid union remains
retained. Unknown or malformed requirements are not retained. Missing or invalid task IDs and
exhausted retention capacity now return errors; the older draft's private-lane fallback is not
used. Retention is per route and process, limited to 4,096 nonempty task entries under IDs of at most
512 bytes, with no expiry, reset API or replica coordination.

State is merged under short locks, never across an await. Selection uses an admission snapshot:
requests that already selected a group are not recalled when another request tightens the task.
After an awaited assessment, selection must incorporate requirements retained by concurrent turns.

## Optional privacy classification

The privacy classifier keeps its existing six verdicts: `no_sensitive_content`, `personal_data`,
`credentials`, `confidential_data`, `regulated_data` and `uncertain`. It does not invent arbitrary
target labels. A valid sensitive verdict adds `private`; a clear verdict cannot remove host or
retained requirements. Uncertainty, invalid output, oversized input, a low clear score or a failed
call requires `private` for the current request, without permanently retaining that precaution.
Only an affirmative sensitive verdict becomes sticky when task retention is enabled.

Before the classifier sees content, its target must satisfy all requirements already known from
the host and task, as well as its privacy approval. For example, an EU-constrained request cannot
be sent to a US-only classifier. Failure to find an eligible classifier must block the call.

Generic LLM and typed decision classifiers remain alternatives, not consecutive judges. To combine
one with the eligibility example, add either classifier block below. The eligibility table owns
`accept_external_signal` and `restriction_scope`; the privacy table supplies only its classifier.
Do not repeat the legacy privacy table's signal or task settings in a combined route.

```toml
[targets.privacy_classifier]
id = "internal-classifier"
llm_client = "eu_endpoint"
labels = ["private", "eu"]

[routes.coding.privacy.classifier]
type = "llm"
target = "privacy_classifier"
clear_threshold = 0.90
response_format_type = "json_schema"
```

The typed alternative uses a decision target with the same approval labels:

```toml
[decision_targets.privacy_decision]
id = "internal-decision-model"
decision_client = "internal_decisions"
labels = ["private", "eu"]

[routes.coding.privacy.classifier]
type = "decision"
target = "privacy_decision"
clear_threshold = 0.90
```

The LLM client must declare `timeout_ms`. The LLM path supports `json_schema` and `json_object`,
with the schema in the prompt and typed local validation. Custom prompts follow the shared
contract, which rejects `{{RESPONSE_SCHEMA}}` placeholders because Switchyard supplies the schema.
Structural validation bounds content before classification; it does not detect PII. Input includes
content-bearing instructions, tools, extensions and preserved request bodies, but excludes transport metadata,
cached responses and remote MCP tool definitions. Serialized input is bounded to 64 KiB.
`clear_score` is a model score, not a calibrated guarantee of safety.

## Responsibilities and API impact

| Component | Responsibility |
|---|---|
| Embedded application | Supply trusted required labels and task identity. |
| Libsy | Enforce allowed model IDs; build and validate optional semantic judge calls. |
| Runner | Prebuild eligible groups, bind their clients and auxiliaries, retain task requirements. |
| LLM client | Execute provider requests, retries and eligible candidate fallbacks. |

`RuntimeModels::with_target_restriction()` enforcement is unchanged: it covers `Driver`, the
default `run_stream` and `drive`, including rewritten steps from a custom stream. A custom
consumer must enforce its own boundary. Model IDs do not distinguish deployments exposing the
same model, so hosts must bind the correct clients. This does not sandbox arbitrary Rust or
network calls made by the host.

The additive generic API is `require_target_labels()`. Existing privacy APIs remain compatibility
adapters: `PrivacyPreflight`, `PrivacyAssessment`, `mark_privacy_restricted()`, `PrivacyDecision`
and `Route::execute_with_privacy_observer()`. No published Rust signature is removed or renamed;
the protocol types, Python surface and native Relay plugin ABI are unchanged. No generic public
observer was added for this slice. TOML labels and eligibility tables are additive. Generic judge
helpers remain private. Runner uses existing workspace packages; this generalization introduces
no new dependency versions.

Existing mixed-privacy restrictions on forwarded caller auth, prefill routing and OpenAI Responses
continuation state are not relaxed. Embedded hosts must not reuse a broad client router's stored
continuation across eligibility groups. Provider policies remain the deployment owner's responsibility.

## Branch ownership and validation

The planned branch ownership below is not a claim that these local changes have been restacked:

| Branch | Responsibility |
|---|---|
| #924 | Build finite eligible execution lanes and preserve router roles and target metadata. |
| #926 | Add target labels, declared requirement sets and the trusted request helper. |
| #929 / #930 | Compose typed / LLM privacy classification with the shared eligibility engine. |
| #931 | Retain accumulated task requirements and recheck concurrent updates. |
| #932 | Add bounded eligibility and classifier tracing. |
| #933 | Preserve Relay privacy marks and the plugin ABI. |

Generalized tests must cover exact label matching, reordered requirement sets, role and weight
preservation, unsupported requirement sets, private fallbacks and auxiliaries, and duplicate model
IDs across clients. Task tests must cover sequential and concurrent unions, cancellation,
capacity, invalid identity and requirements that stay retained after a no-eligible error.

Existing classifier coverage checks bounded context, malformed verdicts, thresholds and cancellation;
synthetic verdicts validate wiring, not classification accuracy. Classifier calls use standard libsy
spans and metrics, with an INFO context span for route attribution and buffered response IDs/models.
Abandoned calls retain the existing `error` outcome rather than a new `cancelled` label.

Generic required labels currently appear only in DEBUG tracing. Relay's existing privacy marks
are unchanged: lane, source, reason and optional classifier score/threshold, with no arbitrary
`required_labels`. After successful admission, those marks survive subsequent execution failure.
Generic eligibility decisions are not yet carried in Relay marks. This is a current telemetry
gap; telemetry must never include request content.
