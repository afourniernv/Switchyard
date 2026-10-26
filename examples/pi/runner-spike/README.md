# Experimental Pi → Runner local-only demo

This is a separate, opt-in spike. It does not replace the normal Pi extension or
add exports to the published Node bindings. It needs the local-only Runner stack.

## Who decides local-only?

The operator does: start Pi with `--local-only`, or enter `/local` while idle.
Without either, the extension supplies an explicit `false`. There is no content
classifier or automatic inspection of files, resumes or PII.

A `true` is saved in Pi's session entries. Every virtual-model request reads it,
including tool continuations, retries and direct calls such as compaction.
The extension cannot clear it. Start a new blank session **without the flag** to
return to normal routing; a startup flag remains active within that Pi process.

## How it reaches Runner

```text
Pi flag or /local command → saved session flag
  → virtual-model route callback: boolean + Pi session ID + routing messages
  → experimental Node addon: require_local_targets(&mut request, local_only)
  → Route::decide: admit request, select standard or local lane, run Stage
  → return approved model ID → Pi sends the original request to that model
```

`spike.toml` has standard and local efficient/capable pairs. `local = true` is an
operator approval, not endpoint detection. The route's flat overrides replace
both Stage roles when required; Stage still chooses between those roles normally.
`scope = "task"` retains a true signal under the supplied Pi session ID. The
Pi session entry additionally restores the signal when the addon restarts.

The addon only chooses a model. It makes no terminal provider calls. Model IDs
must match physical entries in Pi's catalog; Pi owns their URLs and credentials.
The `unused-*` TOML URLs are deliberately not contacted by this fixture.

## Build and check

Use Node 22.19+, the repository Rust toolchain and the Pi versions in the parent
example's development dependencies. From the repository root:

```bash
npm --prefix bindings/typescript ci
npm --prefix bindings/typescript run build
npm --prefix examples/pi ci
node examples/pi/runner-spike/build.mjs
examples/pi/node_modules/.bin/tsc -p examples/pi/runner-spike/tsconfig.json
node --test examples/pi/runner-spike/selection.test.mjs
```

The standalone native crate is `publish = false`, outside workspace CI. Its
generated lockfile, library and build outputs are ignored. Pass `--offline` to
`build.mjs` when dependencies are already cached; it builds the debug addon.

## Real local smoke

With Ollama already listening on `127.0.0.1:11434` and `qwen3.5:9b` installed:

```bash
python3 examples/pi/runner-spike/run_pi_smoke.py
```

`PI_RUNNER_SMOKE_MODEL` can select a different installed Ollama model. The script
uses an isolated Pi agent directory, synthetic prompts and a loopback audit proxy
on port 48663. It does not download models or use cloud credentials. Both local
roles point to the same Ollama model for this smoke; the standard path is an
explicit fixture, **not a successful cloud-model test**.

Checks cover ordinary routing, real local completion, session resume without the
flag, a direct call, a tool continuation, and local failure without standard
fallback. Results and session files stay in ignored `smoke-output/`; the proxy
stops afterward. No personal files or normal Pi settings are changed.

## Limits before this could ship

- Only Stage without a judge or passthrough is supported. Configured judge targets
  are rejected; Pi's typed Decision callback is not connected to Runner.
- The return value is only a model ID. Request rewrites, target prompts/settings
  and ordered answer fallback lists are not passed back to Pi.
- The addon passes `task_id`, not `session_id`. Stage's capable hold resets between
  calls, and Pi's branch-carried CompositeState is not bridged. Supplying a session
  ID could retain in-memory Stage state within a lane, not restore branch state.
- Cancellation is checked around selection but does not cancel the native future.
- It governs only `switchyard/runner-spike`; other Pi providers and tools' network
  calls are outside this path. Local approvals and Pi endpoint bindings must agree.

No new core Rust API or existing binding is changed. This spike adds its own
experimental Node interface and a dependency on Runner; it is not production-ready.
