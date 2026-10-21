# Embedded Go routing

Run Switchyard algorithms in a Go host that owns model eligibility, credentials,
provider calls, and retries. The default backend runs libsy in WebAssembly through
wazero, with no cgo dependency. Native builds can use the C ABI instead.

## Build and test

Requires Go 1.25+. Test the embedded WASM backend without a Rust or C toolchain:

```sh
cd bindings/go
CGO_ENABLED=0 go test ./...
go vet ./...
```

For native tests, build the Rust library for the same platform as the Go host.
This path also requires cgo and a C compiler:

```sh
cargo build -p switchyard-ffi --locked
cd bindings/go
go test -tags switchyard_native -race ./...
GOEXPERIMENT=cgocheck2 go test -tags switchyard_native -race ./...
go vet -tags switchyard_native ./...
```

Native builds load the shared library from `target/debug`; no static archive is required.
Rebuild Go with `go test -a` after changing the C header or native library; Go's
cache does not track headers outside the package directory. Native artifacts
are not downloaded or published by this package.

After changing Rust code, run `scripts/build-go-wasm.sh` from the repository root
to rebuild the embedded WASM file and its checksum. This requires the repository's
Rust toolchain and its `wasm32-wasip1` target.

### Go versions

`go 1.25.0` in `go.mod` is the compatibility minimum required by wazero, not a
compiler pin. Use the latest patch of a [supported Go release](https://go.dev/doc/devel/release#policy)
for development and production; compatibility with an older release does not
make it a recommended production toolchain.

Before merging changes, run the checks above with the latest patch of the
declared minimum and the latest stable Go. Raise the minimum only when the
binding or a dependency needs newer Go, and call out that change.

## Use

```go
router, err := switchyard.New(json.RawMessage(`{
  "algorithm": {"type": "random", "targets": ["fast", "capable"]},
  "targets": {"fast": "fast-model", "capable": "capable-model"}
}`))
if err != nil {
    return err
}
defer router.Close()

outcome, err := router.Route(ctx, switchyard.Input{
    Request: json.RawMessage(`{"messages":[{
        "role":"user", "content":[{"type":"text", "text":"Fix this bug"}]
    }]}`),
    Models: map[string][]string{"any": {"fast-model", "capable-model"}},
}, nil)
```

Import `encoding/json` and
`github.com/NVIDIA-NeMo/Switchyard/bindings/go`. This is a source-only module;
use a local Go module `replace` to point your host at this directory.

`New` uses native code when built with `cgo` and `switchyard_native`; otherwise it
uses WASM. Pass `WithEngine(WASM)` or `WithEngine(Native)` to choose explicitly.

Supply the model groups needed by the configured algorithm. Composite needs
`any` for eligible answer targets, plus `capable`, `efficient`, and `judge`.
Random and signal-only Stage need no host callback. For Composite, supply a
`HostCall` that handles `Work.Type`:

- `call_model`: call the ordered `Work.Models` candidates and return `Reply` with
  a normalized response and the target ID that answered in `ServedModel`.
- `call_decision`: send `Work.Request` to `Work.Model` and return its normalized
  decision response. `ServedModel` does not apply to decision calls.

Requests and replies use Switchyard's protocol JSON. `DecodeRequest`,
`EncodeRequest`, and `DecodeResponse` use the existing translation codecs for
`openai_chat`, `openai_responses`, and `anthropic_messages`. Typed decision
requests use the decision protocol directly.

`Outcome` contains ordered targets, the prepared request, optional routing
evidence, and an optional answer. When it has no answer, the host calls the
selected target. Routing-produced responses are buffered; the host may stream
the final selected provider response separately.

## Cancellation and ownership

`Route` waits in its calling goroutine. A host that needs immediate return can
call it in a goroutine; waiting for routing steps does not occupy a blocked
native-reader thread. Concurrent routes share one router. The WASM backend uses
a pool of instances and keeps each session on the same instance to retain routing
state. A trapped instance is replaced; retained state in that instance is lost.

Host callbacks may run concurrently and must honor their context. Cancellation
stops routing and cancels the host context, but cannot forcibly stop Go code that
ignores it. WASM checks cancellation between guest calls. Late host results cannot
access a freed run.
Host errors cross the boundary as messages, not concrete Go error types.

`Close` prevents new runs; existing runs retain their algorithm and finish under
their own contexts. Cancel those contexts to stop them. Router value copies
share closed state, and repeated `Close` calls are safe. Native callback handles
are released only after native cleanup has waited for callbacks to return.
Call `Close` when finished; Go garbage collection does not release the router.

`New` checks ABI generation 1 against the native library or WASM module. Scope session and agent
IDs to the host's tenant and conversation. `Input.Metadata` and
`Input.SubagentModels` carry routing context separately from provider requests.
