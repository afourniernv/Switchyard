# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real Pi→Node→Runner path, with real Ollama and a standard fixture."""

import collections
import http.server
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PI = ROOT.parent / "node_modules" / ".bin" / "pi"
OUTPUT = ROOT / "smoke-output"
MODEL = os.environ.get("PI_RUNNER_SMOKE_MODEL", "qwen3.5:9b")
PORT = 48663
STATE = {"case": "setup", "fail_local": False, "events": []}


class AuditProxy(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def reply(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path not in ("/local/v1/chat/completions", "/standard/v1/chat/completions"):
            self.reply(500, {"error": "Unexpected path: Runner must not execute provider calls"})
            return
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        backend = "local" if self.path.startswith("/local/") else "standard_fixture"
        event = {
            "case": STATE["case"],
            "backend": backend,
            "model": payload.get("model"),
            "max_tokens": payload.get("max_tokens"),
            "has_tool_result": any(m.get("role") == "tool" for m in payload.get("messages", [])),
        }
        STATE["events"].append(event)
        if len(STATE["events"]) > 15 or event["max_tokens"] != 128:
            event["status"] = 429
            self.reply(429, {"error": "Smoke request budget exceeded"})
            return
        if backend == "standard_fixture":
            event["status"] = 200
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for choice in [
                {
                    "delta": {"role": "assistant", "content": "STANDARD_FIXTURE_OK"},
                    "finish_reason": None,
                },
                {"delta": {}, "finish_reason": "stop"},
            ]:
                chunk = {
                    "id": "fixture",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": payload["model"],
                    "choices": [{"index": 0, **choice}],
                }
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        if STATE["fail_local"]:
            event["status"] = 503
            self.reply(503, {"error": {"message": "Injected local outage", "type": "unavailable"}})
            return
        # Distinct routing roles share one installed model only for this smoke.
        payload["model"] = MODEL
        request = urllib.request.Request(
            "http://127.0.0.1:11434/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=60) as upstream:
                event["status"] = upstream.status
                self.send_response(upstream.status)
                self.send_header(
                    "Content-Type", upstream.headers.get("Content-Type", "text/event-stream")
                )
                self.end_headers()
                while chunk := upstream.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as error:
            event["status"] = error.code
            self.reply(error.code, {"error": "Ollama rejected the smoke request"})
        finally:
            event["elapsed_seconds"] = round(time.monotonic() - started, 3)


def configure():
    OUTPUT.mkdir(exist_ok=True)
    agent = OUTPUT / "agent"
    agent.mkdir(exist_ok=True)
    (OUTPUT / "workdir").mkdir(exist_ok=True)
    (OUTPUT / "sessions").mkdir(exist_ok=True)
    cost = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    providers = {}
    for provider in ("local", "standard"):
        providers[provider] = {
            "baseUrl": f"http://127.0.0.1:{PORT}/{provider}/v1",
            "api": "openai-completions",
            "apiKey": "placeholder-not-a-secret",
            "compat": {
                "supportsStore": False,
                "supportsDeveloperRole": False,
                "supportsReasoningEffort": False,
                "maxTokensField": "max_tokens",
                "supportsStrictMode": False,
            },
            "models": [
                {
                    "id": role,
                    "name": f"{provider} {role} smoke alias",
                    "reasoning": False,
                    "input": ["text"],
                    "contextWindow": 32768,
                    "maxTokens": 128,
                    "samplingParams": {"reasoning_effort": "none"},
                    "cost": cost,
                }
                for role in ("efficient", "capable")
            ],
        }
    (agent / "models.json").write_text(json.dumps({"providers": providers}, indent=2) + "\n")
    (agent / "auth.json").write_text("{}\n")
    (agent / "settings.json").write_text(
        json.dumps(
            {
                "cacheWarming": "off",
                "compaction": {"enabled": False},
                "enableInstallTelemetry": False,
                "enableAnalytics": False,
                "retry": {"enabled": False, "provider": {"maxRetries": 0, "timeoutMs": 60000}},
            },
            indent=2,
        )
        + "\n"
    )


def run_case(name, prompt, *, local=False, session=None, tools=False, fail_local=False):
    STATE.update(case=name, fail_local=fail_local)
    offset = len(STATE["events"])
    command = [
        str(PI),
        "--offline",
        "--no-approve",
        "--no-extensions",
        "--no-mcp",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--no-context-files",
        "-e",
        str(ROOT / "extension.ts"),
        "--model",
        "switchyard/runner-spike",
        "--mode",
        "json",
        "--print",
        "--system-prompt",
        "You are a synthetic smoke-test responder. Follow the requested marker or tool instruction exactly.",
    ]
    command += ["--session", str(session)] if session else ["--no-session"]
    command += ["--tools", "smoke_echo"] if tools else ["--no-tools"]
    if local:
        command.append("--local-only")
    command += ["--", prompt]
    environment = {
        "PATH": os.environ["PATH"],
        "PI_CODING_AGENT_DIR": str(OUTPUT / "agent"),
        "PI_RUNNER_SPIKE_CONFIG": str(ROOT / "spike.toml"),
        "PI_OFFLINE": "1",
        "PI_TELEMETRY": "0",
        "NO_COLOR": "1",
        "TERM": "dumb",
    }
    started = time.monotonic()
    process = subprocess.run(
        command,
        cwd=OUTPUT / "workdir",
        env=environment,
        capture_output=True,
        text=True,
        timeout=150,
    )
    messages = []
    for line in process.stdout.splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = entry.get("message", {})
        if entry.get("type") == "message_end" and message.get("role") == "assistant":
            messages.append(
                {
                    "provider": message.get("provider"),
                    "model": message.get("model"),
                    "stop_reason": message.get("stopReason"),
                    "error": message.get("errorMessage"),
                    "text": "".join(
                        b.get("text", "")
                        for b in message.get("content", [])
                        if b.get("type") == "text"
                    ),
                    "tool_calls": sum(
                        b.get("type") == "toolCall" for b in message.get("content", [])
                    ),
                }
            )
    routes, direct, errors = [], [], []
    for line in process.stderr.splitlines():
        if line.startswith("SPIKE_ROUTE "):
            routes.append(json.loads(line.removeprefix("SPIKE_ROUTE ")))
        elif line.startswith("SPIKE_DIRECT "):
            direct.append(json.loads(line.removeprefix("SPIKE_DIRECT ")))
        else:
            errors.append(line)
    events = STATE["events"][offset:]
    result = {
        "case": name,
        "exit_code": process.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "routes": routes,
        "direct": direct,
        "counts": dict(collections.Counter(e["backend"] for e in events)),
        "events": events,
        "assistant_messages": messages,
        "stderr": errors,
    }
    print(json.dumps(result), flush=True)
    return result


def main():
    configure()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), AuditProxy)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    session = OUTPUT / "sessions" / ("local-" + str(time.time_ns()) + ".jsonl")
    try:
        results = [
            run_case("standard_fixture", "Reply exactly STANDARD_FIXTURE_OK."),
            run_case(
                "local_only_real_ollama",
                "Reply exactly LOCAL_RUNNER_OK.",
                local=True,
                session=session,
            ),
            run_case(
                "persisted_resume_without_flag", "Reply exactly RESUME_LOCAL_OK.", session=session
            ),
            run_case("direct_call_local", "/direct-smoke", local=True),
            run_case(
                "local_tool_continuation",
                "Call smoke_echo exactly once. After its result, reply exactly TOOL_CONTINUATION_OK. Do not call it again.",
                local=True,
                tools=True,
            ),
            run_case(
                "local503_no_standard_fallback",
                "Reply exactly MUST_NOT_FALL_BACK.",
                local=True,
                fail_local=True,
            ),
        ]
        by_name = {r["case"]: r for r in results}
        passed = {
            "standard_fixture_dispatch": by_name["standard_fixture"]["counts"]
            == {"standard_fixture": 1},
            "real_local_completion": by_name["local_only_real_ollama"]["counts"] == {"local": 1}
            and any(
                m["text"] == "LOCAL_RUNNER_OK" and m["stop_reason"] == "stop"
                for m in by_name["local_only_real_ollama"]["assistant_messages"]
            ),
            "persisted_resume_local": by_name["persisted_resume_without_flag"]["counts"]
            == {"local": 1}
            and any(
                m["text"] == "RESUME_LOCAL_OK" and m["stop_reason"] == "stop"
                for m in by_name["persisted_resume_without_flag"]["assistant_messages"]
            ),
            "direct_call_local": by_name["direct_call_local"]["counts"] == {"local": 1}
            and any(r["reason"] == "direct" for r in by_name["direct_call_local"]["routes"])
            and any(m["text"] == "DIRECT_LOCAL_OK" for m in by_name["direct_call_local"]["direct"]),
            "tool_continuation_local": by_name["local_tool_continuation"]["counts"] == {"local": 2}
            and any(e["has_tool_result"] for e in by_name["local_tool_continuation"]["events"])
            and any(
                m["text"] == "TOOL_CONTINUATION_OK"
                for m in by_name["local_tool_continuation"]["assistant_messages"]
            ),
            "failure_no_standard_fallback": by_name["local503_no_standard_fallback"]["counts"]
            == {"local": 1}
            and by_name["local503_no_standard_fallback"]["events"][0]["status"] == 503
            and any(
                m["stop_reason"] == "error"
                for m in by_name["local503_no_standard_fallback"]["assistant_messages"]
            ),
        }
        (OUTPUT / "pi-smoke-results.json").write_text(
            json.dumps({"checks": passed, "results": results}, indent=2) + "\n"
        )
        print("CHECKS " + json.dumps(passed), flush=True)
        if not all(passed.values()):
            raise SystemExit(1)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
        print("AUDIT_PROXY_STOPPED", flush=True)


if __name__ == "__main__":
    main()
