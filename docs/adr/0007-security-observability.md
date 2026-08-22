# ADR 0007: API security and safe observability

Status: accepted
Date: 2026-08-21

## Context

Phase 8 hardens the HTTP boundary before deployment. The expected threat model
is an Internet-facing API receiving malformed requests, oversized inputs,
unauthorized calls, prompt-injection text, and provider failures. The existing
grounding boundary remains:

```text
HTTP -> AgentCore -> ProfileService -> public profile evidence
```

Security and observability must not move retrieval, visibility enforcement, or
generation policy into the transport layer.

## Decision

Add a small security and observability boundary around the existing API:

- optional bearer authentication controlled by `AGENT_API_KEY`;
- server-generated request correlation IDs;
- safe one-line JSON logs to stdout;
- stable public error categories with sanitized messages;
- conservative request, text, transcript, and content-part limits;
- a deterministic `/ready` endpoint that does not call the provider.

No new dependency, persistence layer, provider SDK, or deployment component is
introduced.

## Authentication

`AGENT_API_KEY` is the key for this application's inbound API boundary. It is
distinct from `OPENAI_API_KEY`, which is used only for outbound provider
authentication and is never returned or logged.

When `AGENT_API_KEY` is absent or empty, authentication is intentionally open
for local development. When configured, `POST /agent/prepare` and
`POST /v1/responses` require `Authorization: Bearer <AGENT_API_KEY>` and reject
missing, malformed, or incorrect credentials with HTTP 401. `/health` and
`/ready` remain public so a platform can probe the service.

Authentication is enforced by the ASGI boundary before either protected route
can read, parse, or validate its request body. Therefore an unauthenticated
request receives 401 even when its body is absent, invalid JSON, or otherwise
would produce a FastAPI 422. After authentication, each route preserves its
existing payload validation contract.

The comparison is constant-time after basic header validation. The server
always generates its own request ID; a client-supplied `X-Request-ID` is not
trusted or reused.

## Request correlation

Every HTTP request receives a fresh opaque `X-Request-ID` response header. The
same ID is attached to events emitted during that request, including provider
failure events. IDs contain no user, profile, credential, or provider data.

## Safe structured logging

The application emits one-line JSON records through the `reto_banorte` logger,
which defaults to `LOG_LEVEL=INFO`. Events include request start/completion,
generation start/completion/failure, and sanitized request failure categories.
Safe fields are limited to method, path, status, duration, request ID, agent
status, provider-invoked flag, provider model name, input character count, and
stable error categories.

`provider_invoked` is true only when the provider-neutral generator reports
that an outbound provider request was actually attempted. Missing
`OPENAI_API_KEY` therefore remains false, while provider auth, timeout, and
rate-limit responses after an attempted call are true. `input_chars` is the
total character count of accepted textual request/transcript content; it never
contains the content itself.

The logger does not record request bodies, prompts, generated text, evidence,
authorization headers, cookies, API keys, stack traces, or arbitrary exception
messages. Provider model names are configuration identifiers, not credentials.
This stdout-oriented format can be collected by Cloud Run and Cloud Logging
without adding an application telemetry SDK.

## Error sanitization

`/v1/responses` keeps its local error envelope and returns stable categories for
invalid input, unsupported features, authentication, size limits, provider
failures, and unexpected internal errors. Unexpected details are replaced with
`Internal server error.` and are not serialized to clients. Existing
`/agent/prepare` validation behavior, including its 422 responses, is
preserved. No global exception handler changes the contracts of other routes.

## Input limits

The application applies conservative limits:

| Boundary | Limit |
| --- | ---: |
| request body | 64 KiB |
| total text input | 12,000 characters |
| transcript messages | 32 |
| content parts in one message | 32 |

The ASGI boundary checks the declared length as an early optimization and also
counts bytes while reading chunks for both protected POST endpoints. It
buffers at most the permitted body and replays a valid body to FastAPI, so
`65,536` bytes is accepted and `65,537` bytes is rejected even without a
`Content-Length` header. The byte limit is separate from the character limit.
The limiter is deliberately not global: non-protected routes receive the
original ASGI `receive` callable and are neither read nor reconstructed by this
boundary.
These checks reduce accidental abuse but are not a complete denial-of-service
control: a production gateway should also enforce connection, timeout,
concurrency, and streaming/chunk limits.

## Health and readiness

`GET /health` remains the liveness contract and returns `{"status":"ok"}`.
`GET /ready` returns `{"status":"ready"}` only when the initialized
AgentCore/profile retrieval boundary, policy, and Open Responses adapter are
usable and consistent, including a deterministic local preparation probe that
exercises retrieval without generation. It returns HTTP 503 with
`{"status":"not_ready"}` when an essential local component or probe is
unavailable. This checks local serving readiness only; it does not verify
OpenAI or any external provider. Neither endpoint calls OpenAI, reads a
request body, or exposes profile content.

## Deliberately out of scope

This phase does not add an in-memory rate limiter, distributed quota system,
Docker or Cloud Run configuration, IAM, WAF, persistent audit storage, tracing
SDK, metrics backend, streaming, or a new provider. Cloud Run or an API gateway
should provide multi-instance rate limiting, network controls, and platform
telemetry later. A process-local limiter would be misleading across multiple
instances.

## Validation

The regression suite covers optional authentication, malformed credentials,
request ID isolation, structured-log redaction, provider and unexpected error
sanitization, body/input limits, route-specific validation, and provider-free
readiness. Offline evals remain unchanged and no live provider calls are part
of normal validation.

## Consequences and known limitations

The API is safer to expose behind a gateway and easier to diagnose from stdout,
while remaining lightweight and provider-neutral. Authentication is opt-in for
local compatibility, so production configuration must set `AGENT_API_KEY` at
the deployment boundary. Logs intentionally omit payloads, so detailed
semantic debugging still requires a separate privacy-reviewed workflow. Rate
limiting and platform-level request controls remain deployment responsibilities.
