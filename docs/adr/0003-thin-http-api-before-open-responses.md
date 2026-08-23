# ADR 0003 — Thin HTTP API before Open Responses compatibility

## Status

Accepted.

## Context

The project already has a deterministic retrieval layer and a provider-neutral
`AgentCore`. The next roadmap step is to make that behavior reachable over HTTP,
without coupling the core to the Banorte/Open Responses transport contract or to
an LLM provider before those requirements are implemented.

## Decision

Phase 4 introduces a thin FastAPI boundary with two endpoints:

- `GET /health` for process-level liveness.
- `POST /agent/prepare` for converting a user query into the existing
  `PreparedAgentTurn` contract.

The HTTP layer owns only transport concerns:

- request parsing and schema validation;
- HTTP status mapping;
- stable JSON serialization of the internal prepared turn.

`AgentCore` remains responsible for query normalization, evidence limits and
public-only orchestration. `ProfileService` remains the trusted visibility
boundary. The API does not perform retrieval or visibility filtering itself.

The Phase 4 endpoint is intentionally **not** Open Responses compatible.
Open Responses compatibility is deferred to Phase 5 as an adapter over the
same agent core rather than a rewrite of it.

## Consequences

### Positive

- The HTTP transport can be tested independently from future LLM behavior.
- The core remains independent from FastAPI and Open Responses.
- Phase 5 can introduce `/v1/responses` semantics without changing retrieval or
  grounding rules.
- The API contract is explicit and machine-testable.

### Trade-offs

- `POST /agent/prepare` is an internal project contract, not the final Banorte
  integration contract.
- A small amount of schema mapping exists between internal dataclasses and HTTP
  Pydantic models to keep transport concerns out of the agent domain.

## Deferred

- Open Responses request/response compatibility.
- LLM generation and provider selection.
- authentication, rate limiting, CORS policy and production observability.
- container and cloud deployment configuration.
