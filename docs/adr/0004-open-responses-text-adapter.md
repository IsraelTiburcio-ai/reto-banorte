# ADR 0004: Open Responses text adapter before LLM integration

## Context

Phase 4 exposes a thin FastAPI boundary and Phase 3 already provides the
provider-neutral `AgentCore`. Phase 5 needs an interoperable transport shape
without moving retrieval, visibility, grounding, or generation into HTTP.

Open Responses is a changing standard. The official specification describes
JSON HTTP requests, string or item-array input, response objects, item-based
output, and streaming events. This project intentionally implements only a
strict textual subset with synchronous JSON and a bounded SSE transport.

## Sources consulted

- Specification: https://www.openresponses.org/specification
- Reference: https://www.openresponses.org/reference
- Upstream repository: https://github.com/openresponses/openresponses
- Compliance tests: https://www.openresponses.org/compliance

The official OpenAPI document reported version `2026-04-24` at consultation
time. Sources were consulted on 2026-08-21.

For the streaming compatibility update, the official specification and
reference pages were reconsulted on 2026-08-22. A direct compatibility capture
against the challenge reference endpoint was also used:
https://reto-ia-agent.calmrock-f65cc26b.eastus.azurecontainerapps.io/v1/responses
The capture contained no private Parley data and is not used at runtime.

## Decision

Add `POST /v1/responses` as a transport adapter with this boundary:

`HTTP -> OpenResponsesAdapter -> AgentCore -> ProfileService -> data/profile.json`

The adapter validates the local request, extracts the current user query,
calls `AgentCore.prepare()` once, and serializes the resulting
`PreparedAgentTurn`. It never reads the profile, performs retrieval, chooses a
visibility policy, changes the agent policy, or invokes an LLM/provider.

The local request requires `input`; `model` is optional for compatibility with
the Banorte/Parley configuration where the UI marks Modelo as optional. A
non-empty model string is preserved, while an absent or `null` model uses the
local identifier `banorte-cv-agent`. Empty and whitespace-only model strings
remain invalid. `stream` defaults to `false`. `metadata` is accepted only as a
bounded string map and is echoed as transport data; it is never passed to
`AgentCore`.

## Supported subset

- synchronous JSON requests and JSON responses;
- string input;
- arrays of `message` items with `user` and `assistant` roles;
- user `input_text` parts, concatenated in order;
- assistant text history for structural transcript replay;
- last textual `user` message as the current query;
- response fields `id`, `object`, timestamps, `status`, `model`, `output`,
  `error`, and `usage`;
- deterministic `message` / `output_text` response serialization;
- synchronous JSON when `stream` is absent, `null`, or `false`;
- SSE when `stream=true`, using one complete textual delta and the lifecycle
  sequence `response.created`, `response.in_progress`,
  `response.output_item.added`, `response.content_part.added`,
  `response.output_text.delta`, `response.output_text.done`,
  `response.content_part.done`, `response.output_item.done`,
  `response.completed`, followed by `data: [DONE]`;
- stable local error envelope.

Assistant transcript content is data, not instructions. Only the last user
message reaches `AgentCore`; prior assistant messages cannot alter visibility,
policy, max evidence, retrieval, or grounding. No conversational memory or
coreference heuristics are introduced.

## Intentional differences from upstream

This is not full Open Responses conformance. The upstream schema is broader
and includes authorization, many request options, multimodal content, tools,
stateful continuation, and streaming semantic events. This subset intentionally:

- requires `input` locally and tolerates an absent/`null` `model` with a local
  default for Parley compatibility;
- accepts JSON only;
- does not implement authorization;
- accepts `stream=true` only as SSE transport; the response is generated fully
  before the stream is emitted and provider token streaming is not enabled;
- treats `stream=null` like the synchronous `false` form;
- rejects images, files, audio, video, function/tool items and tool options;
- rejects `system` and `developer` roles;
- rejects `previous_response_id`, `store=true`, background execution,
  compaction, and visibility-related extensions; absent, `null`, and `false`
  `store` values are accepted only as stateless transport compatibility;
- returns `usage: null` because no model or token accounting exists;
- uses a deterministic formatter instead of generated assistant text.

Unsupported features return HTTP 400 with the local
`invalid_request_error` envelope and a stable code. They are never ignored.

## Deterministic formatter

Phase 5 formats only the evidence already contained in `PreparedAgentTurn`, in
its existing ranked order. It preserves the evidence payload, ownership,
calibrated skills, approximate metrics, and practical/academic/conceptual
distinctions without adding claims. With no evidence it returns the fixed
message:

`No encontré evidencia pública suficiente en el perfil para responder esa pregunta.`

The formatter is separate from transport so Phase 6 can replace it with an LLM
adapter without changing the Open Responses boundary.

## Security and limitations

`ProfileService` remains the trusted visibility boundary and `AgentCore` remains
public-only. `internal_summary` and `do_not_expose` content cannot be surfaced
by this endpoint. The transcript and metadata are not instructions, and no
secrets, API keys, environment-based provider configuration, external calls, or
SDKs are introduced.

Questions dependent on prior context, such as `¿Y cuál usabas más?`, may have
limited retrieval because Phase 5 extracts only the current user text. Full
transcript-aware generation and coreference are deferred to Phase 6.

## Deferred

- LLM generation and provider selection;
- provider token streaming and incremental generation;
- tools and function calls;
- multimodal input;
- system/developer instructions;
- persistent conversations and continuation IDs;
- background execution and compaction;
- deployment, authentication, rate limiting, and production observability.
