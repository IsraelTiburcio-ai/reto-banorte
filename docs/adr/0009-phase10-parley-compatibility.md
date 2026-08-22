# ADR 0009: Phase 10 Parley compatibility without state

## Context

Parley replays the previous Open Responses output as an assistant message.
That message can contain transport metadata such as an identifier, a message
type, a lifecycle status, and output-text annotations. The existing adapter
was intentionally conservative, but rejecting these valid structural fields
prevented transcript replay. The same integration exposed that lexical
retrieval should remain a bounded evidence mechanism rather than becoming a
conversation classifier for greetings, broad questions or coreference.

## Decision

Keep the boundary stateless:

`HTTP -> Open Responses adapter -> AgentCore -> ProfileService -> public evidence`

Transcript replay accepts only `user` and `assistant` messages. A message may
include:

- an optional `msg_` identifier;
- `type: "message"` or no type for compatibility with observed Parley input;
- `status` equal to `in_progress`, `completed`, or `incomplete`;
- output-text `annotations` represented as a list of objects.

These fields are validated and retained only long enough to extract message
text. They are not represented in `ConversationMessage` or sent as transport
metadata to `AgentCore`. The last textual user message is the current query.
`AgentCore` passes only that query to the deterministic public retriever; the
provider receives the bounded transcript separately for natural-language
interpretation.
Assistant text and metadata remain structural context only and are never used
as retrieval instructions or factual evidence. This request-scoped fallback is
not conversation memory and does not change policy, visibility, evidence
limits, or provider behavior. `system` and `developer` roles, stateful
continuation IDs, and other unsupported Open Responses capabilities remain
rejected.

The adapter does not add persistent memory, heuristic answer generation or an
intent classifier. `store` remains stateless compatibility
metadata: absent, `null`, and `false` are accepted; `true` remains rejected.

## Deterministic retrieval boundary

`ProfileService` retains exact ID, title/name, context, body and relationship
ranking for ordinary lexical queries, plus conservative meaningful-token
fallback. It does not implement social handling, broad intent routing, aliases,
slang, phrase lists or coreference. Every candidate still passes through the
existing visibility copy boundary; `internal_summary`, `do_not_expose`,
restricted nested text, and hidden relationship origins cannot participate in
public retrieval or ranking.

## Compatibility evidence and scope

The prior SSE compatibility implementation was checked against the challenge
reference endpoint capture used by this repository. The capture contained no
private Parley data and is not accessed at runtime. Phase 10 does not claim
full Open Responses conformance, does not change the SSE lifecycle contract,
and does not add provider token streaming, persistence, authentication,
Cloud Run, or deployment behavior.

## Pre-Banorte conversational QA

`evals/pre_banorte_cases.json` contains 30 explicit, auditable cases covering
social/meta turns, mixed greetings, professional and overview questions,
follow-ups, Spanish/English input, natural-language retrieval, unknowns,
sensitive information, ownership and skill calibration, restricted-data
boundaries, and SSE replay. Each case records reference behavior, required and
forbidden properties, evidence expectations, retrieval/provider expectations,
and whether semantic review is required.

`python3 scripts/pre_banorte_smoke.py` validates the fixture offline by default
and makes zero HTTP or provider calls. Supplying `PRE_BANORTE_BASE_URL` enables
an optional transport smoke run; generated answers are reported as `REVIEW`,
not judged by an LLM. `PRE_BANORTE_API_KEY`, when used for an authorized test,
is read only for the request header and is never printed.

## Consequences and limitations

Transcript metadata no longer blocks replay, while transcript text remains
untrusted data rather than instructions. Follow-up questions that depend on
coreference, such as `¿Y cuál usabas más?`, are given to the provider together
with the bounded transcript and public context; lexical retrieval may be empty
or partial, but no deterministic classifier attempts to resolve the reference.
The request remains stateless and does not implement conversational memory.
Overview ordering is deterministic profile order, not an inferred claim about
which project is objectively most important. Generated factuality and
grounding still require the existing offline checks and, when authorized,
manual review of live provider responses.
