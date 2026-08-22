# ADR 0009: Phase 10 Parley compatibility without state

## Context

Parley replays the previous Open Responses output as an assistant message.
That message can contain transport metadata such as an identifier, a message
type, a lifecycle status, and output-text annotations. The existing adapter
was intentionally conservative, but rejecting these valid structural fields
prevented transcript replay. The same integration also exposed two retrieval
usability gaps: standalone greetings should not require a provider, and broad
career, project, and identity questions should not rank accidental generic
matches above the relevant profile sections.

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
text. They are not represented in `ConversationMessage`, cannot reach
`AgentCore`, and cannot reach the provider prompt. The last textual user
message is the current query. For a bounded set of clearly context-dependent
follow-ups identified by normalized anaphoric markers (demonstratives,
referential `cuál/cuáles`, continuation markers, and object-reference verbs),
the adapter may retry retrieval once with the last few user messages joined to
that current query; the returned turn keeps the current user question.
Assistant text and metadata remain structural context only and are never used
as retrieval instructions or factual evidence. This request-scoped fallback is
not conversation memory and does not change policy, visibility, evidence
limits, or provider behavior. `system` and `developer` roles, stateful
continuation IDs, and other unsupported Open Responses capabilities remain
rejected.

The adapter does not add persistent memory or heuristic answer generation.
The bounded marker detection only decides whether to enrich the current
retrieval query with recent user text. `store` remains stateless compatibility
metadata: absent, `null`, and `false` are accepted; `true` remains rejected.

## Deterministic social and meta handling

The adapter recognizes standalone normalized greetings such as `hola`,
`hello`, `hey`, and the supported Spanish greetings. It also handles a small
deterministic set of thanks, goodbyes, agent identity/capability questions,
sensitive-data requests, and explicit out-of-scope prompts. Punctuation, case,
whitespace, and accents are normalized for these checks. These local responses
perform no retrieval and no provider call. A greeting with additional question
text is not a shortcut; it follows the normal grounded pipeline.

The deterministic responses are deliberately narrow and do not claim to be a
general conversational model. Sensitive and out-of-scope requests receive a
safe domain redirect with `insufficient_evidence` status; no restricted data is
retrieved or exposed.

## Deterministic overview retrieval

`ProfileService` retains exact ID, title/name, context, body, and relationship
ranking for ordinary lexical queries. When a bounded overview intent is
recognized, it returns visible copies in canonical profile order:

- project overviews return public project entities;
- academic and professional project follow-ups use bounded profile-derived
  scopes, without treating prior assistant claims as facts;
- career overviews prioritize `career_story`, `professional_summary`,
  `identity`, and then public experience entities;
- identity overviews return `identity`, `professional_summary`, and
  `career_story`.

The intent vocabulary is deliberately small and token-based. It is not fuzzy
matching, embeddings, a vector database, or an LLM. Unknown terms prevent the
overview shortcut, so accidental words do not silently broaden retrieval.
Every candidate still passes through the existing visibility copy boundary;
`internal_summary`, `do_not_expose`, restricted nested text, and hidden
relationship origins cannot participate in public retrieval or ranking.

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
coreference, such as `¿Y cuál usabas más?`, may still have limited retrieval;
the adapter only performs the bounded prior-user retry described above and does
not implement conversational memory or heuristic answer generation.
Overview ordering is deterministic profile order, not an inferred claim about
which project is objectively most important. Generated factuality and
grounding still require the existing offline checks and, when authorized,
manual review of live provider responses.
