# ADR 0010: Bounded long-transcript replay and domain retrieval

Status: accepted
Date: 2026-08-22

## Context

Parley uses stateless transcript replay, so each request can contain more
history than the previous request. The Phase 10 adapter validated each message
correctly but also summed every transcript character and rejected the request
when that aggregate exceeded `MAX_INPUT_TEXT_CHARS` (12,000). This made a
valid request fail before current-user extraction, retrieval, or generation,
even when the complete HTTP body was within the 64 KiB security limit.

The same integration exposed natural, public-profile questions about the
agent's capabilities, education, and cloud skills that should not depend on
provider generation when deterministic public retrieval is sufficient.

## Decision

Keep the HTTP security boundary unchanged:

```text
HTTP body <= 64 KiB
  -> strict message/content validation
  -> bounded request-scoped history
  -> current user query
  -> AgentCore -> ProfileService -> public evidence
```

The adapter continues to enforce:

- at most 128 transcript messages;
- at most 32 content parts per message;
- at most 12,000 characters in a simple input or individual message/content
  text;
- the existing authentication, content-type, request-ID, and safe logging
  behavior.

The old aggregate transcript-character rejection is removed. A transcript may
therefore exceed 12,000 aggregate characters only when it still fits the HTTP
body and individual-message limits. After structural validation, the adapter
keeps a suffix of complete prior messages bounded by both:

- 8 prior messages;
- 8,000 prior-history characters.

The current user question is extracted from the last user message, validated
against its individual limit, and never truncated. It is carried once as
`TextGenerationRequest.query`; the bounded `transcript` contains prior
conversation messages only. This avoids serializing the current question both
as conversation history and as the explicit current question.

## Retrieval context versus generation context

Retrieval remains stateless and public-only. For a clearly referential
follow-up, only recent user messages from the bounded history can enrich the
retrieval query. Assistant text is never used as a retrieval term, evidence,
policy, or trusted factual source. Independent questions perform ordinary
current-query retrieval and do not inherit the previous topic.

Generation may receive the bounded recent user/assistant history as
conversation data so a provider can answer naturally. The prompt continues to
label that data separately from `public_evidence`; factual claims must remain
grounded in evidence returned by `ProfileService`.

No memory, session, cache, `previous_response_id`, Redis, or persistent state
is introduced. Every request remains self-contained.

## Deterministic domain coverage

The formatter recognizes capability questions through normalized question and
asking terms, including requests for suggested questions or topics. These
requests return the existing safe capability response without retrieval or a
provider call.

`ProfileService` adds bounded overview retrieval for education/formación and
generic cloud/nube questions using existing public profile data. Education
results are formal education and academic-origin records. Generic cloud
results are the documented AWS, GCP, and Oracle Cloud skill records. A direct
provider query such as AWS retains the existing lexical ranking and calibrated
claim limits. No new profile facts, provider inference, or DevOps/SRE claim is
introduced.

Academic projects remain a separate scope from education. Visibility filtering,
nested filtering, relationship filtering, and public-only AgentCore behavior
are unchanged.

## Consequences

Long but valid Parley replays no longer fail solely because history accumulated.
The provider prompt has a predictable bounded size, while body and per-message
abuse protections remain active. Very old context can be omitted, and
coreference that depends on omitted history may still have limited retrieval;
this is an intentional stateless trade-off.

The regression suite covers long valid transcripts, body overflow, individual
message overflow, current-query preservation, bounded generation history,
assistant non-authority, independent and referential long-chat queries, and
long-transcript SSE replay with `store=false`.
