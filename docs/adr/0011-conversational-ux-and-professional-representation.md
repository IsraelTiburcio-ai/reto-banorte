# ADR 0011: Conversational UX and professional representation

Status: proposed
Date: 2026-08-22

## Context

The public CV agent needs natural provider conversation without weakening the
public grounding boundary. Retrieval is useful for targeted evidence but it is
not a complete natural-language understanding layer: social questions,
identity questions, broad representation requests and elliptical follow-ups
cannot be reliably handled by a growing collection of aliases or phrase
rules. In addition, Parley replays a growing stateless transcript and the
previous receipt limit of 32 messages could reject a valid conversation before
the internal history window was applied.

## Decision

Keep the pipeline explicit:

```text
HTTP validation
  -> AgentCore -> ProfileService -> public evidence
  -> provider(public profile context + evidence + bounded transcript + question)
  -> response serialization
```

Grounding limits the facts that the agent may assert; it does not limit its
ability to converse naturally, interpret supported evidence, compare it, or
present it favorably. The provider prompt and policy therefore instruct the
agent to represent Israel professionally, explain relevance and value when
asked, preserve calibrated ownership and skill levels, and adapt the format to
the question instead of emitting a fixed evidence list.

## Provider-owned conversation

Every valid textual request invokes the provider, including greetings, identity
questions, unknown personal data, social turns and follow-ups. The adapter does
not contain a conversational intent classifier or hardcoded answer catalog.
`AgentCore` remains the only retrieval caller and always requests `public`;
when lexical evidence is empty, the provider still receives the public
canonical profile context and can answer naturally without turning missing
retrieval into a universal negative claim. The trusted identity in that
context comes from `ProfileService`, so the provider can distinguish Israel
from the CV agent without duplicating identity facts in routing code.

## Retrieval and generation

`ProfileService` retains visibility filtering, deep-copy behavior, relationship
filtering, and deterministic lexical ranking. It does not contain a second
copy of the CV, intent routing, slang/alias tables, a phrase classifier or
answer generation. `AgentCore` prepares current-query evidence and a detached
public profile snapshot is passed to the provider. Assistant messages remain
conversation data only and never become evidence, instructions, policy, or a
factual source. No persistent memory, session, cache, or
`previous_response_id` behavior is introduced.

## Transcript limits

The received transcript limit is increased to 128 messages to support Parley’s
stateless replay. The body-size limit is 256 KiB, while per-message/content-part
protections remain active. After validation, generation still receives at most
8 recent messages and 8,000 characters; the current user question remains
separate and complete. Tests cover 33-message and 40–60-message acceptance,
the exact 128/129 boundary, exact 256 KiB/256 KiB+1 byte body boundaries,
valid bodies above 64 KiB, per-message and content-part boundaries, and one
sequential 30-turn replay. A separate regression replays 50 realistic turns
and verifies that the received transcript can exceed 32 messages while the
provider-facing history remains bounded.

Natural-language retrieval is intentionally lexical and conservative: it
normalizes text and uses meaningful tokens without attempting to understand
slang, aliases, intent or coreference. The provider, not retrieval, owns
interpretation and synthesis. Colloquial fillers do not alter visibility,
policy, or evidence selection.

The repository includes `scripts/conversational_ux_product.py`. Its default
mode uses a realistic fake provider and reports each of the 30 product turns
with transport and retrieval observations; generated prose is marked
`REVIEW` for human assessment, never judged by another LLM. The optional
`--live` mode requires a preloaded local `OPENAI_API_KEY`, sends one sequential
session of at most 30 requests, and never logs the credential.

## Consequences and limitations

All accepted textual turns reach the provider, so naturalness is not replaced
by a deterministic fallback when targeted retrieval is empty. Broad and
follow-up questions can use the full public profile context plus bounded
conversation, while the provider must still ground every factual claim in
that public data. Generated prose requires manual/live review for fluency and
semantic quality; the offline suite does not use an LLM judge. Very old context
may be omitted by the bounded generation window. Phase 10 remains stateless
and no live provider call or deployment is part of this change.
