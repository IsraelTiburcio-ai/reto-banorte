# ADR 0011: Conversational UX and professional representation

Status: proposed
Date: 2026-08-22

## Context

The public CV agent was technically grounded but treated every input as a
retrieval query. Greetings, basic identity questions, and unavailable personal
data therefore received the same generic insufficient-evidence sentence as an
unknown professional technology. Broad education and professional questions
also depended too heavily on incidental lexical matches. In addition, Parley
replays a growing stateless transcript and the previous receipt limit of 32
messages could reject a valid conversation before the internal history window
was applied.

## Decision

Keep the pipeline explicit:

```text
HTTP validation
  -> deterministic conversation/identity/safety handling when applicable
  -> AgentCore -> ProfileService -> public evidence
  -> provider generation for evidence-backed professional synthesis
```

Grounding limits the facts that the agent may assert; it does not limit its
ability to converse naturally, interpret supported evidence, compare it, or
present it favorably. The provider prompt and policy therefore instruct the
agent to represent Israel professionally, explain relevance and value when
asked, preserve calibrated ownership and skill levels, and adapt the format to
the question instead of emitting a fixed evidence list.

## Deterministic conversational handling

Normalized standalone greetings, repeated-letter greetings such as
`holaaaaaaaa`, small talk, thanks including an informal suffix, and goodbyes
are answered locally without retrieval or provider calls. A mixed greeting is
not local handling when it contains a professional question; the professional
part follows the grounded pipeline. A small out-of-scope social question may
receive a brief safe response and a CV-domain invitation.

Basic person identity questions use the public `identity.full_name` read by
`ProfileService`; they do not depend on a lexical search hit. The agent and the
person remain distinct: the agent identifies itself as a CV agent and does not
impersonate Israel. Unknown personal data, such as age, gets a data-specific
calibrated response without inventing a value. Unknown technology keeps the
distinction between “not enough information to affirm” and an absolute
negative claim.

## Retrieval and generation

`ProfileService` retains visibility filtering, deep-copy behavior, relationship
filtering, and deterministic ranking. It adds bounded professional and
professional-representation overview modes that return public documents,
experiences, and projects in canonical order. These modes provide evidence for
questions such as professional trajectory, strengths, role fit, and hiring
relevance; they do not contain a second copy of the CV or generate claims.

Context-dependent follow-ups may enrich retrieval with recent user messages.
Assistant messages remain conversation data only and never become evidence,
instructions, policy, or a factual source. Independent questions continue to
use only the current query. No persistent memory, session, cache, or
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

Natural-language retrieval uses generic question noise removal and a small
suffix normalization only during deterministic token fallback. This lets
ordinary variants such as a Spanish infinitive and an English gerund reach the
same documented public skill without adding phrase-specific production rules.
Colloquial fillers do not alter visibility, policy, or evidence selection.

The repository includes `scripts/conversational_ux_product.py`. Its default
mode uses a realistic fake provider and reports each of the 30 product turns
with transport and retrieval observations; generated prose is marked
`REVIEW` for human assessment, never judged by another LLM. The optional
`--live` mode requires a preloaded local `OPENAI_API_KEY`, sends one sequential
session of at most 30 requests, and never logs the credential.

## Consequences and limitations

Social and identity turns no longer spend retrieval/provider work, and broad
professional questions can receive evidence suitable for natural synthesis.
Generated prose still requires manual/live review for fluency and semantic
quality; the offline suite does not use an LLM judge. Very old context may be
omitted by the bounded generation window, and coreference remains limited by
the available recent user context. Phase 10 remains stateless and no live
provider call or deployment is part of this change.
