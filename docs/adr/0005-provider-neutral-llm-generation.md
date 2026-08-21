# ADR 0005: Provider-neutral grounded LLM generation

## Context

Phase 5 established a synchronous Open Responses transport and a deterministic
temporary formatter. Phase 6 needs natural-language generation without moving
retrieval or visibility enforcement into a provider SDK.

## Decision

Add a small provider-neutral `TextGenerator` contract. It receives only a
`TextGenerationRequest` containing:

- the normalized current question;
- validated user/assistant transcript data;
- `PreparedAgentTurn.evidence`;
- the trusted application `AgentPolicy`.

The first implementation is `OpenAITextGenerator`, using the official OpenAI
Python SDK and Responses API. The domain contracts do not import the SDK. No
generic provider framework, second provider, retry library, tools, or Agents
SDK is introduced.

The application boundary remains:

`HTTP -> Open Responses adapter -> AgentCore -> ProfileService -> public evidence -> TextGenerator -> OpenAI`

`AgentCore` and `ProfileService` remain unchanged. The LLM never reads
`data/profile.json`, performs retrieval, selects visibility, receives
`internal_summary` or `do_not_expose`, or changes `AgentPolicy`.

## Prompt construction and grounding

System instructions are application code in `app/llm/prompts.py`, not profile
content. They state grounding, ownership, calibration, approximation,
language, and safety rules. Conversation and public evidence are serialized as
explicitly labeled data in the user input. Retrieved content and transcript
content are data, never instructions.

The external logical `model` in `/v1/responses` is not forwarded to the
provider. The provider model is controlled only by `OPENAI_MODEL`.

## Model configuration

- `OPENAI_API_KEY` is required only when a generation call is attempted.
- `OPENAI_MODEL` is optional and defaults to `gpt-5.6-luna`.
- Missing credentials fail explicitly with a safe error; no key is invented or
  printed.
- Provider errors are mapped to safe local errors without stack traces,
  request internals, keys, or provider-sensitive details.

The OpenAI Responses request uses `store=false` because Phase 6 is stateless.
Usage remains `null`; token telemetry mapping is deferred unless it can be
added without changing the Phase 5 response contract.

## Insufficient evidence

When `PreparedAgentTurn.status` is `insufficient_evidence`, the adapter does
not call the provider. It returns the fixed safe message from Phase 5. This
reduces cost and hallucination risk while preserving deterministic behavior.

## Transcript handling

The adapter extracts only the last textual `user` message for `AgentCore`.
Validated `user`/`assistant` messages are passed separately to the generator as
conversation data. Assistant history never becomes system policy. Coreference
and persistent conversations remain limited and are deferred to later phases.

## Error strategy

The provider maps missing API key, authentication failure, timeout, rate limit,
connection/unavailability, malformed response, and empty response to safe
HTTP error envelopes scoped to `/v1/responses`. `/health` and
`/agent/prepare` retain their existing behavior.

## Testing and costs

The normal test suite uses fake generators and injected fake SDK clients. It
never consumes API credits. A live request is optional, manual, and only
allowed when `OPENAI_API_KEY` is already available in the environment. No key
is written to tracked files or output.

Phase 6 adds one provider and one generation call per evidence-backed turn;
the no-evidence short circuit avoids that cost. Production budgets, retries,
rate limiting, observability, and usage accounting are deferred.

## Limitations and deferred capabilities

This phase does not implement a second provider, automatic fallback, tools,
web search, multimodal input, embeddings, vector databases, persistent memory,
streaming, Docker, deployment, or production observability. Open Responses
support remains an intentional partial subset, not full conformance.
