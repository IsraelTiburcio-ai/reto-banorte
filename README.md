# Reto IA Banorte — CV Agent

## Objetivo

Construir un agente conversacional que permita explorar el perfil profesional de un candidato y que posteriormente será integrado mediante una interfaz compatible con Open Responses.

## Estado

Phase 10 — Parley compatibility and retrieval robustness

La Phase 6 reemplaza el formateador determinista temporal por generación grounded mediante el SDK oficial de OpenAI. El LLM solo recibe la evidencia pública ya preparada por `AgentCore`; no hace retrieval ni lee el perfil canónico.

Endpoints actuales:

- `GET /health`
- `GET /ready`
- `POST /agent/prepare`
- `POST /v1/responses`

`POST /agent/prepare` es un contrato interno. `POST /v1/responses` es un contrato interoperable parcial: Phase 5/6 no afirma full conformance con Open Responses.

## Arquitectura inicial

- `api`: interfaz HTTP delgada, schemas, adaptador Open Responses y serialización.
- `llm`: contratos provider-neutral, prompts de aplicación y proveedor OpenAI.
- `agent`: política de comportamiento y orquestación provider-neutral del agente.
- `models`: modelos internos y contratos provider-neutral.
- `services`: lógica reutilizable e integraciones externas.
- `core`: configuración, logging, seguridad y manejo de errores.
- `data`: fuente de verdad estructurada del CV.
- `evals`: datasets y scripts para evaluar el agente.
- `tests`: pruebas automatizadas.
- `docs`: arquitectura, decisiones técnicas y diagramas.

Flujo actual:

`POST /v1/responses -> Open Responses adapter -> AgentCore -> ProfileService -> public evidence -> LLM generator -> OpenAI Responses API`

`ProfileService` conserva la responsabilidad de enforcement de visibilidad y `AgentCore` continúa siendo public-only. La API no implementa retrieval ni reglas de exposición propias.

El subset soporta texto síncrono, input string, mensajes `user`/`assistant`, partes `input_text` y replay estructural sin estado. Phase 6 genera texto mediante el provider en cada request textual válido, incluso cuando el retrieval dirigido está vacío; recibe el contexto público base y evidencia específica cuando existe. Los hechos sobre Israel siguen limitados al contexto/evidence públicos y grounded. `stream` ausente, `null` o `false` devuelve JSON; `stream=true` devuelve SSE con la respuesta textual completa materializada antes de emitirla. Rechaza `system`/`developer`, tools, multimodalidad, persistencia y capacidades conversacionales avanzadas.

## Open Responses subset

```bash
curl -X POST http://localhost:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"banorte-cv-agent","input":"¿Qué experiencia tiene Israel con MCP?"}'
```

El request requiere `input`; `model` es opcional por compatibilidad con la configuración de Parley/Banorte y usa `banorte-cv-agent` cuando está ausente o es `null`. Un string no vacío se preserva y un valor vacío o whitespace-only se rechaza. `stream` es `false` por defecto: `null` y `false` conservan JSON, mientras `true` usa `text/event-stream; charset=utf-8`, `Cache-Control: no-store` y la secuencia SSE de ciclo de vida de Open Responses. El endpoint no mantiene conversaciones: acepta `store` ausente, `null` o `false` como formas stateless, pero rechaza `store=true`; también rechaza `previous_response_id`, `background`, `compaction`, tools o visibilidad seleccionable por el cliente.

## Phase 6 — LLM integration

Configura el proveedor únicamente mediante variables de entorno:

```bash
export OPENAI_API_KEY='...'
export OPENAI_MODEL='gpt-5.6-luna'  # opcional; este es el default
```

`model` en la request es el identificador lógico externo y no selecciona el modelo del proveedor. `OPENAI_MODEL` controla el modelo OpenAI. Nunca publiques API keys en el repositorio, README, logs o tests.

El generador usa la Responses API oficial con `store=false`, sin tools, token streaming, multimodalidad, memoria persistente ni fallback automático a otro proveedor. El SSE del endpoint es streaming de transporte de una respuesta ya completa; no activa streaming del provider. El modelo lógico recibe el contexto público base y la evidencia pública aprobada por `AgentCore`; cualquier hecho sobre Israel debe permanecer grounded en esos datos.

## Phase 7 — Evals

La suite reproducible de evaluación mide factuality, groundedness, relevance,
abstention, ownership y skill calibration, métricas aproximadas, distinción
profesional/académica/conceptual, preguntas fuera de alcance, prompt injection,
información restringida, follow-ups, natural-language retrieval y entradas en
español e inglés.

Las evals offline son deterministas, usan `AgentCore` y no consumen la API real:

```bash
python3 -m evals.runner
```

El reporte separa `OFFLINE CASE STATUS`, `CHECK COVERAGE` y
`LIVE/MANUAL COVERAGE`. `PASS` offline significa únicamente que los checks de
retrieval/contrato aplicables pasaron; `NOT_EVALUATED` cubre factualidad
semántica, groundedness de la prosa, ownership, skill calibration y paráfrasis
que requieren respuesta generada y revisión humana. No se presenta el pass
rate offline como calidad general del agente. `N/A` significa que el caso no
declaró esa expectativa; `NOT_EVALUATED` significa que sí la declaró pero el
modo actual no puede evaluarla. Coverage es checks `PASS + FAIL` sobre checks
aplicables (`PASS + FAIL + NOT_EVALUATED`), excluyendo `N/A`.

También pueden ejecutarse los tests de infraestructura con la suite normal:

```bash
python3 -m unittest discover -v tests
```

Las evals live están separadas y requieren confirmación explícita y un límite
pequeño de casos:

```bash
python3 -m evals.runner --live --confirm-live --limit 1
```

No se usa LLM-as-a-judge en esta fase. El modo offline no guarda respuestas
generadas como verdad absoluta y no imprime secretos. El follow-up
`¿Y cuál usabas más?` permanece como limitación conocida y puede fallar por
coreferencia.

## Phase 8 — Security & Observability

Phase 8 agrega controles pequeños y explícitos en el borde HTTP sin mover
retrieval, grounding ni visibilidad fuera de `AgentCore` y `ProfileService`:

- `AGENT_API_KEY` habilita autenticación Bearer opcional para
  `/agent/prepare` y `/v1/responses`; no es la API key de OpenAI.
- Cuando está configurada, la autenticación ocurre antes de que FastAPI lea o
  valide el body; payloads inválidos sin credenciales reciben 401.
- `GET /health` y `GET /ready` permanecen públicos para liveness/readiness y no
  llaman al proveedor.
- `/ready` valida la policy, la estructura del adapter y ejecuta un probe local
  de `AgentCore -> ProfileService`; devuelve `503 {"status":"not_ready"}` si
  un componente esencial no es usable. No verifica ni llama a OpenAI.
- Cada respuesta incluye un `X-Request-ID` nuevo generado por el servidor.
- Los logs son JSON de una línea y solo contienen campos operativos seguros:
  ruta, estado, duración, categoría de error, estado del agente y modelo del
  proveedor. No registran payloads, respuestas, evidencia, headers, cookies ni
  secretos.
- El límite real del body es 256 KiB (262,144 bytes) en los dos POST protegidos,
  incluso sin `Content-Length` o cuando llegan varios chunks. Además se
  rechaza texto simple, mensaje o parte de contenido mayor a 12,000 caracteres,
  transcripts de más de 128 mensajes y mensajes de más de 32 partes. Un transcript histórico
  válido puede superar el antiguo total agregado mientras el body permanezca
  dentro de 256 KiB; antes de retrieval/generación se conserva solo una ventana
  reciente de hasta 8 mensajes y 8,000 caracteres de historial. La pregunta
  actual se conserva completa y viaja una sola vez como `current_user_question`.
- El límite de 256 KiB no es global: las rutas públicas conservan el `receive`
  ASGI original y no consumen ni reconstruyen su body mediante este middleware.
- `provider_invoked` solo es `true` cuando el generador confirma un intento
  outbound al proveedor; `input_chars` es el total de texto aceptado del
  request/transcript y nunca contiene el texto.
- Los errores inesperados se convierten en respuestas sanitizadas; los errores
  422 existentes de `/agent/prepare` se conservan.

Plantilla local segura:

```bash
cp .env.example .env
```

Configura `AGENT_API_KEY` únicamente en el entorno donde se necesite proteger
la API. Con la key configurada, la UI/API de Banorte debe enviar una credencial
del agente como header conceptual:

```text
Authorization: Bearer <AGENT_API_KEY>
```

La API key de la UI de Banorte debe corresponder a `AGENT_API_KEY`; nunca se
debe introducir `OPENAI_API_KEY` en Banorte. `OPENAI_API_KEY` continúa siendo
exclusivamente una credencial de salida hacia el proveedor y nunca debe
aparecer en código, logs, README o tests. El rate limiting distribuido, WAF,
IAM, tracing/metrics backend, Docker y despliegue quedan fuera de Phase 8.

Validación:

```bash
python3 -m unittest discover -v tests
python3 -m evals.runner
```

## Phase 9 — Containerization / Docker

La imagen se construye desde el `pyproject.toml` canónico, incluye únicamente
el runtime de la aplicación y `data/profile.json`, y ejecuta Uvicorn como un
usuario no root. El contenedor escucha en `0.0.0.0`, respeta `PORT` y conserva
los logs estructurados en stdout.

### Validación estática

Los tests de contrato parsean las instrucciones efectivas del Dockerfile y
evalúan las reglas relevantes de `.dockerignore`, incluyendo continuaciones,
orden de `USER`, semántica de `CMD`, declaraciones `ENV`/`ARG`, destinos
`COPY` y negaciones last-match-wins. También contienen regresiones contra
mutaciones conocidas del review. La suite normal no requiere Docker.

Build local:

```bash
docker build -t reto-banorte-cv-agent:local .
```

Run sin proveedor:

```bash
docker run --rm -p 8080:8080 reto-banorte-cv-agent:local
```

Configuración de runtime con placeholders — nunca uses claves reales en el
Dockerfile, argumentos de build, README o imagen:

```bash
docker run --rm \
  -p 8080:8080 \
  -e AGENT_API_KEY="<agent-api-key>" \
  -e OPENAI_API_KEY="<openai-api-key>" \
  -e OPENAI_MODEL="gpt-5.6-luna" \
  reto-banorte-cv-agent:local
```

`AGENT_API_KEY` protege nuestros endpoints privados; `OPENAI_API_KEY` es la
credencial outbound del proveedor. Ambas se inyectan únicamente al ejecutar
el contenedor. `.env` está excluido del contexto y no se copia a la imagen.

Endpoints para smoke tests:

- `GET /health` — liveness, no requiere proveedor.
- `GET /ready` — readiness local, no llama a OpenAI.
- `POST /v1/responses` — contrato interoperable; requiere auth si
  `AGENT_API_KEY` está configurada.

Phase 9 solo prepara la imagen. Cloud Run, GCP, Artifact Registry, Secret
Manager, IAM, Compose, Kubernetes y deployment permanecen fuera de alcance.

## Phase 10 — Parley compatibility and retrieval robustness

Phase 10 keeps `POST /v1/responses` stateless while accepting the structural
metadata emitted when Parley replays a transcript. Assistant messages may
include a valid `msg_` `id`, `type: "message"`, a supported lifecycle
`status`, and `output_text` annotations. These fields are validated as
transport metadata, then discarded before `AgentCore` or the provider prompt;
only message roles and text remain. `system` and `developer` roles,
stateful continuation, and unsupported capabilities remain rejected.

All valid textual requests use the normal grounded pipeline and invoke the
configured provider, including greetings, identity, social, out-of-scope and
follow-up questions. The adapter does not contain a conversational intent
classifier or a catalog of fixed answers. Safety and visibility remain
enforced by the trusted policy, `AgentCore`, `ProfileService` and provider
instructions.

The auditable pre-Banorte QA fixture is `evals/pre_banorte_cases.json` (30
cases). Run `python3 scripts/pre_banorte_smoke.py` for the default offline
validation; it makes zero HTTP/provider calls and does not use an LLM judge.
Set `PRE_BANORTE_BASE_URL` only when an authorized transport smoke run is
intended; generated-answer cases are reported for manual review.

The lexical retriever keeps exact ID, name/title, substring, context and
deterministic token matching for the current query. It does not classify
intent, resolve coreference, or maintain a second conversational vocabulary.
Assistant text is never treated as evidence or retrieval instructions. A
bounded recent transcript is provided to generation as conversation data, but
every factual claim must come from the public context or public evidence
returned by `ProfileService`. Results remain ordered public evidence copies
only; they do not infer dates, ownership, relevance, skills, or technologies
that are not present in the profile.
Exact IDs, names, titles, and existing substring ranking remain unchanged.

### Conversational UX and professional representation

The agent separates conversation from factual grounding. Every valid textual
request reaches the provider, including social, identity, unknown-data and
follow-up questions. `AgentCore` still performs the current-query public-only
retrieval, but an empty or partial retrieval package does not short-circuit
conversation. The provider receives a detached public canonical profile
context, specific public evidence when available, bounded transcript data and
the current question, then handles natural-language interpretation and
professional synthesis without inventing facts or ownership.

The grounding boundary limits what may be asserted about Israel; it does not
prevent natural explanation or favorable synthesis of supported facts. The
public profile context is generated from `ProfileService.get_profile("public")`
and measured at 36,899 characters / 37,305 UTF-8 bytes for the current
canonical profile. A request may contain up to 128 replayed transcript
messages, while only the most recent 8 messages and 8,000 characters are
passed to generation. The conversation remains stateless and assistant
transcript text is never evidence.
The protected request body limit is 256 KiB (262,144 bytes), with the exact
boundary accepted and the next byte rejected; individual messages remain
limited to 12,000 characters and content parts to 32. Generic token suffix
normalization remains limited to deterministic lexical retrieval and does not
attempt to understand slang, aliases, follow-up intent or conversational
coreference.

The local product runner exercises the 30-turn Parley-style conversation with
a fake provider by default:

```bash
python3 scripts/conversational_ux_product.py
```

It reports HTTP status, provider invocation, evidence count, request sizes at
turns 10/20/30, transcript size, and semantic `REVIEW` markers for generated
prose. It never uses an LLM-as-a-judge. A live run is opt-in only after
`OPENAI_API_KEY` has been loaded into the process:

```bash
python3 scripts/conversational_ux_product.py --live
```

The live option makes one sequential local session of at most 30 requests and
never retries. It does not print or log the provider credential.

Phase 10 does not add conversation memory, provider streaming, new
dependencies, or private Parley data. The SSE transport remains the existing
materialized-response compatibility path.

### Validación local del contenedor

La siguiente validación se ejecutó localmente en macOS Apple Silicon `arm64`
con Docker `29.7.2`; no es una validación de Cloud Run:

- `docker build` pasó para `reto-banorte-cv-agent:phase9`.
- Tamaño de imagen: `56,021,395` bytes (aproximadamente `56 MB`).
- Usuario efectivo: `uid=999(app) gid=999(app) groups=999(app)`.
- Con el puerto default `8080`, `/health` y `/ready` devolvieron `200`.
- Con `PORT=9090`, `/health` devolvió `200`.
- Con una `AGENT_API_KEY` fake: requests sin autorización y con Bearer
  incorrecto devolvieron `401`; Bearer correcto permitió `/agent/prepare` con
  retrieval público MCP válido.
- `/v1/responses` autenticado con una pregunta sin evidencia pública devolvió
  `200`, respuesta Open Responses válida y abstención segura sin requerir
  `OPENAI_API_KEY` ni invocar al proveedor.
- `docker stop -t 10` terminó limpiamente en aproximadamente `0.387 s`.
- La ejecución `--read-only` mantuvo `/health` y `/ready` en `200`.
- `/app/.env` estuvo ausente; `Config.Env` no incluyó `OPENAI_API_KEY` ni
  `AGENT_API_KEY`; `docker history --no-trunc` no mostró credenciales de la
  aplicación.

La compatibilidad con Cloud Run continúa siendo una decisión de diseño
preparatoria. El despliegue real y sus recursos están fuera del trabajo actual
de compatibilidad de Phase 10.

## Ejecución local

```bash
uvicorn app.api.main:app --reload
```

Ejemplo:

```bash
curl -X POST http://127.0.0.1:8000/agent/prepare \
  -H 'Content-Type: application/json' \
  -d '{"query":"experiencia con MCP","max_results":5}'
```

## Roadmap

0. Project foundation
1. Professional knowledge base
2. Profile retrieval layer
3. Agent core
4. HTTP API
5. Open Responses compatibility
6. LLM integration
7. Evaluations
8. Security and observability
9. Containerization
10. Parley compatibility and retrieval robustness
11. Cloud deployment
12. Documentation and demo

## Filosofía de arquitectura

Se seguirá una estrategia incremental:

`build the smallest reliable layer first`

Cada etapa debe tener una responsabilidad clara antes de introducir la siguiente. Las decisiones sobre proveedor de LLM, modelo, RAG, embeddings, vector database, cloud provider y framework de agentes se tomarán después, cuando existan suficientes requisitos.
