# Reto IA Banorte — CV Agent

## Objetivo

Construir un agente conversacional que permita explorar el perfil profesional de un candidato y que posteriormente será integrado mediante una interfaz compatible con Open Responses.

## Estado

Phase 8 — Security & Observability

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

El subset soporta texto síncrono, input string, mensajes `user`/`assistant`, partes `input_text` y replay estructural sin estado. Phase 6 genera texto grounded solo cuando existe evidencia; sin evidencia conserva el fallback fijo y evita la llamada al proveedor. Rechaza `system`/`developer`, streaming, tools, multimodalidad, persistencia y capacidades conversacionales avanzadas.

## Open Responses subset

```bash
curl -X POST http://localhost:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"banorte-cv-agent","input":"¿Qué experiencia tiene Israel con MCP?"}'
```

El request requiere `input`; `model` es opcional por compatibilidad con la configuración de Parley/Banorte y usa `banorte-cv-agent` cuando está ausente o es `null`. Un string no vacío se preserva y un valor vacío o whitespace-only se rechaza. `stream` es `false` por defecto y `metadata` se conserva únicamente como dato de transporte. El endpoint no mantiene conversaciones y no acepta `previous_response_id`, `store`, `background`, `compaction`, tools o visibilidad seleccionable por el cliente.

## Phase 6 — LLM integration

Configura el proveedor únicamente mediante variables de entorno:

```bash
export OPENAI_API_KEY='...'
export OPENAI_MODEL='gpt-5.6-luna'  # opcional; este es el default
```

`model` en la request es el identificador lógico externo y no selecciona el modelo del proveedor. `OPENAI_MODEL` controla el modelo OpenAI. Nunca publiques API keys en el repositorio, README, logs o tests.

El generador usa la Responses API oficial con `store=false`, sin tools, streaming, multimodalidad, memoria persistente ni fallback automático a otro proveedor. El modelo lógico recibe texto grounded únicamente con evidencia pública aprobada por `AgentCore`.

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
- `GET /health` y `GET /ready` permanecen públicos para liveness/readiness y no
  llaman al proveedor.
- Cada respuesta incluye un `X-Request-ID` nuevo generado por el servidor.
- Los logs son JSON de una línea y solo contienen campos operativos seguros:
  ruta, estado, duración, categoría de error, estado del agente y modelo del
  proveedor. No registran payloads, respuestas, evidencia, headers, cookies ni
  secretos.
- Se rechazan cuerpos mayores a 64 KiB, texto total mayor a 12,000 caracteres,
  transcripts de más de 32 mensajes y mensajes de más de 32 partes.
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
10. Cloud deployment
11. Documentation and demo

## Filosofía de arquitectura

Se seguirá una estrategia incremental:

`build the smallest reliable layer first`

Cada etapa debe tener una responsabilidad clara antes de introducir la siguiente. Las decisiones sobre proveedor de LLM, modelo, RAG, embeddings, vector database, cloud provider y framework de agentes se tomarán después, cuando existan suficientes requisitos.
