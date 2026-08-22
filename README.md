# Reto IA Banorte — CV Agent

## Objetivo

Construir un agente conversacional que permita explorar el perfil profesional de un candidato y que posteriormente será integrado mediante una interfaz compatible con Open Responses.

## Estado

Phase 7 — Evals

La Phase 6 reemplaza el formateador determinista temporal por generación grounded mediante el SDK oficial de OpenAI. El LLM solo recibe la evidencia pública ya preparada por `AgentCore`; no hace retrieval ni lee el perfil canónico.

Endpoints actuales:

- `GET /health`
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
generadas como verdad absoluta y no imprime secretos.

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
