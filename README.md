# Reto IA Banorte — CV Agent

## Objetivo

Construir un agente conversacional que permita explorar el perfil profesional de un candidato y que posteriormente será integrado mediante una interfaz compatible con Open Responses.

## Estado

Phase 5 — Open Responses compatibility

La Phase 5 agrega un adaptador síncrono y textual para un subset estricto de Open Responses. El adaptador valida el transporte, extrae el último mensaje `user`, llama a `AgentCore` y serializa una respuesta determinista. No hay LLM todavía.

Endpoints actuales:

- `GET /health`
- `POST /agent/prepare`
- `POST /v1/responses`

`POST /agent/prepare` es un contrato interno. `POST /v1/responses` es un contrato interoperable parcial: Phase 5 no afirma full conformance con Open Responses.

## Arquitectura inicial

- `api`: interfaz HTTP delgada, schemas, adaptador Open Responses y serialización.
- `agent`: política de comportamiento y orquestación provider-neutral del agente.
- `models`: modelos internos y contratos provider-neutral.
- `services`: lógica reutilizable e integraciones externas.
- `core`: configuración, logging, seguridad y manejo de errores.
- `data`: fuente de verdad estructurada del CV.
- `evals`: datasets y scripts para evaluar el agente.
- `tests`: pruebas automatizadas.
- `docs`: arquitectura, decisiones técnicas y diagramas.

Flujo actual:

`POST /v1/responses -> Open Responses adapter -> AgentCore -> ProfileService -> data/profile.json`

`ProfileService` conserva la responsabilidad de enforcement de visibilidad y `AgentCore` continúa siendo public-only. La API no implementa retrieval ni reglas de exposición propias.

El subset soporta texto síncrono, input string, mensajes `user`/`assistant`, partes `input_text`, replay estructural sin estado y un formateador determinista temporal. Rechaza `system`/`developer`, streaming, tools, multimodalidad, persistencia y capacidades conversacionales avanzadas.

## Open Responses subset

```bash
curl -X POST http://localhost:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"banorte-cv-agent","input":"¿Qué experiencia tiene Israel con MCP?"}'
```

El request requiere `input`; `model` es opcional por compatibilidad con la configuración de Parley/Banorte y usa `banorte-cv-agent` cuando está ausente o es `null`. Un string no vacío se preserva y un valor vacío o whitespace-only se rechaza. `stream` es `false` por defecto y `metadata` se conserva únicamente como dato de transporte. El endpoint no usa un LLM, no mantiene conversaciones y no acepta `previous_response_id`, `store`, `background`, `compaction`, tools o visibilidad seleccionable por el cliente.

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
