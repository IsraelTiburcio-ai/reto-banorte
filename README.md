# Reto IA Banorte — CV Agent

## Objetivo

Construir un agente conversacional que permita explorar el perfil profesional de un candidato y que posteriormente será integrado mediante una interfaz compatible con Open Responses.

## Estado

Phase 4 — HTTP API

La Phase 4 expone el `AgentCore` mediante una API HTTP delgada en FastAPI. La capa HTTP valida el contrato de entrada, serializa `PreparedAgentTurn` a JSON y traduce errores de input a respuestas HTTP sin mover reglas de retrieval, visibilidad o grounding fuera de sus capas existentes.

Endpoints actuales:

- `GET /health`
- `POST /agent/prepare`

`POST /agent/prepare` es un contrato interno del proyecto y **todavía no es Open Responses**. La compatibilidad con Open Responses corresponde a Phase 5.

## Arquitectura inicial

- `api`: interfaz HTTP delgada, schemas y serialización.
- `agent`: política de comportamiento y orquestación provider-neutral del agente.
- `models`: modelos internos y contratos provider-neutral.
- `services`: lógica reutilizable e integraciones externas.
- `core`: configuración, logging, seguridad y manejo de errores.
- `data`: fuente de verdad estructurada del CV.
- `evals`: datasets y scripts para evaluar el agente.
- `tests`: pruebas automatizadas.
- `docs`: arquitectura, decisiones técnicas y diagramas.

Flujo actual:

`HTTP -> AgentCore -> ProfileService -> data/profile.json`

`ProfileService` conserva la responsabilidad de enforcement de visibilidad y `AgentCore` continúa siendo public-only. La API no implementa retrieval ni reglas de exposición propias.

Todavía no existe generación con LLM, compatibilidad Open Responses ni deployment.

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
