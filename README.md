# Reto IA Banorte — CV Agent

## Objetivo

Construir un agente conversacional que permita explorar el perfil profesional de un candidato y que posteriormente será integrado mediante una interfaz compatible con Open Responses.

## Estado

Phase 2 — Profile Retrieval Layer

La Phase 2 está implementada en la rama de trabajo: `data/profile.json` sigue siendo la fuente canónica y `ProfileService` permite cargarla, recuperar entidades por ID y realizar búsqueda lexical determinista.

## Arquitectura inicial

- `api`: interfaz HTTP y endpoints compatibles con Open Responses.
- `agent`: configuración, instrucciones, definiciones de tools y orquestación del agente.
- `models`: modelos internos y esquemas de request/response.
- `services`: lógica reutilizable e integraciones externas.
- `core`: configuración, logging, seguridad y manejo de errores.
- `data`: fuente de verdad estructurada del CV.
- `evals`: datasets y scripts para evaluar el agente.
- `tests`: pruebas automatizadas.
- `docs`: arquitectura, decisiones técnicas y diagramas.

La capa de retrieval aplica reglas de visibilidad y no modifica la knowledge base en memoria. Todavía no existe LLM, agente, API, Open Responses ni deployment.

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
