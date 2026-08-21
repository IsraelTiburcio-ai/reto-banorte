# Reto IA Banorte — CV Agent

## Objetivo

Construir un agente conversacional que permita explorar el perfil profesional de un candidato y que posteriormente será integrado mediante una interfaz compatible con Open Responses.

## Estado

Phase 3 — Agent Core

La Phase 3 introduce un `AgentCore` independiente de proveedor. El core valida consultas, fuerza retrieval público, limita la evidencia recuperada y prepara turnos grounded mediante un contrato estructurado que podrá consumir posteriormente un adaptador de LLM.

## Arquitectura inicial

- `api`: interfaz HTTP y endpoints compatibles con Open Responses.
- `agent`: política de comportamiento y orquestación provider-neutral del agente.
- `models`: modelos internos y esquemas de request/response.
- `services`: lógica reutilizable e integraciones externas.
- `core`: configuración, logging, seguridad y manejo de errores.
- `data`: fuente de verdad estructurada del CV.
- `evals`: datasets y scripts para evaluar el agente.
- `tests`: pruebas automatizadas.
- `docs`: arquitectura, decisiones técnicas y diagramas.

La capa de retrieval sigue siendo determinista y `AgentCore` solo prepara contexto público y reglas de comportamiento. Todavía no existe generación con LLM, HTTP API, Open Responses ni deployment.

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
