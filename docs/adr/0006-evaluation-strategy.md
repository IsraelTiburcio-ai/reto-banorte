# ADR 0006 — Reproducible evaluation strategy

## Contexto

El agente ya cuenta con retrieval público, un core provider-neutral, un
adaptador HTTP y generación grounded opcional. Antes de optimizar prompts o
respuestas necesitamos medir qué evidencia recupera, cuándo se abstiene y qué
garantías de seguridad son observables.

## Decisión

Phase 7 agrega un dataset auditable en `evals/cases.json`, un runner offline y
reportes deterministas. La suite cubre factuality, groundedness, relevance,
abstention, ownership calibration, skill calibration, métricas aproximadas,
distinción profesional/académica/conceptual, out-of-scope, prompt injection,
información restringida, follow-ups, natural-language retrieval y entradas en
español e inglés.

Cada caso declara input, comportamiento esperado, hechos requeridos, claims
prohibidos, IDs de evidencia esperados, IDs prohibidos y si debe abstenerse.
Las expectativas describen contratos verificables; no guardan respuestas LLM
como ground truth.

## Checks deterministas

El modo offline ejecuta `AgentCore` con `ProfileService` local y verifica:

- status `ready` o `insufficient_evidence` según el caso;
- evidencia no vacía cuando corresponde;
- IDs esperados y ausencia de IDs prohibidos;
- visibilidad `public` en toda la evidencia;
- ausencia de marcadores `internal_summary`, `do_not_expose` y API keys;
- ausencia literal de claims prohibidos dentro de la evidencia serializada.

Esto mide retrieval y la frontera de grounding, no la corrección completa de un
texto generado.

## Tests vs evals

Los unit tests comprueban contratos de implementación y regresiones puntuales.
Las evals organizan preguntas representativas y expectativas de calidad para
observar el comportamiento agregado por categoría. Ambos deben ejecutarse sin
llamadas externas en el flujo normal.

## Offline vs live

Offline es el default y no crea un cliente LLM. Live requiere explícitamente
`--live --confirm-live --limit N`, limita la muestra a cinco casos y usa el
endpoint `/v1/responses`. Las evals live deben planearse manualmente porque
cada caso con evidencia puede producir una llamada al proveedor; los casos
sin evidencia deben conservar el fallback sin llamar al proveedor.

## Costos y secretos

Phase 7 no ejecuta batch live automáticamente y no usa LLM-as-a-judge. El
runner nunca imprime `OPENAI_API_KEY` ni incluye `.env` o credenciales en el
dataset o en los resultados. Antes de cualquier ejecución live se debe
reportar número de casos, modelo y costo aproximado.

## Limitaciones

La evaluación offline no puede juzgar de forma completa fluidez, factualidad de
cada frase generada, ownership wording, skill calibration lingüística o
grounding semántico más allá de la evidencia disponible. Los follow-ups que
dependen de coreference pueden tener recuperación limitada hasta una fase
posterior. No se optimiza el agente con estos resultados todavía.

Una futura estrategia de LLM-as-a-judge queda diferida y requerirá una decisión
separada sobre reproducibilidad, costo, privacidad y calibración. No forma parte
de Phase 7.
