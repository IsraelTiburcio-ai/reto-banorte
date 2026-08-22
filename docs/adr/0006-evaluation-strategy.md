# ADR 0006 — Reproducible evaluation strategy

## Contexto

El agente ya cuenta con retrieval público, un core provider-neutral, un
adaptador HTTP y generación grounded opcional. Phase 7 necesita medir qué
evidencia recupera y cuándo se abstiene sin vender esa señal como calidad de
una respuesta generada.

Una eval de retrieval no es una eval de respuesta generada. El modo offline es
pre-generation; no produce texto del proveedor y no puede demostrar
factualidad semántica, groundedness de la prosa o una paráfrasis segura.

## Decisión

Phase 7 agrega `evals/cases.json`, un runner offline, una ruta live explícita y
reportes con estados `PASS`, `FAIL` y `NOT_EVALUATED`. La suite cubre
factuality, groundedness, relevance, abstention, ownership calibration, skill
calibration, métricas aproximadas, distinción profesional/académica/conceptual,
out-of-scope, prompt injection, información restringida, follow-ups,
natural-language retrieval y entradas en español e inglés.

Cada caso declara input, comportamiento esperado, `required_facts`,
`forbidden_claims`, `required_evidence_ids`, expectativas opcionales de ranking
(`top_evidence_ids` y `top_k`), `forbidden_evidence_ids` y si debe abstenerse.
Estas declaraciones no son respuestas LLM almacenadas como ground truth.

## Checks offline deterministas

El modo offline ejecuta `AgentCore` con `ProfileService` local y verifica:

- status `ready` o `insufficient_evidence` según el caso;
- evidencia no vacía cuando corresponde;
- `required_evidence_ids` en cualquier posición;
- `top_evidence_ids` dentro del prefijo rankeado `top_k`;
- ausencia de `forbidden_evidence_ids` cuando existe una expectativa real;
- visibilidad `public` en toda la evidencia;
- ausencia de marcadores `internal_summary`, `do_not_expose` y API keys;
- señal literal de claims prohibidos dentro de la evidencia serializada.

Una lista vacía de evidencia prohibida es `NOT_EVALUATED`, no PASS. Las
expectativas de ranking no exigen que toda evidencia adicional sea idéntica a
una lista cerrada.

`required_facts` no se evalúa mediante substring ni matching semántico falso.
Los checks `required_facts_semantics`, `forbidden_claims_semantics` y
`generated_answer_semantics` permanecen `NOT_EVALUATED` offline. Por tanto un
hecho imposible añadido por mutación no puede producir un PASS que afirme
haberlo validado.

## Estados y cobertura

El reporte separa:

- `OFFLINE CASE STATUS`: estado del caso completo; si contiene expectativas
  semánticas no ejecutables, queda `NOT_EVALUATED`, salvo que falle un check
  determinista;
- `CHECK COVERAGE`: checks `PASS`/`FAIL` ejecutados frente a checks declarados;
- `LIVE/MANUAL COVERAGE`: expectativas diferidas.

El pass rate solo usa casos completamente evaluados y se etiqueta como
`OFFLINE EXECUTABLE PASS RATE`. No se presenta como factuality, groundedness ni
calidad general. La salida por categoría muestra también cuántos checks fueron
ejecutados y cuántos quedaron `NOT_EVALUATED`.

## Offline, live y revisión humana

Offline es el default y no crea un cliente LLM. Live requiere explícitamente
`--live --confirm-live --limit N`, limita la muestra a cinco casos y usa
`/v1/responses`. El runner live comprueba solo señales deterministas como HTTP,
schema, forma `assistant/output_text` y marcadores restringidos; devuelve
`NOT_EVALUATED` y una lista `manual_review` para semántica.

Requieren revisión humana o un semantic judge futuro: factualidad completa,
groundedness de la prosa, paráfrasis de claims prohibidos, ownership wording,
skill calibration, métricas aproximadas, relevancia semántica y fluidez o
idioma. Los literales/patrones deterministas no garantizan equivalencia
semántica.

## Tests vs evals

Los unit tests comprueban contratos de implementación y mutaciones del runner.
Las evals organizan preguntas representativas y expectativas observables por
categoría. Ambos deben ejecutarse sin llamadas externas en el flujo normal.

## Costos y secretos

Phase 7 no ejecuta batch live automáticamente y no usa LLM-as-a-judge. El
runner nunca imprime `OPENAI_API_KEY` ni incluye `.env`, tokens o credenciales
en el dataset, errores o resultados. Antes de cualquier ejecución live se debe
reportar número de casos, modelo y costo aproximado.

## Limitaciones

El follow-up `¿Y cuál usabas más?` permanece como fallo conocido: sin reglas de
coreferencia el retrieval puede devolver evidencia incorrecta. No se modifica
AgentCore para subir el score. Preguntas dependientes de contexto requieren
una fase posterior.

Una estrategia de LLM-as-a-judge queda diferida y requerirá una decisión
separada sobre reproducibilidad, costo, privacidad y calibración. No forma
parte de Phase 7.
