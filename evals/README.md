# Phase 7 — Evals

Esta suite mide contratos observables del CV Agent sin alterar su arquitectura
ni consumir la API real durante la ejecución normal. Una eval de retrieval no
es una eval de respuesta generada: recuperar evidencia pública no demuestra
que una respuesta posterior sea factual, relevante o esté bien redactada.

## Offline / pre-generation

```bash
python3 -m evals.runner
```

El modo offline usa `AgentCore` y `ProfileService` locales. Ejecuta checks
deterministas de status, paquete de evidencia, IDs requeridos, ranking dentro
de `top_k`, abstention, visibilidad pública, marcadores restringidos y
ausencia literal de claims prohibidos dentro de la evidencia serializada.

El dataset usa estas expectativas de ranking:

- `required_evidence_ids`: deben aparecer en cualquier posición;
- `top_evidence_ids` + `top_k`: deben aparecer dentro del prefijo rankeado;
- `forbidden_evidence_ids`: solo se comprueban cuando la lista no está vacía;
  una lista vacía se reporta como `NOT_EVALUATED`, no como PASS.

El reporte separa explícitamente:

- `OFFLINE CASE STATUS`: `PASS`, `FAIL` o `NOT_EVALUATED`;
- `CHECK COVERAGE`: checks ejecutados frente a checks aplicables;
- `LIVE/MANUAL COVERAGE`: expectativas que requieren respuesta generada.

Los `required_facts` no se validan mediante substring ni matching semántico
falso. La factualidad completa, groundedness de la prosa, paráfrasis de claims
prohibidos, ownership, skill calibration, métricas aproximadas, relevancia
semántica e idioma requieren revisión humana o un semantic judge futuro. Por
eso, cuando están declarados, se marcan como `NOT_EVALUATED` offline. Cuando
una expectativa no está declarada —por ejemplo una lista vacía de evidencia
requerida, ranking o evidencia prohibida— el check es `N/A`.

El porcentaje offline, cuando existe, es únicamente de casos completamente
evaluados respecto de sus checks aplicables. `PASS + FAIL` forman los checks
ejecutados; `NOT_EVALUATED` permanece en el denominador aplicable y `N/A` queda
fuera. No es un porcentaje de calidad, factualidad ni groundedness del agente.

El CLI imprime para cada caso sus checks agrupados bajo `PASS`, `FAIL`,
`NOT_EVALUATED` y `N/A`, además del status general del caso.

## Live / post-generation

El modo live nunca se activa por defecto. Requiere confirmación explícita y
limita la muestra a cinco casos:

```bash
python3 -m evals.runner --live --confirm-live --limit 1
```

El runner live puede comprobar HTTP, schema de respuesta, forma de
`assistant/output_text` y marcadores restringidos. Devuelve además campos
`NOT_EVALUATED` y `manual_review` para factualidad, groundedness, ownership,
skills y métricas; no autoaprueba la calidad semántica de una respuesta.
Antes de ejecutar un batch live se debe reportar el número de casos, el modelo
configurado y la estimación de llamadas/costo. Esta implementación no ejecuta
ningún batch live automáticamente.

## Dataset y límites

Los casos auditables están en `evals/cases.json`. Cada expectativa queda
clasificada por su check offline, por revisión live/manual o por una futura
evaluación semántica. No se almacenan respuestas LLM como ground truth.

La limitación de follow-up `¿Y cuál usabas más?` permanece visible como fallo
conocido de retrieval sin coreferencia; no se modifica el agente para mejorar
el score. No se usa LLM-as-a-judge en Phase 7.
