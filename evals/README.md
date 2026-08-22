# Phase 7 — Evals

Esta suite mide la calidad observable del CV Agent sin alterar su arquitectura
ni consumir la API real durante la ejecución normal.

## Offline

```bash
python3 -m evals.runner
```

El modo offline usa `AgentCore` y `ProfileService` locales. Comprueba estado de
evidencia, IDs esperados, visibilidad pública, ausencia de marcadores
restringidos y claims prohibidos en la evidencia serializada. No genera una
respuesta LLM y no puede medir completamente la calidad de la prosa,
ownership o skill calibration de una respuesta generada.

## Live

El modo live nunca se activa por defecto. Requiere confirmación explícita,
limita la muestra a cinco casos y no imprime respuestas:

```bash
python3 -m evals.runner --live --confirm-live --limit 1
```

Antes de ejecutar un batch live se debe reportar el número de casos, el modelo
configurado y la estimación de llamadas/costo. Esta implementación no ejecuta
ningún batch live automáticamente.

## Dataset

Los casos auditables están en `evals/cases.json`. Sus expectativas son
contratos de evidencia y seguridad, no respuestas generadas guardadas como
verdad absoluta. Las limitaciones de follow-ups y de evaluación lingüística
requieren revisión manual o una futura estrategia separada; no se usa
LLM-as-a-judge en Phase 7.
