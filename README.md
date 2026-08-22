# Reto IA Banorte — CV Agent

## Objetivo

Construir un agente conversacional que permita explorar el perfil profesional de un candidato y que posteriormente será integrado mediante una interfaz compatible con Open Responses.

## Estado

Phase 9 — Containerization / Docker

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

El subset soporta texto síncrono, input string, mensajes `user`/`assistant`, partes `input_text` y replay estructural sin estado. Phase 6 genera texto grounded solo cuando existe evidencia; sin evidencia conserva el fallback fijo y evita la llamada al proveedor. `stream` ausente, `null` o `false` devuelve JSON; `stream=true` devuelve SSE con la respuesta textual completa materializada antes de emitirla. Rechaza `system`/`developer`, tools, multimodalidad, persistencia y capacidades conversacionales avanzadas.

## Open Responses subset

```bash
curl -X POST http://localhost:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"banorte-cv-agent","input":"¿Qué experiencia tiene Israel con MCP?"}'
```

El request requiere `input`; `model` es opcional por compatibilidad con la configuración de Parley/Banorte y usa `banorte-cv-agent` cuando está ausente o es `null`. Un string no vacío se preserva y un valor vacío o whitespace-only se rechaza. `stream` es `false` por defecto: `null` y `false` conservan JSON, mientras `true` usa `text/event-stream; charset=utf-8`, `Cache-Control: no-store` y la secuencia SSE de ciclo de vida de Open Responses. El endpoint no mantiene conversaciones: acepta `store` ausente, `null` o `false` como formas stateless, pero rechaza `store=true`; también rechaza `previous_response_id`, `background`, `compaction`, tools o visibilidad seleccionable por el cliente.

## Phase 6 — LLM integration

Configura el proveedor únicamente mediante variables de entorno:

```bash
export OPENAI_API_KEY='...'
export OPENAI_MODEL='gpt-5.6-luna'  # opcional; este es el default
```

`model` en la request es el identificador lógico externo y no selecciona el modelo del proveedor. `OPENAI_MODEL` controla el modelo OpenAI. Nunca publiques API keys en el repositorio, README, logs o tests.

El generador usa la Responses API oficial con `store=false`, sin tools, token streaming, multimodalidad, memoria persistente ni fallback automático a otro proveedor. El SSE del endpoint es streaming de transporte de una respuesta ya completa; no activa streaming del provider. El modelo lógico recibe texto grounded únicamente con evidencia pública aprobada por `AgentCore`.

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
- Cuando está configurada, la autenticación ocurre antes de que FastAPI lea o
  valide el body; payloads inválidos sin credenciales reciben 401.
- `GET /health` y `GET /ready` permanecen públicos para liveness/readiness y no
  llaman al proveedor.
- `/ready` valida la policy, la estructura del adapter y ejecuta un probe local
  de `AgentCore -> ProfileService`; devuelve `503 {"status":"not_ready"}` si
  un componente esencial no es usable. No verifica ni llama a OpenAI.
- Cada respuesta incluye un `X-Request-ID` nuevo generado por el servidor.
- Los logs son JSON de una línea y solo contienen campos operativos seguros:
  ruta, estado, duración, categoría de error, estado del agente y modelo del
  proveedor. No registran payloads, respuestas, evidencia, headers, cookies ni
  secretos.
- El límite real del body es 64 KiB (65,536 bytes) en los dos POST protegidos,
  incluso sin `Content-Length` o cuando llegan varios chunks. Además se
  rechaza texto total mayor a 12,000 caracteres, transcripts de más de 32
  mensajes y mensajes de más de 32 partes.
- El límite de 64 KiB no es global: las rutas públicas conservan el `receive`
  ASGI original y no consumen ni reconstruyen su body mediante este middleware.
- `provider_invoked` solo es `true` cuando el generador confirma un intento
  outbound al proveedor; `input_chars` es el total de texto aceptado del
  request/transcript y nunca contiene el texto.
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

## Phase 9 — Containerization / Docker

La imagen se construye desde el `pyproject.toml` canónico, incluye únicamente
el runtime de la aplicación y `data/profile.json`, y ejecuta Uvicorn como un
usuario no root. El contenedor escucha en `0.0.0.0`, respeta `PORT` y conserva
los logs estructurados en stdout.

### Validación estática

Los tests de contrato parsean las instrucciones efectivas del Dockerfile y
evalúan las reglas relevantes de `.dockerignore`, incluyendo continuaciones,
orden de `USER`, semántica de `CMD`, declaraciones `ENV`/`ARG`, destinos
`COPY` y negaciones last-match-wins. También contienen regresiones contra
mutaciones conocidas del review. La suite normal no requiere Docker.

Build local:

```bash
docker build -t reto-banorte-cv-agent:local .
```

Run sin proveedor:

```bash
docker run --rm -p 8080:8080 reto-banorte-cv-agent:local
```

Configuración de runtime con placeholders — nunca uses claves reales en el
Dockerfile, argumentos de build, README o imagen:

```bash
docker run --rm \
  -p 8080:8080 \
  -e AGENT_API_KEY="<agent-api-key>" \
  -e OPENAI_API_KEY="<openai-api-key>" \
  -e OPENAI_MODEL="gpt-5.6-luna" \
  reto-banorte-cv-agent:local
```

`AGENT_API_KEY` protege nuestros endpoints privados; `OPENAI_API_KEY` es la
credencial outbound del proveedor. Ambas se inyectan únicamente al ejecutar
el contenedor. `.env` está excluido del contexto y no se copia a la imagen.

Endpoints para smoke tests:

- `GET /health` — liveness, no requiere proveedor.
- `GET /ready` — readiness local, no llama a OpenAI.
- `POST /v1/responses` — contrato interoperable; requiere auth si
  `AGENT_API_KEY` está configurada.

Phase 9 solo prepara la imagen. Cloud Run, GCP, Artifact Registry, Secret
Manager, IAM, Compose, Kubernetes y deployment permanecen fuera de alcance.

### Validación local del contenedor

La siguiente validación se ejecutó localmente en macOS Apple Silicon `arm64`
con Docker `29.7.2`; no es una validación de Cloud Run:

- `docker build` pasó para `reto-banorte-cv-agent:phase9`.
- Tamaño de imagen: `56,021,395` bytes (aproximadamente `56 MB`).
- Usuario efectivo: `uid=999(app) gid=999(app) groups=999(app)`.
- Con el puerto default `8080`, `/health` y `/ready` devolvieron `200`.
- Con `PORT=9090`, `/health` devolvió `200`.
- Con una `AGENT_API_KEY` fake: requests sin autorización y con Bearer
  incorrecto devolvieron `401`; Bearer correcto permitió `/agent/prepare` con
  retrieval público MCP válido.
- `/v1/responses` autenticado con una pregunta sin evidencia pública devolvió
  `200`, respuesta Open Responses válida y abstención segura sin requerir
  `OPENAI_API_KEY` ni invocar al proveedor.
- `docker stop -t 10` terminó limpiamente en aproximadamente `0.387 s`.
- La ejecución `--read-only` mantuvo `/health` y `/ready` en `200`.
- `/app/.env` estuvo ausente; `Config.Env` no incluyó `OPENAI_API_KEY` ni
  `AGENT_API_KEY`; `docker history --no-trunc` no mostró credenciales de la
  aplicación.

La compatibilidad con Cloud Run continúa siendo una decisión de diseño
preparatoria. El despliegue real y sus recursos pertenecen a Phase 10 y aún no
se han iniciado.

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
