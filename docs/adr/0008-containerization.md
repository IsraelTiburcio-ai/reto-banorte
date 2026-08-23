# ADR 0008: Containerización segura del CV Agent

- Estado: Propuesto
- Fecha: 2026-08-22
- Fase: 9 — Containerization / Docker

## Contexto

El servicio debe poder empaquetarse de forma reproducible para una futura
ejecución en Cloud Run. Esta decisión cubre únicamente la imagen Linux y su
arranque local; no cubre despliegue, IAM, Artifact Registry ni recursos de GCP.

## Decisión

Se agrega un `Dockerfile` de una sola etapa basado en
`python:3.11-slim-bookworm`. La imagen instala el proyecto desde el
`pyproject.toml` canónico con `pip --no-cache-dir --target /app`, sin duplicar
versiones en `requirements.txt` y sin instalar los extras de test/evals. La
instalación se ejecuta sobre un árbol temporal que contiene únicamente
`pyproject.toml` y `app/`; el repositorio completo no es instalable directamente
con `pip install .` porque el descubrimiento flat-layout de setuptools también
detecta `data/` y `evals/` como paquetes top-level. `data/profile.json` se copia
por separado como recurso runtime, sin cambiar el packaging del proyecto.

El código se instala bajo `/app/app` y el perfil se copia explícitamente a
`/app/data/profile.json`. Esto conserva la resolución existente de
`ProfileService`, que calcula `data/profile.json` relativa al paquete local, y
evita modificar retrieval o visibility para funcionar dentro del contenedor.

No se usa multi-stage: el proyecto es Python puro y las wheels runtime se
instalan directamente; una segunda etapa no aporta una reducción necesaria en
esta fase y aumentaría la complejidad de mantenimiento.

## Usuario y runtime

El proceso se ejecuta como el usuario/grupo no privilegiado `app:app`. El
contenedor expone documentalmente el puerto `8080`, pero Uvicorn escucha en
`0.0.0.0` y toma `PORT`, con default `8080`:

```text
exec python -m uvicorn app.api.main:app --host 0.0.0.0 --port "${PORT:-8080}"
```

El uso de `exec` deja a Uvicorn como proceso principal y permite que SIGTERM
llegue directamente al servidor. No se usa `--reload` ni se agregan workers
adicionales.

`PYTHONDONTWRITEBYTECODE=1` y `PYTHONUNBUFFERED=1` son las únicas opciones de
runtime configuradas. Los logs JSON existentes continúan saliendo a stdout.

## Archivos y secretos

El contexto se restringe con `.dockerignore`: excluye `.git`, entornos
virtuales, caches, artefactos de build, tests/evals/docs y archivos `.env`.
`data/profile.json` no se excluye. El Dockerfile usa `COPY` explícitos y nunca
copia el contexto completo.

No se declaran `ARG` ni `ENV` para `OPENAI_API_KEY` o `AGENT_API_KEY`, no se
copia `.env` y no se hornean credenciales en capas, labels ni archivos. Las
claves, si se necesitan en una ejecución posterior, deben inyectarse solamente
como configuración de runtime. La imagen de Phase 9 se valida sin proveedor y
sin claves.

## Health, readiness y filesystem

`/health` y `/ready` siguen siendo endpoints públicos. Ambos pueden validarse
sin `OPENAI_API_KEY`: `/ready` comprueba la policy, el adapter y el probe local
`AgentCore -> ProfileService`, pero no verifica disponibilidad de OpenAI ni
llama al proveedor. No se agrega un `HEALTHCHECK` que obligue a instalar
`curl`/`wget`; la plataforma podrá configurar probes posteriormente.

La aplicación solo necesita leer código y `data/profile.json`; el usuario
runtime no requiere permisos de escritura sobre ellos. La ejecución local con
`--read-only` mantuvo `/health` y `/ready` en `200`; `/tmp` queda disponible
para montarse como tmpfs si un entorno futuro lo requiere.

## Validación local posterior

La validación siguiente se ejecutó localmente en macOS Apple Silicon `arm64`
con Docker `29.7.2`. Son smoke tests del contenedor local, no pruebas de Cloud
Run:

- `docker build` pasó para `reto-banorte-cv-agent:phase9`; tamaño:
  `56,021,395` bytes (aproximadamente `56 MB`).
- El usuario efectivo fue `uid=999(app) gid=999(app) groups=999(app)`.
- Con `PORT=8080`, `/health` y `/ready` devolvieron `200`; con `PORT=9090`,
  `/health` devolvió `200`.
- Con una `AGENT_API_KEY` fake, ausencia de autorización y Bearer incorrecto
  devolvieron `401`; Bearer correcto permitió `/agent/prepare` y retrieval MCP
  público válido.
- Una request autenticada a `/v1/responses` sin evidencia pública devolvió
  `200` con respuesta Open Responses válida y abstención segura, sin requerir
  `OPENAI_API_KEY` ni llamar al proveedor.
- `docker stop -t 10` terminó limpiamente en aproximadamente `0.387 s`.
- La ejecución `--read-only` conservó `/health` y `/ready` en `200`.
- `/app/.env` estuvo ausente; `Config.Env` no incluyó `OPENAI_API_KEY` ni
  `AGENT_API_KEY`; `docker history --no-trunc` no mostró credenciales de la
  aplicación.

## Alcance y compatibilidad futura

La imagen es stateless, escribe logs a stdout/stderr, respeta `PORT` y maneja
señales mediante el proceso Uvicorn, por lo que es compatible conceptualmente
con Cloud Run; esa compatibilidad no equivale a un despliegue probado. No se
agrega Docker Compose porque existe un solo servicio sin dependencias locales
obligatorias. No se implementan Cloud Run, GCP, Artifact
Registry, Cloud Build, Secret Manager, IAM, Terraform, Kubernetes, CI/CD ni
balanceo; esos temas pertenecen a fases posteriores.

## Limitaciones conocidas

- El tag de imagen es responsabilidad del flujo de build; no se publica una
  imagen desde esta fase.
- No hay healthcheck Docker embebido; `/health` y `/ready` son los contratos
  disponibles para la plataforma.
- El proceso usa un solo worker, sin optimización prematura de concurrencia.
- Los tests Python estáticos y los smoke tests Docker locales son validaciones
  separadas; ninguno reemplaza una futura validación y despliegue cloud.
