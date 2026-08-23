# Reto IA Banorte — CV Agent

Agente conversacional público para representar la trayectoria profesional de
Israel Tiburcio Suchil con respuestas útiles, claras y grounded. Puede conversar
sobre experiencia, habilidades, proyectos, decisiones técnicas, forma de trabajo,
aprendizajes y áreas de crecimiento.

El proyecto se construyó de forma incremental: primero una fuente profesional
estructurada y un retrieval determinista; después una frontera de AgentCore,
una API compatible con un subset de Open Responses, generación grounded,
evaluaciones, seguridad, observabilidad, containerización y compatibilidad con
replay stateless de Parley.

## Demo / endpoint

Servicio público:

https://cv-agent-540272372336.us-central1.run.app

Base de la API interoperable:

https://cv-agent-540272372336.us-central1.run.app/v1

Endpoint Open Responses:

POST /v1/responses

Modelo lógico:

banorte-cv-agent

Health:

GET /health

Readiness:

GET /ready

Los endpoints POST requieren Bearer auth cuando AGENT_API_KEY está configurada
en el servicio. La credencial se envía como:

    Authorization: Bearer <AGENT_API_KEY>

No se publica ninguna API key, token o valor de Secret Manager en este
repositorio. OPENAI_API_KEY es únicamente una credencial de salida hacia el
provider y AGENT_API_KEY protege la API del agente; son secretos distintos.

## Arquitectura

El flujo principal es:

~~~mermaid
flowchart TD
    C[Cliente / Parley] --> O[Open Responses API]
    O --> A[AgentCore]
    A --> P[ProfileService]
    P --> E[Perfil público y evidencia dirigida]
    E --> G[LLM provider]
    G --> R[Respuesta grounded]
~~~

Las responsabilidades están separadas:

- La API valida el transporte, extrae la pregunta actual y conserva el
  transcript como contexto estructural.
- AgentCore llama a ProfileService con visibility pública y limita la evidencia
  que llega al generador.
- ProfileService carga el perfil canónico, aplica visibility, filtra contenido
  nested y realiza búsqueda determinista.
- El contrato TextGenerator es provider-neutral. La implementación actual usa
  la Responses API oficial de OpenAI.
- El provider recibe el contexto público y la evidencia ya preparada; no lee
  data/profile.json ni realiza retrieval por su cuenta.

La API pública no duplica reglas de retrieval, grounding o visibilidad.

## 1. Diseñar

### Perfil estructurado como fuente canónica

data/profile.json contiene el conocimiento profesional preparado para el agente.
Incluye identidad, resumen profesional, experiencia, proyectos, educación,
formación, skills calibradas, áreas de conocimiento, working style, decisiones,
colaboración, aprendizajes, fortalezas, crecimiento y narrativas profesionales.

El perfil utiliza tres niveles de visibilidad:

- public: puede utilizarse como conocimiento factual sobre Israel;
- internal_summary: se reserva para una política interna explícita;
- do_not_expose: nunca se entrega.

El flujo público solo permite public. La política también preserva niveles de
habilidad, ownership, métricas aproximadas y la diferencia entre experiencia
práctica, académica, histórica, conceptual y autoevaluada.

### Retrieval sin convertirlo en gate

El retrieval selecciona evidencia especialmente relevante para la pregunta
actual. No decide si el agente tiene permitido conversar.

El principio es:

> El retrieval determina qué evidencia específica está disponible sobre Israel,
> no si el agente tiene permitido hablar.

Por eso, un retrieval dirigido vacío no bloquea automáticamente la generación:
el provider aún recibe el contexto público base y puede responder una pregunta
general, una interacción social o una pregunta fuera de alcance de forma
natural, sin inventar hechos sobre Israel. Cuando sí existe evidencia dirigida,
se añade al mismo contexto público y se conserva su ranking determinista.

### Conocimiento general frente a hechos sobre Israel

El modelo puede explicar conocimiento general. Por ejemplo, puede explicar la
diferencia entre RAG y fine-tuning sin atribuir esa explicación a la experiencia
de Israel.

En cambio, cualquier afirmación sobre Israel —experiencia, proyectos, skills,
educación, métricas u ownership— debe estar respaldada por el contexto público
de la aplicación o por evidencia pública recuperada. El transcript no se
convierte automáticamente en evidencia factual.

### Por qué no utilicé una base vectorial

El corpus actual es pequeño, estructurado y mantenido explícitamente. Preferí
búsqueda exacta, substring y token matching determinista porque es fácil de
inspeccionar, probar y proteger con visibility.

No considero que embeddings o una vector database sean una mala solución. Serían
una evolución razonable si el corpus creciera significativamente, aparecieran
muchos documentos o el retrieval lexical dejara de ser suficiente. Hoy añadirían
complejidad operacional sin resolver una necesidad actual.

### Provider-neutral

El contrato del agente no depende del modelo externo. TextGenerator define una
interfaz pequeña para generar una respuesta a partir de una solicitud ya
grounded. Cambiar el modelo o implementar otro provider no requiere cambiar el
contrato Open Responses ni mover retrieval al provider.

### Conversaciones stateless

El backend no almacena memoria conversacional persistente. El cliente reenvía el
transcript en cada request y el adaptador conserva una ventana reciente para
generación.

Elegí este diseño para mantener un comportamiento predecible, reducir estado
en el servidor, limitar exposición de datos y ser compatible con replay de
transcripciones de Parley. El transcript aporta contexto conversacional, no
autoridad factual.

## 2. Integrar

La solución integra:

- un perfil profesional estructurado;
- retrieval determinista y public-only;
- AgentCore;
- el subset textual de Open Responses;
- un provider LLM intercambiable;
- la implementación OpenAI actual;
- replay stateless de Parley/Banorte;
- autenticación, límites y observabilidad;
- evaluaciones reproducibles;
- un contenedor preparado para Cloud Run.

Evidence y transcript tienen responsabilidades distintas:

- EVIDENCE: hechos públicos preparados sobre Israel para el turno;
- TRANSCRIPT: contexto de la conversación.

Una afirmación del usuario o una respuesta anterior del assistant no se convierte
automáticamente en un hecho profesional. El prompt del provider etiqueta ambos
como datos de referencia y mantiene las instrucciones de la aplicación fuera
del contenido recuperado.

## 3. Desplegar

La instancia pública del reto se ejecuta con un flujo de containerización y
servicios administrados:

source → Docker build → Google Cloud Build → Artifact Registry → Cloud Run

El stack de despliegue contempla:

- Docker para empaquetar la aplicación;
- Google Cloud Build para construir la imagen;
- Artifact Registry para almacenarla;
- Cloud Run en us-central1 para ejecutar el servicio;
- Secret Manager para credenciales;
- una service account dedicada para reducir permisos.

Cloud Run aporta escalado serverless. /health y /ready permiten distinguir
liveness de readiness sin llamar al provider.

El contenedor escucha en 0.0.0.0, respeta PORT, ejecuta Uvicorn como usuario
no root e incluye únicamente el runtime y el perfil necesario. Las credenciales
se inyectan en runtime; no se incluyen en Dockerfile, argumentos de build,
README ni imagen.

## 4. Operar

### Seguridad

- Bearer authentication opcional mediante AGENT_API_KEY.
- OPENAI_API_KEY separada de la credencial del agente.
- Secretos fuera del repositorio y administrados por el entorno de despliegue.
- service account dedicada.
- filtering public-only antes de scoring y generación.
- contenido recuperado tratado como datos, nunca como instrucciones.
- transcript separado de evidence.
- store=false hacia el provider OpenAI.
- sin persistencia conversacional, tools, memoria del servidor ni llamadas
  externas adicionales.
- logs sin payloads, respuestas, evidencia, headers, cookies o secretos.

### Límites operativos

Los límites finales se definen en app/core/limits.py:

| Límite | Valor |
| --- | ---: |
| body HTTP protegido | 512 KiB (524,288 bytes) |
| texto simple o mensaje | 12,000 caracteres |
| mensajes de transcript recibidos | 256 |
| partes de contenido por mensaje | 32 |
| history enviado al generador | 8 mensajes |
| caracteres de history enviados al generador | 8,000 |

El sistema puede aceptar un transcript largo, pero eso no significa que todo el
transcript se envíe al modelo. Antes de generación se conserva únicamente una
ventana reciente de hasta 8 mensajes y 8,000 caracteres de historial; la
pregunta actual se conserva completa y se envía separada.

Estos límites permiten aproximadamente 120 pares pregunta/respuesta dentro de
un request, siempre que el body completo permanezca dentro de 512 KiB. El límite
de partes de contenido es independiente: 32 partes aplican dentro de un solo
mensaje, no al número total de preguntas.

### Observabilidad

Cada request recibe un X-Request-ID generado por el servidor. Los logs
estructurados registran únicamente campos operativos como ruta, status, duración,
categoría de error, estado del agente, provider model y si se intentó una
llamada outbound.

provider_invoked solo es true cuando el generador confirma un intento hacia el
provider. input_chars es un conteo de caracteres aceptados; nunca contiene el
payload.

/ready valida localmente la policy, AgentCore, ProfileService y el adapter. No
llama a OpenAI ni requiere OPENAI_API_KEY.

## 5. Verificar

### Tests automatizados

La suite actual contiene 168 tests y cubre:

- AgentCore y límites de evidencia;
- ProfileService, ranking, deep copies y visibility;
- grounding y ausencia de contenido restringido;
- autenticación y errores sanitizados;
- Open Responses JSON y envelope de errores;
- SSE y coherencia de IDs, índices y eventos;
- replay de transcript user/assistant;
- límites de body, mensajes, texto y content parts;
- store, metadata compatible con Parley y campos no soportados;
- fallos del provider;
- health y readiness;
- Dockerfile, .dockerignore y mutaciones del contrato de contenedor;
- follow-ups y representación profesional;
- integración de contexto público enriquecido.

Validaciones locales ejecutadas:

~~~bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q app evals tests
python3 -m json.tool data/profile.json > /dev/null
git diff --check
~~~

### Evals offline

La suite reproducible contiene 21 casos y no consume la API real:

~~~bash
python3 -m evals.runner
~~~

Resultado actual:

- checks PASS: 138;
- checks FAIL: 0;
- checks NOT_EVALUATED: 56;
- checks N/A: 37;
- checks ejecutados: 138;
- checks aplicables: 194;
- coverage: 71.1%.

El runner separa contrato/retrieval de semántica generada. Las checks
NOT_EVALUATED requieren una respuesta del provider y revisión humana; no se usa
LLM-as-a-judge. N/A significa que no existe una expectativa para ese caso. El
coverage no es una métrica de calidad general ni de factualidad.

### Pre-Banorte smoke

evals/pre_banorte_cases.json contiene 30 casos auditables. El smoke offline
valida el contrato sin hacer llamadas HTTP ni provider:

~~~bash
python3 scripts/pre_banorte_smoke.py
~~~

También existe una ejecución productiva local de 30 turnos estilo Parley para
revisar conversación natural, follow-ups, preguntas ambiguas, calibración,
conocimiento general, tecnologías no dominadas y conversaciones largas:

~~~bash
python3 scripts/conversational_ux_product.py
~~~

El modo live es opt-in y no reintenta automáticamente:

~~~bash
python3 scripts/conversational_ux_product.py --live
~~~

Las respuestas generadas se revisan semánticamente; los marcadores REVIEW no
son un PASS automático.

### Pruebas live

Se ejecutaron sesiones live controladas de 30 turnos para observar conversación
natural, follow-ups, preguntas ambiguas, ownership, skill calibration,
conocimiento general y preguntas fuera de alcance. Los logs no se incorporan al
README ni contienen credenciales.

### Mutaciones dirigidas

Durante la validación se probaron mutaciones deliberadas sobre guardrails
críticos para comprobar que las regresiones fueran detectables, incluyendo
límites de history, aceptación de metadata legítima de Parley y el contrato del
contenedor. Esto complementa la suite normal; no se presenta como una suite
genérica de mutation testing.

## Contexto profesional

El perfil comenzó enfocado en hechos, proyectos y habilidades. Después se
enriqueció para representar también:

- forma de trabajar;
- toma de decisiones y trade-offs;
- colaboración y ownership;
- validación y pruebas;
- aprendizajes;
- fortalezas;
- áreas de crecimiento;
- narrativas profesionales.

El enriquecimiento está en la knowledge base pública, no en respuestas
hardcodeadas. El modelo puede sintetizar distintas preguntas a partir del mismo
contexto, manteniendo calibración y límites de evidencia.

## Compatibilidad Open Responses

La API implementa un subset síncrono, textual y stateless:

- input como string;
- mensajes user y assistant;
- partes input_text;
- metadata como mapa de strings compatible;
- respuesta JSON;
- SSE materializado para stream=true;
- store ausente, null o false;
- replay estructural de transcript.

El modelo lógico externo es banorte-cv-agent. El campo model puede omitirse o ser
null; en ese caso se usa ese identificador local. Un model vacío se rechaza.

El SSE materializa primero la respuesta completa y después emite la secuencia de
eventos compatible. No activa token streaming del provider.

No se afirma full conformance con Open Responses. Persistencia stateful,
previous_response_id, tools, multimodalidad, system/developer, background,
compaction y capacidades avanzadas quedan fuera del subset actual.

## Ejecutar localmente

Requisitos:

- Python 3.11 o posterior;
- dependencias definidas en pyproject.toml;
- Docker solo para probar el contenedor.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
~~~

Configura variables por nombre en el entorno local cuando correspondan:

~~~bash
export OPENAI_API_KEY="<configurar-localmente>"
export OPENAI_MODEL="gpt-5.6-luna"
export AGENT_API_KEY="<configurar-si-se-requiere-auth>"
~~~

No publiques los valores de esas variables ni los agregues a tests, README,
Dockerfile o Git. .env está ignorado y .env.example solo contiene placeholders.

Arranque local:

~~~bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
~~~

Comprobaciones:

~~~bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready

curl -X POST http://127.0.0.1:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"banorte-cv-agent","input":"¿Qué experiencia tiene Israel con MCP?"}'
~~~

Si AGENT_API_KEY está configurada, agrega:

~~~bash
-H 'Authorization: Bearer <AGENT_API_KEY>'
~~~

Suite local:

~~~bash
python3 -m unittest discover -s tests -v
python3 -m evals.runner
~~~

## Roadmap

Las fases completadas cubren:

1. foundation;
2. knowledge base;
3. retrieval;
4. AgentCore;
5. HTTP API y Open Responses;
6. grounded LLM generation;
7. evals;
8. security and observability;
9. containerization;
10. Parley compatibility, transcript replay y professional UX.

El siguiente paso de producto es operar y evolucionar el despliegue público con
controles de plataforma apropiados. La implementación de nuevas capacidades
debe preservar la frontera public-only y el comportamiento stateless.

## Filosofía

Elegí construir la capa confiable más pequeña antes de añadir complejidad:
perfil estructurado, retrieval determinista, AgentCore, API, provider grounded,
evaluaciones y controles operativos.

La simplicidad aquí no es una renuncia a evolucionar. Es una forma de mantener
las decisiones auditables: cada afirmación profesional tiene una fuente pública,
cada límite tiene un contrato probado y cada integración puede cambiar sin mover
la frontera de seguridad.
