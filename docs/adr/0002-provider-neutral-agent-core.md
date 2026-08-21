# ADR 0002: Provider-neutral agent core before LLM integration

## Context

La Phase 2 ya ofrece retrieval determinista y controlado sobre el perfil profesional. El siguiente paso necesita coordinar una consulta de usuario con evidencia pública y reglas de comportamiento, pero todavía no existe una razón para acoplar la aplicación a un proveedor o modelo específico.

## Decision

Introducir un `AgentCore` provider-neutral que prepare cada turno antes de cualquier generación de lenguaje.

El core:

- valida y normaliza la consulta;
- fuerza retrieval con visibilidad `public`;
- limita la cantidad de evidencia entregada a capas posteriores;
- conserva ranking, procedencia y campos de explicación del retrieval;
- produce un `PreparedAgentTurn` con estado `ready` o `insufficient_evidence`;
- adjunta una política de comportamiento versionada y testeable;
- no llama ningún LLM ni genera una respuesta final.

La información profesional es evidencia. Las reglas de comportamiento del agente son configuración de aplicación y se mantienen en código, fuera del contenido recuperable, para que datos del perfil no puedan convertirse en instrucciones ejecutables.

## Reasons

- mantiene determinista la orquestación previa al modelo;
- crea una frontera explícita entre datos y comportamiento;
- evita que una capa externa eleve la visibilidad del retrieval;
- facilita probar grounding y manejo de evidencia insuficiente antes de introducir un LLM;
- evita acoplamiento temprano a OpenAI, Gemini, Anthropic, Groq u otro proveedor;
- deja una interfaz pequeña para HTTP, Open Responses y el futuro adaptador de modelo.

## Consequences

### Positivas

- comportamiento previo al LLM predecible y testeable;
- frontera pública de datos centralizada;
- política del agente explícita y auditable;
- integración futura de modelos sin reescribir retrieval;
- menor riesgo de mezclar instrucciones con evidencia.

### Limitaciones

- el sistema todavía no produce respuestas conversacionales finales;
- la calidad semántica sigue limitada por el retrieval lexical actual;
- el adaptador de LLM deberá transformar el turno preparado en mensajes o input del proveedor en una fase posterior.
