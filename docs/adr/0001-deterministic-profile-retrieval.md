# ADR 0001: Deterministic profile retrieval

## Context

El corpus profesional inicial es pequeño, estructurado y controlado. Antes de introducir comportamiento probabilístico, necesitamos comprobar que el conocimiento puede cargarse y consultarse de forma predecible.

## Decision

Utilizar retrieval determinista basado en la estructura de `data/profile.json` y búsqueda lexical normalizada con Python estándar.

## Reasons

- comportamiento predecible;
- facilidad de testing;
- menor complejidad;
- cero costo externo;
- permite evaluar posteriormente si embeddings realmente aportan valor;
- evita introducir infraestructura antes de necesitarla.

## Consequences

### Positivas

- simple;
- rápido;
- explicable;
- confiable.

### Limitaciones

- menor capacidad semántica;
- las búsquedas conceptuales pueden requerir mayor contexto;
- la capa podrá evolucionar posteriormente si los requisitos lo justifican.
