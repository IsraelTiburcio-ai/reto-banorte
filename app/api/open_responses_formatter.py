"""Safe fixed text used when retrieval has insufficient public evidence."""

from __future__ import annotations

import re
import unicodedata

INSUFFICIENT_EVIDENCE_TEXT = (
    "No encontré evidencia pública suficiente en el perfil para responder esa pregunta."
)

GREETING_RESPONSE_TEXT = (
    "¡Hola! Soy el agente de CV de Israel Tiburcio. Puedo contarte sobre su "
    "experiencia profesional, habilidades, proyectos y trayectoria. ¿Qué te "
    "gustaría conocer?"
)

ENGLISH_GREETING_RESPONSE_TEXT = (
    "Hello! I'm Israel Tiburcio's CV agent. I can tell you about his "
    "professional experience, skills, projects, and career. What would you "
    "like to know?"
)

THANKS_RESPONSE_TEXT = (
    "¡Con gusto! Puedo ayudarte a explorar la trayectoria, experiencia, "
    "habilidades y proyectos profesionales de Israel."
)

GOODBYE_RESPONSE_TEXT = "¡Hasta luego! Si quieres, aquí estaré para hablar sobre su perfil profesional."

AGENT_IDENTITY_RESPONSE_TEXT = (
    "Sí. Soy el agente de CV de Israel Tiburcio. Puedo ayudarte a conocer su "
    "trayectoria profesional, experiencia, habilidades y proyectos."
)

AGENT_CAPABILITIES_RESPONSE_TEXT = (
    "Como agente de CV, puedo responder preguntas sobre la trayectoria profesional de Israel, su "
    "experiencia, proyectos, habilidades y conocimientos, usando únicamente "
    "la evidencia pública disponible en su perfil."
)

SENSITIVE_REQUEST_RESPONSE_TEXT = (
    "No puedo proporcionar contraseñas ni datos personales o sensibles. "
    "Puedo ayudarte con la información profesional pública de Israel."
)

OUT_OF_SCOPE_RESPONSE_TEXT = (
    "Mi función es ayudarte a conocer el perfil profesional de Israel. "
    "Puedo responder sobre su trayectoria, experiencia, proyectos y habilidades."
)

_PURE_GREETINGS = frozenset(
    {
        "hola",
        "hello",
        "hey",
        "buenos dias",
        "buen dia",
        "buenas tardes",
        "buenas noches",
    }
)
_ENGLISH_GREETINGS = frozenset({"hello", "hey"})

_PURE_THANKS = frozenset({
    "gracias",
    "muchas gracias",
    "perfecto gracias",
})

_PURE_GOODBYES = frozenset({
    "adios",
    "bye",
    "hasta luego",
    "nos vemos",
})

_META_RESPONSES = {
    "eres el agente de israel": AGENT_IDENTITY_RESPONSE_TEXT,
    "este es el agente de israel": AGENT_IDENTITY_RESPONSE_TEXT,
    "con quien estoy hablando": AGENT_IDENTITY_RESPONSE_TEXT,
    "que puedes hacer": AGENT_CAPABILITIES_RESPONSE_TEXT,
    "que te puedo preguntar": AGENT_CAPABILITIES_RESPONSE_TEXT,
    "para que sirves": AGENT_CAPABILITIES_RESPONSE_TEXT,
    "puedes responder preguntas sobre su cv": AGENT_CAPABILITIES_RESPONSE_TEXT,
}

_SENSITIVE_MARKERS = (
    "contrasena",
    "password",
    "direccion",
    "domicilio",
    "cuanto gana",
    "salario",
    "religion",
    "opinion politica",
    "diagnostico",
    "diagnosticar",
)

_OUT_OF_SCOPE_EXACT = frozenset({
    "cuentame un chiste",
    "quien gano el mundial",
    "quien gano la copa del mundo",
})


def _normalize_intent(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    normalized = re.sub(r"[^\w]+", " ", without_accents.casefold())
    return " ".join(normalized.split())


def is_pure_cv_greeting(value: str) -> bool:
    """Recognize only standalone greetings, never greeting-plus-question input."""

    return _normalize_intent(value) in _PURE_GREETINGS


def _is_capabilities_question(normalized: str) -> bool:
    """Recognize generic questions about what this agent can answer."""

    tokens = set(normalized.split())
    question_terms = tokens.intersection({"pregunta", "preguntas", "ask"}) or {
        token for token in tokens if token.startswith("pregunt")
    }
    ask_verbs = tokens.intersection(
        {
            "sugieres",
            "sugiero",
            "deberia",
            "puedo",
            "puedes",
            "ask",
            "hacer",
        }
    ) or question_terms
    if not ask_verbs:
        return False
    return bool(question_terms or tokens.intersection({"cosas", "ideas", "sobre"}))


def deterministic_response_for(value: str) -> tuple[str, str] | None:
    """Return a safe local response and agent status for non-professional turns."""

    normalized = _normalize_intent(value)
    if normalized in _PURE_GREETINGS:
        greeting = (
            ENGLISH_GREETING_RESPONSE_TEXT
            if normalized in _ENGLISH_GREETINGS
            else GREETING_RESPONSE_TEXT
        )
        return greeting, "ready"
    if normalized in _PURE_THANKS:
        return THANKS_RESPONSE_TEXT, "ready"
    if normalized in _PURE_GOODBYES:
        return GOODBYE_RESPONSE_TEXT, "ready"
    meta_response = _META_RESPONSES.get(normalized)
    if meta_response is not None:
        return meta_response, "ready"
    if _is_capabilities_question(normalized):
        return AGENT_CAPABILITIES_RESPONSE_TEXT, "ready"
    if any(marker in normalized for marker in _SENSITIVE_MARKERS):
        return SENSITIVE_REQUEST_RESPONSE_TEXT, "insufficient_evidence"
    if normalized in _OUT_OF_SCOPE_EXACT:
        return OUT_OF_SCOPE_RESPONSE_TEXT, "insufficient_evidence"
    return None
