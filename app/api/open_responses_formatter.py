"""Safe fixed text used when retrieval has insufficient public evidence."""

from __future__ import annotations

import re
import unicodedata

INSUFFICIENT_EVIDENCE_TEXT = (
    "No encontré evidencia pública suficiente en el perfil para responder esa pregunta."
)

UNKNOWN_TECHNOLOGY_RESPONSE_TEXT = (
    "No tengo información suficiente para afirmar que Israel haya trabajado con esa tecnología."
)

UNKNOWN_PERSONAL_DATA_RESPONSE_TEXT = (
    "No tengo ese dato personal exacto registrado en la información pública que puedo consultar."
)

UNKNOWN_AGE_RESPONSE_TEXT = (
    "No tengo la edad exacta de Israel registrada en la información pública que puedo consultar."
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

OUT_OF_SCOPE_SOCIAL_RESPONSE_TEXT = (
    "No tengo gustos propios 😄, pero puedo contarte sobre los proyectos y la experiencia "
    "profesional de Israel."
)

_PURE_GREETINGS = frozenset(
    {
        "hola",
        "holi",
        "hello",
        "hey",
        "buenas",
        "buenos dias",
        "buen dia",
        "buenas tardes",
        "buenas noches",
        "hola como estas",
        "hola como andas",
        "hola que tal",
        "holi como estas",
        "que onda",
        "que tal",
        "como estas",
        "como andas",
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
    "de quien es este agente": AGENT_IDENTITY_RESPONSE_TEXT,
    "que puedes hacer": AGENT_CAPABILITIES_RESPONSE_TEXT,
    "que te puedo preguntar": AGENT_CAPABILITIES_RESPONSE_TEXT,
    "para que sirves": AGENT_CAPABILITIES_RESPONSE_TEXT,
    "puedes responder preguntas sobre su cv": AGENT_CAPABILITIES_RESPONSE_TEXT,
}

_UNKNOWN_PERSONAL_MARKERS = (
    "edad",
    "anos",
    "cumpleanos",
    "cumple",
    "birthday",
    "age",
)
_TECHNOLOGY_QUESTION_VERBS = frozenset(
    {"sabe", "conoce", "experiencia", "trabajado", "usado", "utiliza", "domina"}
)

_CAPABILITIES_FRAMING_TERMS = frozenset(
    {
        "a",
        "about",
        "acerca",
        "al",
        "and",
        "ask",
        "como",
        "con",
        "cosas",
        "deberia",
        "cuál",
        "cuales",
        "cual",
        "dame",
        "de",
        "del",
        "el",
        "en",
        "for",
        "hacer",
        "hacerte",
        "ideas",
        "israel",
        "la",
        "las",
        "los",
        "me",
        "of",
        "on",
        "para",
        "pregunta",
        "preguntas",
        "preguntar",
        "preguntarte",
        "puede",
        "puedes",
        "puedo",
        "podria",
        "que",
        "qué",
        "sugiere",
        "sugieres",
        "sugiero",
        "sobre",
        "su",
        "sus",
        "te",
        "the",
        "tiburcio",
        "what",
        "you",
    }
)

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
    # Treat exaggerated social spelling as the same intent while leaving
    # ordinary doubled letters such as the ``ll`` in ``llama`` untouched.
    normalized = re.sub(r"([a-zñ])\1{2,}", r"\1", normalized)
    return " ".join(normalized.split())


def is_pure_cv_greeting(value: str) -> bool:
    """Recognize only standalone greetings, never greeting-plus-question input."""

    return _normalize_intent(value) in _PURE_GREETINGS


def _is_person_identity_question(normalized: str) -> bool:
    tokens = set(normalized.split())
    if "nombre" in tokens and ("completo" in tokens or "full" in tokens):
        return True
    return "llama" in tokens and ("israel" in tokens or "nombre" in tokens)


def _is_unknown_personal_question(normalized: str) -> bool:
    return any(marker in normalized for marker in _UNKNOWN_PERSONAL_MARKERS)


def _is_age_question(normalized: str) -> bool:
    return any(marker in normalized for marker in {"edad", "anos", "cumpleanos", "age"})


def _is_unknown_technology_question(normalized: str) -> bool:
    tokens = set(normalized.split())
    return bool(tokens.intersection(_TECHNOLOGY_QUESTION_VERBS)) and any(
        token not in _CAPABILITIES_FRAMING_TERMS
        and token not in _TECHNOLOGY_QUESTION_VERBS
        for token in tokens
    )


def _thanks_with_informal_suffix(normalized: str) -> bool:
    tokens = normalized.split()
    return bool(tokens) and tokens[0] in {"gracias", "perfecto", "muchas"} and set(
        tokens
    ) <= {"gracias", "perfecto", "muchas", "cawn"}


def _person_identity_response(identity_name: str | None) -> str:
    if identity_name:
        return f"Israel se llama {identity_name}."
    return "No tengo el nombre completo de Israel disponible en la información pública que puedo consultar."


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
    if not (question_terms or tokens.intersection({"cosas", "ideas", "sobre"})):
        return False
    substantive_terms = tokens - _CAPABILITIES_FRAMING_TERMS
    return not substantive_terms


def deterministic_response_for(
    value: str, *, identity_name: str | None = None
) -> tuple[str, str] | None:
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
    if _thanks_with_informal_suffix(normalized):
        return THANKS_RESPONSE_TEXT, "ready"
    if normalized in _PURE_GOODBYES:
        return GOODBYE_RESPONSE_TEXT, "ready"
    if _is_person_identity_question(normalized):
        return _person_identity_response(identity_name), "ready"
    if normalized == "de quien es este agente":
        subject = identity_name or "Israel"
        return f"Este es el agente de CV de {subject}.", "ready"
    meta_response = _META_RESPONSES.get(normalized)
    if meta_response is not None:
        return meta_response, "ready"
    if _is_capabilities_question(normalized):
        return AGENT_CAPABILITIES_RESPONSE_TEXT, "ready"
    if any(marker in normalized for marker in _SENSITIVE_MARKERS):
        return SENSITIVE_REQUEST_RESPONSE_TEXT, "insufficient_evidence"
    if _is_unknown_personal_question(normalized):
        response = (
            UNKNOWN_AGE_RESPONSE_TEXT
            if _is_age_question(normalized)
            else UNKNOWN_PERSONAL_DATA_RESPONSE_TEXT
        )
        return response, "insufficient_evidence"
    if normalized == "te gusta el futbol":
        return OUT_OF_SCOPE_SOCIAL_RESPONSE_TEXT, "insufficient_evidence"
    if normalized in _OUT_OF_SCOPE_EXACT:
        return OUT_OF_SCOPE_RESPONSE_TEXT, "insufficient_evidence"
    return None


def insufficient_response_for(value: str) -> str:
    """Choose a calibrated local fallback without turning absence into a fact."""

    normalized = _normalize_intent(value)
    if _is_unknown_technology_question(normalized):
        return UNKNOWN_TECHNOLOGY_RESPONSE_TEXT
    if _is_unknown_personal_question(normalized):
        return (
            UNKNOWN_AGE_RESPONSE_TEXT
            if _is_age_question(normalized)
            else UNKNOWN_PERSONAL_DATA_RESPONSE_TEXT
        )
    return INSUFFICIENT_EVIDENCE_TEXT
