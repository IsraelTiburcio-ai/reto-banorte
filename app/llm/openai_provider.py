"""OpenAI Responses API implementation of the provider-neutral generator."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from app.models.generation import TextGenerationRequest, TextGenerator
from app.llm.errors import (
    EmptyProviderResponseError,
    MalformedProviderResponseError,
    MissingAPIKeyError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    TextGenerationError,
)
from app.llm.prompts import build_model_input, build_system_instructions


DEFAULT_PROVIDER_MODEL = "gpt-5.6-luna"
DEFAULT_TIMEOUT_SECONDS = 30.0
ClientFactory = Callable[[str, float], Any]


def _default_client_factory(api_key: str, timeout: float) -> OpenAI:
    return OpenAI(api_key=api_key, timeout=timeout)


class OpenAITextGenerator(TextGenerator):
    """Generate grounded text through the official OpenAI Responses API."""

    def __init__(
        self,
        *,
        api_key_env: str = "OPENAI_API_KEY",
        model_env: str = "OPENAI_MODEL",
        default_model: str = DEFAULT_PROVIDER_MODEL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self._api_key_env = api_key_env
        self._model_env = model_env
        self._default_model = default_model
        self._timeout_seconds = timeout_seconds
        self._client_factory = client_factory or _default_client_factory
        self._client: Any | None = None
        self._client_key: str | None = None

    @property
    def provider_model(self) -> str:
        configured = os.getenv(self._model_env, "").strip()
        return configured or self._default_model

    def generate(self, request: TextGenerationRequest) -> str:
        api_key = os.getenv(self._api_key_env, "").strip()
        if not api_key:
            raise MissingAPIKeyError()

        try:
            client = self._get_client(api_key)
            response = client.responses.create(
                model=self.provider_model,
                instructions=build_system_instructions(request),
                input=build_model_input(request),
                store=False,
            )
        except TextGenerationError:
            raise
        except AuthenticationError:
            raise ProviderAuthenticationError() from None
        except APITimeoutError:
            raise ProviderTimeoutError() from None
        except RateLimitError:
            raise ProviderRateLimitError() from None
        except (APIConnectionError, APIError, AttributeError, TypeError):
            raise ProviderUnavailableError() from None
        except Exception:
            # Do not expose arbitrary SDK/provider details to the HTTP boundary.
            raise ProviderUnavailableError() from None

        return self._extract_text(response)

    def _get_client(self, api_key: str) -> Any:
        if self._client is None or self._client_key != api_key:
            self._client = self._client_factory(api_key, self._timeout_seconds)
            self._client_key = api_key
        return self._client

    @staticmethod
    def _extract_text(response: object) -> str:
        try:
            text = getattr(response, "output_text")
        except (AttributeError, TypeError):
            raise MalformedProviderResponseError() from None
        except Exception:
            raise MalformedProviderResponseError() from None
        if not isinstance(text, str):
            raise MalformedProviderResponseError()
        if not text.strip():
            raise EmptyProviderResponseError()
        return text
