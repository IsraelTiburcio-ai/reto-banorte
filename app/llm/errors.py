"""Safe, provider-neutral errors for grounded text generation."""

from __future__ import annotations


class TextGenerationError(RuntimeError):
    """Base error whose public fields are safe for the HTTP adapter."""

    code = "provider_unavailable"
    status_code = 503
    public_message = "Text generation is temporarily unavailable."


class MissingAPIKeyError(TextGenerationError):
    code = "missing_api_key"
    status_code = 503
    public_message = "Text generation is not configured on this server."


class ProviderAuthenticationError(TextGenerationError):
    code = "provider_authentication_failed"
    status_code = 502
    public_message = "Text generation provider authentication failed."


class ProviderTimeoutError(TextGenerationError):
    code = "provider_timeout"
    status_code = 504
    public_message = "Text generation provider timed out."


class ProviderRateLimitError(TextGenerationError):
    code = "provider_rate_limited"
    status_code = 429
    public_message = "Text generation provider rate limit reached."


class ProviderUnavailableError(TextGenerationError):
    code = "provider_unavailable"
    status_code = 503
    public_message = "Text generation provider is temporarily unavailable."


class MalformedProviderResponseError(TextGenerationError):
    code = "malformed_provider_response"
    status_code = 502
    public_message = "Text generation provider returned an invalid response."


class EmptyProviderResponseError(TextGenerationError):
    code = "empty_provider_response"
    status_code = 502
    public_message = "Text generation provider returned no text."
