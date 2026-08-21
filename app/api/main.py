"""FastAPI application exposing the Phase 4 HTTP boundary."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.agent.core import AgentCore, AgentInputError
from app.api.schemas import AgentPrepareRequest, AgentPrepareResponse, HealthResponse


def create_app(agent_core: AgentCore | None = None) -> FastAPI:
    """Create the HTTP application around a trusted ``AgentCore`` instance."""

    app = FastAPI(
        title="Reto IA Banorte — CV Agent",
        version="0.4.0",
        description="Phase 4 HTTP API for deterministic CV-agent preparation.",
    )
    app.state.agent_core = agent_core if agent_core is not None else AgentCore()

    @app.exception_handler(AgentInputError)
    async def handle_agent_input_error(
        _request: Request,
        exc: AgentInputError,
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/agent/prepare", response_model=AgentPrepareResponse)
    def prepare_agent_turn(payload: AgentPrepareRequest) -> AgentPrepareResponse:
        core: AgentCore = app.state.agent_core
        turn = core.prepare(
            payload.query,
            max_results=payload.max_results,
        )
        return AgentPrepareResponse.from_turn(turn)

    return app


app = create_app()
