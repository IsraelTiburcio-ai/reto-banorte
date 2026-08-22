FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install only the runtime project and dependencies from the canonical metadata.
COPY pyproject.toml /tmp/build/pyproject.toml
COPY app /tmp/build/app
RUN python -m pip install --no-cache-dir --target /app /tmp/build \
    && rm -rf /tmp/build \
    && groupadd --system app \
    && useradd --system --gid app --no-create-home --home-dir /nonexistent app \
    && mkdir -p /app/data \
    && chown -R app:app /app

COPY data/profile.json /app/data/profile.json
RUN chown app:app /app/data/profile.json

USER app:app

EXPOSE 8080

CMD ["sh", "-c", "exec python -m uvicorn app.api.main:app --host 0.0.0.0 --port \"${PORT:-8080}\""]
