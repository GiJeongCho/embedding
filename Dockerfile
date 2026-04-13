FROM docker.io/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

ENV PYTHONPATH=/app


EXPOSE ${APP_PORT}
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${APP_PORT}"]