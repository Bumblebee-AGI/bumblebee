FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY . /app

# canary.yaml is gitignored locally; ship Canary from the canonical example for BUMBLEBEE_ENTITY=canary.
RUN cp configs/entities/canary.example.yaml configs/entities/canary.yaml

RUN pip install --upgrade pip && \
    pip install ".[railway,api]"
