"""Structured logging: console + JSON file with entity context."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog


def setup_logging(
    entity_name: str,
    level: str = "INFO",
    log_file: str | None = None,
    log_format: str = "json",
) -> None:
    # log_format: reserved for file sink; console is always human-readable.
    expanded_file: str | None = None
    if log_file:
        expanded_file = str(Path(log_file).expanduser())
        Path(expanded_file).parent.mkdir(parents=True, exist_ok=True)

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        timestamper,
        _entity_processor(entity_name),
    ]

    structlog.configure(
        processors=shared
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Console: always readable (JSON in the terminal breaks REPL / Telegram-only runs).
    handler_console = logging.StreamHandler(sys.stdout)
    handler_console.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=True),
            foreign_pre_chain=shared,
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler_console)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if expanded_file:
        fh = logging.FileHandler(expanded_file, encoding="utf-8")
        fh.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
                foreign_pre_chain=shared,
            )
        )
        root.addHandler(fh)


def _entity_processor(entity_name: str):
    def processor(logger, method_name, event_dict):
        event_dict.setdefault("entity_name", entity_name)
        return event_dict

    return processor


def get_logger(module: str):
    return structlog.get_logger(module)


def bind_context(**kwargs: Any) -> None:
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(**kwargs)
