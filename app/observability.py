import contextvars
import json
import logging
from datetime import UTC, datetime
from typing import Any

REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", REQUEST_ID.get()),
        }
        extra_fields = getattr(record, "extra_fields", None)
        if isinstance(extra_fields, dict):
            payload.update(extra_fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str) -> None:
    root = logging.getLogger()
    if not any(isinstance(handler.formatter, JsonFormatter) for handler in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        root.handlers.clear()
        root.addHandler(handler)
    root.setLevel(level.upper())


def set_request_id(request_id: str) -> contextvars.Token[str]:
    return REQUEST_ID.set(request_id)


def reset_request_id(token: contextvars.Token[str]) -> None:
    REQUEST_ID.reset(token)


def get_request_id() -> str:
    return REQUEST_ID.get()


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    logger.info(event, extra={"extra_fields": {"event": event, **fields}})
