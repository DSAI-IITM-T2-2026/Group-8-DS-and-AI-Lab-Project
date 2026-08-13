"""Error contract: docs section 5.

Every error response is::

    {
      "code": "...",
      "message": "...",
      "requestId": "...",
      "fieldErrors": {"field": "reason"}   # optional
    }
"""

from __future__ import annotations

from typing import Any


class ApiError(Exception):
    """Base class for errors that map straight onto the documented envelope."""

    status_code = 500
    code = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        field_errors: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        self.field_errors = field_errors


class NotFoundError(ApiError):
    status_code = 404
    code = "not_found"


class ValidationFailedError(ApiError):
    """Malformed request (docs: 400)."""

    status_code = 400
    code = "bad_request"


class UnsupportedFeatureOverrideError(ApiError):
    """Invalid feature override or unsupported timestamp (docs: 422)."""

    status_code = 422
    code = "unprocessable_request"


class DataUnresolvableError(ApiError):
    """Requested data/model combination cannot be resolved (docs: 409)."""

    status_code = 409
    code = "data_unresolvable"


class ServiceUnavailableError(ApiError):
    """Model or required data source is unavailable (docs: 503).

    Raised instead of ever fabricating a probability, feature value, or
    validation record -- see Milestone 6/Application/api/README.md for which endpoints
    depend on assets (a trained model artifact, GCS credentials, a local
    historical archive) that are not bundled with this repository.
    """

    status_code = 503
    code = "service_unavailable"


def error_body(exc: ApiError, request_id: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "code": exc.code,
        "message": exc.message,
        "requestId": request_id,
    }
    if exc.field_errors:
        body["fieldErrors"] = exc.field_errors
    return body
