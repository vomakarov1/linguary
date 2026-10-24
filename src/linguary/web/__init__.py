"""Local Web UI: FastAPI application, endpoints, and external DTOs (PRD § 21).

Public entry point of the layer is `create_app`. The layer does not assemble dependencies:
they arrive from `linguary.composition`, so Web UI, CLI, and end-to-end tests work with
the exact same application (ADR § 3).
"""

from linguary.web.app import create_app

__all__ = ["create_app"]
