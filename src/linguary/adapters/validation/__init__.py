"""Data validation at application external boundaries.

The adapter's public entry point is `EpubCheckValidator` and its configuration `EpubCheckSettings`:
implementation of the `ExportValidator` port from `linguary.application.ports.export_validation`
on top of external EPUBCheck. The application layer sees only the port and has no knowledge of
Java, jar files, or process execution (ADR § 3).
"""

from linguary.adapters.validation.epubcheck import (
    EpubCheckAvailability,
    EpubCheckSettings,
    EpubCheckValidator,
)

__all__ = ["EpubCheckAvailability", "EpubCheckSettings", "EpubCheckValidator"]
