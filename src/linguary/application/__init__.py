"""Application layer: use cases and ports.

The layer depends only on the domain layer and standard library. Access to DB, EPUB, LLM,
file system, and HTTP is defined by ports (Protocol/ABC) and implemented by adapters.
"""
