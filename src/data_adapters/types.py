from dataclasses import dataclass


@dataclass(frozen=True)
class AdapterOptions:
    """Common runtime options for data adapters."""

    max_rows: int | None = None
    debug_rows: int | None = None
    strict: bool = True
