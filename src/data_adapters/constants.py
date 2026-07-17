from typing import Final

REQUIRED_COLUMNS: Final[tuple[str, str, str, str]] = (
    "text",
    "doc_id",
    "label_ids",
    "label_texts",
)
