import re
from typing import Callable

_BULLET = re.compile(r"^\s*-\s+")
# Only "(S. <digit or roman numeral> ...)", so author initials like "(S. Freud)" survive.
_PAGE_NUMBER = re.compile(r"\s*\(S\.\s*[\dIVXLCivxlc][^)]*\)\s*$")
# Max. 3 digits per level so that headings starting with a year ("1848 ...") survive.
_NUMBERING = re.compile(r"^(?:(?:\d{1,3}(?:\.\d{1,3})*[.)]?|[IVXLC]+\.)\s+)+")


def keep_text(text: str) -> str:
    return text


def clean_toc(text: str) -> str:
    """Remove bullets/indentation, page numbers "(S. 5)" and chapter numbering from a TOC."""
    cleaned = []
    for line in text.splitlines():
        line = _PAGE_NUMBER.sub("", line)
        # Loop because Docling nests bullets and numbers: "- - 1 - 1.1. Titel".
        previous = None
        while line != previous:
            previous = line
            line = _NUMBERING.sub("", _BULLET.sub("", line))
        line = line.strip()
        if line:
            cleaned.append(line)
    return "\n".join(cleaned)


CLEANERS: dict[str, Callable[[str], str]] = {
    "none": keep_text,
    "toc": clean_toc,
}
