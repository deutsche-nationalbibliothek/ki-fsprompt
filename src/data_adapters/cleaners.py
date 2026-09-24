import re
import unicodedata
from typing import Callable

_BULLET = re.compile(r"^\s*-\s+")
# Only "(S. <digit or roman numeral> ...)", so author initials like "(S. Freud)" survive.
_PAGE_NUMBER = re.compile(r"\s*\(S\.\s*[\dIVXLCivxlc][^)]*\)\s*$")
# Max. 3 digits per level so that headings starting with a year ("1848 ...") survive.
_NUMBERING = re.compile(r"^(?:(?:\d{1,3}(?:\.\d{1,3})*[.)]?|[IVXLC]+\.)\s+)+")
# Docling nests markdown headers inside bullets ("- ## Abbildungen"), so the header
# marker only becomes visible after the bullet is stripped.
_MD_HEADER = re.compile(r"^#{1,6}\s*")
# Dot leaders between heading and page number (". . . . ." or "......").
_DOT_LEADER = re.compile(r"(?:\s*\.\s*){3,}")
# Page number without parentheses at the end of a line ("... Jahre. 217").
_TRAILING_PAGE = re.compile(r"[\s.]+\d{1,4}\s*$")

# Structural/apparatus headings that appear in almost every thesis and therefore carry
# no information for retrieval. Only removed when the WHOLE line consists of such a
# term: "Einleitung: Der Veggie-Boom" and "Abstract Factory" must survive.
TOC_STOPWORDS: frozenset[str] = frozenset({
    # Front and back matter
    "vorwort", "geleitwort", "vorbemerkung", "danksagung", "dank", "danksagungen",
    "widmung", "lebenslauf", "curriculum vitae", "erklärung",
    "eidesstattliche erklärung", "selbständigkeitserklärung",
    "publikationen", "publikationsliste", "thesen",
    # Indexes and reference lists
    "inhalt", "inhaltsverzeichnis", "verzeichnisse", "abbildungsverzeichnis",
    "tabellenverzeichnis", "abkürzungsverzeichnis", "literaturverzeichnis",
    "quellenverzeichnis", "literatur", "literaturübersicht", "bibliographie",
    "glossar", "index", "register", "anhang", "anhänge", "anlagen",
    # Generic IMRaD chapters
    "einleitung", "einführung", "motivation", "zielsetzung", "fragestellung",
    "problemstellung", "aufbau der arbeit", "grundlagen", "allgemeines",
    "stand der forschung", "stand der technik",
    "material", "methoden", "methodik", "material und methoden",
    "material und methode", "statistik", "statistische auswertung",
    "ergebnisse", "resultate", "diskussion", "diskussion der ergebnisse",
    "zusammenfassende diskussion", "zusammenfassung",
    "zusammenfassung und ausblick", "ausblick", "fazit", "zwischenfazit",
    "schluss", "schlussbetrachtung", "schlussfolgerung", "schlussfolgerungen",
    "abstract", "summary",
    # English equivalents
    "introduction", "conclusion", "conclusions", "contents", "references",
    "acknowledgements", "acknowledgments", "appendix", "bibliography",
    "discussion", "results", "methods", "materials and methods",
})


def keep_text(text: str) -> str:
    return text


def _clean_toc_line(line: str) -> str:
    """Strip bullets, header markers, chapter numbering, dot leaders and page numbers."""
    line = _PAGE_NUMBER.sub("", line)
    # Loop because Docling nests bullets, headers and numbers: "- - ## 1 - 1.1. Titel".
    previous = None
    while line != previous:
        previous = line
        line = _NUMBERING.sub("", _MD_HEADER.sub("", _BULLET.sub("", line)))
    line = _DOT_LEADER.sub(" ", line)
    line = _TRAILING_PAGE.sub("", line)
    return re.sub(r"\s+", " ", line).strip()


def _lookup_key(line: str) -> str:
    """Normalize a line for comparison against TOC_STOPWORDS and for deduplication."""
    key = unicodedata.normalize("NFKC", line).lower()
    key = re.sub(r"[\s.:_–—-]+$", "", key)
    return re.sub(r"\s+", " ", key).strip()


def clean_toc(text: str) -> str:
    """Remove TOC formatting: bullets, numbering, dot leaders and page numbers."""
    cleaned = []
    for line in text.splitlines():
        line = _clean_toc_line(line)
        if line:
            cleaned.append(line)
    return "\n".join(cleaned)


def clean_toc_strict(text: str) -> str:
    """clean_toc plus removal of structural headings and of lines repeated within a document.

    Both are content decisions rather than formatting fixes, which is why they live in a
    separate cleaner: `toc` stays available as the comparison baseline.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = _clean_toc_line(line)
        if not line:
            continue
        key = _lookup_key(line)
        if key in TOC_STOPWORDS or key in seen:
            continue
        seen.add(key)
        cleaned.append(line)
    return "\n".join(cleaned)


CLEANERS: dict[str, Callable[[str], str]] = {
    "none": keep_text,
    "toc": clean_toc,
    "toc_strict": clean_toc_strict,
}
