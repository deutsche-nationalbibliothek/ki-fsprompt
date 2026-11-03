"""What gnd_context shows the model about a candidate: definition, using instructions, subject categories,
broader, narrower and related terms, and how the candidate relates to the agent's other candidates."""

# The kinds of broader relations in the DNB dumps, as shown to the model.
BROADER_KIND = {
    "Generic": "Kandidat ist eine Art davon",
    "Partitive": "Kandidat ist ein Teil davon",
    "Instantial": "Kandidat ist eine Instanz davon",
}
NARROWER_KIND = {
    "Generic": "ist eine Art des Kandidaten",
    "Partitive": "ist ein Teil des Kandidaten",
    "Instantial": "ist eine Instanz des Kandidaten",
}

# Long texts (definitions, using instructions) are cut after this many characters.
MAX_TEXT = 350


def by_frequency(frequency, ids):
    """Most often assigned first; ties by GND ID. Without the tie-break the order would follow Python's set order,
    which changes with every process (hash randomisation), so gnd_context could show different terms per run."""
    return sorted(ids, key=lambda i: (-frequency.get(i, 0), i))


def _short(text):
    return (text[:MAX_TEXT] + " …") if text and len(text) > MAX_TEXT else text


def _narrower_to_show(gnd, gnd_id, state, mapper, max_narrower):
    """At most max_narrower narrower terms, in this order:
    1. those that occur in the text read so far,
    2. if more are left than fit: the ones closest in meaning to the document,
    3. the rest, most often assigned first.
    Returns all narrower terms, the ones shown, and those found in the text."""
    narrower = gnd.g["narrower"].get(gnd_id, set())
    text = " ".join(state["text"]).lower()

    in_text = by_frequency(
        gnd.frequency,
        (i for i in narrower if len(gnd.label(i)) > 3 and gnd.label(i).lower() in text),
    )

    rest = narrower - set(in_text)
    room = max_narrower - len(in_text)
    similar = []
    if len(rest) > room > 0:
        # The first characters hold title and abstract.
        similar = mapper.closest(" ".join(state["text"])[:1000], rest, room)

    shown = (in_text + similar + by_frequency(gnd.frequency, rest - set(similar)))[:max_narrower]
    return narrower, shown, in_text


def _relations_to_candidates(gnd, gnd_id, state):
    """How the term relates to the other mapped candidates, e.g. "ist Oberbegriff deines Kandidaten „Malerei“"."""
    relations = []
    own_broader = gnd.broader_terms(gnd_id, 3)

    for k in state["mapped"]:
        if k == gnd_id:
            continue
        if gnd_id in gnd.broader_terms(k, 3):
            relations.append(f"ist Oberbegriff deines Kandidaten „{gnd.label(k)}“")
        elif k in own_broader:
            relations.append(f"ist Unterbegriff deines Kandidaten „{gnd.label(k)}“")
        elif k in gnd.neighbours(gnd_id):
            relations.append(f"ist verwandt mit deinem Kandidaten „{gnd.label(k)}“")

    return relations


def gnd_context(gnd, gnd_id, state, mapper, max_narrower):
    """What the GND knows about a candidate, so the agent can check its mappings."""
    g = gnd.g
    if gnd_id not in g["pref"]:
        return {"gnd_id": gnd_id, "hinweis": "unbekannte GND-ID"}

    details = gnd.details.get(gnd_id, {})

    # 1. The neighbours to show.
    narrower, narrower_shown, in_text = _narrower_to_show(gnd, gnd_id, state, mapper, max_narrower)

    # A broader term outside the vocabulary has no name and could never be a hit.
    broader_ids = (i for i in g["broader"].get(gnd_id, ()) if i in g["pref"])
    broader = by_frequency(gnd.frequency, broader_ids)[:5]

    # Only relations recorded in the concept's own record (reverse direction: e.g. Adorno -> USA).
    related_ids = (i for i in g["related_out"].get(gnd_id, set()) if i in gnd.details)
    related = by_frequency(gnd.frequency, related_ids)[:5]

    # 2. Everything shown may be assigned in finish() from now on.
    for i in broader + narrower_shown + related:
        state["graph"][i] = gnd.label(i)

    # 3. The answer for the model.
    def relation(words, child, parent):
        """The kind of a broader relation, where the GND records it (not for the unspecific broaderTermGeneral)."""
        kind = gnd.details.get(child, {}).get("broader_types", {}).get(parent)
        return {"beziehung": words[kind]} if kind else {}

    # Description (e.g. of places) and using instructions are only shown where the GND has them.
    optional = {
        "beschreibung": _short(details.get("description")),
        "benutzungshinweis": _short(details.get("using_instructions")),
    }

    return {
        "gnd_id": gnd_id,
        "begriff": gnd.label(gnd_id),
        "definition": _short(details.get("definition")),
        **{key: value for key, value in optional.items() if value},
        "sachgruppen": gnd.categories(gnd_id),
        "in_trainingsdokumenten_vergeben": gnd.frequency.get(gnd_id, 0),
        "oberbegriffe": [
            {"gnd_id": i, "begriff": gnd.label(i), **relation(BROADER_KIND, gnd_id, i)}
            for i in broader
        ],
        "unterbegriffe": {
            "anzahl": len(narrower),
            "auswahl": [
                {
                    "gnd_id": i,
                    "begriff": gnd.label(i),
                    **relation(NARROWER_KIND, i, gnd_id),
                    **({"im_text": True} if i in in_text else {}),
                }
                for i in narrower_shown
            ],
        },
        "verwandte_begriffe": [{"gnd_id": i, "begriff": gnd.label(i)} for i in related],
        "beziehungen_zu_deinen_kandidaten": _relations_to_candidates(gnd, gnd_id, state),
    }
