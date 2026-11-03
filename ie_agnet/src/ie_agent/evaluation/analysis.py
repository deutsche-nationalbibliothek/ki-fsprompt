"""Analysis of one run: how the agent used its tools, how each decision branch scored, which tool led to hits.

Branch scores are descriptive: the agent chooses its own path, so long paths also mean difficult documents. Whether
a tool helps can only be shown by comparing runs on the same documents with and without it.
"""
import numpy as np

TOOLS = ["read_abstract_intro", "read_toc", "get_examples", "map_to_gnd", "gnd_context", "finish"]
READ_SECTION = {"read_abstract_intro": "abstract_intro", "read_toc": "toc"}
STAGE_OF_TOOL = {"read_abstract_intro": "after abstract", "read_toc": "after toc"}


# ---------------------------------------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------------------------------------

def prf(predicted, gold):
    """Set precision, recall and F1 of one document."""
    predicted = set(predicted)
    hits = len(predicted & gold)
    p = hits / len(predicted) if predicted else 0.0
    r = hits / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if hits else 0.0
    return p, r, f1


def score(entries, gold, ids):
    """Set P/R/F1 per document against the title gold, averaged (as in auswertung.py). Documents without a
    prediction (also after an error) count as 0. Also reads logs of the older agents (German field names)."""
    by_id = {e["doc_id"]: e for e in entries}

    rows = []
    for doc in ids:
        e = by_id.get(doc, {})
        p, r, f1 = prf(e.get("gnd_ids", []), gold[doc])
        n_headings = len(set(e.get("gnd_ids", [])))
        n_steps = e.get("n_steps", e.get("n_schritte", 0))
        aborted = bool(e.get("abort") or e.get("abbruch") or e.get("error") or not e)
        rows.append((p, r, f1, n_headings, n_steps, aborted))

    t = np.array(rows, dtype=float)
    return {
        "P": t[:, 0].mean(),
        "R": t[:, 1].mean(),
        "F1": t[:, 2].mean(),
        "n": t[:, 3].mean(),
        "steps": t[:, 4].mean(),
        "aborted": int(t[:, 5].sum()),
        "docs": len(t),
    }


def branch(entries, gold, total):
    """Scores and cost of a group of documents (one branch of the agent's decisions)."""
    if not entries:
        return {"docs": 0, "share": 0.0}

    scores = np.array([prf(e["gnd_ids"], gold[e["doc_id"]]) for e in entries])
    return {
        "docs": len(entries),
        "share": len(entries) / total,
        "P": scores[:, 0].mean(),
        "R": scores[:, 1].mean(),
        "F1": scores[:, 2].mean(),
        "headings": np.mean([len(e["gnd_ids"]) for e in entries]),
        "steps": np.mean([e["n_steps"] for e in entries]),
        "tokens_in": np.mean([e["tokens_in"] for e in entries]),
    }


def _means(scores):
    return dict(zip(("P", "R", "F1"), np.mean(scores, axis=0)))


def _bootstrap(differences, seed=1233415, rounds=5000):
    """95 % confidence interval of the mean difference."""
    rng = np.random.default_rng(seed)
    means = [differences[rng.integers(0, len(differences), len(differences))].mean() for _ in range(rounds)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------------------------------------------------
# Tool usage and decision branches
# ---------------------------------------------------------------------------------------------------------

def calls(entry, tool):
    """How often a document called a tool."""
    return sum(s["tool"] == tool for s in entry["steps"])


def tool_usage(entries):
    usage = {}
    for tool in TOOLS:
        steps = [s for e in entries for s in e["steps"] if s["tool"] == tool]
        usage[tool] = {
            "calls": len(steps),
            "docs": sum(calls(e, tool) > 0 for e in entries),
            "per_doc": len(steps) / len(entries),
            "refused": sum(bool(s.get("refused")) for s in steps),
        }
    return usage


def _reading(entry):
    """How far a document was read."""
    if "toc" in entry["read"]:
        return "title + abstract + toc"
    if "abstract_intro" in entry["read"]:
        return "title + abstract"
    return "title only"


def branches(entries, gold):
    n = len(entries)

    def group(selected):
        return branch(selected, gold, n)

    def by_calls(tool, top):
        """Documents with 0, 1, ... calls of a tool; the last group is "top or more"."""
        groups = {str(c): group([e for e in entries if calls(e, tool) == c]) for c in range(top)}
        groups[f"{top} or more"] = group([e for e in entries if calls(e, tool) >= top])
        return groups

    # How far the documents were read.
    read = {"title only": [], "title + abstract": [], "title + abstract + toc": []}
    for e in entries:
        read[_reading(e)].append(e)

    result = {"reading": {k: group(v) for k, v in read.items()}}

    # The same, split by whether get_examples was used.
    result["reading and examples"] = {}
    for k, v in read.items():
        result["reading and examples"][f"{k}, no get_examples"] = group([e for e in v if calls(e, "get_examples") == 0])
        result["reading and examples"][f"{k}, get_examples"] = group([e for e in v if calls(e, "get_examples") > 0])

    result["gnd_context calls"] = by_calls("gnd_context", 4)
    result["map_to_gnd calls"] = by_calls("map_to_gnd", 3)

    result["ending"] = {
        "finish by itself": group([e for e in entries if not e["forced"] and not e["abort"]]),
        "finish forced": group([e for e in entries if e["forced"] and not e["abort"]]),
        "aborted": group([e for e in entries if e["abort"]]),
    }

    result["term from the graph"] = {
        "yes": group([e for e in entries if e["from_graph"]]),
        "no": group([e for e in entries if not e["from_graph"]]),
    }
    return result


# ---------------------------------------------------------------------------------------------------------
# Which tool led to hits
# ---------------------------------------------------------------------------------------------------------

def tool_hits(entries, gold):
    """Which tool led to hits. Needs the step details written since the analysis exists; older logs return None."""
    if not all("title_example_ids" in e for e in entries):
        return None

    # The counters.
    stages = {
        s: {"mapped": 0, "in_gold": 0, "assigned": 0, "assigned_correct": 0}
        for s in ("title only", "after abstract", "after toc")
    }
    examples = {
        s: {"docs": 0, "docs_with_gold": 0, "gold_in_examples": 0, "of_these_assigned": 0}
        for s in ("title", "abstract_intro", "toc")
    }
    mapping = {"candidates": 0, "accepted": 0, "rejected_low_similarity": 0}
    gnd_context = {
        "ids_checked": 0,
        "terms_shown": 0,
        "shown_in_gold": 0,
        "assigned_from_graph": 0,
        "assigned_from_graph_correct": 0,
        "kept_correct": 0,
        "kept_wrong": 0,
        "dropped_correct": 0,
        "dropped_wrong": 0,
        "checked_dropped_correct": 0,
        "checked_dropped_wrong": 0,
    }
    before_context, after_context, all_mapped, final_scores = [], [], [], []

    def count_examples(section, ids, g, final):
        found = set(ids) & g
        ex = examples[section]
        ex["docs"] += 1
        ex["docs_with_gold"] += bool(found)
        ex["gold_in_examples"] += len(found)
        ex["of_these_assigned"] += len(found & final)

    for e in entries:
        g = gold[e["doc_id"]]
        final = set(e["gnd_ids"])
        count_examples("title", e["title_example_ids"], g, final)

        # 1. Go through the steps: where each ID was first mapped, what gnd_context checked and showed.
        stage = "title only"
        first_stage = {}  # GND ID -> stage in which it was first mapped
        checked = set()  # IDs given to gnd_context
        snapshot = None  # what was mapped before the first gnd_context call
        for s in e["steps"]:
            if s.get("refused"):
                continue
            tool = s["tool"]
            stage = STAGE_OF_TOOL.get(tool, stage)

            if "example_ids" in s:
                section = READ_SECTION.get(tool) or s["args"].get("section")
                count_examples(section, s["example_ids"], g, final)

            if tool == "map_to_gnd":
                for r in s.get("results", []):
                    mapping["candidates"] += 1
                    mapping["accepted" if r["gnd_id"] else "rejected_low_similarity"] += 1
                    if r["gnd_id"]:
                        first_stage.setdefault(r["gnd_id"], stage)

            if tool == "gnd_context":
                if snapshot is None:
                    snapshot = set(first_stage)
                checked |= {str(i) for i in s["args"].get("gnd_ids", [])}
                gnd_context["terms_shown"] += len(s.get("shown", []))
                gnd_context["shown_in_gold"] += len(set(s.get("shown", [])) & g)

        # 2. What became of every mapped ID.
        for gnd_id, st in first_stage.items():
            correct = "correct" if gnd_id in g else "wrong"
            row = stages[st]
            row["mapped"] += 1
            row["in_gold"] += gnd_id in g
            row["assigned"] += gnd_id in final
            row["assigned_correct"] += gnd_id in final and gnd_id in g

            kept = "kept" if gnd_id in final else "dropped"
            gnd_context[f"{kept}_{correct}"] += 1
            if gnd_id in checked and gnd_id not in final:
                gnd_context[f"checked_dropped_{correct}"] += 1

        gnd_context["ids_checked"] += len(checked)
        gnd_context["assigned_from_graph"] += len(e["from_graph"])
        gnd_context["assigned_from_graph_correct"] += len(set(e["from_graph"]) & g)

        # 3. Scores: everything mapped, the final result, and before/after the first gnd_context.
        all_mapped.append(prf(first_stage, g))
        final_scores.append(prf(final, g))
        if snapshot is not None:
            before_context.append(prf(snapshot, g))
            after_context.append(prf(final, g))

    for row in stages.values():
        row["precision_of_assigned"] = row["assigned_correct"] / row["assigned"] if row["assigned"] else None

    result = {
        "map_to_gnd": mapping,
        "first_mapped": stages,
        "examples": examples,
        "gnd_context": gnd_context,
        "selection": {"all mapped": _means(all_mapped), "final": _means(final_scores)},
    }

    if before_context:
        diff = np.array(after_context)[:, 2] - np.array(before_context)[:, 2]
        low, high = _bootstrap(diff)
        result["before_and_after_gnd_context"] = {
            "docs": len(diff),
            "before": _means(before_context),
            "after": _means(after_context),
            "delta_F1": diff.mean(),
            "ci95": [low, high],
            "better": int((diff > 0).sum()),
            "worse": int((diff < 0).sum()),
            "same": int((diff == 0).sum()),
        }
    return result


# ---------------------------------------------------------------------------------------------------------
# Where gold headings got lost
# ---------------------------------------------------------------------------------------------------------

REACHED = [
    "assigned",
    "mapped, not assigned",
    "shown by gnd_context, not assigned",
    "only in the examples",
    "never seen",
]
RECALL_MISSES = ("only in the examples", "never seen")


def _where(gnd_id, final, mapped, shown, in_examples):
    """How far one gold heading got."""
    if gnd_id in final:
        return "assigned"
    if gnd_id in mapped:
        return "mapped, not assigned"
    if gnd_id in shown:
        return "shown by gnd_context, not assigned"
    if gnd_id in in_examples:
        return "only in the examples"
    return "never seen"


def gold_coverage(entries, gold, expansions=None):
    """How far each gold heading got. Selection errors: it was a candidate (mapped or shown by gnd_context), but not
    assigned. Recall errors: it never was. expansions (name -> function GND ID -> IDs) measures how many recall errors
    an expansion of the mapped terms would have reached, and how many new candidates it costs per document."""
    if not all("title_example_ids" in e for e in entries):
        return None

    reached = dict.fromkeys(REACHED, 0)
    expansions = expansions or {}
    potential = {name: {"found": 0, "new_candidates": 0} for name in expansions}

    for e in entries:
        # 1. What the agent saw of the GND for this document.
        mapped, shown, in_examples = set(), set(), set(e["title_example_ids"])
        for s in e["steps"]:
            if s.get("refused"):
                continue
            mapped |= {r["gnd_id"] for r in s.get("results", []) if r["gnd_id"]}
            shown |= set(s.get("shown", []))
            in_examples |= set(s.get("example_ids", []))

        # 2. How far each gold heading got.
        final = set(e["gnd_ids"])
        misses = set()
        for gnd_id in gold[e["doc_id"]]:
            where = _where(gnd_id, final, mapped, shown, in_examples)
            reached[where] += 1
            if where in RECALL_MISSES:
                misses.add(gnd_id)

        # 3. Would an expansion of the mapped terms have reached the recall misses?
        for name, expand in expansions.items():
            new = {n for m in mapped for n in expand(m)} - mapped - shown
            potential[name]["found"] += len(new & misses)
            potential[name]["new_candidates"] += len(new)

    total = sum(reached.values())
    recall_misses = sum(reached[k] for k in RECALL_MISSES)
    for p in potential.values():
        p["share_of_recall_misses"] = p["found"] / recall_misses if recall_misses else None
        p["new_candidates_per_doc"] = p.pop("new_candidates") / len(entries) if entries else 0.0

    return {
        "gold": total,
        "reached": reached,
        "selection_misses": reached["mapped, not assigned"] + reached["shown by gnd_context, not assigned"],
        "recall_misses": recall_misses,
        "expansions": potential,
    }


# ---------------------------------------------------------------------------------------------------------
# Everything together, and as Markdown
# ---------------------------------------------------------------------------------------------------------

def analyze(entries, gold, expansions=None):
    ok = [e for e in entries if "error" not in e]
    return {
        "docs": len(ok),
        "errors": len(entries) - len(ok),
        "overall": branch(ok, gold, len(ok)),
        "tools": tool_usage(ok),
        "branches": branches(ok, gold),
        "tool_hits": tool_hits(ok, gold),
        "gold_coverage": gold_coverage(ok, gold, expansions),
    }


def _table(header, rows):
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    return lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


def _branch_rows(group):
    rows = []
    for label, b in group.items():
        if not b["docs"]:
            rows.append([label, 0, "0.0%", "–", "–", "–", "–", "–", "–"])
            continue
        rows.append([
            label,
            b["docs"],
            f"{b['share']:.1%}",
            f"{b['P']:.3f}",
            f"{b['R']:.3f}",
            f"**{b['F1']:.3f}**",
            f"{b['headings']:.1f}",
            f"{b['steps']:.1f}",
            f"{b['tokens_in']:,.0f}",
        ])
    return rows


def _prf(row):
    return f"{row['P']:.3f} / {row['R']:.3f} / {row['F1']:.3f}"


def _share(value):
    return "–" if value is None else f"{value:.1%}"


def to_markdown(a, title):
    o = a["overall"]

    # 1. Head: overall scores and how to read the rest.
    lines = [
        f"# Analysis: {title}",
        "",
        f"Created: {a.get('created', 'unknown')}",
        "",
        f"{a['docs']} documents, {a['errors']} errors · P {o['P']:.3f} · R {o['R']:.3f} · F1 {o['F1']:.3f} · "
        f"{o['headings']:.1f} headings, {o['steps']:.1f} steps, {o['tokens_in']:,.0f} input tokens per document",
        "",
        "Scores here count documents without a result as 0; `scores_at_100_R.yaml` (casimir) is the official value.",
        "",
        "**How to read this:** the agent chooses its own path, so long paths also mean difficult documents. "
        "Branch scores show how the agent behaves, not whether a tool helps. That needs a comparison on the same "
        "documents with and without the tool (`--no-gnd-context`, `--no-toc`, `--examples-with-sections`).",
        "",
        "## Tool usage",
        "",
    ]

    # 2. Tool usage and branches.
    lines += _table(
        ["tool", "calls", "documents", "per document", "refused"],
        [[t, u["calls"], u["docs"], f"{u['per_doc']:.2f}", u["refused"]] for t, u in a["tools"].items()],
    )
    head = ["branch", "docs", "share", "P", "R", "F1", "headings", "steps", "input tokens"]
    for name, group in a["branches"].items():
        lines += ["", f"## Branches: {name}", ""] + _table(head, _branch_rows(group))

    # 3. Hits per tool (not in logs written before the step details existed).
    hits = a["tool_hits"]
    if hits is None:
        lines += [
            "",
            "## Hits per tool",
            "",
            "Not available: this log was written before the step details existed (the column `refused` above "
            "is not recorded there either).",
        ]
        return "\n".join(lines) + "\n"

    m = hits["map_to_gnd"]
    lines += [
        "",
        "## Hits per tool",
        "",
        "### map_to_gnd",
        "",
        f"{m['candidates']} candidates: {m['accepted']} accepted, {m['rejected_low_similarity']} rejected "
        "(cosine below the limit).",
        "",
        "### When a term was first mapped",
        "",
    ]
    lines += _table(
        ["stage", "mapped", "in gold", "assigned", "assigned and correct", "precision of assigned"],
        [
            [
                s,
                r["mapped"],
                r["in_gold"],
                r["assigned"],
                r["assigned_correct"],
                "–" if r["precision_of_assigned"] is None else f"{r['precision_of_assigned']:.3f}",
            ]
            for s, r in hits["first_mapped"].items()
        ],
    )

    lines += ["", "### Examples: did they contain gold headings?", ""]
    lines += _table(
        ["examples of", "docs", "docs with a gold heading", "gold headings in examples", "of these assigned"],
        [
            [s, r["docs"], r["docs_with_gold"], r["gold_in_examples"], r["of_these_assigned"]]
            for s, r in hits["examples"].items()
        ],
    )

    k = hits["gnd_context"]
    lines += ["", "### gnd_context and the final selection", ""]
    lines += _table(["what", "count"], [
        ["IDs checked", k["ids_checked"]],
        ["new terms shown from the graph", k["terms_shown"]],
        ["of these in gold", k["shown_in_gold"]],
        ["assigned from the graph", k["assigned_from_graph"]],
        ["of these correct", k["assigned_from_graph_correct"]],
        ["mapped and kept: correct", k["kept_correct"]],
        ["mapped and kept: wrong", k["kept_wrong"]],
        ["mapped and dropped: wrong (good)", k["dropped_wrong"]],
        ["mapped and dropped: correct (lost hit)", k["dropped_correct"]],
        ["checked and dropped: wrong (good)", k["checked_dropped_wrong"]],
        ["checked and dropped: correct (lost hit)", k["checked_dropped_correct"]],
    ])

    s = hits["selection"]
    lines += [
        "",
        f"All mapped terms as result (P / R / F1): {_prf(s['all mapped'])}. Final result: {_prf(s['final'])}.",
    ]

    b = hits.get("before_and_after_gnd_context")
    if b:
        lines += [
            "",
            "Before the first `gnd_context` (everything mapped so far) against the final result, "
            f"{b['docs']} documents: {_prf(b['before'])} → {_prf(b['after'])}, ΔF1 {b['delta_F1']:+.3f}, 95 % CI "
            f"[{b['ci95'][0]:+.3f}; {b['ci95'][1]:+.3f}]; better {b['better']}, worse {b['worse']}, "
            f"same {b['same']}.",
            "",
            "This also contains later mapping and the model's own selection. The clean test of gnd_context is a run "
            "with `--no-gnd-context`.",
        ]

    # 4. Where gold headings got lost.
    c = a.get("gold_coverage")
    if c and c["gold"]:
        lines += [
            "",
            "## Where gold headings got lost",
            "",
            "Selection error: the heading was a candidate (mapped or shown by `gnd_context`), but not assigned. "
            "Recall error: it never was. More graph knowledge for checking candidates mostly helps against "
            "selection errors; recall errors need new candidates.",
            "",
        ]
        lines += _table(
            ["gold heading", "count", "share"],
            [[k, n, f"{n / c['gold']:.1%}"] for k, n in c["reached"].items()],
        )
        lines += ["", f"Selection errors: {c['selection_misses']} · recall errors: {c['recall_misses']}."]

        if c["expansions"]:
            lines += ["", "### Recall errors an expansion of the mapped terms would have reached", ""]
            lines += _table(
                ["expansion", "recall errors reached", "share", "new candidates per document"],
                [
                    [name, p["found"], _share(p["share_of_recall_misses"]), f"{p['new_candidates_per_doc']:.1f}"]
                    for name, p in c["expansions"].items()
                ],
            )
            lines += ["", "An upper bound: the model would still have to pick them among the new candidates."]

    return "\n".join(lines) + "\n"
