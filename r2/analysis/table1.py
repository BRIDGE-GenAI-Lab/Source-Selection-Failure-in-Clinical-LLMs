"""Text of the Table 1 rows (analysis_original/table1_examples.csv, r2.analysis.original_corrected.table1_examples):
the patient line, the edit by field and the quoted rationale."""
MOD_CHARS = 120
QUOTE_CHARS = 160
CONTEXT_WORDS = 4
ELLIPSIS = "…"
SEX = {"F": "woman", "M": "man"}


def _clip(text, n=MOD_CHARS):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n].rstrip() + ELLIPSIS


def quote(text, n=QUOTE_CHARS):
    """The first n characters of the rationale, verbatim, with an ellipsis when cut."""
    text = str(text).strip()
    return text if len(text) <= n else text[:n] + ELLIPSIS


def _side(x):
    """One side of a change as a list of lines (text split at newlines, a list item by item)."""
    if x is None:
        return []
    if isinstance(x, str):
        return x.splitlines() if "\n" in x else [x]
    if isinstance(x, (list, tuple, set)):
        return [str(i) for i in x]
    return [str(x)]


def _focus(before, after):
    """Drop the shared opening words of a changed line (keeping a few for context) so the edit is visible."""
    a, b = str(before).split(), str(after).split()
    common = 0
    while common < min(len(a), len(b)) and a[common] == b[common]:
        common += 1
    cut = common - CONTEXT_WORDS
    if cut <= 0:
        return " ".join(a), " ".join(b)
    return ELLIPSIS + " ".join(a[cut:]), ELLIPSIS + " ".join(b[cut:])


def modification(edit_record):
    """The edit, by field: "Removed '…'", "Added '…'" or "Changed '…' → '…'"."""
    parts = []
    for ch in edit_record["changes"]:
        before = [x for x in _side(ch["before"]) if x.strip()]
        after = [x for x in _side(ch["after"]) if x.strip()]
        removed = [x for x in before if x not in after]
        added = [x for x in after if x not in before]
        field = ch["field"]
        if len(removed) == 1 and len(added) == 1:
            old, new = _focus(removed[0], added[0])
            parts.append(f"{field}: Changed '{_clip(old)}' → '{_clip(new)}'")
            continue
        actions = [f"{verb} {_and([_quoted(x) for x in items])}"
                   for verb, items in (("Removed", removed), ("Added", added)) if items]
        if actions:
            parts.append(f"{field}: {', '.join(actions)}")
    return "; ".join(parts)


def _quoted(text):
    return "'" + _clip(text) + "'"


def _and(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def patient(case):
    demo = case.get("demographics") or {}
    age = demo.get("age")
    who = f"{age}-year-old {SEX.get(demo.get('sex'), demo.get('sex') or 'patient')}" if age is not None \
        else f"Age not recorded, {SEX.get(demo.get('sex'), demo.get('sex') or 'patient')}"
    return f"{who}: {case.get('chief_complaint') or 'chief complaint not recorded'}"
