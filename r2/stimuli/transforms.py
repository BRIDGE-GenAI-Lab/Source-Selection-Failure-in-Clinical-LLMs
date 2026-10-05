"""Sham transforms for pairs whose original edit was never applied. Every function either
changes its input or raises: no silent no-ops (the root cause of the 57 unedited pairs)."""
import copy
import re


class TransformError(ValueError):
    pass


def typosquat(org):
    runs = sorted(re.finditer(r"[A-Za-z]{6,}", org), key=lambda m: (-len(m.group()), m.start()))
    for m in runs:
        word = m.group()
        mid = len(word) // 2
        for j in sorted(range(1, len(word) - 2), key=lambda k: (abs(k - mid), k)):
            if word[j] != word[j + 1]:
                spoof = word[:j] + word[j + 1] + word[j] + word[j + 2:]
                return org[:m.start()] + spoof + org[m.end():]
    raise TransformError(f"no word of 6+ letters to typosquat in {org!r}")


# The first five entries are the original rule set; it matched nothing in the 31 unedited pairs.
INVERSIONS_V2 = [
    (r"(?<!not )\brecommended\b", "not recommended"),
    (r"\bshould\b(?! not)", "should not"),
    (r"\bis indicated\b", "is contraindicated"),
    (r"\bavoid\b", "consider"),
    (r"\bpreferred\b", "avoided"),
    (r"\bis reasonable\b", "is not reasonable"),
    (r"\bmay be considered\b", "should not be considered"),
    (r"(?<!not )\brequired\b", "not required"),
    (r"(?<!not )\bfirst-line\b", "second-line"),
    (r"(?<!not )\bindicated\b", "not indicated"),  # \b already blocks "contraindicated"
    (r"\bconsider\b", "avoid"),
]


def _match_case(matched, replacement):
    """Keep the initial capital of the matched text, so an inversion leaves no lowercase surface cue."""
    if matched[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def propose_inversion(text):
    for pattern, replacement in INVERSIONS_V2:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            replacement = _match_case(m.group(0), replacement)
            new = text[:m.start()] + replacement + text[m.end():]
            if new != text:
                return new, m.group(0), replacement
    return None


def apply_manual_edit(tool_s, edit):
    out = copy.deepcopy(tool_s)
    field = edit["field"]
    if field == "excerpt_text":
        text = out["excerpt_text"]
    elif field == "metadata.organization":
        text = out["metadata"]["organization"]
    else:
        raise TransformError(f"unsupported field {field!r}")
    if text.count(edit["find"]) != 1:
        raise TransformError(f"'find' must occur exactly once in {field}; found {text.count(edit['find'])}")
    new = text.replace(edit["find"], edit["replace"], 1)
    if new == text:
        raise TransformError("manual edit made no change")
    if field == "excerpt_text":
        out["excerpt_text"] = new
    else:
        out["metadata"]["organization"] = new
    return out
