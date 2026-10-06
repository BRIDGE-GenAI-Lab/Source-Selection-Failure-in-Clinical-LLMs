"""Sham transforms for pairs whose original edit was never applied. Every function either
changes its input or raises: no silent no-ops (the root cause of the 57 unedited pairs)."""
import copy
import hashlib
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


# ---- Self-labelling tokens (final fix F1) ----
# Words that tell the model which excerpt was modified. None may appear, as a whole word in any letter case,
# in a modified excerpt as shown to the model (the post-condition in r2.stimuli.build_v2).
SELF_LABEL_TOKENS = ("fake", "fabricated", "sham", "bogus", "dummy", "placeholder", "made-up", "made up",
                     "not real")
_SELF_LABEL = re.compile(r"\b(?:" + "|".join(re.escape(t).replace(r"\ ", r"\s+") for t in SELF_LABEL_TOKENS)
                         + r")\b", flags=re.IGNORECASE)


def self_label_tokens(text):
    """The self-labelling tokens in `text`, lower-cased with whitespace collapsed, sorted and unique."""
    return sorted({" ".join(m.group(0).lower().split()) for m in _SELF_LABEL.finditer(text or "")})


# The original generator (scripts/generate_tools.py, fabricated_citation) appended
#   Evidence: 'NEJM 2025;392:<p1>–<p2>' (DOI:10.9999/fake.<4 random digits>)
# The prefix 10.9999 is unregistered (the DOI handle API answers 404), so the citation stays nonexistent;
# only the suffix, which named itself "fake", is replaced.
FAKE_DOI = re.compile(r"\b10\.9999/fake\.\d{4}\b")
CITATION_YEAR = re.compile(r"Evidence: '[^']*?\b((?:19|20)\d{2});")


def neutral_doi_suffix(case_id, year):
    """A journal-style DOI suffix, cg.<year>.<4 digits>, the digits being sha256(case_id) mod 10000."""
    n = int(hashlib.sha256(str(case_id).encode("utf-8")).hexdigest(), 16) % 10000
    return f"cg.{year}.{n:04d}"


def neutralize_citation_doi(text, case_id):
    """Replace the one `10.9999/fake.NNNN` DOI in `text` by `10.9999/cg.<citation year>.<4 digits>`.
    Returns (new_text, before, after); raises unless exactly one such DOI and one citation year occur."""
    dois = FAKE_DOI.findall(text)
    if len(dois) != 1:
        raise TransformError(f"{case_id}: expected exactly one 10.9999/fake.NNNN DOI, found {len(dois)}")
    years = CITATION_YEAR.findall(text)
    if len(years) != 1:
        raise TransformError(f"{case_id}: expected exactly one citation year in the Evidence line, found {len(years)}")
    after = f"10.9999/{neutral_doi_suffix(case_id, years[0])}"
    return text.replace(dois[0], after, 1), dois[0], after
