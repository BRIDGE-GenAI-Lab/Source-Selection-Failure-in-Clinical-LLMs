"""docs/r2/vignette_generation_prompt.md: a 3-line provenance header, then the prompt byte for byte."""
import hashlib

import pytest

from r2.paths import LEGACY_EVAL, REPO

DOC = REPO / "docs" / "r2" / "vignette_generation_prompt.md"
SOURCE = LEGACY_EVAL / "supplementary_appendix.md" if LEGACY_EVAL else None
FIRST, LAST = 1442, 1542  # 1-based, inclusive
BODY_SHA256 = "6baefa808f25daded201ad9e1d640586759698c3c1243da6591dc8e6c0091159"


def _split():
    lines = DOC.read_bytes().splitlines(keepends=True)
    return b"".join(lines[:3]).decode("utf-8"), b"".join(lines[3:])


def test_header_states_provenance():
    header, _ = _split()
    for phrase in ("Google AI Studio", "Gemini 3 Pro", "zero-shot", "supplementary_appendix.md",
                   "1442-1542"):
        assert phrase in header, phrase
    assert header.count("\n") == 3


def test_body_sha256_is_pinned():
    _, body = _split()
    assert hashlib.sha256(body).hexdigest() == BODY_SHA256


@pytest.mark.skipif(SOURCE is None or not SOURCE.exists(), reason="legacy appendix not mounted (R2_LEGACY_ROOT)")
def test_body_equals_appendix_lines_1442_to_1542():
    _, body = _split()
    src = SOURCE.read_bytes().splitlines(keepends=True)[FIRST - 1:LAST]
    assert len(src) == 101
    assert body == b"".join(src)
