"""Self-labelling tokens: words that tell the model which excerpt was modified (r2.audit.self_label_tokens)."""
import re

# None may appear, as a whole word in any letter case, in a modified excerpt as shown to the model.
SELF_LABEL_TOKENS = ("fake", "fabricated", "sham", "bogus", "dummy", "placeholder", "made-up", "made up",
                     "not real")
_SELF_LABEL = re.compile(r"\b(?:" + "|".join(re.escape(t).replace(r"\ ", r"\s+") for t in SELF_LABEL_TOKENS)
                         + r")\b", flags=re.IGNORECASE)


def self_label_tokens(text):
    """The self-labelling tokens in `text`, lower-cased with whitespace collapsed, sorted and unique."""
    return sorted({" ".join(m.group(0).lower().split()) for m in _SELF_LABEL.finditer(text or "")})
