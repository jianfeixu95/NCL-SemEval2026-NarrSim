import re
from typing import List, Dict, Optional

_token_re = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

def _tokens(text: str) -> set:
    return set(_token_re.findall(text.lower()))

def jaccard_similarity(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def mine_hard_negative(
        anchor: str,
        negative_examples: List[Dict],
        *,
        prefer_contradiction: bool = True,
) -> Optional[Dict]:
    """
    Select a hard negative for DocNLI-style triplet construction.

    Parameters
    ----------
    anchor : str
        The premise / anchor document.
    negative_examples : List[Dict]
        Each dict must contain:
          - "hypothesis": str
          - "label": "neutral" or "contradiction"
    prefer_contradiction : bool
        If True, contradiction candidates are prioritized when overlap is similar.

    Returns
    -------
    Dict or None
        The selected hard negative example (original dict).
    """

    if not negative_examples:
        return None

    anchor_tokens = _tokens(anchor)

    best = None
    best_score = -1.0

    for ex in negative_examples:
        hyp = ex.get("hypothesis", "")
        label = ex.get("label", "").lower()

        if not hyp:
            continue

        hyp_tokens = _tokens(hyp)
        overlap = jaccard_similarity(anchor_tokens, hyp_tokens)

        # contradiction bonus (optional)
        if prefer_contradiction and label == "contradiction":
            overlap += 0.05  # small bias, do not dominate lexical similarity

        if overlap > best_score:
            best_score = overlap
            best = ex

    return best
