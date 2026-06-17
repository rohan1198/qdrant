"""
QueryIntentDetector — rule-based routing between precision and recall modes.

Precision mode (Poincaré): MRR=0.97, P@10=0.76  — best single answer
Recall mode   (cosine):    MRR=0.42, P@10=0.63  — many relevant docs

Signals used
------------
1. Alpha (conformal factor):  α = 1 / (1 - c‖x‖²)
   High α → query lands near the ball boundary → leaf/specific content → precision
   Low  α → query lands near the ball centre  → broad/general topic  → recall

2. Text heuristics:
   - Named-entity proxies: tokens that look like proper nouns (title-cased or
     all-caps), version strings, quoted phrases, numeric references.
   - Specificity words: vocabulary that signals a narrow, fact-finding intent
     ("how", "what exactly", "define", "who is", "when did", etc.)
   - Breadth words: vocabulary that signals an exploratory/survey intent
     ("overview", "survey", "list", "compare", "all", "examples of", etc.)

The "auto" mode combines both signals via a weighted score.
Alpha carries the dominant weight (0.6) because it directly encodes where in
the Poincaré hierarchy the query lands; text heuristics add 0.4.

Returned intent string
-----------------------
  "precision"  → use Poincaré pipeline with content-only tier filter
  "recall"     → use cosine pipeline with broad tier filter
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Thresholds and weights (tune via offline calibration if needed)
# ---------------------------------------------------------------------------

# Alpha of a query vector at radius r with curvature c:  α = 1/(1 - c·r²)
# Content tier: r≈0.90, c=1 → α ≈ 1/(1-0.81) ≈ 5.26
# Story tier:   r≈0.70, c=1 → α ≈ 1/(1-0.49) ≈ 1.96
# Narrative:    r≈0.45, c=1 → α ≈ 1/(1-0.20) ≈ 1.25
# Theme:        r≈0.15, c=1 → α ≈ 1/(1-0.02) ≈ 1.02
# We use α ≥ 3.0 as the "near the boundary" cutoff (between story and content).
ALPHA_PRECISION_THRESHOLD = 3.0   # α above this → lean precision
ALPHA_RECALL_THRESHOLD    = 1.6   # α below this → lean recall
# Between the two thresholds the text signal breaks the tie.

# Combination weights  (must sum to 1.0)
# NOTE: For PCA-encoded queries (L2-normalised SentenceTransformer output), all query
# vectors are pinned to the content tier radius (r=0.90) → α≈5.26 for every query.
# The alpha signal therefore cannot differentiate specific vs broad queries at encode
# time, so we rely entirely on text heuristics for routing.  The alpha signal becomes
# useful again when queries are embedded by a model that produces naturally varying
# norms (e.g., a fine-tuned hyperbolic encoder), at which point ALPHA_WEIGHT can be
# increased back to 0.6.
ALPHA_WEIGHT = 0.0
TEXT_WEIGHT  = 1.0

# Score ≥ this → precision; < this → recall  (in the [0,1] combined scale)
# With TEXT_WEIGHT=1.0 this directly thresholds the text heuristic score:
#   text_score ≥ 0.55  → net specificity ≥ +0.3 hits  → precision
#   text_score < 0.55  → breadth-leaning or neutral   → recall
COMBINED_PRECISION_CUTOFF = 0.55


# ---------------------------------------------------------------------------
# Text-signal patterns
# ---------------------------------------------------------------------------

# Specificity markers → push toward precision
_SPECIFICITY_PATTERNS = [
    r'\b(who is|who was|what is|what was|when did|where did|how does|how do)\b',
    r'\b(define|definition of|meaning of|explain|describe)\b',
    r'\b(exactly|specifically|precisely|in particular)\b',
    r'\b(the (?:first|last|only|original|best|worst))\b',
    r'"\w[\w\s]*"',                    # quoted phrase
    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',   # multi-word proper noun (Title Case)
    r'\b[A-Z]{2,}\b',                  # acronym / all-caps token
    r'\bv?\d+\.\d+(?:\.\d+)?\b',       # version number
    r'\b\d{4}\b',                       # year
]
_SPECIFICITY_RE = re.compile('|'.join(_SPECIFICITY_PATTERNS), re.IGNORECASE)

# Breadth markers → push toward recall
_BREADTH_PATTERNS = [
    r'\b(overview|survey|introduction|summary|tutorial|guide)\b',
    r'\b(all|every|various|different|multiple|many|several|list of)\b',
    r'\b(compare|comparison|contrast|versus|vs\.?)\b',
    r'\b(examples? of|kinds? of|types? of|categories? of)\b',
    r'\b(what are|which are|tell me about|show me)\b',
    r'\b(landscape|ecosystem|space|field|domain|area)\b',
]
_BREADTH_RE = re.compile('|'.join(_BREADTH_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------

class QueryIntentDetector:
    """
    Detects query intent and recommends a retrieval mode.

    Parameters
    ----------
    curvature:
        Poincaré ball curvature c (must match the one used during encoding).
    alpha_precision_threshold / alpha_recall_threshold:
        Override the default alpha cutoffs if you have calibrated them from
        your own corpus.
    """

    def __init__(
        self,
        curvature: float = 1.0,
        alpha_precision_threshold: float = ALPHA_PRECISION_THRESHOLD,
        alpha_recall_threshold: float = ALPHA_RECALL_THRESHOLD,
    ):
        self.curvature = curvature
        self.alpha_precision_threshold = alpha_precision_threshold
        self.alpha_recall_threshold = alpha_recall_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        query_text: str,
        query_poincare: np.ndarray,
        mode: str = "auto",
    ) -> str:
        """
        Return "precision" or "recall".

        Parameters
        ----------
        query_text:
            Raw query string.
        query_poincare:
            128-d Poincaré vector for the query (1-d or 2-d with batch=1).
        mode:
            "auto"      — combine alpha + text signals (default)
            "alpha"     — use only the Poincaré alpha signal
            "text"      — use only text heuristics
            "precision" — always return "precision"  (pass-through)
            "recall"    — always return "recall"     (pass-through)
        """
        if mode in ("precision", "recall"):
            return mode

        alpha_score = self._alpha_signal(query_poincare)   # 0=recall, 1=precision
        text_score  = self._text_signal(query_text)        # 0=recall, 1=precision

        if mode == "alpha":
            combined = alpha_score
        elif mode == "text":
            combined = text_score
        else:  # "auto"
            combined = ALPHA_WEIGHT * alpha_score + TEXT_WEIGHT * text_score

        intent = "precision" if combined >= COMBINED_PRECISION_CUTOFF else "recall"
        return intent

    # ------------------------------------------------------------------
    # Internal signal computation
    # ------------------------------------------------------------------

    def _alpha_signal(self, query_poincare: np.ndarray) -> float:
        """
        Map the query's conformal factor α to a [0, 1] precision score.

        α is computed as 1 / (1 - c‖x‖²).  Higher α → closer to ball
        boundary → more specific / leaf content → precision.
        """
        vec = np.asarray(query_poincare, dtype=np.float64).ravel()
        norm_sq = float(np.dot(vec, vec))
        alpha = 1.0 / max(1.0 - self.curvature * norm_sq, 1e-6)

        if alpha >= self.alpha_precision_threshold:
            return 1.0
        if alpha <= self.alpha_recall_threshold:
            return 0.0
        # Linear interpolation between the two thresholds
        span = self.alpha_precision_threshold - self.alpha_recall_threshold
        return (alpha - self.alpha_recall_threshold) / span

    def _text_signal(self, query_text: str) -> float:
        """
        Score the query text on a [0, 1] scale.

        Counts weighted hits of specificity markers (+1 each) vs breadth
        markers (-1 each), then maps to [0, 1] via a soft sigmoid-like clip.
        """
        text = query_text.strip()
        specificity_hits = len(_SPECIFICITY_RE.findall(text))
        breadth_hits     = len(_BREADTH_RE.findall(text))

        raw_score = specificity_hits - breadth_hits   # signed integer

        # Saturate: ±3 hits is decisive
        clipped = max(-3, min(3, raw_score))

        # Map [-3, 3] → [0, 1]
        return (clipped + 3) / 6.0

    # ------------------------------------------------------------------
    # Convenience: return tier_filter and pipeline name together
    # ------------------------------------------------------------------

    def routing(
        self,
        query_text: str,
        query_poincare: np.ndarray,
        mode: str = "auto",
    ) -> dict:
        """
        Return a dict with all routing decisions for this query.

        Keys
        ----
        intent         : "precision" | "recall"
        tier_filter    : "content"   | "content_or_story"
        pipeline       : "alpha"     | "cosine"
        alpha          : raw conformal factor (diagnostic)
        text_score     : raw text signal 0-1 (diagnostic)
        alpha_score    : raw alpha signal 0-1 (diagnostic)
        """
        intent = self.detect(query_text, query_poincare, mode=mode)

        vec = np.asarray(query_poincare, dtype=np.float64).ravel()
        norm_sq = float(np.dot(vec, vec))
        raw_alpha = 1.0 / max(1.0 - self.curvature * norm_sq, 1e-6)

        return {
            "intent":      intent,
            "tier_filter": "content"          if intent == "precision" else "content_or_story",
            "pipeline":    "alpha"            if intent == "precision" else "cosine",
            "alpha":       raw_alpha,
            "text_score":  self._text_signal(query_text),
            "alpha_score": self._alpha_signal(query_poincare),
        }


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

_default_detector: Optional[QueryIntentDetector] = None


def detect_intent(
    query_text: str,
    query_poincare: np.ndarray,
    mode: str = "auto",
    curvature: float = 1.0,
) -> str:
    """
    Stateless convenience wrapper.  Returns "precision" or "recall".

    Uses a module-level cached detector (reset if curvature changes).
    """
    global _default_detector
    if _default_detector is None or _default_detector.curvature != curvature:
        _default_detector = QueryIntentDetector(curvature=curvature)
    return _default_detector.detect(query_text, query_poincare, mode=mode)
