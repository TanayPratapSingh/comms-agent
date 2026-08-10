"""Tools the agent can call.

Each tool has a name, a description, and a callable, mirroring the tool-calling
pattern used with function-calling LLMs. The orchestrator records every
invocation so the run produces an auditable trace of what the agent did.
"""
from __future__ import annotations
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_TOKEN = re.compile(r"[a-z0-9]+")

KEY_MESSAGES = ["AI-powered", "privacy-first", "carbon neutral by 2030"]

# lightweight keyword signals per topic (self-contained, no external model)
_TOPIC_SIGNALS = {
    "Product Launch": ["unveils", "launches", "launch", "new", "reveal", "first look"],
    "Earnings": ["earnings", "revenue", "beats", "misses", "estimates", "quarter", "margins", "guidance"],
    "Leadership": ["ceo", "executive", "names", "appointment", "chief", "departs", "summit"],
    "Partnership": ["partners", "partnership", "deal", "integration", "alliance", "ecosystem"],
    "Sustainability": ["carbon", "solar", "green", "sustainability", "neutral", "environmental"],
    "Controversy": ["backlash", "regulators", "inquiry", "sued", "lawsuit", "litigation", "data handling"],
    "Layoffs": ["layoffs", "workforce", "job cuts", "restructuring", "cuts", "morale"],
    "Reviews": ["review", "hands-on", "upgrade", "reviewer", "impresses", "disappointed"],
}

_POS = {"strong", "beats", "beat", "praised", "impresses", "solid", "welcome",
        "upgrade", "standout", "wonderful", "fantastic", "great", "good",
        "growth", "expands", "renewed", "broaden", "profitability"}
_NEG = {"backlash", "sued", "inquiry", "cuts", "layoffs", "dips", "misses",
        "miss", "slows", "pressure", "thinning", "vague", "questions",
        "disappointed", "quirks", "departs", "departure", "critics", "risk"}


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable

    def __call__(self, *a, **k):
        return self.fn(*a, **k)


def fetch_articles(path: str | None = None) -> list[dict]:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data" / "sample_articles.json"
    return json.loads(Path(path).read_text())


def classify_article(text: str) -> dict:
    low = text.lower()
    # topic via keyword scoring + temperature-scaled softmax for a confidence.
    # The temperature (T>1) sharpens the distribution so a clear single-topic
    # article reads as high-confidence, while genuinely dual-topic articles,
    # which score on two labels at once, stay low-confidence and get reviewed.
    T = 1.8
    scores = {t: sum(low.count(w) for w in words) for t, words in _TOPIC_SIGNALS.items()}
    total = sum(math.exp(T * v) for v in scores.values())
    label = max(scores, key=scores.get)
    conf = math.exp(T * scores[label]) / total if total else 0.0
    if max(scores.values()) == 0:
        conf = 0.0
    # sentiment via lexicon
    toks = _tokens(text)
    sval = sum(1 for w in toks if w in _POS) - sum(1 for w in toks if w in _NEG)
    sentiment = "positive" if sval > 0 else "negative" if sval < 0 else "neutral"
    s_conf = min(1.0, abs(sval) / 2.0)  # one clear tone word -> 0.5, two -> 1.0
    messages = [m for m in KEY_MESSAGES if m.lower() in low]
    return {
        "topic": label, "topic_conf": round(conf, 3),
        "sentiment": sentiment, "sentiment_conf": round(s_conf, 3),
        "messages": messages,
        "confidence": round(min(conf, s_conf), 3),
    }


def score_relevance(article: dict, brand: str) -> float:
    hits = article["text"].lower().count(brand.lower())
    return round(min(1.0, 0.4 * min(1.0, hits / 2) + 0.6 * article.get("reach_weight", 0.5)), 3)


def needs_review(prediction: dict, threshold: float) -> bool:
    return prediction["confidence"] < threshold


def build_registry(brand: str) -> dict[str, Tool]:
    return {
        "fetch_articles": Tool("fetch_articles", "Load recent brand mentions.", fetch_articles),
        "classify_article": Tool("classify_article", "Classify one mention for topic and sentiment with confidence.", classify_article),
        "score_relevance": Tool("score_relevance", "Score how relevant a mention is to the brand.", lambda a: score_relevance(a, brand)),
        "needs_review": Tool("needs_review", "Decide if a prediction should go to a human.", needs_review),
    }
