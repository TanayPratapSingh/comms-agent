"""LLM abstraction.

The agent depends only on this small interface, so the default MockLLM (no
API key, deterministic) can be swapped for a real provider by implementing
`complete()`. `summarize()` turns aggregated stats into a short narrative; the
mock builds it from a template so the whole pipeline is reproducible offline.
"""
from __future__ import annotations
from typing import Protocol


class LLM(Protocol):
    def complete(self, prompt: str) -> str: ...
    def summarize(self, stats: dict) -> str: ...


class MockLLM:
    """Deterministic stand-in for a hosted model."""

    name = "mock-llm"

    def complete(self, prompt: str) -> str:
        return f"[mock-llm response to {len(prompt)}-char prompt]"

    def plan(self, brand: str) -> list[str]:
        return [
            f"ingest recent mentions of {brand}",
            "classify each mention (sentiment + topic) with a confidence",
            "route low-confidence mentions to human review",
            "aggregate communications metrics",
            "draft a weekly summary for the comms team",
        ]

    def summarize(self, stats: dict) -> str:
        s = stats
        top_topic = max(s["topic_mix"], key=s["topic_mix"].get) if s["topic_mix"] else "n/a"
        pos = s["sentiment_mix"].get("positive", 0)
        neg = s["sentiment_mix"].get("negative", 0)
        tone = "net positive" if pos > neg else "net negative" if neg > pos else "mixed"
        msgs = ", ".join(f"{k} ({v:.0%})" for k, v in list(s["message_pull_through"].items())[:3]) or "none detected"
        return (
            f"Weekly communications summary for {s['brand']}\n"
            f"{s['n']} mentions analyzed. Coverage tone was {tone} "
            f"({pos} positive, {neg} negative, {s['sentiment_mix'].get('neutral', 0)} neutral). "
            f"The dominant storyline was {top_topic}. "
            f"Key messages landing in coverage: {msgs}. "
            f"{s['flagged']} mentions were low-confidence and sent to human review, "
            f"of which {s['corrected']} were corrected before reporting."
        )


class OpenAILLM:  # pragma: no cover - requires api key + package
    """Example real adapter. Implement complete(); summarize() is inherited in
    spirit via the same prompt contract. Not exercised in tests."""

    def __init__(self, model: str = "gpt-4o-mini", client=None):
        self.model = model
        self.client = client

    def complete(self, prompt: str) -> str:
        raise NotImplementedError("Wire up your OpenAI client here.")

    def summarize(self, stats: dict) -> str:
        return self.complete("Summarize these comms metrics as a short brief:\n" + str(stats))
