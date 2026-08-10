"""Tests for comms-agent. Runs under pytest or standalone."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from commsagent.graph import build_comms_graph
from commsagent.orchestrator import State, StateGraph, END
from commsagent.tools import classify_article, build_registry

STATE = build_comms_graph(threshold=0.50).run(State())


def test_graph_reaches_report():
    assert "report" in STATE.data and STATE.data["report"]
    assert STATE.trace[-1] == "enter:__end__"


def test_all_mentions_classified():
    assert len(STATE.data["preds"]) == len(STATE.data["articles"])
    for p in STATE.data["preds"]:
        assert p["sentiment"] in {"positive", "negative", "neutral"}
        assert 0.0 <= p["confidence"] <= 1.0


def test_human_gate_triggered_and_logged():
    # sample data contains ambiguous items, so review must have run
    assert STATE.data["stats"]["flagged"] > 0
    assert "human_review" in [t.split(":")[1] for t in STATE.trace if t.startswith("enter:")]
    assert STATE.data["corrected"] == len(STATE.data["audit"])


def test_tool_calls_recorded():
    names = {c["tool"] for c in STATE.tool_calls}
    assert {"fetch_articles", "classify_article", "llm.summarize"} <= names


def test_share_of_voice_normalized():
    sov = STATE.data["stats"]["share_of_voice"]
    assert abs(sum(sov.values()) - 1.0) < 1e-6


def test_classify_tool_is_deterministic():
    a = classify_article("Northwind launches new Aurora X1 with AI-powered features")
    b = classify_article("Northwind launches new Aurora X1 with AI-powered features")
    assert a == b and a["topic"] == "Product Launch"


def test_conditional_edge_skips_review_when_confident():
    # a graph with a trivially high threshold-free path: build a tiny graph
    g = StateGraph()
    g.add_node("a", lambda s: s).add_node("b", lambda s: s)
    g.set_entry("a").add_conditional_edge("a", lambda s: "b").add_edge("b", END)
    out = g.run(State())
    assert [t for t in out.trace if t.startswith("enter:")] == ["enter:a", "enter:b", "enter:__end__"]


def test_max_steps_guard():
    g = StateGraph(max_steps=5)
    g.add_node("loop", lambda s: s).set_entry("loop").add_edge("loop", "loop")
    out = g.run(State())
    assert any("ABORT" in t for t in out.trace)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")
