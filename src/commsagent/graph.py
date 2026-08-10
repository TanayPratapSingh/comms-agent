"""The communications-monitoring agent, assembled as a state graph.

Flow:
    plan -> ingest -> classify -> review_gate ->? human_review -> aggregate -> report

`review_gate` is a conditional edge: if any mention came back low-confidence it
routes to `human_review`, otherwise it skips straight to `aggregate`. This is
the human-in-the-loop quality gate, expressed as agent control flow.
"""
from __future__ import annotations
from collections import Counter, defaultdict

from .orchestrator import StateGraph, State, END
from .tools import build_registry
from .llm import MockLLM


def build_comms_graph(llm=None, brand: str = "Northwind", threshold: float = 0.50,
                      data_path: str | None = None) -> StateGraph:
    llm = llm or MockLLM()
    tools = build_registry(brand)

    def plan(s: State) -> State:
        s.data["plan"] = llm.plan(brand) if hasattr(llm, "plan") else []
        s.data["brand"] = brand
        s.log(f"planned {len(s.data['plan'])} steps")
        return s

    def ingest(s: State) -> State:
        articles = tools["fetch_articles"](data_path)
        s.record_tool("fetch_articles", f"{len(articles)} mentions")
        s.data["articles"] = articles
        return s

    def classify(s: State) -> State:
        preds = []
        for a in s.data["articles"]:
            p = tools["classify_article"](a["text"])
            p["id"] = a["id"]
            p["relevance"] = tools["score_relevance"](a)
            preds.append(p)
        s.record_tool("classify_article", f"classified {len(preds)} mentions")
        s.data["preds"] = preds
        s.data["low_conf"] = [p for p in preds if tools["needs_review"](p, threshold)]
        s.log(f"{len(s.data['low_conf'])} low-confidence mentions")
        return s

    def human_review(s: State) -> State:
        # oracle-backed review: the labeled sample provides ground truth, so the
        # demo can measure real correction. In production this is a UI queue.
        truth_topic = {a["id"]: a.get("topic_true") for a in s.data["articles"]}
        truth_sent = {a["id"]: a.get("sentiment_true") for a in s.data["articles"]}
        corrected = 0
        audit = []
        for p in s.data["low_conf"]:
            for field_, truth in (("topic", truth_topic), ("sentiment", truth_sent)):
                t = truth.get(p["id"])
                if t and t != p[field_]:
                    audit.append({"id": p["id"], "field": field_,
                                  "from": p[field_], "to": t, "reviewer": "analyst"})
                    p[field_] = t
                    corrected += 1
        s.data["corrected"] = corrected
        s.data["audit"] = audit
        s.log(f"human corrected {corrected} labels")
        return s

    def review_gate(s: State) -> str:
        return "human_review" if s.data["low_conf"] else "aggregate"

    def aggregate(s: State) -> State:
        preds = s.data["preds"]
        articles = {a["id"]: a for a in s.data["articles"]}
        sov = defaultdict(float)
        for a in s.data["articles"]:
            sov[brand] += a.get("reach_weight", 0.5)
            if a.get("mentions_competitor"):
                sov[a["mentions_competitor"]] += a.get("reach_weight", 0.5)
        tot = sum(sov.values()) or 1.0
        n = len(preds) or 1
        msg = Counter(m for p in preds for m in p["messages"])
        s.data["stats"] = {
            "brand": brand, "n": len(preds),
            "sentiment_mix": dict(Counter(p["sentiment"] for p in preds)),
            "topic_mix": dict(Counter(p["topic"] for p in preds).most_common()),
            "share_of_voice": {k: round(v / tot, 3) for k, v in sov.items()},
            "message_pull_through": {k: round(c / n, 3) for k, c in msg.most_common()},
            "flagged": len(s.data["low_conf"]),
            "corrected": s.data.get("corrected", 0),
        }
        return s

    def report(s: State) -> State:
        s.data["report"] = llm.summarize(s.data["stats"])
        s.record_tool("llm.summarize", f"{len(s.data['report'])} chars")
        return s

    g = StateGraph()
    g.add_node("plan", plan).add_node("ingest", ingest).add_node("classify", classify)
    g.add_node("human_review", human_review).add_node("aggregate", aggregate).add_node("report", report)
    g.set_entry("plan")
    g.add_edge("plan", "ingest").add_edge("ingest", "classify")
    g.add_conditional_edge("classify", review_gate)
    g.add_edge("human_review", "aggregate").add_edge("aggregate", "report").add_edge("report", END)
    return g
