"""Run the comms-agent end to end on the sample data.

    python scripts/run_agent.py [runs/<utc>-demo.json]   # optional path writes the review counts as JSON
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from commsagent.graph import build_comms_graph
from commsagent.orchestrator import State


def main():
    graph = build_comms_graph(threshold=0.50)
    state = graph.run(State())

    print("PLAN")
    for i, step in enumerate(state.data["plan"], 1):
        print(f"  {i}. {step}")

    print("\nTOOL CALLS")
    for c in state.tool_calls:
        print(f"  {c['tool']:18s} {c['detail']}")

    print("\nHUMAN-IN-THE-LOOP")
    print(f"  flagged low-confidence: {state.data['stats']['flagged']}")
    print(f"  corrected by human:     {state.data['stats']['corrected']}")

    print("\nDRAFT REPORT")
    print("  " + state.data["report"].replace("\n", "\n  "))

    print("\nEXECUTION TRACE")
    print("  " + " -> ".join(t.split(":")[1] for t in state.trace if t.startswith("enter:")))

    if len(sys.argv) > 1:
        # unflagged predictions skip review, so any mismatch with the labels reaches the report
        flagged = {p["id"] for p in state.data["low_conf"]}
        arts = {a["id"]: a for a in state.data["articles"]}
        unflagged_wrong = [p["id"] for p in state.data["preds"] if p["id"] not in flagged and any(
            arts[p["id"]].get(f + "_true") not in (None, p[f]) for f in ("topic", "sentiment"))]
        Path(sys.argv[1]).write_text(json.dumps({
            "command": "python " + " ".join(sys.argv),
            "llm": "MockLLM",  # build_comms_graph default; labels come from the rule based classify_article tool
            "mentions": len(state.data["preds"]),
            "flagged": len(flagged),
            "flagged_mentions_corrected": len({a["id"] for a in state.data["audit"]}),
            "label_fixes": state.data["stats"]["corrected"],
            "unflagged": len(arts) - len(flagged),
            "unflagged_mislabelled_reported_unreviewed": len(unflagged_wrong),
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
