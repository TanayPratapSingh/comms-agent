"""Run the comms-agent end to end on the sample data.

    python scripts/run_agent.py
"""
import sys
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


if __name__ == "__main__":
    main()
