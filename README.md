# comms-agent

An agent that automates the communications-monitoring loop: it plans the work, ingests brand mentions, classifies each one, routes the low-confidence tail to a human reviewer through a conditional gate, aggregates the metrics, and drafts a weekly summary. It is built on a small state-graph orchestrator so the control flow is explicit and auditable.

This is the automation layer that sits on top of a monitoring pipeline like [media-pulse](../media-pulse). Where media-pulse is the measurement system, comms-agent is the thing that runs it on a schedule and produces the brief.

## Why this exists

It targets three qualifications that come up together in AI and communications roles: agent orchestration (planning, tool-calling, stateful control flow), AI-driven automation that combines model outputs with human-in-the-loop quality assurance, and responsible-AI practice through an explicit confidence gate and a full audit trail. It runs with zero third-party dependencies so the orchestration itself is the focus.

## The agent

```
plan -> ingest -> classify -> [review_gate] --low-confidence--> human_review -> aggregate -> report
                                     |                                              ^
                                     +-------------- all confident ----------------+
```

The planner asks the LLM for a step plan. Ingest and classify call tools and record every invocation. `review_gate` is a conditional edge: if any mention came back below the confidence threshold it routes to `human_review`, otherwise it skips straight to aggregation. Human review corrects the flagged labels (oracle-backed in the demo, a UI queue in production) and logs each change. Aggregate computes the comms metrics, and report asks the LLM to phrase them as a short brief.

On the sample data the agent flags 9 of 24 mentions as low-confidence and a reviewer corrects 7 of them before anything is reported, so the numbers the comms team sees are the human-verified ones.

## The orchestrator

`orchestrator.py` is a compact state-graph engine with the pieces that matter in production agent frameworks: named nodes over a shared mutable state, fixed and conditional edges, per-node retries, a max-step guard against infinite loops, and a complete execution trace. It is deliberately framework-free; porting the same graph to LangGraph is a mechanical change, and the README points at where an LLM or tool backend swaps in.

## Layout

```
src/commsagent/
  llm.py           LLM interface, deterministic MockLLM, optional real adapter
  tools.py         fetch / classify / relevance / needs-review tools + registry
  orchestrator.py  the state-graph engine (nodes, edges, retries, trace)
  graph.py         the comms-monitoring agent assembled as a graph
scripts/run_agent.py   end to end run with plan, tool trace, report, trace
tests/test_agent.py    graph, gate, tools, and engine tests
data/sample_articles.json
```

## Run it

```bash
python scripts/run_agent.py        # runs the agent, prints plan + report + trace
python tests/test_agent.py         # or: pytest -q
```

No installation required; the core runs on the standard library. To replace the MockLLM with a hosted model, implement `complete()` on a class in `llm.py` (an `OpenAILLM` skeleton is included) and pass it to `build_comms_graph(llm=...)`. Nothing else changes, because the agent depends only on the LLM interface, not the provider.

## Note on the LLM

The default `MockLLM` is deterministic so runs and tests are reproducible offline. It produces the plan from a template and the report by filling the aggregated stats into a narrative. Swapping in a real model upgrades the wording, not the pipeline.
