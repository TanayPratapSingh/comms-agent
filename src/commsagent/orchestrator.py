"""A small but real state-graph engine for agent orchestration.

It captures the ideas that matter in production agent frameworks (LangGraph and
similar) without the dependency weight:

  * named nodes that read and write a shared mutable State
  * fixed edges and conditional edges (a router picks the next node)
  * per-node retries with a max-step guard against infinite loops
  * a full execution trace (nodes visited, tool calls, errors)

Swapping this for LangGraph later is mechanical: nodes become graph nodes and
conditional edges become `add_conditional_edges`.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

END = "__end__"


@dataclass
class State:
    data: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)

    def log(self, msg: str):
        self.trace.append(msg)

    def record_tool(self, name: str, summary: str):
        self.tool_calls.append({"tool": name, "detail": summary})


class StateGraph:
    def __init__(self, max_steps: int = 50):
        self.nodes: dict[str, Callable[[State], State]] = {}
        self.edges: dict[str, str] = {}
        self.cond: dict[str, Callable[[State], str]] = {}
        self.entry: str | None = None
        self.max_steps = max_steps

    def add_node(self, name: str, fn: Callable[[State], State]):
        self.nodes[name] = fn
        return self

    def add_edge(self, src: str, dst: str):
        self.edges[src] = dst
        return self

    def add_conditional_edge(self, src: str, router: Callable[[State], str]):
        self.cond[src] = router
        return self

    def set_entry(self, name: str):
        self.entry = name
        return self

    def _next(self, node: str, state: State) -> str:
        if node in self.cond:
            return self.cond[node](state)
        return self.edges.get(node, END)

    def run(self, state: State | None = None, retries: int = 1) -> State:
        if self.entry is None:
            raise ValueError("entry node not set")
        state = state or State()
        node = self.entry
        steps = 0
        while node != END:
            if steps >= self.max_steps:
                state.log(f"ABORT: exceeded max_steps={self.max_steps}")
                break
            fn = self.nodes[node]
            attempt = 0
            while True:
                try:
                    state.log(f"enter:{node}")
                    state = fn(state)
                    break
                except Exception as e:  # retry transient node failures
                    attempt += 1
                    state.log(f"error:{node}:{e} (attempt {attempt})")
                    if attempt > retries:
                        state.log(f"FAIL:{node}")
                        raise
            node = self._next(node, state)
            steps += 1
        state.log("enter:__end__")
        return state
