"""Stage 1 LangGraph pilot: test chat question routing without changing the RAG app.

Run from project root in your Windows virtualenv:
    python src/langgraph_router_pilot.py

Dependency: pip install langgraph
No Ollama calls, database writes, or network requests are made by this script.
"""
from __future__ import annotations

import re
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

Route = Literal["new_topic", "follow_up", "recall"]


class RouterState(TypedDict, total=False):
    question: str
    recent_user_question: str
    route: Route
    explanation: str
    search_scope: str


RECALL_RE = re.compile(
    r"\b(?:we discussed|we concluded|did we discuss|earlier (?:in|during) (?:this|our) chat|"
    r"remember (?:when|what|the)|what was the limitation we discussed|"
    r"what did we (?:say|conclude|decide)|summari[sz]e what we have discussed)\b",
    re.I,
)

# Questions with a clear new subject should not be rewritten against an old one.
EXPLICIT_SUBJECT_RE = re.compile(
    r"\b(?:AgentChaos|OWASP|vector database|vector similarity search|"
    r"semantic search|keyword search|retrieval.augmented generation|"
    r"machine.learning model|software fault|vulnerability dataset)\b",
    re.I,
)

FOLLOWUP_RE = re.compile(
    r"\b(?:it|that|this|these|those|such|them|the earlier (?:answer|example)|"
    r"can you (?:explain|simplify|give me|show me)|give me (?:a|an) example)\b",
    re.I,
)


def identify_intent(state: RouterState) -> dict:
    question = state["question"].strip()
    if RECALL_RE.search(question):
        route: Route = "recall"
        why = "The user explicitly asks about an earlier discussion."
    elif EXPLICIT_SUBJECT_RE.search(question):
        route = "new_topic"
        why = "The question names a subject; do not carry over the previous topic."
    elif FOLLOWUP_RE.search(question) and state.get("recent_user_question"):
        route = "follow_up"
        why = "The question refers to the preceding discussion without naming a new subject."
    else:
        route = "new_topic"
        why = "No reliable reference to the preceding conversation was found."
    return {"route": route, "explanation": why}


def choose_branch(state: RouterState) -> Route:
    return state["route"]


def new_topic(state: RouterState) -> dict:
    return {"search_scope": "documents; independently interpret latest question"}


def follow_up(state: RouterState) -> dict:
    return {"search_scope": "recent chat turns; resolve references, then document evidence"}


def recall(state: RouterState) -> dict:
    return {"search_scope": "original messages in active chat; summaries are only an aid"}


def build_graph():
    graph = StateGraph(RouterState)
    graph.add_node("identify_intent", identify_intent)
    graph.add_node("new_topic", new_topic)
    graph.add_node("follow_up", follow_up)
    graph.add_node("recall", recall)
    graph.add_edge(START, "identify_intent")
    graph.add_conditional_edges(
        "identify_intent", choose_branch,
        {"new_topic": "new_topic", "follow_up": "follow_up", "recall": "recall"},
    )
    for route in ("new_topic", "follow_up", "recall"):
        graph.add_edge(route, END)
    return graph.compile()


TEST_CASES = [
    (
        "How does a vector database help retrieval-augmented generation?",
        "What is the difference between a crash fault and an omission fault in AgentChaos?",
        "new_topic",
    ),
    (
        "Can you give me a practical example of vector similarity search?",
        "How does a vector database help retrieval-augmented generation?",
        "new_topic",
    ),
    (
        "Can you explain that in simple terms?",
        "How does a vector database help retrieval-augmented generation?",
        "follow_up",
    ),
    (
        "Can you give me a practical example?",
        "How does a vector database help retrieval-augmented generation?",
        "follow_up",
    ),
    (
        "What was the limitation we discussed earlier about AgentChaos and internal reasoning failures?",
        "What is the difference between a vulnerability and a software fault?",
        "recall",
    ),
    (
        "Can you summarize what we have discussed about vulnerability evaluation?",
        "What is the difference between a vulnerability and a software fault?",
        "recall",
    ),
]


def main():
    router = build_graph()
    failed = 0
    print("LangGraph routing pilot — no database modifications\n")
    for number, (question, previous, expected) in enumerate(TEST_CASES, start=1):
        result = router.invoke({"question": question, "recent_user_question": previous})
        actual = result["route"]
        ok = actual == expected
        failed += not ok
        print(f"{number}. {'PASS' if ok else 'FAIL'} — {actual}")
        print(f"   {question}")
        print(f"   Next: {result['search_scope']}\n")
    print(f"Result: {len(TEST_CASES) - failed}/{len(TEST_CASES)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
