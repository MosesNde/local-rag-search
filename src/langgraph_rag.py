"""Read-only LangGraph integration pilot for the existing Local RAG Search project.

Place this file beside ``query_documents.py`` and ``langgraph_router_pilot.py``
inside ``src/``. No Streamlit or SQLite changes are required.

Run a mocked, no-Ollama/no-ChromaDB test:
    python src/langgraph_rag.py --self-test

Run one actual read-only Ollama+ChromaDB topic-switch test:
    python src/langgraph_rag.py --live-test

Test an Ollama-powered follow-up using the real answer from the prior test:
    python src/langgraph_rag.py --live-followup-test

Test a real Ollama recall using earlier chat messages (no DB changes):
    python src/langgraph_rag.py --live-recall-test

Later, call run_question(question, history, summary, recall_history, chat_id)
from the app. This pilot does not itself persist or condense memory.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable, Literal, TypedDict

from langgraph.graph import START, END, StateGraph
from langgraph_router_pilot import identify_intent

Route = Literal["new_topic", "follow_up", "recall"]
Answer = tuple[str, list[str]]


class RAGState(TypedDict, total=False):
    question: str
    history: list[dict[str, Any]]
    summary: str
    recall_history: list[dict[str, Any]]
    chat_id: str
    route: Route
    explanation: str
    answer: str
    sources: list[str]


def _last_user_question(history: list[dict[str, Any]]) -> str:
    """Only history prior to the new question belongs here."""
    for msg in reversed(history):
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            return msg["content"]
    return ""


def route_question(state: RAGState) -> dict[str, Any]:
    question = state["question"].strip()
    if not question:
        raise ValueError("Question must not be empty")
    decision = identify_intent({
        "question": question,
        "recent_user_question": _last_user_question(state.get("history") or []),
    })
    return {"route": decision["route"], "explanation": decision["explanation"]}


def build_graph(backend: Any | None = None):
    """Compile a 3-branch workflow using an injectable existing RAG backend.

    The backend exposes answer_question(...) and answer_conversation(...).
    A fake backend can be injected for tests; otherwise the project's existing
    query_documents module is loaded only when a question is actually answered.
    """
    if backend is None:
        import query_documents as backend

    def new_topic(state: RAGState) -> dict[str, Any]:
        # Deliberately DO NOT pass old chat context to an independent question.
        answer, sources = backend.answer_question(state["question"])
        return {"answer": answer, "sources": sources}

    def follow_up(state: RAGState) -> dict[str, Any]:
        answer, sources = backend.answer_conversation(
            state["question"], state.get("history") or [],
            summary=state.get("summary") or "",
        )
        return {"answer": answer, "sources": sources}

    def recall(state: RAGState) -> dict[str, Any]:
        # Original chat messages take priority; a summary is not a fact source.
        answer, sources = backend.answer_conversation(
            state["question"], state.get("history") or [],
            summary=state.get("summary") or "",
            original_recall_history=(state.get("recall_history")
                                     or state.get("history") or []),
        )
        return {"answer": answer, "sources": sources}

    graph = StateGraph(RAGState)
    graph.add_node("route", route_question)
    graph.add_node("new_topic", new_topic)
    graph.add_node("follow_up", follow_up)
    graph.add_node("recall", recall)
    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route", lambda state: state["route"],
        {"new_topic": "new_topic", "follow_up": "follow_up", "recall": "recall"},
    )
    for branch in ("new_topic", "follow_up", "recall"):
        graph.add_edge(branch, END)
    return graph.compile()


def run_question(question: str, history: list[dict[str, Any]] | None = None,
                 summary: str = "", recall_history: list[dict[str, Any]] | None = None,
                 chat_id: str = "", backend: Any | None = None) -> RAGState:
    """Run one question. No DB writes, summarization, or history edits."""
    return build_graph(backend).invoke({
        "question": question,
        "history": list(history or []),
        "summary": summary,
        "recall_history": list(recall_history or history or []),
        "chat_id": chat_id,
    })


def run_conversation_payload(payload: dict, backend: Any | None = None) -> dict:
    """Streamlit-compatible JSON contract, read-only until the caller stores the summary.

    The original message archive remains in SQLite. The condensed summary is
    updated with the existing engine, but LangGraph chooses the answer route.
    """
    import query_documents as engine

    question = str(payload.get("question", "")).strip()
    if not question:
        raise ValueError("question must not be empty")
    history = payload.get("history", [])
    if not isinstance(history, list):
        raise ValueError("history must be a list")
    history = [m for m in history if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
    summary = str(payload.get("summary", ""))[:engine.MAX_SUMMARY_CHARS]
    checkpoint = int(payload.get("covered_messages", 0))
    if checkpoint < 0:
        raise ValueError("Invalid memory checkpoint")

    # Preserve exactly the existing condensation policy and checkpoint logic.
    updated_summary, consumed = engine.condense_older_history(history, summary, checkpoint)
    recent = history[consumed:]
    recall_history = payload.get("recall_history")
    if not isinstance(recall_history, list):
        recall_history = list(history)
    recall_history = [m for m in recall_history if isinstance(m, dict)
                      and m.get("role") in ("user", "assistant")]

    # Source of truth for recall is the original chat, never the condensed summary.
    result = run_question(question, history=recent, summary=updated_summary,
                          recall_history=recall_history,
                          chat_id=str(payload.get("chat_id", "")), backend=backend)
    return {"answer": result["answer"], "sources": result.get("sources", []),
            "summary": updated_summary, "covered_messages": checkpoint + consumed,
            "route": result["route"]}


def conversation_json_main() -> None:
    """Called by the existing Streamlit subprocess; stdout is one JSON object."""
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        response = run_conversation_payload(payload)
        print(json.dumps(response, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        raise SystemExit(1)


class FakeBackend:
    """Fake responses let us test routing without loading actual models or DBs."""
    def answer_question(self, question: str) -> Answer:
        if "vector database" in question.casefold():
            return ("A vector database finds semantically similar passages for RAG.",
                    ["[1] rag_explanation.txt (chunk 3)"])
        return ("Independent document question: " + question, [])

    def answer_conversation(self, question: str, history: list[dict[str, Any]],
                            summary: str = "", original_recall_history=None) -> Answer:
        if original_recall_history is not None:
            return ("Recall from original messages in this chat.", [])
        return ("Follow-up using recent chat context.", [])


def self_test() -> None:
    backend = FakeBackend()
    agentchaos_history = [
        {"role": "user", "content": "What is a crash fault in AgentChaos?"},
        {"role": "assistant", "content": "A crash fault interrupts an API call."},
    ]
    cases: list[tuple[str, str, list[dict[str, Any]]]] = [
        ("New topic after AgentChaos", "How does a vector database help retrieval-augmented generation?",
         agentchaos_history),
        ("Explicit new example", "Can you give me a practical example of vector similarity search?",
         agentchaos_history),
        ("Referential follow-up", "Can you give me a practical example?", agentchaos_history),
        ("Earlier-chat recall", "What was the limitation we discussed earlier about AgentChaos and internal reasoning failures?",
         agentchaos_history),
    ]
    expected = ["new_topic", "new_topic", "follow_up", "recall"]
    passed = 0
    for (label, question, history), wanted in zip(cases, expected):
        result = run_question(question, history, chat_id="dry-run-only", backend=backend)
        ok = result["route"] == wanted
        if label == "New topic after AgentChaos":
            ok = ok and "vector database" in result["answer"].lower()
            ok = ok and "AgentChaos" not in result["answer"]
            ok = ok and result["sources"] == ["[1] rag_explanation.txt (chunk 3)"]
        passed += bool(ok)
        print(f"{'PASS' if ok else 'FAIL'} - {label}: {result['route']}")
        print(f"  Answer: {result['answer']}")
    print(f"Result: {passed}/{len(cases)} passed (mocked backend; no live retrieval)")
    if passed != len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversation-json", action="store_true",
                        help="Read Streamlit conversation JSON on stdin")
    parser.add_argument("--self-test", action="store_true", help="Run safe mocked routing tests")
    parser.add_argument("--live-test", action="store_true",
                        help="Run one real Ollama+ChromaDB topic-switch query (read-only)")
    parser.add_argument("--live-followup-test", action="store_true",
                        help="Test a real Ollama follow-up using previous verified answer (read-only)")
    parser.add_argument("--live-recall-test", action="store_true",
                        help="Test Ollama recall from older original messages (read-only)")
    args = parser.parse_args()
    if args.conversation_json:
        conversation_json_main()
    elif args.self_test:
        self_test()
    elif args.live_test:
        context = [
            {"role": "user", "content": "What is a crash fault in AgentChaos?"},
            {"role": "assistant", "content": "It simulates a failed LLM API interaction."},
        ]
        result = run_question(
            "How does a vector database help retrieval-augmented generation?",
            history=context, chat_id="pilot-read-only",
        )
        print("Route:", result["route"])
        print("Answer:", result["answer"])
        print("Sources:")
        for source in result.get("sources", []):
            print(" -", source)
        if result["route"] != "new_topic":
            raise SystemExit("FAIL: independent topic was not routed correctly")
    elif args.live_followup_test:
        # This is the *actual* answer and citations supplied by the user in
        # the preceding --live-test. Reuse it instead of rerunning retrieval.
        # The first two messages provide a realistic earlier topic change.
        context = [
            {"role": "user", "content": "What is a crash fault in AgentChaos?"},
            {"role": "assistant", "content": "It simulates a failed LLM API interaction."},
            {"role": "user", "content": "How does a vector database help retrieval-augmented generation?"},
            {"role": "assistant", "content": (
                "The vector database helps retrieval-augmented generation by searching "
                "for chunks that are semantically similar to the question. [1] [2] "
                "The retrieved chunks are then provided to the language model as "
                "context, enabling it to generate an answer based on the relevant information."
            ), "sources": [
                "[1] rag_explanation.txt (chunk 0)",
                "[2] rag_explanation.txt (chunk 3)",
            ]},
        ]
        result = run_question("Can you give me a practical example?",
                              history=context, chat_id="pilot-read-only")
        print("Route:", result["route"])
        print("Answer:", result["answer"])
        print("Sources:")
        for source in result.get("sources", []):
            print(" -", source)
        if result["route"] != "follow_up":
            raise SystemExit("FAIL: follow-up did not route to conversation context")
        if "vector" not in result["answer"].casefold() and "retriev" not in result["answer"].casefold():
            print("REVIEW: Answer does not explicitly mention the vector/RAG retrieval mechanism.")
        if "AgentChaos" in result["answer"] or "Fibonacci" in result["answer"]:
            print("REVIEW: Answer appears to contain an unrelated prior topic.")
    elif args.live_recall_test:
        # Construct an isolated, synthetic chat. The important limitation is
        # established early, then displaced from the recent message window.
        # The condensed summary intentionally mentions only OWASP: this
        # reproduces our earlier summary-omission failure.
        early_discussion = [
            {"role": "user", "content": (
                "What potential limitation might AgentChaos have when evaluating "
                "failures entirely within an AI agent's reasoning or planning?"
            )},
            {"role": "assistant", "content": (
                "We discussed a possible architectural limitation: AgentChaos "
                "injects faults at LLM API/HTTP interactions, so it may not directly "
                "exercise errors occurring entirely inside an agent's own reasoning "
                "or planning. This is our inference, not a limitation the paper "
                "necessarily established experimentally."
            ), "sources": ["Retrieved context (citation unverified): agentchaos.pdf (page 11, chunk 49)"]},
        ]
        other_topics = [
            ("What is the purpose of an OWASP vulnerability dataset?",
             "It supplies examples categorized by software vulnerability type."),
            ("How does a vector database help RAG?",
             "It retrieves semantically similar chunks to support a response."),
            ("What is the difference between keyword and semantic search?",
             "Keyword search matches terms; semantic search seeks related meaning."),
            ("What is a false positive in vulnerability detection?",
             "A safe example incorrectly classified as vulnerable."),
            ("What does an execution trace record?",
             "It can record a sequence of actions and their outcomes."),
            ("How can we evaluate a machine-learning classifier?",
             "We can study precision, recall, and other evaluation measures."),
            ("What is the OWASP Top Ten?",
             "It categorizes important web application security risks."),
        ]
        history = list(early_discussion)
        for user_question, assistant_answer in other_topics:
            history += [
                {"role": "user", "content": user_question},
                {"role": "assistant", "content": assistant_answer},
            ]
        # Mimic a bounded recent-turn context and a lossy older summary.
        # The complete original history is passed only to the recall branch.
        recent = history[-8:]
        incomplete_summary = (
            "Earlier conversation discussed OWASP Top Ten categories and "
            "vulnerability detection. No AgentChaos details were retained "
            "in this summary."
        )
        question = (
            "What was the limitation we discussed earlier about AgentChaos "
            "and internal reasoning failures?"
        )
        result = run_question(
            question,
            history=recent,
            summary=incomplete_summary,
            recall_history=history,
            chat_id="pilot-read-only",
        )
        print("Route:", result["route"])
        print("Answer:", result["answer"])
        print("Sources:")
        for src in result.get("sources", []):
            print(" -", src)
        if result["route"] != "recall":
            raise SystemExit("FAIL: earlier-chat question was not routed to recall")
        answer_lower = result["answer"].casefold()
        # Avoid claiming success when the model returns an empty or unrelated answer.
        if not ("agentchaos" in answer_lower
                and ("reason" in answer_lower or "planning" in answer_lower)
                and ("api" in answer_lower or "http" in answer_lower)):
            raise SystemExit("FAIL: did not recall the AgentChaos reasoning limitation")
        if result.get("sources"):
            print("REVIEW: chat recall should not promote earlier references to verified citations")
        print("PASS: earlier discussion recalled despite topic changes and lossy summary")
    else:
        parser.print_help()
