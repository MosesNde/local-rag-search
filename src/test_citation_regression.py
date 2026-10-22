"""Offline tests for source attribution and answer completion.

Put this file in src/ and run from the project root:
    python src/test_citation_regression.py

Mocks all Ollama chat responses; never accesses the ChromaDB database.
Designed against query_documents.py at GitHub commit c78d0c7.
A compound citation test currently exposes an attribution weakness; do not suppress it.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import query_documents as rag


RAG = "rag_explanation.txt"
PDF = "agentchaos.pdf"


def record(source, chunk, text, file_type="txt", page=None):
    meta = {"source": source, "chunk_number": chunk, "file_type": file_type}
    if page is not None:
        meta["page"] = page
    return text, meta


class CitationTests(unittest.TestCase):
    def annotate(self, answer, records):
        # Production uses these same records and metadata objects.
        return rag.annotate_answer_with_evidence(
            answer, records, [meta for _, meta in records]
        )

    def test_three_distinct_text_citations(self):
        records = [
            record(RAG, 3, "The vector database searches for chunks that are semantically similar to the question."),
            record(RAG, 4, "The retrieved chunks are then provided to the language model as context."),
            record(RAG, 5, "The language model uses that context to generate an answer."),
        ]
        answer = (
            "The vector database searches for chunks that are semantically similar to the question. "
            "The retrieved chunks are then provided to the language model as context. "
            "The language model uses that context to generate an answer."
        )
        annotated, sources = self.annotate(answer, records)
        self.assertEqual(len(sources), 3)
        self.assertIn("[1] rag_explanation.txt (chunk 3)", sources)
        self.assertIn("[2] rag_explanation.txt (chunk 4)", sources)
        self.assertIn("[3] rag_explanation.txt (chunk 5)", sources)
        self.assertIn("an answer. [3]", annotated)

    def test_overlapping_chunk_prefers_original(self):
        records = [
            record(RAG, 4, "The vector database searches for chunks that are semantically similar to the question. The retrieved chunks are then provided to the language model as context."),
            record(RAG, 3, "When a user asks a question, the question is converted into an embedding. The vector database searches for chunks that are semantically similar to the question."),
        ]
        answer = "The vector database searches for chunks that are semantically similar to the question."
        _, sources = self.annotate(answer, records)
        self.assertEqual(sources, ["[1] rag_explanation.txt (chunk 3)"])

    def test_different_documents_get_different_references(self):
        records = [
            record(RAG, 0, "Retrieval-augmented generation combines information retrieval with large language models."),
            record(PDF, 25, "AgentChaos is a framework for chaos engineering in agent systems through programmatic fault injection.", "pdf", 1),
        ]
        answer = (
            "Retrieval-augmented generation combines information retrieval with large language models. "
            "AgentChaos is a framework for chaos engineering in agent systems through programmatic fault injection."
        )
        _, sources = self.annotate(answer, records)
        self.assertEqual(len(sources), 2)
        self.assertIn(RAG, sources[0])
        self.assertIn("chunk 0", sources[0])
        self.assertIn(PDF, sources[1])
        self.assertIn("page 1", sources[1])

    def test_compound_sentence_uses_multiple_passages(self):
        records = [
            record(RAG, 2, "Each chunk can be converted into an embedding and stored in a vector database. When a user asks a question, the question is converted into an embedding."),
            record(RAG, 3, "The vector database searches for chunks that are semantically similar to the question."),
            record(RAG, 4, "The retrieved chunks are then provided to the language model as context."),
        ]
        answer = (
            "When a user asks a question, it is converted into an embedding, the vector database searches "
            "for chunks that are semantically similar to the question, and the retrieved chunks are provided "
            "to the language model as context."
        )
        annotated, sources = self.annotate(answer, records)
        self.assertGreaterEqual(len(sources), 2, (annotated, sources))
        self.assertTrue(all(RAG in s for s in sources))
        self.assertIn("[1]", annotated)

    def test_unrelated_claim_not_given_citation(self):
        records = [
            record(RAG, 3, "The vector database searches for chunks that are semantically similar to the question."),
        ]
        answer = "The vector database secretly transmits users' bank passwords to remote attackers."
        annotated, sources = self.annotate(answer, records)
        self.assertEqual(sources, [])
        self.assertNotIn("[1]", annotated)

    def test_exact_numeric_mismatch_not_cited(self):
        records = [
            record(PDF, 10, "The experiment reported 49.66 for the selected metric.", "pdf", 7),
        ]
        answer = "The experiment reported 99.99 for the selected metric."
        annotated, sources = self.annotate(answer, records)
        self.assertEqual(sources, [])
        self.assertNotIn("[1]", annotated)

    def test_table_and_figure_metadata_are_preserved(self):
        table = rag.format_source({"source": "paper.pdf", "file_type": "pdf_table",
                                   "table_label": "Table II", "page": 6, "row_number": 1})
        figure = rag.format_source({"source": "agentchaos.pdf", "file_type": "pdf_figure",
                                    "figure_number": "3", "page": 5})
        self.assertIn("Table II, page 6, row 1", table)
        self.assertIn("Figure 3, page 5", figure)


class FakeOllamaResponse(dict):
    """Mimic Ollama ChatResponse: dict-style data + attribute access."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


class CompletionTests(unittest.TestCase):
    @staticmethod
    def response(content, reason="stop", tokens=24):
        return FakeOllamaResponse(
            message={"content": content}, done_reason=reason,
            eval_count=tokens, load_duration=0,
            prompt_eval_duration=0, eval_duration=0, prompt_eval_count=10,
        )

    def test_multi_part_detection(self):
        self.assertTrue(rag.is_multi_part_question(
            "What is retrieval-augmented generation, and what does AgentChaos do?"
        ))
        self.assertFalse(rag.is_multi_part_question("What is retrieval-augmented generation?"))

    def test_truncated_answer_retried(self):
        replies = [
            self.response("AgentChaos helps improve robustness", "length", 256),
            self.response("AgentChaos helps improve robustness by testing agent systems.", "stop", 31),
        ]
        with patch.object(rag.ollama, "chat", side_effect=replies) as chat:
            answer = rag.generate_grounded_answer(
                "What is RAG, and what does AgentChaos do?", "Grounded context."
            )
        self.assertEqual(chat.call_count, 2)
        self.assertTrue(answer.endswith("systems."))
        self.assertEqual(chat.call_args_list[0].kwargs["options"]["num_predict"], 256)
        self.assertGreater(chat.call_args_list[1].kwargs["options"]["num_predict"], 256)

    def test_permanently_truncated_answer_rejected(self):
        replies = [
            self.response("The unfinished explanation", "length", 256),
            self.response("Still an unfinished explanation", "length", 512),
        ]
        with patch.object(rag.ollama, "chat", side_effect=replies) as chat:
            answer = rag.generate_grounded_answer(
                "What is RAG, and how does AgentChaos work?", "Grounded context."
            )
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(answer, "INSUFFICIENT_INFORMATION")

    def test_completed_response_not_retried(self):
        with patch.object(
            rag.ollama, "chat",
            return_value=self.response("The answer is complete.", "stop", 17)
        ) as chat:
            answer = rag.generate_grounded_answer("What is RAG?", "Relevant context.")
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(answer, "The answer is complete.")


if __name__ == "__main__":
    print("Offline-only citation and completion tests. No Ollama calls or database writes.",
          flush=True)
    unittest.main(verbosity=2)
