"""Read-only regression checks for the existing Local RAG Search Engine.

Install ONLY this file in src/test_rag_core_regression.py.
Run from project root: python src/test_rag_core_regression.py

Tests routing, mocked LangGraph dispatch, append-only memory, and UI provenance
without invoking Ollama, ChromaDB, Streamlit, or modifying SQLite.
"""
from __future__ import annotations

import ast
import re
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parent


def extract_functions(filename, names, namespace):
    """Load only the named pure/helper functions; don't import service dependencies."""
    tree = ast.parse((SRC / filename).read_text(encoding='utf-8'), filename=filename)
    found = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    actual = {node.name for node in found}
    if actual != set(names):
        raise AssertionError(f'{filename}: missing expected functions {set(names) - actual}')
    minimal = ast.Module(body=found, type_ignores=[])
    exec(compile(ast.fix_missing_locations(minimal), str(SRC / filename), 'exec'), namespace)
    return namespace


class DummyOllama:
    def __init__(self):
        self.notes = []
        self.fail = False

    def chat(self, **kwargs):
        if self.fail:
            raise RuntimeError('Simulated unavailable Ollama')
        prompt = kwargs['messages'][0]['content']
        self.notes.append(prompt)
        if 'AgentChaos' in prompt:
            note = 'AgentChaos was discussed; its API-layer fault injection may not exercise failures internal to agent planning.'
        elif 'vector database' in prompt:
            note = 'Vector databases retrieve semantically similar text passages for retrieval augmented generation.'
        else:
            note = 'The conversation discussed a new topic and a possible follow-up question.'
        return {'message': {'content': note}}


class TestRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import langgraph_rag
        except ImportError as exc:
            raise unittest.SkipTest(f'LangGraph is not installed: {exc}')
        cls.graph_module = langgraph_rag

    def test_all_routes_using_fake_backend(self):
        class FakeBackend:
            calls = []
            def answer_question(self, question):
                self.calls.append(('new_topic', question))
                return 'Doc answer [1]', ['[1] example.txt (chunk 0)']
            def answer_conversation(self, question, history, **kwargs):
                self.calls.append(('conversation', question, kwargs))
                return 'Conversation answer', []
        fake = FakeBackend()
        graph = self.graph_module.build_graph(backend=fake)
        history = [{'role':'user', 'content':'Tell me about AgentChaos'},
                   {'role':'assistant','content':'AgentChaos injects API faults.'}]
        direct = graph.invoke({'question':'How does a vector database help retrieval-augmented generation?', 'history':history})
        self.assertEqual(direct['route'], 'new_topic')
        self.assertEqual(direct['sources'], ['[1] example.txt (chunk 0)'])
        self.assertEqual(fake.calls[-1][0], 'new_topic')
        follow = graph.invoke({'question':'Can you explain that in simple terms?', 'history':history, 'summary':'old summary'})
        self.assertEqual(follow['route'], 'follow_up')
        self.assertEqual(fake.calls[-1][0], 'conversation')
        self.assertEqual(fake.calls[-1][2]['summary'], 'old summary')
        full = history + [{'role':'user', 'content':'Earlier important detail'}]
        recall = graph.invoke({'question':'What was the limitation we discussed earlier about AgentChaos?',
                               'history':history[-1:], 'recall_history':full, 'summary':'partial'})
        self.assertEqual(recall['route'], 'recall')
        self.assertEqual(fake.calls[-1][2]['original_recall_history'], full)


class TestCumulativeMemory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ollama = DummyOllama()
        cls.ns = extract_functions('query_documents.py', ['condense_older_history'], {
            're':re, 'ollama':cls.ollama, 'ANSWER_MODEL':'fake',
            'RECENT_VERBATIM_MESSAGES':12, 'SUMMARY_BATCH_MESSAGES':8,
            'MAX_SUMMARY_CHARS':10000, 'MAX_BATCH_NOTE_CHARS':1500,
            'profile_log': lambda _: None,
        })

    def make_history(self, count=18):
        topics = ['What is the purpose of fault injection in AgentChaos?',
                  'What potential internal reasoning limitation might AgentChaos have?',
                  'How does a vector database help RAG?',
                  'What is an embedding?', 'What is OWASP?', 'What is access control?',
                  'What is SAST?', 'What is a code property graph?', 'What is CFG?']
        result=[]
        for idx in range(count):
            if idx % 2 == 0:
                result.append({'role':'user','content':topics[(idx//2)%len(topics)]})
            else:
                result.append({'role':'assistant','content':'A grounded response to that question.'})
        return result

    def test_early_threshold_does_not_advance(self):
        summary, consumed=self.ns['condense_older_history'](self.make_history(12))
        self.assertEqual((summary,consumed), ('',0))

    def test_append_only_segments_and_source_positions(self):
        history=self.make_history(20)
        first, count=self.ns['condense_older_history'](history[:14], '', 0)
        self.assertEqual(count,2)
        self.assertIn('[Chat messages 1-2]',first)
        second, new_count=self.ns['condense_older_history'](history[2:16],first,2)
        self.assertEqual(new_count,2)
        self.assertTrue(second.startswith(first))
        self.assertIn('[Chat messages 3-4]',second)
        self.assertIn('internal reasoning limitation',second)
        third, extra=self.ns['condense_older_history'](history[4:18],second,4)
        self.assertEqual(extra,2)
        self.assertTrue(third.startswith(second))
        self.assertIn('[Chat messages 5-6]',third)
        self.assertIn('vector database',third)

    def test_model_failure_keeps_watermark(self):
        self.ollama.fail=True
        try:
            summary,count=self.ns['condense_older_history'](self.make_history(14),'previous stable segment',2)
        finally:
            self.ollama.fail=False
        self.assertEqual((summary,count),('previous stable segment',0))

    def test_capacity_limit_preserves_earlier_memory(self):
        self.ns['MAX_SUMMARY_CHARS']=100
        try:
            summary,count=self.ns['condense_older_history'](self.make_history(14),'old segments',2)
            self.assertEqual((summary,count),('old segments',0))
        finally:
            self.ns['MAX_SUMMARY_CHARS']=10000


class TestProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.labels=staticmethod(extract_functions('web_app.py',['answer_provenance_labels'], {'re':re})['answer_provenance_labels'])

    def test_recall_not_document_evidence(self):
        got=self.labels('**From our earlier discussion in this chat:** AgentChaos might miss an internal failure.',[], 'What did we discuss earlier?')
        self.assertTrue(any('Conversation recall' in x for x in got))
        self.assertTrue(any('No independent document verification' in x for x in got))

    def test_verified_passage_label(self):
        got=self.labels('Documented statement. [1]',['[1] agentchaos.pdf (page 3)'],'What is AgentChaos?')
        self.assertTrue(any('Document citations matched' in x for x in got))

    def test_hypothetical_and_unverified_are_independent(self):
        got=self.labels('A fault might cause a tool call to fail.',
                        ['Retrieved context (citation unverified): agentchaos.pdf'],
                        'Give me a hypothetical example of how AgentChaos injects a tool-call fault')
        self.assertTrue(any('Hypothetical example' in x for x in got))
        self.assertTrue(any('Claim-level citation unverified' in x for x in got))

    def test_inference_separate_from_evidence(self):
        got=self.labels('A vulnerability could occur.',[], 'What can we infer about the system?')
        self.assertTrue(any('Reasoned inference' in x for x in got))


class TestExampleRequest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wants=staticmethod(extract_functions('query_documents.py',['wants_concrete_example'],{'re':re})['wants_concrete_example'])

    def test_example_requests_recognized(self):
        self.assertTrue(self.wants('Can you give me a practical example of vector similarity search?'))
        self.assertTrue(self.wants('Provide a hypothetical scenario using AgentChaos.'))
        self.assertFalse(self.wants('What is a vector database?'))


if __name__ == '__main__':
    print('Local RAG regression suite — read-only, no live Ollama/ChromaDB/SQLite access\n')
    unittest.main(verbosity=2)
