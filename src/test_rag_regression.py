"""Read-only regression checks for the Local RAG Search Engine.

Run from project root:
    python src/test_rag_regression.py
    python src/test_rag_regression.py --live

Default: fast extraction-fixture checks (no ChromaDB or Ollama needed).
--live: additionally execute the real query_documents.py against existing ChromaDB.
Never creates, updates or deletes indexed records.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLE_JSON = ROOT / 'data' / 'table_tests' / 'agentchaos_tables.json'
QUERY_SCRIPT = ROOT / 'src' / 'query_documents.py'
DB_PATH = ROOT / 'data' / 'chroma'
IEEE = 'A_Comprehensive_Software_Vulnerability_Dataset_Based_on_OWASP_Top_Ten_Standard.pdf'


def fetch_fixture():
    if not TABLE_JSON.is_file():
        raise FileNotFoundError(f'Expected extracted-table fixture: {TABLE_JSON}\nRun: python src/pdf_tables.py agentchaos.pdf')
    data = json.loads(TABLE_JSON.read_text(encoding='utf-8'))
    return {str(t['number']): t for t in data['tables']}


def record_rows(table):
    for key in ('structured', 'normalized'):
        block = table.get(key) or {}
        if isinstance(block.get('records'), list):
            return block['records']
    return []


class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = fetch_fixture()

    def test_all_nine_table_numbers_present(self):
        self.assertEqual(set(self.tables), {str(i) for i in range(1, 10)})

    def test_numeric_cells_and_missing_values(self):
        rows = record_rows(self.tables['3'])
        self.assertEqual(len(rows), 420, 'Expected 20 system rows x 7 datasets x 3 metrics')
        subset = [r for r in rows if r.get('model') == 'GPT-5.2'
                  and r.get('dataset') == 'HumanEval+']
        metrics = defaultdict(dict)
        for r in subset:
            metrics[r['system']][r['metric']] = r['value']
        self.assertEqual(len(metrics), 5)
        expected = {'MapCoder': 49.66, 'EvoMAC': 26.21,
                    'AutoGen': 22.86, 'MAD': 21.31}
        delta_key = lambda m: next((k for k in m if k in ('Δ','𝚫','∆','delta')), None)
        values = {}
        for system, values_by_metric in metrics.items():
            key = delta_key(values_by_metric)
            self.assertIsNotNone(key, f'Missing delta column for {system}')
            raw = values_by_metric[key]
            values[system] = None if raw is None else float(raw)
        for system, number in expected.items():
            self.assertAlmostEqual(values[system], number, places=2)
        self.assertIsNone(values['Mini-SE'])
        ranked = sorted(((s, v) for s, v in values.items() if v is not None),
                        key=lambda x: x[1], reverse=True)
        self.assertEqual([s for s, _ in ranked], ['MapCoder','EvoMAC','AutoGen','MAD'])

    def test_taxonomy_uncertainty_not_turned_into_false(self):
        table = self.tables['1']
        rows = record_rows(table)
        self.assertEqual(len(rows), 6)
        # Missing checkmark associations must remain missing, not boolean False.
        for row in rows:
            for value in row.values():
                self.assertIsNot(value, False)


def read_count():
    import chromadb  # only required for --live
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_collection('documents').count()


def ask_live(question, timeout):
    # On Windows, the default subprocess encoding may be cp1252, which
    # cannot encode symbols such as the Greek delta (Δ) in test questions.
    # Configure both the parent pipes and the child process for UTF-8.
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    result = subprocess.run(
        [sys.executable, str(QUERY_SCRIPT)],
        input=question + '\n',
        text=True,
        encoding='utf-8',
        errors='replace',
        capture_output=True,
        cwd=str(ROOT),
        timeout=timeout,
        env=env,
    )
    if result.returncode:
        raise AssertionError(f'query_documents.py returned {result.returncode}:\n{result.stderr[-2500:]}')
    if 'Answer:' not in result.stdout:
        raise AssertionError('Missing Answer: output marker')
    return result.stdout


def answer_segment(output):
    return output.split('Answer:', 1)[-1].split('Retrieved sources:', 1)[0]


class LiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not QUERY_SCRIPT.is_file():
            raise FileNotFoundError(QUERY_SCRIPT)
        cls.before = read_count()

    @classmethod
    def tearDownClass(cls):
        after = read_count()
        if after != cls.before:
            raise AssertionError(f'Database count changed during read-only tests: {cls.before} -> {after}')

    def query(self, question):
        return ask_live(question, timeout=240)

    def test_exact_new_table_lookup(self):
        out = self.query('According to Table 3 in agentchaos.pdf, what is the Δpass@1 for MapCoder on HumanEval+ using GPT-5.2?')
        self.assertRegex(answer_segment(out), r'(?<!\d)49\.66(?!\d)')
        self.assertIn('agentchaos.pdf (Table 3, page 7)', out)

    def test_legacy_ieee_table(self):
        out = self.query(f'According to Table II in {IEEE}, what vulnerability categories are listed?')
        answer = answer_segment(out).casefold()
        for category in ('broken access control', 'injection', 'server-side request forgery'):
            self.assertIn(category, answer)
        self.assertIn('Table II, page 6', out)

    def test_python_sorted_ranking_and_missing_value(self):
        out = self.query('According to Table 3 in agentchaos.pdf, rank all agent systems evaluated on HumanEval+ using GPT-5.2 from highest to lowest Δpass@1, and provide their values.')
        answer = answer_segment(out)
        ranks = re.findall(r'^\s*([1-4])\.\s*(MapCoder|EvoMAC|AutoGen|MAD)\s*:\s*(\d+(?:\.\d+)?)', answer, re.M)
        self.assertEqual([(int(i),s) for i,s,_ in ranks], [(1,'MapCoder'),(2,'EvoMAC'),(3,'AutoGen'),(4,'MAD')])
        for (_, system, actual), expected in zip(ranks, [49.66,26.21,22.86,21.31]):
            self.assertAlmostEqual(float(actual), expected, places=2, msg=system)
        self.assertRegex(answer, r'(?i)Mini-SE.*not reported|not reported.*Mini-SE')
        self.assertIn('agentchaos.pdf (Table 3, page 7)', out)

    def test_general_document_question(self):
        out = self.query('According to agentchaos.pdf, what is the purpose of fault injection in the AgentChaos approach?')
        answer = answer_segment(out)
        self.assertGreater(len(answer.strip()), 30)
        self.assertNotIn("I don't have enough information", answer)
        self.assertIn('agentchaos.pdf', out)


def run_suite(test_case):
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(test_case)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Also test the actual CLI and existing ChromaDB; may call Ollama')
    args = parser.parse_args()
    print('Read-only RAG regression suite. No indexing or DB writes.\n', flush=True)
    passed = run_suite(FixtureTests)
    if args.live and passed:
        print('\nLive tests against query_documents.py and existing ChromaDB:\n', flush=True)
        passed = run_suite(LiveTests) and passed
    elif not args.live:
        print('\nTip: pass --live to test end-to-end retrieval against your database.')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
