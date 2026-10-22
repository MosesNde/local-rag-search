"""Read-only focused regression tests for explicit citations."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import query_documents as r

PDF='owasp.pdf'
OWASP=[('Broken Access Control','Python, C, Java, JavaScript'),('Cryptographic Failures','Java, JavaScript, Python, C++'),('Injection','PHP, JavaScript, Python, Java')]
def records():
    return [(f'Row {i}: OWASP Category: {category} | Most Affected Languages: {languages} | Description of Vulnerabilities: ...', {'source':PDF,'file_type':'pdf_table','table_label':'Table II','page':6,'row_number':i}) for i,(category,languages) in enumerate(OWASP,1)]
def cite(question, answer, rows):
    return r.cite_explicit_answer(question,answer,rows,[meta for _,meta in rows])

class StructuredCitationTests(unittest.TestCase):
    def test_real_indexed_record_with_source_headers(self):
        doc = ('Source: A_Comprehensive_Software_Vulnerability_Dataset_Based_on_OWASP_Top_Ten_Standard.pdf\n'
               'Page: 6\nTable: Table II\n'
               'Caption: DISTRIBUTION OF SOFTWARE VULNERABILITIES ACROSS OWASP CATEGORIES\n'
               'Row 1: OWASP Category: Broken Access Control | Most Affected Languages: Python, C, Java, JavaScript | '
               'Description of Vulnerabilities: Enterprise applications have complex access control')
        meta = {'source': 'A_Comprehensive_Software_Vulnerability_Dataset_Based_on_OWASP_Top_Ten_Standard.pdf',
                'file_type': 'pdf_table', 'table_label': 'Table II', 'page': 6, 'row_number': 1}
        answer, sources = cite('According to Table II of the OWASP vulnerability dataset paper, '
                               'which programming languages are most affected by Broken Access Control?',
                               'The programming languages are Python, C, Java, and JavaScript.', [(doc, meta)])
        self.assertIn('[1]', answer)
        self.assertEqual(len(sources), 1)
        self.assertIn('Table II, page 6, row 1', sources[0])

    def test_exact_broken_access_control_row(self):
        answer,sources=cite('According to Table II, what languages are affected by Broken Access Control?','Python, C, Java, JavaScript.',records())
        self.assertIn('[1]',answer);self.assertEqual(len(sources),1);self.assertIn('row 1',sources[0])
    def test_exact_cryptographic_failures_row(self):
        answer,sources=cite('According to Table II, what languages are affected by Cryptographic Failures?','Java, JavaScript, Python, C++.',records())
        self.assertIn('[1]',answer);self.assertEqual(len(sources),1);self.assertIn('row 2',sources[0])
    def test_wrong_or_missing_language_not_cited(self):
        answer,sources=cite('According to Table II, what languages are affected by Broken Access Control?','Python, C and JavaScript.',records())
        self.assertEqual(sources,[]);self.assertNotIn('[1]',answer)
    def test_whole_table_count_uses_whole_table(self):
        answer,sources=cite('How many categories are listed in Table II?','The table lists 3 categories.',records())
        self.assertIn('[1]',answer);self.assertIn('rows 1-3',sources[0])
    def test_wrong_count_not_cited(self):
        answer,sources=cite('How many categories are listed in Table II?','The table lists 8 categories.',records())
        self.assertEqual(sources,[])
    def test_figure_exact(self):
        rows=[('The framework receives an agent system, tasks, and a fault configuration as inputs.',{'source':'agentchaos.pdf','file_type':'pdf_figure','figure_number':3,'page':5})]
        answer,sources=cite('According to Figure 3 of agentchaos.pdf, what inputs are provided?','The inputs include an agent system, tasks, and a fault configuration.',rows)
        self.assertIn('[1]',answer);self.assertIn('Figure 3, page 5',sources[0])
    def test_extremum(self):
        rows=[]
        for i,(system,n) in enumerate([('MapCoder','49.66'),('EvoMAC','26.21'),('AutoGen','22.86')],1):
            doc=f'Row {i}: Model: GPT-5.2 | System: {system} | Dataset: HumanEval+ | delta pass@1 (Δ): {n}'
            rows.append((doc,{'source':'agentchaos.pdf','file_type':'pdf_table','table_number':3,'row_number':i,'page':7}))
        a,s=cite('According to Table 3 in agentchaos.pdf, which GPT-5.2 system on HumanEval+ has the highest reported delta pass@1?','MapCoder has the highest reported delta pass@1 of 49.66.',rows)
        self.assertIn('[1]',a);self.assertEqual(len(s),1);self.assertIn('row 1',s[0])
if __name__=='__main__':unittest.main(verbosity=2)
