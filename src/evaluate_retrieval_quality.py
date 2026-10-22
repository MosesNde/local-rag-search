"""Read-only evidence retrieval evaluation for local-rag-search.

Run from project root:
    python src/evaluate_retrieval_quality.py
    python src/evaluate_retrieval_quality.py --no-embed

Uses existing ChromaDB and Ollama embeddings. Never calls an answer model,
creates a collection, or modifies any existing database records.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / 'data' / 'chroma'
REPORT_DIR = ROOT / 'data' / 'retrieval_tests'
COLLECTION = 'documents'

try:
    from config import EMBEDDING_MODEL
except ImportError:
    EMBEDDING_MODEL = 'nomic-embed-text'

IEEE = 'A_Comprehensive_Software_Vulnerability_Dataset_Based_on_OWASP_Top_Ten_Standard.pdf'
AGENT = 'agentchaos.pdf'

# A small, deliberately explicit gold set, grounded in the documents
# previously inspected. Extend it with more sources as needed.
TESTS = [
    {'id':'text_rag_definition', 'question':'What is retrieval-augmented generation?',
     'source':'rag_explanation.txt', 'top_k':3, 'expected_kind':'txt',
     'required_facts':{
         'definition':r'retrieval.augmented generation|information retrieval',
         'question_embedding':r'question.{0,80}embedding',
         'vector_search':r'vector database searches|semantically similar',
         'retrieved_context':r'retrieved chunks.{0,90}context',
         'answer_generation':r'language model uses that context|generate an answer',
     }},
    {'id':'agentchaos_table3', 'question':'GPT-5.2 MapCoder HumanEval+ delta pass@1 49.66',
     'source':AGENT, 'expected_kind':'pdf_table', 'table_number':3, 'top_k':5,
     'required_facts':{'value':r'49[.]66','model':r'GPT-5[.]2',
                       'system':r'MapCoder','dataset':r'HumanEval\+'}},
    {'id':'ieee_table_ii', 'question':'OWASP vulnerability distribution in Table II',
     'source':IEEE, 'expected_kind':'pdf_table', 'table_number':2, 'top_k':5,
     'required_facts':{'broken_access_control':r'Broken Access Control'}},
    {'id':'agentchaos_figure3', 'question':'AgentChaos framework architecture Figure 3',
     'source':AGENT, 'expected_kind':'pdf_figure', 'figure_number':3, 'top_k':5,
     'required_facts':{}},
]


def normalize(s):
    return re.sub(r'\s+', ' ', str(s or '')).strip().casefold()


def table_number(meta):
    if meta.get('table_number') is not None:
        try:
            return int(meta['table_number'])
        except (ValueError, TypeError):
            return None
    match = re.search(r'\btable\s+([ivxlcdm]+|\d+)\b', str(meta.get('table_label','')), re.I)
    if not match:
        return None
    raw = match.group(1).upper()
    if raw.isdigit():
        return int(raw)
    values = dict(I=1,V=5,X=10,L=50,C=100,D=500,M=1000)
    total = previous = 0
    for char in reversed(raw):
        value = values[char]
        total += -value if value < previous else value
        previous = value
    return total


def source_matches(meta, test):
    if meta.get('source') != test['source']:
        return False
    if meta.get('file_type') != test['expected_kind']:
        return False
    if 'table_number' in test and table_number(meta) != test['table_number']:
        return False
    if 'figure_number' in test:
        try:
            return int(meta.get('figure_number', -1)) == test['figure_number']
        except (ValueError, TypeError):
            return False
    return True


def examine_duplicates(ids, docs, metas):
    groups = defaultdict(list)
    for id_, doc, meta in zip(ids, docs, metas):
        groups[(meta.get('source'), meta.get('file_type'), normalize(doc))].append(id_)
    exact = [ {'source':key[0], 'file_type':key[1], 'record_count':len(found), 'ids':found[:10]}
              for key,found in groups.items() if len(found)>1 and key[2] ]
    exact.sort(key=lambda item:(-item['record_count'],str(item['source'])))
    figures=defaultdict(list)
    tables=defaultdict(list)
    for id_, meta in zip(ids, metas):
        if meta.get('file_type') == 'pdf_figure':
            figures[(meta.get('source'),meta.get('page'),str(meta.get('figure_number')))].append(id_)
        if meta.get('file_type') == 'pdf_table':
            tables[(meta.get('source'),meta.get('page'),table_number(meta))].append(id_)
    figure_groups=[{'source':s,'page':p,'figure_number':n,'records':len(v),'ids':v[:20]}
                   for (s,p,n),v in figures.items() if len(v)>1]
    table_groups=[{'source':s,'page':p,'table_number':n,'records':len(v)}
                  for (s,p,n),v in tables.items()]
    return {'exact_document_duplicates':exact, 'exact_duplicate_group_count':len(exact),
            'figure_groups_with_multiple_records':figure_groups,
            'table_groups':table_groups,
            'note':'Multiple table rows per table are expected. Multiple figures with the same identity may indicate legacy duplicates. Repeated text with different IDs is not the same as repeated query output.'}


def eval_one(test, entries):
    matching=[e for e in entries if source_matches(e['metadata'],test)]
    first_match=next((i+1 for i,e in enumerate(entries) if source_matches(e['metadata'],test)),None)
    joined='\n'.join(e['document'] for e in matching)
    fact_flags={name:bool(re.search(pattern, joined, flags=re.I|re.S))
                for name,pattern in test.get('required_facts',{}).items()}
    # Precision here measures target source/type, not externally adjudicated relevance.
    return {'test_id':test['id'],'question':test['question'],'target_source':test['source'],
            'top_k':test['top_k'],'retrieved':len(entries), 'target_hits':len(matching),
            'first_target_rank':first_match,
            'target_precision_at_k':round(len(matching)/len(entries),3) if entries else 0,
            'target_recall_at_k':bool(matching),
            'evidence_flags':fact_flags,
            'evidence_coverage':round(sum(fact_flags.values())/len(fact_flags),3) if fact_flags else None,
            'retrieved_sources':[{'source':e['metadata'].get('source'),
                'file_type':e['metadata'].get('file_type'),'page':e['metadata'].get('page'),
                'table_number':table_number(e['metadata']) if e['metadata'].get('file_type')=='pdf_table' else None,
                'figure_number':e['metadata'].get('figure_number'),
                'chunk':e['metadata'].get('chunk_number'),
                'distance':round(e['distance'],4) if e['distance'] is not None else None}
                for e in entries]}


def run(skip_embed=False, show_samples=False):
    t0=time.perf_counter()
    client=chromadb.PersistentClient(path=str(DB_PATH))
    collection=client.get_collection(COLLECTION)  # Never create missing collection.
    before=collection.count()
    all_ = collection.get(include=['documents','metadatas'])
    ids=all_['ids']
    docs=all_.get('documents') or []
    metas=all_.get('metadatas') or []
    duplicates=examine_duplicates(ids,docs,metas)
    result={'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'mode':'metadata_only' if skip_embed else 'semantic_and_metadata',
            'record_count_before':before,'file_types':dict(Counter(m.get('file_type','unknown') for m in metas)),
            'duplicates':duplicates, 'tests':[]}

    import ollama
    for test in TESTS:
        if skip_embed:
            # Exhaustive metadata-only evidence availability: not a retrieval score.
            entries=[{'document':d,'metadata':m,'distance':None}
                     for d,m in zip(docs,metas) if source_matches(m,test)]
            scored=eval_one(test,entries)
            scored['evaluation_mode']='evidence_availability_only'
        else:
            emb=ollama.embed(model=EMBEDDING_MODEL,input=test['question'])['embeddings'][0]
            output=collection.query(query_embeddings=[emb],n_results=min(test['top_k'],before),
                         include=['documents','metadatas','distances'])
            entries=[{'document':d,'metadata':m,'distance':dist}
                     for d,m,dist in zip(output['documents'][0],output['metadatas'][0],output['distances'][0])]
            scored=eval_one(test,entries)
            scored['evaluation_mode']='semantic_top_k'
        result['tests'].append(scored)
        print(f"{test['id']}: target hits={scored['target_hits']}/{scored['retrieved']} "
              f"evidence={scored['evidence_coverage']} first rank={scored['first_target_rank']}")
        if show_samples:
            for source in scored['retrieved_sources']:
                print('  ',source)
    after=collection.count()
    result['record_count_after']=after
    result['database_unchanged']=before==after
    result['elapsed_seconds']=round(time.perf_counter()-t0,2)
    REPORT_DIR.mkdir(parents=True,exist_ok=True)
    path=REPORT_DIR / ('retrieval_quality_metadata.json' if skip_embed else 'retrieval_quality_semantic.json')
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f"Records: {before} -> {after} | unchanged: {result['database_unchanged']}")
    print(f"Exact duplicate text groups: {duplicates['exact_duplicate_group_count']}")
    print(f'Report: {path}')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-embed',action='store_true',help='Inspect evidence and duplicates without Ollama')
    parser.add_argument('--show-sources',action='store_true',help='Print retrieved metadata for each test')
    args=parser.parse_args()
    run(skip_embed=args.no_embed,show_samples=args.show_sources)
