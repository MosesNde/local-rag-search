r"""Read-only timing breakdown for a local Ollama + ChromaDB RAG pipeline.

Install in src/profile_rag.py and run from the project root:
    python .\src\profile_rag.py
    python .\src\profile_rag.py --skip-generation
    python .\src\profile_rag.py --question "What is retrieval-augmented generation?"

This profiles the common *semantic* path (embed -> Chroma query -> generate).
Explicit table lookup and deterministic Python rankings follow different paths.
No writes or indexing operations are performed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import chromadb
import ollama

try:
    from config import EMBEDDING_MODEL, ANSWER_MODEL, OLLAMA_HOST
except ImportError:
    EMBEDDING_MODEL = "nomic-embed-text"
    ANSWER_MODEL = "qwen3:8b"
    OLLAMA_HOST = "http://127.0.0.1:11434"

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "chroma"
DEFAULT_QUESTION = "What is retrieval-augmented generation?"


def elapsed(start):
    return round(time.perf_counter() - start, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--repeat", type=int, default=1, help="Repeat identical query to see warm-cache effects")
    parser.add_argument("--skip-generation", action="store_true", help="Only measure embedding and Chroma retrieval")
    parser.add_argument("--results", type=int, default=10)
    args = parser.parse_args()
    if args.repeat < 1 or args.repeat > 10:
        parser.error("--repeat must be between 1 and 10")
    if args.results < 1:
        parser.error("--results must be positive")

    print("Read-only RAG performance profile (no database writes)", flush=True)
    print(f"Question: {args.question}", flush=True)
    print(f"Embedding model: {EMBEDDING_MODEL} | Answer model: {ANSWER_MODEL}", flush=True)
    client = ollama.Client(host=OLLAMA_HOST)

    t = time.perf_counter()
    db = chromadb.PersistentClient(path=str(DB_DIR))
    collection = db.get_collection(name="documents")
    count_before = collection.count()
    print(f"ChromaDB startup: {elapsed(t):.3f}s | records: {count_before}", flush=True)
    if not count_before:
        print("Database has no records. Nothing to profile.")
        return

    runs = []
    for index in range(1, args.repeat + 1):
        print(f"\nRun {index}/{args.repeat}", flush=True)
        times = {}
        t = time.perf_counter()
        embedding = client.embed(model=EMBEDDING_MODEL, input=args.question)["embeddings"][0]
        times["embedding_s"] = elapsed(t)
        print(f"  Embedding: {times['embedding_s']:.3f}s", flush=True)

        t = time.perf_counter()
        result = collection.query(
            query_embeddings=[embedding],
            n_results=min(args.results, count_before),
            include=["documents", "metadatas", "distances"],
        )
        times["retrieval_s"] = elapsed(t)
        texts = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        print(f"  Chroma retrieval: {times['retrieval_s']:.3f}s | hits: {len(texts)}", flush=True)
        if distances:
            print(f"  Nearest distance: {distances[0]:.4f}", flush=True)

        t = time.perf_counter()
        parts = []
        for doc, meta in zip(texts, metadatas):
            source = (meta or {}).get("source", "Unknown")
            parts.append(f"Source: {source}\nContent: {doc}")
        context = "\n\n".join(parts)[:18000]
        times["context_build_s"] = elapsed(t)
        print(f"  Context construction: {times['context_build_s']:.3f}s | chars: {len(context)}", flush=True)

        if not args.skip_generation:
            prompt = (
                "Answer using only the following retrieved context. "
                "If it is insufficient, say so. Do not invent information.\n\n"
                f"Context:\n{context}\n\nQuestion: {args.question}"
            )
            t = time.perf_counter()
            response = client.chat(
                model=ANSWER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                think=False,
            )
            times["generation_s"] = elapsed(t)
            message = response.get("message", {}).get("content", "")
            eval_count = response.get("eval_count")
            prompt_eval_count = response.get("prompt_eval_count")
            print(f"  Qwen3 generation: {times['generation_s']:.3f}s | answer chars: {len(message)}", flush=True)
            if eval_count is not None:
                print(f"  Generated tokens: {eval_count} | prompt tokens: {prompt_eval_count}", flush=True)
        times["total_measured_s"] = round(sum(times.values()), 3)
        print(f"  Total measured: {times['total_measured_s']:.3f}s", flush=True)
        runs.append(times)

    count_after = collection.count()
    print("\nSummary (seconds)", flush=True)
    for key in runs[0]:
        avg = statistics.mean(run[key] for run in runs)
        print(f"  {key}: {avg:.3f}", flush=True)
    print(f"Records before: {count_before} | after: {count_after}", flush=True)
    if count_before != count_after:
        print("WARNING: collection count changed during profiling. This script does not write.", flush=True)
    else:
        print("PASS: database record count unchanged.", flush=True)
    print("NOTE: This measures a representative semantic path, not your query script's exact end-to-end latency.")


if __name__ == "__main__":
    main()
