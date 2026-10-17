import chromadb
import ollama 

DISTANCE_THRESHOLD = 1.0

client = chromadb.PersistentClient(path="data/chroma")

collection = client.get_collection(
    name="documents"
)

query = input("Enter your question: ")

query_response = ollama.embed(
    model="nomic-embed-text",
    input=query,
)

query_embedding = query_response["embeddings"][0]

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10,
)

best_distance = results["distances"][0][0]

if best_distance > DISTANCE_THRESHOLD:
    print("\nAnswer:")
    print("I don't have enough information in the documents to answer that question.")
    exit()
    
context_parts = []

for document, metadata in zip(
    results["documents"][0],
    results["metadatas"][0],
):
    if metadata:
        context_parts.append(
            f"Source: {metadata['source']}\n"
            f"Chunk: {document}"
        )
    else:
        context_parts.append(
            f"Chunk: {document}"
        )

context = "\n\n".join(context_parts)


prompt = f"""
Answer the question using ONLY the provided context.

If the context does not contain enough information to answer
the question, respond with exactly:

INSUFFICIENT_INFORMATION

Do not guess or use outside knowledge.

Context:
{context}

Question:
{query}
"""

response = ollama.chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    think=False,
)

answer = response["message"]["content"].strip()

print("\nAnswer:")

if answer == "INSUFFICIENT_INFORMATION":
    print(
        "I don't have enough information in the documents "
        "to answer that question."
    )
else:
    print(answer)

    sources = set()

    for metadata in results["metadatas"][0]:
        if metadata and "source" in metadata:
            sources.add(
                (
                    metadata["source"],
                    metadata.get("page"),
                    metadata["chunk_number"],
                )
            )

    print("\nRetrieved sources:")

    for source, page, chunk_number in sorted(
        sources,
        key=lambda item: (item[0], item[1] or 0, item[2])
    ):
        if page is not None:
            print(f"- {source} (page {page}, chunk {chunk_number})")
        else:
            print(f"- {source} (chunk {chunk_number})")
