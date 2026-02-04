from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import os

# 👉 Put your Pinecone key here
PINECONE_API_KEY = "pcsk_759xwF_S1byE3zR7gZH8uTDTEg4xzaFSH4Dqj1xBgDRmw3HyLT2rmnz4AAc8NunpBNcwTx"

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "jarvis-ai"

# Create index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,   # for MiniLM model
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("data/notes.txt") as f:
    texts = [line.strip() for line in f if line.strip()]

vectors = model.encode(texts)

data = []
for i, vec in enumerate(vectors):
    data.append((str(i), vec.tolist(), {"text": texts[i]}))

index.upsert(data)

print("✅ Knowledge successfully stored in Pinecone!")
