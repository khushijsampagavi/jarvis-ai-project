import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import ollama

PINECONE_API_KEY = "pcsk_759xwF_S1byE3zR7gZH8uTDTEg4xzaFSH4Dqj1xBgDRmw3HyLT2rmnz4AAc8NunpBNcwTx"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("jarvis-ai")

model = SentenceTransformer("all-MiniLM-L6-v2")

st.title("🤖 Jarvis AI Assistant")

query = st.text_input("Ask a question:")

if query:
    q_vector = model.encode([query])[0].tolist()

    results = index.query(
        vector=q_vector,
        top_k=2,
        include_metadata=True
    )

    context = " ".join(
        match["metadata"]["text"] for match in results["matches"]
    )

    prompt = f"""
Use this information to answer:

{context}

Question: {query}
"""

    response = ollama.chat(
        model="llama2",
        messages=[{"role": "user", "content": prompt}]
    )

    st.success(response["message"]["content"])
