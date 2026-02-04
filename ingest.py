#upload text file data into pinecone vector database
import os
from dotenv import load_dotenv
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
load_dotenv()
# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
pinecone.init
(api_key=pinecone_api_key, environment=pinecone_environment)
index_name = os.getenv("PINECONE_INDEX_NAME")
# Load Llama model
llama_model_path = os.getenv("LLAMA_MODEL_PATH")
# Create embedding function
embedding_function = LlamaCppEmbeddings(model_path=llama_model_path)
# Load Pinecone vector store
vectorstore = Pinecone(index_name=index_name, embedding_function=embedding_function)
# Function to ingest text data
def ingest_text_data(text_data, doc_id):
    embedding = embedding_function.embed_query(text_data)
    vectorstore.add_texts(texts=[text_data], ids=[doc_id])
# Example usage
if __name__ == "__main__":
    sample_text = "This is a sample document to be ingested into Pinecone."
    sample_doc_id = "doc_1"
    ingest_text_data(sample_text, sample_doc_id)
    print(f"Document {sample_doc_id} ingested into Pinecone.")
    