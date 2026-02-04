# create a streamlit chatbot using pinecone and llama model
import streamlit as st      
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import LlamaCpp
import pinecone
import os
from dotenv import load_dotenv
load_dotenv()
# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
index_name = os.getenv("PINECONE_INDEX_NAME")
# Load Llama model
llama_model_path = os.getenv("LLAMA_MODEL_PATH")
llm = LlamaCpp(model_path=llama_model_path)
# Load Pinecone vector store
vectorstore = Pinecone(index_name=index_name, embedding_function=LlamaCppEmbeddings(model_path=llama_model_path))
# Create Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())
# Streamlit app
st.title("Chatbot with Pinecone and Llama")
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
user_input = st.text_input("You: ", "")
if user_input:
    response = qa_chain.run(input=user_input, chat_history=st.session_state.chat_history)
    st.session_state.chat_history.append((user_input, response))
    for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
        st.write(f"User: {user_msg}")
        st.write(f"Bot: {bot_msg}")
# Run the app with: streamlit run app.py