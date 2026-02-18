import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

# Streamlit Page Config
st.set_page_config(page_title="Medical RAG Chatbot", page_icon="🩺")
st.title("🩺 Medical RAG Chatbot")
st.write("Ask medical questions based on the uploaded medical encyclopedia.")

# Load API Key from Streamlit Secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Load LLM
def load_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0.5
    )

# Load FAISS Vector Store
@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "vectorstore/db_faiss",
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return db

db = load_vectorstore()
retriever = db.as_retriever(search_kwargs={"k": 3})

# Prompt
custom_prompt_template = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know.
Do not make up an answer.

Context:
{context}

Question:
{question}

Start the answer directly.
"""

prompt = PromptTemplate(
    template=custom_prompt_template,
    input_variables=["context", "question"]
)

# Build RAG Chain
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | load_llm()
    | StrOutputParser()
)

# Chat UI
user_query = st.text_input("Ask your medical question:")

if user_query:
    with st.spinner("Generating answer..."):
        response = rag_chain.invoke(user_query)

    st.subheader("Answer")
    st.write(response)
