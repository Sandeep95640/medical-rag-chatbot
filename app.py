import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

# Load Environment Variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Page Config
st.set_page_config(page_title="Medical AI Assistant", layout="wide")

st.title("🩺 AI Medical Knowledge Assistant")
st.markdown("Ask medical questions based on the uploaded medical encyclopedia.")


# Load LLM
@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3
    )


# Load Vector Store
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

llm = load_llm()
db = load_vectorstore()
retriever = db.as_retriever(search_kwargs={"k": 5})


# Custom Prompt
custom_prompt_template = """
You are a medical knowledge assistant.

Use ONLY the provided context to answer the question.
If the answer is not in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer clearly and concisely.
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
    | llm
    | StrOutputParser()
)


# Chat Interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("Ask your medical question...")

if user_query:
    st.session_state.chat_history.append(("user", user_query))

    with st.spinner("Thinking..."):
        response = rag_chain.invoke(user_query)

    st.session_state.chat_history.append(("assistant", response))


# Display Chat
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)
