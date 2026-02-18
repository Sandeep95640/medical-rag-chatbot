import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS


# Load Environment Variables
load_dotenv()

# Step 1: Load Groq LLM
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3
    )

# Step 2: Load FAISS Vector Store
DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 3})


# Step 3: Custom Prompt
custom_prompt_template = """
Use ONLY the information provided in the context below.

If the answer is not in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=custom_prompt_template,
    input_variables=["context", "question"]
)


# Helper: Format Documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Step 4: Build RAG Chain (Correct Way)
rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
        "source_docs": retriever
    }
    | RunnableLambda(lambda x: {
        "answer": (
            load_llm().invoke(
                prompt.format(
                    context=x["context"],
                    question=x["question"]
                )
            )
        ),
        "source_docs": x["source_docs"]
    })
)


# Step 5: Run Query
while True:
    user_query = input("\nWrite Query Here (type 'exit' to quit): ")

    if user_query.lower() == "exit":
        break

    result = rag_chain.invoke(user_query)

    print("\nAnswer:\n", result["answer"].content)

    print("\n📚 Source Documents Used:")
    for i, doc in enumerate(result["source_docs"]):
        print(f"\nDocument {i+1}")
        print("File:", doc.metadata.get("source"))
        print("Page:", doc.metadata.get("page"))
