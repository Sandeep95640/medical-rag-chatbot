# 🩺 Medical RAG Chatbot

An AI-powered Medical Question Answering system built using **Retrieval-Augmented Generation (RAG)** with **Groq LLaMA 3.1**, **LangChain**, **FAISS**, and **Streamlit**.

🔗 **Live App:**  
https://medical-rag-chatbot-ai.streamlit.app/

---

## 🚀 Overview

This application allows users to ask medical-related questions based on a trusted medical encyclopedia (The Gale Encyclopedia of Medicine).

Instead of generating generic AI answers, the system:

- Retrieves relevant sections from the medical PDF
- Sends contextual information to the LLM
- Generates answers strictly grounded in retrieved content
- Displays source document page numbers

This ensures:

- Higher factual accuracy  
- Reduced hallucinations  
- Transparent answer sourcing  

---

## 🧠 Architecture

User Question  
→ FAISS Vector Search  
→ Retrieve Top 3 Relevant Chunks  
→ Custom Prompt Template  
→ Groq LLaMA 3.1 Model  
→ Context-Grounded Answer + Sources  

---

## 🏗️ Tech Stack

- **LLM:** Groq (LLaMA 3.1 8B Instant)
- **Framework:** LangChain (LCEL)
- **Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database:** FAISS
- **UI:** Streamlit
- **Deployment:** Streamlit Cloud
- **Environment Management:** python-dotenv

---

## 📂 Project Structure

```
medical-rag-chatbot/
│
├── app.py
├── create_memory_for_llm.py
├── connect_memory_with_llm.py
├── requirements.txt
├── .gitignore
│
├── data/
│   └── The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf
│
└── vectorstore/
    └── db_faiss/
        ├── index.faiss
        └── index.pkl
```

---

## ⚙️ How It Works

### 1️⃣ Document Processing

- Load PDF using `PyPDFLoader`
- Split into 500-token chunks
- Generate embeddings using `all-MiniLM-L6-v2`
- Store embeddings in FAISS vector database

### 2️⃣ Retrieval-Augmented Generation (RAG)

- Retrieve top 3 relevant chunks using semantic search
- Inject retrieved context into custom prompt
- Generate answer using Groq LLaMA 3.1
- Return answer strictly based on retrieved content

### 3️⃣ Output

- AI-generated response
- Source document references
- Page numbers for transparency

---

## 🔐 Environment Variables

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

### For Streamlit Cloud Deployment

Go to:

App → Settings → Secrets  

Add:

```
GROQ_API_KEY = "your_groq_api_key_here"
```

---

## 📦 Installation (Local Setup)

```bash
git clone https://github.com/Sandeep95640/medical-rag-chatbot.git
cd medical-rag-chatbot

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

---

## 🧪 Example Queries

- What are the treatment options for cancer?
- What are the treatments and management methods for diabetes?
- What causes hypertension?
- What are symptoms of asthma?

---

## 📊 Key Features

✅ Context-grounded responses  
✅ Page number citations  
✅ Groq ultra-fast inference  
✅ FAISS semantic search  
✅ Clean modular architecture  
✅ Production-ready Streamlit deployment  

---

## 🎯 Why This Project Matters

This project demonstrates:

- Real-world RAG implementation
- LLM + Vector Database integration
- Context-aware prompting
- Secure deployment with secrets
- Practical GenAI engineering skills

It showcases production-level AI system design beyond basic chatbot applications.

---

## 🔮 Future Improvements

- Add conversational memory
- Multi-document upload support
- Highlight exact citation text
- Add PDF preview panel
- Upgrade to larger medical dataset
- Docker-based deployment



---

## ⭐ If You Found This Helpful

Give this repository a ⭐ on GitHub!
