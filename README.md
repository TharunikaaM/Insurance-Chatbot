# 🤖 Insurance Chatbot (RAG-Based)

AI-powered chatbot that answers insurance-related queries using **Retrieval-Augmented Generation (RAG)**, ensuring responses are grounded in actual policy documents.

---

## 🎯 Problem

Insurance policies are complex and difficult to navigate.
Users often struggle to find accurate and relevant information quickly.

This system solves the problem by retrieving **relevant policy content** and generating **fact-based responses**.

---

## ⚙️ Tech Stack

* **Backend:** FastAPI (Python)
* **Frontend:** React
* **AI/LLM:** LangChain
* **Vector Database:** Chroma
* **Embeddings:** GPT4AllEmbeddings
* **LLM:** LLaMA (NVIDIA API / Ollama)

---

## 🧠 Architecture

```id="chatbot-arch"
User Query → Embedding → Chroma Retrieval → LLM → Response
```

---

## 🔄 Workflow

1. Policy documents (PDFs) are ingested and processed
2. Text is split into chunks and converted into embeddings
3. Embeddings are stored in Chroma vector database
4. User sends a query from the frontend
5. System retrieves relevant document chunks
6. LLM generates response using retrieved context
7. Answer is returned to the user

---

## 📊 Features

* RAG-based question answering
* Semantic search for accurate retrieval
* Context-aware responses using chat history
* Multi-LLM support (cloud and local models)

---

## 📈 Output

* Accurate answers to insurance-related queries
* Context-based explanations derived from policy documents
* Improved reliability compared to standard chatbots

---

## 📂 Project Structure

```id="chatbot-structure"
insurance-bot/
├── backend/
│   ├── main.py
│   ├── services/
│   ├── models/
│   └── utils/
├── frontend/
│   ├── src/
│   └── components/
├── Policies/
├── savedEmbeddings/
```

---

## 🛠️ Setup

### 1. Clone Repository

```id="clone-chatbot"
git clone <repository-url>
cd insurance-bot
```

---

### 2. Backend Setup

```id="backend-chatbot"
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` file:

```id="env-chatbot"
NVIDIA_API_KEY=your_key_here
OLLAMA_BASE_URL=http://localhost:11434
```

Run backend:

```id="run-chatbot"
uvicorn main:app --reload
```

---

### 3. Frontend Setup

```id="frontend-chatbot"
cd ../frontend
npm install
npm run dev
```

---

## 🛠️ My Contribution

* Built RAG pipeline using LangChain and Chroma
* Implemented document ingestion, chunking, and embedding pipeline
* Integrated LLMs for context-aware response generation
* Developed backend APIs using FastAPI

---

## 📌 Note

This project was developed for learning and demonstration purposes.
