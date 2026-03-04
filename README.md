# 📄 RetrievalDocBot – RAG Based PDF Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) chatbot that enables intelligent question answering from uploaded PDF documents using semantic search and large language models.

---

## 🚀 Project Overview

RetrievalDocBot allows users to upload multiple PDF files and ask context-aware questions.  
The system retrieves relevant document chunks using vector similarity search and generates accurate answers strictly based on the uploaded content.

This project demonstrates the implementation of an end-to-end RAG pipeline using LangChain, OpenAI, and FAISS.

---

## ✨ Features

- 📂 Upload up to 10 PDF documents
- 🔍 Semantic search using FAISS
- 🤖 Context-based answer generation using OpenAI LLM
- 📊 Document statistics (Total pages, files, characters)
- 💬 Interactive chat interface (Streamlit)
- 📋 Copy response feature
- ⬇ Download full chat history
- 🗑 Clear chat functionality
- ⚙ Adjustable model parameters (Temperature & Max Tokens)

---

## 🧠 Tech Stack

| Technology     | Purpose                     |
|---------------|-----------------------------|
| Python        | Backend Logic               |
| Streamlit     | Web Interface               |
| LangChain     | RAG Pipeline                |
| OpenAI API    | LLM & Embeddings            |
| FAISS         | Vector Similarity Search    |
| pdfplumber    | PDF Text Extraction         |

---

## 🔄 How It Works

1. User uploads PDF documents.
2. Text is extracted using pdfplumber.
3. Text is split into chunks using LangChain.
4. OpenAI embeddings convert chunks into vectors.
5. FAISS stores vectors for semantic retrieval.
6. Relevant chunks are retrieved using MMR search.
7. Retrieved context is passed to GPT model.
8. Model generates an accurate answer based only on provided context.

---

## 🛠 Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/RAG-PDF-Chatbot.git
cd RAG-PDF-Chatbot
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # For Mac/Linux
venv\Scripts\activate      # For Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add OpenAI API Key

Create a file named:

```
config.py
```

Add:

```python
OPENAI_API_KEY = "your_openai_api_key_here"
```

### 5️⃣ Run Application

```bash
streamlit run app.py
```

---

## 📸 Application Preview

- Upload PDFs  
- Ask questions  
- Get AI-generated answers  
- Download chat history  

---

## 🎯 Learning Outcomes

- Built a complete Retrieval-Augmented Generation pipeline
- Implemented semantic search using FAISS
- Worked with OpenAI embeddings & GPT models
- Applied prompt engineering
- Designed an interactive AI application
- Implemented session-based chat history management

---

## 📌 Future Enhancements

- Persistent vector database storage
- User authentication system
- Multi-user chat support
- Source highlighting in answers
- Cloud deployment (Docker / AWS / OCI)

---

## 📄 License

This project is developed for educational and portfolio purposes.

---

## 👨‍💻 Author

**Ananya Mangaj**  
B.E. Artificial Intelligence & Data Science  
AI/ML Enthusiast | RAG | Generative AI | Data Science
