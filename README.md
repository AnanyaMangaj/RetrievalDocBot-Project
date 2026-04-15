# 📄 RetrievalDocBot – RAG Based PDF Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) chatbot that enables intelligent question answering from uploaded PDF documents using semantic search and large language models.

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
