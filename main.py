import pdfplumber
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from config import OPENAI_API_KEY

# -----------------------------
# GET API KEY SECURELY
# -----------------------------

if not OPENAI_API_KEY:
    st.error("OpenAI API Key not found. Please set it in Streamlit Secrets.")
    st.stop()

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# CUSTOM BUTTON COLORS
# -----------------------------
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #dc3545;
    color: white;
}
div.stButton > button:first-child:hover {
    background-color: #c82333;
}
div.stDownloadButton > button:first-child {
    background-color: #28a745;
    color: white;
}
div.stDownloadButton > button:first-child:hover {
    background-color: #218838;
}
</style>
""", unsafe_allow_html=True)

st.title("📄 RetrievalDocBot")
st.caption("Upload multiple PDFs and ask intelligent questions")

# -----------------------------
# SESSION STATE
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# COPY BUTTON FUNCTION
# -----------------------------
def add_copy_button(text):
    st.markdown(f"""
        <button onclick="navigator.clipboard.writeText(`{text}`)"
        style="
            margin-top:8px;
            background-color:#28a745;
            color:white;
            border:none;
            padding:6px 12px;
            border-radius:6px;
            cursor:pointer;
            font-size:14px;">
            📋 Copy
        </button>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("⚙ Control Panel")

    files = st.file_uploader(
        "Upload up to 10 PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if files and len(files) > 10:
        st.error("Maximum 10 PDFs allowed.")
        files = files[:10]

    st.markdown("---")

    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    max_tokens = st.slider("Max Tokens", 200, 2000, 1000, 100)

    st.markdown("---")

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.success("Chat cleared!")

    if st.session_state.chat_history:
        chat_text = ""
        for role, message in st.session_state.chat_history:
            chat_text += f"{role.capitalize()}: {message}\n\n"

        st.download_button(
            label="⬇ Download Chat History",
            data=chat_text,
            file_name="chat_history.txt",
            mime="text/plain",
            use_container_width=True
        )

# -----------------------------
# MAIN LOGIC
# -----------------------------
if files:

    text = ""
    total_pages = 0

    for file in files:
        with pdfplumber.open(file) as pdf:
            total_pages += len(pdf.pages)
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"

    with st.sidebar:
        st.subheader("📊 Document Info")
        st.write(f"Files: {len(files)}")
        st.write(f"Total Pages: {total_pages}")
        st.write(f"Characters: {len(text)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )

    vector_store = FAISS.from_texts(chunks, embeddings)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=OPENAI_API_KEY
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer the question using ONLY the provided PDF context.\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])

    chain = (
        {"context": retriever | format_docs,
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    user_question = st.chat_input("💬 Ask a question")

    if user_question:
        st.session_state.chat_history.append(("user", user_question))

        with st.spinner("🤖 Thinking..."):
            response = chain.invoke(user_question)

        st.session_state.chat_history.append(("assistant", response))
        # -----------------------------
        # DISPLAY CHAT
        # -----------------------------
    for role, message in st.session_state.chat_history:

        with st.chat_message(role):

            if role == "assistant":
                st.markdown(message)
                st.code(message, language="text")
            else:
                st.markdown(message)
else:
    st.info("👈 Upload PDFs to start chatting.")