import streamlit as st
import os
import datetime

from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================
# 🔐 Load API Key (Streamlit Cloud)
# =========================
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# =========================
# 🧠 Initialize LLM (UPDATED MODEL)
# =========================
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# =========================
# 📦 Load / Create Vector DB
# =========================
@st.cache_resource
def load_vectordb():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if not os.path.exists("./faq_db"):
        st.info("🔄 Creating knowledge base... (first run only)")

        loader = PyPDFLoader("faq.pdf")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        vectordb = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory="./faq_db"
        )
    else:
        vectordb = Chroma(
            persist_directory="./faq_db",
            embedding_function=embeddings
        )

    return vectordb


vectordb = load_vectordb()

# =========================
# 🧠 Query Classification
# =========================
def classify_query(query):
    prompt = f"""
    Classify this query into one of these categories:
    - Warranty
    - Delivery
    - Returns
    - Product Info
    - General FAQ

    Query: {query}

    Only return the category name.
    """

    response = llm.invoke(prompt)
    return response.content.strip()


# =========================
# 🔎 RAG + Answer Generator
# =========================
def get_answer(query):
    intent = classify_query(query)

    enhanced_query = f"{intent}: {query}"

    docs = vectordb.similarity_search(enhanced_query, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are an e-commerce customer support assistant.

    Answer ONLY from the context below.
    If the answer is not present, say:
    "I don't have that information."

    Do NOT guess.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    response = llm.invoke(prompt)

    return response.content, intent


# =========================
# 📊 Logging
# =========================
def log_query(query, intent, response):
    with open("logs.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now()} | {intent} | {query} | {response}\n")


# =========================
# 🎨 UI
# =========================
st.set_page_config(page_title="E-Commerce Bot", page_icon="🛍️")

st.title("🛍️ E-Commerce Support Bot")
st.caption("Ask about delivery, returns, warranty & more")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input
if prompt := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    response, intent = get_answer(prompt)

    log_query(prompt, intent, response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

    st.caption(f"🧠 Intent detected: {intent}")
