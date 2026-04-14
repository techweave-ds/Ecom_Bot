import streamlit as st
import os
import datetime

from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================
# 🔐 API Key
# =========================
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# =========================
# 🧠 LLM CALL FUNCTION
# =========================
def call_llm(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# =========================
# 📦 Load / Create Vector DB
# =========================
@st.cache_resource
def load_vectordb():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if not os.path.exists("./faq_db"):
        st.info("🔄 Creating knowledge base (first run)...")

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
    Classify this query into one of these:
    Warranty, Delivery, Returns, Product Info, General FAQ.

    Query: {query}

    Only return category.
    """
    return call_llm(prompt).strip()


# =========================
# 🔎 RAG
# =========================
def get_answer(query):
    intent = classify_query(query)

    enhanced_query = f"{intent}: {query}"

    docs = vectordb.similarity_search(enhanced_query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are an e-commerce support assistant.

    Answer ONLY from context.
    If not found, say:
    "I don't have that information."

    Context:
    {context}

    Question:
    {query}
    """

    response = call_llm(prompt)

    return response, intent


# =========================
# 📊 Logging
# =========================
def log_query(query, intent, response):
    with open("logs.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now()} | {intent} | {query} | {response}\n")


# =========================
# 🎨 UI
# =========================
st.title("🛍️ E-Commerce Support Bot")
st.caption("Ask about delivery, returns, warranty & more")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    response, intent = get_answer(prompt)

    log_query(prompt, intent, response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

    st.caption(f"🧠 Intent: {intent}")
