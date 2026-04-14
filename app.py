import streamlit as st
import os
import datetime
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# =========================
# 🔐 API KEY
# =========================
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# =========================
# 🧠 LLM CALL
# =========================
def call_llm(prompt):
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "meta-llama/llama-3.1-8b-instruct",
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    return response.json()["choices"][0]["message"]["content"]

# =========================
# 📄 LOAD PDF
# =========================
def load_pdf():
    reader = PdfReader("faq.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# =========================
# ✂️ CHUNKING
# =========================
def chunk_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

# =========================
# 🧠 EMBEDDINGS
# =========================
@st.cache_resource
def setup_embeddings():
    model = SentenceTransformer('all-MiniLM-L6-v2')

    text = load_pdf()
    chunks = chunk_text(text)

    embeddings = model.encode(chunks)

    return model, chunks, embeddings

model, chunks, embeddings = setup_embeddings()

# =========================
# 🔍 SEARCH
# =========================
def search(query):
    q_emb = model.encode([query])[0]

    similarities = np.dot(embeddings, q_emb)
    top_indices = np.argsort(similarities)[-3:]

    return [chunks[i] for i in top_indices]

# =========================
# 🧠 CLASSIFICATION
# =========================
def classify_query(query):
    prompt = f"""
    Classify into:
    Warranty, Delivery, Returns, Product Info, FAQ.

    Query: {query}

    Only return category.
    """
    return call_llm(prompt)

# =========================
# 🤖 ANSWER
# =========================
def get_answer(query):
    intent = classify_query(query)

    docs = search(query)
    context = "\n\n".join(docs)

    prompt = f"""
    Answer ONLY from context.
    If not found say:
    "I don't have that information."

    Context:
    {context}

    Question:
    {query}
    """

    return call_llm(prompt), intent

# =========================
# 📊 LOGGING
# =========================
def log_query(query, intent, response):
    with open("logs.txt", "a") as f:
        f.write(f"{datetime.datetime.now()} | {intent} | {query} | {response}\n")

# =========================
# 🎨 UI
# =========================
st.title("🛍️ E-Commerce Support Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask something..."):
    st.chat_message("user").write(prompt)

    response, intent = get_answer(prompt)

    log_query(prompt, intent, response)

    st.chat_message("assistant").write(response)
    st.caption(f"Intent: {intent}")
