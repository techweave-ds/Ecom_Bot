import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import datetime

# =========================
# 🔐 Load Environment
# =========================
load_dotenv()

print("API KEY:", os.getenv("GROQ_API_KEY"))
# =========================
# 🧠 Initialize LLM
# =========================
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# =========================
# 📦 Load Vector DB
# =========================
@st.cache_resource
def load_vectordb():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

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
    # Step 1: Classify
    intent = classify_query(query)

    # Step 2: Enhance query with intent
    enhanced_query = f"{intent}: {query}"

    # Step 3: Retrieve docs
    docs = vectordb.similarity_search(enhanced_query, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    # Step 4: LLM Prompt
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
# 📊 Logging (for analysis)
# =========================
def log_query(query, intent, response):
    with open("logs.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now()} | {intent} | {query} | {response}\n")


# =========================
# 🎨 Streamlit UI
# =========================
st.set_page_config(page_title="E-Commerce Bot", page_icon="🛍️")

st.title("🛍️ E-Commerce Support Bot")
st.caption("Ask about delivery, returns, warranty & more")

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
if prompt := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Get response
    response, intent = get_answer(prompt)

    # Log it
    log_query(prompt, intent, response)

    # Display response
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

    # Show detected intent
    st.caption(f"🧠 Intent detected: {intent}")