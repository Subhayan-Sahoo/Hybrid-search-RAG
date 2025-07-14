import os
import time
import json
import numpy as np
import streamlit as st
import psycopg2
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# === Configuration ===
CONNECTION_STRING = "host=localhost dbname=dense+sparse user=postgres password=test port=5432"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_api_key")

# === Model Setup ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
chat_engine = ChatGroq(model="meta-llama/Llama-4-Scout-17B-16E-Instruct", groq_api_key=GROQ_API_KEY)

# === Sparse Embedding ===
def sparse_embedding(query, feat_dim=1024):
    vectorizer = TfidfVectorizer(max_features=feat_dim)
    tfidf_sparse = vectorizer.fit_transform([query]).toarray().astype(np.float32)
    if tfidf_sparse.shape[1] < feat_dim:
        tfidf_sparse = np.pad(tfidf_sparse, ((0, 0), (0, feat_dim - tfidf_sparse.shape[1])), mode='constant')
    elif tfidf_sparse.shape[1] > feat_dim:
        tfidf_sparse = tfidf_sparse[:, :feat_dim]
    return tfidf_sparse[0].tolist()

# === Streamlit UI ===
st.set_page_config(page_title="🔍 Hybrid Search QA", layout="wide")
st.title("💼 Hybrid QA on Financial Reports")

# --- Input Section ---
col1, col2 = st.columns(2)

with col1:
    select_company = st.selectbox("📁 Select Company", ["Wipro", "Suzlon", "marksanspharma", "Infosys", "Gabriel"])

with col2:
    query = st.text_input("❓ Ask your question about the reports", placeholder="e.g. Shareholding pattern as on March 31, 2025")

# === Main Logic ===
if query:
    st.info("🚀 Running hybrid retrieval and generating answer...")
    t1 = time.time()

    # --- Embedding Generation ---
    dense_vect = embedding_model.embed_query(query)
    sparse_vect = sparse_embedding(query)

    # --- PostgreSQL Hybrid Search ---
    with psycopg2.connect(CONNECTION_STRING) as conn:
        with conn.cursor() as cur:
            # Dense
            cur.execute("""
                SELECT content, dense_embedding <#> %s::vector AS score
                FROM annual_reports_index
                WHERE company_name = %s
                ORDER BY score ASC
                LIMIT 5;
            """, (dense_vect, select_company.lower()))
            dense_results = cur.fetchall()

            # Sparse
            cur.execute("""
                SELECT content, sparse_embedding <#> %s::vector AS score
                FROM annual_reports_index
                WHERE company_name = %s
                ORDER BY score ASC
                LIMIT 5;
            """, (sparse_vect, select_company.lower()))
            sparse_results = cur.fetchall()

    # === Merge and Deduplicate Chunks ===
    context_hybrid = []
    for ctx in dense_results + sparse_results:
        if ctx[0] not in context_hybrid:
            context_hybrid.append(ctx[0])

    # === Save Logs ===
    os.makedirs("logs", exist_ok=True)
    with open("logs/dense_retrievals.txt", "w") as f:
        f.write("\n\n".join([r[0] for r in dense_results]))
    with open("logs/sparse_retrievals.txt", "w") as f:
        f.write("\n\n".join([r[0] for r in sparse_results]))
    with open("logs/hybrid_context.txt", "w") as f:
        f.write("\n\n".join(context_hybrid))

    # === Show Retrieved Context (Optional) ===
    with st.expander("🔎 View Retrieved Chunks"):
        st.markdown("**Dense Matches**")
        for i, r in enumerate(dense_results):
            st.code(f"[{i+1}] {r[0]}")
        st.markdown("**Sparse Matches**")
        for i, r in enumerate(sparse_results):
            st.code(f"[{i+1}] {r[0]}")

    # === Construct Prompt ===
    final_prompt = (
        f"You are an AI assistant specialized in analyzing documents and answering questions based on provided context. "
        f"The context is provided in a list of text-chunks. Some may be irrelevant. Analyze them judicially. Give a concise and accurate answer.\n\n"
        f"User query: {query}\n\n"
        f"Context:\n{context_hybrid}\n\n"
        f"If the context does not contain the answer, respond with:\n"
        f"\"Answer could not be found in provided context\".\n"
        
    )
    
    sparse_prompt = (
        f"You are an AI assistant specialized in analyzing documents and answering questions based on provided context. "
        f"The context is provided in a list of text-chunks. Some may be irrelevant. Analyze them judicially. give a concise and accurate answer.\n\n"
        f"User query: {query}\n\n"
        f"Context:\n{sparse_results}\n\n"
        f"If the context does not contain the answer, respond with:\n"
        f"\"Answer could not be found in provided context\".\n"
        
    )
    
    dense_prompt = (
        f"You are an AI assistant specialized in analyzing documents and answering questions based on provided context. "
        f"The context is provided in a list of text-chunks. Some may be irrelevant. Analyze them judicially. give a concise and accurate answer.\n\n"
        f"User query: {query}\n\n"
        f"Context:\n{dense_results}\n\n"
        f"If the context does not contain the answer, respond with:\n"
        f"\"Answer could not be found in provided context\".\n"
        
    )
    response = chat_engine.invoke(final_prompt)
    response1 = chat_engine.invoke(sparse_prompt)
    response2 = chat_engine.invoke(dense_prompt)
    # === Invoke LLM ===
    col1,col2,col3 = st.columns(3)
    with col1:
        
        st.markdown("### 🧾 Hybrid Answer")
        st.write(response.content)
    with col2:
    
        
        st.markdown("### 🧾 Sparse Answer")
        st.write(response1.content)
    with col3:
    
        
        st.markdown("### 🧾 Dense Answer")
        st.write(response2.content)
    t2 = time.time()

    # === Show Final Output ===
    st.success(f"✅ Answer generated in {t2 - t1:.2f} seconds.")
    

