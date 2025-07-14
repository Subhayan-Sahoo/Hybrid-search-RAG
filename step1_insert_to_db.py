import os
import time
import numpy as np
import psycopg2
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# === Configuration ===
PDF_DIR = './company_docs'
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
TFIDF_DIM = 1024
CONNECTION_STRING = "host=localhost dbname=dense+sparse user=postgres password=test port=5432"

# === Load Models ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# === Get All PDFs ===
pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
if not pdf_files:
    print("⚠️ No PDF files found in the directory.")
    exit()

# === Process Each PDF ===
t_start = time.time()
for doc in pdf_files:
    pdf_path = os.path.join(PDF_DIR, doc)
    print(f"\n📄 Processing: {doc}")

    company_name = os.path.splitext(doc)[0].lower()
    print(f"🏷️ Company Name: {company_name}")

    # Load PDF
    loader = PyPDFLoader(file_path=pdf_path, mode="single")
    pages = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents([pages[0]])
    chunk_texts = [chunk.page_content for chunk in chunks]

    # Dense Embedding
    dense_vectors = embedding_model.embed_documents(chunk_texts)

    # Sparse Embedding
    vectorizer = TfidfVectorizer(max_features=TFIDF_DIM)
    tfidf_sparse = vectorizer.fit_transform(chunk_texts).toarray().astype(np.float32)

    # Pad sparse vectors
    padded_sparse = np.zeros((len(tfidf_sparse), TFIDF_DIM), dtype=np.float32)
    for i, vec in enumerate(tfidf_sparse):
        padded_sparse[i, :len(vec)] = vec[:TFIDF_DIM]

    # Insert into PostgreSQL
    with psycopg2.connect(CONNECTION_STRING) as conn:
        with conn.cursor() as cur:
            now = datetime.utcnow()
            for i, (text, dense_vec) in enumerate(zip(chunk_texts, dense_vectors)):
                sparse_vec = padded_sparse[i]
                clean_text = ' '.join(text.split())
                chunk_id = int(now.strftime('%Y%m%d%H%M%S%f')) + i  # unique ID

                cur.execute("""
                    INSERT INTO annual_reports_index (
                        chunk_index, company_name, content,
                        dense_embedding, sparse_embedding
                    ) VALUES (%s, %s, %s, %s, %s)
                """, (
                    chunk_id,
                    company_name,
                    clean_text,
                    dense_vec if isinstance(dense_vec, list) else list(dense_vec),
                    sparse_vec.tolist()
                ))
        conn.commit()
    print(f"✅ Uploaded {len(chunk_texts)} chunks.")

t_end = time.time()
print("\n🚀 Document processing completed.")
print(f"⏱️ Total time taken: {t_end - t_start:.2f} seconds")

