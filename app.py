import os
import streamlit as st
from PyPDF2 import PdfReader
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# === Load Models ===
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
generator = pipeline("text-generation", model="gpt2")

# === Read PDF ===
def read_pdf(pdfs):
    text = ""
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

# === Split Text ===
def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# === Create FAISS Index ===
def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings, chunks

# === Generate Answer ===
def generate_answer(query, index, chunks):
    query_embedding = embedder.encode([query])
    _, indices = index.search(np.array(query_embedding), k=3)
    context = "\n".join([chunks[i] for i in indices[0]])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    output = generator(prompt, max_length=200, num_return_sequences=1)
    return output[0]["generated_text"]

# === Streamlit App ===
def main():
    st.set_page_config(page_title="PDF Q&A App", layout="wide")
    st.title("📄  RICCHI'S RAG BASED CHAT PDF")

    if "index" not in st.session_state:
        st.session_state.index = None
        st.session_state.chunks = None

    question = st.text_input("Ask a question about your PDF content:")

    if question and st.session_state.index:
        with st.spinner("Generating answer..."):
            answer = generate_answer(question, st.session_state.index, st.session_state.chunks)
            st.markdown(f"**Answer:** {answer}")

    with st.sidebar:
        st.header("Upload PDFs")
        pdfs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)

        if st.button("Process PDFs"):
            if not pdfs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing..."):
                    raw_text = read_pdf(pdfs)
                    chunks = split_text(raw_text)
                    index, _, chunk_data = create_faiss_index(chunks)
                    st.session_state.index = index
                    st.session_state.chunks = chunk_data
                    st.success("PDFs processed successfully!")

if __name__ == "__main__":
    main()
