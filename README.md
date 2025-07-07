# 📄 RMO Insight – Chat with Your PDF using AI

**RMO Insight** is an intelligent, lightweight, and user-friendly web application that allows users to upload PDF files and interact with them using natural language queries. Built using Streamlit, Sentence Transformers, FAISS, and Hugging Face models, this app delivers context-aware answers from document content without needing OpenAI APIs.

---

## 🚀 Features

- 📥 Upload multiple PDF documents
- 🤖 Ask natural language questions
- 🧠 Semantic search with FAISS + Sentence Transformers
- ✍️ Contextual answer generation using Hugging Face (GPT-2)
- 🔓 Open-source and privacy-friendly (no external API keys required)
- 🌐 Deployable on Streamlit Cloud

---

## 🧠 Tech Stack

| Tool / Library         | Purpose                                |
|------------------------|----------------------------------------|
| **Streamlit**          | Web UI and deployment                  |
| **PyPDF2**             | PDF text extraction                    |
| **Sentence-Transformers** | Text embedding model (`all-MiniLM-L6-v2`) |
| **FAISS**              | Vector storage & semantic search       |
| **Transformers (Hugging Face)** | Answer generation (GPT-2)           |
| **dotenv**             | Environment variable management        |
| **Torch**              | Backend for Transformers               |

---

## 📂 Folder Structure

RMO-Insight/
│
├── app.py # Main Streamlit application
├── faiss_db/ # Stores FAISS index (created after upload)
├── requirements.txt # Required dependencies
└── README.md # Project documentation

## TO RUN IN VisualCODE
       python -m run stramlit app.py
## DEPLOYED ON STREAM LIT
       https://rag-rmo-insight.streamlit.app/

