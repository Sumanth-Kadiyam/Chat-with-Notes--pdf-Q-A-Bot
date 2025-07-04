# 🧠 Chat with Your PDF Notes – Gemini + LangChain + FAISS (RAG)

This is an AI-powered **PDF Question & Answer Bot** that allows users to upload a PDF and interact with it using natural language questions. It leverages **Google Gemini 1.5 Flash**, **LangChain**, **FAISS**, and **Streamlit**, providing intelligent responses using **Retrieval-Augmented Generation (RAG)**.

---

## 🚀 Features

- 📄 Upload and process any PDF file
- 💬 Ask questions based on the PDF's content
- 🔍 Semantic search using FAISS vector store
- 🧠 Gemini 1.5 Flash LLM integration for accurate answers
- ⛓️ LangChain for document chunking, embeddings, and chaining
- 🎯 Clean, responsive UI with Streamlit

---

## 🛠️ Tech Stack

| Tool                 | Purpose                                  |
|----------------------|------------------------------------------|
|   Streamlit          | UI and front-end interaction             |
|   Gemini 1.5 Flash   | LLM for generating answers               |
|   LangChain          | RAG pipeline, embeddings, QA chains      |
|   FAISS              | Vector store for similarity search       |
|   PyMuPDF (fitz)     | Extracts text from PDF documents         |

---

## 📦 Installation

1. **Clone this repository**:

```bash
git clone https://github.com/your-username/pdf-qa-bot.git
cd pdf-qa-bot
