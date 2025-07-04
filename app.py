import os
import fitz  # PyMuPDF
import streamlit as st
import google.generativeai as genai

from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.base import LLM
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Set your Gemini API Key
GEMINI_API_KEY = "AIzaSyDNyAKdaLdLz23kdMJlljJCrky8ACoCp0U"  # Replace with your actual key
genai.configure(api_key=GEMINI_API_KEY)

# LangChain-compatible Gemini LLM wrapper
class GeminiLLM(LLM):
    def _call(self, prompt, stop=None):
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    @property
    def _llm_type(self):
        return "gemini-1.5-flash"

# PDF text extractor
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Streamlit UI setup
st.set_page_config(page_title="📚 ChatBot PDF Q&A (RAG)")
st.title("🧠 Chat with Your Notes 📄---- PDF Q&A Bot (●'◡'●)")

# Init session memory
if "history" not in st.session_state:
    st.session_state.history = []

# Upload PDF
uploaded_file = st.file_uploader("📥 Upload your PDF 📁 File", type="pdf")

# Show chat history
if st.session_state.history:
    for q, a in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(f"**You:** {q}")
        with st.chat_message("assistant"):
            st.markdown(f"**AI:** {a}")

# Ask question
with st.chat_message("user"):
    user_question = st.text_input("💬...Ask a question about the document📚...", key="user_input")

# Main logic
if uploaded_file:
    with st.spinner("📖 Reading & processing PDF🧠..."):
        # Extract & chunk text
        raw_text = extract_text_from_pdf(uploaded_file)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_text(raw_text)
        docs = [Document(page_content=t) for t in texts]

        # Embeddings & FAISS vector index
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        vectorstore = FAISS.from_documents(docs, embeddings)

if uploaded_file and user_question:
    with st.spinner("🤖 Generating answer 🤔..."):
        # Build context with history
        history_context = ""
        for q, a in st.session_state.history:
            history_context += f"User: {q}\nAI: {a}\n"
        history_context += f"User: {user_question}\nAI:"

        # RAG Retrieval
        retriever_docs = vectorstore.similarity_search(user_question, k=3)

        # LLM + Chain
        llm = GeminiLLM()
        chain = load_qa_chain(llm, chain_type="stuff")
        answer = chain.run(input_documents=retriever_docs, question=history_context)

        # Show answer & update memory
        with st.chat_message("assistant"):
            st.markdown(f"**AI:** {answer}")
        st.session_state.history.append((user_question, answer))

# Clear history button
if st.button("🧹 Clear Chat History..."):
    st.session_state.history = []
    st.rerun()