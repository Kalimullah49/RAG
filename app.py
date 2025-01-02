import os
from groq import Groq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import streamlit as st
from tempfile import NamedTemporaryFile

# Initialize the Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

# Function to create embeddings and store them in FAISS
def create_embedding_and_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_texts(chunks, embeddings)
    return vector_db

# Function to query the vector database and interact with Groq
def query_vector_db(query, vector_db):
    docs = vector_db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    try:
        chat_completion = client.chat_completion.create(
            messages=[
                {"role": "system", "content": f"use following context:\n{context}"},
                {"role": "user", "content": query}
            ],
            model="llama3-8b_8192"
        )
        return chat_completion['choices'][0]['message']['content']
    except Exception as e:
        return f"Error interacting with Groq: {str(e)}"

# Streamlit app
st.title("PDF Chatbot")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        pdf_path = temp_file.name

    try:
        text = extract_text_from_pdf(pdf_path)
        st.write("Text extracted successfully.")
    except Exception as e:
        st.write(f"Error extracting text from PDF: {str(e)}")
        text = None

    if text:
        chunks = chunk_text(text)
        st.write("Text chunked successfully.")
        vector_db = create_embedding_and_store(chunks)
        st.write("Text embeddings generated and stored successfully.")
        
        user_query = st.text_input("Enter your query")
        if user_query:
            response = query_vector_db(user_query, vector_db)
            st.write("Response from Groq:")
            st.write(response)
