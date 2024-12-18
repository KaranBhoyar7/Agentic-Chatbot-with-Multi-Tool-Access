import streamlit as st
import requests
import wikipedia
from googlesearch import search
from bs4 import BeautifulSoup
import os
from pathlib import Path
import shutil
import fitz
from io import BytesIO
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_ollama import ChatOllama
from langchain.chains.question_answering import load_qa_chain
# Constants
EMBEDDINGS_DIR = "embeddings"

# Streamlit App Title
st.title("Agentic Chatbot with Multi-Tool Access")

# Function: Query ChatOllama
def query_chatollama(query, index_name="default_index"):
    """API endpoint to search the FAISS index and query ChatOllama."""
    try:
        # Check if FAISS index exists
        embedding_path = Path(os.path.join(EMBEDDINGS_DIR, index_name))
        if not embedding_path.exists() or not embedding_path.is_dir():
            return {"result": "FAISS index not found. Please train your resume first."}

        # Load FAISS index
        embeddings = HuggingFaceEmbeddings()
        docsearch = FAISS.load_local(embedding_path, embeddings, allow_dangerous_deserialization=True)

        # Perform similarity search
        docs = docsearch.similarity_search(query)
        if not docs:
            return {"result": "No relevant documents found in the FAISS index."}

        # Initialize the QA chain
        combine_prompt_template = """
        Given the extracted content and the question, create a final answer.
        If the answer is not contained in the context, say "Answer not available in context."

        Context: 
        {context}

        Question: 
        {question}

        Answer:
        """
        combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(ChatOllama(model="Llama3.2:3b"), chain_type="stuff", prompt=combine_prompt)
        result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)

        return {"result": result.get("output_text", "Answer not available.")}
    except Exception as e:
        return {"result": f"Error: {str(e)}"}

# Function: Load PDF Text
def load_pdf_text(file):
    """Extract text from a PDF file using PyMuPDF."""
    try:
        pdf_content = file.read()
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            text += pdf_document[page_num].get_text()
        pdf_document.close()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# Function: Train FAISS Index
def train_faiss_index(text, index_name="default_index"):
    """Train and save a FAISS index."""
    try:
        # Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings()

        # Prepare embedding directory
        embedding_path = Path(os.path.join(EMBEDDINGS_DIR, index_name))
        if embedding_path.exists():
            shutil.rmtree(embedding_path)
        embedding_path.mkdir(parents=True, exist_ok=True)

        # Create FAISS index and save
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(embedding_path)
        return True
    except Exception as e:
        st.error(f"Error training FAISS index: {e}")
        return False

# Function: Wikipedia Search
def wikipedia_search(query):
    """Search Wikipedia for a query."""
    try:
        summary = wikipedia.summary(query, sentences=3)
        url = wikipedia.page(query).url
        return summary, url
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found: {e.options}", None
    except wikipedia.exceptions.HTTPTimeoutError:
        return "Wikipedia request timed out. Try again later.", None
    except Exception as e:
        return f"Error: {e}", None

# Function: Web Search
def web_search(query):
    """Perform a web search and retrieve results."""
    try:
        results = []
        for url in search(query, num_results=5):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.string if soup.title else "No Title"
            description = soup.find("meta", attrs={"name": "description"})
            description = description["content"] if description else "No Description"
            results.append({"title": title, "description": description, "url": url})
        return results
    except Exception as e:
        return [{"title": "Error", "description": str(e), "url": ""}]

# Upload and Train Resume
st.header("Upload and Train Resume")
uploaded_file = st.file_uploader("Upload your CV (PDF only)", type=["pdf"])
if uploaded_file and st.button("Train Resume"):
    text = load_pdf_text(uploaded_file)
    if text:
        success = train_faiss_index(text)
        if success:
            st.success("Resume trained successfully.")
        else:
            st.error("Training failed.")

# Query Input Section
st.header("Inference and Search Tools")
query = st.text_input("Enter your query here")

if query:
    tool_choice = st.selectbox("Choose a tool", ["ChatFAISS", "Wikipedia Search", "Web Search"])

    if tool_choice == "ChatFAISS":
        response = query_chatollama(query)
        st.write(f"ChatFAISS Response: {response['result']}")

    elif tool_choice == "Wikipedia Search":
        summary, url = wikipedia_search(query)
        if summary:
            st.write(f"Summary: {summary}")
            if url:
                st.write(f"[Read more on Wikipedia]({url})")

    elif tool_choice == "Web Search":
        results = web_search(query)
        for res in results:
            st.write(f"Title: {res['title']}\nDescription: {res['description']}\nURL: {res['url']}")
else:
    st.info("Enter a query to proceed.")
