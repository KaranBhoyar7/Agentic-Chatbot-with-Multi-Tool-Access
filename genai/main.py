import json
import logging
import os
import shutil
import http.client
from pathlib import Path
from io import BytesIO
from fastapi import FastAPI, UploadFile, File
import PyPDF2
import psutil
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain
from langchain_ollama import ChatOllama
import wikipedia
from wikipedia.exceptions import PageError, DisambiguationError

# Initialize FastAPI app
app = FastAPI()

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EMBEDDINGS_DIR = "embeddings"

# Helper function to log memory usage
def log_memory_usage():
    """Log memory usage"""
    process = psutil.Process(os.getpid())
    logger.info(f"Memory usage: {process.memory_info().rss / (1024 ** 2):.2f} MB")

# Function to load and extract text from PDF
def load_pdf_documents(file: UploadFile):
    """Load and extract text from a PDF file."""
    try:
        pdf_content = file.file.read()
        reader = PyPDF2.PdfReader(BytesIO(pdf_content))
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        doc = Document(page_content=text, metadata={"source": file.filename})
        return [doc]
    except Exception as e:
        logger.error(f"Error processing PDF file: {e}")
        return []

# Endpoint for training the model on PDF
@app.post("/train-faiss")
async def train(pdf_file: UploadFile = File(...), index_name: str = "default_index"):
    """API endpoint to train model with PDF input."""
    try:
        logger.info("Starting training process.")
        log_memory_usage()

        # Load documents from PDF
        all_docs = load_pdf_documents(pdf_file)
        logger.info(f"Loaded {len(all_docs)} document(s) from PDF.")
        log_memory_usage()

        # Split documents into smaller chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(all_docs)
        logger.info(f"Split into {len(split_docs)} document chunks.")
        log_memory_usage()

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings()

        # Prepare embedding path
        embedding_path = Path(os.path.join(EMBEDDINGS_DIR, index_name))
        if embedding_path.exists() and embedding_path.is_dir():
            logger.info(f"Clearing existing embedding directory: {embedding_path}")
            shutil.rmtree(embedding_path, ignore_errors=True)
        
        embedding_path.mkdir(parents=True, exist_ok=True)

        # Create FAISS index and save it
        db = FAISS.from_documents(split_docs, embeddings)
        db.save_local(embedding_path)
        logger.info("Training and saving index completed.")
        log_memory_usage()

        return {"message": "Training Done"}

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return {"Message": str(e)}

# Endpoint to search the FAISS index (RAG functionality)
@app.post("/search-faiss")
async def search(query: str, index_name: str = "default_index"):
    """API endpoint to search the FAISS index."""
    try:
        # Load the FAISS index
        embedding_path = Path(os.path.join(EMBEDDINGS_DIR, index_name))
        if embedding_path.exists() and embedding_path.is_dir():
            docsearch = FAISS.load_local(embedding_path, HuggingFaceEmbeddings(), allow_dangerous_deserialization=True)
            logger.info(f"Loaded FAISS index from: {embedding_path}")

            # Perform similarity search
            docs = docsearch.similarity_search(query)

            # Initialize the QA chain and get the response
            combine_prompt_template = """Given the extracted content and the question, create a final answer.
            If the answer is not contained in the context, say "answer not available in context." 

            Context: 
            {context}

            Question: 
            {question}

            Answer:
            """
            combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["context", "question"])
            chain = load_qa_chain(ChatOllama(model="Llama3.2:3b"), chain_type="stuff", prompt=combine_prompt)
            stuff_answer = chain({"input_documents": docs, "question": query}, return_only_outputs=True)

            result = stuff_answer.get("output_text", "Answer not available.")
            return {"result": result}
        else:
            return {"Message": "FAISS index not found."}
    except Exception as e:
        return {"Message": str(e)}

# Endpoint for Wikipedia search tool
@app.post("/search_wikipedia")
async def search_wikipedia_endpoint(query: str):
    """Search Wikipedia for the given query and return a summary."""
    try:
        result = wikipedia.summary(query, sentences=3)
        return {"result": result}
    except PageError:
        return {"result": "No Wikipedia page found for the given query."}
    except DisambiguationError as e:
        return {"result": f"The query is ambiguous. Suggestions: {e.options}"}

@app.post("/search_bing")
async def bing_search_endpoint(query: str):
    """Search Bing for the given query and return the top results."""
    conn = http.client.HTTPSConnection("bing-search-apis.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': "1855d9c335msh5057e7e3990f6f8p157c92jsn1baa0cc59087",
        'x-rapidapi-host': "bing-search-apis.p.rapidapi.com"
    }

    formatted_query = query.replace(" ", "-")
    request_url = f"/api/rapid/web_search?keyword={formatted_query}&page=0&size=10"

    try:
        conn.request("GET", request_url, headers=headers)
        res = conn.getresponse()
        if res.status == 200:
            data = res.read()
            results = json.loads(data.decode("utf-8"))
            print(results)

            if 'data' in results and 'items' in results['data']:
                # Returning only the 'items' part of the response
                return results['data']['items']
            else:
                return {"result": "No search results found."}
        else:
            return {"result": f"Request failed with status: {res.status}"}
    except Exception as e:
        return {"result": f"An error occurred: {e}"}
    finally:
        conn.close()