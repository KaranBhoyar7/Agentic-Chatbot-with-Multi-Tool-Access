# GenAI
# Agentic Chatbot with Multi-Tool Access

This project is a Streamlit-based chatbot application that combines various tools like FAISS, ChatOllama, Wikipedia search, and web search to provide an intelligent query answering system.

## Features
- **Resume Training**: Upload a PDF resume to train a FAISS index.
- **Inference Tools**:
  - Query a ChatOllama-powered knowledge base.
  - Search Wikipedia for quick summaries.
  - Perform web searches to retrieve top results.
- **Streamlit UI**: An interactive interface to upload files and query tools.

## Requirements
- Python 3.8+
- Required Libraries:
  - `streamlit`
  - `requests`
  - `wikipedia`
  - `googlesearch-python`
  - `beautifulsoup4`
  - `langchain`
  - `fitz` (PyMuPDF)
  - `HuggingFaceEmbeddings`
  - `FAISS`

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-URL>
   cd <project-directory>

### Prerequisites
Make sure you have [Python](https://www.python.org/downloads/) and [pip](https://pip.pypa.io/en/stable/installation/) installed on your system.

### Installation and Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
2. **Run the Streamlit app:**

   ```bash
      streamlit run main.py
   ```


 
