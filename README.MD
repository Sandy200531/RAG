# Document-Based RAG System

A local **Retrieval-Augmented Generation (RAG)** system that answers questions from user-provided documents. The project combines document retrieval with a local LLM using **LangChain, ChromaDB, and Ollama**.

## Features

- Supports **TXT, PDF, and DOCX** documents
- Splits documents into smaller chunks
- Generates embeddings using **Ollama**
- Stores embeddings in **ChromaDB**
- Retrieves the most relevant document chunks
- Generates answers using a local **Llama 2** model
- Displays relevant document sources with each answer
- Runs completely locally

## Technologies Used

- Python
- LangChain
- ChromaDB
- Ollama
- Llama 2
- Nomic Embed Text
- LCEL

## RAG Workflow

```text
Document
   ↓
Document Loader
   ↓
Text Chunking
   ↓
Ollama Embeddings
   ↓
ChromaDB Vector Store
   ↓
Retriever
   ↓
Relevant Chunks
   ↓
Prompt + Llama 2
   ↓
Generated Answer
```

## Supported Documents

| Format | Loader |
|---|---|
| `.txt` | TextLoader |
| `.pdf` | PyPDFLoader |
| `.docx` | Docx2txtLoader |

## Models Used

### Embedding Model

```text
nomic-embed-text
```

Used to convert document chunks and user questions into vector representations.

### LLM

```text
llama2
```

Used to generate answers based on the retrieved document context.

## Installation

Install the required Python packages:

```bash
pip install langchain langchain-community langchain-text-splitters langchain-ollama chromadb pypdf docx2txt
```

Make sure Ollama is installed and the required models are available:

```bash
ollama pull llama2
ollama pull nomic-embed-text
```

## Run the Project

Place your document in the project directory and update:

```python
file_path = "sample_document.txt"
```

Then run:

```bash
python rag.py
```

The system will prompt:

```text
=== RAG System Ready ===
Ask questions about the document. Type 'exit' to quit.

Your question:
```

## Example

**Question:**

```text
What is the main topic of the document?
```

**Answer:**

```text
The answer is generated using the relevant information
retrieved from the document.
```

The system also displays the source chunks used to generate the answer.

## Key Learning Outcomes

- Understanding the **RAG architecture**
- Loading and processing different document formats
- Document chunking and preprocessing
- Creating embeddings with local models
- Using ChromaDB for vector search
- Building retrieval pipelines with LangChain
- Connecting a local LLM to retrieved context
- Implementing source-based question answering

## Future Improvements

- Add a Streamlit web interface
- Support multiple documents
- Add conversation memory
- Display similarity scores
- Add a document upload interface
- Experiment with different embedding and LLM models