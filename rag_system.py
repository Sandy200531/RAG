from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Step 1: Load documents (supports .txt, .pdf, .docx)
def load_documents(file_path):
    """Load documents from various file formats"""
    if file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    elif file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    return loader.load()

# Specify your document file path here
file_path = "sample_document.txt"  # Change to your actual file (.txt, .pdf, or .docx)

# Load the document
documents = load_documents(file_path)

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

print(f"Loaded {len(documents)} document(s)")
print(f"Split into {len(chunks)} chunks")

# Step 3: Create embeddings + vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="rag_collection"
)

print(f"Vector store created with {vectorstore._collection.count()} chunks")

# Step 4: Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Step 5: Create LLM
llm = ChatOllama(model="llama2", temperature=0.7)

# Step 6: Create prompt template
prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Step 7: Create RAG chain using LCEL
rag_chain = (
    {"context": retriever, "question": lambda x: x}
    | prompt
    | llm
    | StrOutputParser()
)

print("RAG chain ready!")

# Step 8: Query loop
print("\n=== RAG System Ready ===")
print("Ask questions about the document. Type 'exit' to quit.\n")

while True:
    question = input("Your question: ").strip()
    
    if question.lower() == "exit":
        print("Goodbye!")
        break
    
    if not question:
        continue
    
    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(question)
    
    # Generate answer
    answer = rag_chain.invoke(question)
    
    print(f"\nAnswer: {answer}")
    print("\n--- Sources ---")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"{i}. {doc.page_content[:150]}...")
    print()