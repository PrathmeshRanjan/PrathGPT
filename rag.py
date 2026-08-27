from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import docx2txt

load_dotenv()

# ---------------------------------------------------------------------------
# Directory & Vector Store Initialization
# ---------------------------------------------------------------------------
# Ensure storage directories exist for uploaded files and persistent vector database
Path("uploads").mkdir(exist_ok=True)
Path("chroma_db").mkdir(exist_ok=True)

# Google Gemini Embedding model used to convert text chunks into vector representations
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# ChromaDB vector store configured to persist embeddings locally on disk
vectorstore = Chroma(
    collection_name="agentic_chatbot_docs",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)


# ---------------------------------------------------------------------------
# Document Parsing
# ---------------------------------------------------------------------------
def read_file_text(file_path: str) -> str:
    """
    Extracts raw text from multiple supported document types:
    - PDF (.pdf) using pypdf
    - Word (.docx) using docx2txt
    - Plain text / code (.txt, .md, .py, .csv) using native UTF-8 reading
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    # Extract text from each page of a PDF
    if suffix == ".pdf":
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""
            text += "\n"

        return text

    # Extract text from DOCX Word documents
    if suffix == ".docx":
        return docx2txt.process(file_path)

    # Read plain text, Markdown, Python code, or CSV files directly
    if suffix in [".txt", ".md", ".py", ".csv"]:
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError("Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV.")


# ---------------------------------------------------------------------------
# Ingestion & Indexing Pipeline
# ---------------------------------------------------------------------------
def add_document_to_rag(file_path: str, thread_id: str):
    """
    Ingests an uploaded document into the ChromaDB vector database:
    1. Extracts text from the file.
    2. Splits text into manageable overlapping chunks.
    3. Attaches metadata (thread_id and filename) for thread-level session isolation.
    4. Computes embeddings and stores vectors in ChromaDB.
    """
    text = read_file_text(file_path)

    if not text.strip():
        raise ValueError("No text could be extracted from this file.")

    # Chunking strategy: 900 chars per chunk with 150 chars overlap to preserve context across boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)

    # Attach thread_id to metadata so documents in one chat session don't leak into others
    docs: List[Document] = [
        Document(
            page_content=chunk,
            metadata={
                "thread_id": thread_id,
                "source": Path(file_path).name
            }
        )
        for chunk in chunks
    ]

    # Add chunk documents and their generated embeddings to the persistent store
    vectorstore.add_documents(docs)

    return {
        "filename": Path(file_path).name,
        "chunks": len(docs)
    }


# ---------------------------------------------------------------------------
# Retrieval & Querying
# ---------------------------------------------------------------------------
def retrieve_from_rag(query: str, thread_id: str, k: int = 4) -> str:
    """
    Performs similarity search to find relevant document passages:
    - Filters results strictly by `thread_id` so the agent only accesses documents uploaded in the current chat.
    - Retrieves top-k most similar chunks (default 4).
    - Formats passages with citation sources for the LLM to cite in its response.
    """
    # Similarity search filtered by the current conversation's thread_id
    docs = vectorstore.similarity_search(
        query,
        k=k,
        filter={"thread_id": thread_id}
    )

    if not docs:
        return "No relevant uploaded document content found."

    results = []

    # Format the retrieved chunks with clear source labels
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "uploaded document")
        results.append(
            f"[Source {i}: {source}]\n{doc.page_content}"
        )

    return "\n\n".join(results)