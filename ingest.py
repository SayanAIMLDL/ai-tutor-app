import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONSTRUCT ABSOLUTE PATHS for robustness ---
project_root = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(project_root, "data", "Document")
CHROMA_PATH = os.path.join(project_root, "chroma")


def clean_text(text: str) -> str:
    """
    Cleans the input text by encoding and decoding it, ignoring errors.
    This effectively removes or replaces characters that can't be encoded in UTF-8.
    """
    return text.encode('utf-8', 'ignore').decode('utf-8')


def main():
    """
    Main function to orchestrate the data ingestion process.
    - Clears any existing database.
    - Loads and cleans PDF documents.
    - Splits them into manageable chunks.
    - Creates and persists a new vector database.
    """
    load_dotenv()
    print("--- Starting Data Ingestion ---")

    # 1. Clear out the old database
    if os.path.exists(CHROMA_PATH):
        print(f"Clearing existing database at {CHROMA_PATH}...")
        shutil.rmtree(CHROMA_PATH)

    # 2. Load and clean documents
    documents = load_documents()
    if not documents:
        print("No documents were found or loaded. Exiting.")
        return

    # 3. Split documents
    text_chunks = split_documents(documents)
    if not text_chunks:
        print("Document splitting resulted in no chunks. Exiting.")
        return
        
    # 4. Create the vector store
    create_vector_store(text_chunks)
    print("--- Ingestion Complete ---")


def load_documents():
    """Loads all PDF documents from the DATA_PATH and cleans their text content."""
    print(f"Loading documents from directory: '{DATA_PATH}'...")
    all_pages = []
    
    if not os.path.isdir(DATA_PATH):
        print(f"Error: The directory '{DATA_PATH}' does not exist.")
        return all_pages
    
    for filename in os.listdir(DATA_PATH):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(DATA_PATH, filename)
            try:
                loader = PyPDFLoader(file_path, extract_images=False)
                # Load pages from the current PDF
                pages = loader.load()
                # Clean the text of each page before adding it
                for page in pages:
                    page.page_content = clean_text(page.page_content)
                all_pages.extend(pages)
                print(f"  - Successfully loaded and cleaned {filename}")
            except Exception as e:
                print(f"  - Error loading {filename}: {e}")
    
    print(f"Loaded a total of {len(all_pages)} pages from all PDF files.")
    return all_pages


def split_documents(documents):
    """Splits the documents into smaller text chunks."""
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    return chunks


def create_vector_store(text_chunks):
    """Creates and persists a Chroma vector store from the text chunks."""
    print("Creating vector store...")
    
    # Initialize Ollama embeddings using the corrected, non-deprecated import
    ollama_embeddings = OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBED_MODEL"),
        base_url=os.getenv("OLLAMA_BASE_URL")
    )
    
    # Create a new Chroma database from the chunks
    Chroma.from_documents(
        documents=text_chunks,
        embedding=ollama_embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"Vector store created at '{CHROMA_PATH}' successfully.")


if __name__ == "__main__":
    main()