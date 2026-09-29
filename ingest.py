import os
import sys
import logging
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ingest")


def load_and_split_pdf(pdf_path: str) -> List[Document]:
    """Load PDF using PyPDFLoader and chunk text using RecursiveCharacterTextSplitter."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

    logger.info(f"Loading PDF document from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    logger.info(f"Successfully loaded {len(pages)} pages from PDF.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(pages)
    logger.info(f"Created {len(chunks)} chunks using chunk_size={settings.chunk_size} & chunk_overlap={settings.chunk_overlap}.")
    return chunks


def setup_pinecone_index(index_name: str, dimension: int = 1536):
    """Ensure Pinecone index exists with correct dimension and metric."""
    from pinecone import Pinecone, ServerlessSpec

    if not settings.pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable is missing.")

    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        logger.info(f"Creating new Pinecone serverless index: {index_name} (dim={dimension})...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        logger.info(f"Pinecone index '{index_name}' created successfully.")
    else:
        logger.info(f"Pinecone index '{index_name}' already exists.")


def ingest_documents(pdf_path: str = "ebook.pdf"):
    """Run full ingestion pipeline: PDF -> Chunking -> Embeddings -> Pinecone."""
    try:
        chunks = load_and_split_pdf(pdf_path)

        if not settings.openai_api_key or not settings.pinecone_api_key:
            logger.warning("API keys not fully set in environment. Skipping Pinecone upload.")
            logger.info("PDF processing and chunking verified successfully.")
            return chunks

        # 1. Setup Pinecone Index
        setup_pinecone_index(settings.pinecone_index_name, dimension=1536)

        # 2. Upload Chunks & Embeddings
        from langchain_pinecone import PineconeVectorStore
        from app.vector_store import get_embeddings

        embeddings = get_embeddings()
        logger.info(f"Upserting {len(chunks)} document chunks to Pinecone index '{settings.pinecone_index_name}'...")

        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=settings.pinecone_index_name,
            pinecone_api_key=settings.pinecone_api_key
        )

        logger.info("🎉 Ingestion complete! Chunks successfully stored in Pinecone.")
        return chunks

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise e


if __name__ == "__main__":
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else settings.pdf_path
    ingest_documents(pdf_file)
