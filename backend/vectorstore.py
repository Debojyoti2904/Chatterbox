"""
VECTOR STORE MODULE (BACKEND MEMORY LAYER)
=========================================

PURPOSE:
--------
This module is responsible for:
1. Converting raw text into numerical representations (embeddings)
2. Storing those embeddings in a vector database (Pinecone)
3. Retrieving semantically similar content when queried

In simple terms:
----------------
Text → Numbers → Stored in cloud → Searchable by meaning
"""

# -----------------------------
# Imports
# -----------------------------

import os

# Pinecone SDK:
# Used to connect to Pinecone's cloud-based vector database
from pinecone import Pinecone, ServerlessSpec

# LangChain integration for Pinecone:
# Provides a high-level interface to store and retrieve vectors
from langchain_pinecone import PineconeVectorStore

# HuggingFace embeddings:
# Converts human-readable text into numerical vectors that represent meaning
from langchain_huggingface import HuggingFaceEmbeddings

# Text splitter:
# Breaks large text into smaller, overlapping chunks for better embedding quality
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration import:
# Centralized place where environment variables are defined/validated
from config import PINECONE_API_KEY


# -----------------------------
# Environment Configuration
# -----------------------------

# Fetch Pinecone API key securely from environment variables.
# This avoids hardcoding secrets in the codebase.
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# -----------------------------
# Global Constants
# -----------------------------

# Name of the vector index (acts like a database name).
# All embeddings for this application are stored here.
INDEX_NAME = "rag-index"


# -----------------------------
# Initialize Pinecone Client
# -----------------------------

# This creates an authenticated connection to Pinecone’s cloud service.
# Think of this as "logging into" Pinecone so we can create and query indexes.
pc = Pinecone(api_key=PINECONE_API_KEY)


# -----------------------------
# Define Embedding Model
# -----------------------------

# This model converts text into a fixed-length numerical vector (384 numbers).
# Semantically similar text → similar vectors.
# This is the core intelligence behind semantic search.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# RETRIEVER CREATION (SEARCH SETUP)
# =========================================================

def get_retriever():
    """
    PURPOSE:
    --------
    Ensures a Pinecone index exists and returns a retriever object
    capable of performing semantic similarity search.

    WHAT THIS ENABLES:
    ------------------
    - Search stored text by meaning, not keywords
    - Retrieve the most relevant chunks for a query
    - Act as the "memory lookup" for the backend

    RETURNS:
    --------
    A retriever object that other parts of the backend can use
    without worrying about Pinecone internals.
    """

    # Step 1: Check whether the vector index already exists in Pinecone.
    # This prevents unnecessary re-creation and data loss.
    if INDEX_NAME not in pc.list_indexes().names():
        print("Creating new Pinecone index...")

        # Step 2: Create the index if it does not exist.
        # - dimension=384 must match the embedding model output size
        # - cosine similarity is ideal for text embeddings
        # - serverless spec tells Pinecone where to host the index
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    print("Pinecone index is ready.")

    # Step 3: Connect the embedding model with the Pinecone index.
    # This object knows:
    # - how to embed text
    # - where to store/retrieve vectors
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    # Step 4: Convert vectorstore into a retriever abstraction.
    # A retriever:
    # - accepts a query
    # - embeds it
    # - searches the index
    # - returns relevant chunks
    return vectorstore.as_retriever()


# =========================================================
# DOCUMENT INGESTION (ADDING KNOWLEDGE)
# =========================================================

def add_document_to_vectorstore(text_content: str):
    """
    PURPOSE:
    --------
    Takes raw text content and adds it to the vector database
    so it becomes searchable by semantic meaning.

    PIPELINE OVERVIEW:
    ------------------
    Raw Text
        ↓
    Split into overlapping chunks
        ↓
    Convert chunks into embeddings
        ↓
    Store embeddings in Pinecone

    PARAMETERS:
    -----------
    text_content : str
        The raw document text to be indexed.
    """

    # Safety check:
    # Prevents indexing empty or invalid documents.
    if not text_content:
        raise ValueError("Document content cannot be empty")
    
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Index '{INDEX_NAME}' not found. Creating it...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        import time
        time.sleep(10) # Wait for initialization

    # Step 1: Initialize text splitter.
    # Large documents are split into smaller chunks to:
    # - Preserve semantic clarity
    # - Improve retrieval precision
    # - Avoid embedding model limitations
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # Balanced size for semantic meaning
        chunk_overlap=200,     # Preserves context across chunks
        add_start_index=True   # Enables traceability to original text
    )

    # Step 2: Split raw text into structured document chunks.
    documents = text_splitter.create_documents([text_content])

    print(f"Splitting document into {len(documents)} chunks for indexing")

    # Step 3: Connect to the Pinecone vector store.
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    # Step 4: Embed and store each chunk in the vector database.
    # Pinecone handles indexing and similarity search internally.
    vectorstore.add_documents(documents)

    print(
        f"Successfully added {len(documents)} chunks "
        f"to Pinecone index '{INDEX_NAME}'"
    )
