import spacy
from typing import List, Dict, Any
import unicodedata
import re
import uuid
from datetime import datetime
import os

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


def detect_file_type(file_name):
    """
    Detect file type based on extension.

    Args:
        file_name (str): Name of the file

    Returns:
        str: File type ('pdf', 'docx', 'xlsx', or 'unsupported')
    """
    extension = os.path.splitext(file_name.lower())[1]

    if extension == ".pdf":
        return "pdf"
    elif extension == ".docx":
        return "docx"
    elif extension in [".xlsx", ".xls"]:
        return "excel"
    else:
        return "unsupported"


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 75) -> List[str]:
    """
    Split text into chunks using a sliding window approach with overlap.
    Each chunk will be approximately chunk_size characters with overlap characters
    of overlap between consecutive chunks.

    Args:
        text (str): The text to chunk
        chunk_size (int): Target size of each chunk in characters
        overlap (int): Number of characters to overlap between chunks

    Returns:
        List[str]: List of text chunks
    """
    # Process the text with spaCy for sentence boundaries
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_size = len(sentence)

        # If adding this sentence would exceed the chunk size
        if current_size + sentence_size > chunk_size and current_chunk:
            # Join current chunk and add to results
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

            # Start new chunk with overlap
            if overlap > 0:
                # Calculate how many sentences to keep for overlap
                overlap_size = 0
                overlap_sentences = []
                for sent in reversed(current_chunk):
                    if overlap_size + len(sent) <= overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_size += len(sent)
                    else:
                        break

                current_chunk = overlap_sentences
                current_size = overlap_size
            else:
                current_chunk = []
                current_size = 0

        current_chunk.append(sentence)
        current_size += sentence_size

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing.

    Args:
        text (str): Text to clean

    Returns:
        str: Cleaned text
    """
    # Remove extra whitespace
    text = " ".join(text.split())
    return text


def sanitize_vector_id(id_str: str) -> str:
    """
    Sanitize vector ID to ensure it contains only ASCII characters.

    Args:
        id_str (str): The string to sanitize

    Returns:
        str: Sanitized string safe to use as vector ID
    """
    normalized = unicodedata.normalize("NFKD", id_str)
    ascii_str = normalized.encode("ASCII", "ignore").decode("ASCII")
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", ascii_str)
    return sanitized


def prepare_vector_batch(
    document_name: str, chunks: List[str], namespace: str, customer_id: str = None
) -> List[Dict[str, Any]]:
    """
    Prepares a batch of vectors for Pinecone server-side embedding.

    Args:
        document_name (str): Name of the document
        chunks (List[str]): List of text chunks
        namespace (str): Pinecone namespace to use
        customer_id (str, optional): Customer ID for organization

    Returns:
        List[Dict]: List of vector records ready for Pinecone
    """
    batch_id = f"batch_{uuid.uuid4().hex}"
    current_time = datetime.now().isoformat()

    vectors = []
    for i, chunk in enumerate(chunks):
        # Create a unique and sanitized vector ID
        base_id = sanitize_vector_id(document_name)
        vector_id = f"{base_id}_chunk_{i}"

        # Create vector record for server-side embedding
        vector = {
            "id": vector_id,
            "text": chunk,  # For server-side embedding
            "metadata": {
                # Document information
                "document_name": document_name,
                "source_type": "document",
                # Chunk information
                "chunk_index": i,
                "total_chunks": len(chunks),
                # Content information
                "text": chunk,  # Also stored in metadata for retrieval
                # Tracking information
                "upload_timestamp": current_time,
                "batch_id": batch_id,
                # Customer information
                "customer_id": customer_id if customer_id else "",
            },
        }
        vectors.append(vector)

    return vectors
