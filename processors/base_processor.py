import os
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid
import time

from dotenv import load_dotenv
from loguru import logger
from pinecone import Pinecone

# Load environment variables
load_dotenv()


class BaseDocumentProcessor:
    """
    Base class for all document processors. Handles common functionality
    such as chunking, vector creation, and Pinecone uploads.
    """

    def __init__(self, document_registry=None):
        """Initialize with configuration from environment variables."""
        # Chunking parameters
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "75"))
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

        # Document registry for tracking documents
        self.document_registry = document_registry

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logger

    def process_file(
        self,
        file_path: str,
        index=None,
        namespace: str = None,
        customer_id: str = None,
        company_id: str = None,
        title: str = None,
    ) -> bool:
        """
        Process a document file and store its content in Pinecone.

        Args:
            file_path: Path to the document file
            index: Pinecone index object
            namespace: Pinecone namespace
            customer_id: Customer ID for namespace
            company_id: Company ID for the document registry
            title: Document title for the document registry

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Use customer_id as company_id if company_id not provided
            if company_id is None and customer_id is not None:
                company_id = customer_id

            # Extract text from document
            text_content = self.extract_text(file_path)
            if not text_content:
                self.logger.error(f"No text content extracted from {file_path}")
                return False

            # Create chunks
            chunks = self.chunk_text(text_content)
            if not chunks:
                self.logger.error(f"No chunks created from {file_path}")
                return False

            # Get file information
            file_path_obj = Path(file_path)
            filename = file_path_obj.name
            file_type = file_path_obj.suffix.lower()[1:]  # Remove the dot

            # If title not provided, use filename without extension
            if title is None:
                title = file_path_obj.stem

            # Generate document ID
            document_name = f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Register document in registry if available
            document_id = None
            if self.document_registry is not None and company_id is not None:
                # Default namespace if not provided
                if namespace is None:
                    if customer_id:
                        namespace = f"documents_{customer_id}"
                    else:
                        namespace = "documents"

                # Register with initial chunk count, will update after processing
                document_id = self.document_registry.register_document(
                    company_id=company_id,
                    title=title,
                    document_name=document_name,
                    filename=filename,
                    file_type=file_type,
                    total_chunks=len(chunks),
                    embedding_model=self.embedding_model,
                    pinecone_namespace=namespace,
                )

            # Create and upsert vectors
            success = self.create_and_upsert_vectors(
                chunks=chunks,
                file_path=file_path,
                index=index,
                namespace=namespace,
                customer_id=customer_id,
                document_id=document_id,
            )

            if not success:
                self.logger.error(
                    f"Failed to create and upsert vectors for {file_path}"
                )
                return False

            self.logger.info(f"Successfully processed file: {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            return False

    def extract_text(self, file_path: str) -> str:
        """
        Extract text from document. Must be implemented by subclasses.

        Args:
            file_path: Path to the document file

        Returns:
            str: Extracted text content
        """
        raise NotImplementedError("Subclasses must implement extract_text()")

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on configuration.

        Args:
            text: Text to split into chunks

        Returns:
            List[str]: List of text chunks
        """
        if not text.strip():
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Get chunk of size chunk_size
            end = min(start + self.chunk_size, text_length)

            # If this is not the last chunk, try to break at a sentence or paragraph
            if end < text_length:
                # Look for paragraph break
                paragraph_break = text.rfind("\n\n", start, end)
                if (
                    paragraph_break != -1
                    and paragraph_break > start + self.chunk_size // 2
                ):
                    end = paragraph_break + 2  # Include the newlines
                else:
                    # Look for sentence break (period, question mark, exclamation)
                    sentence_break = max(
                        text.rfind(". ", start, end),
                        text.rfind("? ", start, end),
                        text.rfind("! ", start, end),
                    )
                    if (
                        sentence_break != -1
                        and sentence_break > start + self.chunk_size // 2
                    ):
                        end = sentence_break + 2  # Include the period and space

            # Add the chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move to next chunk with overlap
            start = end - self.chunk_overlap if end < text_length else text_length

        return chunks

    def generate_vector_id(
        self, text: str, file_path: str, chunk_index: int, document_id: str = None
    ) -> str:
        """
        Generate a unique ID for the vector.

        Args:
            text: Text chunk
            file_path: Path to the original file
            chunk_index: Index of the chunk
            document_id: Document ID from registry (if available)

        Returns:
            str: Unique ID
        """
        # If document_id is provided, use it with hierarchical structure
        if document_id:
            return f"{document_id}#chunk_{chunk_index}"

        # Otherwise, use legacy method
        filename = Path(file_path).name
        file_type = Path(file_path).suffix.lower()[1:]  # Remove the dot

        # Generate a stable, unique ID
        id_base = f"{file_type}_{filename}_{chunk_index}_{datetime.now().isoformat()}"
        return hashlib.md5(id_base.encode()).hexdigest()

    def create_and_upsert_vectors(
        self,
        chunks: List[str],
        file_path: str,
        index=None,
        namespace: str = None,
        customer_id: str = None,
        document_id: str = None,
    ) -> bool:
        """
        Create and upsert vectors to Pinecone.
        Uses server-side embedding functionality.

        Args:
            chunks: List of text chunks
            file_path: Path to the original file
            index: Pinecone index object
            namespace: Pinecone namespace
            customer_id: Customer ID for namespace
            document_id: Document ID from registry (if available)

        Returns:
            bool: True if successful, False otherwise
        """
        if not chunks:
            self.logger.warning("No chunks to process")
            return False

        if index is None:
            self.logger.error("No index provided")
            return False

        # Default namespace if not provided
        if namespace is None:
            if customer_id:
                namespace = f"documents_{customer_id}"
            else:
                namespace = "documents"

        try:
            # Get file metadata
            file_path_obj = Path(file_path)
            filename = file_path_obj.name
            file_type = file_path_obj.suffix.lower()[1:]  # Remove the dot

            # Create a batch ID for tracking this upload
            batch_id = f"batch_{uuid.uuid4().hex}"
            current_time = datetime.now().isoformat()

            # Create records with metadata for server-side embedding
            formatted_records = []
            for i, chunk in enumerate(chunks):
                # Create a unique vector ID
                vector_id = self.generate_vector_id(chunk, file_path, i, document_id)

                # Format record for upsert_records with server-side embedding
                formatted_record = {
                    # ID field for upsert_records
                    "_id": vector_id,
                    # Text field for embedding
                    "chunk_text": chunk,
                    # Additional metadata
                    "filename": filename,
                    "file_type": file_type,
                    "original_path": str(file_path_obj.absolute()),
                    "source_type": f"local_{file_type}",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk),
                    "token_count": len(chunk.split()),
                    "upload_timestamp": current_time,
                    "batch_id": batch_id,
                    "customer_id": customer_id if customer_id else "",
                    "document_id": document_id if document_id else "",
                }

                formatted_records.append(formatted_record)

            # Batch records into smaller chunks for processing
            batch_size = 100
            total_batches = (len(formatted_records) + batch_size - 1) // batch_size
            total_upserted = 0

            # Upload batches using upsert_records for server-side embedding
            for i in range(total_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(formatted_records))
                batch = formatted_records[start_idx:end_idx]

                # Use upsert_records which handles server-side embedding
                self.logger.info(
                    f"Upserting batch {i+1}/{total_batches} "
                    f"with {len(batch)} records"
                )
                index.upsert_records(namespace, batch)

                total_upserted += len(batch)

                # Small pause between batches to avoid rate limiting
                if i < total_batches - 1:
                    time.sleep(0.5)

            self.logger.info(
                f"Successfully upserted {total_upserted} vectors "
                f"to Pinecone {namespace} namespace"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error creating and upserting vectors: {str(e)}")
            return False

    def dry_run(
        self,
        file_path: str,
        output_dir: str = "dry_run_output",
        company_id: str = None,
        title: str = None,
    ) -> bool:
        """
        Process a document without upserting to Pinecone.
        Instead, save the chunks as markdown files for review.

        Args:
            file_path: Path to the document file
            output_dir: Directory to save chunk files
            company_id: Company ID for the document registry
            title: Document title for the document registry

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Extract text
            text_content = self.extract_text(file_path)
            if not text_content:
                self.logger.error(f"No text content extracted from {file_path}")
                return False

            # Create chunks
            chunks = self.chunk_text(text_content)
            if not chunks:
                self.logger.error(f"No chunks created from {file_path}")
                return False

            # Ensure output directory exists
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)

            # Generate base filename
            base_filename = Path(file_path).stem
            file_type = Path(file_path).suffix.lower()[1:]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # If title not provided, use filename without extension
            if title is None:
                title = base_filename

            # Generate mock document ID for preview
            if company_id:
                mock_document_id = f"{company_id}#{title}#{base_filename}_{timestamp}"
                self.logger.info(f"Dry run document ID structure: {mock_document_id}")

            # Write each chunk to a file
            for i, chunk in enumerate(chunks):
                # Format chunk for markdown
                md_content = self.format_chunk_for_markdown(
                    chunk, file_path, i, len(chunks), company_id=company_id, title=title
                )

                # Create filename
                filename = (
                    f"{base_filename}_chunk_{i+1}_of_{len(chunks)}_{timestamp}.md"
                )
                file_path = output_path / filename

                # Write to file
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(md_content)

                self.logger.info(f"Saved chunk {i+1}/{len(chunks)} to {file_path}")

            return True

        except Exception as e:
            self.logger.error(f"Error in dry run for {file_path}: {str(e)}")
            return False

    def format_chunk_for_markdown(
        self,
        chunk: str,
        file_path: str,
        chunk_index: int,
        total_chunks: int,
        company_id: str = None,
        title: str = None,
    ) -> str:
        """
        Format a chunk as a markdown file with metadata.

        Args:
            chunk: Text chunk
            file_path: Path to the original file
            chunk_index: Index of the chunk
            total_chunks: Total number of chunks
            company_id: Company ID for the document registry
            title: Document title for the document registry

        Returns:
            str: Formatted markdown content
        """
        filename = Path(file_path).name
        file_type = Path(file_path).suffix.lower()[1:]
        timestamp = datetime.now().isoformat()

        # If title not provided, use filename without extension
        if title is None:
            title = Path(file_path).stem

        # Format metadata
        metadata = {
            "filename": filename,
            "file_type": file_type,
            "chunk_index": chunk_index + 1,
            "total_chunks": total_chunks,
            "chunk_size": len(chunk),
            "token_count": len(chunk.split()),
            "timestamp": timestamp,
        }

        # Add company and title info if available
        if company_id:
            metadata["company_id"] = company_id
        if title:
            metadata["title"] = title

        # Add the document ID structure preview
        if company_id and title:
            doc_name = (
                f"{Path(file_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            metadata["document_id_structure"] = (
                f"{company_id}#{title}#{doc_name}#chunk_{chunk_index}"
            )

        # Create markdown content
        md_content = "# Document Chunk Preview\n\n"
        md_content += "## Metadata\n\n"

        for key, value in metadata.items():
            md_content += f"- **{key}**: {value}\n"

        md_content += "\n## Content\n\n"
        md_content += f"```\n{chunk}\n```\n"

        return md_content
