import os
import time
import uuid
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger.add("pdf_processor.log", rotation="10 MB")


class PdfProcessor:
    """
    Processes PDF documents and uploads to Pinecone.

    This is a wrapper around the existing PDF parser that adapts it to
    our Pinecone integration requirements.
    """

    def __init__(self):
        # Chunking parameters
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "75"))

    def process_file(
        self, file_path: str, index=None, namespace: str = None, customer_id: str = None
    ) -> bool:
        """
        Process a PDF file and store its content in Pinecone.

        Args:
            file_path: Path to the PDF file
            index: Pinecone index object
            namespace: Pinecone namespace
            customer_id: Customer ID for namespace

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Extract text from PDF
            text_content = self._extract_text_from_pdf(file_path)
            if not text_content:
                logger.error("No text content extracted from PDF")
                return False

            # Create chunks
            chunks = self._chunk_text(text_content)

            # Create and upsert vectors
            if not self._create_and_upsert_vectors(
                chunks, index, namespace, customer_id
            ):
                logger.error("Failed to create and upsert vectors")
                return False

            logger.info(f"Successfully processed PDF file: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error processing PDF file: {str(e)}")
            return False

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file with improved cleaning.

        Args:
            file_path: Path to the PDF file

        Returns:
            str: Extracted and cleaned text content
        """
        try:
            # Try to use the complex PDF parser
            try:
                from RAGFlowPdfParser import RAGFlowPdfParser, PlainParser

                # Choose the appropriate parser
                use_complex_parser = True
                if use_complex_parser:
                    parser = RAGFlowPdfParser()
                else:
                    parser = PlainParser()

                # Extract text from PDF
                text_content, tables = parser(file_path)
                if not text_content:
                    logger.warning("No text content extracted from PDF")
                    return ""

                # Process extracted text
                if isinstance(text_content, list):
                    # If text_content is a list of (text, position) tuples
                    text_parts = [text for text, _ in text_content if text]
                    # Join with a single space instead of newlines for better readability
                    text_content = " ".join(text_parts)
                elif isinstance(text_content, str):
                    # If text_content is already a string
                    pass
                else:
                    logger.warning(
                        f"Unexpected text content type: {type(text_content)}"
                    )
                    return ""

                # Enhanced text cleaning
                # 1. Replace multiple spaces with a single space
                # 2. Replace multiple newlines with a single newline
                # 3. Remove excessive whitespace at the beginning and end of lines
                text_content = " ".join(text_content.split())

                # Process tables if needed
                if tables and isinstance(tables, list) and len(tables) > 0:
                    table_texts = []
                    for table_data in tables:
                        if isinstance(table_data, tuple) and len(table_data) > 1:
                            # Clean up table text as well
                            table_text = str(table_data[1])
                            table_text = " ".join(table_text.split())
                            table_texts.append(table_text)

                    if table_texts:
                        text_content += "\n\nTables:\n" + "\n\n".join(table_texts)

                return text_content

            except ImportError:
                logger.warning("RAGFlowPdfParser not available, using fallback method")
                return self._use_fallback_parser(file_path)

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    def _use_fallback_parser(self, file_path: str) -> str:
        """
        Use a simple fallback parser when the complex one is not available.

        Args:
            file_path: Path to the PDF file

        Returns:
            str: Extracted and cleaned text content
        """
        try:
            from PyPDF2 import PdfReader

            with open(file_path, "rb") as file:
                pdf_reader = PdfReader(file)
                text_parts = []

                # Process each page
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text and text.strip():
                        # Enhanced text cleaning
                        text = " ".join(text.split())  # Remove extra whitespace
                        text_parts.append(text)

                # Join pages with double newlines for better separation
                return "\n\n".join(text_parts)

        except Exception as e:
            logger.error(f"Error in fallback parser for {file_path}: {str(e)}")
            return ""

    def _process_text_content(self, text_content, file_path: str) -> None:
        """
        Process extracted text content and upload to Pinecone.

        Args:
            text_content: Extracted text content from the PDF
            file_path: Path to the PDF file
        """
        # Base metadata for all chunks from this file
        metadata_base = {
            "filename": Path(file_path).name,
            "file_type": "pdf",
            "original_path": str(Path(file_path).absolute()),
            "source_type": "local_pdf",
            "upload_timestamp": datetime.now().isoformat(),
        }

        # Process the text content
        all_chunks = []
        for i, (text, position) in enumerate(text_content):
            if not text:
                continue

            # Add position information if available
            chunk_metadata = metadata_base.copy()
            if position:
                chunk_metadata["position"] = position

            # Chunk the text (if it's not already chunked)
            chunks = self._chunk_text(text)
            all_chunks.extend(chunks)

            # Create and upsert vectors in batches
            if len(all_chunks) >= 50:
                self._create_and_upsert_vectors(all_chunks, chunk_metadata)
                all_chunks = []

        # Process any remaining chunks
        if all_chunks:
            self._create_and_upsert_vectors(all_chunks, chunk_metadata)

    def _process_tables(self, tables, file_path: str) -> None:
        """
        Process extracted tables and upload to Pinecone.

        Args:
            tables: Extracted tables from the PDF
            file_path: Path to the PDF file
        """
        # Base metadata for all chunks from this file
        metadata_base = {
            "filename": Path(file_path).name,
            "file_type": "pdf",
            "original_path": str(Path(file_path).absolute()),
            "source_type": "local_pdf_table",
            "upload_timestamp": datetime.now().isoformat(),
        }

        # Process each table
        for i, table_data in enumerate(tables):
            # Extract table content
            table_content = (
                str(table_data[1])
                if isinstance(table_data, tuple) and len(table_data) > 1
                else ""
            )

            if not table_content:
                continue

            # Add table information to metadata
            table_metadata = metadata_base.copy()
            table_metadata["content_type"] = "table"
            table_metadata["table_index"] = i

            # Chunk the table content
            chunks = self._chunk_text(table_content)

            # Create and upsert vectors
            self._create_and_upsert_vectors(chunks, table_metadata)

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on configuration.

        Args:
            text: The text to chunk

        Returns:
            List of text chunks
        """
        if not text or len(text.strip()) == 0:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Calculate end position with overlap
            end = min(start + self.chunk_size, text_length)

            # Get the chunk
            chunk = text[start:end]

            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk)

            # Move start position for next chunk
            start = end - self.chunk_overlap if end < text_length else text_length

        return chunks

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using Pinecone's server-side embedding.

        Args:
            text: Text to embed

        Returns:
            Vector embedding
        """
        try:
            # Use server-side embedding by passing the text directly
            # The embedding will be generated by Pinecone
            return text
        except Exception as e:
            logger.error("Error preparing text for server-side embedding: " f"{str(e)}")
            raise

    def _generate_vector_id(self, text: str, metadata: Dict[str, Any]) -> str:
        """
        Generate a unique ID for a vector.

        Args:
            text: The text content
            metadata: Metadata for the vector

        Returns:
            Unique vector ID
        """
        # Create a unique hash based on content and metadata
        content_to_hash = text

        # Add key metadata to the hash
        filename = metadata.get("filename", "")
        if filename:
            content_to_hash += filename

        chunk_index = metadata.get("chunk_index", 0)
        content_to_hash += str(chunk_index)

        # Generate a hash
        content_hash = hashlib.md5(content_to_hash.encode("utf-8")).hexdigest()

        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Combine with a UUID component
        unique_id = f"file_{content_hash}_{timestamp}_{str(uuid.uuid4())[:8]}"

        return unique_id

    def _create_and_upsert_vectors(
        self,
        chunks: List[str],
        index=None,
        namespace: str = None,
        customer_id: str = None,
    ) -> bool:
        """
        Create vectors from chunks and upsert to Pinecone with improved metadata.

        Args:
            chunks: List of text chunks
            index: Pinecone index object
            namespace: Pinecone namespace
            customer_id: Customer ID for namespace

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            vectors = []
            timestamp = datetime.now().isoformat()

            for i, chunk in enumerate(chunks):
                # Create enhanced metadata for each chunk
                metadata = {
                    "text": chunk,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "upload_timestamp": timestamp,
                    "source_type": "local_pdf",
                }

                # Generate a unique ID for the vector
                vector_id = f"pdf_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}_{int(time.time())}"

                # Create vector with properly formatted values for server-side embedding
                vector = {
                    "id": vector_id,
                    "values": [0.1]
                    * 1024,  # Placeholder values for server-side embedding
                    "metadata": metadata,
                }
                vectors.append(vector)

            # Upsert vectors to Pinecone
            if index and vectors:
                # Use customer-specific namespace if provided
                effective_namespace = namespace
                if customer_id and namespace:
                    effective_namespace = f"{namespace}_{customer_id}"

                # Use batch processing for better performance
                batch_size = 100
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i : i + batch_size]
                    index.upsert(vectors=batch, namespace=effective_namespace)
                    logger.info(f"Upserted batch of {len(batch)} vectors to Pinecone")

                logger.info(
                    f"Successfully upserted total of {len(vectors)} vectors to Pinecone"
                )
            else:
                if not index:
                    logger.warning("No Pinecone index provided, skipping upsert")
                if not vectors:
                    logger.warning("No vectors to upsert")

            return True

        except Exception as e:
            logger.error(f"Error upserting vectors: {str(e)}")
            return False

    def dry_run(self, file_path: str, output_dir: str = "dry_run_output") -> bool:
        """
        Process the file without uploading to Pinecone, saving chunks as structured markdown files.

        Args:
            file_path: Path to the PDF file
            output_dir: Directory to save the markdown files

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            # Extract text from PDF with improved cleaning
            text_content = self._extract_text_from_pdf(file_path)
            if not text_content:
                logger.error("No text content extracted from PDF")
                return False

            # Create chunks with appropriate size and overlap
            chunks = self._chunk_text(text_content)
            if not chunks:
                logger.warning(f"No chunks created from {file_path}")
                return False

            # Common timestamp for all chunks in this file
            timestamp = datetime.now().isoformat()

            # Get file information
            file_name = Path(file_path).name
            file_stem = Path(file_path).stem
            file_path_abs = str(Path(file_path).absolute())

            # Save chunks to markdown files with comprehensive metadata
            for i, chunk in enumerate(chunks):
                # Create comprehensive metadata for each chunk
                metadata = {
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "filename": file_name,
                    "file_type": "pdf",
                    "original_path": file_path_abs,
                    "source_type": "local_pdf",
                    "upload_timestamp": timestamp,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "chunk_length_chars": len(chunk),
                    "chunk_length_tokens": len(chunk.split()),
                }

                # Create filename with chunk number
                filename = f"{file_stem}_chunk_{i+1}_of_{len(chunks)}.md"
                chunk_path = output_path / filename

                # Format chunk for better readability
                formatted_chunk = self._format_chunk_for_markdown(chunk)

                # Write chunk to file with improved formatting
                with open(chunk_path, "w", encoding="utf-8") as f:
                    f.write(f"# {file_name} - Chunk {i+1} of {len(chunks)}\n\n")
                    f.write("## Content\n\n")
                    f.write(formatted_chunk)
                    f.write("\n\n## Metadata\n\n")

                    # Write metadata in a clean, organized format
                    for key, value in sorted(metadata.items()):
                        f.write(f"- **{key}**: {value}\n")

            logger.info(
                f"Dry run completed. {len(chunks)} chunks saved to {output_dir}"
            )
            return True

        except Exception as e:
            logger.error(f"Error in dry run: {str(e)}")
            return False

    def _format_chunk_for_markdown(self, chunk: str) -> str:
        """
        Format a text chunk for better readability in markdown.

        Args:
            chunk: The text chunk to format

        Returns:
            Formatted text suitable for markdown display
        """
        # Ensure the chunk is properly formatted for markdown
        # 1. Replace newlines with double newlines for proper paragraph breaks
        # 2. Escape any markdown special characters if needed

        # Wrap the content in a markdown code block if it contains special characters
        # that might interfere with markdown rendering
        special_chars = ["#", "*", "_", "`", "[", "]", "(", ")", "<", ">", "|"]
        needs_code_block = any(char in chunk for char in special_chars)

        if needs_code_block:
            return f"```\n{chunk}\n```"

        # Otherwise, just ensure paragraphs are properly separated
        formatted = chunk.replace("\n", "\n\n")
        return formatted


def process_pdf(file_path: str) -> None:
    """
    Process a PDF file and upload to Pinecone.

    Args:
        file_path: Path to the PDF file
    """
    processor = PdfProcessor()
    processor.process_file(file_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process a PDF file and upload to Pinecone"
    )
    parser.add_argument("file_path", help="Path to the PDF file")

    args = parser.parse_args()
    process_pdf(args.file_path)

    args = parser.parse_args()
    process_pdf(args.file_path)
