#!/usr/bin/env python3
"""
Document Processor

A central script to process documents of various types (PDF, Word, Excel)
and upload their content to Pinecone as vector embeddings.

This script can be used directly from the command line or imported by the
API and Streamlit interfaces.
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
from typing import Optional, List, Dict, Any
import argparse
import sys

from processors import get_processor_for_file, DocumentRegistry, DocumentManager
from loguru import logger

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger.add("document_processor.log", rotation="10 MB")


class DocumentProcessorService:
    """
    Service for processing documents and uploading to Pinecone.

    This class centralizes the document processing functionality used by
    the CLI, API, and Streamlit interfaces.
    """

    def __init__(self, dry_run: bool = False):
        """
        Initialize the document processor service.

        Args:
            dry_run: If True, don't upload to Pinecone, instead save chunks locally
        """
        self.dry_run = dry_run

        # Load configuration from environment variables
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "document-embeddings")
        self.namespace = os.getenv("PINECONE_NAMESPACE", "documents")
        self.customer_id = os.getenv("CUSTOMER_ID", "default")
        self.index_host = os.getenv("PINECONE_INDEX_HOST")

        # Only initialize Pinecone if not in dry run mode
        self.pc = None
        self.index = None

        if not dry_run:
            self._initialize_pinecone()

    def _initialize_pinecone(self):
        """Initialize connection to Pinecone."""
        try:
            # Initialize Pinecone using the documentation example
            api_key = os.getenv("PINECONE_API_KEY")

            # Initialize Pinecone client
            self.pc = Pinecone(api_key=api_key)

            if self.index_host:
                # Connect directly using the provided host
                self.index = self.pc.Index(name=self.index_name, host=self.index_host)
            else:
                # Get the index info to retrieve the host
                try:
                    index_info = self.pc.describe_index(self.index_name)
                    self.index_host = index_info.host
                    logger.info(f"Retrieved index host: {self.index_host}")
                    self.index = self.pc.Index(
                        name=self.index_name, host=self.index_host
                    )
                except Exception as e:
                    logger.error(
                        f"Error connecting to Pinecone index '{self.index_name}': {e}"
                    )
                    raise
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise

    def process_document(self, file_path: str) -> bool:
        """
        Process a document and upload to Pinecone (or save locally in dry run mode).

        Args:
            file_path: Path to the document to process

        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            file_path_obj = Path(file_path)

            if not file_path_obj.exists():
                logger.error(f"File not found: {file_path}")
                return False

            # Get processor for this file type
            processor = get_processor_for_file(str(file_path))

            if not processor:
                logger.error(f"Unsupported file type: {file_path_obj.suffix}")
                return False

            logger.info(
                f"Processing {file_path_obj.name} using {processor.__class__.__name__}"
            )

            # Process using dry run or normal mode
            if self.dry_run:
                return processor.dry_run(file_path)
            else:
                return processor.process_file(
                    file_path,
                    index=self.index,
                    namespace=self.namespace,
                    customer_id=self.customer_id,
                )

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return False

    def process_documents(self, file_paths: List[str]) -> Dict[str, bool]:
        """
        Process multiple documents.

        Args:
            file_paths: List of paths to documents to process

        Returns:
            Dict[str, bool]: Dictionary of file paths and success status
        """
        results = {}

        for file_path in file_paths:
            results[file_path] = self.process_document(file_path)

        return results


def setup_logging():
    """Configure logging"""
    logging.basicConfig(level=logging.INFO)


def get_pinecone_index(index_name: Optional[str] = None):
    """
    Initialize and return a Pinecone index.

    Args:
        index_name: Name of index to use. If None, uses PINECONE_INDEX_NAME from env.

    Returns:
        Pinecone index instance
    """
    # Get Pinecone credentials from environment variables
    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        logger.error("PINECONE_API_KEY environment variable not set")
        sys.exit(1)

    # Use provided index name or get from environment
    if not index_name:
        index_name = os.getenv("PINECONE_INDEX_NAME")

    if not index_name:
        logger.error("PINECONE_INDEX_NAME environment variable not set")
        sys.exit(1)

    # Initialize Pinecone
    pc = Pinecone(api_key=api_key)

    # Check if index exists
    try:
        indexes = pc.list_indexes()
        if index_name not in [idx["name"] for idx in indexes]:
            logger.error(
                f"Index '{index_name}' does not exist. Please create it before processing documents."
            )
            sys.exit(1)

        # Get index
        return pc.Index(index_name)
    except Exception as e:
        logger.error(f"Error connecting to Pinecone: {str(e)}")
        sys.exit(1)


def process_document(
    file_path: str,
    dry_run: bool = False,
    index_name: Optional[str] = None,
    namespace: Optional[str] = None,
    customer_id: Optional[str] = None,
    company_id: Optional[str] = None,
    title: Optional[str] = None,
) -> bool:
    """
    Process a document file and either upload to Pinecone or run in dry-run mode.

    Args:
        file_path: Path to the document file
        dry_run: If True, run in dry-run mode without uploading to Pinecone
        index_name: Name of Pinecone index to use
        namespace: Pinecone namespace to use
        customer_id: Customer ID for namespace
        company_id: Company ID for document registry
        title: Title for the document

    Returns:
        bool: True if processing was successful, False otherwise
    """
    # Check if file exists
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False

    # Initialize document registry
    registry = DocumentRegistry()

    # Use customer_id as company_id if company_id not provided
    if company_id is None and customer_id is not None:
        company_id = customer_id

    # Get processor for file type
    try:
        processor = get_processor_for_file(str(file_path), document_registry=registry)
    except ValueError as e:
        logger.error(str(e))
        return False

    # Determine if we're using dry-run mode
    if dry_run:
        # Run in dry-run mode (no Pinecone upload)
        dry_run_dir = os.getenv("DRY_RUN_OUTPUT_DIR", "dry_run_output")

        logger.info(f"Running in dry-run mode. Saving chunks to '{dry_run_dir}'")
        success = processor.dry_run(
            str(file_path), output_dir=dry_run_dir, company_id=company_id, title=title
        )

        if success:
            logger.info(f"Dry run completed successfully for {file_path}")
        else:
            logger.error(f"Dry run failed for {file_path}")

        return success
    else:
        # Process document and upload to Pinecone
        index = get_pinecone_index(index_name)

        # Use default namespace if not provided
        if not namespace and customer_id:
            namespace = f"documents_{customer_id}"
        elif not namespace:
            namespace = "documents"

        logger.info(
            f"Processing {file_path} and uploading to Pinecone namespace '{namespace}'"
        )
        success = processor.process_file(
            str(file_path),
            index=index,
            namespace=namespace,
            customer_id=customer_id,
            company_id=company_id,
            title=title,
        )

        if success:
            logger.info(f"Successfully processed {file_path}")
        else:
            logger.error(f"Failed to process {file_path}")

        return success


def process_directory(
    directory_path: str,
    dry_run: bool = False,
    index_name: Optional[str] = None,
    namespace: Optional[str] = None,
    customer_id: Optional[str] = None,
    company_id: Optional[str] = None,
    title_prefix: Optional[str] = None,
) -> bool:
    """
    Process all documents in a directory.

    Args:
        directory_path: Path to directory containing documents
        dry_run: If True, run in dry-run mode without uploading to Pinecone
        index_name: Name of Pinecone index to use
        namespace: Pinecone namespace to use
        customer_id: Customer ID for namespace
        company_id: Company ID for document registry
        title_prefix: Prefix to add to all document titles

    Returns:
        bool: True if all documents were processed successfully, False otherwise
    """
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        logger.error(f"Directory not found: {directory}")
        return False

    # Supported file extensions
    supported_extensions = [".pdf", ".docx", ".doc", ".xlsx", ".xls"]

    # Process all files in directory with supported extensions
    success_count = 0
    failure_count = 0

    for file_path in directory.glob("**/*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            # Generate title from filename if title_prefix is provided
            title = None
            if title_prefix:
                title = f"{title_prefix}_{file_path.stem}"

            # Process file
            success = process_document(
                str(file_path),
                dry_run=dry_run,
                index_name=index_name,
                namespace=namespace,
                customer_id=customer_id,
                company_id=company_id,
                title=title,
            )

            if success:
                success_count += 1
            else:
                failure_count += 1

    total = success_count + failure_count
    logger.info(
        f"Processed {total} files: {success_count} succeeded, {failure_count} failed"
    )

    # Return True if all files were processed successfully
    return failure_count == 0


def delete_document(
    document_id: str,
    index_name: Optional[str] = None,
) -> bool:
    """
    Delete a document from Pinecone and the document registry.

    Args:
        document_id: ID of document to delete
        index_name: Name of Pinecone index

    Returns:
        bool: True if deletion was successful, False otherwise
    """
    # Initialize document registry and manager
    registry = DocumentRegistry()
    manager = DocumentManager(registry, index_name=index_name)

    # Delete document
    success, message = manager.delete_document(document_id)

    if success:
        logger.info(message)
    else:
        logger.error(message)

    return success


def delete_company_documents(
    company_id: str,
    index_name: Optional[str] = None,
) -> bool:
    """
    Delete all documents for a company.

    Args:
        company_id: Company ID
        index_name: Name of Pinecone index

    Returns:
        bool: True if deletion was successful, False otherwise
    """
    # Initialize document registry and manager
    registry = DocumentRegistry()
    manager = DocumentManager(registry, index_name=index_name)

    # Delete company documents
    success, message = manager.delete_company_documents(company_id)

    if success:
        logger.info(message)
    else:
        logger.error(message)

    return success


def delete_documents_by_title(
    company_id: str,
    title: str,
    index_name: Optional[str] = None,
) -> bool:
    """
    Delete all documents with a specific title for a company.

    Args:
        company_id: Company ID
        title: Document title
        index_name: Name of Pinecone index

    Returns:
        bool: True if deletion was successful, False otherwise
    """
    # Initialize document registry and manager
    registry = DocumentRegistry()
    manager = DocumentManager(registry, index_name=index_name)

    # Delete documents by title
    success, message = manager.delete_documents_by_title(company_id, title)

    if success:
        logger.info(message)
    else:
        logger.error(message)

    return success


def list_company_documents(
    company_id: str,
) -> List[dict]:
    """
    List all documents for a company.

    Args:
        company_id: Company ID

    Returns:
        List of document metadata dictionaries
    """
    # Initialize document registry
    registry = DocumentRegistry()

    # Get company documents
    documents = registry.get_company_documents(company_id)

    # Print document information
    if documents:
        logger.info(f"Found {len(documents)} documents for company {company_id}:")
        for doc in documents:
            logger.info(f"ID: {doc['id']}")
            logger.info(f"  Title: {doc['title']}")
            logger.info(f"  Filename: {doc['original_filename']}")
            logger.info(f"  Type: {doc['file_type']}")
            logger.info(f"  Uploaded: {doc['upload_date']}")
            logger.info(f"  Chunks: {doc['total_chunks']}")
            logger.info("---")
    else:
        logger.info(f"No documents found for company {company_id}")

    return documents


def main():
    """Main function to handle command-line processing"""
    setup_logging()

    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Process documents and upload to Pinecone vector database"
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process documents")
    process_parser.add_argument("path", help="Path to file or directory to process")
    process_parser.add_argument(
        "--dry-run", action="store_true", help="Run without uploading to Pinecone"
    )
    process_parser.add_argument("--index", help="Pinecone index name")
    process_parser.add_argument("--namespace", help="Pinecone namespace")
    process_parser.add_argument("--customer-id", help="Customer ID for namespace")
    process_parser.add_argument("--company-id", help="Company ID for document registry")
    process_parser.add_argument("--title", help="Document title")
    process_parser.add_argument(
        "--title-prefix", help="Prefix to add to all document titles (for directories)"
    )

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete documents")
    delete_subparsers = delete_parser.add_subparsers(
        dest="delete_type", help="Type of deletion"
    )

    # Delete by document ID
    delete_doc_parser = delete_subparsers.add_parser(
        "document", help="Delete document by ID"
    )
    delete_doc_parser.add_argument("document_id", help="Document ID to delete")
    delete_doc_parser.add_argument("--index", help="Pinecone index name")

    # Delete by company ID
    delete_company_parser = delete_subparsers.add_parser(
        "company", help="Delete all documents for a company"
    )
    delete_company_parser.add_argument("company_id", help="Company ID")
    delete_company_parser.add_argument("--index", help="Pinecone index name")

    # Delete by title
    delete_title_parser = delete_subparsers.add_parser(
        "title", help="Delete documents by title"
    )
    delete_title_parser.add_argument("company_id", help="Company ID")
    delete_title_parser.add_argument("title", help="Document title")
    delete_title_parser.add_argument("--index", help="Pinecone index name")

    # List command
    list_parser = subparsers.add_parser("list", help="List documents")
    list_subparsers = list_parser.add_subparsers(
        dest="list_type", help="Type of listing"
    )

    # List by company ID
    list_company_parser = list_subparsers.add_parser(
        "company", help="List all documents for a company"
    )
    list_company_parser.add_argument("company_id", help="Company ID")

    # Parse arguments
    args = parser.parse_args()

    if args.command == "process":
        # Check if path is a file or directory
        path = Path(args.path)

        if path.is_file():
            # Process single file
            success = process_document(
                str(path),
                dry_run=args.dry_run,
                index_name=args.index,
                namespace=args.namespace,
                customer_id=args.customer_id,
                company_id=args.company_id,
                title=args.title,
            )
        elif path.is_dir():
            # Process directory
            success = process_directory(
                str(path),
                dry_run=args.dry_run,
                index_name=args.index,
                namespace=args.namespace,
                customer_id=args.customer_id,
                company_id=args.company_id,
                title_prefix=args.title_prefix,
            )
        else:
            logger.error(f"Path not found: {path}")
            return 1

        return 0 if success else 1

    elif args.command == "delete":
        if args.delete_type == "document":
            # Delete document by ID
            success = delete_document(args.document_id, index_name=args.index)

        elif args.delete_type == "company":
            # Delete all documents for a company
            success = delete_company_documents(args.company_id, index_name=args.index)

        elif args.delete_type == "title":
            # Delete documents by title
            success = delete_documents_by_title(
                args.company_id, args.title, index_name=args.index
            )

        else:
            parser.print_help()
            return 1

        return 0 if success else 1

    elif args.command == "list":
        if args.list_type == "company":
            # List all documents for a company
            documents = list_company_documents(args.company_id)
            return 0 if documents is not None else 1

        else:
            parser.print_help()
            return 1

    else:
        # If no command specified, print help
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

# Explicitly export the DocumentProcessorService class
__all__ = [
    "DocumentProcessorService",
    "process_document",
    "process_directory",
    "get_pinecone_index",
]
