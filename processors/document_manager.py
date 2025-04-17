import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from uuid import uuid4
from pinecone import Pinecone
from .document_registry import DocumentRegistry
from loguru import logger


class DocumentManager:
    """Manager for document operations in Pinecone vector database"""

    def __init__(
        self,
        registry: DocumentRegistry,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        index_name: Optional[str] = None,
    ):
        """Initialize document manager with Pinecone connection

        Args:
            registry: DocumentRegistry instance for metadata tracking
            api_key: Pinecone API key. If None, uses PINECONE_API_KEY environment variable.
            environment: Pinecone environment. If None, uses PINECONE_ENVIRONMENT environment variable.
            index_name: Name of the Pinecone index. If None, uses PINECONE_INDEX_NAME environment variable.
        """
        self.registry = registry
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
        self.index_name = index_name or os.environ.get("PINECONE_INDEX_NAME")

        if not self.api_key:
            raise ValueError(
                "Pinecone API key not provided and PINECONE_API_KEY environment variable not set"
            )

        if not self.index_name:
            raise ValueError(
                "Pinecone index name not provided and PINECONE_INDEX_NAME environment variable not set"
            )

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)

        # Connect to index
        try:
            indexes = self.pc.list_indexes()
            if self.index_name not in [idx["name"] for idx in indexes]:
                raise ValueError(
                    f"Pinecone index '{self.index_name}' does not exist. Please create it before using DocumentManager."
                )

            # Get index
            self.index = self.pc.Index(self.index_name)
        except Exception as e:
            logger.error(f"Error connecting to Pinecone: {e}")
            raise

    def delete_document(self, document_id: str) -> Tuple[bool, str]:
        """Delete a specific document by its ID

        Args:
            document_id: Document ID to delete

        Returns:
            Tuple of (success, message)
        """
        # Get document info from registry
        doc_info = self.registry.get_document(document_id)

        if not doc_info:
            return False, f"Document {document_id} not found in registry"

        namespace = doc_info["pinecone_namespace"]

        try:
            # Delete from Pinecone using prefix
            deleted_count = 0

            # Fetch and delete chunks in batches
            for ids in self.index.list(prefix=f"{document_id}#", namespace=namespace):
                if ids:
                    self.index.delete(ids=ids, namespace=namespace)
                    deleted_count += len(ids)

            # Remove from registry
            self.registry.delete_document(document_id)

            return True, f"Deleted {deleted_count} chunks for document {document_id}"
        except Exception as e:
            return False, f"Error deleting document: {str(e)}"

    def delete_company_documents(self, company_id: str) -> Tuple[bool, str]:
        """Delete all documents for a company

        Args:
            company_id: Company ID

        Returns:
            Tuple of (success, message)
        """
        try:
            # Get all documents for company
            documents = self.registry.get_company_documents(company_id)

            if not documents:
                return False, f"No documents found for company {company_id}"

            deleted_docs = 0
            failed_docs = []

            for doc in documents:
                document_id = doc["id"]
                success, _ = self.delete_document(document_id)

                if success:
                    deleted_docs += 1
                else:
                    failed_docs.append(document_id)

            if failed_docs:
                return (
                    (deleted_docs > 0),
                    f"Deleted {deleted_docs} documents, but failed to delete {len(failed_docs)} documents for company {company_id}",
                )

            return (
                True,
                f"Successfully deleted {deleted_docs} documents for company {company_id}",
            )
        except Exception as e:
            return False, f"Error deleting company documents: {str(e)}"

    def delete_documents_by_title(
        self, company_id: str, title: str
    ) -> Tuple[bool, str]:
        """Delete all documents with a specific title for a company

        Args:
            company_id: Company ID
            title: Document title

        Returns:
            Tuple of (success, message)
        """
        try:
            # Get all documents with the specific title for the company
            documents = self.registry.get_documents_by_title(company_id, title)

            if not documents:
                return (
                    False,
                    f"No documents found with title '{title}' for company {company_id}",
                )

            deleted_docs = 0
            failed_docs = []

            for doc in documents:
                document_id = doc["id"]
                success, _ = self.delete_document(document_id)

                if success:
                    deleted_docs += 1
                else:
                    failed_docs.append(document_id)

            if failed_docs:
                return (
                    (deleted_docs > 0),
                    f"Deleted {deleted_docs} documents, but failed to delete {len(failed_docs)} documents with title '{title}' for company {company_id}",
                )

            return (
                True,
                f"Successfully deleted {deleted_docs} documents with title '{title}' for company {company_id}",
            )
        except Exception as e:
            return False, f"Error deleting documents by title: {str(e)}"

    def list_company_documents(self, company_id: str) -> List[Dict[str, Any]]:
        """List all documents for a company

        Args:
            company_id: Company ID

        Returns:
            List of document metadata dictionaries
        """
        return self.registry.get_company_documents(company_id)

    def list_documents_by_title(
        self, company_id: str, title: str
    ) -> List[Dict[str, Any]]:
        """List all documents with a specific title for a company

        Args:
            company_id: Company ID
            title: Document title

        Returns:
            List of document metadata dictionaries
        """
        return self.registry.get_documents_by_title(company_id, title)

    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific document

        Args:
            document_id: Document ID

        Returns:
            Document metadata dictionary or None if not found
        """
        return self.registry.get_document(document_id)
