import sqlite3
import os
from datetime import datetime
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    DateTime,
    Index,
    Table,
    MetaData,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class Documents(Base):
    """SQLAlchemy model for documents table"""

    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    company_id = Column(String, nullable=False)
    title = Column(String, nullable=False)
    document_name = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    total_chunks = Column(Integer, nullable=False)
    upload_date = Column(DateTime, nullable=False)
    embedding_model = Column(String, nullable=False)
    pinecone_namespace = Column(String, nullable=False)

    # Create indexes for faster lookups
    __table_args__ = (
        Index("company_idx", "company_id"),
        Index("title_idx", "title"),
    )


class DocumentRegistry:
    """Registry for tracking document metadata for retrieval and deletion"""

    def __init__(self, db_connection_string=None):
        """Initialize document registry with database connection

        Args:
            db_connection_string: Database connection string. If None, uses SQLite.
                For production, use PostgreSQL or another robust database.
        """
        if db_connection_string is None:
            # Default to SQLite for development
            db_path = os.environ.get("DOCUMENT_REGISTRY_PATH", "document_registry.db")
            self.engine = create_engine(f"sqlite:///{db_path}")
        else:
            # Use provided connection string for production
            self.engine = create_engine(db_connection_string)

        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def register_document(
        self,
        company_id,
        title,
        document_name,
        filename,
        file_type,
        total_chunks,
        embedding_model,
        pinecone_namespace,
    ):
        """Register a new document in the registry

        Args:
            company_id: Identifier for the company that owns the document
            title: Document title
            document_name: Unique document identifier/name
            filename: Original filename
            file_type: Type of document (pdf, docx, xlsx, etc.)
            total_chunks: Number of chunks the document was split into
            embedding_model: Name of embedding model used
            pinecone_namespace: Namespace in Pinecone where vectors are stored

        Returns:
            document_id: Prefixed ID used in Pinecone
        """
        # Create document ID with multi-level prefix for hierarchical organization
        document_id = f"{company_id}#{title}#{document_name}"

        session = self.Session()
        try:
            # Create new document record
            doc = Documents(
                id=document_id,
                company_id=company_id,
                title=title,
                document_name=document_name,
                original_filename=filename,
                file_type=file_type,
                total_chunks=total_chunks,
                upload_date=datetime.now(),
                embedding_model=embedding_model,
                pinecone_namespace=pinecone_namespace,
            )

            session.add(doc)
            session.commit()
            return document_id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def update_chunk_count(self, document_id, total_chunks):
        """Update the total chunk count for a document

        Args:
            document_id: ID of document to update
            total_chunks: New total chunk count
        """
        session = self.Session()
        try:
            doc = session.query(Documents).filter_by(id=document_id).first()
            if doc:
                doc.total_chunks = total_chunks
                session.commit()
            else:
                raise ValueError(f"Document with ID {document_id} not found")
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_document(self, document_id):
        """Get document metadata by ID

        Args:
            document_id: ID of document to retrieve

        Returns:
            Document metadata as dictionary
        """
        session = self.Session()
        try:
            doc = session.query(Documents).filter_by(id=document_id).first()
            if doc:
                return {
                    "id": doc.id,
                    "company_id": doc.company_id,
                    "title": doc.title,
                    "document_name": doc.document_name,
                    "original_filename": doc.original_filename,
                    "file_type": doc.file_type,
                    "total_chunks": doc.total_chunks,
                    "upload_date": doc.upload_date,
                    "embedding_model": doc.embedding_model,
                    "pinecone_namespace": doc.pinecone_namespace,
                }
            return None
        finally:
            session.close()

    def get_company_documents(self, company_id):
        """Get all documents for a specific company

        Args:
            company_id: Company identifier

        Returns:
            List of document metadata dictionaries
        """
        session = self.Session()
        try:
            docs = session.query(Documents).filter_by(company_id=company_id).all()
            return [
                {
                    "id": doc.id,
                    "company_id": doc.company_id,
                    "title": doc.title,
                    "document_name": doc.document_name,
                    "original_filename": doc.original_filename,
                    "file_type": doc.file_type,
                    "total_chunks": doc.total_chunks,
                    "upload_date": doc.upload_date,
                    "embedding_model": doc.embedding_model,
                    "pinecone_namespace": doc.pinecone_namespace,
                }
                for doc in docs
            ]
        finally:
            session.close()

    def get_documents_by_title(self, company_id, title):
        """Get all documents with a specific title for a company

        Args:
            company_id: Company identifier
            title: Document title

        Returns:
            List of document metadata dictionaries
        """
        session = self.Session()
        try:
            docs = (
                session.query(Documents)
                .filter_by(company_id=company_id, title=title)
                .all()
            )
            return [
                {
                    "id": doc.id,
                    "company_id": doc.company_id,
                    "title": doc.title,
                    "document_name": doc.document_name,
                    "original_filename": doc.original_filename,
                    "file_type": doc.file_type,
                    "total_chunks": doc.total_chunks,
                    "upload_date": doc.upload_date,
                    "embedding_model": doc.embedding_model,
                    "pinecone_namespace": doc.pinecone_namespace,
                }
                for doc in docs
            ]
        finally:
            session.close()

    def delete_document(self, document_id):
        """Delete document metadata from registry

        Args:
            document_id: ID of document to delete

        Returns:
            bool: True if document was deleted, False otherwise
        """
        session = self.Session()
        try:
            doc = session.query(Documents).filter_by(id=document_id).first()
            if doc:
                session.delete(doc)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def list_all_documents(self):
        """List all documents in the registry

        Returns:
            List of document metadata dictionaries
        """
        session = self.Session()
        try:
            docs = session.query(Documents).all()
            return [
                {
                    "id": doc.id,
                    "company_id": doc.company_id,
                    "title": doc.title,
                    "document_name": doc.document_name,
                    "original_filename": doc.original_filename,
                    "file_type": doc.file_type,
                    "total_chunks": doc.total_chunks,
                    "upload_date": doc.upload_date,
                    "embedding_model": doc.embedding_model,
                    "pinecone_namespace": doc.pinecone_namespace,
                }
                for doc in docs
            ]
        finally:
            session.close()
