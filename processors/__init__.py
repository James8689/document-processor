# processors package
from pathlib import Path
from typing import Optional

from processors.base_processor import BaseDocumentProcessor
from processors.pdf.pdf_processor import PdfProcessor
from processors.docx.docx_processor import DocxProcessor
from processors.xlsx.xlsx_processor import XlsxProcessor
from processors.document_registry import DocumentRegistry
from processors.document_manager import DocumentManager

# Dictionary mapping file extensions to processor classes
PROCESSOR_MAPPING = {
    "pdf": PdfProcessor,
    "docx": DocxProcessor,
    "doc": DocxProcessor,  # Use DocxProcessor for .doc files too
    "xlsx": XlsxProcessor,
    "xls": XlsxProcessor,  # Use XlsxProcessor for .xls files too
}


def get_processor_for_file(file_path, document_registry=None):
    """
    Get the appropriate processor for a file based on its extension.

    Args:
        file_path: Path to the file
        document_registry: Optional DocumentRegistry instance

    Returns:
        BaseDocumentProcessor: Processor instance for the file type
    """
    # Extract extension from file path (lowercase, without the dot)
    extension = file_path.split(".")[-1].lower()

    # Get processor class from mapping
    processor_class = PROCESSOR_MAPPING.get(extension)

    if processor_class is None:
        raise ValueError(f"No processor available for file type: {extension}")

    # Initialize processor with registry if provided
    if document_registry is not None:
        return processor_class(document_registry=document_registry)

    # Otherwise initialize without registry
    return processor_class()


__all__ = [
    "BaseDocumentProcessor",
    "PdfProcessor",
    "DocxProcessor",
    "XlsxProcessor",
    "DocumentRegistry",
    "DocumentManager",
    "get_processor_for_file",
    "PROCESSOR_MAPPING",
]
