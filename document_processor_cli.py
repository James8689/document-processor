#!/usr/bin/env python3
"""
Document Processor - Command Line Interface

This script provides a command-line interface for processing documents and uploading to Pinecone.
It determines the document type based on file extension and routes to the appropriate processor.

Note: This script is separate from document_processor_api.py, which provides a REST API interface
for the same functionality. Use this script for command-line batch processing from the filesystem.

Usage:
    python document_processor_cli.py /path/to/file1.pdf /path/to/file2.docx
"""
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Union

from loguru import logger

# Set up logging
logging.basicConfig(level=logging.INFO)
logger.add("document_processor.log", rotation="10 MB")


def process_document(file_path: str) -> None:
    """
    Process a document and upload to Pinecone.

    This function determines the document type and routes it to the appropriate processor.

    Args:
        file_path: Path to the document
    """
    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        logger.error(f"File not found: {file_path}")
        return

    # Get file extension (lowercase)
    file_ext = file_path_obj.suffix.lower()

    try:
        # Process based on file type
        if file_ext in [".pdf"]:
            logger.info(f"Processing PDF: {file_path}")
            from pdf_loader.pdf_processor import process_pdf

            process_pdf(file_path)

        elif file_ext in [".docx", ".doc"]:
            logger.info(f"Processing Word document: {file_path}")
            from docx_loader.docx_processor import process_docx

            process_docx(file_path)

        elif file_ext in [".xlsx", ".xls", ".csv"]:
            logger.info(f"Processing Excel/CSV: {file_path}")
            from xlsx_loader.xlsx_processor import process_xlsx

            process_xlsx(file_path)

        else:
            logger.error(f"Unsupported file type: {file_ext}")
            print(f"Unsupported file type: {file_ext}")
            print("Currently supported file types: PDF, DOCX, DOC, XLSX, XLS, CSV")

    except ImportError as e:
        logger.error(f"Failed to import processor module: {str(e)}")
        print("Error: Missing required module. Please install requirements.txt")

    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        print(f"Error processing document: {str(e)}")


def process_documents(file_paths: List[str]) -> None:
    """
    Process multiple documents and upload to Pinecone.

    Args:
        file_paths: List of paths to the documents
    """
    for file_path in file_paths:
        process_document(file_path)


def main():
    parser = argparse.ArgumentParser(
        description="Process documents and upload to Pinecone"
    )
    parser.add_argument(
        "file_paths", nargs="+", help="Paths to the documents to process"
    )

    args = parser.parse_args()

    # Process all provided documents
    process_documents(args.file_paths)


if __name__ == "__main__":
    main()
