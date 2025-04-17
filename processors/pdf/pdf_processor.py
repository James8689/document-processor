from pathlib import Path
import fitz  # PyMuPDF
import PyPDF2
import os

from processors.base_processor import BaseDocumentProcessor


class PdfProcessor(BaseDocumentProcessor):
    """
    Processes PDF documents and uploads to Pinecone.
    Extends the BaseDocumentProcessor for PDF-specific functionality.
    """

    def extract_text(self, file_path: str) -> str:
        """
        Extract text content from PDF.

        Args:
            file_path: Path to the PDF file

        Returns:
            str: Extracted text content
        """
        try:
            # First try PyMuPDF (better text extraction)
            doc = fitz.open(file_path)
            text_content = ""

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_content += page.get_text()

            doc.close()

            # If no text was extracted, fall back to PyPDF2
            if not text_content.strip():
                self.logger.info(
                    f"No text extracted with PyMuPDF, falling back to PyPDF2 for {file_path}"
                )
                return self._extract_with_pypdf2(file_path)

            return text_content

        except Exception as e:
            self.logger.error(f"Error extracting text with PyMuPDF: {e}")
            # Fall back to PyPDF2
            return self._extract_with_pypdf2(file_path)

    def _extract_with_pypdf2(self, file_path: str) -> str:
        """
        Extract text using PyPDF2 as a fallback.

        Args:
            file_path: Path to the PDF file

        Returns:
            str: Extracted text content
        """
        try:
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text_content = ""

                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text_content += page.extract_text() + "\n\n"

                return text_content

        except Exception as e:
            self.logger.error(f"Error extracting text with PyPDF2: {e}")
            return ""
