import docx
from typing import List, Tuple

from processors.base_processor import BaseDocumentProcessor


class DocxProcessor(BaseDocumentProcessor):
    """
    Processes DOCX documents and uploads to Pinecone.
    Extends the BaseDocumentProcessor for DOCX-specific functionality.
    """

    def extract_text(self, file_path: str) -> str:
        """
        Extract text content from DOCX.

        Args:
            file_path: Path to the DOCX file

        Returns:
            str: Extracted text content
        """
        try:
            doc = docx.Document(file_path)
            text_content = []

            # Process paragraphs
            for para in doc.paragraphs:
                if para.text.strip():
                    text_content.append(para.text)

            # Process tables
            for table in doc.tables:
                for row in table.rows:
                    cells = [
                        cell.text.strip() for cell in row.cells if cell.text.strip()
                    ]
                    if cells:
                        text_content.append(" | ".join(cells))

            return "\n\n".join(text_content)

        except Exception as e:
            self.logger.error(f"Error extracting text from DOCX: {e}")
            return ""
