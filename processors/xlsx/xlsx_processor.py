import os
import re
import warnings
from datetime import datetime
from enum import Enum, auto
from typing import List, Dict, Optional

import pandas as pd
import tiktoken
from dataclasses import dataclass

from processors.base_processor import BaseDocumentProcessor


# suppress pandas "could not infer format" warnings when parsing dates
warnings.filterwarnings(
    "ignore",
    message=(
        "Could not infer format, so each element will be parsed " "individually.*"
    ),
)


class ColumnType(Enum):
    IDENTIFIER = auto()
    DATE = auto()
    NUMERIC = auto()
    TEXT = auto()
    UNKNOWN = auto()


@dataclass
class ProcessingError(Exception):
    """Custom exception for processing errors."""

    message: str
    file_path: str
    sheet_name: Optional[str] = None
    details: Optional[str] = None


class XlsxProcessor(BaseDocumentProcessor):
    """
    Processes Excel documents into semantic chunks optimized for RAG context.
    Each chunk contains related rows with clear context for LLM consumption.
    """

    def __init__(
        self,
        max_rows_per_chunk: int = 25,
        max_tokens_per_chunk: int = 500,
        chunk_overlap: int = 2,
        skip_sheets: List[str] = None,
    ):
        if max_rows_per_chunk < 1:
            raise ValueError("max_rows_per_chunk must be at least 1")
        if max_tokens_per_chunk < 100:
            raise ValueError("max_tokens_per_chunk must be at least 100")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")

        super().__init__()
        self.max_rows = max_rows_per_chunk
        self.max_tokens = max_tokens_per_chunk
        self.overlap = chunk_overlap
        self.skip_sheets = skip_sheets or ["summary", "contents", "index"]
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def extract_text(self, file_path: str) -> str:
        """Extract text content from Excel file while preserving structure."""
        try:
            chunks = self.extract_chunks(file_path)
            if not chunks:
                self.logger.warning(f"No chunks extracted from {file_path}")
                return ""

            combined = []
            for chunk in chunks:
                combined.append(chunk["metadata"]["chunk_text"])
                combined.append("\n" + "-" * 50 + "\n")

            return "\n".join(combined)
        except ProcessingError as e:
            self.logger.error(f"Processing error in {file_path}: {e.message}")
            if e.sheet_name:
                self.logger.error(f"Sheet: {e.sheet_name}")
            if e.details:
                self.logger.error(f"Details: {e.details}")
            return ""
        except Exception as e:
            self.logger.error(f"Unexpected error in extract_text: {str(e)}")
            return ""

    def extract_chunks(self, file_path: str) -> List[Dict]:
        """Read an XLSX and return optimized chunks for RAG."""
        if not os.path.exists(file_path):
            raise ProcessingError("File not found", file_path=file_path)

        try:
            sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")
        except pd.errors.EmptyDataError:
            raise ProcessingError("Excel file is empty", file_path=file_path)
        except pd.errors.ParserError as e:
            raise ProcessingError(
                "Error parsing Excel file", file_path=file_path, details=str(e)
            )
        except Exception as e:
            raise ProcessingError(
                "Failed to read Excel", file_path=file_path, details=str(e)
            )

        all_chunks = []
        file_timestamp = datetime.now().isoformat()

        for sheet_name, df in sheets.items():
            if df.empty or sheet_name.lower() in self.skip_sheets:
                self.logger.info(f"Skipping empty or excluded sheet: {sheet_name}")
                continue

            try:
                # Convert all columns to string first to handle mixed types
                df = df.astype(str)
                df = self._clean_column_names(df)
                df = self._clean_data(df)
                column_types = self._analyze_columns(df)
                chunks = self._chunk_sheet(
                    df, sheet_name, file_path, file_timestamp, column_types
                )
                all_chunks.extend(chunks)
            except Exception as e:
                self.logger.error(f"Error processing sheet {sheet_name}: {str(e)}")
                # Continue processing other sheets even if one fails
                continue

        if not all_chunks:
            raise ProcessingError(
                "No valid chunks extracted from any sheet", file_path=file_path
            )

        return all_chunks

    def _chunk_sheet(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        file_path: str,
        file_timestamp: str,
        column_types: Dict[str, ColumnType],
    ) -> List[Dict]:
        """Break a DataFrame into semantic chunks with overlap."""
        chunks = []
        total_rows = len(df)
        cols = df.columns.tolist()
        file_name = os.path.basename(file_path)

        # Calculate chunk size and step for overlap
        rows_per_chunk = self.max_rows
        step = max(1, rows_per_chunk - self.overlap)

        for start in range(0, total_rows, step):
            end = min(start + rows_per_chunk, total_rows)
            block = df.iloc[start:end]

            # Build chunk text
            lines = [
                f"Data from '{file_name}', sheet '{sheet_name}'.",
                f"Rows {start+1}–{end} of {total_rows}.",
                "Columns: " + ", ".join(cols) + ".",
                "Data:",
            ]

            # Format each row
            for i, row_vals in enumerate(block.values.tolist(), start=start + 1):
                # Only include non-empty values
                pairs = [
                    f"{col}: {val}"
                    for col, val in zip(cols, row_vals)
                    if val and str(val).strip()  # Skip empty strings and whitespace
                ]
                if pairs:  # Only add row if it has data
                    lines.append(f"Row {i}: " + "; ".join(pairs))

            text = "\n".join(lines)
            tok_count = len(self.encoding.encode(text))

            if tok_count > self.max_tokens:
                self.logger.warning(
                    f"Chunk rows {start+1}-{end} = {tok_count} tokens "
                    f"(>{self.max_tokens})"
                )

            # Create chunk with minimal metadata
            chunk = {
                "id": f"{file_name}_{sheet_name}_{start+1}_{end}",
                "metadata": {
                    "chunk_text": text,
                    "source": file_name,
                    "sheet": sheet_name,
                    "rows": f"{start+1}-{end}",
                    "columns": cols,
                },
            }
            chunks.append(chunk)

        return chunks

    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, ColumnType]:
        types: Dict[str, ColumnType] = {}
        for col in df.columns:
            low = col.lower()
            if any(k in low for k in ("id", "code", "number")):
                types[col] = ColumnType.IDENTIFIER
            elif self._is_date_column(df[col]):
                types[col] = ColumnType.DATE
            elif self._is_numeric_column(df[col]):
                types[col] = ColumnType.NUMERIC
            else:
                types[col] = ColumnType.TEXT
        return types

    def _is_date_column(self, s: pd.Series) -> bool:
        conv = pd.to_datetime(s, errors="coerce")
        return conv.notna().mean() > 0.5

    def _is_numeric_column(self, s: pd.Series) -> bool:
        num = pd.to_numeric(s, errors="coerce")
        return num.notna().mean() > 0.5

    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names to be more readable."""
        df = df.copy()
        new_cols: List[str] = []

        # First, try to get meaningful column names from the first row
        first_row = df.iloc[0].tolist()

        for i, col in enumerate(df.columns):
            # Try to get a meaningful name from the first row
            if pd.isna(col) or str(col).startswith("Unnamed"):
                if i < len(first_row) and pd.notna(first_row[i]):
                    name = str(first_row[i]).strip()
                else:
                    # If no meaningful name in first row, try to find one in the column
                    sample = df.iloc[:, i].dropna()
                    if not sample.empty:
                        name = str(sample.iloc[0]).strip()
                    else:
                        name = f"Column {i+1}"
            else:
                name = str(col).strip()

            # Clean the name
            name = re.sub(r"[^a-zA-Z0-9\s]", "", name)
            name = re.sub(r"\s+", " ", name)
            name = name.strip()

            # Ensure unique and valid
            if not name:  # If name is empty after cleaning
                name = f"Column {i+1}"

            # Make name unique
            base = name
            counter = 1
            while name in new_cols:
                name = f"{base} {counter}"
                counter += 1
            new_cols.append(name)

        df.columns = new_cols
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and format data values."""
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()

        for col in df.columns:
            # Convert to string and clean
            df.loc[:, col] = df[col].astype(str).str.strip()

            # Handle numeric values
            try:
                num = pd.to_numeric(df[col], errors="coerce")
                if num.notna().any():
                    df.loc[:, col] = num.apply(
                        lambda x: f"{x:,.2f}" if pd.notna(x) else ""
                    )
            except (ValueError, TypeError):
                pass

            # Remove empty values and nan
            df.loc[:, col] = df[col].replace(
                ["nan", "None", "none", "null", "NULL", "", "NaN", "NAN"], ""
            )

        return df
