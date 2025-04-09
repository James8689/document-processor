"""
Document Processor - API Interface

This script provides a REST API interface for processing documents and uploading to Pinecone.
It accepts document uploads via HTTP requests and routes them to the appropriate processor
based on file type.

Note: This script is separate from document_processor_cli.py, which provides a command-line interface
for processing documents directly from the filesystem. Use this API for web/programmatic access.

Endpoints:
    POST /process-document - Process a single document
    POST /batch-process - Process multiple documents at once

Usage:
    uvicorn document_processor_api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import os
import logging
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Optional
from pydantic import BaseModel
from pinecone import Pinecone
from utils import detect_file_type

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Pinecone using V3 SDK
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Get index settings
index_name = os.getenv("PINECONE_INDEX_NAME", "document-embeddings")
namespace = os.getenv("PINECONE_NAMESPACE", "documents")
customer_id = os.getenv("CUSTOMER_ID", "default")
index_host = os.getenv("PINECONE_HOST")

# Connect to index
if index_host:
    index = pc.Index(name=index_name, host=index_host)
else:
    # Get the index info to retrieve the host
    try:
        index_info = pc.describe_index(index_name)
        index_host = index_info.host
        logger.info(f"Retrieved index host: {index_host}")
        index = pc.Index(name=index_name, host=index_host)
    except Exception as e:
        logger.error(f"Error connecting to Pinecone index '{index_name}': {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to connect to Pinecone index. Please ensure the index exists.",
        )

app = FastAPI(
    title="Document Processor API",
    description="API for processing documents and uploading to Pinecone",
    version="1.0.0",
)


class ProcessResponse(BaseModel):
    status: str
    message: str
    file_type: str
    chunks: Optional[int] = None


@app.post("/process-document", response_model=ProcessResponse)
async def process_document(file: UploadFile = File(...), custom_id: str = Form(None)):
    """Process a single document and store in Pinecone."""
    temp_file = None
    try:
        # Save uploaded file to temp location
        _, temp_file_path = tempfile.mkstemp()
        temp_file = Path(temp_file_path)

        # Write content to temp file
        with open(temp_file, "wb") as f:
            contents = await file.read()
            f.write(contents)

        # Use the provided custom ID or the original filename
        document_id = custom_id if custom_id else file.filename
        logger.info(f"Processing document: {document_id}")

        # Detect file type
        file_type = detect_file_type(file.filename)

        if file_type == "pdf":
            # Process PDF
            from pdf_loader.pdf_processor import PdfProcessor

            processor = PdfProcessor()
            processor.process_file(str(temp_file))

        elif file_type == "docx":
            # Process DOCX
            from docx_loader.docx_processor import DocxProcessor

            processor = DocxProcessor()
            processor.process_file(str(temp_file))

        elif file_type == "excel":
            # Process Excel
            from xlsx_loader.xlsx_processor import XlsxProcessor

            processor = XlsxProcessor()
            processor.process_file(str(temp_file))

        else:
            # Unsupported file type
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type: {file_type}"
            )

        return {
            "status": "success",
            "message": f"Successfully processed {file.filename}",
            "file_type": file_type,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if temp_file and temp_file.exists():
            temp_file.unlink()


@app.post("/batch-process")
async def batch_process_documents(files: List[UploadFile] = File(...)):
    """Process multiple documents in batch."""
    results = []

    for file in files:
        temp_file = None
        try:
            # Save uploaded file to temp location
            _, temp_file_path = tempfile.mkstemp()
            temp_file = Path(temp_file_path)

            # Write content to temp file
            with open(temp_file, "wb") as f:
                contents = await file.read()
                f.write(contents)

            # Detect file type
            file_type = detect_file_type(file.filename)

            if file_type == "pdf":
                # Process PDF
                from pdf_loader.pdf_processor import PdfProcessor

                processor = PdfProcessor()
                processor.process_file(str(temp_file))

            elif file_type == "docx":
                # Process DOCX
                from docx_loader.docx_processor import DocxProcessor

                processor = DocxProcessor()
                processor.process_file(str(temp_file))

            elif file_type == "excel":
                # Process Excel
                from xlsx_loader.xlsx_processor import XlsxProcessor

                processor = XlsxProcessor()
                processor.process_file(str(temp_file))

            else:
                # Unsupported file type
                results.append(
                    {
                        "filename": file.filename,
                        "status": "error",
                        "message": f"Unsupported file type: {file_type}",
                        "file_type": file_type,
                    }
                )
                continue

            results.append(
                {
                    "filename": file.filename,
                    "status": "success",
                    "message": f"Successfully processed {file.filename}",
                    "file_type": file_type,
                }
            )

        except Exception as e:
            logger.error(f"Error processing document {file.filename}: {str(e)}")
            results.append(
                {
                    "filename": file.filename,
                    "status": "error",
                    "message": str(e),
                    "file_type": file_type if "file_type" in locals() else "unknown",
                }
            )
        finally:
            # Clean up temp file
            if temp_file and temp_file.exists():
                temp_file.unlink()

    return {"results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
