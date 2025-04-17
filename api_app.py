"""
Document Processor API

A FastAPI application that provides REST endpoints for processing documents
and uploading their content to Pinecone as vector embeddings.

Usage:
    uvicorn api_app:app --host 0.0.0.0 --port 8000
"""

import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel

from document_processor import DocumentProcessorService

# Initialize FastAPI app
app = FastAPI(
    title="Document Processor API",
    description="API for processing documents and uploading to Pinecone",
    version="1.0.0",
)

# Initialize document processor service
processor_service = DocumentProcessorService()


class ProcessResponse(BaseModel):
    """Response model for document processing."""

    filename: str
    status: str
    message: str
    file_type: str


@app.post("/process-document", response_model=ProcessResponse)
async def process_document(file: UploadFile = File(...), custom_id: str = Form(None)):
    """
    Process a single document and store in Pinecone.

    Args:
        file: The document file to process
        custom_id: Optional custom identifier for the document

    Returns:
        ProcessResponse: Processing result
    """
    temp_file = None
    try:
        # Save file to temp location
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            temp_file = tmp.name

        # Log the custom ID if provided
        if custom_id:
            print(f"Processing document with custom ID: {custom_id}")

        # Get file type
        file_type = Path(file.filename).suffix.lower()[1:]  # Remove the dot

        # Process the document
        success = processor_service.process_document(temp_file)

        if success:
            return {
                "filename": file.filename,
                "status": "success",
                "message": f"Successfully processed {file.filename}",
                "file_type": file_type,
            }
        else:
            raise HTTPException(
                status_code=500, detail=f"Failed to process {file.filename}"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


@app.post("/batch-process")
async def batch_process(files: List[UploadFile] = File(...)):
    """
    Process multiple documents in batch.

    Args:
        files: List of document files to process

    Returns:
        Dict: Dictionary containing results for each file
    """
    results = []

    for file in files:
        temp_file = None
        try:
            # Save file to temp location
            suffix = Path(file.filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                temp_file = tmp.name

            # Get file type
            file_type = Path(file.filename).suffix.lower()[1:]  # Remove the dot

            # Process the document
            success = processor_service.process_document(temp_file)

            if success:
                results.append(
                    {
                        "filename": file.filename,
                        "status": "success",
                        "message": f"Successfully processed {file.filename}",
                        "file_type": file_type,
                    }
                )
            else:
                results.append(
                    {
                        "filename": file.filename,
                        "status": "error",
                        "message": f"Failed to process {file.filename}",
                        "file_type": file_type,
                    }
                )

        except Exception as e:
            results.append(
                {
                    "filename": file.filename,
                    "status": "error",
                    "message": str(e),
                    "file_type": (
                        Path(file.filename).suffix.lower()[1:]
                        if hasattr(file, "filename")
                        else "unknown"
                    ),
                }
            )
        finally:
            # Clean up temp file
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)

    return {"results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
