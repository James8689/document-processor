# Document Processor

A comprehensive document processing system for extracting, chunking, and embedding text from various document formats into Pinecone vector databases.

## Features

- **Multi-format Support**: Process PDF, DOCX, and XLSX files
- **Intelligent Chunking**: Split documents into optimal chunks for vector embeddings
- **Pinecone Integration**: Upload document chunks to Pinecone with full metadata
- **Document Management**: Track, organize, and delete documents by company, title, or ID
- **Dry Run Mode**: Preview chunks before uploading to Pinecone
- **Multiple Interfaces**: Command line, API, and Streamlit interfaces

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in your configuration

## Environment Variables

```
# Pinecone settings
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index_name

# Embedding model
EMBEDDING_MODEL=text-embedding-ada-002

# Chunking parameters
CHUNK_SIZE=500
CHUNK_OVERLAP=75

# Dry run output
DRY_RUN_OUTPUT_DIR=dry_run_output

# Document registry
DOCUMENT_REGISTRY_PATH=document_registry.db
```

## Command Line Usage

### Processing Documents

Process a single document:

```bash
python document_processor.py process path/to/document.pdf --company-id acme --title "Annual Report 2025"
```

Process a directory of documents:

```bash
python document_processor.py process path/to/directory --customer-id customer123 --title-prefix "Financial Reports"
```

Run in dry-run mode (no upload to Pinecone):

```bash
python document_processor.py process path/to/document.pdf --dry-run --company-id acme --title "Annual Report"
```

### Managing Documents

List all documents for a company:

```bash
python document_processor.py list company acme_corp
```

Delete a specific document:

```bash
python document_processor.py delete document acme_corp#Annual_Report#document_20250416.pdf_20250416_120135
```

Delete all documents for a company:

```bash
python document_processor.py delete company acme_corp
```

Delete all documents with a specific title for a company:

```bash
python document_processor.py delete title acme_corp "Annual Report"
```

## Document ID Structure

Documents are stored with a hierarchical ID structure:

```
company_id#title#document_name#chunk_n
```

Example: `acme_corp#Annual_Report#report_2025_20250416_120135#chunk_1`

This structure allows efficient organization and deletion at various levels:
- Delete all documents for a company
- Delete all documents with a specific title
- Delete a specific document
- Delete a specific chunk

## API Usage

The system also provides a RESTful API for document processing and management:

```python
from api_app import app

# Run the API server
app.run(debug=True)
```

Check the API documentation at `http://localhost:5000/docs` when running.

## Web Interface

A Streamlit-based web interface is available for easy document processing:

```bash
streamlit run streamlit_app.py
```

This provides a user-friendly interface for uploading, processing, and managing documents.

## Architecture

The system uses a modular architecture:

- `processors/`: Base and specific document processors (PDF, DOCX, XLSX)
- `document_registry.py`: Document metadata tracking using SQLite
- `document_manager.py`: Interface for document management operations
- `document_processor.py`: Command-line interface
- `api_app.py`: FastAPI-based REST API
- `streamlit_app.py`: Streamlit web interface

## Dry Run Output

When running in dry-run mode, the system saves document chunks as markdown files in the `dry_run_output` directory, with the following structure:

```markdown
# Document Chunk Preview

## Metadata
- **filename**: document.pdf
- **file_type**: pdf
- **chunk_index**: 1
- **total_chunks**: 5
- **chunk_size**: 1245
- **token_count**: 213
- **timestamp**: 2025-04-16T12:01:35.123456
- **company_id**: acme_corp
- **title**: Annual Report
- **document_id_structure**: acme_corp#Annual_Report#document_20250416_120135#chunk_0

## Content

```
Actual text content of the chunk goes here
```