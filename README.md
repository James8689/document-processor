# Document Processor

A document processing system that extracts content from various document types (PDF, Word, Excel) and stores the vector embeddings in Pinecone for retrieval.

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Configure your environment:
   - Copy `.env.example` to `.env`
   - Fill in your Pinecone credentials and other configuration values

## Configuration

The system uses environment variables for configuration. These can be set in a `.env` file:

- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_INDEX_NAME`: Name of your Pinecone index
- `PINECONE_INDEX_HOST`: Direct host URL for your Pinecone index
- `CUSTOMER_ID`: Customer identifier used in namespacing
- `PINECONE_NAMESPACE`: Namespace for storing document vectors (defaults to `documents_${CUSTOMER_ID}`)
- `EMBEDDING_MODEL`: Model to use for generating embeddings
- `CHUNK_SIZE`: Character size for text chunking
- `CHUNK_OVERLAP`: Character overlap between chunks

## Usage Options

### 1. Streamlit Web Interface (Recommended)

The Streamlit web interface is the recommended way to use this system. It provides a user-friendly way to upload documents and has both regular and dry run modes.

```bash
# Start the Streamlit web interface in normal mode
streamlit run upload_documents.py

# Start in dry run mode (saves chunks as markdown files without uploading to Pinecone)
streamlit run upload_documents.py -- --dry-run
```

Features:
- Upload multiple documents at once
- Support for PDF, DOCX, and Excel files
- Option to use dry run mode to preview chunks before uploading
- Detailed progress information

### 2. Command Line Interface

For batch processing of documents directly from the command line:

```bash
# Process files using the CLI script
python document_processor_cli.py /path/to/your/document.pdf

# Process specific document types directly
python pdf_loader/pdf_processor.py /path/to/your/document.pdf
python docx_loader/docx_processor.py /path/to/your/document.docx
python xlsx_loader/xlsx_processor.py /path/to/your/document.xlsx
```

### 3. API Service

Process documents programmatically via REST API:

```bash
# Start the FastAPI server
uvicorn document_processor_api:app --host 0.0.0.0 --port 8000
```

API endpoints:
- `POST /process-document`: Upload and process a single document
- `POST /batch-process`: Upload and process multiple documents

## Document Processors

The system includes processors for different document types:

- **PDF Processor**: Extracts text from PDF files with enhanced structure preservation
- **Word Processor**: Extracts text from DOCX files, including tables and formatting
- **Excel Processor**: Extracts text from XLSX files while preserving table structure

Each processor:
1. Extracts text from the specific file type
2. Chunks the text into manageable segments
3. Generates vector embeddings using Pinecone's server-side embedding
4. Uploads the vectors to Pinecone with appropriate metadata

## Dry Run Mode

The system supports a "dry run" mode which processes documents but saves the chunks as markdown files instead of uploading to Pinecone. This is useful for:

- Previewing how documents will be chunked
- Verifying content extraction quality
- Testing without consuming Pinecone resources

When using dry run mode, chunk files are saved to the `dry_run_output` directory with:
- Content section showing the exact text that would be uploaded
- Comprehensive metadata including character counts and token counts
- File naming that includes chunk position (e.g., chunk_1_of_4)

## Metadata

Each document chunk is stored with metadata including:
- Filename and file type
- Original file path
- Source type (e.g., local_pdf, local_xlsx)
- Chunk information (index, total chunks, chunk size, overlap)
- Character and token counts
- Processing timestamp
- Additional file-specific metadata (e.g., sheet names for Excel) 