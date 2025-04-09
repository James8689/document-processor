import streamlit as st
from pinecone import Pinecone
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import time
import argparse
from typing import Tuple
from pdf_loader.pdf_processor import PdfProcessor
from docx_loader.docx_processor import DocxProcessor
from xlsx_loader.xlsx_processor import XlsxProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Run in dry run mode (save chunks as markdown files)",
)
args = parser.parse_args()

# Load environment variables
load_dotenv()

# Initialize Pinecone only if not in dry run mode
if not args.dry_run:
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
            st.error(
                "Failed to connect to Pinecone index. Please ensure the index exists."
            )
            raise


# Define our own detect_file_type function
def detect_file_type(file_name):
    """
    Detect file type based on extension.

    Args:
        file_name (str): Name of the file

    Returns:
        str: File type ('pdf', 'docx', 'xlsx', or 'unsupported')
    """
    extension = os.path.splitext(file_name.lower())[1]

    if extension == ".pdf":
        return "pdf"
    elif extension == ".docx":
        return "docx"
    elif extension in [".xlsx", ".xls"]:
        return "excel"
    else:
        return "unsupported"


def process_document(file, dry_run: bool = False) -> Tuple[bool, str]:
    """
    Process a single document and store in Pinecone.

    Args:
        file: Streamlit UploadedFile object
        dry_run: If True, save chunks as markdown files instead of uploading to Pinecone

    Returns:
        Tuple of (success, message)
    """
    try:
        # Save uploaded file temporarily
        file_path = Path(f"temp_{file.name}")
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        # Get file extension
        file_ext = file_path.suffix.lower()

        # Process based on file type
        if file_ext == ".pdf":
            processor = PdfProcessor()
        elif file_ext == ".docx":
            processor = DocxProcessor()
        elif file_ext == ".xlsx":
            processor = XlsxProcessor()
        else:
            return False, f"Unsupported file type: {file_ext}"

        # Process the file
        if dry_run:
            success = processor.dry_run(str(file_path))
            message = f"Dry run completed for {file.name}. Chunks saved to dry_run_output directory."
        else:
            # Only pass Pinecone index when not in dry run mode
            success = processor.process_file(
                str(file_path),
                index=index,
                namespace=namespace,
                customer_id=customer_id,
            )
            message = (
                f"Successfully processed {file.name}"
                if success
                else f"Failed to process {file.name}"
            )

        # Close any open file handles
        if hasattr(processor, "close"):
            processor.close()

        # Delete the temporary file
        try:
            file_path.unlink()
        except PermissionError:
            # If we can't delete it immediately, try again after a longer delay
            time.sleep(2)  # Increased from 1 to 2 seconds
            try:
                file_path.unlink()
            except PermissionError as e:
                logger.warning(f"Could not delete temporary file {file_path}: {e}")
                # Try one more time after another delay
                time.sleep(3)  # Wait 3 more seconds
                try:
                    file_path.unlink()
                except PermissionError as e:
                    logger.warning(f"Final attempt to delete {file_path} failed: {e}")

        return success, message

    except Exception as e:
        logger.error(f"Error processing {file.name}: {str(e)}")
        return False, f"Error processing {file.name}: {str(e)}"


# Streamlit UI
st.title("Document Upload to Pinecone")

# Add dry run checkbox, but disable it if --dry-run was passed
dry_run = args.dry_run or st.checkbox(
    "Dry Run (save chunks as markdown files)", value=args.dry_run, disabled=args.dry_run
)

if args.dry_run:
    st.info("Running in dry run mode. Chunks will be saved as markdown files.")

uploaded_files = st.file_uploader(
    "Upload documents (PDF, DOCX, Excel)",
    type=["pdf", "docx", "xlsx", "xls"],
    accept_multiple_files=True,
)

if uploaded_files:
    for file in uploaded_files:
        with st.spinner(f"Processing {file.name}..."):
            success, message = process_document(file, dry_run=dry_run)
            if success:
                st.success(message)
            else:
                st.error(message)
