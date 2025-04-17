import streamlit as st
import tempfile
import os
from pathlib import Path
import argparse
import sys
import importlib

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the current directory to sys.path
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import document_processor module
document_processor = importlib.import_module("document_processor")
DocumentProcessorService = getattr(document_processor, "DocumentProcessorService")


def main():
    """Run the Streamlit application."""
    # Setup arg parser for dry run mode
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry run mode (save chunks as markdown files)",
    )
    args = parser.parse_args()

    # Set page config
    st.set_page_config(page_title="Document Processor", page_icon="📄", layout="wide")

    # App title and description
    st.title("Document Processor")
    st.markdown(
        """
        Upload documents (PDF, Word, Excel) to extract text and create vector embeddings in Pinecone.
        """
    )

    # Display dry run info if active
    dry_run = args.dry_run
    if dry_run:
        st.info("Running in dry run mode. Chunks will be saved as markdown files.")

    # Add dry run checkbox (only if not already set by command line)
    if not args.dry_run:
        dry_run = st.checkbox(
            "Dry Run (save chunks as markdown files instead of uploading to Pinecone)"
        )

    # Initialize document processor
    processor_service = DocumentProcessorService(dry_run=dry_run)

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "xlsx", "xls", "csv"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        # Process each file
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # Save the file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=f"_{uploaded_file.name}"
                ) as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    temp_path = temp_file.name

                # Process the file
                success = processor_service.process_document(temp_path)

                # Show result
                if success:
                    st.success(f"Successfully processed {uploaded_file.name}")
                else:
                    st.error(f"Failed to process {uploaded_file.name}")

                # Clean up the temp file
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    st.warning(f"Could not delete temporary file: {e}")


if __name__ == "__main__":
    main()
