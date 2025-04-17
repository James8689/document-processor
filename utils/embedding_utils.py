from pinecone import Pinecone
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class EmbeddingManager:
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "document-embeddings")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "llama-text-embed-v2")

        if not self.api_key:
            raise ValueError("Missing required Pinecone API key")

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)

        # Create or connect to index
        try:
            # Check if index exists
            indexes = self.pc.list_indexes()
            if not indexes.names() or self.index_name not in indexes.names():
                # Create new index with server-side embedding
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1024,  # Dimension for llama-text-embed-v2
                    metric="cosine",
                    spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
                )
                logger.info(f"Created new index: {self.index_name}")

            # Connect to index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to index: {self.index_name}")

        except Exception as e:
            logger.error(f"Error setting up Pinecone index: {e}")
            raise

    def prepare_vectors(
        self, chunks: List[str], source_type: str, filename: str
    ) -> List[Dict[str, Any]]:
        """Prepare vectors for upsert with server-side embedding."""
        records = []
        timestamp = datetime.now().isoformat()

        for i, chunk in enumerate(chunks):
            # Create record for server-side embedding with flat metadata
            record = {
                "id": f"{source_type}_{filename}_{i}_{timestamp}",
                "chunk_text": chunk,
                "metadata.document_id": filename,
                "metadata.chunk_id": f"document_{filename}_{i}",
                "metadata.timestamp": timestamp,
            }
            records.append(record)

        return records

    def upsert_vectors(self, vectors: List[Dict[str, Any]], namespace: str = "") -> int:
        """Upsert vectors to Pinecone with server-side embedding."""
        try:
            total_records = len(vectors)
            logger.info(f"Upserting {total_records} records")

            # Process in batches of 50
            batch_size = 50
            records_upserted = 0

            for i in range(0, total_records, batch_size):
                batch = vectors[i : i + batch_size]

                # Upsert records directly - no transformation needed
                self.index.upsert_records(records=batch, namespace=namespace)
                records_upserted += len(batch)
                logger.info(
                    f"Upserted batch {i//batch_size + 1}/{(total_records-1)//batch_size + 1}"
                )

            return records_upserted

        except Exception as e:
            logger.error(f"Error during upsert: {e}")
            raise

    def search(self, query: str, namespace: str = "", top_k: int = 3) -> Dict:
        """Search for similar vectors using the Llama model."""
        try:
            query_payload = {"inputs": {"chunk_text": query}, "top_k": top_k}

            results = self.index.search(namespace=namespace, query=query_payload)
            return results

        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise
