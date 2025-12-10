import os
import argparse
import uuid # Added for UUID generation
from qdrant_client import QdrantClient, models
from openai import OpenAI
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Configuration from environment variables
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "humanoid_ai_book"
CHUNK_SIZE = 1000  # characters

# Initialize clients
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def get_markdown_files(docs_path="../docs/"):
    """Recursively find all markdown files in the given path."""
    markdown_files = []
    for root, _, files in os.walk(docs_path):
        for file in files:
            if file.endswith(".md"):
                markdown_files.append(os.path.join(root, file))
    logging.info(f"Found {len(markdown_files)} markdown files in {docs_path}")
    return markdown_files

def chunk_text(text: str, chunk_size: int) -> list[str]:
    """Splits text into chunks of specified size."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generates OpenAI embeddings for a list of texts."""
    response = openai_client.embeddings.create(input=texts, model=OPENAI_EMBEDDING_MODEL)
    return [d.embedding for d in response.data]

def ingest_documents(docs_path="../docs/"):
    """
    Ingests markdown documents into Qdrant.
    Reads markdown files, chunks text, generates embeddings, and uploads to Qdrant.
    """
    logging.info(f"Starting ingestion process from {docs_path}")

    # Recreate collection
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )
    logging.info(f"Collection '{COLLECTION_NAME}' recreated in Qdrant.")

    markdown_files = get_markdown_files(docs_path)
    if not markdown_files:
        logging.warning("No markdown files found to ingest.")
        return

    points = []
    for md_file in markdown_files:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple metadata extraction (can be enhanced for Docusaurus frontmatter)
        filename = os.path.basename(md_file)
        
        chunks = chunk_text(content, CHUNK_SIZE)
        embeddings = generate_embeddings(chunks)

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{filename}_{i}")), # Generate stable UUID
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "filename": filename,
                        "chunk_index": i,
                        "source": md_file # Store full path for retrieval
                    },
                )
            )
    
    if points:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=points
        )
        logging.info(f"Successfully ingested {len(points)} points into Qdrant collection '{COLLECTION_NAME}'.")
    else:
        logging.warning("No points to upsert to Qdrant.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest markdown documents into Qdrant.")
    parser.add_argument("--docs_path", type=str, default="../docs/",
                        help="Path to the directory containing markdown files.")
    args = parser.parse_args()
    ingest_documents(args.docs_path)
