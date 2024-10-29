import chromadb
from chromadb.config import Settings
import os


def get_chroma_client():
    # Create persist directory if it doesn't exist
    persist_directory = os.path.join(os.getcwd(), "chroma_db")
    os.makedirs(persist_directory, exist_ok=True)

    # Initialize the client with the new configuration
    client = chromadb.PersistentClient(
        path=persist_directory
    )

    return client


def get_or_create_collection(client, name="documents"):
    try:
        # Try to get existing collection
        collection = client.get_collection(name=name)
    except ValueError:
        # Create new collection if it doesn't exist
        collection = client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
    return collection
