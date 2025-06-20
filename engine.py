import os
from dotenv import load_dotenv

# --- NEW: Import Settings and the correct model classes ---
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# Load environment variables
load_dotenv()

# Constants
DATA_DIR = "./data"
PERSIST_DIR = "./storage"


def create_and_persist_index():
    """Creates and persists the vector store index from documents in DATA_DIR."""
    if not os.path.exists(PERSIST_DIR):
        print(f"Creating index from documents in {DATA_DIR}...")

        # --- NEW: Configure models globally using Settings ---
        # Set up the LLM
        llm = GoogleGenAI(model_name="models/gemini-1.5-flash-latest")
        Settings.llm = llm

        # Set up the embedding model
        embed_model = GoogleGenAIEmbedding(
            model_name="models/embedding-001", api_key=os.getenv("GEMINI_API_KEY")
        )
        Settings.embed_model = embed_model

        # Load the documents
        documents = SimpleDirectoryReader(DATA_DIR).load_data()

        # Create and build the index (no need to pass models here anymore)
        index = VectorStoreIndex.from_documents(documents)

        # Store it for later
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        print(f"Index created and saved to {PERSIST_DIR}.")
    else:
        print(f"Index already exists at {PERSIST_DIR}. No action taken.")


# --- Main execution ---
if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        raise EnvironmentError("GEMINI_API_KEY not found in .env file.")

    # --- IMPORTANT: Delete the old storage directory before running ---
    if os.path.exists(PERSIST_DIR):
        import shutil

        print(
            f"Found old index. Deleting {PERSIST_DIR} to rebuild with new embedding model."
        )
        shutil.rmtree(PERSIST_DIR)

    create_and_persist_index()
    print("Script finished.")
