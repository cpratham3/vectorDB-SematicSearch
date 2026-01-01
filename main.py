import os
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Load keys from .env
load_dotenv()

# Initialize Clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "similarity-demo"
MODEL = "text-embedding-3-small"  # Outputs 1536 dimensions

# Setup Index
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating index: {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric='cosine',  # Measures the angle between ideas
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

index = pc.Index(INDEX_NAME)


def get_vector(text):
    """Helper to convert text to embeddings."""
    return client.embeddings.create(input=[text], model=MODEL).data[0].embedding


def save_to_database(text):
    """Saves a string as a vector into Pinecone."""
    vector = get_vector(text)
    # Metadata stores the original text so we can read it later
    index.upsert(vectors=[{"id": str(uuid.uuid4()), "values": vector, "metadata": {"text": text}}])
    print(f"Successfully saved: '{text}'")


def compare_and_search(query_text):
    """Compares a query to saved vectors and returns proximity."""
    query_vector = get_vector(query_text)
    results = index.query(vector=query_vector, top_k=3, include_metadata=True)

    print(f"\nResults for: '{query_text}'")
    for match in results['matches']:
        # Score is 0.0 to 1.0 (1.0 is a perfect semantic match)
        print(f"--- Closeness Score: {match['score']:.4f} | Text: {match['metadata']['text']}")


# --- RUNNING THE APP ---
if __name__ == "__main__":
    # 1. Add some data to your brain
    save_to_database("I love eating fresh green apples.")
    save_to_database("The server is currently undergoing maintenance.")
    save_to_database("Python is a popular language for data science.")

    # 2. Compare a new search term to your saved data
    search_term = input("\nEnter a phrase to compare: ")
    compare_and_search(search_term)