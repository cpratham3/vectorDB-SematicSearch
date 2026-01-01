import os
import uuid
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# 1. SETUP
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "interactive-demo"
MODEL = "text-embedding-3-small"

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

index = pc.Index(INDEX_NAME)


# 2. THE MATH (UNDER THE HOOD)
def calculate_manual_similarity(vec1, vec2):
    """How the computer actually 'compares' two 1536-dimensional points."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    # Cosine Similarity Formula: (A · B) / (||A|| * ||B||)
    dot_product = np.dot(v1, v2)
    norm_a = np.linalg.norm(v1)
    norm_b = np.linalg.norm(v2)
    return dot_product / (norm_a * norm_b)


# 3. CORE FUNCTIONS
def get_embedding(text):
    return client.embeddings.create(input=[text], model=MODEL).data[0].embedding


def add_entry(text):
    vector = get_embedding(text)
    index.upsert(vectors=[{"id": str(uuid.uuid4()), "values": vector, "metadata": {"text": text}}])
    print(f"\n Entry Saved to Vector DB: '{text}'")
    return vector


def run_comparison():
    print("\n--- VECTOR DB INTERACTIVE LAB ---")
    print("1. Use hardcoded examples (Fruit, Coding, Servers)")
    print("2. Add your own custom text to the database")
    choice = input("Select an option (1 or 2): ")

    if choice == '1':
        examples = ["I love eating fresh green apples.",
                    "Python is a popular language for data science.",
                    "The server is currently undergoing maintenance."]
        for ex in examples: add_entry(ex)
    else:
        custom_text = input("Enter the text you want to store in the DB: ")
        add_entry(custom_text)

    # The Comparison Phase
    query = input("\nNow, enter a search term to compare against the DB: ")
    query_vector = get_embedding(query)

    # Query the DB
    results = index.query(vector=query_vector, top_k=1, include_metadata=True)

    if results['matches']:
        best_match = results['matches'][0]
        db_text = best_match['metadata']['text']
        db_score = best_match['score']

        # Manual verification to show the user the "Work"
        db_vector = index.fetch([best_match['id']])['vectors'][best_match['id']]['values']
        manual_score = calculate_manual_similarity(query_vector, db_vector)

        print(f"\n--- RESULTS ---")
        print(f"Your Query: '{query}'")
        print(f"Closest Match in DB: '{db_text}'")
        print(f"Cloud DB Similarity Score: {db_score:.4f}")
        print(f"Manual Math Verification:  {manual_score:.4f}")
        print(f"Insight: This is {manual_score * 100:.1f}% semantically similar.")


if __name__ == "__main__":
    run_comparison()