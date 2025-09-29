import chromadb
from chromadb.config import Settings
import numpy as np


#Chroma uses uclidean distance to measure similarity, could pass it other distance metrics
# Initialize persistent ChromaDB client
client = chromadb.PersistentClient(
    path="./chroma",
    settings=Settings(anonymized_telemetry=False)
)

# Load your collection
collection = client.get_collection(name="jobs")

# CLI loop
print("🔍 Semantic Job Search CLI")
print("Type your query to find matching jobs. Type 'exit' to quit.\n")

while True:
    query = input("Enter your job-related query: ").strip()
    if query.lower() in ["exit", "quit", "stop"]:
        print("👋 Exiting. Thanks for searching!")
        break

    try:
        results = collection.query(query_texts=[query], n_results=5)
        print("\nTop Matches:\n")
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results['distances'][0]):
            print(f"📄 {doc[:250]}")
            print(f"🏷️  Title: {meta.get('Job title')}")
            print(f"🏢 Company: {meta.get('Company')}")
            print(f"📍 Location: {meta.get('Location')}")
            print(f"🛠️  Skills: {meta.get('Skills')}")
            print(f"🔗 URL: {meta.get('URL')}")
            print(f"🥇 Similarity ranking: {np.round(dist, 2)}\n")
    except Exception as e:
        print(f"⚠️ Error during query: {e}\n")