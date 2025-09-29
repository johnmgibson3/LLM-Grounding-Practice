from sentence_transformers import SentenceTransformer
import pandas as pd
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient
from pathlib import Path

model = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_excel("Jobs for Semantic Search.xlsx")
skilsDesc = df["Skills"] + " : " + df["Description"]

texts = skilsDesc.tolist()

#This is how the metadata is created and converted to a dictionary
metadatas = df[["Job title", "Company", "Location", "Skills", "URL"]].copy()

metadatas = metadatas.to_dict(orient="records")

embeddings = model.encode(texts)

#Important: only use the command below to create a chromadb client in memory
#client = chromadb.Client(Settings(anonymized_telemetry=False))

#This is how an instance of a persistent client is created and used to then create a collection
client = chromadb.PersistentClient(
    path="./chroma",
    settings=Settings(anonymized_telemetry=False)
)

collection = client.create_collection(name="jobs")

collection.add(
    documents=texts,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=[f"job_{i}" for i in range(len(texts))]
)