from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [

    "Ice cream is almsot the best thing in the world",
    "I love to go hiking in the mountains",
    "The University of Utah sucks.",
    "I cannot think of anything cool to say today in class.",
    "Chococlate chip cookies are the best dessert ever. Period.",
    "California expereinces a ton of earthquakes.",
    "My grandmother used to get me to go to sleep at night by telling my stories of nuclear fusion.",
    "My first car was black in color",
    "This class sux."


]

# Step 1: Generate embeddings for our 'documents'
embeddings = model.encode(texts)

# Step 2a: Inspect the embeddings
for i, emb in enumerate(embeddings):
    print(f"Text {i} preview: ", np.round(emb[:5], 3))

# Step 2b: Visualize the embeddings' similarities in 2d space 
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

plt.figure(figsize=(6, 4))
plt.scatter(reduced[:, 0], reduced[:, 1])
for i, text in enumerate(texts):
    plt.annotate(f"Text {i}: {text}", (reduced[i, 0], reduced[i, 1]))
plt.title("Embeddings Visualized (PCA)")
plt.show() 