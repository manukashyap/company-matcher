import pandas as pd
import spacy
import faiss
import numpy as np
from sklearn.preprocessing import normalize

# Load spacy model for embeddings (e.g., using a transformer model for better quality embeddings)
nlp = spacy.load('en_core_web_md')

# Load datasets
df_large = pd.read_csv('preprocessed_dataset1.csv')
df_small = pd.read_csv('preprocessed_dataset2.csv')

# Helper function to create a sentence embedding for each row
def generate_embedding(row):
    text = f"{row['name']} {row['domain']} {row['website']} {row['linkedin']} {row['country']}"
    doc = nlp(text)
    return doc.vector

# Generate embeddings for the larger dataset
embeddings_large = np.array([generate_embedding(row) for _, row in df_large.iterrows()])
embeddings_large = normalize(embeddings_large)

# Create FAISS Index and add the large dataset embeddings
dimension = embeddings_large.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_large)

# Helper function to find the closest match using FAISS
def find_closest_match(embedding, top_k=3):
    embedding = embedding.reshape(1, -1)  # Reshape embedding to match FAISS input format
    distances, indices = index.search(embedding, top_k)
    return distances, indices


# Generate embeddings for the smaller dataset and find the closest matches
for idx, row in df_small.iterrows():
    embedding_small = generate_embedding(row)
    embedding_small = normalize(embedding_small.reshape(1, -1))  # Normalize the small embedding

    # Find the top 3 closest matches in the large dataset
    distances, indices = find_closest_match(embedding_small, top_k=3)

    # Retrieve the matching rows from the larger dataset
    for i, (distance, index) in enumerate(zip(distances[0], indices[0])):
        matching_row = df_large.iloc[index]
        print(f"Small Entry: {row['name']} | Closest Match: {matching_row['name']} | Distance: {distance}")