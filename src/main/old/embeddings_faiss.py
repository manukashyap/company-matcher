# Embedding Generation and FAISS Indexing

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# Load the pre-trained knowledge-based model
knowledge_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


# FAISS indexer class to handle embedding storage
class FAISSIndexer:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)  # Using L2 distance

    def add_to_index(self, embeddings):
        embeddings = normalize(embeddings)  # Normalize embeddings
        self.index.add(embeddings)

    def search(self, embedding, top_k=3):
        embedding = embedding.reshape(1, -1)
        return self.index.search(embedding, top_k)


def generate_knowledge_embeddings(df):
    """Generate knowledge-based embeddings for a dataset."""
    embeddings = []
    for _, row in df.iterrows():
        try:
            entity_representation = f"{row['name']} {row['country']} {row['website']} {row['linkedin']}"
            embedding = knowledge_model.encode(entity_representation)
            embeddings.append(embedding)
        except Exception:
            print(row)
            print("Error")
    return np.array(embeddings)


def generate_knowledge_embeddings_batch(df, batch_size=10):
    """Generate knowledge-based embeddings for a dataset in batches."""
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        embeddings = []
        for _, row in batch.iterrows():
            try:
                entity_representation = f"{row['name']} {row['country']} {row['website']} {row['linkedin']}"
                embedding = knowledge_model.encode(entity_representation)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing row: {row}, Error: {e}")

        yield np.array(embeddings, dtype=np.float16)


def add_embeddings_to_faiss_index(df, faiss_indexer, batch_size=10):
    """Generate and add embeddings to the FAISS index incrementally."""
    for batch_embeddings in generate_knowledge_embeddings_batch(df, batch_size=batch_size):
        if len(batch_embeddings) > 0:
            faiss_indexer.add(batch_embeddings.astype('float16'))


if __name__ == "__main__":
    # Load the preprocessed large dataset
    df_large = pd.read_csv("../data/large_dataset_cleaned.csv")
    # df_large = df_large.loc[0:10]
    # Generate embeddings for the large dataset
    embeddings_large = generate_knowledge_embeddings(df_large.loc[0:10])

    # Initialize FAISS indexer and add embeddings
    dimension = knowledge_model.get_sentence_embedding_dimension()
    faiss_indexer = faiss.IndexFlatL2(dimension)
    add_embeddings_to_faiss_index(df_large, faiss_indexer, batch_size=10)

    # Save the FAISS index and embeddings
    faiss.write_index(faiss_indexer, "data/faiss_index.bin")
    np.save("../data/embeddings_large.npy", embeddings_large)
