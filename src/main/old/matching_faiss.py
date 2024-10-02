import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer

from src.main.old.embeddings_faiss import generate_knowledge_embeddings

# Load pre-trained knowledge model
knowledge_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


def load_faiss_index():
    """Load the FAISS index from disk."""
    index = faiss.read_index("faiss_index.bin")
    return index



def find_top_matches(faiss_index, embedding, top_k=3):
    """Search FAISS index for top K matches."""
    embedding = embedding.reshape(1, -1)
    distances, indices = faiss_index.search(embedding, top_k)
    return distances[0], indices[0]


if __name__ == "__main__":
    # Load the FAISS index and the large dataset
    faiss_index = load_faiss_index()
    df_large = pd.read_csv("../data/large_dataset_cleaned.csv")

    # Load the preprocessed small dataset
    df_small = pd.read_csv("../data/small_dataset_cleaned.csv")

    # Generate embeddings for the smaller dataset
    embeddings_small = generate_knowledge_embeddings(df_small)

    # Find top 3 matches for each entry in the smaller dataset
    results = []
    for i, row in df_small.iterrows():
        embedding = embeddings_small[i]
        distances, indices = find_top_matches(faiss_index, embedding)

        # Get the top 3 matches
        top_3_matches = [(df_large.iloc[idx]['id'], df_large.iloc[idx]['name']) for idx in indices]

        # Create result row: (id, name, match1_id, match1_name, match2_id, match2_name, match3_id, match3_name)
        result_row = [row['id'], row['name']] + [item for sublist in top_3_matches for item in sublist]
        results.append(result_row)

    # Create result dataframe and save to CSV
    result_df = pd.DataFrame(results,
                             columns=['id', 'name', 'match1_id', 'match1_name', 'match2_id', 'match2_name', 'match3_id',
                                      'match3_name'])
    result_df.to_csv("matching_results.csv", index=False)