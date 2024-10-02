from Levenshtein import distance as levenshtein_distance
from sklearn.preprocessing import normalize

from src.main.embeddings import find_closest_match, generate_embedding


# Function to combine FAISS embedding distance with Levenshtein distance
def combined_score(small_row, large_row, embedding_distance):
    # Calculate L dist for name and domain as additional scores
    name_score = levenshtein_distance(small_row['name'], large_row['name'])
    domain_score = levenshtein_distance(small_row['domain'], large_row['domain'])

    # Combine embedding distance and Levenshtein score (tune the weights as per your need)
    total_score = embedding_distance + (name_score / 100.0) + (domain_score / 100.0)  # Normalized score
    return total_score

def find_closest(df_small):
    # Refine the closest match using a combination of FAISS distance and fuzzy matching
    for idx, row in df_small.iterrows():
        embedding_small = generate_embedding(row)
        embedding_small = normalize(embedding_small.reshape(1, -1))

        # Find top 3 candidates from FAISS
        distances, indices = find_closest_match(embedding_small, top_k=3)

        # Re-rank the results using fuzzy matching (Levenshtein distance)
        best_score = float('inf')
        best_match = None

        for i, (distance, index) in enumerate(zip(distances[0], indices[0])):
            matching_row = df_large.iloc[index]
            score = combined_score(row, matching_row, distance)

            if score < best_score:
                best_score = score
                best_match = matching_row

        print(f"Best match for {row['name']} -> {best_match['name']} with combined score: {best_score}")