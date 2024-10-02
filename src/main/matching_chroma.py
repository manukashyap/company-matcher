import pandas as pd

from embeddings_chroma import collection, generate_composite_embedding


def match_companies(smaller_df, k=3):
    results = []

    count = collection.count()
    print('Chroma vector count : ' + str(count) )

    for _, row in smaller_df.iterrows():
        query_embedding = generate_composite_embedding(row)

        # Perform the search in ChromaDB
        matches = collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        top_matches = matches['metadatas'][0]
        top_ids = matches['ids'][0]

        # print(top_matches)
        # print(top_ids)
        # Prepare a result row
        result_row = {
            'small_id': row['id'],
            'small_name': row['name'],
            'match_1_id': top_ids[0] if len(top_ids) > 0 else None,
            'match_1_value': str(top_matches[0]) if len(top_matches) > 0 else None,
            'match_2_id': top_ids[1] if len(top_ids) > 1 else None,
            'match_2_name': str(top_matches[1]) if len(top_matches) > 1 else None,
            'match_3_id': top_ids[2] if len(top_ids) > 2 else None,
            'match_3_name': str(top_matches[2]) if len(top_matches) > 2 else None,
        }
        results.append(result_row)

    result_df = pd.DataFrame(results)
    return result_df


def run_matching(df_small, k=3):
    matched_df = match_companies(df_small, k=k)
    # Save the matched results to a CSV
    matched_df.to_csv("data/matched_companies.csv", index=False)
    print("Matching completed. Results saved to 'matched_companies.csv'.")