import preprocess
from src.main.embeddings_chroma import add_embeddings_to_chroma
from src.main.matching_chroma import run_matching

if __name__ == "__main__":
    # Assume 'large_dataset.csv' and 'small_dataset.csv' are the dataset files
    df_large = preprocess.load_and_preprocess('data/large_dataset.csv')
    df_small = preprocess.load_and_preprocess('data/small_dataset.csv')
    # Testing smaller sample
    # df_large = df_large.loc[0:1000]
    # df_small = df_small.loc[0:100]
    # Further processing or saving the cleaned dataset
    df_large.to_csv('data/large_dataset_cleaned.csv', index=False)
    df_small.to_csv('data/small_dataset_cleaned.csv', index=False)

    add_embeddings_to_chroma(df_large)
    run_matching(df_small)
