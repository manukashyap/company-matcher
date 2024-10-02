import pandas as pd

from src.main.preprocess import apply_preprocessing


def match_company():
    company_dataset_path = "/Users/manukashysp/Downloads/sample_company_dataset.csv"
    company_short_dataset_path = "/Users/manukashysp/Downloads/sample_person_dataset.csv"

    company_df = pd.read_csv(company_dataset_path)
    company_short_df = pd.read_csv(company_short_dataset_path)

    company_df_clean,  company_short_df_clean = apply_preprocessing(company_df, company_short_df)
    print(company_df.head())


if __name__ == "__main__":
    match_company()