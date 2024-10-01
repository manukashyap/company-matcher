import pandas as pd

from src.main.preprocess import preprocess_company_data

company_dataset_path = "/Users/manukashysp/Downloads/sample_company_dataset.csv"
person_dataset_path = "/Users/manukashysp/Downloads/sample_person_dataset.csv"

company_df = pd.read_csv(company_dataset_path)
person_df = pd.read_csv(person_dataset_path)

company_df_clean = preprocess_company_data(company_df)
print(company_df.head())