import dask.dataframe as dd
import pandas


def preprocess_company_data(company_df: pandas.DataFrame):
    head = company_df.head()
    print(head)
