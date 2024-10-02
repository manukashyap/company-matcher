import pandas as pd
import re
from urllib.parse import urlparse

from pandas import DataFrame

# Constants for URL scheme
HTTP = "http://"
HTTPS = "https://"


# Helper function to ensure proper URL format
def ensure_http_scheme(website):
    """Ensure that the website URL starts with either http or https."""
    if not website.startswith(HTTP) and not website.startswith(HTTPS):
        return HTTP + website  # Default to HTTP if no scheme is provided
    return website


# Updated website preprocessing function
def preprocess_website(website):
    """Standardize the website URL by extracting the domain, handling both http and https."""
    if pd.isna(website) or not website.strip():
        return 'unknown website'

    website = website.strip()  # Remove extra whitespace
    website = ensure_http_scheme(website)  # Ensure the URL starts with http/https

    return extract_domain_from_url(website)


# Example usage of constants
def extract_domain_from_url(url):
    """Extract domain name from a given URL."""
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        if domain.startswith("www."):
            domain = domain[4:]  # Remove 'www.' if present
        return domain
    except Exception:
        return ''

def normalize_linkedin_url(linkedin_url):
    """Extract company identifier from a LinkedIn URL."""
    try:
        if 'linkedin' in linkedin_url:
            return linkedin_url.split('/')[-1]  # Extract last part as company identifier
        return linkedin_url
    except Exception:
        return linkedin_url

def preprocess_name(name):
    """Normalize the company name by converting to lowercase and stripping whitespace."""
    if pd.isna(name):
        return ''
    return name.lower().strip()

def preprocess_linkedin(linkedin_url):
    """Extract meaningful company identifier from LinkedIn URLs."""
    if pd.isna(linkedin_url):
        return 'N/A'
    return normalize_linkedin_url(linkedin_url)

def preprocess_country(country):
    """Clean and standardize country names."""
    if pd.isna(country):
        return 'Unknown'  # Fill missing country with 'Unknown'
    return country.strip()

def preprocess_domain(domain, website):
    if pd.isna(domain) or domain == '':
        return "unknown_domain"
    return domain.strip()


def apply_preprocessing(df1: DataFrame, df2: DataFrame):
    # Apply preprocessing to Dataset 1
    df1['name'] = df1['name'].apply(preprocess_name)
    df1['domain'] = df1.apply(lambda row: preprocess_domain(row['domain'], row['website']), axis=1)
    df1['website'] = df1['website'].apply(preprocess_website)
    df1['linkedin'] = df1['linkedin'].apply(preprocess_linkedin)
    df1['country'] = df1['country'].apply(preprocess_country)

    # Apply preprocessing to Dataset 2
    df2['name'] = df2['name'].apply(preprocess_name)
    df2['domain'] = df2.apply(lambda row: preprocess_domain(row['domain'], row['website']), axis=1)
    df2['website'] = df2['website'].apply(preprocess_website)
    df2['linkedin'] = df2['linkediin'].apply(preprocess_linkedin)  # Note: fixed typo in column named 'linkediin'
    df2['country'] = df2['country'].apply(preprocess_country)

    # Saving
    save_data_sample_csv(df1, df2)

    return df1, df2


def save_data_sample_csv(df1: DataFrame, df2: DataFrame):
    # Save the cleaned data back to CSV files
    df1.sample(50).to_csv('preprocessed_dataset1.csv', index=False)
    df2.sample(50).to_csv('preprocessed_dataset2.csv', index=False)
    print("Preprocessing complete. Cleaned datasets saved as CSV files.")