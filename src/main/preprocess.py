import pandas as pd
import re
from urllib.parse import urlparse

# Constants for LinkedIn and website matching
LINKEDIN_PREFIX = "linkedin.com/company/"
WEB_PREFIX = "www."
HTTP_PREFIX = "http://"
HTTP_SECURE_PREFIX = "https://"


def preprocess_country(country):
    """Convert country to lowercase and strip whitespace."""
    if pd.isna(country):
        return ''
    return country.strip().lower()


def preprocess_name(name):
    """Convert name to lowercase and strip whitespace."""
    if pd.isna(name):
        return ''
    return name.strip().lower()


def extract_domain_from_url(url):
    """Extract domain from a URL."""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    if domain.startswith(WEB_PREFIX):
        domain = domain[4:]  # Remove 'www.' prefix
    return domain


def preprocess_website(website):
    """Standardize the website URL by extracting domain."""
    if pd.isna(website):
        return ''
    website = website.strip()
    if not (website.startswith(HTTP_PREFIX) or website.startswith(HTTP_SECURE_PREFIX)):
        website = HTTP_PREFIX + website  # Ensure the URL has the proper format
    return extract_domain_from_url(website)


def preprocess_domain(domain):
    """Standardize domain by stripping unnecessary spaces and ensuring it is lowercase."""
    if pd.isna(domain):
        return ''
    return domain.strip().lower()


def preprocess_linkedin(linkedin):
    """Standardize LinkedIn URL by extracting the company handle."""
    if pd.isna(linkedin):
        return ''
    linkedin = linkedin.strip().lower()
    if LINKEDIN_PREFIX in linkedin:
        linkedin = linkedin.split(LINKEDIN_PREFIX)[-1]  # Extract the company name after prefix
    return linkedin


def preprocess_dataset(df):
    """Apply preprocessing steps to the dataset."""
    df['country'] = df['country'].apply(preprocess_country)
    df['name'] = df['name'].apply(preprocess_name)
    df['domain'] = df['domain'].apply(preprocess_domain)
    df['website'] = df['website'].apply(preprocess_website)
    df['linkedin'] = df['linkedin'].apply(preprocess_linkedin)
    return df


# Example usage
def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df = preprocess_dataset(df)
    return df
