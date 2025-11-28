"""Module for data preprocessing and feature engineering."""
import pandas as pd
import re


def extract_zip_code(geocode):
    """
    Extract zip code from geography string.

    Args:
        geocode: String containing geographic information

    Returns:
        Extracted 5-digit zip code or None
    """
    pattern = r'\d{5}$'
    match = re.search(pattern, geocode)

    if match:
        return match.group(0)
    return None


def add_zip_codes_to_population(population_df):
    """
    Add zip codes column to population dataframe.

    Args:
        population_df: DataFrame with 'Geography' column

    Returns:
        DataFrame with added 'Zip Code' column
    """
    df = population_df.copy()
    df['Zip Code'] = df['Geography'].apply(extract_zip_code)
    return df


def merge_cafe_with_population(cafe_df, population_df):
    """
    Merge cafe data with population data.

    Args:
        cafe_df: Coffee shop DataFrame
        population_df: Population DataFrame

    Returns:
        Merged DataFrame
    """
    cafe_data = cafe_df.copy()
    cafe_data['Zip Code'] = cafe_data['Zip Code'].astype(str)

    df = pd.merge(cafe_data, population_df)

    columns = cafe_data.columns.values.tolist() + ['Total']
    df = df[columns]
    df = df.rename(columns={'Total': 'Population'})

    return df


def add_coffee_shop_counts(df):
    """
    Add coffee shop count per zip code to the dataframe.

    Args:
        df: DataFrame with 'Zip Code' column

    Returns:
        DataFrame with added 'Coffee Shop Count' column
    """
    coffee_shop_counts = df['Zip Code'].value_counts().reset_index()
    coffee_shop_counts.columns = ['Zip Code', 'Coffee Shop Count']

    df = df.copy()
    df['Zip Code'] = df['Zip Code'].astype(str)
    coffee_shop_counts['Zip Code'] = coffee_shop_counts['Zip Code'].astype(str)

    df = df.merge(coffee_shop_counts, on='Zip Code', how='left')

    return df


def select_features(df, features=None):
    """
    Select specific features from dataframe.

    Args:
        df: Input DataFrame
        features: List of feature names. If None, uses default features.

    Returns:
        DataFrame with selected features
    """
    if features is None:
        features = ['Zip Code', 'Rating', 'Median Salary', 'Latte Price', 'Population']

    return df[features]


def get_top_zip_codes(df, n=5):
    """
    Get top N zip codes based on population, coffee shop count, rating, and salary.

    Args:
        df: Input DataFrame
        n: Number of top zip codes to return (default: 5)

    Returns:
        DataFrame with data for top N zip codes
    """
    sorted_df = df.sort_values(
        by=['Population', 'Coffee Shop Count', 'Rating', 'Median Salary'],
        ascending=[False, True, True, False]
    ).reset_index(drop=True)

    top_zip_codes = []
    for i in range(len(sorted_df)):
        if len(top_zip_codes) >= n:
            break
        zip_code = sorted_df['Zip Code'][i]
        if zip_code not in top_zip_codes:
            top_zip_codes.append(zip_code)

    top_zip_codes_df = sorted_df[sorted_df['Zip Code'].isin(top_zip_codes)]

    return top_zip_codes_df


def preprocess_data(cafe_df, population_df):
    """
    Complete preprocessing pipeline.

    Args:
        cafe_df: Raw coffee shop DataFrame
        population_df: Raw population DataFrame

    Returns:
        Tuple of (processed_df, top_zip_codes_df)
    """
    # Add zip codes to population data
    population_df = add_zip_codes_to_population(population_df)

    # Merge datasets
    df = merge_cafe_with_population(cafe_df, population_df)

    # Select features
    df = select_features(df)

    # Add coffee shop counts
    df = add_coffee_shop_counts(df)

    # Get top zip codes
    top_zip_codes_df = get_top_zip_codes(df)

    return df, top_zip_codes_df
