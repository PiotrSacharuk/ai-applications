"""Module for loading data files."""
import pandas as pd
from pathlib import Path


def get_data_dir():
    """Get the data directory path relative to this module."""
    # modules/data_loader.py -> src -> ai-applications -> data
    return Path(__file__).parent.parent.parent / 'data'


def load_coffee_shop_data(filepath=None):
    """
    Load coffee shop data from Excel file.

    Args:
        filepath: Optional path to the file. If None, uses default location.

    Returns:
        DataFrame with coffee shop data
    """
    if filepath is None:
        filepath = get_data_dir() / 'Coffee_Shop_data.xlsx'

    df = pd.read_excel(filepath)
    return df


def load_population_data(filepath=None, skiprows=None):
    """
    Load population data from CSV file.

    Args:
        filepath: Optional path to the file. If None, uses default location.
        skiprows: Rows to skip when reading CSV. Default is [0].

    Returns:
        DataFrame with population data
    """
    if filepath is None:
        filepath = get_data_dir() / 'population.csv'

    if skiprows is None:
        skiprows = [0]

    population = pd.read_csv(filepath, skiprows=skiprows)
    return population


def load_all_data():
    """
    Load all required data files.

    Returns:
        Tuple of (coffee_shop_df, population_df)
    """
    coffee_df = load_coffee_shop_data()
    population_df = load_population_data()

    return coffee_df, population_df
