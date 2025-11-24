"""
Utility Functions

Helper functions for data loading, preprocessing, and common operations.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import os


def load_experiment_data(
    data_dir: str = '../data',
    daily_file: str = 'daily_ai_saas_experiment.csv',
    user_file: str = 'user_ai_saas_experiment.csv',
    use_parquet: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load daily and user-level experiment data.
    
    Parameters
    ----------
    data_dir : str, default='../data'
        Directory containing data files
    daily_file : str, default='daily_ai_saas_experiment.csv'
        Name of daily data file
    user_file : str, default='user_ai_saas_experiment.csv'
        Name of user data file
    use_parquet : bool, default=False
        Whether to load parquet files instead of CSV
        
    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        Daily and user dataframes
    """
    if use_parquet:
        daily_file = daily_file.replace('.csv', '.parquet')
        user_file = user_file.replace('.csv', '.parquet')
        daily_df = pd.read_parquet(os.path.join(data_dir, daily_file))
        user_df = pd.read_parquet(os.path.join(data_dir, user_file))
    else:
        daily_df = pd.read_csv(os.path.join(data_dir, daily_file))
        user_df = pd.read_csv(os.path.join(data_dir, user_file))
    
    return daily_df, user_df


def check_data_quality(
    df: pd.DataFrame,
    name: str = 'DataFrame'
) -> dict:
    """
    Perform data quality checks and return summary.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check
    name : str, default='DataFrame'
        Name for reporting
        
    Returns
    -------
    dict
        Dictionary with quality metrics
    """
    quality_report = {
        'name': name,
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'dtypes': df.dtypes.to_dict()
    }
    
    return quality_report


def print_data_summary(
    df: pd.DataFrame,
    name: str = 'DataFrame'
) -> None:
    """
    Print comprehensive data summary.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to summarize
    name : str, default='DataFrame'
        Name for display
    """
    print("=" * 80)
    print(f"{name.upper()}")
    print("=" * 80)
    print(f"Shape: {df.shape}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")
    
    if 'cohort' in df.columns:
        print(f"\nCohort Distribution:\n{df['cohort'].value_counts()}")
    
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\nSummary Statistics:")
    print(df.describe())


def filter_cohorts(
    df: pd.DataFrame,
    cohorts: List[str],
    cohort_col: str = 'cohort'
) -> pd.DataFrame:
    """
    Filter dataframe to specific cohorts.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    cohorts : list of str
        List of cohort names to keep
    cohort_col : str, default='cohort'
        Name of cohort column
        
    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    return df[df[cohort_col].isin(cohorts)].copy()


def add_derived_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add derived features to dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    pd.DataFrame
        Dataframe with additional features
    """
    df = df.copy()
    
    # Add engagement rate if relevant columns exist
    if 'ai_calls' in df.columns and 'time_on_platform' in df.columns:
        df['ai_calls_per_hour'] = df['ai_calls'] / (df['time_on_platform'] / 60 + 1e-6)
    
    # Add productivity rate
    if 'tasks_completed' in df.columns and 'time_on_platform' in df.columns:
        df['tasks_per_hour'] = df['tasks_completed'] / (df['time_on_platform'] / 60 + 1e-6)
    
    # Add revenue per task
    if 'revenue' in df.columns and 'tasks_completed' in df.columns:
        df['revenue_per_task'] = df['revenue'] / (df['tasks_completed'] + 1e-6)
    
    return df


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_col: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train and test sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    test_size : float, default=0.2
        Proportion of data for test set
    random_state : int, default=42
        Random seed
    stratify_col : str, optional
        Column name for stratified splitting
        
    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        Train and test dataframes
    """
    from sklearn.model_selection import train_test_split
    
    stratify = df[stratify_col] if stratify_col else None
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    return train_df, test_df


def calculate_cohort_balance(
    df: pd.DataFrame,
    cohort_col: str = 'cohort',
    feature_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate balance statistics across cohorts.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    cohort_col : str, default='cohort'
        Name of cohort column
    feature_cols : list of str, optional
        List of features to check balance for
        If None, uses all numeric columns
        
    Returns
    -------
    pd.DataFrame
        Balance statistics (mean and std) by cohort
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if cohort_col in feature_cols:
            feature_cols.remove(cohort_col)
    
    balance_stats = df.groupby(cohort_col)[feature_cols].agg(['mean', 'std'])
    
    return balance_stats


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format value as percentage string.
    
    Parameters
    ----------
    value : float
        Value to format (0-1 or 0-100)
    decimals : int, default=1
        Number of decimal places
        
    Returns
    -------
    str
        Formatted percentage string
    """
    if value <= 1:
        value *= 100
    return f"{value:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """
    Format number with thousands separator.
    
    Parameters
    ----------
    value : float
        Value to format
    decimals : int, default=2
        Number of decimal places
        
    Returns
    -------
    str
        Formatted number string
    """
    return f"{value:,.{decimals}f}"


def save_results(
    df: pd.DataFrame,
    filename: str,
    output_dir: str = '../results',
    format: str = 'csv'
) -> None:
    """
    Save results to file.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    filename : str
        Output filename (without extension)
    output_dir : str, default='../results'
        Output directory
    format : str, default='csv'
        Output format ('csv' or 'parquet')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, f"{filename}.{format}")
    
    if format == 'csv':
        df.to_csv(filepath, index=False)
    elif format == 'parquet':
        df.to_parquet(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Results saved to: {filepath}")
