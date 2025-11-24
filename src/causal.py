"""
Causal Inference Utilities

Implementation of causal inference methods including X-learner meta-learner,
CATE estimation, and treatment effect analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.base import BaseEstimator
from econml.metalearners import XLearner
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_causal_data(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    feature_cols: List[str],
    categorical_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data for causal inference by encoding categoricals and extracting arrays.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    treatment_col : str
        Name of treatment column (binary: 0/1)
    outcome_col : str
        Name of outcome column
    feature_cols : list of str
        List of feature column names
    categorical_cols : list of str, optional
        List of categorical columns to one-hot encode
        
    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray, list of str)
        - X: Feature matrix
        - T: Treatment vector
        - Y: Outcome vector
        - feature_names: List of feature names after encoding
    """
    df_encoded = df.copy()
    
    # One-hot encode categorical variables
    if categorical_cols:
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=False)
        
        # Update feature columns to include encoded columns
        encoded_cols = [col for col in df_encoded.columns 
                       if any(col.startswith(cat + '_') for cat in categorical_cols)]
        feature_cols_updated = [col for col in feature_cols if col not in categorical_cols]
        feature_cols_updated.extend(encoded_cols)
    else:
        feature_cols_updated = feature_cols
    
    X = df_encoded[feature_cols_updated].values
    T = df_encoded[treatment_col].values
    Y = df_encoded[outcome_col].values
    
    return X, T, Y, feature_cols_updated


def train_xlearner(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    base_model: Optional[BaseEstimator] = None,
    propensity_model: Optional[BaseEstimator] = None,
    random_state: int = 42
) -> XLearner:
    """
    Train X-learner meta-learner for CATE estimation.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    T : np.ndarray
        Treatment vector (binary)
    Y : np.ndarray
        Outcome vector
    base_model : BaseEstimator, optional
        Base model for treatment effect estimation
        Default: RandomForestRegressor
    propensity_model : BaseEstimator, optional
        Model for propensity score estimation
        Default: GradientBoostingRegressor
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    XLearner
        Trained X-learner model
    """
    if base_model is None:
        base_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state
        )
    
    if propensity_model is None:
        propensity_model = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=random_state
        )
    
    xlearner = XLearner(
        models=base_model,
        propensity_model=propensity_model
    )
    
    xlearner.fit(Y, T, X=X)
    
    return xlearner


def estimate_cate(
    xlearner: XLearner,
    X: np.ndarray
) -> np.ndarray:
    """
    Estimate Conditional Average Treatment Effect (CATE) for each observation.
    
    Parameters
    ----------
    xlearner : XLearner
        Trained X-learner model
    X : np.ndarray
        Feature matrix
        
    Returns
    -------
    np.ndarray
        CATE estimates for each observation
    """
    return xlearner.effect(X)


def get_cate_statistics(cate: np.ndarray) -> dict:
    """
    Calculate summary statistics for CATE distribution.
    
    Parameters
    ----------
    cate : np.ndarray
        Array of CATE estimates
        
    Returns
    -------
    dict
        Dictionary with statistics including mean, std, min, max, percentiles
    """
    return {
        'mean': np.mean(cate),
        'std': np.std(cate),
        'min': np.min(cate),
        'max': np.max(cate),
        'p10': np.percentile(cate, 10),
        'p25': np.percentile(cate, 25),
        'p50': np.percentile(cate, 50),
        'p75': np.percentile(cate, 75),
        'p90': np.percentile(cate, 90)
    }


def extract_feature_importance(
    xlearner: XLearner,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extract feature importance from X-learner models.
    
    Parameters
    ----------
    xlearner : XLearner
        Trained X-learner model
    feature_names : list of str
        List of feature names
        
    Returns
    -------
    pd.DataFrame
        DataFrame with features and their importance scores, sorted by importance
    """
    try:
        # Try to access internal models
        model_t = xlearner.models_t[0]
        model_c = xlearner.models_c[0]
        
        importance_t = model_t.feature_importances_
        importance_c = model_c.feature_importances_
        
        # Average importances
        avg_importance = (importance_t + importance_c) / 2
        
    except (AttributeError, IndexError):
        # Fallback: train a simple model on CATE
        from sklearn.ensemble import RandomForestRegressor
        
        # This is a workaround - would need X and cate passed in
        # For now, return None
        return None
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': avg_importance
    }).sort_values('importance', ascending=False)
    
    return importance_df


def plot_cate_distribution(
    cate: np.ndarray,
    treatment: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Plot CATE distribution with histogram and optional box plot by treatment.
    
    Parameters
    ----------
    cate : np.ndarray
        Array of CATE estimates
    treatment : np.ndarray, optional
        Treatment assignment vector for box plot
    figsize : tuple, default=(14, 5)
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    if treatment is not None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(7, 5))
        axes = [axes]
    
    # Histogram
    axes[0].hist(cate, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0].axvline(cate.mean(), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {cate.mean():.2f}')
    axes[0].set_xlabel('CATE (Treatment Effect)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Distribution of Treatment Effects', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Box plot by treatment (if provided)
    if treatment is not None:
        df_plot = pd.DataFrame({'CATE': cate, 'Treatment': treatment})
        df_plot.boxplot(column='CATE', by='Treatment', ax=axes[1])
        axes[1].set_xlabel('Treatment Group', fontsize=11)
        axes[1].set_ylabel('CATE', fontsize=11)
        axes[1].set_title('CATE by Treatment Assignment', fontsize=12, fontweight='bold')
        axes[1].get_figure().suptitle('')
    
    plt.tight_layout()
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot feature importance as horizontal bar chart.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns
    top_n : int, default=15
        Number of top features to display
    figsize : tuple, default=(10, 8)
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(top_features)), top_features['importance'], color='#2ecc71')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_ylabel('Feature', fontsize=11)
    ax.set_title(f'Top {top_n} Features Driving Treatment Response', 
                 fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def calculate_ate(cate: np.ndarray) -> float:
    """
    Calculate Average Treatment Effect (ATE) from CATE estimates.
    
    Parameters
    ----------
    cate : np.ndarray
        Array of CATE estimates
        
    Returns
    -------
    float
        Average treatment effect
    """
    return np.mean(cate)


def identify_high_responders(
    cate: np.ndarray,
    percentile: float = 80
) -> np.ndarray:
    """
    Identify high responders based on CATE percentile threshold.
    
    Parameters
    ----------
    cate : np.ndarray
        Array of CATE estimates
    percentile : float, default=80
        Percentile threshold (0-100)
        
    Returns
    -------
    np.ndarray
        Boolean array indicating high responders
    """
    threshold = np.percentile(cate, percentile)
    return cate >= threshold
