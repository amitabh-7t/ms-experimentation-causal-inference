"""
A/B Testing Utilities

Statistical testing framework for experimentation including t-tests,
lift calculations, and result visualization.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def perform_ttest(
    df: pd.DataFrame,
    cohort1: str,
    cohort2: str,
    metric: str,
    cohort_col: str = 'cohort'
) -> Dict:
    """
    Perform independent t-test between two cohorts for a given metric.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    cohort1 : str
        Name of first cohort
    cohort2 : str
        Name of second cohort
    metric : str
        Name of metric column to test
    cohort_col : str, default='cohort'
        Name of cohort column
        
    Returns
    -------
    dict
        Dictionary containing test results including:
        - metric: metric name
        - cohort_1, cohort_2: cohort names
        - mean_1, mean_2: cohort means
        - t_statistic: t-test statistic
        - p_value: p-value
        - lift_pct: percentage lift from cohort1 to cohort2
        - significant: whether result is significant at α=0.05
    """
    group1 = df[df[cohort_col] == cohort1][metric]
    group2 = df[df[cohort_col] == cohort2][metric]
    
    t_stat, p_value = stats.ttest_ind(group1, group2)
    
    mean1 = group1.mean()
    mean2 = group2.mean()
    lift_pct = ((mean2 - mean1) / mean1) * 100 if mean1 != 0 else 0
    
    return {
        'metric': metric,
        'cohort_1': cohort1,
        'cohort_2': cohort2,
        'mean_1': mean1,
        'mean_2': mean2,
        't_statistic': t_stat,
        'p_value': p_value,
        'lift_pct': lift_pct,
        'significant': 'Yes' if p_value < 0.05 else 'No'
    }


def run_ab_test_suite(
    df: pd.DataFrame,
    cohorts: List[str],
    metrics: List[str],
    cohort_col: str = 'cohort'
) -> pd.DataFrame:
    """
    Run pairwise t-tests for all cohort combinations and metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    cohorts : list of str
        List of cohort names to compare
    metrics : list of str
        List of metric columns to test
    cohort_col : str, default='cohort'
        Name of cohort column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with test results for all combinations
    """
    results = []
    
    # Generate all pairwise combinations
    for i in range(len(cohorts)):
        for j in range(i + 1, len(cohorts)):
            cohort1, cohort2 = cohorts[i], cohorts[j]
            for metric in metrics:
                result = perform_ttest(df, cohort1, cohort2, metric, cohort_col)
                results.append(result)
    
    return pd.DataFrame(results)


def calculate_cohort_statistics(
    df: pd.DataFrame,
    metrics: List[str],
    cohort_col: str = 'cohort'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Calculate summary statistics for each cohort.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    metrics : list of str
        List of metric columns
    cohort_col : str, default='cohort'
        Name of cohort column
        
    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.Series)
        - Cohort means
        - Cohort standard deviations
        - Cohort sizes
    """
    cohort_means = df.groupby(cohort_col)[metrics].mean()
    cohort_stds = df.groupby(cohort_col)[metrics].std()
    cohort_counts = df.groupby(cohort_col).size()
    
    return cohort_means, cohort_stds, cohort_counts


def plot_cohort_comparison(
    cohort_means: pd.DataFrame,
    metrics: List[str],
    figsize: Tuple[int, int] = (16, 10),
    colors: Optional[List[str]] = None
) -> plt.Figure:
    """
    Create bar charts comparing cohorts across metrics.
    
    Parameters
    ----------
    cohort_means : pd.DataFrame
        DataFrame with cohort means (cohorts as index, metrics as columns)
    metrics : list of str
        List of metrics to plot
    figsize : tuple, default=(16, 10)
        Figure size
    colors : list of str, optional
        List of colors for cohorts
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    if colors is None:
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        cohort_means[metric].plot(kind='bar', ax=ax, color=colors[:len(cohort_means)])
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Cohort', fontsize=10)
        ax.set_ylabel('Mean Value', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    # Remove extra subplots
    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    return fig


def create_ab_test_summary(
    results_df: pd.DataFrame,
    control_cohort: str,
    treatment_cohorts: List[str]
) -> pd.DataFrame:
    """
    Create summary table comparing treatment cohorts to control.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with all test results from run_ab_test_suite
    control_cohort : str
        Name of control cohort
    treatment_cohorts : list of str
        List of treatment cohort names
        
    Returns
    -------
    pd.DataFrame
        Summary table with lifts and p-values vs control
    """
    summary_data = []
    
    metrics = results_df['metric'].unique()
    
    for metric in metrics:
        row = {'Metric': metric}
        
        # Get control mean
        control_result = results_df[
            (results_df['metric'] == metric) & 
            (results_df['cohort_1'] == control_cohort)
        ].iloc[0] if len(results_df[
            (results_df['metric'] == metric) & 
            (results_df['cohort_1'] == control_cohort)
        ]) > 0 else None
        
        if control_result is not None:
            row['Control Mean'] = control_result['mean_1']
        
        # Get treatment results
        for treatment in treatment_cohorts:
            treatment_result = results_df[
                (results_df['metric'] == metric) & 
                (results_df['cohort_1'] == control_cohort) &
                (results_df['cohort_2'] == treatment)
            ]
            
            if len(treatment_result) > 0:
                treatment_result = treatment_result.iloc[0]
                row[f'{treatment} Lift %'] = treatment_result['lift_pct']
                row[f'{treatment} p-value'] = treatment_result['p_value']
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)


def calculate_statistical_power(
    effect_size: float,
    sample_size: int,
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> float:
    """
    Calculate statistical power for a t-test.
    
    Parameters
    ----------
    effect_size : float
        Cohen's d effect size
    sample_size : int
        Sample size per group
    alpha : float, default=0.05
        Significance level
    alternative : str, default='two-sided'
        Alternative hypothesis ('two-sided', 'larger', 'smaller')
        
    Returns
    -------
    float
        Statistical power (0-1)
    """
    from statsmodels.stats.power import ttest_power
    
    power = ttest_power(
        effect_size=effect_size,
        nobs=sample_size,
        alpha=alpha,
        alternative=alternative
    )
    
    return power
