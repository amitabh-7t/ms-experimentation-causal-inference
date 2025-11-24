"""
Uplift Modeling Utilities

Functions for uplift-based user segmentation, segment profiling,
and business strategy recommendations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def create_uplift_segments(
    uplift: np.ndarray,
    n_segments: int = 5,
    labels: Optional[List[str]] = None
) -> np.ndarray:
    """
    Create uplift-based segments using quantile bucketing.
    
    Parameters
    ----------
    uplift : np.ndarray
        Array of uplift scores (CATE estimates)
    n_segments : int, default=5
        Number of segments to create
    labels : list of str, optional
        Custom labels for segments
        Default: ['Very Low', 'Low', 'Medium', 'High', 'Very High'] for 5 segments
        
    Returns
    -------
    np.ndarray
        Segment labels for each observation
    """
    if labels is None:
        if n_segments == 5:
            labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        elif n_segments == 3:
            labels = ['Low', 'Medium', 'High']
        else:
            labels = [f'Segment_{i+1}' for i in range(n_segments)]
    
    segments = pd.qcut(uplift, q=n_segments, labels=labels, duplicates='drop')
    
    return segments


def calculate_segment_profiles(
    df: pd.DataFrame,
    segment_col: str,
    metrics: List[str],
    segment_order: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate aggregated metrics for each segment.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with segment assignments and metrics
    segment_col : str
        Name of segment column
    metrics : list of str
        List of metric columns to aggregate
    segment_order : list of str, optional
        Desired order of segments in output
        
    Returns
    -------
    pd.DataFrame
        Segment profiles with mean values for each metric
    """
    profiles = df.groupby(segment_col)[metrics].mean().round(2)
    
    if segment_order:
        profiles = profiles.reindex(segment_order)
    
    return profiles


def plot_segment_profiles(
    segment_profiles: pd.DataFrame,
    metrics: List[str],
    figsize: Tuple[int, int] = (16, 10),
    colors: Optional[List[str]] = None
) -> plt.Figure:
    """
    Create bar charts showing segment profiles across metrics.
    
    Parameters
    ----------
    segment_profiles : pd.DataFrame
        DataFrame with segments as index and metrics as columns
    metrics : list of str
        List of metrics to plot
    figsize : tuple, default=(16, 10)
        Figure size
    colors : list of str, optional
        Colors for segments
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    if colors is None:
        colors = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#27ae60']
    
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        segment_profiles[metric].plot(kind='bar', ax=ax, color=colors[:len(segment_profiles)])
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Uplift Segment', fontsize=10)
        ax.set_ylabel('Mean Value', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    # Remove extra subplots
    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    return fig


def plot_segment_heatmap(
    segment_profiles: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 6),
    cmap: str = 'RdYlGn'
) -> plt.Figure:
    """
    Create heatmap of standardized segment characteristics.
    
    Parameters
    ----------
    segment_profiles : pd.DataFrame
        DataFrame with segments as index and metrics as columns
    figsize : tuple, default=(10, 6)
        Figure size
    cmap : str, default='RdYlGn'
        Colormap name
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    # Standardize metrics for better visualization
    scaler = StandardScaler()
    profiles_normalized = pd.DataFrame(
        scaler.fit_transform(segment_profiles),
        index=segment_profiles.index,
        columns=segment_profiles.columns
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(profiles_normalized.T, annot=True, fmt='.2f', 
                cmap=cmap, center=0, cbar_kws={'label': 'Standardized Value'},
                ax=ax)
    ax.set_title('Segment Characteristics Heatmap (Standardized)', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Uplift Segment', fontsize=11)
    ax.set_ylabel('Metric', fontsize=11)
    
    plt.tight_layout()
    return fig


def get_segment_strategies() -> Dict[str, Dict[str, str]]:
    """
    Get business strategy recommendations for each uplift segment.
    
    Returns
    -------
    dict
        Dictionary mapping segment names to strategy details
    """
    strategies = {
        'Very Low': {
            'profile': 'Low baseline productivity, minimal AI response',
            'strategy': 'Deprioritize AI rollout. Focus on basic product improvements and alternative engagement.',
            'resource_allocation': '5%',
            'actions': [
                'Focus on core product improvements',
                'Provide basic onboarding and support',
                'Monitor for churn with non-AI retention tactics'
            ]
        },
        'Low': {
            'profile': 'Moderate baseline, limited AI response',
            'strategy': 'Gradual rollout with education and training.',
            'resource_allocation': '10%',
            'actions': [
                'Provide guided AI feature adoption (tooltips, tutorials)',
                'A/B test different AI UX patterns',
                'Offer educational content'
            ]
        },
        'Medium': {
            'profile': 'Average across dimensions, moderate uplift',
            'strategy': 'Standard rollout with basic support.',
            'resource_allocation': '15%',
            'actions': [
                'Standard feature rollout',
                'Monitor usage patterns for sub-segmentation',
                'Identify upsell opportunities'
            ]
        },
        'High': {
            'profile': 'Higher productivity, strong AI response',
            'strategy': 'Priority rollout of new AI features.',
            'resource_allocation': '30%',
            'actions': [
                'Invest in advanced AI capabilities',
                'Use as beta testers for new features',
                'Leverage for case studies and marketing'
            ]
        },
        'Very High': {
            'profile': 'Highest productivity, maximum uplift',
            'strategy': 'Immediate comprehensive rollout with white-glove service.',
            'resource_allocation': '50%',
            'actions': [
                'White-glove onboarding for AI features',
                'Offer premium/enterprise AI tiers',
                'Co-development partnerships',
                'Use as advocates and references'
            ]
        }
    }
    
    return strategies


def print_segment_recommendations(
    segment_profiles: pd.DataFrame,
    strategies: Optional[Dict] = None
) -> None:
    """
    Print formatted segment recommendations.
    
    Parameters
    ----------
    segment_profiles : pd.DataFrame
        DataFrame with segment profiles
    strategies : dict, optional
        Custom strategies dictionary
    """
    if strategies is None:
        strategies = get_segment_strategies()
    
    print("=" * 80)
    print("SEGMENT-SPECIFIC BUSINESS STRATEGIES")
    print("=" * 80)
    
    for segment in segment_profiles.index:
        if segment in strategies:
            strategy = strategies[segment]
            print(f"\n{'🔴' if segment == 'Very Low' else '🟠' if segment == 'Low' else '🟡' if segment == 'Medium' else '🟢'} {segment.upper()} UPLIFT SEGMENT")
            print("-" * 80)
            print(f"Profile: {strategy['profile']}")
            print(f"Strategy: {strategy['strategy']}")
            print(f"Resource Allocation: {strategy['resource_allocation']}")
            print(f"Key Actions:")
            for action in strategy['actions']:
                print(f"  • {action}")


def calculate_segment_sizes(
    segments: np.ndarray
) -> pd.Series:
    """
    Calculate size of each segment.
    
    Parameters
    ----------
    segments : np.ndarray
        Array of segment labels
        
    Returns
    -------
    pd.Series
        Segment sizes
    """
    return pd.Series(segments).value_counts().sort_index()


def identify_target_segments(
    segment_profiles: pd.DataFrame,
    uplift_col: str,
    threshold_percentile: float = 60
) -> List[str]:
    """
    Identify segments to target based on uplift threshold.
    
    Parameters
    ----------
    segment_profiles : pd.DataFrame
        DataFrame with segment profiles
    uplift_col : str
        Name of uplift column
    threshold_percentile : float, default=60
        Percentile threshold for targeting
        
    Returns
    -------
    list of str
        List of segment names to target
    """
    threshold = segment_profiles[uplift_col].quantile(threshold_percentile / 100)
    target_segments = segment_profiles[segment_profiles[uplift_col] >= threshold].index.tolist()
    
    return target_segments


def calculate_incremental_value(
    df: pd.DataFrame,
    segment_col: str,
    uplift_col: str,
    target_segments: List[str]
) -> Dict[str, float]:
    """
    Calculate incremental value from targeting specific segments.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with segments and uplift
    segment_col : str
        Name of segment column
    uplift_col : str
        Name of uplift column
    target_segments : list of str
        List of segments to target
        
    Returns
    -------
    dict
        Dictionary with value metrics
    """
    targeted_users = df[df[segment_col].isin(target_segments)]
    all_users = df
    
    return {
        'targeted_users': len(targeted_users),
        'total_users': len(all_users),
        'targeting_rate': len(targeted_users) / len(all_users),
        'avg_uplift_targeted': targeted_users[uplift_col].mean(),
        'avg_uplift_all': all_users[uplift_col].mean(),
        'total_incremental_value': targeted_users[uplift_col].sum()
    }
