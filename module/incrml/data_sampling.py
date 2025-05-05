"""
Incremental Machine Learning Data Sampling Module

This module provides various data sampling methods for processing large-scale datasets, including importance sampling, random sampling, uncertainty sampling, time window sampling, and uniform sampling.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Optional, Tuple, Any
from sklearn.neighbors import NearestNeighbors

from module.serve.test_file import predict, predict_proba
from module.utils.log import get_logger

logger = get_logger(__name__)


def data_sampling(
    history: Union[pd.DataFrame, List[str], str], 
    config: Optional[Dict] = None, 
    method: str = 'importance', 
    size: int = 500000
) -> pd.DataFrame:
    """
    Sample data using various sampling methods
    
    Args:
        history: Input data, can be a DataFrame, a list of CSV file paths, or a single CSV file path
        config: Configuration parameter dictionary, default is None
        method: Sampling method, options are 'importance', 'random', 'uncertainty', 'timewindow', 'uniform', default is 'importance'
        size: Sample size, default is 500000
        
    Returns:
        Sampled DataFrame
        
    Raises:
        TypeError: When the input data type does not meet requirements
        ValueError: When the sampling method is not supported
    """
    # Process input data
    if isinstance(history, pd.DataFrame):
        df = history
    else:
        # Process string type (single CSV file path)
        if isinstance(history, str):
            try:
                df = pd.read_csv(history)
            except Exception as e:
                logger.error(f"Failed to read CSV file: {e}")
                raise
        # Process list type (multiple CSV file paths)
        elif isinstance(history, list):
            if not history:  # Check if list is empty
                return pd.DataFrame([])
                
            # Check if all files are in CSV format
            if not all(f.endswith('.csv') for f in history):
                raise ValueError('Please check file format, all files must be in CSV format')
                
            # Read and merge all CSV files
            try:
                df = pd.concat([pd.read_csv(f) for f in history], ignore_index=True)
            except Exception as e:
                logger.error(f"Failed to merge CSV files: {e}")
                raise
        else:
            raise TypeError('history parameter must be a DataFrame, a list of CSV file paths, or a single CSV file path')
    
    # If data volume exceeds threshold, perform sampling
    if len(df) > 5000000:
        logger.info(f"Data exceeds 5,000,000 rows, using {method} method for sampling")
        
        # Process according to different sampling methods
        if method == 'importance':
            return importance_sampling(df, size)
            
        elif method == 'random':
            return df.sample(size, random_state=config.get('Feature', {}).get('RandomSeed', 42) if config else 42)
            
        elif method == 'uncertainty':
            if config is None:
                raise ValueError("uncertainty sampling method requires config parameter")
            return uncertainty_sampling(df, config, size)
            
        elif method == 'timewindow':
            return timewindow_sampling(df, size)
            
        elif method == 'uniform':
            if config is None:
                raise ValueError("uniform sampling method requires config parameter")
            return uniform_sampling(df, config, size)
            
        else:
            raise ValueError(f"Unsupported sampling method: {method}")
    
    # If data volume does not exceed threshold, return directly
    logger.info(f"Data contains {len(df)} rows, no sampling needed")
    return df

def uniform_sampling(df: pd.DataFrame, config: Dict, size: int = 500000) -> pd.DataFrame:
    """
    Uniform sampling method
    
    For classification tasks, samples uniformly by class; for regression tasks, divides the target variable into two parts and samples uniformly
    
    Args:
        df: Input DataFrame
        config: Configuration parameter dictionary, must contain 'Feature' key with 'TargetName', 'TaskType', and 'RandomSeed'
        size: Sample size, default is 500000
        
    Returns:
        Sampled DataFrame
    """
    # 获取配置参数
    target = config['Feature']['TargetName']
    task_type = config['Feature']['TaskType']
    seed = config['Feature']['RandomSeed']
    
    # 分类任务处理
    if task_type == 'classification':
        logger.info(f"Performing uniform sampling for classification task, sampling uniformly by class")
        # Sample uniformly by class
        sample_size = min(size, df[target].value_counts().min())
        df_sampled = df.groupby(target).apply(
            lambda x: x.sample(n=sample_size, random_state=seed)
        ).reset_index(drop=True)
        return df_sampled
    
    # 回归任务处理
    logger.info(f"Performing uniform sampling for regression task, dividing target variable into two parts")
    # Divide target variable into two parts
    df['_qcut'] = pd.qcut(df[target].rank(method='first'), q=2, labels=False)
    
    # Sample uniformly by group
    sample_size = min(size, df.groupby('_qcut').size().min())
    df_sampled = df.groupby('_qcut').apply(
        lambda x: x.sample(n=sample_size, random_state=seed)
    ).reset_index(drop=True)
    
    # Delete temporary column
    df_sampled.drop(columns=['_qcut'], inplace=True)
    return df_sampled


def timewindow_sampling(df: pd.DataFrame, size: int = 500000) -> pd.DataFrame:
    """
    Time window sampling method
    
    Select the most recent size records
    
    Args:
        df: Input DataFrame
        size: Sample size, default is 500000
        
    Returns:
        Sampled DataFrame
    """
    logger.info(f"Using time window sampling method, selecting the most recent {size} records")
    # Select the most recent size records
    return df.tail(size)


def uncertainty_sampling(df: pd.DataFrame, config: Dict, size: int = 500000) -> pd.DataFrame:
    """
    Uncertainty sampling method
    
    Select samples with the highest model prediction uncertainty
    
    Args:
        df: Input DataFrame
        config: Configuration parameter dictionary, must contain 'Feature' key with 'TargetName' and 'TaskType'
        size: Sample size, default is 500000
        
    Returns:
        Sampled DataFrame
    """
    logger.info(f"Using uncertainty sampling method, selecting {size} records with highest model prediction uncertainty")
    
    # 获取配置参数
    target = config['Feature']['TargetName']
    task_type = config['Feature']['TaskType']
    
    # Predict data
    pred = predict(config, df, save=False)
    
    # 分类任务处理
    if task_type == 'classification':
        # Get prediction probabilities
        proba = predict_proba(config, df, save=False)
        proba = proba if isinstance(proba, np.ndarray) else proba.values
        
        # Calculate uncertainty for each sample (distance from correct class probability)
        # Get true class index for each sample
        true_class_indices = df[target].values.tolist()
        # Get predicted probability for true class of each sample
        true_class_probs = np.array([proba[i, idx] for i, idx in enumerate(true_class_indices)])
        # Calculate uncertainty (1-true class probability)
        distance = 1 - true_class_probs
    else:
        # Process regression task, calculate absolute error between predicted and true values
        pred = pred if isinstance(pred, np.ndarray) else pred.values
        distance = np.abs(df[target].values - pred).reshape(-1)
    
    # Sort by uncertainty in descending order, select top size samples
    argsort_dis = np.argsort(distance)[::-1]  # Sort in descending order
    return df.iloc[argsort_dis[:size]]


def importance_sampling(df: pd.DataFrame, size: int = 500000) -> pd.DataFrame:
    """
    Importance sampling method
    
    Select the most important samples based on data density
    
    Args:
        df: Input DataFrame
        size: Sample size, default is 500000
        
    Returns:
        Sampled DataFrame
    """
    logger.info(f"Using importance sampling method, selecting {size} records based on data density")
    
    # Process non-numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        logger.warning("No numeric columns in DataFrame, will use random sampling instead")
        return df.sample(min(size, len(df)))
    
    # Calculate density for each sample
    try:
        local_densities = compute_local_density(numeric_df, min(10, len(numeric_df.columns)))
        # Select samples based on density
        selected_indices = select_samples_based_on_density(numeric_df, local_densities, min(size, len(df)))
        return df.iloc[selected_indices]
    except Exception as e:
        logger.error(f"Importance sampling failed: {e}")
        logger.warning("Will use random sampling instead")
        return df.sample(min(size, len(df)))


def compute_local_density(X: pd.DataFrame, k: int = 10) -> np.ndarray:
    """
    Calculate local density of data
    
    Use k-nearest neighbors algorithm to calculate the average distance from each sample to its k nearest neighbors
    
    Args:
        X: Input DataFrame, must contain only numeric columns
        k: Number of neighbors, default is 10
        
    Returns:
        Array of local density for each sample (average of distances)
    """
    # Ensure number of neighbors does not exceed number of samples
    k = min(k, len(X) - 1)
    if k <= 0:
        return np.zeros(len(X))
    
    # Initialize k-nearest neighbors model
    neigh = NearestNeighbors(n_neighbors=k+1)  # +1 is because the first neighbor is the sample itself
    neigh.fit(X)
    
    # Calculate distances
    distances, indices = neigh.kneighbors(X)
    
    # Ignore the first distance (distance to self is 0)
    densities = np.mean(distances[:, 1:], axis=1)
    return densities


def select_samples_based_on_density(X: pd.DataFrame, local_densities: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Select samples based on local density
    
    Select samples with the lowest local density (i.e., the sparsest areas)
    
    Args:
        X: Input DataFrame
        local_densities: Array of local densities
        num_samples: Number of samples to select
        
    Returns:
        Array of indices of selected samples
    """
    # Ensure sample count does not exceed total sample count
    num_samples = min(num_samples, len(X))
    
    # Sort by local density in descending order, select samples with lowest density (sparsest areas)
    selected_indices = np.argsort(local_densities)[:num_samples]
    return selected_indices