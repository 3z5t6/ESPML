#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Drift Detection Module

This module provides various methods for detecting changes in data distribution, including:
- Chi-squared test
- T-test
- KL divergence
- Cluster analysis
- DDM (Drift Detection Method)
- EDDM (Early Drift Detection Method)

These methods can be used to monitor changes in data distribution and timely detect data drift issues.
"""

import os
import glob
from typing import List, Dict, Union, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score

from module.utils.log import get_logger

# Import prediction functions from appropriate module
# These functions are used in the drift detection methods but are not defined in this file
from module.utils.test_file import predict, predict_proba

# Configure logger
logger = get_logger(__name__)

def drift_detect(history: List[str], new: List[str], config: Optional[Dict[str, Any]] = None, 
                threshold: float = 0.05, method: str = 'ddm', mode: str = 'local_dir') -> bool:
    """
    Detect data drift
    
    Detects distribution drift between historical data and new data using the specified method
    
    Args:
        history: List of historical data file paths
        new: List of new data file paths
        config: Configuration parameter dictionary containing feature information, etc.
        threshold: Drift detection threshold, default is 0.05
        method: Drift detection method, options are 'chi', 'ttest', 'kl', 'Clustering', 'ddm', 'eddm', default is 'ddm'
        mode: Data reading mode, currently only supports 'local_dir' (read from local files)
        
    Returns:
        Whether data drift is detected
    """
    # If there is no historical data, assume drift exists
    if not history:
        logger.warning("No historical data, assuming data drift by default")
        return True
    
    # Read data
    if mode == 'local_dir':
        logger.info(f"Reading data from local files, number of historical data files: {len(history)}, number of new data files: {len(new)}")
        try:
            history_data = pd.concat((pd.read_csv(f) for f in history), ignore_index=True)
            new_data = pd.concat((pd.read_csv(f) for f in new), ignore_index=True)
        except Exception as e:
            logger.error(f"Failed to read data files: {e}")
            return False
    else:
        logger.error(f"Unsupported data reading mode: {mode}")
        return False
    
    # Detect drift according to the specified method
    if method == 'chi':
        logger.info("Using chi-squared test to detect data drift")
        return chi_squared_test(history_data, new_data, config['Feature']['CategoricalFeature'], threshold)
    elif method == 'ttest':
        logger.info("Using T-test to detect data drift")
        num_features = [feature for feature in config['Feature']['FeatureName'] 
                       if feature not in config['Feature']['CategoricalFeature']]
        return t_test(history_data, new_data, num_features, threshold)
    elif method == 'kl':
        logger.info("Using KL divergence to detect data drift")
        return kl_divergence(history_data, new_data, config['Feature']['FeatureName'], threshold)
    elif method == 'Clustering':
        logger.info("Using cluster analysis to detect data drift")
        return cluster_analysis(history_data, new_data, config['Feature']['FeatureName'])
    elif method == 'ddm':
        logger.info("Using DDM method to detect data drift")
        return ddm_test(new_data, config, config['Feature']['TaskType'], 
                       config['Feature']['TargetName'], threshold)
    elif method == 'eddm':
        logger.info("Using EDDM method to detect data drift")
        return eddm_test(history_data, new_data, config, config['Feature']['TaskType'], 
                        config['Feature']['TargetName'], threshold)
    else:
        logger.error(f"Unsupported drift detection method: {method}")
        raise ValueError(f"Unsupported drift detection method: {method}")


# Fix function name spelling error
dirft_detect = drift_detect  # For backward compatibility

def chi_squared_test(history: pd.DataFrame, new: pd.DataFrame, columns: List[str], 
                    threshold: float = 0.05) -> bool:
    """
    Chi-squared test
    
    Use chi-squared test to detect distribution changes in categorical features
    
    Args:
        history: Historical DataFrame
        new: New DataFrame
        columns: List of categorical features to detect
        threshold: Significance threshold, default is 0.05
        
    Returns:
        Whether data drift is detected (drift is considered to exist when p-value is greater than or equal to the threshold)
    """
    if not columns:
        logger.warning("No categorical features provided for chi-squared test")
        return False
    
    drift_detected = False
    for column in columns:
        try:
            # Check if column exists in both dataframes
            if column not in history.columns or column not in new.columns:
                logger.warning(f"Column {column} does not exist in the DataFrame, skipping chi-squared test for this column")
                continue
                
            # Create contingency table
            contingency_table = pd.crosstab(history[column], new[column])
            
            # Calculate chi-square test statistic
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            
            # Determine if drift is detected
            if p >= threshold:
                logger.info(f"Column {column} detected data drift,p值 = {p:.4f}")
                drift_detected = True
            else:
                logger.info(f"Column {column} did not detect data drift,p值 = {p:.4f}")
                
        except Exception as e:
            logger.error(f"Error during chi-squared test for column {column}: {e}")
    
    return drift_detected


def t_test(history: pd.DataFrame, new: pd.DataFrame, columns: List[str], 
           threshold: float = 0.05) -> bool:
    """
    T-test
    
    Use T-test to detect distribution changes in numerical features
    
    Args:
        history: Historical DataFrame
        new: New DataFrame
        columns: List of numerical features to detect
        threshold: Significance threshold, default is 0.05
        
    Returns:
        Whether data drift is detected (drift is considered to exist when p-value is greater than or equal to the threshold)
    """
    if not columns:
        logger.warning("No numerical features provided for T-test")
        return False
    
    drift_detected = False
    for column in columns:
        try:
            # Check if column exists in both dataframes
            if column not in history.columns or column not in new.columns:
                logger.warning(f"Column {column} does not exist in the DataFrame, skipping T-test for this column")
                continue
                
            # Calculate T-test statistic (not assuming equal variance)
            stat, p = ttest_ind(history[column], new[column], equal_var=False)
            
            # Determine if drift is detected
            if p >= threshold:
                logger.info(f"Column {column} detected data drift,p值 = {p:.4f}")
                drift_detected = True
            else:
                logger.info(f"Column {column} did not detect data drift,p值 = {p:.4f}")
                
        except Exception as e:
            logger.error(f"Error during T-test for column {column}: {e}")
    
    return drift_detected


def kl_divergence(history: pd.DataFrame, new: pd.DataFrame, columns: List[str], 
                  threshold: float = 0.05) -> bool:
    """
    KL Divergence Test
    
    Use Jensen-Shannon divergence to detect distribution changes in features
    
    Args:
        history: Historical DataFrame
        new: New DataFrame
        columns: List of features to detect
        threshold: Divergence threshold, default is 0.05
        
    Returns:
        Whether data drift is detected (drift is considered to exist when divergence is greater than 0)
    """
    if not columns:
        logger.warning("No features provided for KL divergence test")
        return False
    
    drift_detected = False
    for column in columns:
        try:
            # Check if column exists in both dataframes
            if column not in history.columns or column not in new.columns:
                logger.warning(f"Column {column} does not exist in the DataFrame, skipping KL divergence test for this column")
                continue
                
            # Calculate histogram
            hist_hist, _ = np.histogram(history[column], bins=50, density=True)
            hist_new, _ = np.histogram(new[column], bins=50, density=True)
            
            # Calculate Jensen-Shannon divergence
            js_distance = jensenshannon(hist_hist, hist_new, base=2)
            
            # Determine if drift is detected
            if js_distance > 0:
                logger.info(f"Column {column} detected data drift, Jensen-Shannon divergence = {js_distance:.4f}")
                drift_detected = True
            else:
                logger.info(f"Column {column} did not detect data drift, Jensen-Shannon divergence = {js_distance:.4f}")
                
        except Exception as e:
            logger.error(f"Error during KL divergence test for column {column}: {e}")
    
    return drift_detected


def cluster_analysis(history: pd.DataFrame, new: pd.DataFrame, columns: List[str]) -> bool:
    """
    Cluster Analysis
    
    Use K-means clustering to detect distribution changes in text features
    
    Args:
        history: Historical DataFrame
        new: New DataFrame
        columns: List of features to detect
        
    Returns:
        Whether data drift is detected (drift is considered to exist when clustering results differ)
    """
    if not columns:
        logger.warning("No features provided for cluster analysis")
        return False
    
    drift_detected = False
    for column in columns:
        try:
            # Check if column exists in both dataframes
            if column not in history.columns or column not in new.columns:
                logger.warning(f"Column {column} does not exist in the DataFrame, skipping cluster analysis for this column")
                continue
                
            # Convert text to TF-IDF features
            vectorizer = TfidfVectorizer()
            X_hist = vectorizer.fit_transform(history[column].astype(str))
            X_new = vectorizer.transform(new[column].astype(str))
            
            # Perform K-means clustering on historical and new data
            kmeans_hist = KMeans(n_clusters=min(3, X_hist.shape[0]), random_state=42).fit(X_hist)
            kmeans_new = KMeans(n_clusters=min(3, X_new.shape[0]), random_state=42).fit(X_new)
            
            # Determine if clustering results are consistent
            if not np.array_equal(kmeans_hist.labels_, kmeans_new.labels_):
                logger.info(f"Column {column} detected data drift, clustering results are inconsistent")
                drift_detected = True
            else:
                logger.info(f"Column {column} did not detect data drift, clustering results are consistent")
                
        except Exception as e:
            logger.error(f"Error during cluster analysis for column {column}: {e}")
    
    return drift_detected

def ddm_test(new: pd.DataFrame, config: Dict[str, Any], task_type: str, 
            target_name: str, threshold: float = 0.85) -> bool:
    """
    DDM (Drift Detection Method) Test
    
    Detect data drift based on model performance on new data
    
    Args:
        new: New DataFrame
        config: Configuration parameter dictionary
        task_type: Task type, 'classification' or 'regression'
        target_name: Target variable name
        threshold: Performance threshold, default is 0.85
        
    Returns:
        Whether data drift is detected (drift is considered to exist when model performance is below the threshold)
    """
    try:
        # Predict new data
        pred = predict(config, new, save=False)

        # 分类任务处理
        if task_type == 'classification':
            # Calculate classification accuracy
            accuracy = accuracy_score(new[target_name], pred)
            logger.info(f"DDM test: Classification accuracy = {accuracy:.4f}, threshold = {threshold:.4f}")
            
            # Determine if drift is detected
            if accuracy < threshold:
                logger.info(f"DDM test detected data drift: Accuracy below threshold")
                return True
            else:
                logger.info(f"DDM test did not detect data drift: Accuracy above or equal to threshold")
                return False

        # 回归任务处理
        elif task_type == 'regression':
            # Calculate R-squared value
            r2 = r2_score(new[target_name], pred)
            logger.info(f"DDM test: R-squared value = {r2:.4f}, threshold = {threshold:.4f}")
            
            # Determine if drift is detected
            if r2 < threshold:
                logger.info(f"DDM test detected data drift: R-squared value below threshold")
                return True
            else:
                logger.info(f"DDM test did not detect data drift: R-squared value above or equal to threshold")
                return False
        else:
            logger.error(f"不支持的任务类型: {task_type}")
            raise ValueError(f"不支持的任务类型: {task_type}")

    except Exception as e:
        logger.error(f"DDM测试时出错: {e}")
        return False


def eddm_test(history: pd.DataFrame, new: pd.DataFrame, config: Dict[str, Any], 
              task_type: str, target_name: str, threshold: float = 0.85) -> bool:
    """
    EDDM (Early Drift Detection Method) Test
    
    Detect data drift based on performance difference between model on historical data and new data
    
    Args:
        history: Historical DataFrame
        new: New DataFrame
        config: Configuration parameter dictionary
        task_type: Task type, 'classification' or 'regression'
        target_name: Target variable name
        threshold: Performance threshold, default is 0.85
        
    Returns:
        Whether data drift is detected (drift is considered to exist when new data performance is lower than historical data)
    """
    try:
        # First perform DDM test
        if not ddm_test(new, config, task_type, target_name, threshold):
            # DDM test did not detect drift, continue with EDDM test
            logger.info("Performing EDDM test, comparing model performance on historical and new data")
            
            # Predict historical and new data
            history_pred = predict(config, history, save=False)
            new_pred = predict(config, new, save=False)
            
            # Process classification task
            if task_type == 'classification':
                # Calculate accuracy for historical and new data
                history_accuracy = accuracy_score(history[target_name], history_pred)
                new_accuracy = accuracy_score(new[target_name], new_pred)
                
                logger.info(f"EDDM test: Historical data accuracy = {history_accuracy:.4f}, "
                          f"new data accuracy = {new_accuracy:.4f}")
                
                # Determine if drift is detected
                if new_accuracy < history_accuracy:
                    logger.info(f"EDDM test detected data drift: New data accuracy lower than historical data")
                    return True
                else:
                    logger.info(f"EDDM test did not detect data drift: New data accuracy higher than or equal to historical data")
                    return False
                    
            # Process regression task
            elif task_type == 'regression':
                # Calculate R-squared value for historical and new data
                history_r2 = r2_score(history[target_name], history_pred)
                new_r2 = r2_score(new[target_name], new_pred)
                
                logger.info(f"EDDM test: Historical data R-squared value = {history_r2:.4f}, "
                          f"new data R-squared value = {new_r2:.4f}")
                
                # Determine if drift is detected
                if new_r2 < history_r2:
                    logger.info(f"EDDM test detected data drift: New data R-squared value lower than historical data")
                    return True
                else:
                    logger.info(f"EDDM test did not detect data drift: New data R-squared value higher than or equal to historical data")
                    return False
            else:
                logger.error(f"Unsupported task type: {task_type}")
                raise ValueError(f"Unsupported task type: {task_type}")
        else:
            # DDM test already detected drift, no need for EDDM test
            logger.info("DDM test already detected data drift, no need for EDDM test")
            return True
            
    except Exception as e:
        logger.error(f"EDDM test failed: {e}")
        return False

class DriftDetecter:
    """
    Data Drift Detector
    
    Used to monitor changes in data files in a specified directory and detect data drift
    """

    def __init__(self, directory: str):
        """
        Initialize data drift detector
        
        Args:
            directory: Path to the data directory to monitor
        """
        self.directory = directory
        self.history_data_files = glob.glob(os.path.join(self.directory, '*'))
        logger.info(f"Initializing data drift detector, monitoring directory: {directory}")
        logger.info(f"Current directory contains {len(self.history_data_files)} data files")

    def check_data_change(self) -> bool:
        """
        Check if files in the data directory have changed
        
        Returns:
            Whether file changes are detected
        """
        try:
            # Get all files in the current directory
            current_data_files = glob.glob(os.path.join(self.directory, '*'))
            
            # Check if the file set has changed
            if set(self.history_data_files) != set(current_data_files):
                logger.info(f"Detected data file changes, original file count: {len(self.history_data_files)}, "
                          f"new file count: {len(current_data_files)}")
                
                # Update the history file list
                self.history_data_files = current_data_files
                return True
            else:
                logger.info("No data file changes detected")
                return False
                
        except Exception as e:
            logger.error(f"Error checking data changes: {e}")
            return False

    def history_sampling(self) -> pd.DataFrame:
        """
        Sample historical data
        
        Returns:
            Sampled historical DataFrame
        """
        try:
            if not self.history_data_files:
                logger.warning("No historical data files available")
                return pd.DataFrame()
                
            # Read and merge all historical data files
            logger.info(f"Reading {len(self.history_data_files)} historical data files")
            history_data = pd.concat(
                (pd.read_csv(f) for f in self.history_data_files), 
                ignore_index=True
            )
            
            # Sampling logic can be added here, currently returns all data
            logger.info(f"Historical data reading completed, total {len(history_data)} rows")
            return history_data
            
        except Exception as e:
            logger.error(f"Error during historical data sampling: {e}")
            return pd.DataFrame()

class DataDistributor:
    """
    Data Distribution Detector
    
    Used to detect distribution differences between historical and new data to determine if data drift exists
    """

    def __init__(self, historical_files: List[str], new_files: List[str], config: Dict[str, Any]):
        """
        Initialize data distribution detector
        
        Args:
            historical_files: List of historical data file paths
            new_files: List of new data file paths (should include historical data files)
            config: Configuration parameter dictionary
        """
        logger.info(f"Initializing data distribution detector, historical file count: {len(historical_files)}, "
                  f"new file count: {len(new_files)}")
        
        # Verify that new data contains all historical data
        if not set(historical_files).issubset(set(new_files)):
            error_msg = 'New data does not contain all historical data, please check if historical data has been modified!'
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Load data
        self.historical_data = self.load_data(historical_files)
        self.new_data = self.load_data(new_files)
        self.config = config
        
        logger.info(f"Data loading completed, historical data: {len(self.historical_data)} rows, "
                  f"new data: {len(self.new_data)} rows")

    def detect(self, method: str, historical_files: Optional[List[str]] = None, 
              new_files: Optional[List[str]] = None, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Detect data drift
        
        Args:
            method: Drift detection method, options are 'chi', 'ttest', 'kl', 'Clustering', 'ddm', 'eddm'
            historical_files: List of historical data file paths, if None uses data from initialization
            new_files: List of new data file paths, if None uses data from initialization
            config: Configuration parameter dictionary, if None uses configuration from initialization
        
        Returns:
            Whether data drift is detected
        """
        try:
            # If new file lists are provided, reload data
            if historical_files is not None and new_files is not None:
                logger.info("Loading data using new file lists")
                
                # Verify that new data includes historical data
                if not set(historical_files).issubset(set(new_files)):
                    error_msg = 'New data does not fully include historical data, please check if historical data has been modified!'
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Load historical data
                historical_data = self.load_data(historical_files)
                
                # Only load newly added data files
                new_only_files = [file for file in new_files if file not in historical_files]
                new_data = self.load_data(new_only_files)
            else:
                logger.info("Using data from initialization")
                historical_data = self.historical_data
                new_data = self.new_data
                
            # Use specified configuration or default configuration
            cfg = config if config is not None else self.config
            
            # Detect drift according to the specified method
            logger.info(f"Using {method} method to detect data drift")
            
            # Here should implement the calling logic for various methods
            # Currently only returns default value, need to implement complete detection logic
            return False
            
        except Exception as e:
            logger.error(f"Error during data drift detection: {e}")
            return False

    def load_data(self, file_list: List[str]) -> pd.DataFrame:
        """
        Load data files
        
        Args:
            file_list: List of data file paths
            
        Returns:
            Merged DataFrame
        """
        try:
            if not file_list:
                logger.warning("No data files provided")
                return pd.DataFrame()
                
            logger.info(f"Reading {len(file_list)} data files")
            data = pd.concat((pd.read_csv(f) for f in file_list), ignore_index=True)
            logger.info(f"Data reading completed, total {len(data)} rows")
            return data
            
        except Exception as e:
            logger.error(f"Error reading data files: {e}")
            return pd.DataFrame()

    def chi_squared_test(self, column: Union[str, List[str]], threshold: float = 0.05) -> bool:
        """
        Chi-squared test
        
        Use chi-squared test to detect distribution changes in categorical features
        
        Args:
            column: Name or list of categorical features to detect
            threshold: Significance threshold, default is 0.05
            
        Returns:
            Whether data drift is detected (drift is considered to exist when p-value is less than threshold)
        """
        try:
            # Handle single column or list of columns
            columns = [column] if isinstance(column, str) else column
            
            drift_detected = False
            for col in columns:
                # Check if column exists in both dataframes
                if col not in self.historical_data.columns or col not in self.new_data.columns:
                    logger.warning(f"Column {col} does not exist in the DataFrame, skipping chi-squared test for this column")
                    continue
                    
                # Check if column is categorical type
                if not pd.api.types.is_categorical_dtype(self.historical_data[col]):
                    logger.warning(f"Column {col} is not categorical type, skipping chi-squared test for this column")
                    continue
                
                # Create contingency table
                contingency_table = pd.crosstab(self.historical_data[col], self.new_data[col])
                
                # Calculate chi-squared test statistic
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                
                # Determine if drift is detected
                if p < threshold:
                    logger.info(f"Column {col} detected data drift, p-value = {p:.4f}, threshold = {threshold:.4f}")
                    drift_detected = True
                else:
                    logger.info(f"Column {col} did not detect data drift, p-value = {p:.4f}, threshold = {threshold:.4f}")
            
            return drift_detected
            
        except Exception as e:
            logger.error(f"Error during chi-squared test: {e}")
            return False

    def t_test(self, column: Union[str, List[str]], threshold: float = 0.05) -> bool:
        """
        T-test
        
        Use T-test to detect distribution changes in numerical features
        
        Args:
            column: Name or list of numerical features to detect
            threshold: Significance threshold, default is 0.05
            
        Returns:
            Whether data drift is detected (drift is considered to exist when p-value is less than threshold)
        """
        try:
            # Handle single column or list of columns
            columns = [column] if isinstance(column, str) else column
            
            drift_detected = False
            for col in columns:
                # Check if column exists in both dataframes
                if col not in self.historical_data.columns or col not in self.new_data.columns:
                    logger.warning(f"Column {col} does not exist in the DataFrame, skipping T-test for this column")
                    continue
                    
                # Check if column is numerical type
                if pd.api.types.is_categorical_dtype(self.historical_data[col]):
                    logger.warning(f"Column {col} is categorical type, skipping T-test for this column")
                    continue
                
                # Calculate T-test statistic (not assuming equal variance)
                stat, p = ttest_ind(self.historical_data[col], self.new_data[col], equal_var=False)
                
                # Determine if drift is detected
                if p < threshold:
                    logger.info(f"Column {col} detected data drift, p-value = {p:.4f}, threshold = {threshold:.4f}")
                    drift_detected = True
                else:
                    logger.info(f"Column {col} did not detect data drift, p-value = {p:.4f}, threshold = {threshold:.4f}")
            
            return drift_detected
            
        except Exception as e:
            logger.error(f"Error during T-test: {e}")
            return False

    def kl_divergence(self, column: Union[str, List[str]], threshold: float = 0.05) -> bool:
        """
        KL Divergence Test
        
        Use Jensen-Shannon divergence to detect distribution changes in features
        
        Args:
            column: Name or list of features to detect
            threshold: Divergence threshold, default is 0.05
            
        Returns:
            Whether data drift is detected (drift is considered to exist when divergence is greater than threshold)
        """
        try:
            # Handle single column or list of columns
            columns = [column] if isinstance(column, str) else column
            
            drift_detected = False
            for col in columns:
                # Check if column exists in both dataframes
                if col not in self.historical_data.columns or col not in self.new_data.columns:
                    logger.warning(f"Column {col} does not exist in the DataFrame, skipping KL divergence test for this column")
                    continue
                
                # Calculate histogram
                hist_hist, _ = np.histogram(self.historical_data[col], bins=50, density=True)
                hist_new, _ = np.histogram(self.new_data[col], bins=50, density=True)
                
                # Calculate Jensen-Shannon divergence
                js_distance = jensenshannon(hist_hist, hist_new, base=2)
                
                # Determine if drift is detected
                if js_distance > threshold:
                    logger.info(f"Column {col} detected data drift, Jensen-Shannon divergence = {js_distance:.4f}, threshold = {threshold:.4f}")
                    drift_detected = True
                else:
                    logger.info(f"Column {col} did not detect data drift, Jensen-Shannon divergence = {js_distance:.4f}, threshold = {threshold:.4f}")
            
            return drift_detected
            
        except Exception as e:
            logger.error(f"Error during KL divergence test: {e}")
            return False

    def cluster_analysis(self, text_column: Union[str, List[str]]) -> bool:
        """
        Cluster Analysis
        
        Use K-means clustering analysis to detect distribution changes in text features
        
        Args:
            text_column: Name or list of text features to detect
            
        Returns:
            Whether data drift is detected (drift is considered to exist when clustering results differ)
        """
        try:
            # Handle single column or list of columns
            columns = [text_column] if isinstance(text_column, str) else text_column
            
            drift_detected = False
            for col in columns:
                # Check if column exists in both dataframes
                if col not in self.historical_data.columns or col not in self.new_data.columns:
                    logger.warning(f"Column {col} does not exist in the dataframe, skipping cluster analysis for this column")
                    continue
                
                # Convert text to TF-IDF features
                vectorizer = TfidfVectorizer()
                X_hist = vectorizer.fit_transform(self.historical_data[col].astype(str))
                X_new = vectorizer.transform(self.new_data[col].astype(str))
                
                # Perform K-means clustering on historical and new data
                n_clusters = min(3, X_hist.shape[0], X_new.shape[0])
                kmeans_hist = KMeans(n_clusters=n_clusters, random_state=42).fit(X_hist)
                kmeans_new = KMeans(n_clusters=n_clusters, random_state=42).fit(X_new)
                
                # Determine if clustering results are consistent
                if not np.array_equal(kmeans_hist.labels_, kmeans_new.labels_):
                    logger.info(f"Column {col} detected data drift, clustering results are inconsistent")
                    drift_detected = True
                else:
                    logger.info(f"Column {col} did not detect data drift, clustering results are consistent")
            
            return drift_detected
            
        except Exception as e:
            logger.error(f"Error during cluster analysis: {e}")
            return False

    def ddm_test(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        DDM (Drift Detection Method) Test
        
        Detect data drift based on model performance on new data
        
        Args:
            config: Configuration parameters dictionary, if None then use the configuration at initialization
            
        Returns:
            Whether data drift is detected
        """
        try:
            # Use specified configuration or default configuration
            cfg = config if config is not None else self.config
            
            # Get task type and target variable name
            task_type = cfg['Feature']['TaskType']
            target_name = cfg['Feature']['TargetName']
            threshold = 0.85  # Default threshold
            
            # Call DDM test function
            logger.info("Calling DDM test function for data drift detection")
            return ddm_test(self.new_data, cfg, task_type, target_name, threshold)
            
        except Exception as e:
            logger.error(f"Error during DDM test: {e}")
            return False

    def eddm_test(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        EDDM (Early Drift Detection Method) Test
        
        Detect data drift based on model performance differences between historical and new data
        
        Args:
            config: Configuration parameters dictionary, if None then use the configuration at initialization
            
        Returns:
            Whether data drift is detected
        """
        try:
            # Use specified configuration or default configuration
            cfg = config if config is not None else self.config
            
            # Get task type and target variable name
            task_type = cfg['Feature']['TaskType']
            target_name = cfg['Feature']['TargetName']
            threshold = 0.85  # Default threshold
            
            # Call EDDM test function
            logger.info("Calling EDDM test function for data drift detection")
            return eddm_test(self.historical_data, self.new_data, cfg, task_type, target_name, threshold)
            
        except Exception as e:
            logger.error(f"Error during EDDM test: {e}")
            return False