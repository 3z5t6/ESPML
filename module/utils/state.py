#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
State management module

This module provides the Base class for managing the state of machine learning tasks, including configuration parsing, 
data loading and preprocessing, feature engineering, and model training infrastructure.
"""

import os
import gc
import random
from glob import glob
from typing import Dict, List, Union, Optional, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from module.utils.yaml_parser import YamlParser
from module.utils.log import get_logger
from module.dataprocess.data_processor import DataProcess
from module.autofe.transform import Transform

# Get logger
logger = get_logger(__name__)

class Base:
    """
    Base state class for machine learning tasks
    
    This class provides the basic functionality needed for machine learning tasks, including configuration management,
    data loading, data preprocessing, feature engineering, and data splitting. It serves as a base class for other
    specific task classes, providing shared functionality.
    """


    def __init__(self, config_path: Union[str, Dict[str, Any]], data: Optional[pd.DataFrame] = None) -> None:
        """
        Initialize Base class
        
        Args:
            config_path: Configuration file path or configuration dictionary
            data: Optional DataFrame, if provided, use this data instead of loading from file
        
        Raises:
            ValueError: When config_path is neither a string nor a dictionary
        """
        # Save configuration path
        self.config_path = config_path
        
        # Parse configuration
        if isinstance(config_path, str):
            self.config = YamlParser.parse(config_path)
        elif isinstance(config_path, dict):
            self.config = config_path
        else:
            raise ValueError('Configuration path must be a string or dictionary')
            
        # Check and standardize column names in configuration
        self.config = self.check_config_columns(self.config)
        
        # Extract common parameters from configuration
        self.target_name = self.config['Feature']['TargetName']
        self.metric = self.config['Feature']['Metric']
        self.max_trails = self.config['AutoFE']['maxTrialNum']
        self.task_type = self.config['Feature']['TaskType']
        
        # Task information
        self.author_name = self.config['AuthorName']
        self.task_name = self.config['TaskName']
        self.job_id = self.config['JobID']
        
        # Data source configuration
        self.train_dir = self.config['DataSource']['dir']
        self.val_dir = self.config['DataSource'].get('val_dir', None)
        
        # AutoFE configuration
        self.autofe_running = self.config['AutoFE'].get('Running', True)
        self.random_ratio = float(self.config['AutoFE'].get('RandomRatio', 0))
        self.feat_imp_threshold = float(self.config['AutoFE'].get('FeatImpThreshold', 0))
        self.dfs_layer = self.config['AutoFE'].get('DFSLayers', 1)
        
        # AutoML configuration
        self.automl_running = self.config['AutoML'].get('Running', True)
        self.train_all_data = self.config['AutoML'].get('TrainAllData', False)
        self.automl_time_budget = self.config['AutoML'].get('TimeBudget', 60)
        
        # Feature and data splitting configuration
        self.test_size = float(self.config['Feature'].get('TestSize', 0.25))
        self.id_index = self.config['Feature'].get('id_index', None)
        self.time_index = self.config['Feature'].get('TimeIndex', None)
        self.group_id = self.config['Feature'].get('GroupIndex', None)
        self.ignore_features = self.config['Feature'].get('IgnoreFeature', [])
        self.n_splits = self.config['Feature'].get('NSplits', 5)
        self.split_type = self.config['Feature'].get('SplitType', 'stratified')
        self.seed = self.config['Feature'].get('RandomSeed', 1024)
        self.all_features = self.config['Feature'].get('FeatureName', [])
        self.cat_features = self.config['Feature'].get('CategoricalFeature', [])
        self.sampling_way = self.config['Feature'].get('Sampling', None)
        self.plot = self.config['Feature'].get('Plot', True)
        
        # Incremental learning configuration
        self.incrml = self.config['IncrML'].get('Incrml', True)
        self.model_save_path = self.config['IncrML'].get('SaveModelPath', None)
        
        # Set random seed to ensure reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Load training data
        self.val_df = None  # First initialize to None to avoid referencing undefined variables later
        
        if data is None:
            # Load training data from files
            self.df = pd.DataFrame()
            train_files = glob(os.path.join(self.train_dir, '*'))
            if not train_files:
                logger.warning(f"Training data directory is empty: {self.train_dir}")
            for file in train_files:
                try:
                    file_data = pd.read_csv(file)
                    self.df = pd.concat([self.df, file_data])
                except Exception as e:
                    logger.error(f"Error reading file {file}: {str(e)}")
        else:
            # Use provided data
            self.df = data.copy()
        
        # Load validation data (if available)
        if self.val_dir is not None:
            val_files = glob(os.path.join(self.val_dir, '*'))
            if val_files:
                self.val_df = pd.DataFrame()
                for file in val_files:
                    try:
                        file_data = pd.read_csv(file)
                        self.val_df = pd.concat([self.val_df, file_data])
                    except Exception as e:
                        logger.error(f"Error reading validation file {file}: {str(e)}")
        
        # If data is empty, log warning and return
        if self.df.empty:
            logger.warning("Training data is empty, cannot continue processing")
            return
            
        # Standardize column names
        self.df.columns = self.check_colname(list(self.df.columns))
        if self.val_df is not None:
            self.val_df.columns = self.check_colname(list(self.val_df.columns))
        
        # Handle missing values
        self.df.dropna(subset=[self.target_name], inplace=True)
        if self.val_df is not None:
            self.val_df.dropna(subset=[self.target_name], inplace=True)
        
        # Handle infinite values
        self.df.replace(np.inf, 1000000000.0, inplace=True)
        self.df.replace(-np.inf, -1000000000.0, inplace=True)
        if self.val_df is not None:
            self.val_df.replace(np.inf, 1000000000.0, inplace=True)
            self.val_df.replace(-np.inf, -1000000000.0, inplace=True)
        
        # Initialize transformer and data processor
        self.transform = Transform(**self.config)
        self.dp = DataProcess(**self.config)
        
        # Process training data
        try:
            self.df, high_nan_rate_features = self.dp.processing(df=self.df, transform=self.transform, valation=False)
            
            # Process validation data (if available)
            if self.val_df is not None:
                # Remove high missing rate features from validation data
                self.val_df.drop(columns=[col for col in high_nan_rate_features if col in self.val_df.columns], inplace=True)
                self.val_df = self.dp.processing(df=self.val_df, transform=self.transform, valation=True)
            
            # Update feature lists, remove high missing rate features
            self.cat_features = [column for column in self.cat_features if column not in high_nan_rate_features]
            self.config['Feature']['CategoricalFeature'] = self.cat_features
            self.all_features = [column for column in self.all_features if column not in high_nan_rate_features]
            self.config['Feature']['FeatureName'] = self.all_features
            
            # Process time index features (if available)
            if self.time_index is not None:
                new_time_index = [
                    f'year_{self.time_index}', f'month_{self.time_index}', f'day_{self.time_index}',
                    f'hour_{self.time_index}', f'minute_{self.time_index}', f'second_{self.time_index}',
                    f'dayofweek_{self.time_index}', f'dayofyear_{self.time_index}', f'quarter_{self.time_index}'
                ]
                for new_index in new_time_index:
                    if new_index in self.df.columns:
                        self.cat_features.append(new_index)
            
            # Split training and validation sets
            self._train_test_split()
            
        except Exception as e:
            logger.error(f"Error during data processing: {str(e)}")
            raise

    def _train_test_split(self) -> None:
        """
        Split data into training and validation sets
        
        Split data into training and validation sets according to the configured splitting strategy
        (stratified, time, or random). If a validation set (self.val_df) is already provided, use it directly.
        """
        try:
            # If validation set is already provided, use it directly
            if self.val_df is not None:
                self.X_train = self.df.drop(columns=[self.target_name])
                self.y_train = self.df[self.target_name]
                self.X_val = self.val_df.drop(columns=[self.target_name])
                self.y_val = self.val_df[self.target_name]
            else:
                # Otherwise split data according to splitting strategy
                X = self.df.drop(columns=[self.target_name])
                y = self.df[self.target_name]
                
                # Stratified splitting
                if self.split_type == 'stratified':
                    # For classification tasks, stratify directly by class
                    if self.task_type == 'classification':
                        stratify_array = y
                    # For regression tasks, divide the target variable into two groups for stratification
                    else:
                        stratify_array = pd.qcut(y.rank(method='first'), q=2, labels=False)
                    
                    self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                        X, y, test_size=self.test_size, random_state=self.seed, stratify=stratify_array
                    )
                
                # Time series splitting (take the first part in order as the training set)
                elif self.split_type == 'time':
                    train_size = int(len(X) * (1 - self.test_size))
                    self.X_train = X.iloc[:train_size, :]
                    self.y_train = y.iloc[:train_size]
                    self.X_val = X.iloc[train_size:, :]
                    self.y_val = y.iloc[train_size:]
                
                # Random splitting
                else:
                    self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                        X, y, test_size=self.test_size, random_state=self.seed
                    )
                
                # Release variables that are no longer needed
                del X, y
                gc.collect()
            
            # Reset indices to ensure data consistency
            self.X_train = self.X_train.reset_index(drop=True)
            self.y_train = self.y_train.reset_index(drop=True)
            self.X_val = self.X_val.reset_index(drop=True)
            self.y_val = self.y_val.reset_index(drop=True)
            
            # Log data splitting results
            logger.info(f"Data splitting completed: Training set {len(self.X_train)} samples, validation set {len(self.X_val)} samples")
            
        except Exception as e:
            logger.error(f"Error during data splitting: {str(e)}")
            raise

    def check_config_columns(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check and standardize column names in configuration
        
        Ensure that the Feature section exists in the configuration and standardize the column names in it.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Processed configuration dictionary
        """
        try:
            # Ensure Feature section exists
            if config.get('Feature', None) is None:
                config['Feature'] = {}
                
            # Standardize various feature column names
            config['Feature']['FeatureName'] = self.check_colname(config['Feature'].get('FeatureName', None))
            config['Feature']['CategoricalFeature'] = self.check_colname(config['Feature'].get('CategoricalFeature', None))
            config['Feature']['GroupIndex'] = self.check_colname(config['Feature'].get('GroupIndex', None))
            config['Feature']['IgnoreFeature'] = self.check_colname(config['Feature'].get('IgnoreFeature', None))
            config['Feature']['TimeIndex'] = self.check_colname(config['Feature'].get('TimeIndex', None))
            config['Feature']['TargetName'] = self.check_colname(config['Feature'].get('TargetName', None))
            
            return config
            
        except Exception as e:
            logger.error(f"Error checking configuration column names: {str(e)}")
            # Return original configuration to avoid complete failure
            return config

    def check_colname(self, new_col: Union[List[str], str, None]) -> Union[List[str], str, None]:
        """
        Standardize column names
        
        Replace special characters in column names with underscores to ensure validity and consistency.
        
        Args:
            new_col: Column names to standardize, can be a string, list of strings, or None
            
        Returns:
            Standardized column names, maintaining the original type
        """
        # If column name is None, return directly
        if new_col is None:
            return None
            
        # Special characters to be replaced
        replace_str = [
            ':', '[', ']', '（', '）', '！', '＠', '＃', '￥', '％', '…', 
            '《', '》', '【', '】', ' ', '(', ')', '!', '@', '#', '$', '%', 
            '^', '&', '*', '+', '=', '{', '}', '|', '\\', '/', '?', '<', '>'
        ]
        
        try:
            # If it's a list, process each element
            if isinstance(new_col, list):
                result = []
                for col in new_col:
                    if col is not None:  # Ensure elements in the list are not None
                        processed_col = col.strip()
                        for s in replace_str:
                            processed_col = processed_col.replace(s, '_')
                        result.append(processed_col)
                return result
            # If it's a string, process directly
            elif isinstance(new_col, str):
                processed_col = new_col.strip()
                for s in replace_str:
                    processed_col = processed_col.replace(s, '_')
                return processed_col
            # Return other types directly
            else:
                return new_col
                
        except Exception as e:
            logger.error(f"Error processing column names: {str(e)}")
            return new_col