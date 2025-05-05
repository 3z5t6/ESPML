"""
Data Processing Module

This module provides data processing related functionality, including data cleaning, feature processing, type conversion, and class imbalance handling.
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Union, Optional, Any

from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from module.autofe.transform import Transform
from module.utils.log import get_logger

logger = get_logger(__name__)


class DataProcess:
    """
    Data Processing Class
    
    Provides various data preprocessing and cleaning methods, including time feature processing, type conversion, class imbalance handling, etc.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the data processing class
        
        Args:
            **kwargs: Configuration parameter dictionary, containing feature configuration, task type and other information
        """
        self.kwargs = kwargs
        self.lec_record_dt = {}
        
        # Extract feature related settings from configuration
        feature_config = kwargs.get('Feature', {})
        self.category_features = feature_config.get('CategoricalFeature', [])
        self.time_index = feature_config.get('TimeIndex', None)
        self.group_index = feature_config.get('GroupIndex', None)
        self.ignore_features = feature_config.get('IgnoreFeature', [])
        self.sampling_way = feature_config.get('Sampling', None)
        self.seed = feature_config.get('RandomSeed', 1024)
        self.task_type = feature_config.get('TaskType', None)
        self.target_name = feature_config.get('TargetName', None)

    def processing(self, df: pd.DataFrame, transform: Optional[Transform] = None, valation: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[str]]]:
        """
        Main data processing function, including multiple data preprocessing steps
        
        Processing steps include: feature transformation, time feature processing, removing unused features, removing duplicates,
        type conversion and class imbalance handling, etc.
        
        Args:
            df: Input DataFrame
            transform: Feature transformer, default is None
            valation: Whether it is in validation set processing mode, default is False
            
        Returns:
            If in validation set mode, returns the processed DataFrame;
            If in training set mode, returns the processed DataFrame and the list of deleted features
        """
        # Initialize transformer
        if transform is None:
            transform = Transform()
            
        # Choose different transformation methods based on whether it is validation set mode
        df = transform.fit_transform(df, labelencode=True) if not valation else transform.transform(df, labelencode=True)
        
        # Process time features, generate time-related features
        if self.time_index is not None and self.time_index in df.columns:
            # Convert to datetime type
            df[self.time_index] = pd.to_datetime(df[self.time_index])
            tmp_dt = df[self.time_index].dt
            
            # Create regular time features
            new_columns_dict = {
                f'year_{self.time_index}': tmp_dt.year, 
                f'month_{self.time_index}': tmp_dt.month, 
                f'day_{self.time_index}': tmp_dt.day, 
                f'hour_{self.time_index}': tmp_dt.hour, 
                f'minute_{self.time_index}': tmp_dt.minute, 
                f'second_{self.time_index}': tmp_dt.second, 
                f'dayofweek_{self.time_index}': tmp_dt.dayofweek, 
                f'dayofyear_{self.time_index}': tmp_dt.dayofyear, 
                f'quarter_{self.time_index}': tmp_dt.quarter
            }
            
            # Only add time features that have multiple different values
            for key, value in new_columns_dict.items():
                if key not in df.columns and value.nunique(dropna=False) >= 2:
                    df[key] = value
                    
            # Release temporary variables
            del tmp_dt
        
        # Delete ignored features
        for col in self.ignore_features:
            if col in df.columns and col != self.time_index and col != self.group_index:
                df = df.drop(col, axis=1)
        
        # If not in validation set mode, delete features with only one unique value
        high_nan_rate_features = []
        if not valation:
            for column in df.columns:
                if column != self.time_index and df[column].nunique() <= 1:
                    df.drop(columns=column, inplace=True)
                    high_nan_rate_features.append(column)
        
        # Data cleaning
        df.drop_duplicates(inplace=True)  # Remove duplicate rows
        df.dropna(how='all', inplace=True)  # Remove rows with all NaN values
        
        # Convert percentage signs to decimal
        df = self.percent_sign_conversion(df)
        
        # Sort by time index
        if self.time_index is not None and self.time_index in df.columns:
            df.sort_values(by=self.time_index, inplace=True)
        
        # Convert categorical features to category type
        df = self.astype_category(df, self.category_features)
        
        # Handle class imbalance
        if self.task_type == 'classification' and self.sampling_way is not False and self.target_name in df.columns:
            df = self.class_imbalance(df, self.target_name, self.sampling_way, self.seed)
        
        # Return results
        if not valation:
            return df, high_nan_rate_features
        return df

    def astype_category(self, df: pd.DataFrame, category_feature: List[str]) -> pd.DataFrame:
        """
        Convert specified features to category type
        
        Args:
            df: Input DataFrame
            category_feature: List of features to be converted to category type
            
        Returns:
            Converted DataFrame
        """
        for feature in category_feature:
            if feature in df.columns:
                df[feature] = df[feature].astype('category')
        return df

    def class_imbalance(self, df: pd.DataFrame, target_name: str, sampling_way: Optional[str] = None, seed: int = 1024) -> pd.DataFrame:
        """
        Handle class imbalance problem
        
        Supports three modes:
        1. Automatic mode (sampling_way=None): Supplement classes with fewer than 10 samples to 10 samples
        2. Oversampling mode (sampling_way='upsampling'): Supplement all class samples to the same as the most numerous class
        3. Undersampling mode (sampling_way='downsampling'): Reduce all class samples to the same as the least numerous class
        
        Args:
            df: Input DataFrame
            target_name: Target variable name
            sampling_way: Sampling mode, default is None
            seed: Random seed, default is 1024
            
        Returns:
            DataFrame after handling class imbalance
        """
        # Extract features and target variable
        x, y = df.drop(columns=[target_name]), df[target_name]
        
        # Count the number of samples for each class
        class_counts = Counter(y)
        min_key = min(class_counts, key=class_counts.get)
        
        # Automatic mode: Supplement classes with fewer than 10 samples to 10 samples
        if class_counts[min_key] < 10 and sampling_way is None:
            logger.info('Using oversampling to balance data.')
            logger.info(f'Class {min_key} sample count increased from {class_counts[min_key]} to 10.')
            logger.info('Due to the presence of a small amount of resampled data in the validation set, model accuracy may be affected.')
            
            # Create oversampling strategy, only oversample classes with fewer than 10 samples
            sampling_strategy = {key: 10 for key, values in class_counts.items() if values < 10}
            sampler = RandomOverSampler(random_state=seed, sampling_strategy=sampling_strategy)
            
            # Perform oversampling
            X_resampled, y_resampled = sampler.fit_resample(x, y)
            
            # Merge features and target variable
            df = pd.concat([
                pd.DataFrame(X_resampled, columns=x.columns), 
                pd.DataFrame(y_resampled, columns=[target_name])
            ], axis=1)
            
            return df
            
        # Oversampling mode: Supplement all class samples to the same as the most numerous class
        elif sampling_way == 'upsampling':
            logger.info('Using oversampling to balance data.')
            self.lec_record_dt['oversampling'] = [len(x)]
            
            # Create oversampler
            sampler = RandomOverSampler(random_state=seed)
            
            # Perform oversampling
            X_resampled, y_resampled = sampler.fit_resample(x, y)
            
            logger.info(f'Sample count increased from {len(df)} to {len(X_resampled)}.')
            logger.info('Due to validation set containing a small amount of resampled data, model accuracy may be affected.')
            
            # Merge features and target variable
            df = pd.concat([
                pd.DataFrame(X_resampled, columns=x.columns), 
                pd.DataFrame(y_resampled, columns=[target_name])
            ], axis=1)
            
            return df
            
        # Undersampling mode: Reduce all class samples to the same as the least numerous class
        elif sampling_way == 'downsampling':
            logger.info('Using undersampling to balance data.')
            self.lec_record_dt['oversampling'] = [len(x)]
            
            # Create undersampler
            sampler = RandomUnderSampler(random_state=seed)
            
            # Perform undersampling
            X_resampled, y_resampled = sampler.fit_resample(x, y)
            
            logger.info(f'Sample count reduced from {len(df)} to {len(X_resampled)}.')
            logger.info('Due to validation set containing a small amount of resampled data, model accuracy may be affected.')
            
            # Merge features and target variable
            df = pd.concat([
                pd.DataFrame(X_resampled, columns=x.columns), 
                pd.DataFrame(y_resampled, columns=[target_name])
            ], axis=1)
        
        # Return the processed DataFrame
        return df

    def percent_sign_conversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert percentage sign to decimal
        
        Detect if string and category type columns contain percentage signs; if so, convert them to decimal values
        For example: '50%' converted to 0.5
        
        Args:
            df: Input DataFrame
            
        Returns:
            Converted DataFrame
        """
        # Create result DataFrame
        result_df = pd.DataFrame()
        
        # Iterate through each column
        for column in df.columns:
            column_dtype = df[column].dtype
            
            # Process string and category types
            if column_dtype == 'object' or column_dtype.name == 'category':
                # Convert to string type
                column_data = df[column].astype(str)
                
                # Check if it contains percentage signs
                contains_percentage = column_data.str.endswith('%')
                
                # If it contains percentage signs, convert
                if contains_percentage.any():
                    # Remove percentage sign and convert to float, then divide by 100
                    converted_data = column_data.str.replace('%', '').astype(float) / 100
                    result_df[column] = converted_data
                else:
                    # If it doesn't contain percentage signs, keep as is
                    result_df[column] = df[column]
            else:
                # For non-string and non-category types, keep as is
                result_df[column] = df[column]
                
        return result_df