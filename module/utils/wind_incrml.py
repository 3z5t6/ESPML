#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
WindIncrml module

This module provides the WindIncrml class for wind power prediction incremental learning and model training.
It includes data loading, feature engineering, model training, prediction, and backtesting functions.
"""

# Standard library imports

import os
import gc
import time
import traceback
import hashlib
import shutil
import pickle
import logging
from datetime import datetime
from functools import partial
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import pandas as pd
import polars as pl
from schedule import every, repeat, run_pending, jobs
from sklearn.metrics import r2_score


from module.utils.ml import ESPML
from module.utils.test_file import predict
from module.utils.yaml_parser import YamlParser
from module.utils.log import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

def init_logger(config: Dict[str, Any], task_name: str, logtype: str) -> None:
    """
    Initialize logger
    
    Create logger for specified task and log type, including file handler and stream handler.
    If file handler already exists, remove it first.
    
    Args:
        config: Configuration dictionary, containing log file path etc.
        task_name: Task name, used to construct log file name
        logtype: Log type, such as 'train', 'predict', etc.
    """
    try:
        # Remove existing file handlers to avoid duplicate logging
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
        
        # Create log file path
        log_file_path = os.path.join(config['Path']['LogFile'], f'{task_name}_{logtype}.log')
        
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            logger.info(f"Create log directory: {log_dir}")
        
        # Create file handler
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Create console handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        logger.info(f"Initialize {task_name}_{logtype} logger successfully")
        
    except Exception as e:
        # If logger initialization fails, use standard error output to report the error
        print(f"Logger initialization failed: {str(e)}")
        # Create a basic console handler to ensure some log output
        basic_handler = logging.StreamHandler()
        basic_handler.setLevel(logging.DEBUG)
        basic_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(basic_handler)


class WindIncrml:
    """
    Wind power incremental learning class
    
    This class provides incremental learning and model training for wind power prediction, including data loading, feature engineering, model training, prediction, and backtesting.
    """

    def __init__(self, config: Dict[str, Any], task_name: str, limit_model_num: int = 100) -> None:
        """
        Initialize WindIncrml class
        
        Args:
            config: Configuration dictionary, containing data path, feature configuration, etc.
            task_name: Task name, such as 'Forecast4Hour'
            limit_model_num: Model number limit, default is 100
        """
        try:
            # Save basic parameters
            self.config = config
            self.task_name = task_name
            
            # Extract time column and task information from configuration
            self.time_col = config.get('Features', {}).get('TimeIndex', None)
            self.tasks = config.get('Task', None)
            
            # Check necessary configuration items
            if self.time_col is None or self.tasks is None:
                logger.error("Missing necessary configuration items: TimeIndex or Task")
                raise ValueError("Missing necessary configuration items")
            
            # Calculate time interval related parameters
            self.ahead_rows = config.get('Features', {}).get('AheadRows', 0)
            self.interval = self.tasks[self.task_name]['RowInterval'] + self.ahead_rows
            self.n = self.interval * 15 # Each row is 15 minutes
            
            # Set paths
            self.predict_save_path = config['Path']['Prediction']
            self.label_save_path = os.path.join(config['Path']['Prediction'], 'real_wind_power.csv')
            self.model_path = os.path.join(config['Path']['Model'], self.task_name)
            
            # Set weather data related columns
            self.sc_date = config['Features']['WeatherSCDateIndex']
            self.sc_time = config['Features']['WeatherSCTimeIndex']
            self.pre_time = config['Features']['WeatherPreTimeIndex']
            self.target_name = config['Features']['TargetName']
            
            # Set configuration file path
            base_path = os.path.dirname(os.path.dirname(__file__))
            self.config_pth = os.path.join(base_path, 'project', 'WindPower', 'config.yaml')
            
            # Set time delay related parameters
            self.hourlagtime = config.get('Features', {}).get('HourSubmitLagTime', '0min')
            self.daylagtime = config.get('Features', {}).get('DaySubmitLagTime', '0hour')
            self.time_sleep = int(self.config.get('Features', {}).get('PredictWaiting', 0))
            
            # Set debug CSV save path
            self.debug_csv_save_path = config['Path']['OtherFile']
            
            # Set prediction result save path
            today = datetime.now().strftime('%Y%m%d_%H%M')
            if self.task_name != 'Forecast4Hour':
                self.real_predict_save_path = os.path.join(
                    self.predict_save_path, 
                    f'real_{today}_{self.task_name}.csv'
                )
            else:
                self.real_predict_save_path = os.path.join(
                    self.predict_save_path, 
                    f'{self.task_name}.csv'
                )
            
            # Limit model number
            self.limit_model_num(limit_model_num)
            
            logger.info(f"Initialize WindIncrml class successfully, task name: {task_name}")
            
        except Exception as e:
            logger.error(f"Initialize WindIncrml class failed: {str(e)}")
            raise

    def limit_model_num(self, limit_num: int = 100) -> None:
        """
        Limit model number
        
        When the number of models exceeds the specified limit, delete old model folders.
        The deletion strategy is to delete models from the earliest one until the number of models does not exceed the limit.
        
        Args:
            limit_num: Model number limit, default is 100
        """
        try:
            # Check if model path exists
            if not os.path.exists(self.model_path):
                logger.info(f"Model path does not exist: {self.model_path}")
                return
                
            # Get all folders in the model directory
            try:
                # Try to get all folder names that can be converted to integers
                model_dirs = [d for d in os.listdir(self.model_path) if d.isdigit()]
                model_nums = [int(d) for d in model_dirs]
            except (FileNotFoundError, PermissionError) as e:
                logger.error(f"Cannot access model directory: {str(e)}")
                return
                
            # If there are no models, return directly
            if not model_nums:
                logger.info(f"Model directory is empty: {self.model_path}")
                return
                
            # Find the largest model number
            seq = max(model_nums)
            
            # Find valid model numbers (models containing automl/val_res directory)
            while seq > 0 and not os.path.exists(os.path.join(self.model_path, str(seq), 'automl', 'val_res')):
                logger.warning(f"Model {seq} is missing val_res directory")
                seq -= 1
                
            # If the number of models exceeds the limit, delete old models
            if seq > limit_num:
                rm_num = seq - limit_num
                logger.info(f"Number of models({seq}) exceeds limit({limit_num}), will delete {rm_num} old models")
                
                for i in range(1, rm_num + 1):
                    model_dir = os.path.join(self.model_path, str(i))
                    if os.path.exists(model_dir):
                        try:
                            shutil.rmtree(model_dir, ignore_errors=True)
                            logger.info(f"Delete old model: {model_dir}")
                        except Exception as e:
                            logger.error(f"Delete model {i} failed: {str(e)}")
            else:
                logger.info(f"Current model number({seq}) does not exceed limit({limit_num})")
                
        except Exception as e:
            logger.error(f"Limit model number failed: {str(e)}")
            # Not raise exception, because this is not a critical operation

    def incrml_train(self) -> None:
        """
        Train incremental learning model
        
        Initialize logger and call train_task method for model training.
        """
        try:
            # Initialize logger
            init_logger(self.config, self.task_name, 'train')
            logger.info(f"Start {self.task_name} incremental learning training")
            
            # Call training task
            self.train_task()
            
            logger.info(f"{self.task_name} incremental learning training completed")
        except Exception as e:
            logger.error(f"{self.task_name} incremental learning training failed: {str(e)}")
            raise

    def incrml_predict(self) -> pd.DataFrame:
        """
        Use incremental learning model for prediction
        
        Initialize logger and call predict_task method for prediction.
        
        Returns:
            Prediction result DataFrame
        """
        try:
            # Initialize logger
            init_logger(self.config, self.task_name, 'predict')
            logger.info(f"Start {self.task_name} incremental learning prediction")
            
            # Call prediction task
            result = self.predict_task()
            
            logger.info(f"{self.task_name} incremental learning prediction completed")
            return result
        except Exception as e:
            logger.error(f"{self.task_name} incremental learning prediction failed: {str(e)}")
            # In prediction failure, return empty DataFrame
            return pd.DataFrame()

    def weather_forecast_group(self, timestamp: pd.Series, n: int, weather_max_sc_time: str, predict_time: int = 8, previous_day: int = 0) -> pd.Series:
        """
        Calculate weather forecast group timestamp
        
        Based on the input timestamp series, calculate the weather forecast group timestamp.
        This method adjusts the time to the specified prediction time (default is 8 AM),
        and considers the offset of the previous day.
        
        Args:
            timestamp: Timestamp series
            n: Time interval
            weather_max_sc_time: Maximum weather data release time
            predict_time: Prediction time (hours), default is 8
            previous_day: Offset of the previous day, default is 0
            
        Returns:
            Calculated timestamp series
        """
        try:
            # Calculate the prediction time point for today (default is 08:00)
            eight_pm_today = (
                timestamp + 
                pd.to_timedelta(predict_time - timestamp.dt.hour, unit='h') - 
                pd.to_timedelta(timestamp.dt.minute, unit='m') - 
                pd.to_timedelta(timestamp.dt.second, unit='s') - 
                pd.to_timedelta(timestamp.dt.microsecond, unit='us') - 
                pd.to_timedelta(previous_day, unit='d')
            )
            
            # If the current hour is less than the prediction time, adjust the day back by one
            eight_pm_today[timestamp.dt.hour < predict_time] -= pd.Timedelta(days=1)
            
            # Ensure it does not exceed the maximum weather release time
            max_sc_time = pd.Series(pd.to_datetime([weather_max_sc_time] * len(eight_pm_today)))
            eight_pm_today = pd.concat([eight_pm_today, max_sc_time], axis=1).min(axis=1)
            
            logger.debug(f"Calculate weather forecast group time successfully, prediction time: {predict_time} point, previous day offset: {previous_day}")
            return eight_pm_today
            
        except Exception as e:
            logger.error(f"Calculate weather forecast group time failed: {str(e)}")
            # Return original timestamp when error occurs
            return timestamp

    def weather_data_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process weather data
        
        Preprocess weather data, including creating group column, removing null values, converting time format, and resampling data.
        
        Args:
            df: Input weather data frame
            
        Returns:
            Processed weather data frame
        """
        try:
            # Create a copy of the data frame to avoid modifying the original data
            df_copy = df.copy()
            
            # Check if necessary columns exist
            required_columns = [self.sc_time, self.pre_time]
            if self.sc_date is not None:
                required_columns.append(self.sc_date)
                
            for col in required_columns:
                if col not in df_copy.columns:
                    logger.error(f"Weather data missing necessary column: {col}")
                    return df_copy
            
            # Check if the data frame is empty
            if df_copy.empty:
                logger.warning("Weather data frame is empty")
                return df_copy
                
            # Create group column
            try:
                # Determine the format of the release time
                if len(df_copy[self.sc_time].values[0].split(' ')) > 1 or len(df_copy[self.sc_time].values[0].split('_')) > 1:
                    # If the release time already contains date and time, use it directly
                    df_copy['GROUP'] = df_copy[self.sc_time]
                    logger.debug("Use existing release time as grouping")
                else:
                    # Otherwise combine date and time
                    df_copy['GROUP'] = df_copy[self.sc_date].astype(str) + ' ' + df_copy[self.sc_time].astype(str)
                    logger.debug("Combine date and time as grouping")
            except Exception as e:
                logger.error(f"Error creating group column: {str(e)}")
                # If an error occurs, try using the default method
                df_copy['GROUP'] = df_copy[self.sc_time]
                
            # Delete null values in necessary columns
            df_copy.dropna(subset=['GROUP', self.pre_time], inplace=True)
            logger.debug(f"Number of rows after deleting null values: {len(df_copy)}")
            
            # Convert prediction time to datetime format
            df_copy[self.pre_time] = pd.to_datetime(df_copy[self.pre_time])
            
            # Initialize resampled data list
            resampled_data = []
            
            # Process data by group
            for group, group_data in df_copy.groupby('GROUP'):
                try:
                    # Delete duplicate prediction times
                    group_data = group_data.drop_duplicates(subset=[self.pre_time])
                    
                    # Set prediction time as index
                    group_data = group_data.set_index(self.pre_time)
                    
                    # Only select numeric columns for resampling
                    group_data_numeric = group_data.select_dtypes(include='number')
                    
                    # Resample every 15 minutes and interpolate linearly
                    group_data_resampled = group_data_numeric.resample('15min').interpolate(method='linear')
                    
                    # Add group column
                    group_data_resampled['GROUP'] = group
                    
                    # Reset index and add to result list
                    resampled_data.append(group_data_resampled.reset_index())
                    logger.debug(f"Successfully processed group {group}, resampled data row count: {len(group_data_resampled)}")
                except Exception as e:
                    logger.error(f"Error processing group {group}: {str(e)}")
            
            # If no resampled data, return original data frame
            if not resampled_data:
                logger.warning("No available resampled data")
                return df_copy
                
            # Merge all resampled data
            result = pd.concat(resampled_data, ignore_index=True)
            logger.info(f"Weather data processing completed, final data row count: {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing weather data: {str(e)}")
            # Return original data frame to avoid complete failure
            return df

    def weather_data_merge(self, df: pd.DataFrame, weather_data: pd.DataFrame, previous_day: int = 0) -> pd.DataFrame:
        """
        Merge weather data with input data frame based on timestamp and prediction time.
        
        Args:
            df: Input data frame
            weather_data: Weather data frame
            previous_day: Offset of previous day, default is 0
            
        Returns:
            Merged data frame
        """
        try:
            # Create a copy of the data frame to avoid modifying the original data
            df_copy = df.copy()
            weather_df = weather_data.copy()
            
            # Check if the data frames are empty
            if df_copy.empty or weather_df.empty:
                logger.warning("Input data frame or weather data frame is empty")
                return df_copy
                
            # Check if necessary columns exist
            if 'GROUP' not in weather_df.columns:
                logger.error("Weather data missing GROUP column")
                return df_copy
                
            if self.time_col not in df_copy.columns:
                logger.error(f"Input data missing {self.time_col} column")
                return df_copy
            
            # Extract prediction time point (hour)
            try:
                # Extract prediction time from the first row of the GROUP column
                group_str = weather_df['GROUP'].astype(str).values[0]
                predict_time = int(group_str.split(' ')[1].split(':')[0])
                logger.debug(f"Extracted prediction time point: {predict_time}")
            except Exception as e:
                logger.error(f"Error extracting prediction time point: {str(e)}")
                predict_time = 8  # Default to 8 o'clock
            
            # Calculate the next prediction time point
            df_copy['next_8pm'] = self.weather_forecast_group(
                df_copy[self.time_col], 
                self.n, 
                weather_df['GROUP'].max(), 
                predict_time, 
                previous_day
            )
            
            # Calculate prediction time (current time + n minutes)
            df_copy['pre_time'] = pd.to_datetime(df_copy[self.time_col]) + pd.Timedelta(minutes=self.n)
            
            # Convert the GROUP column of weather data to datetime format
            weather_df['GROUP'] = pd.to_datetime(weather_df['GROUP'])
            
            # Merge data frames
            merged_df = df_copy.merge(
                weather_df, 
                left_on=['pre_time', 'next_8pm'], 
                right_on=[self.pre_time, 'GROUP'], 
                how='left', 
                suffixes=('', f'_previou_day_{previous_day}')
            )
            
            # Drop unnecessary columns
            merged_df.drop(['pre_time', 'GROUP', 'next_8pm', self.pre_time], axis=1, inplace=True)
            
            # Check the merged data
            missing_count = merged_df.isnull().sum().sum()
            if missing_count > 0:
                logger.warning(f"Missing {missing_count} values after merging")
                
            logger.info(f"Weather data merged successfully, result data row count: {len(merged_df)}")
            return merged_df
            
        except Exception as e:
            logger.error(f"Error merging weather data: {str(e)}")
            # Return original data frame to avoid complete failure
            return df

    def merge_all_weather_data(self, df: pd.DataFrame, weather_data: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge all weather data
        
        Merge multiple weather data frames with the input data frame, considering different day offsets.
        
        Args:
            df: Input data frame
            weather_data: List of weather data frames
            
        Returns:
            Merged data frame
        """
        try:
            result_df = df.copy()
            weather_md = int(self.config['Features']['WeatherMergeDays'])
            
            for wd in weather_data:
                for i in range(weather_md):
                    result_df = self.weather_data_merge(result_df, wd, i)
                    
            logger.info(f"All weather data merged successfully, result data row count: {len(result_df)}")
            return result_df
            
        except Exception as e:
            logger.error(f"Error merging all weather data: {str(e)}")
            # Return original data frame to avoid complete failure
            return df

    def add_time_feature_process(self, df: pd.DataFrame, set_label: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process time features for the data frame
        
        Extract time-related features (year, month, day, hour, etc.) from the time column, and optionally set the label.
        Update the task configuration to add the newly added time features to the feature list.
        
        Args:
            df: Input data frame
            set_label: Whether to set the label, default is True
            
        Returns:
            Tuple containing the processed data frame and updated task configuration
        """
        try:
            # Create a copy of the data frame to avoid modifying the original data
            df_copy = df.copy()
            add_features = []
            
            # Check if the necessary columns exist
            if self.time_col not in df_copy.columns:
                logger.error(f"Missing time column: {self.time_col}")
                return df_copy, {}
                
            # Extract time features
            try:
                # Get the date time attribute of the time column
                time_dt = df_copy[self.time_col].dt
                
                # Define time features list
                time_features = {
                    f'year_{self.time_col}': time_dt.year,
                    f'month_{self.time_col}': time_dt.month,
                    f'day_{self.time_col}': time_dt.day,
                    f'hour_{self.time_col}': time_dt.hour,
                    f'minute_{self.time_col}': time_dt.minute,
                    f'second_{self.time_col}': time_dt.second,
                    f'dayofweek_{self.time_col}': time_dt.dayofweek,
                    f'dayofyear_{self.time_col}': time_dt.dayofyear,
                    f'quarter_{self.time_col}': time_dt.quarter
                }
                
                # Add time features to the data frame
                for feature_name, feature_values in time_features.items():
                    add_features.append(feature_name)
                    df_copy[feature_name] = feature_values
                    
                logger.debug(f"Added {len(add_features)} time features")
                    
            except Exception as e:
                logger.error(f"Error adding time features: {str(e)}")
                
            # Set label (if required)
            if set_label and self.target_name in df_copy.columns:
                try:
                    # Create a temporary data frame, including time and target columns
                    temp_df = df_copy[[self.time_col, self.target_name]].copy()
                    
                    # Calculate the time of the previous day (based on the interval)
                    days_offset = self.interval // 96  # 96 = 24小时 * 4（每小时15分钟）
                    temp_df['pre_time'] = pd.to_datetime(temp_df[self.time_col]) - pd.Timedelta(days=days_offset)
                    
                    # Set label
                    temp_df['label'] = temp_df[self.target_name]
                    
                    # Merge back to the original data frame
                    df_copy = df_copy.merge(
                        temp_df[['pre_time', 'label']], 
                        left_on=self.time_col, 
                        right_on='pre_time', 
                        how='left'
                    )
                    
                    # Delete the temporary column
                    df_copy.drop(columns=['pre_time'], inplace=True)
                    logger.debug("Successfully set label")
                    
                except Exception as e:
                    logger.error(f"Error setting label: {str(e)}")
            
            # Parse task configuration
            try:
                task_config = YamlParser.parse(self.config_pth)
                
                # Update feature list
                task_config['Feature']['FeatureName'].extend(add_features)
                task_config['Feature']['CategoricalFeature'].extend(add_features)
                
                # Set target column name
                task_config['Feature']['TargetName'] = 'label'
                
                logger.debug("Successfully update task configuration")
                
            except Exception as e:
                logger.error(f"Error updating task configuration: {str(e)}")
                # Create an empty configuration as a backup option
                task_config = {'Feature': {'FeatureName': add_features, 'CategoricalFeature': add_features, 'TargetName': 'label'}}
            
            return df_copy, task_config
            
        except Exception as e:
            logger.error(f"Error processing time features: {str(e)}")
            # Return original data frame and empty configuration
            return df, {}

    def update_config(self, task_config: Dict[str, Any], task_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Update the task configuration
        
        Update the task configuration based on the current instance configuration, including feature names, categorical features, time index, etc.
        
        Args:
            task_config: Task configuration dictionary to be updated
            task_name: Task name, if provided, use this task name, otherwise use the instance task name
            
        Returns:
            Updated task configuration dictionary
        """
        try:
            # Create a copy of the configuration to avoid modifying the original configuration
            updated_config = task_config.copy()
            
            # Check if the required configuration items exist
            required_config_keys = ['Feature', 'AutoML', 'AutoFE', 'IncrML']
            for key in required_config_keys:
                if key not in updated_config:
                    logger.warning(f"Task configuration missing '{key}' section, will be automatically created")
                    updated_config[key] = {}
            
            # Update feature-related configuration
            try:
                # Feature name
                if 'FeatureName' in self.config.get('Features', {}):
                    updated_config['Feature']['FeatureName'] = self.config['Features']['FeatureName']
                    
                # Categorical feature
                if 'CategoricalFeature' in self.config.get('Features', {}):
                    updated_config['Feature']['CategoricalFeature'] = self.config['Features']['CategoricalFeature']
                    
                # Time index
                if 'TimeIndex' in self.config.get('Features', {}):
                    updated_config['Feature']['TimeIndex'] = self.config['Features']['TimeIndex']
                    
                # Ignore feature
                if 'IgnoreFeature' in self.config.get('Features', {}):
                    updated_config['Feature']['IgnoreFeature'] = self.config['Features']['IgnoreFeature']
                    
                logger.debug("Successfully update feature configuration")
                
            except Exception as e:
                logger.error(f"Error updating feature configuration: {str(e)}")
            
            # Update AutoML-related configuration
            try:
                # Time budget
                if 'AutoMLTimeBudget' in self.config.get('Features', {}):
                    updated_config['AutoML']['TimeBudget'] = self.config['Features']['AutoMLTimeBudget']
                    
                logger.debug("Successfully update AutoML configuration")
                
            except Exception as e:
                logger.error(f"Error updating AutoML configuration: {str(e)}")
            
            # Update AutoFE-related configuration
            try:
                # Maximum number of trials
                if 'AutoFEmaxTrialNum' in self.config.get('Features', {}):
                    updated_config['AutoFE']['maxTrialNum'] = self.config['Features']['AutoFEmaxTrialNum']
                    
                logger.debug("Successfully update AutoFE configuration")
                
            except Exception as e:
                logger.error(f"Error updating AutoFE configuration: {str(e)}")
            
            # Set model save path
            try:
                # Determine which task name to use
                current_task_name = task_name if task_name is not None else self.task_name
                
                # Set save path
                if 'Path' in self.config and 'Model' in self.config['Path']:
                    model_path = os.path.join(self.config['Path']['Model'], current_task_name)
                    updated_config['IncrML']['SaveModelPath'] = model_path
                    logger.debug(f"Model save path set to: {model_path}")
                else:
                    logger.warning("Configuration missing model path information")
                    
            except Exception as e:
                logger.error(f"Error setting model save path: {str(e)}")
            
            return updated_config
            
        except Exception as e:
            logger.error(f"Error updating task configuration: {str(e)}")
            # Return original configuration to avoid complete failure
            return task_config

    def load_data(self) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        """
        Load data from all data sources
        
        Load wind tower data, wind fan data, and weather forecast data from the specified paths,
        and perform necessary data processing and merging operations.
        
        Returns:
            Tuple, containing the processed merged data frame and weather data list
        
        Raises:
            ValueError: When necessary data source configuration is missing
        """
        try:
            logger.info('Start loading data...')
            
            # Get necessary parameters from configuration
            time_col = self.config.get('Features', {}).get('TimeIndex', None)
            windtower_paths = self.config.get('DataSource', {}).get('WindTower', None)
            windfans_paths = self.config.get('DataSource', {}).get('WindFans', None)
            weather_paths = self.config.get('DataSource', {}).get('WeatherForecast', None)
            target_name = self.config.get('Features', {}).get('TargetName', None)
            tasks = self.config.get('Task', None)
            
            # Verify if necessary configuration exists
            if not windfans_paths or not windtower_paths or not tasks:
                error_msg = 'Data source not configured, please check the WindTower, WindFans, and Task sections in the configuration file'
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Initialize data containers
            tower_data = []
            fans_data = []
            weather_data = []
            
            # Load wind tower data
            if windtower_paths:
                for path in windtower_paths:
                    try:
                        logger.info(f'Loading wind tower data: {path}')
                        tower_df = pd.read_csv(path)
                        tower_data.append(tower_df)
                        logger.debug(f'Wind tower data sample: {tower_df.tail(2).values.tolist()}')
                    except Exception as e:
                        logger.error(f'Error loading wind tower data: {path}, error: {str(e)}')
            
            # Load wind fan data
            if windfans_paths:
                for path in windfans_paths:
                    try:
                        logger.info(f'Loading wind fan data: {path}')
                        # Use polars to load large CSV files to improve performance
                        fan_df = pl.read_csv(path).to_pandas()
                        fans_data.append(fan_df)
                        logger.debug(f'Wind fan data sample: {fan_df.tail(2).values.tolist()}')
                    except Exception as e:
                        logger.error(f'Error loading wind fan data: {path}, error: {str(e)}')
            
            # Load weather forecast data
            if weather_paths:
                for path in weather_paths:
                    try:
                        logger.info(f'Loading weather data: {path}')
                        weather_df = pd.read_csv(path)
                        # Process weather data
                        processed_weather = self.weather_data_process(weather_df)
                        weather_data.append(processed_weather)
                    except Exception as e:
                        logger.error(f'Error loading and processing weather data: {path}, error: {str(e)}')
            
            # Merge wind tower data
            tower_merged = None
            if tower_data:
                try:
                    tower_merged = pd.concat(tower_data, ignore_index=True)
                    logger.info(f'Merged wind tower data shape: {tower_merged.shape}')
                except Exception as e:
                    logger.error(f'Error merging wind tower data: {str(e)}')
                    if len(tower_data) == 1:
                        tower_merged = tower_data[0]
            
            # Merge wind fan data and merge with wind tower data
            final_df = None
            if fans_data:
                try:
                    fans_merged = pd.concat(fans_data, ignore_index=False)
                    if tower_merged is not None and time_col in tower_merged.columns:
                        # Outer join merge wind tower and wind fan data
                        final_df = tower_merged.merge(fans_merged, on=time_col, how='outer')
                        logger.info(f'Merged wind tower and wind fan data shape: {final_df.shape}')
                    else:
                        final_df = fans_merged
                        logger.warning('No wind tower data or missing time column, only use wind fan data')
                except Exception as e:
                    logger.error(f'Error merging wind fan data: {str(e)}')
                    final_df = tower_merged
            else:
                final_df = tower_merged
            
            # Check if data is empty
            if final_df is None or final_df.empty:
                error_msg = 'Data is empty after loading, please check the data source'
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Standardize target value range
            if target_name in final_df.columns:
                try:
                    target_mean = final_df[target_name].mean()
                    # If the target value is too large, reduce it by 1000
                    if target_mean > 10000:
                        final_df[target_name] = final_df[target_name] / 1000
                        logger.info(f'Target value average ({target_mean}) is too large, divided by 1000')
                    # If the target value is too small, multiply by 1000
                    elif target_mean < 0.01:
                        final_df[target_name] = final_df[target_name] * 1000
                        logger.info(f'Target value average ({target_mean}) is too small, multiplied by 1000')
                except Exception as e:
                    logger.error(f'Error standardizing target value: {str(e)}')
            
            # Standardize time format
            if time_col in final_df.columns:
                try:
                    # Ensure time column format is unified (add time part if only date)
                    final_df[time_col] = final_df[time_col].apply(
                        lambda x: str(x) + ' 00:00:00' if len(str(x)) == 10 else str(x)
                    )
                    final_df[time_col] = pd.to_datetime(final_df[time_col])
                    
                    # Calculate the time range of the data set
                    max_time = final_df[time_col].max()
                    min_time = final_df[time_col].min().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Consider the maximum time of weather data
                    for weather_df in weather_data:
                        if 'GROUP' in weather_df.columns:
                            weather_max_time = pd.to_datetime(weather_df['GROUP']).max()
                            max_time = max(weather_max_time, max_time)
                    
                    # If you need to expand the time range, perform timestamp padding
                    if max_time != final_df[time_col].max():
                        final_df = self.get_timestamps(
                            final_df, 
                            min_time, 
                            max_time.strftime('%Y-%m-%d %H:%M:%S')
                        )
                        logger.info(f'Expand data time range to: {min_time} to {max_time}')
                except Exception as e:
                    logger.error(f'Error processing time column: {str(e)}')
            
            # Clean up memory
            del fans_data, tower_data
            if 'tower_merged' in locals() and tower_merged is not None:
                del tower_merged
            if 'fans_merged' in locals() and 'fans_merged' in locals():
                del fans_merged
            gc.collect()
            
            logger.info('Data loading completed')
            return (final_df, weather_data)
            
        except Exception as e:
            logger.error(f'Error loading data: {str(e)}')
            # Return empty data frame and empty list in case of complete failure
            return pd.DataFrame(), []

    def get_X_y_train(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Get training data and task configuration
        
        Load original data, merge weather data, generate time features, and update task configuration
        
        Returns:
            Tuple, containing the processed training data frame and task configuration
        """
        try:
            logger.info('Start loading training data...')
            
            # Load original data
            df, weather_data = self.load_data()
            
            # Check if data is valid
            if df.empty:
                logger.error('Data is empty')
                return pd.DataFrame(), {}
            
            # Merge weather data
            if isinstance(weather_data, list) and weather_data:
                try:
                    df = self.merge_all_weather_data(df, weather_data)
                    logger.info(f'Weather data merged data shape: {df.shape}')
                except Exception as e:
                    logger.error(f'Error merging weather data: {str(e)}')
            
            # Add time features and get task configuration
            try:
                df, task_config = self.add_time_feature_process(df)
                task_config = self.update_config(task_config)
                logger.info('Successfully generated time features and updated configuration')
            except Exception as e:
                logger.error(f'Error processing time features or updating configuration: {str(e)}')
                return df, {}
            
            logger.info('Training data loading completed')
            return df, task_config
            
        except Exception as e:
            logger.error(f'Error loading training data: {str(e)}')
            return pd.DataFrame(), {}

    def get_X_test(self, predict_time: str, label_save_path: str = '') -> Tuple[pd.DataFrame, Tuple, float, pd.Timestamp, pd.Timestamp]:
        """
        Get test data
        
        Get test data based on the specified prediction time, merge weather data, and calculate the prediction time range
        
        Args:
            predict_time: Prediction time point
            label_save_path: Label save path, default is empty string
            
        Returns:
            Tuple, containing the processed test data frame, task configuration, average power value, prediction start time, and prediction end time
        """
        try:
            logger.info('Start getting test data...')
            
            # Initialize variables
            mean_power = 0
            res = pd.DataFrame()
            label_df = pd.DataFrame()
            
            # Round down the prediction time to the nearest 15 minutes
            try:
                predict_end_time = pd.to_datetime(predict_time).floor('15min')
                logger.debug(f'Prediction end time (rounded down): {predict_end_time}')
            except Exception as e:
                logger.error(f'Error parsing prediction time: {str(e)}')
                return pd.DataFrame(), {}, 0, None, None
            
            # Calculate delay time based on task type
            try:
                lag_time = 0
                
                # Delay processing for hourly task
                if self.task_name.endswith('Hour'):
                    if self.hourlagtime.endswith('min'):
                        lag_time = int(self.hourlagtime[:-3])
                        logger.debug(f'Hour task delay time: {lag_time} minutes')
                    elif self.hourlagtime.endswith('hour'):
                        lag_time = int(self.hourlagtime[:-4]) * 60
                        logger.debug(f'Hour task delay time: {lag_time} minutes (converted from hours)')
                
                # Delay processing for daily task
                if self.task_name.endswith('Day'):
                    if self.daylagtime.endswith('min'):
                        lag_time = int(self.daylagtime[:-3])
                        logger.debug(f'Day task delay time: {lag_time} minutes')
                    elif self.daylagtime.endswith('hour'):
                        lag_time = int(self.daylagtime[:-4]) * 60
                        logger.debug(f'Day task delay time: {lag_time} minutes (converted from hours)')
                
                # Adjust prediction end time based on delay time
                predict_end_time += pd.Timedelta(minutes=lag_time)
                
            except Exception as e:
                logger.error(f'Error calculating delay time: {str(e)}')
            
            # Calculate prediction start time and adjusted end time
            predict_start_time = predict_end_time - pd.Timedelta(minutes=self.n)
            predict_end_time = predict_end_time - pd.Timedelta(minutes=self.ahead_rows * 15)
            
            logger.info(f'Prediction time range: [{predict_start_time}, {predict_end_time}]')
            
            # If waiting time is configured, sleep for the specified time
            if self.time_sleep > 0:
                logger.info(f'Waiting {self.time_sleep} seconds...')
                time.sleep(self.time_sleep)
            
            # Load data
            df, weather_data = self.load_data()
            
            # Check if data is valid
            if df.empty:
                logger.error('Loaded data is empty')
                return pd.DataFrame(), {}, 0, predict_start_time, predict_end_time
            
            # Merge weather data
            if isinstance(weather_data, list) and weather_data:
                try:
                    df = self.merge_all_weather_data(df, weather_data)
                    logger.info(f'Weather data merged data shape: {df.shape}')
                except Exception as e:
                    logger.error(f'Error merging weather data: {str(e)}')
            
            # Process label save path (if provided and is 4-hour prediction task)
            if label_save_path and self.task_name == 'Forecast4Hour':
                try:
                    if os.path.exists(label_save_path):
                        # Read and process label data
                        label_df = pd.read_csv(label_save_path)
                        
                        # Standardize time format
                        label_df['date_time'] = label_df['date_time'].apply(
                            lambda x: str(x) + ' 00:00:00' if len(str(x)) == 10 else str(x)
                        )
                        label_df['date_time'] = pd.to_datetime(label_df['date_time'])
                        logger.info(f'Loaded label data: {label_save_path}, shape: {label_df.shape}')
                        
                        # Extract new data
                        try:
                            max_date = label_df['date_time'].max()
                            # Filter out data newer than the maximum date
                            res = df[[self.time_col, self.target_name]][
                                df[self.time_col] > max_date
                            ].rename(columns={self.target_name: 'power'})
                            logger.info(f'Extracted {len(res)} new data')
                        except Exception as e:
                            logger.error(f'Error processing existing label data: {str(e)}')
                            res = pd.DataFrame()
                    else:
                        logger.info('First save real label file...')
                        try:
                            # Ensure directory exists
                            os.makedirs(os.path.dirname(label_save_path), exist_ok=True)
                            # Extract all time and target value data
                            res = df[[self.time_col, self.target_name]].rename(
                                columns={self.target_name: 'power'}
                            )
                        except Exception as e:
                            logger.error(f'Error preparing to save label data: {str(e)}')
                            res = pd.DataFrame()

                    # If there is new data, save to label file
                    if not res.empty:
                        try:
                            # Add execution time and date time column
                            res['exec_time'] = pd.Timestamp.today()
                            res['date_time'] = res[self.time_col]
                            # Only keep the required columns
                            res = res[['exec_time', 'date_time', 'power']]
                            logger.info(f'Save {len(res)} label data')
                            
                            # Save data (append or create)
                            if os.path.exists(label_save_path):
                                res.to_csv(label_save_path, index=False, mode='a', header=None)
                                logger.info(f'Successfully appended label data to: {label_save_path}')
                            else:
                                res.to_csv(label_save_path, index=False)
                                logger.info(f'Successfully created and saved label data to: {label_save_path}')
                        except Exception as e:
                            logger.error(f'Error saving label data: {str(e)}')
                    else:
                        logger.info('No new label data to save')
                        res = label_df

                    # Calculate power average value
                    try:
                        if 'power' in res.columns and not res.empty:
                            mean_power = res['power'].mean()
                            logger.debug(f'Power average value: {mean_power}')
                    except Exception as e:
                        logger.error(f'Error calculating power average value: {str(e)}')
                except Exception as e:
                    logger.error(f'Error processing label data: {str(e)}')
                    mean_power = df[self.target_name].mean() if self.target_name in df.columns else 0
            else:
                # If it is not a 4-hour prediction task or no label path is provided
                mean_power = df[self.target_name].mean() if self.target_name in df.columns else 0
                
            # Filter out data within the prediction time range
            df = df[(df[self.time_col] >= predict_start_time) & (df[self.time_col] <= predict_end_time)]
            if df.empty:
                logger.error(f'No valid data in the range [{predict_start_time}, {predict_end_time}]')
                return pd.DataFrame(), {}, mean_power, predict_start_time, predict_end_time
                
            # If you need to expand the time range, perform timestamp padding
            df = self.get_timestamps(
                df, 
                start_time=predict_start_time, 
                end_time=predict_end_time
            )
            logger.info(f'Expand data time range to: {predict_start_time} to {predict_end_time}')
            
            # Add time features and update configuration
            df, task_config = self.add_time_feature_process(df)
            task_config = self.update_config(task_config)
            logger.info('Test data retrieval completed')
            
            return df, task_config, mean_power, predict_start_time, predict_end_time
            
        except Exception as e:
            logger.error(f'Error retrieving test data: {str(e)}')
            return pd.DataFrame(), {}, 0, None, None

    def train_task(self):

        logger.info(f'{self.task_name} task start training.')
        if not os.path.exists(self.debug_csv_save_path):
            os.makedirs(self.debug_csv_save_path, exist_ok=True)
        try:
            df, task_config = self.get_X_y_train()
            df.tail(100).to_csv(os.path.join(self.debug_csv_save_path, f"{self.task_name}_train_{str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}.csv"), index=False)
            logger.info(task_config)
            ESPML(task_config, df).fit()
            logger.info(f'{self.task_name} task finish training successfully.')
        except Exception as e:
            logger.error(f'{e}\n{traceback.format_exc()}')

    def predict_fn(self, X_val: pd.DataFrame, task_config: Dict, 
                 history_mean_value: float, 
                 predict_start_time: pd.Timestamp, 
                 predict_end_time: pd.Timestamp) -> pd.DataFrame:
        """
        Execute prediction function
        
        Execute prediction based on input data and model configuration, use historical average value if an error occurs
        
        Args:
            X_val: Input feature data frame
            task_config: Task configuration dictionary
            history_mean_value: Historical average value, used when an error occurs
            predict_start_time: Prediction start time
            predict_end_time: Prediction end time
            
        Returns:
            Data frame containing prediction results
        """
        # Initialize result data frame
        res = pd.DataFrame()
        
        # Check if input data is empty
        if not X_val.empty:
            try:
                # Create basic result data frame, containing base time and prediction time
                res = pd.DataFrame({
                    'base_time': X_val[self.time_col], 
                    'date_time': pd.to_datetime(X_val[self.time_col]) + pd.Timedelta(minutes=self.n)
                })
                
                # Record task configuration and execute prediction
                logger.info(f'Start prediction, task configuration: {task_config}')
                pred = predict(task_config, X_val, save=False)
                
                # Add prediction results to result data frame
                res['power'] = pred.values
                
                logger.info(f'{self.task_name} task prediction completed successfully')
                
            except Exception as e:
                # Record error when prediction fails and use historical average value
                logger.error(f'Prediction failed: {e}\n{traceback.format_exc()}')
                
                if not res.empty:
                    res['power'] = history_mean_value
                    logger.warning(f'Use historical average value {history_mean_value} as prediction result')
        else:
            # Handle empty data case
            logger.error(f'{self.task_name} got empty input data, please check real-time data')
            
            # Create an empty result data frame based on time range
            base_times = pd.date_range(predict_start_time, predict_end_time, freq='15min')
            date_times = pd.date_range(
                predict_start_time + pd.Timedelta(minutes=self.n), 
                predict_end_time + pd.Timedelta(minutes=self.n), 
                freq='15min'
            )
            
            res = pd.DataFrame({
                'base_time': base_times,
                'date_time': date_times
            })
            
            # Fill with historical average value
            res['power'] = history_mean_value
            logger.warning(f'Use historical average value {history_mean_value} to fill empty result')
            
        # Add execution point information and additional time processing
        if not res.empty and 'date_time' in res.columns:
            # Add execution point and current execution time
            res['exec_point'] = str(res['date_time'].min())
            res['exec_time'] = pd.Timestamp.today()
            
            # Format date time column as string format
            res['base_time'] = res['base_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            res['date_time'] = res['date_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Format execution time column (if exists)
            if 'exec_time' in res.columns:
                res['exec_time'] = res['exec_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
        return res

    def predict_task(self, predict_time: Optional[Union[str, datetime]] = None) -> Optional[pd.DataFrame]:
        """
        Execute prediction task
        
        Load model and use it for prediction, save results to specified path
        
        Args:
            predict_time: The time point to predict, can be a string or datetime object, if None use current time
            
        Returns:
            Prediction result data frame, return None if prediction fails
        """
        try:
            # Record start prediction
            logger.info(f'{self.task_name} task start prediction')
            
            # Process prediction time
            if predict_time is None:
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                logger.debug(f'Use current time: {now}')
            else:
                try:
                    # If input is string, convert to datetime
                    if isinstance(predict_time, str):
                        now = pd.to_datetime(predict_time)
                    else:
                        now = predict_time
                    logger.debug(f'Use specified prediction time: {now}')
                except Exception as e:
                    logger.error(f'Cannot parse prediction time: {str(e)}')
                    return None
            
            # Ensure debug file save path exists
            if not os.path.exists(self.debug_csv_save_path):
                os.makedirs(self.debug_csv_save_path, exist_ok=True)
                logger.debug(f'Create debug file save path: {self.debug_csv_save_path}')
            
            # Check model path and get the latest model
            if not os.path.exists(self.model_path):
                logger.error(f'{self.task_name} model path does not exist: {self.model_path}')
                return None
                
            try:
                # Get all model sequence numbers
                model_dirs = [d for d in os.listdir(self.model_path) if d.isdigit()]
                if not model_dirs:
                    logger.error(f'{self.task_name} model path does not have sequence number folder')
                    return None
                    
                # Get the maximum sequence number
                seq = max([int(num) for num in model_dirs])

                # Find the valid model
                while not os.path.exists(os.path.join(self.model_path, str(seq), 'automl', 'val_res')) and seq > 0:
                    logger.warning(f'Model {seq} validation result does not exist, try previous model')
                    seq -= 1
                
                # Check if there is a valid model
                if seq <= 0:
                    logger.error(f'{self.task_name} did not find valid model')
                    return None
                    
                logger.info(f'{self.task_name} use model with sequence {seq} for prediction')
                
            except Exception as e:
                logger.error(f'Error checking model: {str(e)}')
                return None
            
            # Get test data and configuration
            X_val, task_config, history_mean_value, predict_start_time, predict_end_time = self.get_X_test(now, self.label_save_path)
            
            # Save part of test data for debugging
            if isinstance(X_val, pd.DataFrame) and not X_val.empty:
                debug_file_path = os.path.join(
                    self.debug_csv_save_path, 
                    f"{self.task_name}_predict_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                )
                # Only save the first 50 rows to reduce file size
                X_val.iloc[:50, :].to_csv(debug_file_path, index=False)
                logger.debug(f'Save debug data to: {debug_file_path}')
            
            # Execute prediction
            res = self.predict_fn(X_val, task_config, history_mean_value, predict_start_time, predict_end_time)
            
            # Save prediction results
            if not res.empty:
                try:
                    # Determine save method based on task type
                    if self.task_name == 'Forecast4Hour':
                        if os.path.exists(self.real_predict_save_path):
                            # Append save
                            res.to_csv(self.real_predict_save_path, index=False, mode='a', header=None)
                            logger.info(f'Save prediction results to: {self.real_predict_save_path}')
                        else:
                            # Create new file save
                            res.to_csv(self.real_predict_save_path, index=False)
                            logger.info(f'Save prediction results to: {self.real_predict_save_path}')
                    else:
                        # Other task types directly overwrite save
                        res.to_csv(self.real_predict_save_path, index=False)
                        logger.info(f'Save prediction results to: {self.real_predict_save_path}')
                except Exception as e:
                    logger.error(f'Error saving prediction results: {str(e)}')
            
            return res
            
        except Exception as e:
            logger.error(f'{self.task_name} prediction task error: {str(e)}\n{traceback.format_exc()}')
            return None

    def backtracking(self, start_time: Optional[str] = None, end_time: Optional[str] = None, only_predict: bool = False) -> None:
        """
        Execute backtracking test, train model and evaluate historical performance
        
        This method trains models and evaluates their performance on historical data for each day in the specified time range
        
        Args:
            start_time: Backtracking start date, format: 'YYYY-MM-DD'
            end_time: Backtracking end date, format: 'YYYY-MM-DD'
            only_predict: Whether to only execute prediction without training models
        """
        # Initialize logger
        init_logger(self.config, self.task_name, 'backtracking')
        
        # Initialize metadata information dictionary
        meta_info = {
            'training_type': [], 
            'train_end_date': [], 
            'model_id': [], 
            'exec_time': [], 
            'state': []
        }
        
        # Ensure prediction result save path exists
        if not os.path.exists(self.predict_save_path):
            os.makedirs(self.predict_save_path)
            
        # Set metadata save path
        meta_save_path = os.path.join(self.predict_save_path, 'meta.csv')
        
        # Load data and weather data
        logger.info('Start loading data for backtracking test...')
        df, weather_data = self.load_data()
        
        # Merge weather data
        if isinstance(weather_data, list) and weather_data:
            try:
                df = self.merge_all_weather_data(df, weather_data)
                logger.info(f'Weather data merged data shape: {df.shape}')
            except Exception as e:
                logger.error(f'Error merging weather data: {str(e)}')
                
        # Add time features and get task configuration
        task_df, task_config = self.add_time_feature_process(df)
        
        # Process each training time point
        for i, train_time in enumerate(self.tasks[self.task_name]['TrainTime']):
            try:
                # Calculate time range
                if start_time is None or end_time is None:
                    logger.error('Must provide start and end time')
                    continue
                    
                begining_time = f'{start_time} {train_time}'
                finishing_time = f'{end_time} {train_time}'
                logger.info(f'Process time range: {begining_time} to {finishing_time}')
                
                # Create date range
                try:
                    date_range = pd.date_range(start=begining_time, end=finishing_time, freq='D')
                    if date_range.empty:
                        logger.warning(f'Generated date range is empty, please check time format: {begining_time} - {finishing_time}')
                        continue
                except Exception as e:
                    logger.error(f'Error creating date range: {str(e)}')
                    continue
                    
                # Process each day for modeling and prediction
                for date in date_range:
                    try:
                        logger.info(f'Start processing date: {date}')
                        
                        # Set prediction result save path
                        if self.task_name == 'Forecast4Hour':
                            # 4 hour prediction task uses fixed name
                            backtrack_predict_save_path = os.path.join(self.predict_save_path, f'{self.task_name}.csv')
                        else:
                            # Other task types use date and task name combination
                            backtrack_predict_save_path = os.path.join(
                                self.predict_save_path, 
                                f"backtrack_{date.strftime('%Y%m%d_%H%M')}_{self.task_name}.csv"
                            )
                    
                        # Set task result path and model sequence number
                        task_result_path = os.path.join(self.config['Path']['Model'], f'{self.task_name}_backtrack')
                        
                        # Determine new model sequence number
                        try:
                            if not os.path.exists(task_result_path):
                                os.makedirs(task_result_path, exist_ok=True)
                                seq = 1
                            elif len(os.listdir(task_result_path)) <= 0:
                                seq = 1
                            else:
                                # Find the current maximum sequence number and add 1
                                model_dirs = [d for d in os.listdir(task_result_path) if d.isdigit()]
                                if not model_dirs:
                                    seq = 1
                                else:
                                    seq = max([int(num) for num in model_dirs]) + 1
                        except Exception as e:
                            logger.error(f'Error determining model sequence number: {str(e)}')
                            seq = 1

                        # If the sequence number exceeds 2, clean the old models to save space
                        if seq > 2:
                            try:
                                rm_num = seq - 2  # Only keep the latest 2 models
                                for ii in range(1, rm_num + 1):
                                    old_model_path = os.path.join(task_result_path, str(ii))
                                    if os.path.exists(old_model_path):
                                        logger.info(f'Clean old model: {old_model_path}')
                                        shutil.rmtree(old_model_path, ignore_errors=True)
                            except Exception as e:
                                logger.warning(f'Error cleaning old models: {str(e)}')
                        
                        # Update metadata information
                        meta_info['training_type'].append(self.task_name)
                        meta_info['train_end_date'].append(date)
                        meta_info['model_id'].append(seq)
                        meta_info['exec_time'].append(pd.Timestamp.today())
                        
                        # Start training task
                        logger.info(f'{self.task_name} {date} training task start')
                        
                        # Prepare training data
                        cutoff_time = date - pd.Timedelta(minutes=self.n)
                        temp_df = task_df[task_df[self.time_col] < cutoff_time]
                        
                        # If training data is empty, skip current loop
                        if temp_df.empty:
                            logger.warning(f'Training data for date {date} is empty')
                            meta_info['state'].append('no_data')
                            continue
                            
                        # Update task configuration
                        updated_config = self.update_config(task_config, f'{self.task_name}_backtrack')
                        
                        # Train model and save metadata
                        try:
                            # If not only predict mode, execute training
                            if not only_predict:
                                # Remove empty label data for training
                                train_data = temp_df[~temp_df['label'].isnull()]
                                
                                if train_data.empty:
                                    logger.warning(f'Training data for date {date} is empty')
                                    meta_info['state'].append('no_valid_data')
                                else:
                                    # Execute model training
                                    ESPML(updated_config, train_data).fit()
                                    logger.info(f'{date} training task completed successfully')
                                    meta_info['state'].append('success')
                            else:
                                # Only predict mode
                                logger.info(f'Skip training, only predict (use sequence number: {seq})')
                                meta_info['state'].append('predict_only')
                                
                        except Exception as e:
                            # Record training error
                            logger.error(f'Training error: {str(e)}\n{traceback.format_exc()}')
                            meta_info['state'].append('failed')
                        
                        # Whether training is successful, save metadata
                        finally:
                            try:
                                meta_df = pd.DataFrame(meta_info)
                                
                                if os.path.exists(meta_save_path):
                                    # Append mode save
                                    meta_df.to_csv(meta_save_path, index=False, mode='a', header=False)
                                    logger.debug(f'Metadata has been appended to: {meta_save_path}')
                                else:
                                    # New mode save
                                    meta_df.to_csv(meta_save_path, index=False)
                                    logger.debug(f'Metadata has been saved to: {meta_save_path}')
                            except Exception as e:
                                logger.error(f'Error saving metadata: {str(e)}')
                    except Exception as e:
                        logger.error(f'Error processing date {date}: {str(e)}\n{traceback.format_exc()}')
            except Exception as e:
                logger.error(f'Error processing training time {train_time}: {str(e)}\n{traceback.format_exc()}')
                continue
            
            # Helper function: perform single prediction and process results
            def predict_and_process_result(
                predict_start_time: pd.Timestamp,
                predict_end_time: pd.Timestamp,
                y_start_time: pd.Timestamp,
                y_end_time: pd.Timestamp
            ) -> pd.DataFrame:
                """
                Perform single prediction and process results
                
                Args:
                    predict_start_time: Prediction start time
                    predict_end_time: Prediction end time
                    y_start_time: Label start time
                    y_end_time: Label end time
                    
                Returns:
                    Processed result data frame
                """
                try:
                    logger.debug(f'Prepare prediction: [{predict_start_time}, {predict_end_time}] -> label: [{y_start_time}, {y_end_time}]')
                    
                    # Extract test data
                    X_test = task_df[
                        (task_df[self.time_col] >= predict_start_time) & 
                        (task_df[self.time_col] <= predict_end_time)
                    ]
                    logger.debug(f'Extract test data: {X_test.shape}')
                    
                    # Extract label data
                    y_test = task_df[
                        (task_df[self.time_col] >= y_start_time) & 
                        (task_df[self.time_col] <= y_end_time)
                    ][[self.time_col, self.target_name]]
                    logger.debug(f'Extract label data: {y_test.shape}')
                    
                    # Process empty label data
                    if y_test.empty:
                        logger.warning('Label data is empty, create an empty label data frame')
                        time_range = pd.date_range(y_start_time, y_end_time, freq='15min')
                        y_test = pd.DataFrame({
                            self.time_col: time_range
                        })
                        y_test[self.target_name] = 0
                    
                    # Initialize result data frame
                    result_df = pd.DataFrame()
                    
                    # Process test data and predict
                    if not X_test.empty:
                        try:
                            # Add timestamps to ensure data points for each 15-minute interval
                            X_test = self.get_timestamps(
                                X_test, 
                                start_time=predict_start_time, 
                                end_time=predict_end_time
                            )
                            logger.debug(f'Test data shape after adding timestamps: {X_test.shape}')
                            
                            # Execute prediction
                            y_pred = predict(updated_config, X_test, save=False)
                            logger.debug(f'Prediction result length: {len(y_pred)}')
                            
                            # Create result data frame
                            date_range = pd.date_range(y_start_time, y_end_time, freq='15min')
                            result_df = pd.DataFrame({
                                'date_time': date_range,
                                'y_pred': y_pred
                            })
                            logger.debug(f'Create result data frame: {result_df.shape}')
                            
                        except Exception as e:
                            logger.error(f'Prediction process error: {str(e)}\n{traceback.format_exc()}')
                            # If prediction fails, use historical average
                            date_range = pd.date_range(y_start_time, y_end_time, freq='15min')
                            result_df = pd.DataFrame({'date_time': date_range})
                            history = task_df[task_df[self.time_col] < y_start_time][self.target_name]
                            hist_mean = history.mean() if not history.empty else 0
                            result_df['y_pred'] = hist_mean
                            logger.warning(f'Use historical average ({hist_mean}) as prediction result')
                            
                    else:
                        # If there is no test data, use historical average
                        logger.warning(f'No data for prediction time range: [{predict_start_time}, {predict_end_time}]')
                        date_range = pd.date_range(y_start_time, y_end_time, freq='15min')
                        result_df = pd.DataFrame({'date_time': date_range})
                        
                        # Calculate historical average
                        history = task_df[task_df[self.time_col] < y_start_time][self.target_name]
                        hist_mean = history.mean() if not history.empty else 0
                        result_df['y_pred'] = hist_mean
                        logger.warning(f'Use historical average ({hist_mean}) as prediction result')
                        
                    # Merge prediction results and labels
                    try:
                        result_df = result_df.merge(
                            y_test, 
                            left_on='date_time', 
                            right_on=self.time_col, 
                            how='left'
                        )
                        logger.debug(f'Merge prediction results and labels: {result_df.shape}')
                        
                        # Clean and format results
                        if self.time_col != 'date_time' and self.time_col in result_df.columns:
                            result_df.drop(columns=[self.time_col], inplace=True)
                            
                        # Rename label column
                        result_df.rename(columns={self.target_name: 'y_true'}, inplace=True)
                    except Exception as e:
                        logger.error(f'Error merging prediction results and labels: {str(e)}')
                    
                    # Add metadata columns
                    try:
                        result_df['exec_point'] = str(result_df['date_time'].min())
                        result_df['exec_time'] = pd.Timestamp.today()
                        result_df['model_id'] = seq
                        
                        # Format date column
                        result_df['date_time'] = result_df['date_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                        result_df['exec_time'] = result_df['exec_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    except Exception as e:
                        logger.error(f'Error adding metadata columns: {str(e)}')
                    
                    logger.debug(f'Prediction processing completed, result shape: {result_df.shape}')
                    return result_df
                except Exception as e:
                    logger.error(f'Prediction processing error: {str(e)}\n{traceback.format_exc()}')
                    # If the entire function fails, return an empty data frame
                    date_range = pd.date_range(y_start_time, y_end_time, freq='15min')
                    empty_df = pd.DataFrame({
                        'date_time': date_range,
                        'y_pred': 0,
                        'y_true': 0,
                        'exec_point': str(y_start_time),
                        'exec_time': pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S'),
                        'model_id': seq
                    })
                    return empty_df
                
            try:
                # Execute different prediction logic based on task type
                if self.task_name == 'Forecast4Hour':
                    logger.info(f'Execute 4-hour prediction task logic, date: {date}')
                    
                    # Initialize prediction time range
                    predict_start_time = date - pd.Timedelta(hours=4)
                    predict_end_time = date
                    
                    # Record start prediction time range
                    logger.info(f'Initial prediction time range: [{predict_start_time}, {predict_end_time}]')
                    
                    # Loop through 16 15-minute intervals
                    for interval in range(16):
                        try:
                            # Move forward by 15 minutes
                            predict_start_time += pd.Timedelta(minutes=15)
                            predict_end_time += pd.Timedelta(minutes=15)
                            
                            # Calculate label time range (prediction time + 4 hours)
                            y_start_time = predict_start_time + pd.Timedelta(hours=4)
                            y_end_time = predict_end_time + pd.Timedelta(hours=4)
                            
                            logger.info(f'Process interval {interval+1}/16: prediction range [{predict_start_time}, {predict_end_time}] -> label range [{y_start_time}, {y_end_time}]')
                            
                            # Execute prediction and process results
                            result_df = predict_and_process_result(
                                predict_start_time,
                                predict_end_time,
                                y_start_time,
                                y_end_time
                            )
                            
                            # Save prediction results
                            try:
                                if os.path.exists(backtrack_predict_save_path):
                                    # If the file exists, use append mode
                                    result_df.to_csv(
                                        backtrack_predict_save_path, 
                                        index=False, 
                                        mode='a', 
                                        header=None
                                    )
                                    logger.debug(f'Prediction results have been appended to: {backtrack_predict_save_path}')
                                else:
                                    # If it is the first time to save, create a new file
                                    result_df.to_csv(backtrack_predict_save_path, index=False)
                                    logger.info(f'Initial prediction results have been saved to: {backtrack_predict_save_path}')
                            except Exception as e:
                                logger.error(f'Error saving prediction results: {str(e)}')
                                
                        except Exception as e:
                            logger.error(f'Error processing interval {interval+1}/16: {str(e)}')
                            continue  # Continue processing the next time interval
                            
                    logger.info(f'Completed all 16 time intervals')
                            
                else:
                    # Other prediction tasks: single time point prediction
                    logger.info(f'Execute single point prediction task logic, date: {date}')
                    
                    try:
                        # Get submission time from configuration
                        subtime = self.tasks[self.task_name]['SubmitTime'][i]
                        
                        # Calculate prediction time range
                        date_str = str(date)[:10]  # Get the first 10 characters (date part)
                        predict_end_time = pd.to_datetime(f'{date_str} {subtime}')
                        predict_start_time = predict_end_time - pd.Timedelta(minutes=self.n)
                        
                        # Calculate label time range
                        y_start_time = predict_start_time + pd.Timedelta(minutes=self.n)
                        y_end_time = predict_end_time + pd.Timedelta(minutes=self.n)
                        
                        logger.info(f'Prediction range: [{predict_start_time}, {predict_end_time}] -> label range: [{y_start_time}, {y_end_time}]')
                        
                        # Execute prediction and process results
                        result_df = predict_and_process_result(
                            predict_start_time,
                            predict_end_time,
                            y_start_time,
                            y_end_time
                        )
                        
                        # Save prediction results (overwrite mode)
                        try:
                            result_df.to_csv(backtrack_predict_save_path, index=False)
                            logger.info(f'Prediction results have been saved to: {backtrack_predict_save_path}')
                        except Exception as e:
                            logger.error(f'Error saving prediction results: {str(e)}')
                            
                    except Exception as e:
                        logger.error(f'Error processing single point prediction: {str(e)}\n{traceback.format_exc()}')
                
                # Record successful prediction log
                logger.info(f'{date} training task prediction successful')
                
            except Exception as e:
                logger.error(f'{date} training task prediction failed: {str(e)}\n{traceback.format_exc()}')
        
        # Clean up log handler
        logger.info('Backtracking test completed')
        logger.handlers = []