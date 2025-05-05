"""
Wind power incremental learning utility module.

Provides utility functions for wind power data processing, model training and prediction.
"""

import os
import gc
import time
import traceback
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
from functools import partial
from ctypes import *

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import r2_score
from schedule import every, repeat, run_pending

from module.utils.ml import ESPML
from module.utils.test_file import predict
from module.utils.yaml_parser import YamlParser

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_timestamps(
    df: pd.DataFrame, 
    time_col: str, 
    interval: str = '15min', 
    start_time: Optional[pd.Timestamp] = None, 
    end_time: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Ensure data has points every 15 minutes.
    
    Resample time series data to ensure there is data for each time interval within the specified time range.
    Missing time points are filled using forward fill method.
    
    Args:
        df: Input dataframe
        time_col: Time column name
        interval: Time interval, default is '15min'
        start_time: Start time, default is the earliest time in the data
        end_time: End time, default is the latest time in the data
        
    Returns:
        Resampled dataframe
    """
    # Ensure time column is datetime type
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Set time index and group by time to get mean values
    df.set_index(time_col, inplace=True)
    df = df.groupby(time_col).mean()
    
    # Determine time range
    if start_time is None:
        start_time = df.index.min()
    if end_time is None:
        end_time = df.index.max()
        
    # Create complete time index
    full_index = pd.date_range(
        start=start_time, 
        end=end_time, 
        freq=interval, 
        name=time_col
    )
    
    logger.info(
        "Ensuring data has points every %s in time range [%s, %s]", 
        interval, str(start_time), str(end_time)
    )
    
    # Sort by time and rebuild index, using forward fill to handle missing values
    df.sort_index(inplace=True)
    df = df.reindex(full_index, method='ffill')
    df.reset_index(inplace=True)
    
    return df


def next_eight_pm(timestamp: pd.Series, n: int) -> pd.Series:
    """Calculate the nearest weather forecast time point (20:00).
    
    Based on the current time and prediction lead time n, determine which day's 20:00 weather forecast should be used.
    
    Args:
        timestamp: Timestamp series
        n: Prediction lead time in minutes
        
    Returns:
        Corresponding 20:00 weather forecast time points
    """
    # Calculate today's 20:00 time point
    eight_pm_today = (
        timestamp + 
        pd.to_timedelta(20 - timestamp.dt.hour, unit='h') - 
        pd.to_timedelta(timestamp.dt.minute, unit='m') - 
        pd.to_timedelta(timestamp.dt.second, unit='s') - 
        pd.to_timedelta(timestamp.dt.microsecond, unit='us')
    )
    
    # Adjust which day's forecast to use based on prediction lead time
    if 4 <= n // 60 < 24:
        # If current time is earlier than 16:00, use the previous day's 20:00 forecast
        eight_pm_today[timestamp.dt.hour < 16] -= pd.Timedelta(days=1)
        return eight_pm_today
    
    # If current time is later than or equal to 20:00, use the next day's 20:00 forecast
    eight_pm_today[timestamp.dt.hour >= 20] += pd.Timedelta(days=1)
    return eight_pm_today


def init_logger(config: Dict[str, Any], task_name: str, logtype: str) -> None:
    """Initialize logger configuration.
    
    Set up log handlers for specific tasks and log types, including file and console handlers.
    
    Args:
        config: Configuration dictionary
        task_name: Task name
        logtype: Log type (e.g., 'train', 'predict', 'backtracking', etc.)
    """
    # Remove existing file handlers
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            
    # Set log file path
    log_file_path = os.path.join(
        config['Path']['LogFile'], 
        f'{task_name}_{logtype}.log'
    )
    
    # Ensure log directory exists
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    # Configure file log handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Configure console log handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    logger.info("Initialized %s log for task %s", logtype, task_name)

def weather_data_process(config: Dict[str, Any], df: pd.DataFrame, interval: str = '15min') -> pd.DataFrame:
    """Process weather data, resampling and interpolating each forecast group.
    
    Args:
        config: Configuration dictionary containing column names for weather data
        df: Weather dataframe
        interval: Resampling interval, default is '15min'
        
    Returns:
        Processed weather dataframe
    """
    logger.info("Starting to process weather data, data shape: %s", df.shape)
    
    try:
        # Get column names from configuration
        sc_date = config['Feature']['WeatherSCDateIndex']
        sc_time = config['Feature']['WeatherSCTimeIndex']
        pre_time = config['Feature']['WeatherPreTimeIndex']
        
        # Create forecast group identifier
        df['GROUP'] = df[sc_date].astype(str) + ' ' + df[sc_time].astype(str)
        
        # Convert forecast time to timestamp format
        df[pre_time] = pd.to_datetime(df[pre_time])
        
        # Process each forecast group separately
        resampled_data = []
        group_count = df['GROUP'].nunique()
        logger.debug("Total %d forecast groups to process", group_count)
        
        for group, group_data in df.groupby('GROUP'):
            # Set forecast time as index
            group_data = group_data.set_index(pre_time)
            
            # Only select numeric columns for interpolation
            group_data_numeric = group_data.select_dtypes(include='number')
            
            # Resample at specified interval and apply linear interpolation
            group_data_resampled = group_data_numeric.resample(interval).interpolate(method='linear')
            
            # Add group identifier
            group_data_resampled['GROUP'] = group
            
            # Add to result list
            resampled_data.append(group_data_resampled.reset_index())
            
        # Merge all resampled group data
        result = pd.concat(resampled_data, ignore_index=True)
        logger.info("Weather data processing completed, result shape: %s", result.shape)
        
        return result
        
    except Exception as exc:
        logger.error("Error processing weather data: %s\n%s", exc, traceback.format_exc())
        raise


def weather_data_merge(
    config: Dict[str, Any], 
    df: pd.DataFrame, 
    weather_data: pd.DataFrame, 
    n: int
) -> pd.DataFrame:
    """Merge weather data with the main dataframe.
    
    Based on prediction time and weather forecast time, merge appropriate weather data with the main dataframe.
    
    Args:
        config: Configuration dictionary
        df: Main dataframe
        weather_data: Weather dataframe
        n: Prediction lead time in minutes
        
    Returns:
        Merged dataframe
    """
    logger.info("Starting to merge weather data, main data shape: %s, weather data shape: %s", 
              df.shape, weather_data.shape)
    
    try:
        # Get column names from configuration
        pre_time = config['Feature']['WeatherPreTimeIndex']
        time_col = config['Feature']['TimeIndex']
        
        # Calculate weather forecast time and prediction time
        df['next_8pm'] = next_eight_pm(df[time_col], n)
        df['pre_time'] = pd.to_datetime(df[time_col]) + pd.Timedelta(minutes=n)
        
        # Process the GROUP column in weather data
        weather_data['GROUP'] = pd.to_datetime(weather_data['GROUP'])
        weather_date_min = weather_data[pre_time].min()
        
        # Filter matching weather data
        weather = []
        unique_dates = df['next_8pm'].drop_duplicates()
        logger.debug("Need to process %d unique dates", len(unique_dates))
        
        for date in unique_dates:
            # Skip dates earlier than weather data range
            if date < weather_date_min:
                logger.debug("Skipping date %s (earlier than weather data range)", date)
                continue
                
            # Select weather data for current date
            temp1 = weather_data[weather_data['GROUP'] == date]
            
            # Further filter matching prediction times
            matching_pre_times = df['pre_time'][df['next_8pm'] == date]
            temp = temp1[temp1[pre_time].isin(matching_pre_times)]
            
            if temp.empty:
                logger.debug("Date %s has no matching weather data", date)
                continue
                
            weather.append(temp)
            
        # If no matching weather data, return original dataframe
        if not weather:
            logger.warning("No matching weather data found, returning original dataframe")
            df.drop(columns=['pre_time', 'GROUP', 'next_8pm'], inplace=True)
            return df
            
        # Merge and process weather data
        merged_weather = pd.concat(weather, ignore_index=True)
        processed_weather = merged_weather.groupby(pre_time).mean().sort_index()
        
        # Merge weather data with main dataframe
        result_df = df.merge(processed_weather, left_on='pre_time', right_index=True, how='left')
        
        # Clean up temporary columns
        result_df.drop(columns=['pre_time', 'GROUP', 'next_8pm'], inplace=True)
        
        logger.info("Weather data merging completed, result shape: %s", result_df.shape)
        return result_df
        
    except Exception as exc:
        logger.error("Error merging weather data: %s\n%s", exc, traceback.format_exc())
        raise


def add_time_feature_process(
    df: pd.DataFrame, 
    time_col: str, 
    target_name: str, 
    config_pth: str, 
    interval: int
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Add time features and process configuration.
    
    Add time-related features to the dataframe, create label column, and update configuration information.
    
    Args:
        df: Input dataframe
        time_col: Time column name
        target_name: Target variable name
        config_pth: Configuration file path
        interval: Time interval
        
    Returns:
        Tuple (processed dataframe, updated configuration)
    """
    logger.info("Starting to add time features and process configuration, data shape: %s", df.shape)
    
    try:
        # Initialize feature list
        add_features = []
        
        # Get time attributes from time column
        tmp_dt = df[time_col].dt
        
        # Create time feature dictionary
        new_columns_dict = {
            f'year_{time_col}': tmp_dt.year, 
            f'month_{time_col}': tmp_dt.month, 
            f'day_{time_col}': tmp_dt.day, 
            f'hour_{time_col}': tmp_dt.hour, 
            f'minute_{time_col}': tmp_dt.minute, 
            f'second_{time_col}': tmp_dt.second, 
            f'dayofweek_{time_col}': tmp_dt.dayofweek, 
            f'dayofyear_{time_col}': tmp_dt.dayofyear, 
            f'quarter_{time_col}': tmp_dt.quarter
        }
        
        # Add time features to dataframe
        for key, value in new_columns_dict.items():
            # Check number of unique values in feature
            if value.nunique(dropna=False) >= 2:
                add_features.append(key)
                df[key] = value
                logger.debug("Added time feature: %s", key)
            else:
                logger.debug("Skipping time feature %s (insufficient unique values)", key)
        
        # Create label column (shifted target variable)
        logger.debug("Creating shifted target variable as label column, shift interval: %d", interval)
        df['label'] = df[target_name].shift(-interval)
        
        # Parse configuration file
        logger.debug("Parsing configuration file: %s", config_pth)
        task_config = YamlParser.parse(config_pth)
        
        # Update feature lists in configuration
        if 'Feature' in task_config:
            # Add time features to feature list
            if 'FeatureName' in task_config['Feature']:
                task_config['Feature']['FeatureName'].extend(add_features)
                logger.debug("Added %d time features to feature list", len(add_features))
            
            # Add time features to categorical feature list
            if 'CategoricalFeature' in task_config['Feature']:
                task_config['Feature']['CategoricalFeature'].extend(add_features)
                logger.debug("Added time features to categorical feature list")
            
            # Set target variable name to 'label'
            task_config['Feature']['TargetName'] = 'label'
            logger.debug("Set target variable name to 'label'")
        
        logger.info("Time features and configuration processing completed, added %d features", len(add_features))
        return df, task_config
        
    except Exception as exc:
        logger.error("Error adding time features and processing configuration: %s\n%s", exc, traceback.format_exc())
        raise
    return (df, task_config)

def update_config(config: Dict[str, Any], task_config: Dict[str, Any], task_name: str) -> Dict[str, Any]:
    """Update task-specific configuration from global configuration.
    
    Copy relevant settings from the global configuration to the task-specific configuration.
    
    Args:
        config: Global configuration dictionary
        task_config: Task-specific configuration dictionary
        task_name: Task name
        
    Returns:
        Updated task configuration dictionary
    """
    logger.debug("Updating configuration for task %s", task_name)
    
    try:
        # Update feature-related configuration
        task_config['Feature']['FeatureName'] = config['Feature']['FeatureName']
        task_config['Feature']['CategoricalFeature'] = config['Feature']['CategoricalFeature']
        task_config['Feature']['TimeIndex'] = config['Feature']['TimeIndex']
        task_config['Feature']['IgnoreFeature'] = config['Feature']['IgnoreFeature']
        
        # Update AutoML-related configuration
        task_config['AutoML']['TimeBudget'] = config['Feature']['AutoMLTimeBudget']
        task_config['AutoFE']['maxTrialNum'] = config['Feature']['AutoFEmaxTrialNum']
        
        # Set model save path
        task_config['IncrML']['SaveModelPath'] = os.path.join(
            config['Path']['Model'], 
            task_name
        )
        
        logger.debug("Configuration updated successfully")
        return task_config
        
    except KeyError as exc:
        logger.error("Error updating configuration, missing key: %s\n%s", exc, traceback.format_exc())
        raise


def load_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, Union[pd.DataFrame, List]]:  
    """Load wind tower, turbine and weather data.
    
    Load data from sources specified in the configuration and perform necessary processing.
    
    Args:
        config: Configuration dictionary containing data source paths
        
    Returns:
        Tuple (merged main dataframe, weather dataframe or empty list)
        
    Raises:
        AssertionError: When data source configuration is missing
    """
    logger.info('Starting to load data...')
    
    # Get necessary parameters from configuration
    time_col = config.get('Feature', {}).get('TimeIndex')
    windtower_paths = config.get('DataSource', {}).get('WindTower', [])
    windfans_paths = config.get('DataSource', {}).get('WindFans', [])
    weather_paths = config.get('DataSource', {}).get('WeatherForecast', [])
    tasks = config.get('Task', {})
    
    # Validate data source configuration
    assert windtower_paths and windfans_paths and tasks, 'Data sources not configured'
    
    # Initialize data containers
    tower_data = []
    fans_data = []
    weather_data = []
    
    try:
        # Load wind tower data
        for path in windtower_paths:
            logger.info('Loading wind tower data: %s', path)
            tower_df = pd.read_csv(path)
            logger.debug('Wind tower data sample: %s', tower_df.tail(2).values.tolist())
            tower_data.append(tower_df)
        
        # Load wind turbine data
        for path in windfans_paths:
            logger.info('Loading wind turbine data: %s', path)
            # Use polars to load large files, then convert to pandas
            fan_df = pl.read_csv(path).to_pandas()
            logger.debug('Wind turbine data sample: %s', fan_df.tail(2).values.tolist())
            fans_data.append(fan_df)
        
        # Load weather data
        for path in weather_paths:
            logger.info('Loading weather data: %s', path)
            weather_df = pd.read_csv(path)
            # Process weather data
            processed_weather = weather_data_process(config, weather_df)
            weather_data.append(processed_weather)
        
        # Process wind tower data
        if tower_data:
            # Merge all wind tower data
            tower_df = pd.concat(tower_data, ignore_index=True)
            # Ensure time series is complete
            tower_df = get_timestamps(tower_df, time_col)
            logger.info('Wind tower data processing completed, shape: %s', tower_df.shape)
        else:
            tower_df = pd.DataFrame()
        
        # Process wind turbine data and merge with tower data
        if fans_data:
            # Merge all wind turbine data
            fans_df = pd.concat(fans_data, ignore_index=False)
            # Ensure time series is complete
            fans_df = get_timestamps(fans_df, time_col)
            logger.info('Wind turbine data processing completed, shape: %s', fans_df.shape)
            
            # Merge wind tower and turbine data
            main_df = tower_df.merge(fans_df, on=time_col, how='left')
            logger.info('Wind tower and turbine data merging completed, shape: %s', main_df.shape)
        else:
            main_df = tower_df
        
        # Process weather data
        if weather_data:
            # Merge all weather data
            merged_weather = pd.concat(weather_data, ignore_index=True)
            
            # Remove columns that are all zeros
            zero_cols = []
            for col in merged_weather.columns:
                if (merged_weather[col] == 0).all():
                    zero_cols.append(col)
            
            if zero_cols:
                merged_weather.drop(columns=zero_cols, inplace=True)
                logger.debug('Removed %d columns that were all zeros', len(zero_cols))
                
            logger.info('Weather data processing completed, shape: %s', merged_weather.shape)
        else:
            merged_weather = []
        
        # Clean up memory
        del fans_data, tower_data
        if 'fan_df' in locals():
            del fan_df
        if 'tower_df' in locals():
            del tower_df
        if 'fans_df' in locals():
            del fans_df
        gc.collect()
        
        logger.info('Data loading completed, main data shape: %s', main_df.shape)
        return main_df, merged_weather
        
    except Exception as exc:
        logger.error('Error loading data: %s\n%s', exc, traceback.format_exc())
        raise

def get_X_y_train(config: Dict[str, Any], taskname: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Get training data and task configuration.
    
    Load and process training data, merge weather data, add time features, and update task configuration.
    
    Args:
        config: Global configuration dictionary
        taskname: Task name
        
    Returns:
        Tuple (processed training dataframe, updated task configuration)
    """
    logger.info('Starting to get training data...')
    
    try:
        # Extract necessary parameters from configuration
        ahead_rows = config.get('Feature', {}).get('AheadRows', 0)
        target_name = config.get('Feature', {}).get('TargetName')
        time_col = config.get('Feature', {}).get('TimeIndex')
        tasks = config.get('Task', {})
        
        # Validate necessary parameters
        if not target_name or not time_col or not tasks or taskname not in tasks:
            logger.error('Configuration missing necessary parameters: target_name=%s, time_col=%s, taskname=%s', 
                      target_name, time_col, taskname)
            raise ValueError('Configuration missing necessary parameters')
            
        # Calculate time interval
        interval = tasks[taskname]['RowInterval'] + ahead_rows
        n = interval * 15  # Convert to minutes
        logger.debug('Calculated time interval: interval=%d, n=%d minutes', interval, n)
        
        # Get configuration file path
        try:
            # Try to locate configuration file using package resources
            config_pth = pkg_resources.resource_filename(
                # module/config/config.yaml
                'module', 
                'module/config/config.yaml'
            )
            logger.debug('Located configuration file using package resources: %s', config_pth)
        except Exception as exc:
            # If failed, use relative path
            logger.debug('Package resource location failed, using relative path: %s', exc)
            base_path = os.path.dirname(os.path.dirname(__file__))
            config_pth = os.path.join(base_path, 'module', 'config', 'config.yaml')
            logger.debug('Located configuration file using relative path: %s', config_pth)
        
        # Load data
        logger.info('Loading raw data...')
        df, weather_data = load_data(config)
        
        # Merge weather data (if exists)
        if isinstance(weather_data, pd.DataFrame) and not weather_data.empty:
            logger.info('Merging weather data...')
            df = weather_data_merge(config, df, weather_data, n)
        
        # Add time features and process configuration
        logger.info('Adding time features and processing configuration...')
        df, task_config = add_time_feature_process(
            df, 
            time_col=time_col, 
            target_name=target_name, 
            config_pth=config_pth, 
            interval=interval
        )
        
        # Update task configuration
        logger.info('Updating task configuration...')
        task_config = update_config(config, task_config, taskname)
        
        logger.info('Training data preparation completed, shape: %s', df.shape)
        return df, task_config
        
    except Exception as exc:
        logger.error('Error getting training data: %s\n%s', exc, traceback.format_exc())
        raise


def get_X_test(
    config: Dict[str, Any], 
    taskname: str, 
    predict_time: str, 
    label_save_path: str = ''
) -> Tuple[pd.DataFrame, Dict[str, Any], float, pd.Timestamp, pd.Timestamp]:
    """Get test data and prediction-related parameters.
    
    Based on prediction time and task configuration, prepare data for prediction and related parameters.
    
    Args:
        config: Global configuration dictionary
        taskname: Task name
        predict_time: Prediction time string
        label_save_path: Label save path, default is empty string
        
    Returns:
        Tuple (test dataframe, task configuration, historical average, prediction start time, prediction end time)
    """
    logger.info('Starting to get test data...')
    
    try:
        # Extract necessary parameters from configuration
        target_name = config.get('Feature', {}).get('TargetName')
        time_col = config.get('Feature', {}).get('TimeIndex')
        tasks = config.get('Task', {})
        ahead_rows = config.get('Feature', {}).get('AheadRows', 0)
        hourlagtime = config.get('Feature', {}).get('HourSubmitLagTime', '0min')
        daylagtime = config.get('Feature', {}).get('DaySubmitLagTime', '0hour')
        
        # Validate necessary parameters
        if not target_name or not time_col or not tasks or taskname not in tasks:
            logger.error('Configuration missing necessary parameters: target_name=%s, time_col=%s, taskname=%s', 
                      target_name, time_col, taskname)
            raise ValueError('Configuration missing necessary parameters')
        
        # Calculate time interval
        interval = tasks[taskname]['RowInterval'] + ahead_rows
        n = interval * 15  # Convert to minutes
        logger.debug('Calculated time interval: interval=%d, n=%d minutes', interval, n)
        
        # Calculate prediction end time, floor to 15 minutes
        predict_end_time = pd.to_datetime(predict_time).floor('15min')
        
        # Get configuration file path
        try:
            # Try to locate configuration file using package resources
            config_pth = pkg_resources.resource_filename(
                # module/config/config.yaml
                'module', 
                'module/config/config.yaml'
            )
            logger.debug('Located configuration file using package resources: %s', config_pth)
        except Exception as exc:
            # If failed, use relative path
            logger.debug('Package resource location failed, using relative path: %s', exc)
            base_path = os.path.dirname(os.path.dirname(__file__))
            config_pth = os.path.join(base_path, 'module', 'config', 'config.yaml')
            logger.debug('Located configuration file using relative path: %s', config_pth)
        
        # Adjust prediction time based on task type
        # Time adjustment for hourly tasks
        if taskname.endswith('Hour'):
            lag_time = 0
            if hourlagtime.endswith('min'):
                lag_time = int(hourlagtime[:-3])
                logger.debug('Hourly task time adjustment: %d minutes', lag_time)
            elif hourlagtime.endswith('hour'):
                lag_time = int(hourlagtime[:-4]) * 60
                logger.debug('Hourly task time adjustment: %d hours (%d minutes)', 
                          int(hourlagtime[:-4]), lag_time)
            predict_end_time += pd.Timedelta(minutes=lag_time)
        
        # Time adjustment for daily tasks
        if taskname.endswith('Day'):
            lag_time = 0
            if daylagtime.endswith('min'):
                lag_time = int(daylagtime[:-3])
                logger.debug('Daily task time adjustment: %d minutes', lag_time)
            elif daylagtime.endswith('hour'):
                lag_time = int(daylagtime[:-4]) * 60
                logger.debug('Daily task time adjustment: %d hours (%d minutes)', 
                          int(daylagtime[:-4]), lag_time)
            predict_end_time += pd.Timedelta(minutes=lag_time)
        
        # Calculate prediction start time and adjusted end time
        predict_start_time = predict_end_time - pd.Timedelta(minutes=n)
        predict_end_time = predict_end_time - pd.Timedelta(minutes=ahead_rows * 15)
        
        logger.info('Prediction time range: [%s, %s]', predict_start_time, predict_end_time)
        
        # Wait time (if configured)
        wait_seconds = int(config.get('Feature', {}).get('PredictWaiting', 0))
        if wait_seconds > 0:
            logger.info('Waiting %d seconds before loading data...', wait_seconds)
            time.sleep(wait_seconds)
        
        # Load data
        logger.info('Loading raw data...')
        df, weather_data = load_data(config)
        
        # Merge weather data (if exists)
        if isinstance(weather_data, pd.DataFrame) and not weather_data.empty:
            logger.info('Merging weather data...')
            df = weather_data_merge(config, df, weather_data, n)
        
        # Process real label data (if it's a 4-hour forecast task and label save path is specified)
        if label_save_path and taskname == 'Forecast4Hour':
            logger.info('Processing real label data...')
            
            # If label file already exists, read and append new data
            if os.path.exists(label_save_path):
                logger.debug('Reading existing label file: %s', label_save_path)
                label_df = pd.read_csv(label_save_path)
                label_df['date_time'] = pd.to_datetime(label_df['date_time'])
                max_date = label_df['date_time'].max()
                
                # Extract new label data (data greater than existing maximum date)
                res = df[[time_col, target_name]][
                    df[time_col] > max_date
                ].rename(columns={target_name: 'power'})
                
                logger.debug('New label data sample: %s', res.tail().to_dict(orient='records'))
            else:
                # If label file doesn't exist, create a new file
                logger.info('Creating new real label file...')
                os.makedirs(os.path.dirname(label_save_path), exist_ok=True)
                label_df = pd.DataFrame()
                res = df[[time_col, target_name]].rename(columns={target_name: 'power'})
            
            # If there is new label data, process and save
            if not res.empty:
                # Add metadata
                res['exec_time'] = pd.Timestamp.today()
                res['date_time'] = res[time_col]
                res = res[['exec_time', 'date_time', 'power']]
                
                # Merge new and old data
                res = pd.concat([label_df, res], ignore_index=True)
                logger.debug('Merged label data sample: %s', res.tail().to_dict(orient='records'))
                
                # Save to file
                res.to_csv(label_save_path, index=False)
                logger.info('Successfully saved real label file, total %d records', len(res))
            else:
                logger.info('No new real label data')
                res = label_df
            
            # Calculate historical average
            mean_power = res['power'].mean()
            logger.debug('Historical average power: %.4f', mean_power)
        else:
            # If label processing is not needed, directly calculate average
            mean_power = df[target_name].mean()
            logger.debug('Historical average power: %.4f', mean_power)
        
        # Filter data within prediction time range
        logger.info('Filtering data within prediction time range...')
        df = df[
            (df[time_col] >= predict_start_time) & 
            (df[time_col] <= predict_end_time)
        ]
        
        # Handle special task types
        if not df.empty:
            if taskname == 'Forecast10Day':
                logger.info('Completing time series for 10-day forecast task...')
                df = get_timestamps(
                    df, 
                    time_col, 
                    start_time=predict_start_time, 
                    end_time=predict_end_time
                )
            logger.info('Prediction data preparation completed, shape: %s', df.shape)
        else:
            logger.error('No data obtained within the prediction time range')
        
        # Add time features and process configuration
        logger.info('Adding time features and processing configuration...')
        df, task_config = add_time_feature_process(
            df, 
            time_col=time_col, 
            target_name=target_name, 
            config_pth=config_pth, 
            interval=interval
        )
        
        # Update task configuration
        logger.info('Updating task configuration...')
        task_config = update_config(config, task_config, taskname)
        
        logger.info('Test data preparation completed')
        return df, task_config, mean_power, predict_start_time, predict_end_time
        
    except Exception as exc:
        logger.error('Error getting test data: %s\n%s', exc, traceback.format_exc())
        # Return empty dataframe and default values
        return pd.DataFrame(), {}, 0.0, pd.Timestamp.now(), pd.Timestamp.now()

def train_task(config: dict, task_name: str) -> None:
    """Train and save model by task name.

    Args:
        config: Global configuration dictionary.
        task_name: Task name.
    """
    logger.info("%s task start training.", task_name)

    # Debug CSV save directory
    debug_csv_path: str = config["Path"]["OtherFile"]
    os.makedirs(debug_csv_path, exist_ok=True)

    try:
        df, task_config = get_X_y_train(config, task_name)
        debug_file = os.path.join(
            debug_csv_path,
            f"{task_name}_train_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv",
        )
        df.to_csv(debug_file, index=False)
        ESPML(task_config, df).fit()
        logger.info("%s task finish training successfully.", task_name)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("%s task training failed: %s\n%s", task_name, exc, traceback.format_exc())


def predict_task(config: dict, task_name: str) -> pd.DataFrame:
    """Execute prediction based on the latest model and save results.

    Args:
        config: Global configuration dictionary.
        task_name: Task name.

    Returns:
        Prediction result ``DataFrame``.
    """
    logger.info("%s task start predicting.", task_name)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tasks = config.get("Task", {})
    time_col: str | None = config.get("Feature", {}).get("TimeIndex")
    ahead_rows: int = config.get("Feature", {}).get("AheadRows", 0)

    if not time_col:
        logger.error("TimeIndex not configured, cannot execute prediction.")
        return pd.DataFrame()

    interval: int = tasks.get(task_name, {}).get("RowInterval", 0) + ahead_rows
    minutes_ahead: int = interval * 15

    # Prediction result save path
    prediction_dir: str = config["Path"]["Prediction"]
    task_file_map = {"Forecast4Hour": "4h.csv", **{f"Forecast{i}Day": f"{i}d.csv" for i in range(1, 11)}}
    predict_save_path = os.path.join(prediction_dir, task_file_map[task_name])

    # Label save path
    label_save_path = os.path.join(prediction_dir, "real_wind_power.csv")

    # Latest model path
    model_dir = os.path.join(config["Path"]["Model"], task_name)
    if not os.path.exists(model_dir):
        logger.warning("%s has no available model.", task_name)
        return pd.DataFrame()

    # Get latest valid model sequence number
    seq_list = [int(folder) for folder in os.listdir(model_dir) if folder.isdigit()]
    seq = max(seq_list, default=-1)
    while seq >= 0 and not os.path.exists(os.path.join(model_dir, str(seq), "automl", "val_res")):
        seq -= 1
    if seq < 0:
        logger.warning("%s no valid model sequence found.", task_name)
        return pd.DataFrame()
    logger.info("%s uses model seq %s for prediction.", task_name, seq)

    # Get test data
    X_val, task_config, hist_mean, p_start, p_end = get_X_test(
        config, task_name, now_str, label_save_path
    )

    # Debug CSV save
    debug_csv_path: str = config["Path"]["OtherFile"]
    os.makedirs(debug_csv_path, exist_ok=True)
    if isinstance(X_val, pd.DataFrame) and not X_val.empty:
        debug_file = os.path.join(
            debug_csv_path,
            f"{task_name}_predict_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv",
        )
        X_val.to_csv(debug_file, index=False)

    # Historical prediction results
    if os.path.exists(predict_save_path):
        pre_df = pd.read_csv(predict_save_path)
    else:
        logger.info("Create prediction result file at %s", predict_save_path)
        os.makedirs(os.path.dirname(predict_save_path), exist_ok=True)
        pre_df = pd.DataFrame()

    # Generate prediction result DataFrame
    if not X_val.empty:
        res = pd.DataFrame({
            "base_time": X_val[time_col],
            "date_time": pd.to_datetime(X_val[time_col]) + pd.Timedelta(minutes=minutes_ahead),
        })
        try:
            predictions = predict(task_config, X_val, save=False)
            res["power"] = predictions.values
            logger.info("%s task prediction finished successfully.", task_name)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("%s prediction failed: %s\n%s", task_name, exc, traceback.format_exc())
            res["power"] = hist_mean
    else:
        logger.warning("%s X_val is empty, using historical mean value for backfill.", task_name)
        base_range = pd.date_range(p_start, p_end, freq="15min")[:interval]
        res = pd.DataFrame({
            "base_time": base_range,
            "date_time": base_range + pd.Timedelta(minutes=minutes_ahead),
            "power": [hist_mean] * interval,
        })

    # Append metadata and save
    res["exec_point"] = res["base_time"].max()
    res["exec_time"] = pd.Timestamp.today()

    pd.concat([pre_df, res], ignore_index=True).to_csv(predict_save_path, index=False)
    logger.info("%s prediction results saved to %s", task_name, predict_save_path)

    return res


def incrml_train(config: dict, task_name: str, mode: str = "prod") -> None:
    """Incremental training entry point.

    Args:
        config: Global configuration.
        task_name: Task name.
        mode: Running mode, ``test`` means execute once immediately.
    """
    tasks = config.get("Task", {})
    start_time = tasks.get(task_name, {}).get("TrainTime")

    init_logger(config, task_name, "train")

    if mode == "test":
        train_task(config, task_name)
        return

    model_dir = config["Path"]["Model"]
    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        # Train once on first run
        train_task(config, task_name)

    scheduled_job = partial(train_task, config=config, task_name=task_name)

    if isinstance(start_time, list):
        for t in start_time:
            every().day.at(t).do(scheduled_job)
    else:
        every().day.at(start_time).do(scheduled_job)

    # External loop needs to call run_pending()


def incrml_predict(config: dict, task_name: str, mode: str = "prod") -> None:
    """Incremental prediction entry point.

    Args:
        config: Global configuration.
        task_name: Task name.
        mode: Running mode, ``test`` means execute once immediately.
    """
    tasks = config.get("Task", {})
    start_time = tasks.get(task_name, {}).get("SubmitTime")

    init_logger(config, task_name, "predict")

    if mode == "test":
        predict_task(config, task_name)
        return

    scheduled_job = partial(predict_task, config=config, task_name=task_name)

    if task_name.endswith("Hour") and isinstance(start_time, list):
        for t in start_time:
            every().hours.at(t).do(scheduled_job)
    elif task_name.endswith("Hour") and isinstance(start_time, str):
        every().hours.at(start_time).do(scheduled_job)
    elif isinstance(start_time, list):
        for t in start_time:
            every().days.at(t).do(scheduled_job)
    else:
        every().days.at(start_time).do(scheduled_job)

    # External loop needs to call run_pending()


def backtracking(config: dict) -> None:
    """
    Execute backtesting, train models and evaluate performance on historical data
    
    Args:
        config: Configuration dictionary, containing task parameters, feature information, and paths
    """
    # Initialize logger
    init_logger(config, 'backtracking', 'test')
    logger.info('Starting backtesting')
    
    try:
        # Extract necessary parameters from configuration
        tasks = config.get('Task', None)
        if not tasks:
            logger.error('Task configuration does not exist or is empty')
            return
            
        ahead_rows = config.get('Feature', {}).get('AheadRows', 0)
        time_col = config.get('Feature', {}).get('TimeIndex', None)
        target_name = config.get('Feature', {}).get('TargetName', None)
        
        if not time_col or not target_name:
            logger.error('Time column or target variable name not specified')
            return
            
        # Create save path
        save_path = config['Path']['Prediction']
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
            logger.info(f'Created prediction result save path: {save_path}')
        
        # Initialize results dictionary
        results = {
            'date_time': [], 
            'task_type': [], 
            'rse': [], 
            'r2': []
        }
        
        # Load data
        logger.info('Loading data...')
        df, weather_data = load_data(config)
        logger.info(f'Data loading completed, shape: {df.shape}')
        
        # Add time features
        add_features = add_time_features(df, time_col)
        
        # Process by task name
        for taskname in tasks.keys():
            logger.info(f'Processing task: {taskname}')
            try:
                # Calculate time interval
                interval = tasks[taskname]['RowInterval'] + ahead_rows
                n = interval * 15  # Convert interval to minutes
                
                # Merge weather data (if exists)
                if isinstance(weather_data, pd.DataFrame):
                    task_df = weather_data_merge(config, df, weather_data, n)
                    logger.info(f'Weather data merged, result shape: {task_df.shape}')
                else:
                    task_df = df
                
                # Set date range for backtesting
                date_sample = pd.date_range('2024-10-01 08:00:00', '2024-11-01 08:00:00')
                date_sample = pd.to_datetime(date_sample)
                logger.info(f'Backtesting date range: {date_sample.min()} to {date_sample.max()}')
                
                # Get task configuration path
                config_pth = config['Task'][taskname]['YamlFiles']
                
                # Perform backtesting for each date
                for date in date_sample:
                    logger.info(f'Processing date: {date}')
                    try:
                        # Execute single backtest and collect results
                        y_test, y_pred, X_test = perform_single_backtest(
                            task_df, date, time_col, target_name, 
                            interval, config_pth, add_features
                        )
                        
                        # Calculate evaluation metrics
                        r2 = r2_score(y_test, y_pred)
                        rse = (y_test - y_pred).pow(2).sum() ** 0.5
                        
                        # Record results
                        results['r2'].append(r2)
                        results['rse'].append(rse)
                        results['task_type'].append(taskname)
                        results['date_time'].append(date)
                        
                        # Save single test result
                        single_result_df = pd.DataFrame({
                            'base_time': X_test[time_col], 
                            'pre_time': pd.to_datetime(X_test[time_col]) + pd.Timedelta(minutes=n), 
                            'y_test': y_test.values, 
                            'y_pred': y_pred.values
                        })
                        
                        # Format date for filename
                        date_str = str(date).replace('-', '').replace(' ', '')
                        result_filename = f"{taskname}_{date_str}.csv"
                        result_path = os.path.join(save_path, result_filename)
                        
                        # Save to CSV
                        single_result_df.to_csv(result_path, index=False)
                        logger.info(f'Saved single test result to: {result_path}')
                        logger.info(f'R2: {r2:.4f}, RSE: {rse:.4f}')
                        
                    except Exception as e:
                        logger.error(f'Error processing date {date}: {str(e)}\n{traceback.format_exc()}')
                        continue
            
            except Exception as e:
                logger.error(f'Error processing task {taskname}: {str(e)}\n{traceback.format_exc()}')
                continue
        
        # Save summary results
        results_df = pd.DataFrame(results)
        summary_path = os.path.join(save_path, 'backtracking_res.csv')
        results_df.to_csv(summary_path, index=False)
        logger.info(f'Saved backtesting summary results to: {summary_path}')
        
    except Exception as e:
        logger.error(f'Error executing backtesting: {str(e)}\n{traceback.format_exc()}')
    finally:
        logger.info('Backtesting completed')
        # Clean up log handlers
        logger.handlers = []


def add_time_features(df: pd.DataFrame, time_col: str) -> List[str]:
    """
    Add time features to the dataframe
    
    Args:
        df: Input dataframe
        time_col: Time column name
        
    Returns:
        List of added feature names
    """
    logger.info('Adding time features...')
    add_features = []
    
    try:
        # Get time features
        tmp_dt = df[time_col].dt
        
        # Create time feature dictionary
        new_columns_dict = {
            f'year_{time_col}': tmp_dt.year, 
            f'month_{time_col}': tmp_dt.month, 
            f'day_{time_col}': tmp_dt.day, 
            f'hour_{time_col}': tmp_dt.hour, 
            f'minute_{time_col}': tmp_dt.minute, 
            f'second_{time_col}': tmp_dt.second, 
            f'dayofweek_{time_col}': tmp_dt.dayofweek, 
            f'dayofyear_{time_col}': tmp_dt.dayofyear, 
            f'quarter_{time_col}': tmp_dt.quarter
        }
        
        # Only add features with enough unique values
        for key, value in new_columns_dict.items():
            if key not in df.columns and value.nunique(dropna=False) >= 2:
                add_features.append(key)
                df[key] = value
                logger.debug(f'Added time feature: {key}')
                
        logger.info(f'Added {len(add_features)} time features')
        return add_features
        
    except Exception as e:
        logger.error(f'Error adding time features: {str(e)}')
        return []


def perform_single_backtest(task_df: pd.DataFrame, date: pd.Timestamp, 
                          time_col: str, target_name: str, interval: int,
                          config_pth: str, add_features: list) -> Tuple[pd.Series, np.ndarray, pd.DataFrame]:
    """
    Perform a single backtest
    
    Args:
        task_df: Task dataframe
        date: Prediction date
        time_col: Time column name
        target_name: Target variable name
        interval: Time interval
        config_pth: Configuration file path
        add_features: List of additional features
        
    Returns:
        Tuple (y_test, y_pred, X_test): Actual values, predicted values and test data
    """
    # Extract test label data
    y_test = task_df[target_name][task_df[time_col] >= date][:interval]
    logger.debug(f'Test label data length: {len(y_test)}')
    
    # Extract training data (using past year of data)
    temp_df = task_df[task_df[time_col] < date]
    temp_df = temp_df[temp_df[time_col] >= date - pd.Timedelta(days=365)]
    logger.debug(f'Training data shape: {temp_df.shape}')
    
    # Create label column
    temp_df['label'] = temp_df[target_name].shift(-interval)
    
    # Separate features and labels
    X_test = temp_df[temp_df['label'].isnull()].drop(columns='label')
    X_train = temp_df[~temp_df['label'].isnull()]
    logger.debug(f'Training set shape: {X_train.shape}, Test set shape: {X_test.shape}')
    
    # Parse task configuration
    task_config = YamlParser.parse(config_pth)
    
    # Add additional features to configuration
    task_config['Feature']['FeatureName'].extend(add_features)
    task_config['Feature']['CategoricalFeature'].extend(add_features)
    task_config['Feature']['TargetName'] = 'label'
    
    # Train model
    logger.info('Training model...')
    model = ESPML(task_config, X_train)
    model.fit()
    
    # Predict
    logger.info('Performing prediction...')
    y_pred = predict(task_config, X_test, save=False)
    
    return y_test, y_pred, X_test