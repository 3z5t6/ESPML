"""
Wind power incremental learning utility module.

Provides utility functions for wind power data processing, model training, and prediction.
"""

# Standard library imports
import os
import gc
import time
import traceback
import logging
import hashlib
import pkg_resources
from datetime import datetime
from functools import partial
from typing import Dict, List, Optional, Tuple, Union, Any
from ctypes import *

# Third-party imports
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import r2_score
from schedule import every, repeat, run_pending, jobs

# Project imports
from module.utils.ml import ESPML
from module.utils.test_file import predict
from module.utils.yaml_parser import YamlParser

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_timestamps(
    df: pd.DataFrame, 
    time_col: str, 
    interval: str = '15min', 
    start_time: Optional[pd.Timestamp] = None, 
    end_time: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Ensure data is available every 15 minutes.
    
    Resamples time series data to ensure data is available at each time interval within the specified time range.
    Missing time points are filled forward.
    
    Args:
        df: Input data frame
        time_col: Name of the time column
        interval: Time interval, default is '15min'
        start_time: Start time, default is earliest time in data
        end_time: End time, default is latest time in data
        
    Returns:
        Resampled data frame
    """
    try:
        # Ensure time column is of datetime type
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Set time index and take mean by time group
        df.set_index(time_col, inplace=True)
        df = df.groupby(time_col).mean()
        
        # Determine time range
        if start_time is None:
            start_time = df.index.min()
        if end_time is None:
            end_time = df.index.max()
            
        # Create full time index
        full_index = pd.date_range(
            start=start_time, 
            end=end_time, 
            freq=interval, 
            name=time_col
        )
        
        logger.info(
            "Ensure data is available every %s within time range [%s, %s]", 
            interval, str(start_time), str(end_time)
        )
        
        # Sort by time index and reindex, using forward fill to handle missing values
        df.sort_index(inplace=True)
        df = df.reindex(full_index, method='ffill')
        df.reset_index(inplace=True)
        
        return df
    except Exception as exc:
        logger.error("Error resampling time series: %s\n%s", exc, traceback.format_exc())
        raise

def next_eight_pm(
    timestamp: pd.Series, 
    n: int, 
    predict_time: int = 8, 
    previous_day: int = 0
) -> pd.Series:
    """Calculate the nearest weather forecast time point.
    
    Based on current time and prediction lead time n, determine which day's forecast time to use.
    
    Args:
        timestamp: Time series
        n: Number of minutes to predict ahead
        predict_time: Forecast time (hour), default is 8 (8 AM)
        previous_day: Number of days to offset backwards, default is 0
        
    Returns:
        Corresponding forecast time point series
    """
    try:
        # Calculate the time at predict_time on the current day, and subtract the specified number of days
        eight_pm_today = (
            timestamp + 
            pd.to_timedelta(predict_time - timestamp.dt.hour, unit='h') - 
            pd.to_timedelta(timestamp.dt.minute, unit='m') - 
            pd.to_timedelta(timestamp.dt.second, unit='s') - 
            pd.to_timedelta(timestamp.dt.microsecond, unit='us') - 
            pd.to_timedelta(previous_day, unit='d')
        )
        
        # Adjust forecast time based on different prediction lead times
        n_hours = n // 60  # Convert to hours
        
        # If it's a 4-24 hour prediction, use the forecast from the previous day if current time is earlier than forecast time
        if 4 <= n_hours < 24:
            eight_pm_today[timestamp.dt.hour < predict_time] -= pd.Timedelta(days=1)
            logger.debug("For %d-hour prediction, using forecast from previous day if current time is earlier than %d o'clock", 
                       n_hours, predict_time)
        
        # If it's a 9-11 day prediction, use special handling
        if 216 < n_hours < 264:
            logger.debug("For %d-hour prediction (about %d days), using special handling", 
                       n_hours, n_hours/24)
            return eight_pm_today
        
        # Otherwise, if current time is later than forecast time, use the forecast from the next day
        eight_pm_today[timestamp.dt.hour >= predict_time] += pd.Timedelta(days=1)
        logger.debug("For %d-hour prediction, using forecast from next day if current time is later than %d o'clock", 
                   n_hours, predict_time)
        
        return eight_pm_today
    
    except Exception as exc:
        logger.error("Error calculating forecast time: %s\n%s", exc, traceback.format_exc())
        raise

def init_logger(config: Dict[str, Any], task_name: str, logtype: str) -> None:
    """Initialize logging configuration.
    
    Sets up log handlers for specific tasks and log types, including file log and console log.
    
    Args:
        config: Configuration dictionary
        task_name: Task name
        logtype: Log type (e.g., 'train', 'predict', 'backtracking', etc.)
    """
    try:
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
        
        logger.info("Initialized logging for %s task of type %s", task_name, logtype)
    except Exception as exc:
        print(f"Error initializing logging: {exc}\n{traceback.format_exc()}")
        raise

def weather_data_process(config: Dict[str, Any], df: pd.DataFrame, interval: str = '15min') -> pd.DataFrame:
    """Process weather data, resampling and interpolating each forecast group.
    
    Args:
        config: Configuration dictionary with column name settings
        df: Weather data frame
        interval: Resampling interval, default is '15min'
        
    Returns:
        Processed weather data frame
    """
    logger.info("Starting to process weather data, data shape: %s", df.shape)
    
    try:
        # Get column names from config
        sc_date = config['Features']['WeatherSCDateIndex']
        sc_time = config['Features']['WeatherSCTimeIndex']
        pre_time = config['Features']['WeatherPreTimeIndex']
        
        # Create forecast group identifier
        df['GROUP'] = df[sc_date].astype(str) + ' ' + df[sc_time].astype(str)
        
        # Convert forecast time to timestamp format
        df[pre_time] = pd.to_datetime(df[pre_time])
        
        # Process each forecast group separately
        resampled_data = []
        group_count = df['GROUP'].nunique()
        logger.debug("There are %d forecast groups to process", group_count)
        
        for group, group_data in df.groupby('GROUP'):
            # Set forecast time as index
            group_data = group_data.set_index(pre_time)
            
            # Select only numeric columns for interpolation
            group_data_numeric = group_data.select_dtypes(include='number')
            
            # Resample at specified intervals and linearly interpolate
            group_data_resampled = group_data_numeric.resample(interval).interpolate(method='linear')
            
            # Add group identifier
            group_data_resampled['GROUP'] = group
            
            # Add to result list
            resampled_data.append(group_data_resampled.reset_index())
            
        # Merge all resampled group data
        result = pd.concat(resampled_data, ignore_index=True)
        logger.info("Weather data processing completed, result shape: %s", result.shape)
        
        # NOTE: Fixed issue by removing unreachable code that was here after the return statement
        return result
        
    except Exception as exc:
        logger.error("Error processing weather data: %s\n%s", exc, traceback.format_exc())
        # If an error occurs, try to clean up temporary columns and return original data frame
        try:
            if 'pre_time' in df.columns or 'GROUP' in df.columns or 'next_8pm' in df.columns:
                df.drop(columns=['pre_time', 'GROUP', 'next_8pm'], errors='ignore', inplace=True)
        except Exception:
            pass
        raise

def weather_data_merge(
    config: Dict[str, Any], 
    df: pd.DataFrame, 
    weather_data: pd.DataFrame, 
    n: int, 
    previous_day: int = 0, 
    previous_day_num: int = 5
) -> pd.DataFrame:
    """Merge weather data with the main dataframe.
    
    Based on prediction time and weather forecast time, merges appropriate 
    weather data with the main dataframe.
    
    Args:
        config: Configuration dictionary
        df: Main dataframe
        weather_data: Weather dataframe
        n: Prediction lead time in minutes
        previous_day: Days to offset backwards, default is 0
        previous_day_num: Maximum days to look back, default is 5
        
    Returns:
        Merged dataframe
    """
    logger.info("Starting weather data merge, main data shape: %s, weather data shape: %s", 
              df.shape, weather_data.shape)
    
    try:
        # Get column names from config
        pre_time = config['Features']['WeatherPreTimeIndex']
        time_col = config['Features']['TimeIndex']
        
        # Get forecast time point (hour)
        predict_time = int(weather_data['GROUP'].astype(str).values[0].split(' ')[1].split(':')[0])
        logger.debug("Forecast time point is %d o'clock", predict_time)
        
        # Calculate weather forecast time and prediction time
        df['next_8pm'] = next_eight_pm(df[time_col], n, predict_time, previous_day // previous_day_num)
        df['pre_time'] = pd.to_datetime(df[time_col]) + pd.Timedelta(minutes=n)
        
        # Process the GROUP column in weather data
        weather_data['GROUP'] = pd.to_datetime(weather_data['GROUP'])
        
        # Merge weather data with main dataframe
        result_df = df.merge(
            weather_data, 
            left_on=['pre_time', 'next_8pm'], 
            right_on=[pre_time, 'GROUP'], 
            how='left', 
            suffixes=('', f'_previous_day_{previous_day}')  # Fixed typo: changed 'previou' to 'previous'
        )
        
        # Clean up temporary columns
        result_df.drop(columns=['pre_time', 'GROUP', 'next_8pm', pre_time], inplace=True)
        
        logger.info("Weather data merge complete, result shape: %s", result_df.shape)
        return result_df
        
    except Exception as exc:
        logger.error("Error merging weather data: %s\n%s", exc, traceback.format_exc())
        # If error occurs, try to clean up temporary columns and return original dataframe
        try:
            if 'pre_time' in df.columns or 'GROUP' in df.columns or 'next_8pm' in df.columns:
                df.drop(columns=['pre_time', 'GROUP', 'next_8pm'], errors='ignore', inplace=True)
        except Exception:
            pass
        raise

def merge_all_weather_data(config: Dict[str, Any], df: pd.DataFrame, weather_data: List[pd.DataFrame], n: int) -> pd.DataFrame:
    """Merge multiple weather data sources.
    
    Sequentially merges multiple weather data sources with the main dataframe,
    with each source potentially having multiple time offsets.
    
    Args:
        config: Configuration dictionary
        df: Main dataframe
        weather_data: List of weather dataframes
        n: Prediction lead time in minutes
        
    Returns:
        Merged dataframe
    """
    logger.info("Starting to merge multiple weather data sources, total sources: %d", len(weather_data))
    
    try:
        result_df = df.copy()
        
        # Iterate through each weather data source
        for j, wdata in enumerate(weather_data):
            logger.debug("Processing weather data source #%d, shape: %s", j+1, wdata.shape)
            
            # Try different time offsets for each data source
            for i in range(2):
                previous_day = i + j * 5
                logger.debug("Using time offset: %d days", previous_day)
                
                # Merge weather data
                result_df = weather_data_merge(
                    config, 
                    result_df, 
                    wdata, 
                    n, 
                    previous_day=previous_day, 
                    previous_day_num=5
                )
        
        logger.info("Multiple weather data sources merge complete, result shape: %s", result_df.shape)
        return result_df
        
    except Exception as exc:
        logger.error("Error merging multiple weather data sources: %s\n%s", exc, traceback.format_exc())
        raise

def next_run_time() -> pd.Timestamp:
    """Calculate the next task execution time.
    
    Returns:
        The earliest scheduled execution time for the next task
    """
    try:
        job_next_times = []
        for job in jobs:
            job_next_times.append(job.next_run)
        return pd.to_datetime(job_next_times).min()
    except Exception as exc:
        logger.error("Error calculating next task execution time: %s\n%s", exc, traceback.format_exc())
        # Return current time as default value
        return pd.Timestamp.now()

def add_time_feature_process(
    df: pd.DataFrame, 
    time_col: str, 
    target_name: str, 
    config_pth: str, 
    interval: int
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Add time features to the data frame and update configuration.
    
    Extract various time features (year, month, day, hour, etc.) from the time column and add them to the data frame.
    Also, create a label column based on the target column and specified interval, and update the configuration file.
    
    Args:
        df: Input data frame
        time_col: Name of the time column
        target_name: Name of the target column
        config_pth: Path to configuration file
        interval: Prediction interval
        
    Returns:
        tuple: (Data frame with added time features, updated configuration dictionary)
    """
    logger.info("Starting to add time features to the data, data shape: %s", df.shape)
    
    try:
        # Initialize feature list
        add_features = []
        
        # Get datetime accessor of time column
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            logger.debug("Converting time column %s to datetime type", time_col)
            df[time_col] = pd.to_datetime(df[time_col])
            
        tmp_dt = df[time_col].dt
        
        # Create various time features
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
        
        # Add time features to data frame
        for key, value in new_columns_dict.items():
            add_features.append(key)
            df[key] = value
            logger.debug("Added time feature: %s", key)
        
        # Create label column
        df['label'] = df[target_name].shift(-interval)
        logger.debug("Created label column, using target %s to move forward %d steps", target_name, interval)
        
        # Update configuration
        logger.debug("Loading task configuration from %s", config_pth)
        task_config = YamlParser.parse(config_pth)
        
        # Add new features to configuration
        task_config['Feature']['FeatureName'].extend(add_features)
        task_config['Feature']['CategoricalFeature'].extend(add_features)
        task_config['Feature']['TargetName'] = 'label'
        
        # Save updated configuration
        YamlParser.save(task_config, config_pth)
        logger.info("Updated configuration file %s, added %d time features", config_pth, len(add_features))
        
        return df, task_config
        
    except Exception as exc:
        logger.error("Error adding time features: %s\n%s", exc, traceback.format_exc())
        raise

def update_config(config: Dict[str, Any], task_config: Dict[str, Any], task_name: str) -> Dict[str, Any]:
    """Use main configuration to update task configuration.
    
    Synchronizes relevant settings from the main configuration file to the task-specific configuration, including features, time column, automation parameters, etc.
    
    Args:
        config: Main configuration dictionary
        task_config: Task configuration dictionary
        task_name: Task name
        
    Returns:
        Updated task configuration dictionary
    """
    logger.info("Starting to update configuration for task '%s'", task_name)
    
    try:
        # Update feature-related configuration
        task_config['Feature']['FeatureName'] = config['Features']['FeatureName']
        task_config['Feature']['CategoricalFeature'] = config['Features']['CategoricalFeature']
        task_config['Feature']['TimeIndex'] = config['Features']['TimeIndex']
        task_config['Feature']['IgnoreFeature'] = config['Features']['IgnoreFeature']
        
        # Update automation parameters
        task_config['AutoML']['TimeBudget'] = config['Features']['AutoMLTimeBudget']
        task_config['AutoFE']['maxTrialNum'] = config['Features']['AutoFEmaxTrialNum']
        
        # Set model save path
        model_save_path = os.path.join(config['Path']['Model'], task_name)
        task_config['IncrML']['SaveModelPath'] = model_save_path
        
        # Ensure model save path exists
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path, exist_ok=True)
            logger.debug("Created model save path: %s", model_save_path)
        
        logger.info("Configuration for task '%s' updated successfully", task_name)
        return task_config
        
    except KeyError as exc:
        logger.error("Error updating configuration: %s\n%s", exc, traceback.format_exc())
        raise ValueError(f"Missing required key in configuration: {exc}")
        
    except Exception as exc:
        logger.error("Error updating configuration: %s\n%s", exc, traceback.format_exc())
        raise

def load_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """Load data from multiple sources and preprocess it.
    
    Load wind tower data, wind fan data, and weather forecast data, and merge them into a unified data frame.
    Also, convert the target variable to the appropriate units and standardize the time column.
    
    Args:
        config: Configuration dictionary containing data source paths and feature configuration
        
    Returns:
        tuple: (Merged data frame, weather data list)
        
    Raises:
        ValueError: Raised when necessary configuration items are missing or data sources are missing
    """
    logger.info('Starting to load data...')
    
    try:
        # Get necessary parameters from configuration
        time_col = config.get('Features', {}).get('TimeIndex', None)
        windtower_paths = config.get('DataSource', {}).get('WindTower', None)
        windfans_paths = config.get('DataSource', {}).get('WindFans', None)
        weather_paths = config.get('DataSource', {}).get('WeatherForecast', None)
        target_name = config.get('Features', {}).get('TargetName', None)
        tasks = config.get('Task', None)
        
        # Verify necessary configuration items
        if not all([windtower_paths, tasks]):
            error_msg = 'Missing necessary data source or task'
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Initialize data lists
        tower, fans, weather_data = ([], [], [])
        
        # Load wind tower data
        logger.info('Loading wind tower data...')
        for pth in windtower_paths:
            try:
                logger.debug('Loading wind tower data from %s', pth)
                fan_df = pd.read_csv(pth)
                logger.debug('Wind tower data sample: %s', fan_df.tail(2).values.tolist())
                tower.append(fan_df)
            except Exception as exc:
                logger.error('Error loading wind tower data file %s: %s\n%s', pth, exc, traceback.format_exc())
                raise
        
        # Load wind fan data
        if windfans_paths:
            logger.info('Loading wind fan data...')
            for pth in windfans_paths:
                try:
                    logger.debug('Loading wind fan data from %s', pth)
                    fan_df = pl.read_csv(pth)
                    fans.append(fan_df.to_pandas())
                    logger.debug('Wind fan data sample: %s', fans[-1].tail(2).values.tolist())
                except Exception as exc:
                    logger.error('Error loading wind fan data file %s: %s\n%s', pth, exc, traceback.format_exc())
                    raise
        
        # Load weather forecast data
        if weather_paths:
            logger.info('Loading weather forecast data...')
            for pth in weather_paths:
                try:
                    logger.debug('Loading weather forecast data from %s', pth)
                    weather_df = pd.read_csv(pth)
                    processed_weather = weather_data_process(config, weather_df)
                    weather_data.append(processed_weather)
                except Exception as exc:
                    logger.error('Error loading weather forecast data file %s: %s\n%s', pth, exc, traceback.format_exc())
                    raise
        
        # Merge wind tower data
        if not tower:
            error_msg = 'No available wind tower data'
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info('Merging wind tower data...')
        tower = pd.concat(tower, ignore_index=True)
        
        # Merge wind tower and fan data
        if fans:
            logger.info('Merging wind tower and fan data...')
            fans = pd.concat(fans, ignore_index=False)
            df = tower.merge(fans, on=time_col, how='outer')
        else:
            logger.warning('No wind fan data, only using wind tower data')
            df = tower
        
        # Convert target variable to appropriate units
        if target_name in df.columns:
            target_mean = df[target_name].mean()
            logger.debug('Average value of target variable %s: %f', target_name, target_mean)
            
            if target_mean > 10000:
                logger.info('Target variable value too large, divided by 1000')
                df[target_name] = df[target_name] / 1000
                
            if target_mean < 0.01:
                logger.info('Target variable value too small, multiplied by 1000')
                df[target_name] = df[target_name] * 1000
        else:
            logger.warning('Target variable %s not in data columns', target_name)
        
        # Standardize time column format
        logger.info('Standardizing time column format...')
        df[time_col] = df[time_col].apply(lambda x: str(x) + ' 00:00:00' if len(str(x)) == 10 else str(x))
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Release memory no longer needed
        del fans, tower
        if 'fan_df' in locals():
            del fan_df
        if 'weather_df' in locals():
            del weather_df
        gc.collect()
        
        logger.info('Data loading completed, data shape: %s, number of weather data sources: %d', df.shape, len(weather_data))
        return df, weather_data
        
    except Exception as exc:
        logger.error('Error loading data: %s\n%s', exc, traceback.format_exc())
        raise

def get_X_y_train(config: Dict[str, Any], taskname: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Get training data and task configuration.
    
    Load original data, merge weather data, add time features, and update task configuration.
    
    Args:
        config: Main configuration dictionary
        taskname: Task name
        
    Returns:
        tuple: (Processed data frame, updated task configuration)
    """
    logger.info('Getting training data...')
    
    try:
        # Get necessary parameters from configuration
        ahead_rows = config.get('Features', {}).get('AheadRows', 0)
        target_name = config.get('Features', {}).get('TargetName', None)
        time_col = config.get('Features', {}).get('TimeIndex', None)
        tasks = config.get('Task', None)
        
        # Verify necessary configuration items
        if not all([target_name, time_col, tasks, taskname in tasks]):
            error_msg = f'Missing necessary configuration item or task name {taskname} does not exist'
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Calculate interval and minutes to predict ahead
        interval = tasks[taskname]['RowInterval'] + ahead_rows
        n = interval * 15  # 15 minutes per row, converted to minutes
        logger.debug('Interval for task %s is %d rows, predicting %d minutes ahead', taskname, interval, n)
        
        # Get configuration file path
        try:
            config_pth = pkg_resources.resource_filename('ESPML', 'module/config/config.yaml')
            logger.debug('Using package resource path: %s', config_pth)
        except Exception as exc:
            logger.debug('Failed to get package resource path, using relative path: %s', exc)
            base_path = os.path.dirname(os.path.dirname(__file__))
            config_pth = os.path.join(base_path, 'config', 'config.yaml')
            logger.debug('Using relative path: %s', config_pth)
        
        # Load data
        logger.info('Loading original data...')
        df, weather_data = load_data(config)
        
        # Merge weather data
        if isinstance(weather_data, list) and weather_data:
            logger.info('Merging weather data...')
            df = merge_all_weather_data(config, df, weather_data, n)
        elif isinstance(weather_data, pd.DataFrame):
            logger.info('Merging single weather data source...')
            df = weather_data_merge(config, df, weather_data, n)
        else:
            logger.warning('No available weather data')
        
        # Add time features and update configuration
        logger.info('Adding time features...')
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
        
        logger.info('Training data loading completed, data shape: %s', df.shape)
        return df, task_config
        
    except Exception as exc:
        logger.error('Error getting training data: %s\n%s', exc, traceback.format_exc())
        raise

def get_X_test(
    config: Dict[str, Any], 
    taskname: str, 
    predict_time: Union[str, pd.Timestamp], 
    label_save_path: str = '', 
    pred_n: int = 1
) -> Tuple[pd.DataFrame, Dict[str, Any], float, pd.Timestamp, pd.Timestamp]:
    """Get test data and related configuration.
    
    Based on prediction time and task type, load and process appropriate test data.
    Also, handle label saving and time range calculation.
    
    Args:
        config: Main configuration dictionary
        taskname: Task name
        predict_time: Prediction time
        label_save_path: Path to save labels, default is empty string
        pred_n: Number of prediction steps, default is 1
        
    Returns:
        tuple: (Processed data frame, updated task configuration, target average value, prediction start time, prediction end time)
    """
    logger.info('Getting test data...')
    
    try:
        # Get necessary parameters from configuration
        target_name = config.get('Features', {}).get('TargetName', None)
        time_col = config.get('Features', {}).get('TimeIndex', None)
        tasks = config.get('Task', {})
        ahead_rows = config.get('Features', {}).get('AheadRows', 0)
        hourlagtime = config.get('Features', {}).get('HourSubmitLagTime', '0min')
        daylagtime = config.get('Features', {}).get('DaySubmitLagTime', '0hour')
        
        # Verify necessary configuration items
        if not all([target_name, time_col, tasks, taskname in tasks]):
            error_msg = f'Missing necessary configuration item or task name {taskname} does not exist'
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Calculate interval and minutes to predict ahead
        interval = tasks[taskname]['RowInterval'] * pred_n + ahead_rows
        n = interval * 15  # 15 minutes per row, converted to minutes
        
        # Process prediction time
        predict_end_time = pd.to_datetime(predict_time).floor('15min')
        logger.debug('Original prediction end time: %s', predict_end_time)
        
        # Get configuration file path
        try:
            config_pth = pkg_resources.resource_filename('ESPML', 'module/config/config.yaml')
            logger.debug('Using package resource path: %s', config_pth)
        except Exception as exc:
            logger.debug('Failed to get package resource path, using relative path: %s', exc)
            base_path = os.path.dirname(os.path.dirname(__file__))
            config_pth = os.path.join(base_path, 'config', 'config.yaml')
            logger.debug('Using relative path: %s', config_pth)
        
        # Calculate time offset based on task type
        if taskname.endswith('Hour'):
            logger.debug('Processing time offset for hourly task')
            if hourlagtime.endswith('min'):
                lag_time = int(hourlagtime[:-3])
                logger.debug('Hourly task time offset: %d minutes', lag_time)
            elif hourlagtime.endswith('hour'):
                lag_time = int(hourlagtime[:-4]) * 60
                logger.debug('Hourly task time offset: %d hours (%d minutes)', int(hourlagtime[:-4]), lag_time)
            predict_end_time += pd.Timedelta(minutes=lag_time)
            
        if taskname.endswith('Day'):
            logger.debug('Processing time offset for daily task')
            if daylagtime.endswith('min'):
                lag_time = int(daylagtime[:-3])
                logger.debug('Daily task time offset: %d minutes', lag_time)
            elif daylagtime.endswith('hour'):
                lag_time = int(daylagtime[:-4]) * 60
                logger.debug('Daily task time offset: %d hours (%d minutes)', int(daylagtime[:-4]), lag_time)
            predict_end_time += pd.Timedelta(minutes=lag_time)
        
        # Calculate prediction time range
        predict_start_time = predict_end_time - pd.Timedelta(minutes=n)
        predict_end_time = predict_end_time - pd.Timedelta(minutes=ahead_rows * 15)
        logger.info('Prediction time range: [%s, %s]', predict_start_time, predict_end_time)
        
        # Wait for data (if configured)
        predict_waiting = int(config.get('Features', {}).get('PredictWaiting', 0))
        if predict_waiting > 0:
            logger.info('Waiting %d seconds before loading data...', predict_waiting)
            time.sleep(predict_waiting)
        
        # Load data
        logger.info('Loading original data...')
        df, weather_data = load_data(config)
        
        # Merge weather data
        if isinstance(weather_data, list) and weather_data:
            logger.info('Merging weather data...')
            df = merge_all_weather_data(config, df, weather_data, n)
        elif isinstance(weather_data, pd.DataFrame):
            logger.info('Merging single weather data source...')
            df = weather_data_merge(config, df, weather_data, n)
        else:
            logger.warning('No available weather data')
        
        # Handle label saving (if needed)
        if label_save_path and taskname == 'Forecast4Hour':
            logger.info('Processing hourly prediction label saving...')
            
            # If label file exists, read and process
            if os.path.exists(label_save_path):
                logger.debug('Reading existing label file: %s', label_save_path)
                label_df = pd.read_csv(label_save_path)
                label_df['date_time'] = label_df['date_time'].apply(
                    lambda x: str(x) + ' 00:00:00' if len(str(x)) == 10 else str(x)
                )
                label_df['date_time'] = pd.to_datetime(label_df['date_time'])
                max_date = label_df['date_time'].max()
                logger.debug('Existing label maximum date: %s', max_date)
                
                # Select new data as label
                res = df[[time_col, target_name]][
                    df[time_col] > max_date
                ].rename(columns={target_name: 'power'})
                logger.debug('New label data sample: %s', res.tail().to_dict())
            else:
                # If label file does not exist, create new file
                logger.info('Creating new label file for the first time...')
                os.makedirs(os.path.dirname(label_save_path), exist_ok=True)
                label_df = pd.DataFrame()
                res = df[[time_col, target_name]].rename(columns={target_name: 'power'})
            
            # If there is new data, save it to the label file
            if not res.empty:
                logger.info('There is new data to save to label file')
                res['exec_time'] = pd.Timestamp.today()
                res['date_time'] = res[time_col]
                res = res[['exec_time', 'date_time', 'power']]
                logger.debug('Data to be saved sample: %s', res.tail().to_dict())
                
                # Save data
                if os.path.exists(label_save_path):
                    res.to_csv(label_save_path, index=False, mode='a', header=None)
                    logger.debug('Saved data in append mode')
                else:
                    res.to_csv(label_save_path, index=False)
                    logger.debug('Created new file to save data')
                logger.info('Successfully saved label file')
            else:
                logger.info('No new label data')
                res = label_df
            
            # Calculate target average value
            mean_power = res['power'].mean()
            logger.debug('Target average value of label data: %f', mean_power)
        else:
            # If no label saving is needed, use the average value of current data
            mean_power = df[target_name].mean()
            logger.debug('Current data target average value: %f', mean_power)
        
        # Filter data within prediction time range
        logger.info('Filtering data within prediction time range...')
        df = df[
            (df[time_col] >= predict_start_time) & 
            (df[time_col] <= predict_end_time)
        ]
        
        # Handle specific task timestamps
        if not df.empty:
            if taskname in ['Forecast10Day', 'Forecast4Hour']:
                logger.debug('Handling timestamps for %s task', taskname)
                df = get_timestamps(df, time_col, start_time=predict_start_time, end_time=predict_end_time)
        else:
            error_msg = 'No data obtained within prediction time range, please check real-time data'
            logger.error(error_msg)
            # Note: Here, no exception is raised because the original code also continues execution
        
        # Add time features and update configuration
        logger.info('Adding time features...')
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
        
        logger.info('Test data retrieval completed, data shape: %s', df.shape)
        return df, task_config, mean_power, predict_start_time, predict_end_time
        
    except Exception as exc:
        logger.error('Error getting test data: %s\n%s', exc, traceback.format_exc())
        raise

def train_task(config: Dict[str, Any], task_name: str) -> None:
    """Execute model training task.
    
    Load training data, save debug information, and use ESPML for model training.
    
    Args:
        config: Main configuration dictionary
        task_name: Task name
    """
    logger.info('Starting training task: %s', task_name)
    
    try:
        # Get path for saving debug files
        debug_csv_save_path = config.get('Path', {}).get('OtherFile', '')
        if not debug_csv_save_path:
            logger.warning('Missing path for saving debug files, using default path')
            debug_csv_save_path = os.path.join(os.path.dirname(__file__), 'debug_files')
            
        # Ensure directory for debug files exists
        if not os.path.exists(debug_csv_save_path):
            os.makedirs(debug_csv_save_path, exist_ok=True)
            logger.debug('Created directory for debug files: %s', debug_csv_save_path)
        
        # Get training data and task configuration
        logger.info('Getting training data...')
        df, task_config = get_X_y_train(config, task_name)
        
        # Save training data for debugging
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        debug_file_path = os.path.join(debug_csv_save_path, f"{task_name}_train_{timestamp}.csv")
        logger.debug('Saving training data to: %s', debug_file_path)
        df.to_csv(debug_file_path, index=False)
        
        # Record task configuration
        logger.info('Task configuration: %s', task_config)
        
        # Train model
        logger.info('Starting model training...')
        ESPML(task_config, df).fit()
        
        logger.info('Task %s training completed successfully', task_name)
        
    except Exception as exc:
        logger.error('Error training task %s: %s\n%s', task_name, exc, traceback.format_exc())
        raise

def predict_task(config: Dict[str, Any], task_name: str) -> Optional[pd.DataFrame]:
    """Execute model prediction task.
    
    Load test data, use model to predict, and save prediction results.
    
    Args:
        config: Main configuration dictionary
        task_name: Task name
        
    Returns:
        Prediction result data frame, returns None if an error occurs
    """
    logger.info('Starting prediction task: %s', task_name)
    
    try:
        # Get current time as baseline for prediction
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.debug('Current time: %s', now)
        
        # Get necessary parameters from configuration
        time_col = config.get('Features', {}).get('TimeIndex', None)
        tasks = config.get('Task', None)
        ahead_rows = config.get('Features', {}).get('AheadRows', 0)
        
        # Verify necessary configuration items
        if not all([time_col, tasks, task_name in tasks]):
            error_msg = f'Missing necessary configuration item or task name {task_name} does not exist'
            logger.error(error_msg)
            return None
        
        # Calculate interval and minutes to predict ahead
        interval = tasks[task_name]['RowInterval'] + ahead_rows
        n = interval * 15  # 15 minutes per row, converted to minutes
        logger.debug('Interval for task %s is %d rows, predicting %d minutes ahead', task_name, interval, n)
        
        # Set path for saving prediction results
        predict_save_path = config.get('Path', {}).get('Prediction', '')
        if not predict_save_path:
            logger.warning('Missing path for saving prediction results, using default path')
            predict_save_path = os.path.join(os.path.dirname(__file__), 'predictions')
        
        # Set filename based on task type
        task_save_task_index = {'Forecast4Hour': '4h.csv'}
        for i in range(1, 11):
            task_save_task_index[f'Forecast{i}Day'] = f'{i}d.csv'
            
        if task_name not in task_save_task_index:
            logger.warning('Task %s does not have predefined save file name, using default name', task_name)
            task_save_task_index[task_name] = f'{task_name}.csv'
            
        predict_save_path = os.path.join(predict_save_path, task_save_task_index[task_name])
        logger.debug('Prediction results save path: %s', predict_save_path)
        
        # Set label save path
        label_save_path = os.path.join(config.get('Path', {}).get('Prediction', ''), 'real_wind_power.csv')
        logger.debug('Label save path: %s', label_save_path)
        
        # Set model path
        model_path = os.path.join(config.get('Path', {}).get('Model', ''), task_name)
        logger.debug('Model path: %s', model_path)
        
        # Set debug file save path
        debug_csv_save_path = config.get('Path', {}).get('OtherFile', '')
        if not debug_csv_save_path:
            logger.warning('Debug file save path not configured, using default path')
            debug_csv_save_path = os.path.join(os.path.dirname(__file__), 'debug_files')
            
        # Ensure debug file directory exists
        if not os.path.exists(debug_csv_save_path):
            os.makedirs(debug_csv_save_path, exist_ok=True)
            logger.debug('Created debug file directory: %s', debug_csv_save_path)
        
        # Check if model exists
        if os.path.exists(model_path):
            # Get the latest model version
            try:
                model_versions = [int(num) for num in os.listdir(model_path) if num.isdigit()]
                if not model_versions:
                    logger.error('No valid model versions in model directory %s', model_path)
                    return None
                    
                seq = max(model_versions)
                logger.debug('Found latest model version: %d', seq)
                
                # Find valid model version
                while not os.path.exists(os.path.join(model_path, str(seq), 'automl', 'val_res')) and seq > 0:
                    logger.debug('Model version %d is incomplete, trying previous version', seq)
                    seq -= 1
                    
                if seq <= 0:
                    logger.error('No available model for task %s', task_name)
                    return None
                    
                logger.info('Task %s will use model version %d for prediction', task_name, seq)
            except Exception as exc:
                logger.error('Error getting model version: %s\n%s', exc, traceback.format_exc())
                return None
        else:
            logger.error('Model path %s for task %s does not exist', model_path, task_name)
            return None
        
        # Get test data
        logger.info('Getting test data...')
        X_val, task_config, history_mean_value, predict_start_time, predict_end_time = get_X_test(
            config, task_name, now, label_save_path
        )
        
        # Save test data for debugging
        if isinstance(X_val, pd.DataFrame):
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            debug_file_path = os.path.join(debug_csv_save_path, f"{task_name}_predict_{timestamp}.csv")
            logger.debug('Saving test data to: %s', debug_file_path)
            X_val.to_csv(debug_file_path, index=False)
        
        # If test data is not empty, make prediction
        if not X_val.empty:
            logger.info('Starting prediction...')
            res = pd.DataFrame({
                'base_time': X_val[time_col], 
                'date_time': pd.to_datetime(X_val[time_col]) + pd.Timedelta(minutes=n)
            })
            
            try:
                # Record task configuration
                logger.info('Task configuration: %s', task_config)
                
                # Make prediction
                pred = predict(task_config, X_val, save=False)
                res['power'] = pred.values
                logger.info('Task %s prediction completed successfully', task_name)
                
            except Exception as exc:
                logger.error('Prediction failed: %s\n%s', exc, traceback.format_exc())
                # If prediction fails, use historical average value
                res['power'] = history_mean_value
                logger.warning('Using historical average value %f as prediction result', history_mean_value)
        else:
            # If test data is empty, create dataframe within time range
            logger.error('Task %s got empty test data, please check real-time data', task_name)
            
            # Create dataframe within time range
            base_times = [t for t in pd.date_range(predict_start_time, predict_end_time, freq='15min')]
            date_times = [t for t in pd.date_range(
                predict_start_time + pd.Timedelta(minutes=n), 
                predict_end_time + pd.Timedelta(minutes=n), 
                freq='15min'
            )]
            
            res = pd.DataFrame({
                'base_time': base_times, 
                'date_time': date_times
            })
            res['power'] = history_mean_value
            logger.warning('Using historical average value %f to fill prediction results', history_mean_value)
        
        # Add execution point and time
        res['exec_point'] = res['base_time'].max()
        res['exec_time'] = pd.Timestamp.today()
        
        # Ensure prediction result directory exists
        os.makedirs(os.path.dirname(predict_save_path), exist_ok=True)
        
        # Save prediction results
        if os.path.exists(predict_save_path):
            logger.info('Saving prediction results to existing file in append mode: %s', predict_save_path)
            res.to_csv(predict_save_path, index=False, mode='a', header=None)
        else:
            logger.info('Saving prediction results to new file: %s', predict_save_path)
            res.to_csv(predict_save_path, index=False)
        
        return res
        
    except Exception as exc:
        logger.error('Error in prediction task %s: %s\n%s', task_name, exc, traceback.format_exc())
        return None

def incrml_train(config: Dict[str, Any], task_name: str, mode: str) -> None:
    """Initialize incremental learning training task.
    
    Performs one-time training or sets up scheduled training tasks based on the mode.
    
    Args:
        config: Main configuration dictionary
        task_name: Task name
        mode: Running mode, 'test' for test mode, others for normal mode
    """
    logger.info('Initializing incremental learning training task: %s, mode: %s', task_name, mode)
    
    try:
        # Get necessary parameters from configuration
        tasks = config.get('Task', None)
        if not tasks or task_name not in tasks:
            error_msg = f'Missing necessary task configuration or task name {task_name} does not exist'
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        start_time = tasks[task_name].get('TrainTime', None)
        if not start_time:
            error_msg = f'Task {task_name} has no configured training time'
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        model_path = config.get('Path', {}).get('Model', '')
        if not model_path:
            logger.warning('Model path not configured, using default path')
            model_path = os.path.join(os.path.dirname(__file__), 'models')
        
        # Initialize logger
        init_logger(config, task_name, 'train')
        
        # Test mode: perform training once directly
        if mode == 'test':
            logger.info('Test mode: performing training once directly')
            train_task(config, task_name)
            return
        
        # If model path doesn't exist or is empty, perform training once first
        if not os.path.exists(model_path) or len(os.listdir(model_path)) < 1:
            logger.info('Model path does not exist or is empty, performing training once first')
            train_task(config, task_name)
        
        # Create scheduled task
        logger.info('Creating scheduled task')
        scheduled_job = partial(train_task, config=config, task_name=task_name)
        
        # Set up scheduled tasks based on training time
        if isinstance(start_time, list):
            logger.info('Setting multiple training time points: %s', start_time)
            for stime in start_time:
                logger.debug('Adding daily training task at %s', stime)
                every().day.at(stime).do(scheduled_job)
        else:
            logger.info('Setting single training time point: %s', start_time)
            every().day.at(start_time).do(scheduled_job)
        
        # The following code will never execute, but is kept for possible future use
        if False:  # pragma: no cover
            # Calculate next run time
            next_run = next_run_time()
            if next_run:
                now = datetime.now()
                sleep_seconds = (next_run - now).total_seconds() - 1
                if sleep_seconds > 0:
                    logger.debug('Waiting %d seconds before running task', sleep_seconds)
                    time.sleep(sleep_seconds)
                else:
                    logger.debug('Running pending tasks immediately')
                    run_pending()
            run_pending()
            
        logger.info('Incremental learning training task initialization completed')
        
    except Exception as exc:
        logger.error('Error initializing incremental learning training task: %s\n%s', exc, traceback.format_exc())
        raise

def incrml_predict(config: Dict[str, Any], task_name: str, mode: str) -> None:
    """Initialize incremental learning prediction task.
    
    Performs one-time prediction or sets up scheduled prediction tasks based on the mode.
    Sets different scheduling strategies based on task type (hourly or daily).
    
    Args:
        config: Main configuration dictionary
        task_name: Task name
        mode: Running mode, 'test' for test mode, others for normal mode
    """
    logger.info('Initializing incremental learning prediction task: %s, mode: %s', task_name, mode)
    
    try:
        # Get necessary parameters from configuration
        tasks = config.get('Task', None)
        if not tasks or task_name not in tasks:
            error_msg = f'Missing necessary task configuration or task name {task_name} does not exist'
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        start_time = tasks[task_name].get('SubmitTime', None)
        if not start_time:
            error_msg = f'Task {task_name} has no configured submission time'
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Initialize logger
        init_logger(config, task_name, 'predict')
        
        # Test mode: perform prediction once directly
        if mode == 'test':
            logger.info('Test mode: performing prediction once directly')
            predict_task(config, task_name)
            return
        
        # Create scheduled task
        logger.info('Creating scheduled task')
        scheduled_job = partial(predict_task, config=config, task_name=task_name)
        
        # Set up scheduled tasks based on task type and submission time format
        is_hour_task = task_name.endswith('Hour')
        
        if is_hour_task and isinstance(start_time, list):
            # Hourly task, multiple time points
            logger.info('Hourly task, setting multiple time points within each hour: %s', start_time)
            for stime in start_time:
                logger.debug('Adding prediction task at %s every hour', stime)
                every().hours.at(stime).do(scheduled_job)
                
        elif is_hour_task and isinstance(start_time, str):
            # Hourly task, single time point
            logger.info('Hourly task, setting single time point within each hour: %s', start_time)
            every().hours.at(start_time).do(scheduled_job)
            
        elif isinstance(start_time, list):
            # Daily task, multiple time points
            logger.info('Daily task, setting multiple daily time points: %s', start_time)
            for stime in start_time:
                logger.debug('Adding prediction task at %s every day', stime)
                every().days.at(stime).do(scheduled_job)
                
        else:
            # Daily task, single time point
            logger.info('Daily task, setting single daily time point: %s', start_time)
            every().days.at(start_time).do(scheduled_job)
        
        # The following code will not execute, kept for possible future use
        if False:  # pragma: no cover
            # Calculate next run time
            next_run = next_run_time()
            if next_run:
                now = datetime.now()
                sleep_seconds = (next_run - now).total_seconds() - 1
                if sleep_seconds > 0:
                    logger.debug('Waiting %d seconds before running task', sleep_seconds)
                time.sleep(sleep_seconds)
            else:
                logger.debug('Running pending tasks immediately')
                run_pending()
        run_pending()
            
        logger.info('Incremental learning prediction task initialization completed')
        
    except Exception as exc:
        logger.error('Error initializing incremental learning prediction task: %s\n%s', exc, traceback.format_exc())
        raise

def backtracking(
    config: Dict[str, Any], 
    start_time: Optional[str] = None, 
    end_time: Optional[str] = None, 
    only_predict: bool = False
) -> None:
    """Perform model backtesting evaluation.
    
    Within the specified time range, uses historical data to train models and make predictions,
    recording metadata information for each training and prediction session.
    
    Args:
        config: Main configuration dictionary
        start_time: Backtest start time, format 'YYYY-MM-DD'
        end_time: Backtest end time, format 'YYYY-MM-DD'
        only_predict: Whether to only make predictions without training models
    """
    logger.info('Starting model backtesting evaluation, time range: %s to %s', start_time, end_time)
    
    # Initialize variables
    meta_info = {}
    save_path = ""
    meta_save_path = ""
    
    try:
        # Set up log handlers
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
                
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.debug('Log handlers setup completed')
        
        # Initialize metadata information dictionary
        meta_info = {
            'training_type': [], 
            'train_end_date': [], 
            'model_id': [], 
            'exec_time': [], 
            'state': []
        }
        
        # Get necessary parameters from configuration
        tasks = config.get('Task', None)
        if not tasks:
            error_msg = 'Missing necessary task configuration'
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        ahead_rows = config.get('Features', {}).get('AheadRows', 0)
        time_col = config.get('Features', {}).get('TimeIndex', None)
        target_name = config.get('Features', {}).get('TargetName', None)
        
        # Verify necessary configuration items
        if not all([time_col, target_name]):
            error_msg = 'Missing necessary configuration items (time column or target column)'
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Set save path
        save_path = config.get('Path', {}).get('Prediction', '')
        if not save_path:
            logger.warning('Prediction results save path not configured, using default path')
            save_path = os.path.join(os.path.dirname(__file__), 'predictions')
            
        # Ensure save path exists
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
            logger.debug('Created prediction results save path: %s', save_path)
        
        # Set metadata save path
        timestamp = pd.Timestamp.today().strftime('%Y%m%d_%H%M%S')
        meta_save_path = os.path.join(save_path, f"meta_{timestamp}.csv")
        logger.debug('Metadata save path: %s', meta_save_path)
        
        # Iterate through all tasks
        for taskname in tasks.keys():
            logger.info('Processing task: %s', taskname)
                
            # Get task configuration
            task_yaml_config = tasks[taskname]
            if 'RowInterval' not in task_yaml_config:
                logger.warning('Task %s is missing RowInterval configuration, skipping', taskname)
                continue
                
            if 'TrainTime' not in task_yaml_config:
                logger.warning('Task %s is missing TrainTime configuration, skipping', taskname)
                continue
                
            # Calculate interval and prediction lead time in minutes
            interval = task_yaml_config['RowInterval'] + ahead_rows
            n = interval * 15  # 15 minutes per row, converted to minutes
            logger.debug('Task %s has interval of %d rows, predicting %d minutes ahead', taskname, interval, n)
                
            # Get configuration file path
            config_pth = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'project', 'WindPower', 'config.yaml')
            logger.debug('Configuration file path: %s', config_pth)
            
            # Set prediction results save path
            predict_save_path = os.path.join(save_path, f'{taskname}.csv')
            logger.debug('Prediction results save path: %s', predict_save_path)
            
            # Load data
            logger.info('Loading data...')
            df, weather_data = load_data(config)
                
            # Merge weather data
            if isinstance(weather_data, list) and weather_data:
                logger.info('Merging weather data...')
                df = merge_all_weather_data(config, df, weather_data, n)
            elif isinstance(weather_data, pd.DataFrame):
                logger.info('Merging single weather data source...')
                df = weather_data_merge(config, df, weather_data, n)
                
            # Add time features
            logger.info('Adding time features...')
            task_df, task_config = add_time_feature_process(
                df, 
                time_col=time_col, 
                target_name=target_name, 
                config_pth=config_pth, 
                interval=interval
            )
                
            # Iterate through training time points
            for train_time in task_yaml_config['TrainTime']:
                # Set backtest time range
                if not start_time or not end_time:
                    logger.warning('Unspecified backtest time range, using default range')
                    start_date = (pd.Timestamp.today() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
                    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
                else:
                    start_date = start_time
                    end_date = end_time
                
                beginning_time = f'{start_date} {train_time}'
                finishing_time = f'{end_date} {train_time}'
                logger.info('Backtest time range: %s to %s', beginning_time, finishing_time)
                
                # Generate date range
                date_range = pd.date_range(start=beginning_time, end=finishing_time, freq='D')
                logger.debug('Generated %d backtest dates', len(date_range))
                    
                # Iterate through each backtest date
                for date in date_range:
                    logger.info('Processing backtest date: %s', date)
                    
                    # Set model save path
                    task_result_path = os.path.join(config.get('Path', {}).get('Model', ''), f'{taskname}_backtrack')
                    if not os.path.exists(task_result_path):
                        os.makedirs(task_result_path, exist_ok=True)
                        logger.debug('Created model save path: %s', task_result_path)
                        
                    # Determine model version number
                    try:
                        if not os.path.exists(task_result_path) or len(os.listdir(task_result_path)) == 0:
                            seq = 1
                        else:
                            model_versions = [int(num) for num in os.listdir(task_result_path) if num.isdigit()]
                            seq = max(model_versions) + 1 if model_versions else 1
                        logger.debug('Current model version number: %d', seq)
                    except Exception as exc:
                        logger.error('Error getting model version number: %s\n%s', exc, traceback.format_exc())
                        seq = 1
                        
                    # Record metadata
                    meta_info['training_type'].append(taskname)
                    meta_info['train_end_date'].append(date)
                    meta_info['model_id'].append(seq)
                    meta_info['exec_time'].append(pd.Timestamp.today())
                    
                    # Prepare training data
                    logger.info('Preparing training data, end date: %s', date)
                    temp_df = task_df[task_df[time_col] < date].copy()
                    temp_df.dropna(subset=[target_name], inplace=True)
                    temp_df['label'] = temp_df[target_name].shift(-interval)
                    
                    # Update task configuration
                    task_config = update_config(config, task_config, f'{taskname}_backtrack')
                    
                    # Train model
                    try:
                        if not only_predict:
                            logger.info('Starting model training...')
                            ESPML(task_config, temp_df[~temp_df['label'].isnull()]).fit()
                            logger.info('Model training completed successfully')
                            meta_info['state'].append('success')
                        else:
                            logger.info('Skipping model training (only_predict=True)')
                            meta_info['state'].append('success')  # If only predicting, also mark as success
                    except Exception as exc:
                        logger.error('Model training failed: %s\n%s', exc, traceback.format_exc())
                        meta_info['state'].append('failed')
                    
                    # Save metadata
                    pd.DataFrame(meta_info).to_csv(meta_save_path, index=False)
                    logger.debug('Metadata saved to: %s', meta_save_path)
                        
                    # Perform prediction
                    result_df = None
                    
                    # Process hourly prediction task
                    if taskname == 'Forecast4Hour':
                        logger.info('Processing hourly prediction task...')
                        initial_predict_start_time = date - pd.Timedelta(hours=4)
                        initial_predict_end_time = date
                            
                        # Predict every 15 minutes, 16 times
                        for step in range(16):
                            predict_start_time = initial_predict_start_time + pd.Timedelta(minutes=15 * (step + 1))
                            predict_end_time = initial_predict_end_time + pd.Timedelta(minutes=15 * (step + 1))
                            y_start_time = predict_start_time + pd.Timedelta(hours=4)
                            y_end_time = predict_end_time + pd.Timedelta(hours=4)
                            
                            logger.debug('Step %d prediction, time range: [%s, %s]', step+1, predict_start_time, predict_end_time)
                                
                            # Prepare test data
                            X_test = task_df[
                                (task_df[time_col] >= predict_start_time) & 
                                (task_df[time_col] <= predict_end_time)
                            ]
                            
                            # Get actual values
                            y_test = task_df[
                                (task_df[time_col] >= y_start_time) & 
                                (task_df[time_col] <= y_end_time)
                            ][target_name]
                                
                            # If there is test data, perform prediction
                            if not X_test.empty:
                                logger.debug('Test data is not empty, shape: %s', X_test.shape)
                                X_test = get_timestamps(X_test, time_col, start_time=predict_start_time, end_time=predict_end_time)
                                
                                try:
                                    y_pred = predict(task_config, X_test, save=False)
                                    min_length = min(len(y_pred.values), len(y_test.values))
                                    
                                    result_df = pd.DataFrame({
                                        'base_time': pd.date_range(predict_start_time, predict_end_time, freq='15min').values[-min_length:],
                                        'date_time': pd.date_range(y_start_time, y_end_time, freq='15min').values[-min_length:], 
                                        'y_pred': y_pred.values[-min_length:]
                                    })
                                    
                                    # If there are actual values, add to results
                                    if not y_test.empty:
                                        result_df['y_true'] = y_test.values[-min_length:]
                                        
                                    logger.debug('Prediction succeeded, result shape: %s', result_df.shape)
                                except Exception as exc:
                                    logger.error('Prediction failed: %s\n%s', exc, traceback.format_exc())
                                    continue  # Skip current step, continue to next prediction
                            else:
                                # Handle case with no test data
                                logger.info('No test data, time range: %s to %s', predict_start_time, predict_end_time)
                                
                                # Get historical data average value as prediction
                                history = task_df[task_df[time_col] < y_start_time][target_name]
                                history_mean = history.mean() if not history.empty else 0
                                
                                # Determine data length
                                if y_test.empty:
                                    min_length = len(pd.date_range(predict_start_time, predict_end_time, freq='15min'))
                                    result_df = pd.DataFrame({
                                        'base_time': pd.date_range(predict_start_time, predict_end_time, freq='15min').values[-min_length:], 
                                        'date_time': pd.date_range(y_start_time, y_end_time, freq='15min').values[-min_length:]
                                    })
                                else:
                                    min_length = len(y_test.values)
                                    result_df = pd.DataFrame({
                                        'base_time': pd.date_range(predict_start_time, predict_end_time, freq='15min').values[-min_length:], 
                                        'date_time': pd.date_range(y_start_time, y_end_time, freq='15min').values[-min_length:], 
                                        'y_true': y_test.values[-min_length:]
                                    })
                                    
                                # Use historical average value as prediction
                                result_df['y_pred'] = history_mean
                                
                            # Add metadata information
                            if result_df is not None and not result_df.empty:
                                result_df['exec_point'] = pd.to_datetime(date)
                                result_df['exec_time'] = pd.Timestamp.today()
                                result_df['model_id'] = seq
                                
                                # Save prediction results
                                if os.path.exists(predict_save_path):
                                    result_df.to_csv(predict_save_path, index=False, mode='a', header=False)
                                else:
                                    result_df.to_csv(predict_save_path, index=False)
                    else:
                        # Handle other task type prediction
                        logger.info('Processing regular prediction task: %s', taskname)
                        
                        # Iterate through submission time points
                        for subtime in task_yaml_config.get('SubmitTime', []):
                            # Calculate prediction time window
                            predict_end_time = pd.to_datetime(str(date)[:10] + ' ' + subtime)
                            predict_start_time = predict_end_time - pd.Timedelta(minutes=n)
                            y_start_time = predict_start_time + pd.Timedelta(minutes=n)
                            y_end_time = predict_end_time + pd.Timedelta(minutes=n)
                            
                            logger.debug('Prediction time window: %s to %s', predict_start_time, predict_end_time)
                            logger.debug('Target time window: %s to %s', y_start_time, y_end_time)
                            
                            # Prepare test data
                            X_test = task_df[
                                (task_df[time_col] >= predict_start_time) & 
                                (task_df[time_col] <= predict_end_time)
                            ]
                            
                            # Get actual values
                            y_test = task_df[
                                (task_df[time_col] >= y_start_time) & 
                                (task_df[time_col] <= y_end_time)
                            ][target_name]
                            
                            # Execute prediction if there is test data
                            if not X_test.empty:
                                logger.debug('Test data is not empty, shape: %s', X_test.shape)
                                try:
                                    # Preprocess test data
                                    X_test = get_timestamps(X_test, time_col, start_time=predict_start_time, end_time=predict_end_time)
                                    
                                    # Execute prediction
                                    y_pred = predict(task_config, X_test, save=False)
                                    min_length = min(len(y_pred.values), len(y_test.values))
                                    
                                    # Create result DataFrame
                                    result_df = pd.DataFrame({
                                        'base_time': pd.date_range(predict_start_time, predict_end_time, freq='15min').values[-min_length:], 
                                        'date_time': pd.date_range(y_start_time, y_end_time, freq='15min').values[-min_length:], 
                                        'y_pred': y_pred.values[-min_length:]
                                    })
                                    
                                    # Add actual values
                                    if not y_test.empty:
                                        result_df['y_true'] = y_test.values[-min_length:]
                                    
                                    logger.debug('Prediction succeeded, result shape: %s', result_df.shape)
                                except Exception as exc:
                                    logger.error('Prediction failed: %s\n%s', exc, traceback.format_exc())
                                    continue  # Skip current prediction point, continue to next
                            else:
                                # Handle case with no test data
                                logger.info('No test data, time range: %s to %s', predict_start_time, predict_end_time)
                                
                                # Get historical data average value as prediction
                                history = task_df[task_df[time_col] < y_start_time][target_name]
                                history_mean = history.mean() if not history.empty else 0
                                
                                # Determine data length and create result DataFrame
                                if y_test.empty:
                                    min_length = len(pd.date_range(predict_start_time, predict_end_time, freq='15min'))
                                    result_df = pd.DataFrame({
                                        'base_time': pd.date_range(predict_start_time, predict_end_time, freq='15min').values[-min_length:], 
                                        'date_time': pd.date_range(y_start_time, y_end_time, freq='15min').values[-min_length:]
                                    })
                                else:
                                    min_length = len(y_test.values)
                                    result_df = pd.DataFrame({
                                        'base_time': pd.date_range(predict_start_time, predict_end_time, freq='15min').values[-min_length:], 
                                        'date_time': pd.date_range(y_start_time, y_end_time, freq='15min').values[-min_length:], 
                                        'y_true': y_test.values[-min_length:]
                                    })
                                
                                # Use historical average value as prediction
                                result_df['y_pred'] = history_mean
                            
                            # Add metadata information and save results
                            if result_df is not None and not result_df.empty:
                                result_df['exec_point'] = result_df['base_time'].max()
                                result_df['exec_time'] = pd.Timestamp.today()
                                result_df['model_id'] = seq
                                
                                # Save prediction results
                                if os.path.exists(predict_save_path):
                                    result_df.to_csv(predict_save_path, index=False, mode='a', header=False)
                                else:
                                    result_df.to_csv(predict_save_path, index=False)
                        
                        logger.info('Task %s prediction task completed on date %s', taskname, date)
        
    except Exception as exc:
        # Catch all unhandled exceptions
        error_msg = f'Backtesting process encountered exception: {exc}'
        logger.error('%s\n%s', error_msg, traceback.format_exc())
        raise RuntimeError(error_msg) from exc
    
    finally:
        # Clean up work
        if meta_info and meta_save_path:
            try:
                # Ensure metadata is saved
                pd.DataFrame(meta_info).to_csv(meta_save_path, index=False)
                logger.debug('Metadata saved to: %s', meta_save_path)
            except Exception as e:
                logger.error('Failed to save final metadata: %s', e)
        
        # Record completion information
        logger.info('Backtesting process completed') 