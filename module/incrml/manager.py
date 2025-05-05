#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data monitoring module

This module provides functionality for monitoring data changes, including detecting new data, updating historical data records, etc.
Mainly used in incremental learning scenarios to monitor changes in data sources and trigger corresponding processing workflows.
"""

import os
import glob
from typing import List

from module.incrml.metadata import MetaData
from module.utils.log import get_logger

# Configure logger
logger = get_logger(__name__)


class DataMonitoring:
    """
    Data monitoring class
    
    Monitors changes in data source and supports local directory mode.
    """

    def __init__(self, data_address: str, mode: str = 'local_dir'):
        """
        Initialize data monitoring object
        
        Args:
            data_address: Data source address, for local_dir mode, this is a local directory path
            mode: Data source mode, currently supports 'local_dir' (local directory)
        """
        self.mode = mode
        self.data_address = data_address
        self.historical_files: List[str] = MetaData().data_source
        self.new_files: List[str] = []
        
        logger.info(f"Initialize data monitoring object, data source address: {data_address}, mode: {mode}")
        logger.info(f"Number of historical data files: {len(self.historical_files)}")

    def check_data_change(self) -> bool:
        """
        Check if there is new data in the data source
        
        Detect changes in the data source and verify if the new data contains all historical data.
        
        Returns:
            If data changes are detected, return True, otherwise return False
        
        Raises:
            ValueError: When new data does not contain all historical data, an exception is thrown
        """
        if self.mode == 'local_dir':
            # Get all files in the directory
            try:
                self.new_files = glob.glob(os.path.join(self.data_address, '*'))
                
                # Exclude metadata folder
                if 'metadata' in self.new_files:
                    self.new_files.remove('metadata')
                    
                logger.info(f"Detected {len(self.new_files)} files in the data source")
                
                # Verify if the new data contains all historical data
                if not set(self.historical_files).issubset(set(self.new_files)):
                    error_msg = 'New data does not contain all historical data, please check if historical data has been modified!'
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Check if there are new files
                if self.new_files != self.historical_files:
                    logger.info("Data changes detected")
                    return True
                else:
                    logger.info("No data changes detected")
                    return False
            except Exception as e:
                logger.error(f"Error checking data changes: {e}")
                raise
        else:
            logger.error(f"Unsupported data source mode: {self.mode}")
            raise ValueError(f"Unsupported data source mode: {self.mode}")

    def update(self) -> None:
        """
        Update historical data records
        
        Update the historical file list with the current new file list
        """
        logger.info(f"Update historical data records, from {len(self.historical_files)} files to {len(self.new_files)} files")
        self.historical_files = self.new_files.copy()  # Use copy() to avoid reference issues

    def get_history_files(self) -> List[str]:
        """
        Get historical data file list
        
        Returns:
            List of historical data file paths
        """
        if self.mode == 'local_dir':
            return self.historical_files.copy()  # Return copy to avoid external modification
        else:
            logger.error(f"Unsupported data source mode: {self.mode}")
            raise ValueError(f"Unsupported data source mode: {self.mode}")

    def get_new_files(self) -> List[str]:
        """
        Get new data file list
        
        Returns:
            List of new data file paths
        """
        if self.mode == 'local_dir':
            return self.new_files.copy()  # Return copy to avoid external modification
        else:
            logger.error(f"Unsupported data source mode: {self.mode}")
            raise ValueError(f"Unsupported data source mode: {self.mode}")


# To maintain backward compatibility, keep the misspelled method name
DataMonitoring.get_hitory_files = DataMonitoring.get_history_files