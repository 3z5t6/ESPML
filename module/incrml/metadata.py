#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metadata management module

This module provides functionality for managing metadata in incremental learning scenarios, including tracking sequence numbers, states, and data sources.
Implements singleton pattern to ensure only one metadata instance exists throughout the application lifecycle.
"""

import os
from typing import Dict, List, Optional, Any, Union
from copy import deepcopy

from module.utils.result_saver import ResultSaver
from module.utils.log import get_logger

# Configure logger
logger = get_logger(__name__)


class MetaData:
    """
    Metadata management class
    
    Manages metadata in incremental learning scenarios, including tracking sequence numbers, states, and data sources.
    Implements singleton pattern to ensure only one metadata instance exists throughout the application lifecycle.
    """
    # Singleton pattern related attributes
    __species: Optional['MetaData'] = None  # Store unique instance
    __first_init: bool = True  # Mark whether it is the first initialization

    def __new__(cls, *args: Any, **kwargs: Any) -> 'MetaData':
        """
        Implement singleton pattern __new__ method
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            MetaData instance
        """
        if cls.__species is None:
            logger.debug("Create metadata manager instance")
            cls.__species = object.__new__(cls)
        return cls.__species

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize metadata manager
        
        If it is the first initialization, load or create metadata.
        If already initialized, do not perform any operations.
        
        Args:
            config: Configuration parameter dictionary, containing author name, task name, task ID, etc.
        """
        # If not the first initialization, directly return
        if not self.__first_init:
            return
            
        # Mark as initialized
        self.__class__.__first_init = False
        
        # If no configuration is provided, initialize to empty state
        if config is None:
            logger.warning("No configuration parameters provided, initializing to empty state")
            self.path = None
            self.meta = {'seq': [], 'state': [], 'data_source': []}
            self.seq = 1
            self.data_source: List[str] = []
            self.cur_seq = -1
            return
            
        # Determine metadata storage path
        try:
            # If SaveModelPath is None, use default path
            if config['IncrML'].get('SaveModelPath') is None:
                self.path = ResultSaver.generate_res_path(
                    os.getcwd(), 'results', 
                    str(config['AuthorName']), 
                    str(config['TaskName']), 
                    str(config['JobID']), 
                    'metadata'
                )
            else:
                # Otherwise use specified path
                self.path = os.path.join(config['IncrML']['SaveModelPath'], 'metadata')
                
            logger.info(f"Metadata storage path: {self.path}")
                
            # If metadata file exists, load it
            if os.path.exists(self.path):
                logger.info("Load existing metadata")
                self.meta = ResultSaver.load_json(self.path)
                
                # Iterate through sequences in reverse order to find the most recent success state
                for i in range(len(self.meta['seq']) - 1, -1, -1):
                    if self.meta['state'][i] == 'success':
                        self.seq = self.meta['seq'][i] + 1
                        self.data_source = self.meta['data_source'][i]
                        self.cur_seq = self.meta['seq'][i]
                        logger.info(f"Found the most recent success state, current sequence number: {self.cur_seq}, next sequence number: {self.seq}")
                        return
                
                # If no success state is found, initialize to default value
                logger.info("No success state found, initializing to default value")
                self.seq = 1
                self.data_source = []
                self.cur_seq = -1
            else:
                # If metadata file does not exist, create new metadata
                logger.info("Create new metadata")
                self.meta = {'seq': [], 'state': [], 'data_source': []}
                self.seq = 1
                self.data_source = []
                self.cur_seq = -1
                
        except Exception as e:
            logger.error(f"Error initializing metadata: {e}")
            # Initialize to default value when error occurs
            self.path = None
            self.meta = {'seq': [], 'state': [], 'data_source': []}
            self.seq = 1
            self.data_source = []
            self.cur_seq = -1

    def update(self, state: Dict[str, Any]) -> None:
        """
        Update metadata state
        
        Add new state information to metadata and save to file.
        
        Args:
            state: State information dictionary, containing 'seq', 'state', 'data_source' keys
        """
        try:
            # Update metadata
            for k, v in state.items():
                if k in self.meta:
                    self.meta[k].append(v)
                else:
                    logger.warning(f"Unknown metadata key: {k}")
                    
            # If state is success, update current sequence number and next sequence number
            if state.get('state') == 'success':
                self.cur_seq = state.get('seq', self.cur_seq)
                self.seq = self.cur_seq + 1
                logger.info(f"Update metadata successfully, current sequence number: {self.cur_seq}, next sequence number: {self.seq}")
                
            # Save metadata to file
            if self.path:
                ResultSaver.save_json(self.meta, self.path)
                logger.info(f"Metadata saved to: {self.path}")
            else:
                logger.warning("Metadata storage path not set, cannot save")
                
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
            raise