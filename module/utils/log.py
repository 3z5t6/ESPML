#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logging utility module

Provides logging initialization, logger retrieval, and time measurement decorator functions.
Mainly used to unify log format and level across the entire project, making debugging and problem tracking easier.
"""

import os
import time
import logging
from typing import Any, Callable, Optional, TypeVar, cast

from module.utils.result_saver import ResultSaver

# Define type variable for decorator
F = TypeVar('F', bound=Callable[..., Any])


def init_log(logger: logging.Logger, exp_dir: str, debug_dir: str) -> None:
    """
    Initialize logger
    
    Configure logger handlers, including an INFO level log file for front-end display,
    a DEBUG level debug log file, and a console output handler.

    Args:
        logger: Logger object to configure
        exp_dir: Front-end display INFO level log file path
        debug_dir: DEBUG level debug log file path
    """
    # Only initialize if the logger has no handlers
    if not logger.handlers:
        # Create log directory
        ResultSaver.mkdir(os.path.dirname(exp_dir))
        ResultSaver.mkdir(os.path.dirname(debug_dir))
        
        # Set log format
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format)
        
        # Create front-end display INFO level log file handler
        info_handler = logging.FileHandler(exp_dir, encoding='utf-8')
        info_handler.setFormatter(formatter)
        info_handler.setLevel(logging.INFO)
        logger.addHandler(info_handler)
        
        # Create debug log file handler
        debug_handler = logging.FileHandler(debug_dir, encoding='utf-8')
        debug_handler.setFormatter(formatter)
        debug_handler.setLevel(logging.DEBUG)
        logger.addHandler(debug_handler)
        
        # Create console output handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get logger
    
    Create and configure a logger object, set to DEBUG level and disable propagation.

    Args:
        name: Logger name, usually use __name__

    Returns:
        Configured logger object
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return logger


def timeit_to_logger(logger: Optional[logging.Logger] = None) -> Callable[[F], F]:
    """
    Time measurement decorator
    
    Measure function execution time and record it to the log.

    Args:
        logger: Logger used to record execution time, if None then create a new logger

    Returns:
        Decorator function
    """
    # If no logger is provided, create a new one
    if logger is None:
        logger = get_logger(__name__)
    
    # If the logger has no handlers, initialize it
    if not logger.handlers:
        try:
            # Generate log file path
            log_dir = os.path.join(os.getcwd(), 'logs')
            os.makedirs(log_dir, exist_ok=True)
            
            log_path = os.path.join(log_dir, 'autofe.log')
            debug_path = os.path.join(log_dir, 'autofe_debug.log')
            
            # Initialize logger
            init_log(logger, log_path, debug_path)
        except Exception as e:
            print(f"Error initializing logger: {e}")

    def decorator(func: F) -> F:
        """
        Decorator function
        
        Args:
            func: Function to be decorated
            
        Returns:
            Decorated function
        """
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """
            Wrapper function, measure execution time and record to log
            
            Args:
                *args: Positional arguments passed to the decorated function
                **kwargs: Keyword arguments passed to the decorated function
                
            Returns:
                Return value of the decorated function
            """
            # Record start time
            start_time = time.time()
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Calculate execution time
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Format function name for readability
            function_name = func.__name__.replace('_', ' ')
            
            # Record execution time
            logger.debug(f'{function_name} execution time: {elapsed_time:.2f} seconds')
            
            return result
            
        return cast(F, wrapper)
        
    return decorator