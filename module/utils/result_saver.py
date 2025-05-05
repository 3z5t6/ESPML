#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ResultSaver class

This module provides a ResultSaver class for handling data saving and loading operations,
supporting JSON, Pickle, and CSV file formats.
"""

import os
import pickle
import json
from typing import Any, Dict, List, Tuple, Union, Optional

import pandas as pd


class ResultSaver:
    """
    ResultSaver class
    
    Provides a series of static methods for handling data saving and loading operations,
    supporting JSON, Pickle, and CSV file formats.
    """

    @staticmethod
    def save_json(content: Union[Dict, List, str], *args) -> None:
        """
        Save content as JSON file
        
        Args:
            content: Content to save, can be dictionary, list, or string
            *args: File path components, passed to generate_res_path method
        """
        try:
            file_name = ResultSaver.generate_res_path(*args)
            
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, indent=2)
                
            ResultSaver.to_local_json(file_name, content)
        except Exception as e:
            raise RuntimeError(f"Error saving JSON file: {str(e)}")

    @staticmethod
    def save_pickle(obj: Any, *args) -> None:
        """
        Save object as Pickle file
        
        Args:
            obj: Object to save
            *args: File path components, passed to generate_res_path method
        """
        try:
            file_name = ResultSaver.generate_res_path(*args)
            
            dir_name = os.path.dirname(file_name)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
                
            with open(file_name, 'wb') as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise RuntimeError(f"Error saving Pickle file: {str(e)}")

    @staticmethod
    def save_csv(data: pd.DataFrame, index: bool = False, *args) -> None:
        """
        Save DataFrame as CSV file
        
        Args:
            data: DataFrame to save
            index: Whether to save row index, default is False
            *args: File path components, passed to generate_res_path method
        """
        try:
            file_name = ResultSaver.generate_res_path(*args)
            
            dir_name = os.path.dirname(file_name)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
                
            data.to_csv(file_name, index=index)
        except Exception as e:
            raise RuntimeError(f"Error saving CSV file: {str(e)}")


    @staticmethod
    def load_json(*args) -> Dict[str, Any]:
        """
        Load JSON file
        
        Args:
            *args: File path components, passed to generate_res_path method
            
        Returns:
            Loaded JSON data, usually in dictionary form
            
        Raises:
            RuntimeError: When file does not exist or format is incorrect
        """
        try:
            file_name = ResultSaver.generate_res_path(*args)
            
            with open(file_name, 'rb') as f:
                data_dict = json.loads(f.read())
                return data_dict
        except FileNotFoundError:
            raise RuntimeError(f"File not found: {file_name}")
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON format: {file_name}")
        except Exception as e:
            raise RuntimeError(f"Error reading JSON file: {str(e)}")

    @staticmethod
    def load_pickle(*args) -> Any:
        """
        Load Pickle file
        
        Args:
            *args: File path components, passed to generate_res_path method
            
        Returns:
            Loaded Python object
            
        Raises:
            RuntimeError: When file does not exist or loading fails
        """
        try:
            file_name = ResultSaver.generate_res_path(*args)
            
            with open(file_name, 'rb') as f:
                model = pickle.load(f)
                return model
        except FileNotFoundError:
            raise RuntimeError(f"File not found: {file_name}")
        except pickle.UnpicklingError:
            raise RuntimeError(f"Invalid Pickle format: {file_name}")
        except Exception as e:
            raise RuntimeError(f"Error reading Pickle file: {str(e)}")


    @staticmethod
    def generate_res_path(*args) -> str:
        """
        Generate result file path
        
        Combine multiple path components into a complete file path and ensure the directory exists.
        
        Args:
            *args: Path components, such as directory name, subdirectory name, file name, etc.
            
        Returns:
            Generated complete file path
        """
        try:
            path_args = list(args)
            
            file_name = os.path.join(*path_args)
            
            ResultSaver.mkdir(file_name)
            
            return file_name
        except Exception as e:
            raise RuntimeError(f"Error generating result path: {str(e)}")

    @staticmethod
    def load_flaml_logs(*args) -> Tuple[Dict[str, List], str]:
        """
        Load FLAML log file
        
        Read the log file generated by the FLAML automatic machine learning library and extract the key information.
        
        Args:
            *args: File path components, passed to os.path.join method
            
        Returns:
            Tuple containing extracted log information and file path
            
        Raises:
            RuntimeError: When file does not exist or format is incorrect
        """
        try:
            path_args = list(args)
            file_name = os.path.join(*path_args)
            
            with open(file_name, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            if not lines:
                raise ValueError(f"Log file is empty: {file_name}")
                
            _dt = json.loads(lines[0])
            dt = {}
            
            keepkeys = ['record_id', 'logged_metric', 'config']
            
            for k in _dt.keys():
                if k in keepkeys:
                    if k == 'logged_metric':
                        for kk in _dt[k].keys():
                            if kk.startswith('val'):
                                dt[kk] = []
                    else:
                        dt[k] = []
            
            for line in lines[:-1]:
                logline = json.loads(line)
                
                for key in dt.keys():
                    try:
                        if key in logline:
                            dt[key].append(logline[key])
                        elif 'logged_metric' in logline and key in logline['logged_metric']:
                            dt[key].append(logline['logged_metric'][key])
                    except Exception as e:
                        print(f"Error processing log line: {str(e)}")
            
            return dt, file_name
            
        except FileNotFoundError:
            raise RuntimeError(f"Log file not found: {file_name}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Error decoding JSON: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error loading log file: {str(e)}")

    @staticmethod
    def mkdir(file_name: str) -> None:
        """
        Create directory
        
        Ensure the directory part of the specified file path exists, create it if it does not.
        
        Args:
            file_name: File path, will extract its directory part
        """
        try:
            dir_name = os.path.dirname(file_name)
            
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
        except Exception as e:
            raise RuntimeError(f"Error creating directory: {str(e)}")

    @staticmethod
    def to_local_json(file_name: str, content: str) -> None:
        """
        Write content to JSON file
        
        Args:
            file_name: Target file path
            content: JSON string content to write
        """
        try:
            dir_name = os.path.dirname(file_name)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
                
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            raise RuntimeError(f"Error writing JSON file: {str(e)}")