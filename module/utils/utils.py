#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility Functions Module

This module provides a series of utility functions for various common operations in the machine learning workflow,
including setting random seeds, evaluating model performance, processing AutoML results, and file compression.
"""

import os
import json
import random
import shutil
import zipfile
from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
from module.utils.log import get_logger
from module.utils.yaml_parser import YamlParser

# Get logger
logger = get_logger(__name__)

def set_global_seed(seed: int) -> None:
    """
    Set global random seed

    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)

def eval_score(y_val: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None, task_type: str = 'classification') -> Dict[str, float]:
    """
    Evaluate model performance

    Calculate appropriate evaluation metrics based on the task type (classification or regression).

    Args:
        y_val: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (required for classification tasks)
        task_type: Task type, 'classification' or 'regression'

    Returns:
        Dictionary containing various evaluation metrics
    """
    try:
        # Classification task evaluation
        if task_type == 'classification':
            from sklearn.metrics import (
                log_loss, accuracy_score, precision_score, 
                recall_score, f1_score, roc_auc_score
            )
            
            # Check if y_proba is provided
            if y_proba is None:
                logger.warning("Classification task evaluation requires y_proba, but it was not provided")
                return {'val_error': 'y_proba not provided'}
                
            # 计算各项指标
            val_loss = log_loss(y_val, y_proba)
            val_f1 = f1_score(y_val, y_pred, average='weighted')
            val_accuracy = accuracy_score(y_val, y_pred)
            val_precision = precision_score(y_val, y_pred, average='weighted')
            val_recall = recall_score(y_val, y_pred, average='weighted')
            
            # Calculate AUC-ROC
            if y_proba.shape[1] > 1:  # Multi-class case
                num_classes = y_proba.shape[1]
                val_auc_roc = 0.0
                for class_idx in range(num_classes):
                    val_auc_roc += roc_auc_score(
                        (y_val == class_idx).astype(int), 
                        y_proba[:, class_idx]
                    )
                val_auc_roc /= num_classes  # Take average
            else:  # Binary classification case
                val_auc_roc = roc_auc_score(y_val, y_proba)
                
            return {
                'val_Log_Loss': val_loss, 
                'val_f1': val_f1, 
                'val_Accuracy': val_accuracy, 
                'val_Precision': val_precision, 
                'val_Recall': val_recall, 
                'val_AUC': val_auc_roc
            }
        
        # Regression task evaluation
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            # Calculate metrics
            val_mse = mean_squared_error(y_val, y_pred)
            val_rmse = np.sqrt(val_mse)
            val_mae = mean_absolute_error(y_val, y_pred)
            val_r_squared = r2_score(y_val, y_pred)
            
            # Handle invalid R² values
            if np.isnan(val_r_squared):
                val_r_squared = -1
                logger.warning("R² value is NaN, set to -1")
                
            return {
                'val_MSE': val_mse, 
                'val_RMSE': val_rmse, 
                'val_MAE': val_mae, 
                'val_R-squared': val_r_squared
            }
            
    except Exception as e:
        logger.error(f"Error evaluating model performance: {str(e)}")
        return {'val_error': str(e)}

def get_automl_result(automl_result_path: str, automl_running: bool, model: Any) -> Tuple[str, List[str]]:
    """
    Get AutoML result information
    
    Extract the best model type and feature list from the result file.
    
    Args:
        automl_result_path: AutoML result file path
        automl_running: Whether AutoML was run
        model: Model object
        
    Returns:
        Tuple containing model type and feature list
    """
    try:
        # If AutoML was run, extract information from the result file
        if automl_running:
            # Read result file
            with open(automl_result_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Ensure file is not empty
            if not lines:
                logger.warning(f"AutoML result file is empty: {automl_result_path}")
                return "lgbm", getattr(model, "feature_names_in_", [])
                
            # Parse the last line to get best trial information
            best_trail = json.loads(lines[-1])
            
            # Get best model information
            model_info = json.loads(lines[best_trail['curr_best_record_id']])
            model_type = model_info['learner']
            
            # Get model features
            features = getattr(model, "feature_names_in_", [])
            
            return model_type, features
        
        # If AutoML was not run, use default values
        else:
            # Default to lgbm model type
            model_type = 'lgbm'
            
            # Try different methods to get feature names
            try:
                features = model.feature_name()
            except (AttributeError, TypeError):
                try:
                    features = model.feature_names_in_
                except (AttributeError, TypeError):
                    features = []
                    logger.warning("Unable to get model feature names")
            
            return model_type, features
            
    except Exception as e:
        logger.error(f"Error getting AutoML results: {str(e)}")
        # Return default values on error
        return "lgbm", []

def zip_download(base_dir: str, cfg_path: str) -> None:
    """
    Zip files for download
    
    Package project results, reports, and other files into compressed files for easy download and sharing.
    
    Args:
        base_dir: Project base directory
        cfg_path: Configuration file path
    """
    try:
        # Create result directory
        zip_path = os.path.join(base_dir, 'result', 'result', 'result')
        if not os.path.exists(zip_path):
            os.makedirs(zip_path)
            logger.info(f"Created result directory: {zip_path}")
        
        # Create download directory and copy configuration file
        download_dir = os.path.join(base_dir, 'download')
        config_path = os.path.join(download_dir, 'config.yaml')
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
            logger.info(f"Created download directory: {download_dir}")
        
        # Copy configuration file
        if not os.path.exists(config_path):
            shutil.copyfile(cfg_path, config_path)
            logger.info(f"Copied configuration file: {cfg_path} -> {config_path}")
        
        # Set source and destination paths
        autofe_path = os.path.join(base_dir, 'autofe')
        automl_path = os.path.join(base_dir, 'automl')
        config_dst_path = os.path.join(zip_path, 'config.yaml')
        autofe_dst_path = os.path.join(zip_path, 'autofe')
        automl_dst_path = os.path.join(zip_path, 'automl')
        
        # Clean up old files at destination paths
        if os.path.exists(config_dst_path):
            os.remove(config_dst_path)
        if os.path.exists(autofe_dst_path):
            shutil.rmtree(autofe_dst_path)
        if os.path.exists(automl_dst_path):
            shutil.rmtree(automl_dst_path)
        
        # Copy files and directories
        shutil.copyfile(config_path, config_dst_path)
        
        # Copy AutoFE and AutoML directories (if they exist)
        if os.path.exists(autofe_path):
            shutil.copytree(autofe_path, autofe_dst_path)
        else:
            logger.warning(f"AutoFE directory does not exist: {autofe_path}")
            
        if os.path.exists(automl_path):
            shutil.copytree(automl_path, automl_dst_path)
        else:
            logger.warning(f"AutoML directory does not exist: {automl_path}")
        
        # Copy files from prediction directory
        predict_path = os.path.join(os.getcwd(), 'predict')
        if os.path.exists(predict_path):
            filelist = os.listdir(predict_path)
            for file in filelist:
                src = os.path.join(predict_path, file)
                dst = os.path.join(os.path.dirname(zip_path), file)
                try:
                    # If it's a file, use copyfile
                    if os.path.isfile(src):
                        if os.path.exists(dst):
                            os.remove(dst)
                        shutil.copyfile(src, dst)
                    # If it's a directory, use copytree
                    elif os.path.isdir(src):
                        if os.path.exists(dst):
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                except Exception as e:
                    logger.error(f"Error copying prediction file {src} -> {dst}: {str(e)}")
        else:
            logger.warning(f"Prediction directory does not exist: {predict_path}")
        
        # Create result zip file
        result_zip_path = os.path.join(base_dir, 'download', 'result.zip')
        if os.path.exists(result_zip_path):
            os.remove(result_zip_path)
            
        # Compress result directory
        result_dir = os.path.join(base_dir, 'result')
        if os.path.exists(result_dir):
            with zipfile.ZipFile(result_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(result_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_name = os.path.relpath(file_path, result_dir)
                        zipf.write(file_path, arc_name)
            logger.info(f"Created result zip file: {result_zip_path}")
        else:
            logger.warning(f"Result directory does not exist: {result_dir}")
        
        # Create report zip file
        report_zip_path = os.path.join(base_dir, 'download', 'report.zip')
        if os.path.exists(report_zip_path):
            os.remove(report_zip_path)
            
        # Compress report directory
        report_dir = os.path.join(base_dir, 'report')
        if os.path.exists(report_dir):
            with zipfile.ZipFile(report_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(report_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_name = os.path.relpath(file_path, report_dir)
                        zipf.write(file_path, arc_name)
            logger.info(f"Created report zip file: {report_zip_path}")
        else:
            logger.warning(f"Report directory does not exist: {report_dir}")
            
        logger.info("File packaging completed")
        
    except Exception as e:
        logger.error(f"Error during file packaging: {str(e)}")
        raise