#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Machine learning model management module

This module provides the ESPML class, which integrates automatic feature engineering and automatic machine learning functions,
including model training, evaluation, and result saving.
"""

import os
import time
import json
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from module.utils.state import Base
from module.utils.result_saver import ResultSaver
from module.utils.utils import eval_score, get_automl_result, zip_download
from module.utils.report import (
    draw_and_save, plot_confusion_matrix, plot_roc_curve, 
    mkdirs_save_path, plot_regression_results
)
from module.utils.log import init_log, get_logger
from module.autofe.model import lgb_model_train
from module.autofe.autofe import AutoFE
from module.autofe.utils import feature2table
from module.automl.automl import AutoML
from module.incrml.metadata import MetaData

logger = get_logger(__name__)

__all__ = ['ESPML']

class ESPML(Base):
    """
    ESPML class
    
    Integrates automatic feature engineering (AutoFE) and automatic machine learning (AutoML) functions,
    providing end-to-end machine learning workflow, including data preprocessing, feature engineering, model training, and result evaluation.
    """

    def __init__(self, config_pth: str, data: Optional[pd.DataFrame] = None) -> None:
        """
        Initialize ESPML object
        
        Args:
            config_pth: Configuration file path
            data: Input DataFrame, default is None
        """
        # Call parent class initialization method
        super(ESPML, self).__init__(config_pth, data)
        
        # Determine result save path
        task_result_path = (
            ResultSaver.generate_res_path(
                os.getcwd(), 'results', 
                str(self.AuthorName), str(self.TaskName), str(self.JobID)
            ) if self.model_save_path is None else self.model_save_path
        )
        
        # Determine sequence number
        if not os.path.exists(task_result_path) or len(os.listdir(task_result_path)) <= 0:
            self.seq = 1
        else:
            try:
                # Get the maximum sequence number in the existing directory and add 1
                self.seq = max([int(num) for num in os.listdir(task_result_path) 
                               if num.isdigit()]) + 1
            except (ValueError, FileNotFoundError):
                self.seq = 1
        
        # Create directory and log path
        self.base_dir = ResultSaver.generate_res_path(task_result_path, str(self.seq))
        logpath = ResultSaver.generate_res_path(self.base_dir, 'autofe', 'dfs_log')
        debugpath = ResultSaver.generate_res_path(self.base_dir, 'autofe', 'debug.log')
        self.automl_log_path = ResultSaver.generate_res_path(self.base_dir, 'automl', 'logs')
        
        # Initialize logger
        init_log(logger, logpath, debugpath)
        logger.info(f'Start training the {self.seq}th experiment')
        logger.info('Start ESPML training...')
        
        # Calculate the baseline score of the original data
        _, self.ori_score = lgb_model_train(
            (self.X_train, self.X_val, self.y_train, self.y_val), 
            metric=self.metric, 
            task_type=self.task_type, 
            seed=self.seed, 
            cat_feature=self.cat_features, 
            time_index=self.time_index
        )
        
        # Adjust score display based on metric type
        logger_score = 1 - self.ori_score if self.metric in ['roc_auc', 'f1', 'accuracy', 'auc_mu', 'r2', 'ap'] else self.ori_score
        logger.info(f'Original data training validation set {self.metric} score: {logger_score:.4f}')
        
        # Initialize AutoFE and AutoML components
        if self.autofe_running:
            self.autofe = AutoFE(
                n=self.max_trails, 
                base_score=self.ori_score, 
                logger=logger, 
                transformer=self.transform, 
                **self.config
            )
        else:
            self.autofe = None
            
        if self.automl_running:
            self.automl = AutoML(
                estimator='auto', 
                budget=self.automl_time_budget, 
                metric=self.metric, 
                task_type=self.task_type, 
                n_jobs=-1, 
                seed=self.seed, 
                log_file_name=self.automl_log_path
            )
        else:
            self.automl = None

    def fit(self) -> None:
        """
        Train model
        
        Execute feature engineering and model training process, including:
        1. If AutoFE is enabled, perform feature engineering
        2. If AutoML is enabled, perform automatic model selection and training
        3. If AutoML is not enabled, use LightGBM model for training
        4. Save model and evaluation results
        """
        # Execute feature engineering (if enabled)
        if self.autofe_running:
            logger.info("Start executing automatic feature engineering...")
            X_train, X_val, y_train, y_val, self.select_feature = self.autofe.fit(
                self.X_train, self.y_train, self.X_val, self.y_val
            )
        else:
            logger.info("Auto feature engineering is not enabled, using original features")
            X_train, X_val, y_train, y_val, self.select_feature = (
                self.X_train, self.X_val, self.y_train, self.y_val, []
            )
            
        logger.info(f'Feature engineering completed, selected {len(self.select_feature)} features')
        
        # Execute model training
        if self.automl_running:
            logger.info("Start executing automatic machine learning...")
            self.model = self.automl.fit(X_train, y_train, X_val, y_val)
            # Calculate and adjust metric score
            logger_score = 1 - self.model.best_loss if self.metric in ['roc_auc', 'f1', 'accuracy', 'auc_mu', 'r2', 'ap'] else self.model.best_loss
        else:
            logger.info("Auto machine learning is not enabled, using LightGBM model training")
            # Use LightGBM to train model
            self.model, logger_score = lgb_model_train(
                (X_train, X_val, y_train, y_val), 
                metric=self.metric, 
                task_type=self.task_type, 
                seed=self.seed, 
                cat_feature=self.cat_features, 
                time_index=self.time_index
            )
            # Adjust metric score
            if self.metric in ['auc', 'f1', 'accuracy', 'auc_mu', 'r2', 'ap']:
                logger_score = 1 - logger_score
                
        # Predict on validation set
        pred = self.model.predict(X_val)
        yproba = self.model.predict_proba(X_val) if self.task_type == 'classification' else None
        
        # Record model performance
        logger.info(f'Machine learning completed, best model {self.metric} score: {logger_score:.4f}')
        
        # Save model and results
        self.save(X_train, X_val, y_train, y_val, pred, yproba)
        logger.info('Training completed.')
        
        return None

    def save(self, X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series, 
             y_val: pd.Series, ypred: np.ndarray, yproba: Optional[np.ndarray] = None) -> None:
        """
        Save model and evaluation results
        
        Save model training and validation results to the specified directory, including:
        1. Evaluation metrics and results saved in JSON format
        2. Validation set prediction results saved in CSV format
        3. If visualization is enabled, generate various visualization charts
        4. If AutoML is enabled, save AutoML training process records
        
        Args:
            X_train: Training set feature data
            X_val: Validation set feature data
            y_train: Training set target variable
            y_val: Validation set target variable
            ypred: Validation set prediction results
            yproba: Validation set prediction probabilities (classification tasks)
        """
        logger.info("Start saving model and evaluation results...")
        
        # Prepare result data
        res = {
            'id': 1, 
            'combination_features': self.select_feature
        }
        
        # Save feature engineering results
        ResultSaver.save_json(res, self.base_dir, 'autofe', 'result')
        save_feature_names_table = feature2table(self.select_feature)
        ResultSaver.save_csv(save_feature_names_table, False, self.base_dir, 'autofe', 'features.csv')
        ResultSaver.save_csv(pd.DataFrame({
            'feature_name': [], 
            'trail_result': [], 
            'feature_imp_ratio': [], 
            'feature_length': []
        }), False, self.base_dir, 'autofe', 'record.csv')
        
        # Save converter and model
        ResultSaver.save_pickle(self.transform, self.base_dir, 'autofe', 'labelencoder.pkl')
        ResultSaver.save_pickle(self.model, self.base_dir, 'automl', 'result')
        
        # Calculate evaluation metrics
        json_dt = eval_score(y_val, ypred, yproba, task_type=self.task_type)
        json_dt.update(res)
        ResultSaver.save_json(json_dt, self.base_dir, 'automl', 'val_logs')
        
        # Prepare validation set prediction results
        logger.info("Prepare validation set prediction results...")
        test_res = pd.concat([
            pd.Series(y_val).reset_index(drop=True), 
            pd.Series(ypred).reset_index(drop=True)
        ], axis=1)
        test_res.columns = [self.target_name, 'ypred']
        
        # If the target variable was transformed, restore it
        if (hasattr(self, 'transform') and 
            hasattr(self.transform, '_record') and 
            self.transform._record is not None and 
            self.target_name in self.transform._record):
            logger.info("Restore target variable...")
            for col in test_res.columns.tolist():
                test_res[col] = self.transform._record[self.target_name].inverse_transform(test_res[col])
        
        # If features were transformed, restore them
        X_val_copy = X_val.copy()
        if hasattr(self, 'transform') and hasattr(self.transform, '_record'):
            logger.info("Reverse transform features...")
            for col in self.all_features:
                if col in self.transform._record:
                    X_val_copy[col] = self.transform._record[col].inverse_transform(X_val_copy[col])
        
        # Merge prediction results and features
        test_res = pd.concat([test_res, X_val_copy.reset_index(drop=True)], axis=1)
        
        # Save validation set prediction results as CSV
        logger.info("Save validation set prediction results...")
        ResultSaver.save_csv(test_res, False, self.base_dir, 'automl', 'val_res')
        
        # If visualization is enabled, generate charts
        if self.plot:
            logger.info("Generate visualization charts...")
            result_path = ResultSaver.generate_res_path(self.base_dir, 'report', 'espmlImage')
            
            # Get model type and features
            model_type, features = get_automl_result(
                automl_result_path=self.automl_log_path, 
                automl_running=self.automl_running, 
                model=self.model
            )
            
            # Classification task visualization
            if self.task_type == 'classification':
                # If the target variable is object type, encoding is needed
                y_val_encoded, ypred_encoded = y_val, ypred
                if y_val.dtype == np.object_:
                    logger.info("Encoding classification target...")
                    encoder = LabelEncoder().fit(np.concatenate((y_val, ypred)))
                    y_val_encoded = encoder.transform(y_val)
                    ypred_encoded = encoder.transform(ypred)
                
                # Get class list
                classes = np.unique(y_val_encoded)
                
                # Draw confusion matrix
                logger.info("Draw confusion matrix...")
                save_confusion_matrix_path = mkdirs_save_path(
                    os.path.join(result_path, 'valid_metirc', 'valid_metirc2.png')
                )
                plot_confusion_matrix(y_val_encoded, ypred_encoded, classes, path=save_confusion_matrix_path)
                
                # Draw ROC curve
                logger.info("Draw ROC curve...")
                save_roc_curve_path = mkdirs_save_path(
                    os.path.join(result_path, 'valid_metirc', 'valid_metirc1.png')
                )
                plot_roc_curve(y_val_encoded, yproba, len(classes), path=save_roc_curve_path)
            
            # Regression task visualization
            else:
                logger.info("Draw regression results graph...")
                save_regression_results_path = mkdirs_save_path(
                    os.path.join(result_path, 'valid_metirc', 'valid_metirc1.png')
                )
                plot_regression_results(y_val, ypred, model_type, save_path=save_regression_results_path)
            
            # Draw AutoML model visualization graph
            if self.automl_running:
                logger.info("Draw AutoML model visualization graph...")
                draw_and_save(
                    self.model, model_type, self.task_type, 
                    self.model._X_train_all, self.model._y_train_all, 
                    features, self.target_name, 
                    save_path=result_path, val_result=json_dt
                )
                
                # Process AutoML training process records
                logger.info("Process AutoML training process records...")
                automl_trails_path = ResultSaver.generate_res_path(self.base_dir, 'automl', 'logs')
                
                try:
                    with open(automl_trails_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Parse training process records
                    dt = {}
                    for i in range(len(lines) - 1):
                        try:
                            info = json.loads(lines[i])
                            info2 = info.get('config', {})
                            
                            # Record ID
                            if 'record_id' in dt:
                                dt['record_id'].append(i)
                            else:
                                dt['record_id'] = [i]
                            
                            # Record learner type
                            if 'learner' in dt:
                                dt['learner'].append(info.get('learner', ''))
                            else:
                                dt['learner'] = [info.get('learner', '')]
                            
                            # Remove unnecessary fields
                            if 'FLAML_sample_size' in info2:
                                info2.pop('FLAML_sample_size')
                            
                            # Record evaluation metrics
                            for k, v in info.get('logged_metric', {}).items():
                                if k.startswith('train'):
                                    continue
                                if k in dt:
                                    dt[k].append(v)
                                else:
                                    dt[k] = [v]
                            
                            # Record configuration
                            if 'config' in dt:
                                dt['config'].append(str(info2))
                            else:
                                dt['config'] = [str(info2)]
                        except Exception as e:
                            logger.error(f"Error parsing training process records: {e}")
                    
                    # Save training process records
                    logger.info("Save AutoML training process records...")
                    ResultSaver.save_csv(pd.DataFrame(dt), False, self.base_dir, 'automl', 'automl_trails')
                except Exception as e:
                    logger.error(f"Error processing AutoML training process records: {e}")
            else:
                # If not AutoML, draw LightGBM model visualization graph
                logger.info("Draw LightGBM model visualization graph...")
                draw_and_save(
                    self.model, 'lgbm', self.task_type, 
                    X_train, y_train, features, self.target_name, 
                    save_path=result_path, val_result=json_dt
                )
            
            # Create a downloadable ZIP file
            logger.info("Create a downloadable ZIP file...")
            zip_download(base_dir=self.base_dir, cfg_path=self.config_path)
        
        logger.info("Model and evaluation results saved successfully")
        return None