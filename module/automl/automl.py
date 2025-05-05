"""
Automatic Machine Learning Module

This module provides automatic machine learning related functionality, including custom models, evaluation metrics, and automated training processes.
"""

import os
import time
import hashlib
import numpy as np
import pandas as pd
import flaml
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

from flaml.automl.model import SKLearnEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from module.utils.log import get_logger

logger = get_logger(__name__)


class LRRegression(SKLearnEstimator):
    """
    Linear Regression model estimator for regression tasks
    
    Inherits from FLAML's SKLearnEstimator, encapsulates sklearn's LinearRegression
    """
    
    def __init__(self, task, **config):
        """
        Initialize the Linear Regression model estimator
        
        Args:
            task: Task type object
            **config: Model configuration parameters
        """
        super().__init__(task, **config)
        assert not self._task.is_classification(), 'LinearRegression for regression task only'
        self.estimator_class = LinearRegression


class LRClassifier(SKLearnEstimator):
    """
    Logistic Regression model estimator for classification tasks
    
    Inherits from FLAML's SKLearnEstimator, encapsulates sklearn's LogisticRegression
    """
    
    def __init__(self, task, **config):
        """
        Initialize the Logistic Regression model estimator
        
        Args:
            task: Task type object
            **config: Model configuration parameters
        """
        super().__init__(task, **config)
        assert self._task.is_classification(), 'LogisticRegression for classification task only'
        self.estimator_class = LogisticRegression


def custom_metric(
    X_val: pd.DataFrame, 
    y_val: pd.Series, 
    estimator: Any, 
    labels: Any, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    weight_val: Optional[np.ndarray] = None, 
    weight_train: Optional[np.ndarray] = None, 
    *args
) -> Tuple[float, Dict[str, float]]:
    """
    Custom evaluation metric function
    
    Calculate custom loss on training and validation sets, as well as prediction time
    
    Args:
        X_val: Validation features
        y_val: Validation labels
        estimator: Model estimator
        labels: Label list
        X_train: Training features
        y_train: Training labels
        weight_val: Validation sample weights, default is None
        weight_train: Training sample weights, default is None
        *args: Other parameters
        
    Returns:
        Tuple containing validation loss and additional information
    """
    # Calculate prediction time
    start = time.time()
    y_pred = estimator.predict(X_train)
    pred_time = (time.time() - start) / len(X_train)
    
    # Define constants
    Ci = 147
    en = 0.25
    
    # Initialize error lists
    error_list = []
    edl_list = []
    
    def get_error(realp: np.ndarray, predp: np.ndarray) -> float:
        """
        Calculate prediction error
        
        Args:
            realp: Array of real values
            predp: Array of predicted values
            
        Returns:
            Mean error value
        """
        # Clear error lists for recalculation
        error_list.clear()
        edl_list.clear()
        
        for i in range(len(realp)):
            if realp[i] == 0:
                # Handle case when real value is 0
                v = np.abs(realp[i] - predp[i]) / Ci
                error_list.append(v)
            else:
                # Calculate relative error rate
                err_rate = np.abs((predp[i] - realp[i]) / realp[i])
                error_list.append(err_rate)
                
                # Handle cases where error rate exceeds threshold
                if err_rate > en:
                    if realp[i] > predp[i]:
                        err_dl = np.abs(realp[i] - (1 + en) * predp[i])
                    else:
                        err_dl = np.abs(realp[i] - (1 - en) * predp[i])
                    edl_list.append(err_dl)
                else:
                    edl_list.append(0)
                    
        return np.mean(error_list)
    
    # Handle training data types
    realp = y_train.values if isinstance(y_train, (pd.DataFrame, pd.Series)) else y_train
    predp = y_pred.values if isinstance(y_pred, (pd.DataFrame, pd.Series)) else y_pred
    
    # Calculate training loss
    train_loss = get_error(realp, predp)
    
    # Make predictions on validation set
    y_pred_val = estimator.predict(X_val)
    
    # Handle validation data types
    realp = y_val.values if isinstance(y_val, (pd.DataFrame, pd.Series)) else y_val
    predp = y_pred_val.values if isinstance(y_pred_val, (pd.DataFrame, pd.Series)) else y_pred_val
    
    # Calculate validation loss
    val_loss = get_error(realp, predp)
    
    return (val_loss, {'val_loss': val_loss, 'train_loss': train_loss, 'pred_time': pred_time})

def automl_fit(
    X_train: pd.DataFrame, 
    X_val: pd.DataFrame, 
    y_train: pd.Series, 
    y_val: pd.Series, 
    estimator: List[str], 
    metric: str, 
    budget: int, 
    task_type: str, 
    log_file: str = 'logs', 
    seed: int = 1024, 
    n_jobs: int = -1
) -> flaml.AutoML:
    """
    AutoML model training function
    
    Uses the FLAML library for automatic model selection and hyperparameter optimization
    
    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels
        estimator: List of model estimators
        metric: Evaluation metric
        budget: Training time budget (seconds)
        task_type: Task type, 'classification' or 'regression'
        log_file: Log file name, default is 'logs'
        seed: Random seed, default is 1024
        n_jobs: Number of parallel jobs, default is -1 (use all CPU cores)
        
    Returns:
        Trained AutoML model
    """
    # Initialize AutoML model
    model = flaml.AutoML(metric=metric, log_file_name=log_file, time_budget=budget)
    
    # Add custom learners
    if 'lrr' in estimator:
        model.add_learner('lrr', LRRegression)
    if 'lrc' in estimator:
        model.add_learner('lrc', LRClassifier)
    
    # For multi-class problems, adjust the evaluation metric
    if pd.concat([y_val, y_train], axis=0).nunique() > 2 and metric == 'roc_auc':
        metric = 'roc_auc_ovr'
    
    # Train the model
    model.fit(
        X_train=X_train, 
        y_train=y_train, 
        X_val=X_val, 
        y_val=y_val, 
        task=task_type, 
        time_budget=budget, 
        metric=custom_metric, 
        keep_search_state=True, 
        verbose=False, 
        log_training_metric=True, 
        log_type='all', 
        estimator_list=estimator, 
        seed=seed, 
        n_jobs=n_jobs
    )
    
    # Merge training and validation sets for final model training
    all_data = pd.concat([
        pd.concat([X_train, X_val], axis=0), 
        pd.concat([y_train, y_val], axis=0)
    ], axis=1)
    
    # Retrain using the best model from the log
    model.retrain_from_log(
        log_file_name=log_file, 
        dataframe=all_data, 
        label=y_val.name, 
        task=task_type, 
        train_full=True, 
        train_best=True
    )
    
    return model


class AutoML:
    """
    AutoML class
    
    Encapsulates the training and prediction workflow for automatic machine learning
    """

    def __init__(
        self, 
        estimator: List[str], 
        budget: int, 
        metric: str, 
        task_type: str, 
        seed: int = 1024, 
        n_jobs: int = -1, 
        log_file_name: str = 'logs'
    ):
        """
        Initialize AutoML class
        
        Args:
            estimator: List of model estimators
            budget: Training time budget (seconds)
            metric: Evaluation metric
            task_type: Task type, 'classification' or 'regression'
            seed: Random seed, default is 1024
            n_jobs: Number of parallel jobs, default is -1 (use all CPU cores)
            log_file_name: Log file name, default is 'logs'
        """
        self.estimator = estimator
        self.budget = budget
        self.metric = metric
        self.task_type = task_type
        self.n_jobs = n_jobs
        self.log_file_name = log_file_name
        self.seed = seed
        self.model = None

    def fit(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_val: pd.DataFrame, 
        y_val: pd.Series
    ) -> flaml.AutoML:
        """
        Train AutoML model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Trained AutoML model
        """
        self.model = automl_fit(
            X_train=X_train, 
            X_val=X_val, 
            y_train=y_train, 
            y_val=y_val, 
            estimator=self.estimator, 
            metric=self.metric, 
            budget=self.budget, 
            task_type=self.task_type, 
            log_file=self.log_file_name, 
            n_jobs=self.n_jobs, 
            seed=self.seed
        )
        return self.model
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Test features
            
        Returns:
            Prediction results
        """
        if self.model is None:
            raise ValueError("Model not trained yet, please call the fit method first")
        return self.model.predict(X)