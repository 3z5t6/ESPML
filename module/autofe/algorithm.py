import os
import threading
from typing import List, Dict, Tuple, Optional, Union

import pandas as pd
from logging import Logger

from module.autofe.transform import Transform
from module.autofe.model import lgb_model_train
from module.autofe.utils import normalize_gini_select
from module.utils.log import get_logger

# Configure logger
logger = get_logger(__name__)

# Feature selection using multiple threads
def threads_feature_select(
    df: pd.DataFrame,
    target_name: str,
    feature_candidates: List[str],
    transformer: Optional[Transform] = None,
    return_score: bool = False,
    logger: Optional[Logger] = None
) -> Union[Tuple[List[str], pd.DataFrame], Tuple[List[str], pd.DataFrame, Dict[str, float]]]:
    """
    Feature selection using multiple threads
    
    Args:
        df: Input DataFrame
        target_name: Target variable name
        feature_candidates: List of candidate features
        transformer: Feature transformer, if None a new Transform instance will be created
        return_score: Whether to return feature scores, default is False
        logger: Logger instance, default is None
        
    Returns:
        If return_score is False, returns (list of selected features, feature DataFrame)
        If return_score is True, returns (list of selected features, feature DataFrame, feature score dictionary)
    """
    y = df[target_name]
    max_threads = os.cpu_count() or 1  # Ensure at least one thread
    new_features_scores = {}
    new_features = [0] * max_threads
    feature_selected = []
    threads = []
    
    if transformer is None:
        transformer = Transform()

    def threads_function(process: int) -> None:
        """
        Thread function for feature fusion
        
        Args:
            process: Thread index
        """
        start_idx = process * feature_num
        end_idx = start_idx + feature_num
        current_candidates = feature_candidates[start_idx:end_idx]
        
        if current_candidates:
            fes = transformer.fit_transform(df, current_candidates)
            selected_features, gini_dict = normalize_gini_select(fes, y, list(df.columns))
            new_features[process] = fes[selected_features]
            new_features_scores.update(gini_dict)
            feature_selected.extend(selected_features)
    
    # Calculate the number of features to be processed by each thread
    feature_num = len(feature_candidates) // max_threads
    if len(feature_candidates) % max_threads > 0:
        feature_num += 1
        
    # Create and start threads
    for i in range(max_threads):
        t = threading.Thread(target=threads_function, args=(i,))
        threads.append(t)
        t.start()
        
    # Wait for all threads to complete
    for t in threads:
        t.join()
        
    # Merge feature data
    feature_choose = pd.concat(
        [newfe for newfe in new_features if not isinstance(newfe, int)],
        axis=1
    )
    
    if return_score:
        return feature_selected, feature_choose, new_features_scores
    return feature_selected, feature_choose


def model_features_select(
    fes: pd.DataFrame,
    baseline: float,
    metric: str,
    task_type: str,
    seed: int = 1024,
    cat_features: List[str] = [],
    logger: Optional[Logger] = None,
    time_index: Optional[str] = None
) -> Tuple[List[str], float]:
    """
    Select features through model evaluation, new features are only selected if they perform better than the baseline

    Args:
        fes: DataFrame containing features
        baseline: Baseline score for comparison
        metric: Name of evaluation metric
        task_type: Type of task
        seed: Random seed, default is 1024
        cat_features: List of categorical features, default is empty list
        logger: Logger instance, default is None
        time_index: Time index column name, default is None
        
    Returns:
        (List of selected features, score)
    """
    model, score = lgb_model_train(
        df=fes,
        metric=metric,
        task_type=task_type,
        seed=seed,
        cat_feature=cat_features,
        time_index=time_index
    )
    
    # Sort by feature importance
    feimp = sorted(
        zip(model.feature_name(), model.feature_importance()),
        key=lambda x: x[1],
        reverse=True
    )
    
    feature_selected = []
    
    # For some metrics, convert score to 'smaller is better' form
    inverse_metrics = ['roc_auc', 'f1', 'accuracy', 'auc_mu', 'r2', 'ap']
    logger_score = 1 - score if metric in inverse_metrics else score
    
    if logger:
        logger.info(f'Autofe Trial finished with value: {logger_score}')
    
    # Select features with importance greater than 0
    for k, v in feimp:
        if v > 0:
            feature_selected.append(k)
        else:
            break
    
    # Only return features if score is better than baseline
    if score < baseline:
        return feature_selected, score
    return [], score


def max_threads_name2feature(
    df: pd.DataFrame,
    feature_candidates: List[str],
    transformer: Optional[Transform] = None,
    logger: Optional[Logger] = None
) -> pd.DataFrame:
    """
    Generate features using multiple threads
    
    Args:
        df: Input DataFrame
        feature_candidates: List of candidate features
        transformer: Feature transformer, if None a new Transform instance will be created
        logger: Logger instance, default is None
        
    Returns:
        DataFrame containing generated features
    """
    if not feature_candidates:
        return df
        
    max_threads = os.cpu_count() or 1  # Ensure at least one thread
    new_features = [0] * max_threads
    threads = []
    
    if transformer is None:
        transformer = Transform()

    def threads_function(process: int) -> None:
        """
        Thread function for feature fusion
        
        Args:
            process: Thread index
        """
        start_idx = process * feature_num
        end_idx = start_idx + feature_num
        current_candidates = feature_candidates[start_idx:end_idx]
        
        if current_candidates:
            fes = transformer.fit_transform(df, current_candidates)
            # For the first thread, keep all columns; for other threads, only keep candidate feature columns
            new_features[process] = fes if process == 0 else fes[current_candidates]
    
    # Calculate the number of features to be processed by each thread
    feature_num = len(feature_candidates) // max_threads
    if len(feature_candidates) % max_threads > 0:
        feature_num += 1
        
    # Create and start threads
    for i in range(max_threads):
        t = threading.Thread(target=threads_function, args=(i,))
        threads.append(t)
        t.start()
        
    # Wait for all threads to complete
    for t in threads:
        t.join()
        
    # Filter and merge feature data
    valid_features = [i for i in new_features if isinstance(i, pd.DataFrame)]
    return pd.concat(valid_features, axis=1)