import pandas as pd
import lightgbm as lgb
from typing import Tuple, List, Optional, Dict
from module.utils.log import get_logger

# Configure logger
logger = get_logger(__name__)

def lgb_model_train(
    df: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    metric: str,
    task_type: str,
    seed: int = 1024,
    cat_feature: List[str] = [],
    time_index: Optional[str] = None
) -> Tuple[lgb.Booster, float]:
    """
    Train a LightGBM model and evaluate its performance
    
    Args:
        df: Tuple containing training and validation sets (X_train, X_val, y_train, y_val)
        metric: Evaluation metric name, such as 'roc_auc', 'logloss', 'accuracy', etc.
        task_type: Task type, 'classification' or 'regression'
        seed: Random seed, default is 1024
        cat_feature: List of categorical features, default is empty list
        time_index: Time index column name, if not None it will be removed from features, default is None
        
    Returns:
        Trained LightGBM model and evaluation score on validation set
    """
    # Unpack dataset
    # global class_num
    
    X_train, X_val, y_train, y_val = df
    
    # If time index exists, remove it from features
    if time_index is not None:
        X_train = X_train.drop(time_index, axis=1)
        X_val = X_val.drop(time_index, axis=1)
    
    # Remove duplicate columns
    X_train = X_train.T.drop_duplicates().T
    
    # Ensure validation set columns match training set
    X_val = X_val[X_train.columns]
    
    # Set feature data types
    for col in X_train.columns:
        if col not in cat_feature:
            # Convert numerical features to float32
            X_train[col] = X_train[col].astype('float32')
            X_val[col] = X_val[col].astype('float32')
        else:
            # Convert categorical features to category type
            X_train[col] = X_train[col].astype('category')
            X_val[col] = X_val[col].astype('category')
    
    # Define evaluation metric mapping for classification tasks
    lgb_classification_metric: Dict[str, Dict[str, str]] = {
        'binary': {
            'roc_auc': 'auc',
            'logloss': 'binary_logloss',
            'accuracy': 'accuracy'
        },
        'multiclass': {
            'roc_auc': 'auc_mu',
            'logloss': 'multi_logloss',
            'accuracy': 'accuracy'
        }
    }
    
    # Handle classification tasks
    if task_type == 'classification':
        # Determine if binary or multiclass
        class_num = y_train.nunique()
        if class_num > 2:
            task_type = 'multiclass'
            metric = lgb_classification_metric['multiclass'][metric]
        else:
            task_type = 'binary'
            metric = lgb_classification_metric['binary'][metric]
    
    # Set LightGBM parameters
    params_lgb = {
        'objective': task_type,
        'metric': metric,
        'boosting_type': 'gbdt',
        'verbose': -1,
        'seed': seed,
        'num_threads': -1,
        'num_leaves': 31,
        'learning_rate': 0.05
    }

    # For multiclass tasks, add number of classes
    if task_type == 'multiclass':
        params_lgb['num_class'] = class_num

    # Create LightGBM datasets
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    # Train model
    clf = lgb.train(
        params_lgb,
        lgb_train,
        valid_sets=lgb_val,
        valid_names='eval',
        categorical_feature=cat_feature
    )
    
    # Get score on validation set
    val_score = clf.best_score.get('eval')[metric]
    
    # For some metrics, convert score to 'smaller is better' form
    inverse_metrics = ['auc', 'f1', 'accuracy', 'auc_mu', 'r2', 'ap']
    if metric in inverse_metrics:
        val_score = 1 - val_score
    
    return clf, val_score