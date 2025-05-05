from typing import List, Tuple, Optional, Dict

import pandas as pd
from logging import Logger

from module.autofe.transform import Transform
from module.autofe.utils import feature_space, update_time_span
from module.autofe.algorithm import model_features_select, max_threads_name2feature, threads_feature_select

class AutoFE:
    """
    Automated Feature Engineering class for generating and selecting features
    
    This class implements an iterative process of feature generation and selection
    to optimize feature sets.
    """
    
    def __init__(
        self,
        n: int = 2,
        method: Optional[str] = None,
        base_score: float = 0.5,
        logger: Optional[Logger] = None,
        transformer: Optional[Transform] = None,
        **kwargs
    ) -> None:
        """
        Initialize AutoFE class
        
        Args:
            n: Number of iterations, default is 2
            method: Feature selection method, default is None
            base_score: Baseline score for feature selection comparison, default is 0.5
            logger: Logger instance, default is None
            transformer: Feature transformer, default is None (creates new Transform instance)
            **kwargs: Additional configuration parameters, mainly containing Feature dictionary
        """
        self.n = n
        self.logger = logger
        self.method = method
        self.base_score = base_score
        self.transformer = Transform() if transformer is None else transformer
        
        # Get feature configuration from kwargs
        feature_config = kwargs.get('Feature', {})
        self.task_type = feature_config.get('TaskType', None)
        self.metric = feature_config.get('Metric', None)
        self.seed = feature_config.get('RandomSeed', 1024)
        self.cat_features = feature_config.get('CategoricalFeature', [])
        self.target_name = feature_config.get('TargetName', None)
        self.time_index = feature_config.get('TimeIndex', None)
        self.group_index = feature_config.get('GroupIndex', None)
        self.time_span = update_time_span(feature_config.get('TimeWindow', None))

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
        """
        Train the automated feature engineering model and generate new features
        
        Args:
            X_train: Training feature set
            y_train: Training labels
            X_val: Validation feature set
            y_val: Validation labels
            
        Returns:
            Tuple containing: training set with new features, validation set with new features,
            training labels, validation labels, and list of selected features
        """
        # Merge features and labels
        df_train = pd.concat([X_train, y_train], axis=1)
        df_val = pd.concat([X_val, y_val], axis=1)
        
        # Initialize variables
        already_selected = []  # Already selected features
        best_score = self.base_score  # Best score
        pick_features = []  # Best feature set
        adv_features = []  # Advanced feature set
        
        # Iterative feature selection process
        for i in range(self.n):
            # Generate candidate features
            candidate_feature = feature_space(
                df_train,
                y_train.name,
                already_selected=already_selected,
                time_span=self.time_span,
                time_index=self.time_index,
                group_index=self.group_index
            )
            
            # If no new candidate features, end iteration
            if not candidate_feature:
                break
                
            # Update already selected features list
            already_selected.extend(candidate_feature)
            
            # Use multi-threading to select features
            selected_features, new_df, features_scores = threads_feature_select(
                df_train,
                y_train.name,
                candidate_feature,
                return_score=True,
                transformer=self.transformer
            )
            
            # Update validation and training sets
            df_val = max_threads_name2feature(df_val, selected_features, transformer=self.transformer)
            df_train = pd.concat([df_train, new_df[selected_features]], axis=1)
            
            # Evaluate features using a model
            train_features = df_train.drop(y_train.name, axis=1)
            val_features = df_val.drop(y_train.name, axis=1)
            
            selected_features, fscore = model_features_select(
                # Combine training and validation features
                fes=pd.concat([train_features, val_features]),
                baseline=self.base_score,
                metric=self.metric,
                task_type=self.task_type,
                logger=self.logger,
                seed=self.seed,
                cat_features=self.cat_features,
                time_index=self.time_index
            )
            
            # Filter out original features
            selected_features = [feature for feature in selected_features if feature not in X_train.columns]
            
            if self.logger:
                self.logger.info(
                    f'Iteration {i + 1} searched {len(candidate_feature)} features, found {len(selected_features)} effective features'
                )
            
            # Update best feature set
            if fscore < best_score:
                best_score = fscore
                pick_features = selected_features[:len(X_train.columns)]
                
            # Update advanced feature set
            adv_features.extend([feature for feature in selected_features[:20] if feature not in adv_features])
            
            # Limit feature count
            if len(selected_features) > 20:
                selected_features = selected_features[:20]
                
            # Update datasets, keeping only original features, target variable and selected features
            original_cols = list(X_train.columns)
            df_train = df_train[original_cols + [y_train.name] + selected_features]
            df_val = df_val[original_cols + [y_train.name] + selected_features]
            
            if self.logger:
                self.logger.debug(f'Iteration {i + 1} selected features: {selected_features}')
                
            # If no features selected but iterations remain, use feature score ranking to select features
            if not selected_features and i + 1 < self.n:
                # Select top 20 features by feature score
                top_features = list(dict(sorted(features_scores.items(), key=lambda x: x[1], reverse=True)).keys())[1:21]
                selected_features = [feature for feature in top_features if feature not in original_cols + [y_train.name]]
                
                # Update datasets
                df_train = max_threads_name2feature(df_train, selected_features)
                df_val = max_threads_name2feature(df_val, selected_features)
        
        # Final feature selection
        if adv_features:
            # Regenerate features using advanced feature set
            train_subset = df_train[list(X_train.columns) + [y_train.name]]
            val_subset = df_val[list(X_train.columns) + [y_train.name]]
            
            df_train = max_threads_name2feature(train_subset, adv_features)
            df_val = max_threads_name2feature(val_subset, adv_features)
            
            # Evaluate features again
            train_features = df_train.drop(y_train.name, axis=1)
            val_features = df_val.drop(y_train.name, axis=1)
            
            selected_features, fscore = model_features_select(
                fes=pd.concat([train_features, val_features]),
                baseline=self.base_score,
                metric=self.metric,
                task_type=self.task_type,
                logger=self.logger,
                seed=self.seed,
                cat_features=self.cat_features,
                time_index=self.time_index
            )
            
            # Filter out original features
            selected_features = [feature for feature in selected_features if feature not in X_train.columns]
            
            # If no features selected, use previously saved best feature set
            if not selected_features:
                selected_features = [feature for feature in pick_features if feature not in X_train.columns]
                df_train = max_threads_name2feature(train_subset, selected_features)
                df_val = max_threads_name2feature(val_subset, selected_features)
        
        if self.logger:
            self.logger.debug(f'Final selected features: {selected_features}')

        return (
            df_train[list(X_train.columns) + selected_features],
            df_val[list(X_train.columns) + selected_features],
            y_train,
            y_val,
            selected_features
        )