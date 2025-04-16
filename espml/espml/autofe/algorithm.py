# -*- coding: utf-8 -*-
"""
AutoFE 特征选择和生成算法函数 (espml)
包含并行特征计算、基于 Gini 的快速筛选和基于模型的特征选择
"""

import os
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from loguru import logger


from espml.autofe.transform import Transform
from espml.autofe import utils as autofe_utils

try:
    import lightgbm as lgb
    LGBM_INSTALLED = True
except ImportError:
    logger.warning("LightGBM 未安装,model_features_select 将不可用请运行 'pip install lightgbm'")
    lgb = None
    LGBM_INSTALLED = False

from sklearn.metrics import mean_squared_error, log_loss, accuracy_score, f1_score

# --- 并行特征计算  ---

def max_threads_name2feature(
    df: pd.DataFrame,
    feature_names: List[str],
    transformer: Transform,
    logger: Any,
    n_jobs: int = -1
    ) -> pd.DataFrame:
    """
    使用多线程/多进程并行地根据特征名称列表计算特征值,并将结果添加到 DataFrame
    依赖 Transform.transform 方法计算特征

    Args:
        df (pd.DataFrame): 输入的 DataFrame (包含计算所需的基础特征)
        feature_names (List[str]): 需要计算的 AutoFE 特征名称列表
        transformer (Transform): 已初始化的 Transform 实例,用于执行计算
        logger (logger): loguru logger 实例
        n_jobs (int): 并行工作的数量-1 表示使用所有 CPU 核心,1 表示串行

    Returns:
        pd.DataFrame: 添加了新计算特征列的 DataFrame 副本计算失败的列不会被添加

    Raises:
        TypeError: 如果 transformer 不是 Transform 实例
        ValueError: 如果 feature_names 包含非字符串
    """
    if not isinstance(transformer, Transform):
        raise TypeError("参数 'transformer' 必须是 Transform 类的实例")
    if not isinstance(feature_names, list):
        raise TypeError("'feature_names' 必须是一个列表")
    if not all(isinstance(name, str) for name in feature_names):
         raise ValueError("'feature_names' 列表必须只包含字符串")

    if not feature_names:
        logger.debug("max_threads_name2feature: 输入的特征名称列表为空,无需计算")
        return df.copy()

    logger.info(f"开始并行计算 {len(feature_names)} 个特征 (n_jobs={n_jobs})...")
    start_time = time.perf_counter()

    if n_jobs <= 0: actual_workers = os.cpu_count() or 1
    else: actual_workers = min(n_jobs, os.cpu_count() or 1)
    executor_cls = ThreadPoolExecutor

    logger.debug(f"实际使用 worker 数量: {actual_workers} (类型: {executor_cls.__name__})")

    results_dict: Dict[str, pd.Series] = {}
    futures_map: Dict[Any, str] = {}
    failed_features: List[str] = []

    # --- 内部任务函数 ---
    def _calculate_single_feature_task(feature_name: str) -> Optional[Tuple[str, pd.Series]]:
        try:
            df_with_feature = transformer.transform(df, [feature_name])

            if feature_name in df_with_feature.columns:
                 new_series = df_with_feature[feature_name]
                 if isinstance(new_series, pd.Series):
                     if not new_series.index.equals(df.index):
                          logger.warning(f"特征 '{feature_name}' 计算后索引不匹配,尝试重新对齐")
                          new_series = new_series.reindex(df.index)
                     return feature_name, new_series
                 else: logger.error(f"... 返回类型不是 Series: {type(new_series)}")
            else: logger.error(f"... 未在结果 DataFrame 中找到列 '{feature_name}'")
            return None
        except Exception as e:
            logger.error(f"并行计算特征 '{feature_name}' 时出错: {type(e).__name__}: {e}", exc_info=True)
            return None

    # --- 提交与收集 ---
    processed_count = 0
    names_to_submit = sorted(list(set(name for name in feature_names if name not in df.columns))) # 去重并排序
    logger.debug(f"需要计算 {len(names_to_submit)} 个新特征")

    if actual_workers == 1:
        logger.debug("n_jobs=1,执行串行计算...")
        for name in names_to_submit:
            result = _calculate_single_feature_task(name)
            if result is not None:
                results_dict[result[0]] = result[1]
                processed_count += 1
            else:
                failed_features.append(name)
    else:
        with executor_cls(max_workers=actual_workers) as executor:
            futures = {executor.submit(_calculate_single_feature_task, name): name for name in names_to_submit}
            for future in as_completed(futures):
                feature_name = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        results_dict[result[0]] = result[1]
                        processed_count += 1
                    else:
                        failed_features.append(feature_name)
                except Exception as exc:
                    logger.error(f"获取特征 '{feature_name}' 计算结果时出错: {exc}")
                    failed_features.append(feature_name)

    # --- 合并结果 ---
    logger.debug(f"并行计算完成,成功计算 {processed_count}/{len(names_to_submit)} 个新特征失败: {len(failed_features)} 个")
    if failed_features: logger.warning(f"计算失败的特征列表: {failed_features}")

    res_df = df.copy()
    if results_dict:
        new_features_df = pd.DataFrame(results_dict)
        cols_to_add_ordered = [name for name in feature_names if name in new_features_df.columns and name not in res_df.columns]

        if cols_to_add_ordered:
            if not new_features_df.index.equals(res_df.index):
                logger.warning("并行计算返回的特征索引与 DataFrame 不一致,强制重新对齐")
                new_features_df = new_features_df.reindex(res_df.index)
            res_df = pd.concat([res_df, new_features_df[cols_to_add_ordered]], axis=1)
            logger.debug(f"已添加 {len(cols_to_add_ordered)} 列新特征到 DataFrame")

    end_time = time.perf_counter()
    logger.info(f"并行特征计算 ('max_threads_name2feature') 完成,耗时: {end_time - start_time:.2f} 秒最终 DataFrame 形状: {res_df.shape}")
    gc.collect()
    return res_df


# --- 基于 Gini 的快速特征筛选  ---

def threads_feature_select(
    df: pd.DataFrame,
    target_name: str,
    candidate_feature: List[str],
    transformer: Transform,
    logger: Any,
    metric: str = 'gini',
    return_score: bool = False,
    n_jobs: int = -1,
    gini_threshold: float = 0.0001
    ) -> Tuple[List[str], pd.DataFrame, Dict[str, float]]:
    """
    使用多线程/多进程并行计算候选特征的值,并通过 Gini 指数进行快速筛选
    

    Args:
        df (pd.DataFrame): 包含基础特征和目标列的 DataFrame
        target_name (str): 目标列的名称
        candidate_feature (List[str]): 候选特征名称列表
        transformer (Transform): 用于计算特征值的 Transform 实例
        logger (logger): loguru logger 实例
        metric (str): 筛选指标（固定为 'gini'）
        return_score (bool): 是否返回计算出的 Gini 分数
        n_jobs (int): 并行工作数
        gini_threshold (float): Gini 分数的筛选阈值（大于此值才被选中）

    Returns:
        Tuple[List[str], pd.DataFrame, Dict[str, float]]:
            - 筛选后选中的特征名称列表
            - 只包含筛选后选中特征列的新 DataFrame
            - 包含所有计算特征及其 Gini 分数的字典
    """
    logger.info(f"开始基于 '{metric}' 的并行特征筛选 (共 {len(candidate_feature)} 个候选)...")
    start_time = time.perf_counter()

    selected_features: List[str] = []
    features_scores: Dict[str, float] = {}
    selected_features_df = pd.DataFrame(index=df.index) # 初始化为空

    if not candidate_feature:
        logger.warning("候选特征列表为空,筛选结束")
        return selected_features, selected_features_df, features_scores

    if target_name not in df.columns:
        raise ValueError(f"目标列 '{target_name}' 不在输入的 DataFrame 中")
    y_true_series = df[target_name]

    # 1. 并行计算所有候选特征的值
    logger.debug("步骤 1/3: 并行计算所有候选特征值...")
    df_with_candidates = pd.DataFrame(index=df.index) # 存储新特征
    try:
        # 调用 max_threads_name2feature 计算,返回包含+新特征的 DF
        df_temp_with_all = max_threads_name2feature(
            df=df, feature_names=candidate_feature,
            transformer=transformer, logger=logger, n_jobs=n_jobs
        )
        # 提取新计算出的候选特征列
        calculated_candidate_names = [f for f in candidate_feature if f in df_temp_with_all.columns and f not in df.columns]
        if not calculated_candidate_names:
             logger.warning("未能成功计算任何候选特征的值,筛选结束")
             return selected_features, selected_features_df, features_scores
        logger.debug(f"成功计算了 {len(calculated_candidate_names)} 个新特征的值")
        # 只保留新计算出的特征用于 Gini 计算
        df_with_candidates = df_temp_with_all[calculated_candidate_names].copy()

    except Exception as e:
         logger.exception(f"并行计算候选特征值时出错: {e}")
         return selected_features, selected_features_df, features_scores # 返回空

    # 2. 计算 Gini 分数
    X_candidates = df_with_candidates.select_dtypes(include=np.number) # 只对数值特征计算
    if X_candidates.empty:
         logger.warning("计算出的候选特征均非数值类型,无法计算 Gini")
         # 即使没有数值特征,也要返回空的 scores dict（如果 return_score=True）
         return selected_features, selected_features_df, features_scores

    logger.debug(f"步骤 2/3: 为 {len(X_candidates.columns)} 个数值候选特征计算 Gini 分数...")
    try:
        gini_scores_array = autofe_utils.calc_ginis(X_candidates.to_numpy(), y_true_series.to_numpy())
        features_scores = dict(zip(X_candidates.columns, map(float, gini_scores_array)))
        logger.debug(f"Gini 分数计算完成,共 {len(features_scores)} 个")
    except Exception as e:
         logger.exception(f"计算 Gini 分数时出错: {e}")
         features_scores = {} # 清空分数

    # 3. 根据 Gini 阈值筛选特征
    logger.debug(f"步骤 3/3: 根据 Gini > {gini_threshold} 进行筛选...")
    selected_features = [
        name for name, score in features_scores.items()
        if pd.notna(score) and np.isfinite(score) and score > gini_threshold
    ]
    # 按 Gini 分数降序排序
    selected_features.sort(key=lambda name: features_scores.get(name, -np.inf), reverse=True)
    logger.info(f"根据 Gini > {gini_threshold} 筛选,选中 {len(selected_features)} 个特征")

    # 4. 构建只包含选中特征的 DataFrame
    if selected_features:
        # 从包含所有计算出的候选特征的 DataFrame 中选取
        selected_features_df = df_with_candidates[selected_features].copy()

    end_time = time.perf_counter()
    logger.info(f"并行特征筛选 ('threads_feature_select') 完成,耗时: {end_time - start_time:.2f} 秒")

    # 根据 return_score 返回结果
    if return_score:
        return selected_features, selected_features_df, features_scores
    else:
        # 如果代码不返回分数,则返回空字典
        return selected_features, selected_features_df, {}


# --- 基于模型的特征选择  ---

# @common_utils.log_execution_time(level="INFO")
def model_features_select(
    fes: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    baseline: float,
    metric: str,
    task_type: str,
    logger: Any,
    seed: int,
    cat_features: List[str],
    time_index: Optional[str] = None,
    model_params: Optional[Dict] = None,
    importance_threshold: float = 1e-5 # 重要性阈值
    ) -> Tuple[List[str], float]:
    """
    执行基于模型的特征选择（使用 LightGBM 和特征重要性）
    

    Args:
        fes (Tuple): (X_train, X_val, y_train, y_val)
        baseline (float): 用于比较的基线分数
        metric (str): 评估指标
        task_type (str): 任务类型
        logger (logger): loguru logger 实例
        seed (int): 随机种子
        cat_features (List[str]): 分类特征名称列表
        time_index (Optional[str]): 时间索引名称（未使用）
        model_params (Optional[Dict]): 覆盖默认 LGBM 参数
        importance_threshold (float): 特征重要性的筛选阈值

    Returns:
        Tuple[List[str], float]:
            - 筛选后保留的特征名称列表 (包含特征和被选中的高级特征)
            - 使用筛选后特征在验证集上达到的最终分数
    """
    # 引用参数避免"未存取"警告
    _ = time_index

    if not LGBM_INSTALLED: # 检查 LGBM 是否导入成功
        logger.error("LightGBM 未安装,无法执行 model_features_select")
        # 失败时返回空特征列表和最差分数
        worst_score = -np.inf if metric in {'roc_auc', 'f1', 'accuracy', 'auc_mu', 'r2', 'ap'} else np.inf
        return [], worst_score

    try:
        X_train, X_val, y_train, y_val = fes
        # 深拷贝以防修改数据
        X_train_lgb = X_train.copy()
        X_val_lgb = X_val.copy()
        y_train_lgb = y_train.copy()
        y_val_lgb = y_val.copy()
    except (TypeError, ValueError, IndexError): # 更具体的异常捕获
        raise ValueError("输入 'fes' 必须是包含 (X_train, X_val, y_train, y_val) 且非空的元组")

    logger.info("开始执行基于模型的特征选择 (LightGBM)...")
    logger.debug(f"Input shapes: X_train={X_train_lgb.shape}, X_val={X_val_lgb.shape}")
    logger.debug(f"Task: {task_type}, Metric: {metric}, Baseline: {baseline}")

    # 支持的评估指标
    supported_metrics = {
        'regression': ['rmse', 'mae', 'mse', 'msle', 'r2'],
        'classification': ['auc', 'roc_auc', 'neg_log_loss', 'accuracy', 'f1', 'auc_mu', 'ap']
    }
    
    current_mode = 'regression' if task_type == 'regression' else 'classification'
    if metric not in supported_metrics.get(current_mode, []):
        logger.warning(f"评估指标 '{metric}' 不在推荐的 {current_mode} 指标列表中，可能不受完全支持")
    
    higher_is_better = metric in {'roc_auc', 'f1', 'accuracy', 'auc_mu', 'r2', 'ap'}

    # --- 模型参数准备 ---
    lgbm_params = { # 默认参数
        'objective': 'regression_l1' if task_type == 'regression' else 'binary',
        'metric': 'mae' if task_type == 'regression' else 'auc',
        'n_estimators': 200, 
        'learning_rate': 0.05, 
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 
        'bagging_freq': 1, 
        'lambda_l1': 0.1,
        'lambda_l2': 0.1, 
        'num_leaves': 31, 
        'verbose': -1, 
        'n_jobs': -1,
        'seed': seed, 
        'boosting_type': 'gbdt',
    }
    
    # 任务特定优化
    if task_type != 'regression':
         lgbm_params['is_unbalance'] = True
         # 映射评估指标
         metric_mapping = {
             'neg_log_loss': 'binary_logloss',
             'f1': 'f1',
             'accuracy': 'accuracy',
             'auc': 'auc',
             'roc_auc': 'auc'
         }
         lgbm_params['metric'] = metric_mapping.get(metric, 'auc')

    if model_params: # 用户覆盖
        logger.debug(f"使用用户提供的 LGBM 参数覆盖默认值: {model_params}")
        lgbm_params.update(model_params)

    # --- 数据准备 (分类特征) ---
    categorical_cols = [col for col in cat_features if col in X_train_lgb.columns]
    logger.debug(f"模型选择中使用的分类特征: {categorical_cols}")
    
    # 检查并处理分类特征
    valid_categorical_cols = []
    for col in categorical_cols:
        try: # 尝试统一类型为 category
            # 检查特征是否有足够的值
            if X_train_lgb[col].nunique() <= 1:
                logger.warning(f"分类特征 '{col}' 只有一个唯一值，不作为分类特征处理")
                continue
                
            train_cat = X_train_lgb[col].astype('category')
            val_cat = X_val_lgb[col].astype('category')
            common_cats = pd.api.types.union_categoricals([train_cat, val_cat], sort_categories=True).categories
            X_train_lgb[col] = pd.Categorical(train_cat, categories=common_cats)
            X_val_lgb[col] = pd.Categorical(val_cat, categories=common_cats)
            valid_categorical_cols.append(col)
        except Exception as cat_e:
             logger.warning(f"处理分类特征 '{col}' 时出错,将尝试让 LGBM 自动处理: {cat_e}")
             # 添加到valid_categorical_cols，让LGBM尝试处理
             valid_categorical_cols.append(col)

    # 移除具有过多缺失值的特征
    missing_thresh = 0.9  # 允许的最大缺失比例
    valid_features = []
    for col in X_train_lgb.columns:
        missing_ratio = X_train_lgb[col].isna().mean()
        if missing_ratio < missing_thresh:
            valid_features.append(col)
        else:
            logger.warning(f"特征 '{col}' 缺失值比例 {missing_ratio:.2%} 超过阈值 {missing_thresh:.2%}，被移除")
    
    if len(valid_features) < len(X_train_lgb.columns):
        X_train_lgb = X_train_lgb[valid_features]
        X_val_lgb = X_val_lgb[valid_features]
        # 更新分类特征列表
        valid_categorical_cols = [col for col in valid_categorical_cols if col in valid_features]
        logger.info(f"移除高缺失特征后，保留 {len(valid_features)}/{len(X_train.columns)} 个特征")

    # --- 训练与筛选 ---
    selected_features: List[str] = list(X_train_lgb.columns) # 默认返回所有
    final_score: float = baseline # 默认返回基线
    model = None

    try:
        logger.debug("训练初始 LGBM 模型以获取特征重要性...")
        model_cls = lgb.LGBMRegressor if task_type == 'regression' else lgb.LGBMClassifier
        model = model_cls(**lgbm_params)

        # 预处理数据 - 填充缺失值
        X_train_lgb = X_train_lgb.fillna(X_train_lgb.median())
        X_val_lgb = X_val_lgb.fillna(X_train_lgb.median())  # 用训练集的中位数填充验证集

        # 准备 fit 参数
        fit_params = {
            "X": X_train_lgb, 
            "y": y_train_lgb,
            "eval_set": [(X_val_lgb, y_val_lgb)],
            # 适配 eval_metric (LGBM 可能不支持所有 sklearn metric name)
            "eval_metric": 'logloss' if metric == 'neg_log_loss' else ('l1' if metric == 'mae' else metric),
            "callbacks": [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=0)], # 假设使用早停
            # 显式传递 categorical_feature
            "categorical_feature": valid_categorical_cols if valid_categorical_cols else 'auto'
        }
        
        model.fit(**fit_params)
        logger.debug("初始 LGBM 模型训练完成")

        # 获取重要性并筛选
        if hasattr(model, 'feature_importances_'):
            importances = pd.Series(model.feature_importances_, index=X_train_lgb.columns)
            logger.trace(f"特征重要性: \n{importances.sort_values(ascending=False)}")
            
            # 动态调整重要性阈值
            if importances.max() < 10 * importance_threshold:
                # 如果所有重要性都较低，降低阈值
                adj_threshold = importances.max() / 10
                logger.info(f"特征重要性普遍较低，调整阈值从 {importance_threshold} 到 {adj_threshold}")
                importance_threshold = adj_threshold
                
            selected_features = importances[importances > importance_threshold].index.tolist()
            logger.info(f"根据重要性 (> {importance_threshold}) 筛选出 {len(selected_features)} 个特征")

            if not selected_features:
                 logger.warning("基于重要性的模型筛选未选中任何特征!将保留所有特征")
                 selected_features = list(X_train_lgb.columns)
                 # 重新计算分数（使用所有特征）
                 if metric == 'neg_log_loss': preds = model.predict_proba(X_val_lgb)
                 else: preds = model.predict(X_val_lgb)
                 final_score = _calculate_metric(metric, y_val_lgb, preds, logger) # 使用内部函数
            else:
                # 使用选中特征重新评估分数
                logger.debug(f"使用 {len(selected_features)} 个选定特征重新评估...")
                # 确保只使用选中的列,并处理分类特征
                X_train_selected = X_train_lgb[selected_features]
                X_val_selected = X_val_lgb[selected_features]
                categorical_selected = [c for c in valid_categorical_cols if c in selected_features]

                # 可以在此重新训练,但更常见的是直接用 model.best_score_ (如果用了早停)
                if hasattr(model, 'best_score_') and model.best_score_:
                    try:
                        valid_scores = model.best_score_['valid_0']
                        # 需要映射 metric 到 LGBM 的度量名称
                        lgbm_metric_name = 'l1' if metric == 'mae' else \
                                           'binary_logloss' if metric == 'neg_log_loss' else \
                                           metric # 其他假设名称一致
                        if lgbm_metric_name in valid_scores:
                             final_score = valid_scores[lgbm_metric_name]
                             if metric == 'neg_log_loss': final_score = -final_score # 取反
                             logger.debug(f"使用模型的 best_score_ 作为最终分数: {final_score:.6f}")
                        else: raise KeyError(f"Metric '{lgbm_metric_name}' not found in best_score_")
                    except (KeyError, TypeError, AttributeError) as score_e:
                         logger.warning(f"无法从 model.best_score_ 获取分数: {score_e}使用最后预测评估")
                         # 回退用完整模型在验证集预测
                         if metric == 'neg_log_loss': preds_final = model.predict_proba(X_val_lgb)
                         else: preds_final = model.predict(X_val_lgb)
                         final_score = _calculate_metric(metric, y_val_lgb, preds_final, logger)
                else: # 没有早停或 best_score_,用最后预测评估
                     logger.debug("未使用早停或无 best_score_,使用最终模型在验证集上的评估分数")
                     if metric == 'neg_log_loss': preds_final = model.predict_proba(X_val_lgb)
                     else: preds_final = model.predict(X_val_lgb)
                     final_score = _calculate_metric(metric, y_val_lgb, preds_final, logger)
        else: # 模型不支持重要性
            logger.warning("模型不支持 feature_importances_,返回所有特征")
            selected_features = list(X_train_lgb.columns)
            if metric == 'neg_log_loss': preds = model.predict_proba(X_val_lgb)
            else: preds = model.predict(X_val_lgb)
            final_score = _calculate_metric(metric, y_val_lgb, preds, logger)

    except ImportError as imp_err: # 已在函数开始检查,但再次捕获以防万一
         logger.error(f"ImportError: {imp_err}")
         return list(X_train.columns), baseline
    except Exception as e:
         logger.exception(f"模型特征选择过程中发生严重错误: {e}")
         return list(X_train.columns), baseline # 失败时返回特征和基线

    # 比较分数
    score_improved = (higher_is_better and final_score > baseline) or \
                     (not higher_is_better and final_score < baseline)

    if score_improved: logger.info(f"模型特征选择完成,选中 {len(selected_features)} 个特征,最终分数 {final_score:.6f} 优于基线 {baseline:.6f}")
    else: logger.info(f"模型特征选择完成,选中 {len(selected_features)} 个特征,但最终分数 {final_score:.6f} 未优于基线 {baseline:.6f}")

    return selected_features, float(final_score) # 确保返回 float

def _calculate_metric(metric_name: str, y_true: pd.Series, y_pred: np.ndarray, logger: Any) -> float:
    """(内部函数) 根据配置的 metric 计算得分"""
    try:
        if metric_name == 'rmse':
            y_pred_num = pd.to_numeric(y_pred, errors='coerce')
            if not np.all(np.isfinite(y_pred_num)): return np.inf
            return np.sqrt(mean_squared_error(y_true, y_pred_num))
        elif metric_name == 'neg_log_loss':
             if y_pred.ndim != 2 or y_pred.shape[0] != len(y_true): return np.inf
             eps = 1e-15; y_pred_clip = np.clip(y_pred, eps, 1 - eps)
             return -log_loss(y_true, y_pred_clip, eps=eps)
        elif metric_name == 'accuracy': return accuracy_score(y_true, y_pred)
        elif metric_name == 'f1': return f1_score(y_true, y_pred, average='weighted')
        else: raise NotImplementedError(f"不支持的评估指标: {metric_name}")
    except ValueError as ve:
         logger.error(f"计算指标 '{metric_name}' 时出错: {ve}")
         return np.inf if metric_name in ['rmse', 'neg_log_loss'] else -np.inf
    except Exception as e:
         logger.exception(f"计算指标 '{metric_name}' 时发生未知错误: {e}")
         return np.inf if metric_name in ['rmse', 'neg_log_loss'] else -np.inf

logger.info("AutoFE 算法函数模块 (espml.autofe.algorithm) 加载完成")