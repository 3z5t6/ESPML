# -*- coding: utf-8 -*-
"""
AutoFE 特征评估模型 (espml)
用于快速评估新生成特征的价值（性能提升和重要性）
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score, f1_score
from loguru import logger

# 导入 AutoFE 内部工具 (如果需要清理特征名等)
from espml.autofe import utils as autofe_utils
# 导入项目级通用工具 (如果需要计时器等)
# from espml.util import utils as common_utils

# --- 默认评估模型参数  ---
DEFAULT_EVAL_MODEL_PARAMS_REGRESSION: Dict[str, Any] = {
    'n_estimators': 40,
    'max_depth': 10,
    'max_features': 1.0, # 假设代码包含此参数
    'min_samples_leaf': 5,
    'n_jobs': -1,
    'random_state': None, # 由函数参数传入覆盖
    # 'criterion': 'squared_error' # 较新 sklearn 版本默认
}

DEFAULT_EVAL_MODEL_PARAMS_CLASSIFICATION: Dict[str, Any] = {
    'n_estimators': 40,
    'max_depth': 10,
    'max_features': 'sqrt', # 分类常用默认值
    'min_samples_leaf': 5,
    'n_jobs': -1,
    'random_state': None,
    'class_weight': 'balanced'
    # 'criterion': 'gini' # 默认值
}

# --- 内部评估函数 ---
def _calculate_metric(metric_name: str, y_true: pd.Series, y_pred: np.ndarray) -> float:
    """(内部函数) 根据配置的 metric 计算得分"""
    if metric_name == 'rmse':
        # 确保预测值也是数值类型且有限
        if not pd.api.types.is_numeric_dtype(y_pred): return np.inf
        y_pred = np.nan_to_num(y_pred, nan=np.inf, posinf=np.inf, neginf=-np.inf)
        if not np.all(np.isfinite(y_pred)): return np.inf # 如果仍然有 Inf
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric_name == 'neg_log_loss': # 分类概率
        # y_pred 应该是 N x K 概率数组
        if y_pred.ndim != 2 or y_pred.shape[0] != len(y_true):
             logger.error(f"计算 neg_log_loss 时预测形状不匹配: {y_pred.shape}, 真实值长度: {len(y_true)}")
             return np.inf # 返回最差值
        # 确保概率值在 (0, 1) 范围内
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -log_loss(y_true, y_pred, eps=eps) # 返回负对数似然
    elif metric_name == 'accuracy': # 分类标签
        return accuracy_score(y_true, y_pred)
    elif metric_name == 'f1': # 分类标签
        # 需要指定 average 参数,与代码保持一致,此处假设 'weighted'
        return f1_score(y_true, y_pred, average='weighted')
    # 可以添加 MAE 等其他代码支持的指标
    # elif metric_name == 'mae':
    #    from sklearn.metrics import mean_absolute_error
    #    return mean_absolute_error(y_true, y_pred)
    else:
         logger.error(f"不支持的评估指标: {metric_name}")
         raise NotImplementedError(f"不支持的评估指标: {metric_name}")


# --- 核心评估函数  ---
# @common_utils.log_execution_time(level="DEBUG") # 可选添加计时器
def evaluate_feature_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    new_feature: pd.Series,
    feature_name: str,
    base_features: List[str],
    task_type: str = 'regression',
    metric: str = 'rmse',
    eval_model_params: Optional[Dict[str, Any]] = None,
    validation_size: float = 0.25,
    random_seed: Optional[int] = None
    ) -> Tuple[float, float]:
    """
    使用 ExtraTrees 模型快速评估单个新特征对预测性能的提升

    严格遵循代码的评估流程

    Args:
        X_train (pd.DataFrame): 包含基础特征的训练数据
        y_train (pd.Series): 训练数据的目标变量
        new_feature (pd.Series): 新生成的特征列 (索引需与 X_train 对齐)
        feature_name (str): 新特征的名称
        base_features (List[str]): 用于基线模型的基础特征列表
        task_type (str): 任务类型 ('regression' 或 'classification')
        metric (str): 评估指标 ('rmse', 'neg_log_loss', 'accuracy', 'f1' 等)
        eval_model_params (Optional[Dict[str, Any]]): 覆盖默认评估模型的参数
        validation_size (float): 用于内部验证的拆分比例 (0 < size < 1)
        random_seed (Optional[int]): 用于模型和数据拆分的随机种子

    Returns:
        Tuple[float, float]: (性能提升值, 新特征的重要性分数)
                             性能提升根据 metric 定义（越小越好或越大越好）
                             重要性分数基于模型的 feature_importances_
                             评估失败或特征无效时返回 (-np.inf, 0.0)
    """
    # logger.trace(f"开始评估新特征 '{feature_name}' (Metric: {metric})...")

    # --- 输入验证  ---
    if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series) or not isinstance(new_feature, pd.Series):
        logger.error("EVAL: 输入类型错误")
        return -np.inf, 0.0
    if X_train.empty or y_train.empty:
        logger.error("EVAL: 输入数据为空")
        return -np.inf, 0.0
    if not X_train.index.equals(y_train.index):
         logger.error("EVAL: X_train 和 y_train 索引不匹配")
         return -np.inf, 0.0
    if new_feature.isna().all():
         logger.trace(f"EVAL: 新特征 '{feature_name}' 全为 NaN,评估无效")
         return -np.inf, 0.0
    if not new_feature.index.equals(X_train.index):
        logger.warning(f"EVAL: 新特征 '{feature_name}' 索引与训练数据不匹配,尝试重新对齐...")
        try:
             new_feature = new_feature.reindex(X_train.index)
             if new_feature.isna().all():
                  logger.error(f"EVAL: 特征 '{feature_name}' 重新对齐后全为 NaN")
                  return -np.inf, 0.0
        except Exception as e:
             logger.error(f"EVAL: 重新对齐特征 '{feature_name}' 索引失败: {e}")
             return -np.inf, 0.0

    # --- 数据准备  ---
    valid_base_features = [f for f in base_features if f in X_train.columns]
    if not valid_base_features:
         logger.error("EVAL: 基础特征列表无效或不在 X_train 中")
         return -np.inf, 0.0
    X_base = X_train[valid_base_features].copy()

    # 组合特征
    clean_feature_name = feature_name # 假设代码不清理名称
    if clean_feature_name in X_base.columns:
         logger.warning(f"EVAL: 新特征名 '{clean_feature_name}' 与基础特征冲突")
    X_combined = X_base.copy()
    X_combined[clean_feature_name] = new_feature

    # 预先填充 NaN 为 0
    try:
        X_base = X_base.fillna(0).astype(float) # 确保是浮点数
        X_combined = X_combined.fillna(0).astype(float)
        # 检查填充后是否还有 Inf 或过大/过小的值
        if not np.all(np.isfinite(X_base.values)):
             logger.warning("EVAL: 填充 NaN 后基础特征中仍存在非有限值,评估可能失败")
             # 可以选择替换 Inf: X_base = X_base.replace([np.inf, -np.inf], 0)
        if not np.all(np.isfinite(X_combined.values)):
             logger.warning("EVAL: 填充 NaN 后组合特征中仍存在非有限值,评估可能失败")
    except Exception as fill_e:
        logger.error(f"EVAL: 填充 NaN 或转换类型时失败: {fill_e}")
        return -np.inf, 0.0

    # --- 模型和评估设置  ---
    is_regression = task_type == 'regression'
    model_cls = ExtraTreesRegressor if is_regression else ExtraTreesClassifier
    default_params = DEFAULT_EVAL_MODEL_PARAMS_REGRESSION if is_regression else DEFAULT_EVAL_MODEL_PARAMS_CLASSIFICATION
    # 确定指标方向
    higher_is_better = metric in ['accuracy', 'f1'] # 添加其他越大越好的指标

    current_eval_params = default_params.copy() # 使用副本
    if eval_model_params:
        current_eval_params.update(eval_model_params)
    current_eval_params['random_state'] = random_seed # 确保设置了种子

    # --- 内部验证集拆分  ---
    try:
        # 确保 validation_size 合法
        if not 0 < validation_size < 1:
             raise ValueError(f"validation_size ({validation_size}) 必须在 (0, 1) 之间")
        # 分类任务使用分层抽样
        stratify_opt = y_train if not is_regression and y_train.nunique() > 1 else None
        X_tr_base, X_val_base, y_tr, y_val = train_test_split(
            X_base, y_train, test_size=validation_size,
            random_state=random_seed, shuffle=True, stratify=stratify_opt
        )
        # 使用相同索引获取组合数据的拆分
        X_tr_comb = X_combined.loc[X_tr_base.index]
        X_val_comb = X_combined.loc[X_val_base.index]
        logger.trace(f"EVAL: 内部验证集拆分完成: Train={len(X_tr_base)}, Val={len(X_val_base)}")
    except Exception as e:
         logger.error(f"EVAL: 评估特征 '{feature_name}' 时,拆分验证集失败: {e}")
         return -np.inf, 0.0

    # --- 评估基线模型  ---
    baseline_score = np.inf if not higher_is_better else -np.inf
    model_base = model_cls(**current_eval_params) # 每次都创建新实例
    try:
        # logger.trace(f"EVAL: 训练基线模型...")
        model_base.fit(X_tr_base, y_tr)
        # logger.trace("EVAL: 基线模型训练完成")
        if metric == 'neg_log_loss':
             preds_base = model_base.predict_proba(X_val_base)
        else:
             preds_base = model_base.predict(X_val_base)
        baseline_score = _calculate_metric(metric, y_val, preds_base)
        logger.trace(f"EVAL: 特征 '{feature_name}' 基线得分 ({metric}) = {baseline_score:.6f}")
    except Exception as e:
         logger.error(f"EVAL: 评估基线模型失败 (特征: {feature_name}): {e}", exc_info=True)
         return -np.inf, 0.0 # 基线失败无法继续

    # --- 评估新模型  ---
    new_score = np.inf if not higher_is_better else -np.inf
    feature_importance_val = 0.0
    model_new = model_cls(**current_eval_params) # 每次都创建新实例
    try:
        # logger.trace(f"EVAL: 训练新模型 (含 '{clean_feature_name}')...")
        model_new.fit(X_tr_comb, y_tr)
        # logger.trace("EVAL: 新模型训练完成")
        if metric == 'neg_log_loss':
             preds_new = model_new.predict_proba(X_val_comb)
        else:
             preds_new = model_new.predict(X_val_comb)
        new_score = _calculate_metric(metric, y_val, preds_new)

        # 获取特征重要性 
        if hasattr(model_new, 'feature_importances_'):
            try:
                importances = pd.Series(model_new.feature_importances_, index=X_tr_comb.columns)
                # 使用 feature_name (未清理的) 或清理后的 clean_feature_name 查找
                if clean_feature_name in importances.index:
                     feature_importance_val = float(importances[clean_feature_name])
                elif feature_name in importances.index: # 尝试名称
                     feature_importance_val = float(importances[feature_name])
                else:
                     logger.warning(f"EVAL: 无法在模型重要性中找到特征 '{clean_feature_name}' 或 '{feature_name}',重要性设为 0")
                     feature_importance_val = 0.0
            except Exception as imp_e:
                 logger.warning(f"EVAL: 获取特征 '{clean_feature_name}'/'{feature_name}' 重要性失败: {imp_e}")
                 feature_importance_val = 0.0
        else:
             feature_importance_val = 0.0 # 模型不支持

        logger.trace(f"EVAL: 特征 '{feature_name}' 新得分 ({metric}) = {new_score:.6f}, 重要性 = {feature_importance_val:.6f}")

    except Exception as e:
         logger.error(f"EVAL: 评估加入新特征 '{feature_name}' 的模型失败: {e}", exc_info=True)
         return -np.inf, 0.0 # 新模型失败则无提升

    # --- 计算性能提升  ---
    if not np.isfinite(baseline_score) or not np.isfinite(new_score):
        performance_gain = -np.inf
        logger.warning(f"EVAL: 无法计算特征 '{feature_name}' 的性能提升,因为得分无效 (baseline={baseline_score}, new={new_score})")
    elif higher_is_better:
        performance_gain = new_score - baseline_score
    else: # Lower is better
        performance_gain = baseline_score - new_score

    # logger.debug(f"EVAL: 特征 '{feature_name}' 性能提升 ({metric}) = {performance_gain:.6f}")

    # 返回标准 float
    return float(performance_gain), float(feature_importance_val)


logger.info("AutoFE 特征评估模型模块 (espml.autofe.model) 加载完成")