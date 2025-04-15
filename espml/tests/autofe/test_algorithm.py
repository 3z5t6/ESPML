# tests/autofe/test_algorithm.py
# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring, redefined-outer-name, protected-access, unused-argument, line-too-long
"""
espml.autofe.algorithm 模块核心函数的单元测试
验证并行特征计算、Gini 筛选和模型选择逻辑
需要大量使用 mock
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call, ANY # 导入 mock 工具
from typing import List, Dict, Any, Tuple
import hashlib # 用于模拟特征生成
import os

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split # 用于设置 n_jobs

# 导入被测函数和依赖类
# 需要确保espml在测试环境中是可导入的
try:
    from espml.autofe.algorithm import (
        max_threads_name2feature,
        threads_feature_select,
        model_features_select,
        _calculate_metric # 导入内部函数进行测试
    )
    from espml.autofe.transform import Transform # 需要模拟 Transform 实例
    from espml.autofe import utils as autofe_utils # 需要 Gini 计算
    from espml.util import utils as common_utils # 可能需要
    from loguru import logger # 需要 logger 实例
except ImportError as e:
    pytest.skip(f"跳过 algorithm 测试，因为无法导入 espml 模块: {e}", allow_module_level=True)

# 尝试导入 lightgbm 用于 mock 类型提示
try:
    import lightgbm as lgb
    LGBM_INSTALLED = True
except ImportError:
    lgb = None # type: ignore
    LGBM_INSTALLED = False


# --- Fixtures (部分来自 conftest.py) ---
@pytest.fixture
def mock_transformer(mocker) -> MagicMock:
    """创建一个模拟的 Transform 实例"""
    mock = MagicMock(spec=Transform)
    # 模拟 transform 方法接收 df 和 feature_list，返回包含新列的 df
    def mock_transform_logic(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
        res_df = df.copy()
        for feature_name in feature_list:
            if feature_name not in res_df.columns:
                 # 使用确定性方式生成模拟数据
                 np.random.seed(abs(hash(feature_name)) % (2**32 - 1)) # 根据特征名设种子
                 res_df[feature_name] = pd.Series(np.random.rand(len(df)) * 10, index=df.index)
        return res_df
    mock.transform.side_effect = mock_transform_logic
    # 模拟 fit 方法 (对于需要 fit 的 transform)
    mock.fit.return_value = mock # fit 返回自身
    # 模拟 _record (如果需要)
    mock._record = {}
    return mock

@pytest.fixture
def sample_dataframe_clf() -> pd.DataFrame:
    """提供分类任务的 DataFrame"""
    np.random.seed(42)
    return pd.DataFrame({
        'A': np.random.rand(30) * 10,
        'B': np.random.rand(30) * 5,
        'cat1': np.random.choice(['X', 'Y', 'Z', 'X', 'Y'], 30),
        'target': np.random.randint(0, 2, 30) # 二分类目标
    }, index=pd.date_range('2024-01-01', periods=30, freq='D', name="datetime"))

@pytest.fixture
def sample_dataframe_reg() -> pd.DataFrame:
    """提供回归任务的 DataFrame"""
    np.random.seed(43)
    return pd.DataFrame({
        'feat1': np.random.rand(40) * 100,
        'feat2': np.random.rand(40) * 50 + 10,
        'target': np.random.rand(40) * 20 + 5
    }, index=pd.date_range('2024-02-01', periods=40, freq='D', name="datetime"))

# --- 测试 max_threads_name2feature ---
# (之前的实现已包含基本测试，此处可补充)
def test_max_threads_name2feature_empty_list(sample_dataframe_clf: pd.DataFrame, mock_transformer: MagicMock):
    """测试传入空特征列表"""
    df_result = max_threads_name2feature(
        df=sample_dataframe_clf, feature_names=[],
        transformer=mock_transformer, logger=logger, n_jobs=1
    )
    assert df_result.equals(sample_dataframe_clf) # 应返回 DF
    mock_transformer.transform.assert_not_called() # 不应调用 transform

@patch('espml.autofe.algorithm.ThreadPoolExecutor', MagicMock()) # 禁用实际并行以简化测试
def test_max_threads_name2feature_mixed_results(sample_dataframe_clf: pd.DataFrame, mock_transformer: MagicMock):
    """测试部分特征成功，部分失败"""
    features_to_calc = ["log###A$$$", "error_feat", "add###A|||B$$$"]
    # 配置 mock transform 行为
    def transform_side_effect(df, f_list):
        fname = f_list[0]
        if fname == "error_feat":
            raise ValueError("Simulated Error")
        else: # 成功计算
            res_df = df.copy()
            np.random.seed(abs(hash(fname)) % (2**32 - 1))
            res_df[fname] = pd.Series(np.random.rand(len(df)), index=df.index)
            return res_df
    mock_transformer.transform.side_effect = transform_side_effect

    df_result = max_threads_name2feature(
        df=sample_dataframe_clf, feature_names=features_to_calc,
        transformer=mock_transformer, logger=logger, n_jobs=1 # 使用串行测试逻辑
    )
    assert "log###A$$$" in df_result.columns
    assert "add###A|||B$$$" in df_result.columns
    assert "error_feat" not in df_result.columns # 失败的不添加
    assert mock_transformer.transform.call_count == len(features_to_calc) # 都尝试了

# --- 测试 threads_feature_select ---
# 需要 mock max_threads_name2feature 和 autofe_utils.calc_ginis
@patch('espml.autofe.algorithm.max_threads_name2feature')
@patch('espml.autofe.algorithm.autofe_utils.calc_ginis')
def test_threads_feature_select_all_numeric(mock_calc_ginis, mock_max_threads, sample_dataframe_clf: pd.DataFrame, mock_transformer: MagicMock):
    """测试所有候选特征都是数值型"""
    target = 'target'; y_true = sample_dataframe_clf[target]
    candidates = ["num_feat1", "num_feat2", "num_feat3"]
    # 模拟 max_threads 返回的 DF
    df_with_cands = sample_dataframe_clf.copy()
    df_with_cands["num_feat1"] = np.random.rand(len(y_true))
    df_with_cands["num_feat2"] = np.random.rand(len(y_true)) * 0.1
    df_with_cands["num_feat3"] = np.random.rand(len(y_true)) + 1
    mock_max_threads.return_value = df_with_cands
    # 模拟 Gini 分数
    mock_calc_ginis.return_value = np.array([0.2, 0.005, 0.3]) # 对应 num1, num2, num3

    selected, selected_df, scores = threads_feature_select(
        df=sample_dataframe_clf, target_name=target, candidate_feature=candidates,
        transformer=mock_transformer, logger=logger, return_socre=True,
        gini_threshold=0.01 # 阈值
    )

    mock_max_threads.assert_called_once()
    # 验证传递给 calc_ginis 的是包含三个新特征的数组
    call_args_gini, _ = mock_calc_ginis.call_args
    assert call_args_gini[0].shape[1] == 3
    assert sorted(selected) == sorted(["num_feat1", "num_feat3"]) # num_feat2 被过滤
    assert sorted(list(selected_df.columns)) == sorted(["num_feat1", "num_feat3"])
    assert np.isclose(scores['num_feat2'], 0.005)

@patch('espml.autofe.algorithm.max_threads_name2feature')
@patch('espml.autofe.algorithm.autofe_utils.calc_ginis')
def test_threads_feature_select_no_numeric_candidates(mock_calc_ginis, mock_max_threads, sample_dataframe_clf: pd.DataFrame, mock_transformer: MagicMock):
    """测试候选特征计算后没有数值列"""
    target = 'target'; candidates = ["cat_feat1", "cat_feat2"]
    df_with_cats = sample_dataframe_clf.copy()
    df_with_cats["cat_feat1"] = ['m', 'n'] * 15
    df_with_cats["cat_feat2"] = ['p', 'q'] * 15
    mock_max_threads.return_value = df_with_cats
    # Gini 不会被调用

    selected, selected_df, scores = threads_feature_select(
        df=sample_dataframe_clf, target_name=target, candidate_feature=candidates,
        transformer=mock_transformer, logger=logger, return_socre=True, gini_threshold=0.01
    )
    mock_calc_ginis.assert_not_called() # 不应调用 Gini
    assert selected == []
    assert selected_df.empty
    assert scores == {}

# --- 测试 _calculate_metric ---
# （已在 test_autofe_model.py 或上一步骤中充分测试，此处省略）

# --- 测试 model_features_select ---
# 需要 mock LightGBM
@pytest.mark.skipif(not LGBM_INSTALLED, reason="LightGBM not installed")
@patch('espml.autofe.algorithm.lgb')
def test_model_features_select_regression(mock_lgb, sample_dataframe_reg: pd.DataFrame):
    """测试回归任务的模型选择"""
    X = sample_dataframe_reg[['feat1', 'feat2']]
    y = sample_dataframe_reg['target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)

    mock_model_inst = MagicMock()
    mock_model_inst.feature_importances_ = np.array([100, 5]) # feat1 重要
    mock_model_inst.best_score_ = {'valid_0': {'mae': 2.5}} # 假设 metric='mae'
    mock_lgb.LGBMRegressor.return_value = mock_model_inst

    selected, score = model_features_select(
        fes=(X_train, X_val, y_train, y_val), baseline=np.inf, # 基线设为最差
        metric='mae', task_type='regression', logger=logger, seed=1,
        cat_features=[], importance_threshold=10.0 # 阈值
    )

    mock_lgb.LGBMRegressor.assert_called_once()
    mock_model_inst.fit.assert_called_once()
    # 检查 fit 的 eval_metric 参数是否正确
    assert mock_model_inst.fit.call_args.kwargs.get('eval_metric') == 'mae'

    assert selected == ['feat1'] # 只选中 feat1
    assert np.isclose(score, 2.5) # 返回 best_score

@pytest.mark.skipif(not LGBM_INSTALLED, reason="LightGBM not installed")
@patch('espml.autofe.algorithm.lgb')
def test_model_features_select_classification_f1(mock_lgb, sample_dataframe_clf: pd.DataFrame):
    """测试分类任务的模型选择 (metric=f1)"""
    X = sample_dataframe_clf[['A', 'B', 'cat1']]
    y = sample_dataframe_clf['target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

    mock_model_inst = MagicMock()
    mock_model_inst.feature_importances_ = np.array([5, 20, 15]) # B 和 cat1 重要
    # 模拟 predict (用于计算 F1)
    # 简单预测全为 1
    mock_model_inst.predict.return_value = np.ones(len(y_val), dtype=int)
    mock_lgb.LGBMClassifier.return_value = mock_model_inst

    selected, score = model_features_select(
        fes=(X_train, X_val, y_train, y_val), baseline=-np.inf, # F1 越大越好
        metric='f1', task_type='classification', logger=logger, seed=42,
        cat_features=['cat1'], importance_threshold=10.0
    )

    mock_lgb.LGBMClassifier.assert_called_once()
    mock_model_inst.fit.assert_called_once()
    fit_kwargs = mock_model_inst.fit.call_args.kwargs
    assert fit_kwargs['eval_metric'] == 'f1' # 检查 eval_metric
    assert 'categorical_feature' in fit_kwargs and fit_kwargs['categorical_feature'] == ['cat1']

    assert sorted(selected) == sorted(['B', 'cat1']) # 选中 B 和 cat1
    # 验证分数是重新计算得到的 F1 分数
    expected_f1 = f1_score(y_val, np.ones(len(y_val)), average='weighted')
    assert np.isclose(score, expected_f1)

@pytest.mark.skipif(not LGBM_INSTALLED, reason="LightGBM not installed")
@patch('espml.autofe.algorithm.lgb')
def test_model_features_select_training_fails(mock_lgb, sample_dataframe_reg: pd.DataFrame):
    """测试模型训练失败的情况"""
    X = sample_dataframe_reg[['feat1', 'feat2']]
    y = sample_dataframe_reg['target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)

    mock_model_inst = MagicMock()
    # 模拟 fit 抛出异常
    mock_model_inst.fit.side_effect = ValueError("Simulated training error")
    mock_lgb.LGBMRegressor.return_value = mock_model_inst

    baseline_score = 5.0
    selected, score = model_features_select(
        fes=(X_train, X_val, y_train, y_val), baseline=baseline_score,
        metric='rmse', task_type='regression', logger=logger, seed=1, cat_features=[]
    )

    # 断言返回特征和基线分数
    assert selected == list(X_train.columns)
    assert np.isclose(score, baseline_score)

# --- 结束 tests/autofe/test_algorithm.py ---