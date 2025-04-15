# tests/autofe/test_autofe.py
# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring, redefined-outer-name, protected-access, unused-argument
"""
espml.autofe.autofe 模块 (AutoFE 引擎类) 的单元测试
验证 AutoFE 引擎的初始化、运行流程和特征选择策略
需要 mock 其依赖项 (Transform 和 algorithm 中的函数)
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call, ANY # 导入 ANY 用于匹配部分参数
from typing import Dict, Any, List, Tuple

# 导入被测类和依赖
from espml.autofe.autofe import AutoFE
from espml.autofe.transform import Transform # 需要模拟
# 导入 algorithm 模块以 mock 其函数
from espml.autofe import algorithm
from loguru import logger

# --- Fixtures ---
@pytest.fixture
def sample_config_autofe() -> Dict[str, Any]:
    """提供用于 AutoFE 引擎测试的配置"""
    return {
        "Feature": {
            "TaskType": "regression", "TargetName": "target", "Metric": "rmse",
            "RandomSeed": 123, "CategoricalFeature": ["cat1"], "TestSize": 0.2,
            "TimeWindow": "1,2" # 对应 time_span=[1, 2]
        },
        "AutoFE": {
            "Running": True, "Method": "DFS", "DFSLayers": 2, # n=2
            "RandomRatio": 0.5, "FeatImpThreshold": 0.01, "SaveFeatures": "OnlyGreater",
            "Operators": ["add", "log"],
            "Transforms": [{"name": "Lag", "params": {"periods": 1}}],
            "MaxFeaturesPerLayer": 10
        },
        "Resource": {"MaxWorkers": 1} # 方便 mock，使用串行
    }

@pytest.fixture
def sample_data_autofe() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """提供 AutoFE fit 使用的拆分后数据"""
    np.random.seed(99)
    X_train = pd.DataFrame({'A': np.random.rand(10), 'B': np.random.rand(10)*5, 'cat1': ['x', 'y']*5})
    y_train = pd.Series(np.random.rand(10)*20, name='target')
    X_val = pd.DataFrame({'A': np.random.rand(5), 'B': np.random.rand(5)*5, 'cat1': ['x', 'y', 'x', 'y', 'x']})
    y_val = pd.Series(np.random.rand(5)*20, name='target')
    return X_train, y_train, X_val, y_val

# --- 测试 AutoFE 初始化 ---
# (已在上次交互中提供并通过，此处省略)

# --- 测试 AutoFE fit 方法 (核心流程，需要 mock 依赖) ---
@patch('espml.autofe.autofe.Transform') # Mock Transform 类
@patch('espml.autofe.autofe.feature_space') # Mock 工具函数
@patch('espml.autofe.autofe.threads_feature_select') # Mock 算法函数
@patch('espml.autofe.autofe.max_threads_name2feature') # Mock 算法函数
@patch('espml.autofe.autofe.model_features_select') # Mock 算法函数
def test_autofe_fit_workflow_strict(
    mock_model_select, mock_max_threads, mock_threads_select, mock_feature_space, MockTransform,
    sample_config_autofe: Dict[str, Any],
    sample_data_autofe: Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
):
    """严格测试 AutoFE fit 方法的工作流和函数调用结合"""
    X_train, y_train, X_val, y_val = sample_data_autofe
    mock_transformer_inst = MockTransform.return_value # 获取模拟实例

    # --- 配置 Mock 返回值 (模拟一个成功的两轮迭代) ---
    # 1. feature_space (autofe.utils.feature_space)
    mock_feature_space.side_effect = [
        ["f_cand1", "f_cand2", "f_cand3"], # 第 1 轮候选
        ["f_cand4", "f_cand5"]            # 第 2 轮候选
    ]
    # 2. threads_feature_select (algorithm.threads_feature_select)
    #    返回 (selected_names, selected_df, scores_dict)
    df_f1 = pd.DataFrame({'f_cand1': np.random.rand(len(X_train))}, index=X_train.index)
    df_f4 = pd.DataFrame({'f_cand4': np.random.rand(len(X_train)) + 1}, index=X_train.index)
    mock_threads_select.side_effect = [
        # 第 1 轮: Gini 筛选出 f_cand1
        (["f_cand1"], df_f1, {"f_cand1": 0.8, "f_cand2": 0.0, "f_cand3": 0.1}),
        # 第 2 轮: Gini 筛选出 f_cand4
        (["f_cand4"], df_f4, {"f_cand4": 0.7, "f_cand5": 0.05}),
    ]
    # 3. max_threads_name2feature (algorithm.max_threads_name2feature)
    #    接收 df, feature_names, transformer, logger, n_jobs
    #    返回添加了列的 df
    def mock_add_features_side_effect(df, feature_names, transformer, logger, n_jobs):
        df_res = df.copy()
        # print(f"DEBUG: mock_max_threads called with features: {feature_names}")
        for name in feature_names:
            if name not in df_res.columns:
                 np.random.seed(abs(hash(name)) % (2**32 - 1))
                 df_res[name] = pd.Series(np.random.rand(len(df)) + hash(name) % 5, index=df.index)
        return df_res
    mock_max_threads.side_effect = mock_add_features_side_effect
    # 4. model_features_select (algorithm.model_features_select)
    #    返回 (selected_feature_names, score)
    mock_model_select.side_effect = [
        # 第 1 轮: 输入 X_train + f_cand1, 返回 + f_cand1, 分数改善
        (['A', 'B', 'cat1', 'f_cand1'], 0.4),
        # 第 2 轮: 输入 X_train + f_cand1 + f_cand4, 返回 + f_cand4, 分数改善
        (['A', 'B', 'cat1', 'f_cand4'], 0.3),
        # 最终选择: 输入 X_train + f_cand1 + f_cand4, 返回 + f_cand4
        (['A', 'B', 'cat1', 'f_cand4'], 0.3),
    ]

    # --- 执行 fit ---
    engine = AutoFE(logger_instance=logger, **sample_config_autofe)
    engine.base_score = 0.5 # 设置初始基线

    final_X_train, final_X_val, final_y_train, final_y_val, final_selected_new_features = engine.fit(
        X_train, y_train, X_val, y_val
    )

    # --- 验证调用和结果 ---
    # 1. 验证 feature_space 调用
    assert mock_feature_space.call_count == 2
    # 第一次调用的 df 参数应该是初始 df_train
    pd.testing.assert_frame_equal(mock_feature_space.call_args_list[0].args[0], pd.concat([X_train, y_train], axis=1))
    # 第二次调用的 df 应该是包含第一轮选中特征的 df
    expected_cols_iter2_input = ['A', 'B', 'cat1', 'target'] + ['f_cand1'] # 模型选中的是 A,B,cat1,f1; 传给下一轮的是 A,B,cat1,f1,target? 看代码实现
    # fit 中准备下一轮的 df_train_iter = df_train[[cols_to_keep_next if col in df_train.columns] + [self.target_name]]
    # cols_to_keep_next = original_train_cols + features_to_add_to_adv (本轮模型选中的前20个新特征)
    # 所以第二次 feature_space 输入的 df 确实是 df_train_iter (包含 A,B,cat1,target,f_cand1)
    assert sorted(list(mock_feature_space.call_args_list[1].args[0].columns)) == sorted(expected_cols_iter2_input)


    # 2. 验证 threads_feature_select 调用
    assert mock_threads_select.call_count == 2
    # 第一次调用的 candidate_feature 应是 ['f1', 'f2', 'f3']
    assert mock_threads_select.call_args_list[0].kwargs['candidate_feature'] == ["f_cand1", "f_cand2", "f_cand3"]
    # 第二次调用的 candidate_feature 应是 ['f4', 'f5']
    assert mock_threads_select.call_args_list[1].kwargs['candidate_feature'] == ["f_cand4", "f_cand5"]
    # 验证 transformer 实例被正确传递
    assert mock_threads_select.call_args_list[0].kwargs['transformer'] == mock_transformer_inst
    assert mock_threads_select.call_args_list[1].kwargs['transformer'] == mock_transformer_inst

    # 3. 验证 max_threads_name2feature 调用次数和参数
    # iter1: calc f_cand1 for train & val => 2 calls
    # iter2: calc f_cand4 for train & val => 2 calls
    # final select: calc f_cand1, f_cand4 for train & val => 2 calls
    # final output: calc f_cand4 for train & val => 2 calls
    assert mock_max_threads.call_count == 8
    # 检查部分调用参数
    assert mock_max_threads.call_args_list[0].args[1] == ['f_cand1'] # 第一次为 train 计算 f1
    assert mock_max_threads.call_args_list[1].args[1] == ['f_cand1'] # 第一次为 val 计算 f1
    assert mock_max_threads.call_args_list[4].args[1] == ['f_cand1', 'f_cand4'] # 最终选择前为 train 计算 adv
    assert mock_max_threads.call_args_list[6].args[1] == ['f_cand4'] # 最终输出为 train 计算 final


    # 4. 验证 model_features_select 调用次数和参数
    assert mock_model_select.call_count == 3
    # 第一次调用的 baseline 是 0.5
    assert np.isclose(mock_model_select.call_args_list[0].kwargs['baseline'], 0.5)
    # 第二次调用的 baseline 是第一次返回的 0.4
    assert np.isclose(mock_model_select.call_args_list[1].kwargs['baseline'], 0.4)
    # 第三次（最终）调用的 baseline 是初始的 0.5
    assert np.isclose(mock_model_select.call_args_list[2].kwargs['baseline'], 0.5)

    # 5. 验证最终返回结果
    assert final_selected_new_features == ['f_cand4'] # 最终模型只选择了 f_cand4 作为新特征
    expected_final_cols = ['A', 'B', 'cat1', 'f_cand4']
    assert sorted(list(final_X_train.columns)) == sorted(expected_final_cols)
    assert sorted(list(final_X_val.columns)) == sorted(expected_final_cols)
    assert 'f_cand1' not in final_X_train.columns

# --- 结束 tests/autofe/test_autofe.py ---