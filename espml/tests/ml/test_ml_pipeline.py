# tests/ml/test_ml_pipeline.py
# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring, redefined-outer-name, protected-access, unused-argument, too-many-arguments
"""
espml.ml 模块 (MLPipeline 类) 的单元测试
验证核心 ML 流程的编排、组件调用和状态管理
需要 mock DataProcessor, AutoFE, FlamlAutomlWrapper 等
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from unittest.mock import patch, MagicMock, call, ANY
import datetime

# 导入被测类和依赖
try:
    from espml.ml import MLPipeline
    from espml.dataprocess.data_processor import DataProcessor # Mock Target
    from espml.autofe.autofe import AutoFE # Mock Target
    from espml.autofe.transform import Transform # Mock Target (for loading state)
    from espml.automl.automl import FlamlAutomlWrapper # Mock Target
    from espml.util import const, utils as common_utils
    from loguru import logger
    ML_MODULE_LOADED = True
except ImportError as e:
    ML_MODULE_LOADED = False
    pytest.skip(f"跳过 ml 测试，因为无法导入 espml 模块: {e}", allow_module_level=True)

pytestmark = pytest.mark.skipif(not ML_MODULE_LOADED, reason="espml.ml 或其依赖项无法导入")

# --- Fixtures ---
@pytest.fixture
def sample_ml_config(tmp_path: Path) -> Dict[str, Any]:
    """提供一个用于 MLPipeline 测试的完整配置"""
    task_id = "MLTestTask"
    model_base_path = tmp_path / "ml_pipeline_models" / task_id
    return {
        "TaskName": task_id, # 添加 TaskName
        "Feature": {"TargetName": "target", "RandomSeed": 42, "TestSize": 0.25, "TimeFrequency": "1D", "CapacityKW": 100.0},
        "AutoFE": {"Running": True, "SaveFeatures": "OnlyGreater", "Method": "DFS"}, # 启用 AutoFE
        "AutoML": {"Running": True, "Method": "flaml", "TimeBudget": 10},
        "IncrML": {"Enabled": False, "SaveModelPath": str(model_base_path)}, # IncrML 未启用，但提供路径
        "DataSource": {"dir": "dummy_data"}, # DP 需要
        "Cleaning": {}, # DP 需要
        "Resource": {},
        "Project": {}
    }

@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """提供输入 DataFrame"""
    return pd.DataFrame({'A': range(20), 'B': range(20, 40), 'target': range(0, 40, 2)},
                        index=pd.date_range('2024-01-01', periods=20, freq='D'))

@pytest.fixture
def mock_data_processor(mocker, sample_raw_df: pd.DataFrame) -> MagicMock:
    """模拟 DataProcessor"""
    mock_dp_instance = MagicMock(spec=DataProcessor)
    # 模拟 process 方法返回处理后的数据
    processed_df = sample_raw_df.copy()
    processed_df['A_proc'] = processed_df['A'] * 1.1
    processed_df['target'] = processed_df['target'].rename("label") # 模拟目标重命名
    mock_dp_instance.process.return_value = processed_df
    # 模拟输出列名
    mock_dp_instance.output_feature_names = ['A', 'B', 'A_proc', 'label']
    # Patch DataProcessor 类返回这个 mock 实例
    mocker.patch('espml.ml.DataProcessor', return_value=mock_dp_instance)
    return mock_dp_instance

@pytest.fixture
def mock_autofe_engine(mocker, sample_raw_df: pd.DataFrame) -> MagicMock:
    """模拟 AutoFE 引擎"""
    mock_autofe_instance = MagicMock(spec=AutoFE)
    # 模拟 fit 方法
    def fit_side_effect(X_train, y_train, X_val, y_val):
        X_train_enhanced = X_train.copy(); X_train_enhanced['autofe_feat1'] = 1
        X_val_enhanced = X_val.copy(); X_val_enhanced['autofe_feat1'] = 1
        selected_features = ['autofe_feat1']
        # 模拟 transformer 状态
        mock_autofe_instance.transformer = MagicMock(spec=Transform)
        mock_autofe_instance.transformer._record = {"state": "fitted"}
        return X_train_enhanced, X_val_enhanced, y_train, y_val, selected_features
    mock_autofe_instance.fit.side_effect = fit_side_effect
    # Patch AutoFE 类返回这个 mock 实例
    mocker.patch('espml.ml.AutoFE', return_value=mock_autofe_instance)
    return mock_autofe_instance

@pytest.fixture
def mock_automl_wrapper(mocker) -> MagicMock:
    """模拟 FlamlAutomlWrapper"""
    mock_automl_inst = MagicMock(spec=FlamlAutomlWrapper)
    mock_automl_inst.fit = MagicMock() # fit 无返回值
    mock_automl_inst.save_model.return_value = True # 模拟保存成功
    mock_automl_inst.predict.return_value = np.array([1]*5) # 模拟预测返回固定值
    mock_automl_inst.metric = 'rmse' # 设置属性
    mock_automl_inst.final_val_score = 0.1 # 设置属性
    # Patch FlamlAutomlWrapper 类返回这个 mock 实例
    mocker.patch('espml.ml.FlamlAutomlWrapper', return_value=mock_automl_inst)
    # 同时 Patch load_model 类方法
    mocker.patch.object(FlamlAutomlWrapper, 'load_model', return_value=mock_automl_inst)
    return mock_automl_inst


# --- 测试 MLPipeline 初始化 ---
def test_ml_pipeline_init(sample_ml_config: Dict[str, Any]):
    """测试 MLPipeline 初始化"""
    pipeline = MLPipeline(config=sample_ml_config)
    assert pipeline.target_name == "target"
    assert pipeline.random_seed == 42
    assert pipeline.base_model_save_path == Path(sample_ml_config['IncrML']['SaveModelPath'])

# --- 测试 MLPipeline.train ---
@patch('espml.ml.train_test_split') # Mock 数据拆分
@patch('espml.ml.common_utils.dump_pickle') # Mock 保存 transformer 状态
@patch('espml.ml.common_utils.write_json_file') # Mock 保存特征列表
def test_ml_pipeline_train_workflow(mock_write_json, mock_dump_pickle, mock_split,
                                      sample_ml_config: Dict[str, Any],
                                      sample_raw_df: pd.DataFrame,
                                      mock_data_processor: MagicMock,
                                      mock_autofe_engine: MagicMock,
                                      mock_automl_wrapper: MagicMock):
    """测试完整的 train 方法工作流"""
    pipeline = MLPipeline(config=sample_ml_config)
    run_id = "test_run_123"

    # 配置 mock_split 返回值
    X = sample_raw_df.drop(columns=['target']); y = sample_raw_df['target']
    # 假设 DataProcessor 返回的列
    X_processed = pd.DataFrame({'A': X['A'], 'B': X['B'], 'A_proc': X['A']*1.1})
    y_processed = y.rename("label")
    mock_data_processor.process.return_value = pd.concat([X_processed, y_processed], axis=1)
    mock_data_processor.output_feature_names = list(X_processed.columns) + ['label'] # 模拟 DP 返回的列

    # 模拟 train_test_split
    X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(X_processed, y_processed, test_size=0.25, random_state=42)
    mock_split.return_value = (X_train_s, X_val_s, y_train_s, y_val_s)

    # 执行 train
    success = pipeline.train(df_train_full=sample_raw_df, run_id=run_id)

    # --- 验证 ---
    assert success is True
    # 1. 验证 DataProcessor 调用
    mock_data_processor.process.assert_called_once_with(sample_raw_df)
    # 2. 验证 train_test_split 调用
    mock_split.assert_called_once()
    # 3. 验证 AutoFE 调用
    mock_autofe_engine.fit.assert_called_once()
    call_args_autofe, _ = mock_autofe_engine.fit.call_args
    pd.testing.assert_frame_equal(call_args_autofe[0], X_train_s) # 传入拆分后的 X_train
    pd.testing.assert_series_equal(call_args_autofe[1], y_train_s) # 传入拆分后的 y_train
    # 4. 验证 AutoML 调用
    mock_automl_wrapper.fit.assert_called_once()
    call_kwargs_automl = mock_automl_wrapper.fit.call_args.kwargs
    assert 'X_train' in call_kwargs_automl and 'autofe_feat1' in call_kwargs_automl['X_train'].columns # 确认传入增强后的数据
    assert 'X_val' in call_kwargs_automl and 'autofe_feat1' in call_kwargs_automl['X_val'].columns
    assert call_kwargs_automl['log_dir'] is not None # 确认 log_dir 被传递
    # 5. 验证保存调用
    model_path, tf_path, feat_path, _ = pipeline._get_run_specific_paths(run_id)
    mock_automl_wrapper.save_model.assert_called_once_with(model_path)
    mock_dump_pickle.assert_called_once_with({"state": "fitted"}, tf_path) # 验证保存的状态
    mock_write_json.assert_called_once_with({"selected_features": ['autofe_feat1']}, feat_path) # 验证保存的特征列表
    # 6. 验证内部状态更新
    assert pipeline.last_run_id == run_id
    assert pipeline.last_run_performance == {'rmse': 0.1}
    assert pipeline.last_run_selected_autofe_features == ['autofe_feat1']

@patch('espml.ml.DataProcessor') # 只 Mock DP
def test_ml_pipeline_train_data_processing_fails(MockDP, sample_ml_config, sample_raw_df):
    """测试数据处理失败时 train 的行为"""
    mock_dp_instance = MagicMock()
    mock_dp_instance.process.side_effect = DataProcessingError("Simulated DP Error") # 模拟处理失败
    MockDP.return_value = mock_dp_instance

    pipeline = MLPipeline(config=sample_ml_config)
    success = pipeline.train(df_train_full=sample_raw_df, run_id="fail_run")
    assert success is False # 训练应失败

# --- 测试 MLPipeline.predict ---
@patch('espml.ml.common_utils.read_json_file') # Mock 加载特征列表
@patch('espml.ml.common_utils.load_pickle') # Mock 加载 transformer 状态
@patch('espml.ml.FlamlAutomlWrapper.load_model') # Mock 加载模型
@patch('espml.ml.Transform') # Mock Transform 类
@patch('espml.ml.DataProcessor') # Mock DataProcessor
def test_ml_pipeline_predict_workflow(MockDP, MockTransform, mock_load_model, mock_load_pickle, mock_read_json,
                                      sample_ml_config: Dict[str, Any],
                                      mock_automl_wrapper: MagicMock, # 用于 load_model 返回
                                      mock_transformer: MagicMock): # 用于 Transform 实例化
    """测试 predict 方法的完整流程"""
    pipeline = MLPipeline(config=sample_ml_config)
    run_id_to_load = "predict_run_123"
    X_test = pd.DataFrame({'A': [6], 'B': [11], 'cat1': ['y']})

    # --- 配置 Mock 返回值 ---
    # 1. 加载特征列表
    mock_read_json.return_value = {"selected_features": ["autofe_feat1"]}
    # 2. 加载 Transformer 状态
    mock_transformer_state = {"state": "loaded"}
    mock_load_pickle.return_value = mock_transformer_state
    # 3. 加载 AutoML 模型
    mock_load_model.return_value = mock_automl_wrapper # 返回 mock wrapper
    # 4. DataProcessor.process
    mock_dp_instance = MockDP.return_value
    X_test_processed = pd.DataFrame({'A': [6.1], 'B': [11.1], 'cat1': ['y']}, index=X_test.index) # 模拟处理结果
    mock_dp_instance.process.return_value = X_test_processed
    # 5. Transform 实例化和 transform 方法
    mock_transform_instance = mock_transformer # 使用已有的 mock transformer
    MockTransform.return_value = mock_transform_instance
    # 模拟 transform 添加新特征
    X_test_final = X_test_processed.copy()
    X_test_final['autofe_feat1'] = 1.5
    mock_transform_instance.transform.return_value = X_test_final
    # 6. AutoML predict
    expected_predictions = np.array([15.5])
    mock_automl_wrapper.predict.return_value = expected_predictions

    # --- 执行 predict ---
    predictions = pipeline.predict(X_test=X_test, run_id=run_id_to_load)

    # --- 验证 ---
    # 1. 验证加载函数调用
    model_path, tf_path, feat_path, _ = pipeline._get_run_specific_paths(run_id_to_load)
    mock_read_json.assert_called_once_with(feat_path)
    mock_load_pickle.assert_called_once_with(tf_path)
    mock_load_model.assert_called_once_with(model_path, logger_instance=ANY, config=ANY, global_config=ANY)
    # 2. 验证 DataProcessor 调用
    mock_dp_instance.process.assert_called_once_with(X_test) # 确认传入的是 X_test
    # 3. 验证 Transform 实例化和调用
    MockTransform.assert_called_once() # 验证类被实例化
    # 验证状态是否被加载
    assert mock_transform_instance._record == mock_transformer_state
    mock_transform_instance.transform.assert_called_once_with(X_test_processed, ["autofe_feat1"])
    # 4. 验证 AutoML predict 调用
    mock_automl_wrapper.predict.assert_called_once()
    call_args_pred, _ = mock_automl_wrapper.predict.call_args
    pd.testing.assert_frame_equal(call_args_pred[0], X_test_final) # 确认传入最终特征
    # 5. 验证返回结果
    np.testing.assert_array_equal(predictions, expected_predictions)


def test_ml_pipeline_predict_load_fails(mock_read_json): # 只 mock 一个加载函数
    """测试加载 AutoFE 特征列表失败时 predict 的行为"""
    config = {"Feature": {"TargetName": "target"}, "IncrML": {"SaveModelPath": "dummy"}, "AutoML":{}, "AutoFE":{}}
    pipeline = MLPipeline(config=config)
    mock_read_json.return_value = None # 模拟加载失败
    X_test = pd.DataFrame({'A': [1]})
    predictions = pipeline.predict(X_test=X_test, run_id="fail_load")
    assert predictions is None # 预测应失败

# --- 结束 tests/ml/test_ml_pipeline.py ---