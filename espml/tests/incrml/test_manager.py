# tests/incrml/test_manager.py
# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring, redefined-outer-name, protected-access, unused-argument
"""
espml.incrml.manager 模块的单元测试
验证 IncrmlManager 类的初始化、触发器检查、数据准备和更新流程
需要 mock MLPipeline, BaseSampler, BaseDriftDetector, IncrmlMetadata
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from unittest.mock import patch, MagicMock, call
import datetime

from espml.espml.util.wind_incrml import DEFAULT_TIMEZONE

# 导入被测类和依赖
try:
    from espml.incrml.manager import IncrmlManager
    from espml.incrml.metadata import IncrmlMetadata, ModelVersionInfo # 需要 mock
    from espml.incrml.data_sampling import BaseSampler, WindowSampler, ExemplarSampler # 需要 mock
    from espml.incrml.detect_drift import BaseDriftDetector, DriftDetectorDDM # 需要 mock
    from espml.ml import MLPipeline # 需要 mock
    from espml.util import const
    from loguru import logger
    MANAGER_MODULE_LOADED = True
except ImportError as e:
    MANAGER_MODULE_LOADED = False
    pytest.skip(f"跳过 manager 测试，因为导入失败: {e}", allow_module_level=True)

pytestmark = pytest.mark.skipif(not MANAGER_MODULE_LOADED, reason="espml.incrml.manager 或其依赖项无法导入")


# --- Fixtures ---
@pytest.fixture
def sample_incrml_config() -> Dict[str, Any]:
    """提供 IncrML 配置部分"""
    return {
        "Enabled": True, "Method": "icarl", "Trigger": "OnDataFileIncrease",
        "SaveModelPath": "mock/save/path/MyTask",
        "DataSampling": {"MaxExemplarSetSize": 10, "ExemplarSelectionStrategy": "random"},
        "DriftDetection": {"Enabled": True, "Method": "DDM"} # 启用漂移检测以测试初始化
    }

@pytest.fixture
def full_config_for_manager(sample_incrml_config: Dict[str, Any]) -> Dict[str, Any]:
    """提供包含 IncrML 的完整配置"""
    return {"Feature": {"TargetName": "target", "TimeFrequency": "1H"}, "IncrML": sample_incrml_config}

# 使用 mocker fixture 创建 mock 对象
@pytest.fixture
def mock_metadata(mocker) -> MagicMock:
    """模拟 IncrmlMetadata"""
    mock = mocker.MagicMock(spec=IncrmlMetadata)
    mock.get_current_version.return_value = None
    mock.save.return_value = True
    mock.add_version = MagicMock()
    mock.set_current_version = MagicMock(return_value=True)
    return mock

@pytest.fixture
def mock_sampler(mocker) -> MagicMock:
    """模拟 BaseSampler"""
    mock = mocker.MagicMock(spec=BaseSampler)
    mock.select_data.side_effect = lambda available_data, **kwargs: available_data.copy() # 默认返回全部
    mock.update_state.return_value = None
    return mock

@pytest.fixture
def mock_drift_detector(mocker) -> MagicMock:
    """模拟 BaseDriftDetector"""
    mock = mocker.MagicMock(spec=BaseDriftDetector)
    mock.detected_change.return_value = False
    mock._reset = MagicMock()
    mock.add_element = MagicMock() # 模拟 add_element
    return mock

@pytest.fixture
def mock_ml_pipeline(mocker) -> MagicMock:
    """模拟 MLPipeline"""
    mock = mocker.MagicMock(spec=MLPipeline)
    mock.train.return_value = True # 默认成功
    # 模拟路径获取方法
    def get_paths(run_id):
        run_dir = Path(f"mock_save/path/MyTask/{run_id}")
        return (str(run_dir / f"model_{run_id}.joblib"), str(run_dir / f"tf_state_{run_id}.joblib"),
                str(run_dir / f"features_{run_id}.json"), str(run_dir / "automl" / "logs"))
    mock._get_run_specific_paths = MagicMock(side_effect=get_paths)
    # 模拟性能和 AutoML 包装器
    mock.last_run_performance = {'rmse': 0.1}
    mock.automl_wrapper = MagicMock(metric='rmse', final_val_score=0.1)
    return mock

# --- 测试 IncrmlManager 初始化 ---
# 使用 patch 上下文管理器 mock 工厂函数和类
@patch('espml.incrml.manager.IncrmlMetadata')
@patch('espml.incrml.manager.get_sampler')
@patch('espml.incrml.manager.get_drift_detector')
def test_manager_init_all_components(mock_get_detector, mock_get_sampler, MockMeta,
                                     full_config_for_manager: Dict[str, Any],
                                     mock_metadata, mock_sampler, mock_drift_detector): # 传入 mock 实例用于返回
    """测试初始化是否正确调用依赖项"""
    config = full_config_for_manager.copy()
    config['IncrML']['Trigger'] = 'OnDriftDetected' # 强制测试漂移检测器初始化

    MockMeta.return_value = mock_metadata # 工厂函数返回 mock 实例
    mock_get_sampler.return_value = mock_sampler
    mock_get_detector.return_value = mock_drift_detector

    manager = IncrmlManager(task_id="TestInit", config=config, logger_instance=logger)

    MockMeta.assert_called_once_with(task_id="TestInit", metadata_dir=Path(config['IncrML']['SaveModelPath']) / "metadata")
    mock_get_sampler.assert_called_once_with(config['IncrML'], manager.logger) # 检查 logger 是否传递
    mock_get_detector.assert_called_once_with(config, manager.logger) # 传递了完整 config
    assert manager.metadata == mock_metadata
    assert manager.sampler == mock_sampler
    assert manager.drift_detector == mock_drift_detector

# --- 测试 check_trigger ---
# (需要 mock metadata 和 detector 的返回值)
def test_manager_check_trigger_data_increase(mock_metadata, full_config_for_manager):
    config = full_config_for_manager.copy()
    config['IncrML']['Trigger'] = 'OnDataFileIncrease'
    with patch('espml.incrml.manager.get_sampler'), patch('espml.incrml.manager.get_drift_detector'):
         manager = IncrmlManager("TaskTrigger", config, logger)
         manager.metadata = mock_metadata # 注入 mock

    # 无历史
    mock_metadata.get_current_version.return_value = None
    assert manager.check_trigger(latest_data_timestamp=pd.Timestamp.now(tz=DEFAULT_TIMEZONE)) is True
    # 有历史，新数据更新
    prev_meta = MagicMock(spec=ModelVersionInfo, training_data_end="2024-01-10T12:00:00Z")
    mock_metadata.get_current_version.return_value = prev_meta
    assert manager.check_trigger(latest_data_timestamp=pd.Timestamp("2024-01-10T13:00:00Z")) is True
    # 有历史，新数据未更新
    assert manager.check_trigger(latest_data_timestamp=pd.Timestamp("2024-01-10T11:00:00Z")) is False
    # 未提供时间戳
    assert manager.check_trigger(latest_data_timestamp=None) is False

def test_manager_check_trigger_drift(mock_metadata, mock_drift_detector, full_config_for_manager):
    config = full_config_for_manager.copy()
    config['IncrML']['Trigger'] = 'OnDriftDetected'; config['IncrML']['DriftDetection']['Enabled'] = True
    with patch('espml.incrml.manager.get_sampler'), \
         patch('espml.incrml.manager.get_drift_detector', return_value=mock_drift_detector), \
         patch('espml.incrml.manager.IncrmlMetadata', return_value=mock_metadata):
        manager = IncrmlManager("TaskDrift", config, logger)
        manager.drift_detector = mock_drift_detector # 注入 mock

    preds = pd.Series([0, 1]); truth = pd.Series([0, 0])
    # 漂移未检测到
    mock_drift_detector.detected_change.return_value = False
    assert manager.check_trigger(current_predictions=preds, current_ground_truth=truth) is False
    assert mock_drift_detector.add_element.call_count == 2
    # 漂移检测到
    mock_drift_detector.detected_change.return_value = True
    assert manager.check_trigger(current_predictions=preds, current_ground_truth=truth) is True


# --- 测试 prepare_data ---
def test_manager_prepare_data(mock_metadata, mock_sampler, full_config_for_manager):
    """测试 prepare_data 是否正确调用 sampler"""
    with patch('espml.incrml.manager.get_sampler', return_value=mock_sampler), \
         patch('espml.incrml.manager.get_drift_detector'), \
         patch('espml.incrml.manager.IncrmlMetadata', return_value=mock_metadata):
        manager = IncrmlManager("TaskPrep", full_config_for_manager, logger)

    available_df = pd.DataFrame({'A': range(10)})
    # 配置 mock sampler 返回值
    sampled_df = available_df.tail(3)
    mock_sampler.select_data.return_value = sampled_df
    # 配置 mock metadata 返回值
    prev_meta_obj = MagicMock(spec=ModelVersionInfo); prev_meta_dict = {"id": "prev"}
    prev_meta_obj.to_dict.return_value = prev_meta_dict
    mock_metadata.get_current_version.return_value = prev_meta_obj

    result = manager.prepare_data(available_df)

    mock_sampler.select_data.assert_called_once_with(
        available_data=available_df, previous_model_metadata=prev_meta_dict
    )
    assert_frame_equal(result, sampled_df)


# --- 测试 update ---
def test_manager_update_workflow_success(mock_metadata, mock_sampler, mock_drift_detector,
                                         mock_ml_pipeline, full_config_for_manager):
    """测试成功的增量更新流程"""
    # 初始化 manager，确保它使用 mock 组件
    with patch('espml.incrml.manager.IncrmlMetadata', return_value=mock_metadata), \
         patch('espml.incrml.manager.get_sampler', return_value=mock_sampler), \
         patch('espml.incrml.manager.get_drift_detector', return_value=mock_drift_detector):
        manager = IncrmlManager("TaskUpdate", full_config_for_manager, logger)

    # 准备数据
    index = pd.date_range('2024-01-01', periods=5, freq='D', name='datetime')
    incrml_train_df = pd.DataFrame({'A': range(5)}, index=index)
    new_data_processed = incrml_train_df.tail(2)

    # 配置 mock MLPipeline 行为
    mock_ml_pipeline.train.return_value = True # 训练成功
    mock_ml_pipeline.last_run_performance = {'rmse': 0.09} # 训练后的性能
    mock_ml_pipeline.automl_wrapper = MagicMock(metric='rmse', final_val_score=0.09)
    # 假设 train 方法会设置这些属性，或者通过 _get_run_specific_paths 获取
    run_id_generated = "20240110100000000000" # 假设生成的 ID
    mock_ml_pipeline.last_run_id = run_id_generated
    model_p, tf_p, feat_p, _ = manager.ml_pipeline._get_run_specific_paths(run_id_generated) # 使用 manager 的 pipeline 实例
    mock_ml_pipeline.last_run_model_path = model_p
    mock_ml_pipeline.last_run_transformer_path = tf_p
    mock_ml_pipeline.last_run_features_path = feat_p

    # 配置 mock sampler.update_state 返回值
    sampler_update_dict = {"exemplar_set_path": "/new/exemplars.feather"}
    mock_sampler.update_state.return_value = sampler_update_dict

    # 配置 mock metadata.get_current_version 返回值
    prev_meta = MagicMock(spec=ModelVersionInfo); prev_meta.version_id = "v_old"
    mock_metadata.get_current_version.return_value = prev_meta

    # 配置 mock drift_detector 状态
    mock_drift_detector.detected_change.return_value = True # 模拟检测到漂移

    # 执行 update
    update_ok = manager.update(
        ml_pipeline=mock_ml_pipeline, # 传入 mock pipeline
        incrml_train_df=incrml_train_df,
        new_data_processed=new_data_processed
    )

    # --- 验证 ---
    assert update_ok is True
    # 验证 train 调用
    mock_ml_pipeline.train.assert_called_once()
    call_kwargs_train = mock_ml_pipeline.train.call_args.kwargs
    pd.testing.assert_frame_equal(call_kwargs_train['df_train_full'], incrml_train_df)
    assert call_kwargs_train['run_id'] is not None # 检查 run_id 是否被传递

    # 验证 sampler 更新调用
    mock_sampler.update_state.assert_called_once()
    call_args_sampler = mock_sampler.update_state.call_args.args
    pd.testing.assert_frame_equal(call_args_sampler[0], new_data_processed)
    pd.testing.assert_frame_equal(call_args_sampler[1], incrml_train_df)
    assert call_args_sampler[2]['version_id'] == mock_ml_pipeline.last_run_id

    # 验证 metadata 添加和保存调用
    mock_metadata.add_version.assert_called_once()
    added_version_info = mock_metadata.add_version.call_args[0][0]
    assert isinstance(added_version_info, ModelVersionInfo)
    assert added_version_info.version_id == mock_ml_pipeline.last_run_id
    assert added_version_info.performance_metrics['rmse'] == 0.09
    assert added_version_info.base_model_version == "v_old"
    assert added_version_info.drift_status is True # 验证漂移状态被记录
    assert added_version_info.exemplar_set_path == "/new/exemplars.feather" # 验证 sampler 状态路径被记录
    mock_metadata.save.assert_called_once()

    # 验证 drift detector 重置被调用
    mock_drift_detector._reset.assert_called_once()


def test_manager_update_train_fail(mock_metadata, mock_sampler, mock_drift_detector,
                                   mock_ml_pipeline, full_config_for_manager):
    """测试 update 中 MLPipeline.train 失败的情况"""
    with patch('espml.incrml.manager.IncrmlMetadata', return_value=mock_metadata), \
         patch('espml.incrml.manager.get_sampler', return_value=mock_sampler), \
         patch('espml.incrml.manager.get_drift_detector', return_value=mock_drift_detector):
        manager = IncrmlManager("TaskFail", full_config_for_manager, logger)

    incrml_train_df = pd.DataFrame({'A': [1]})
    new_data = pd.DataFrame({'A': [1]})
    mock_ml_pipeline.train.return_value = False # 模拟训练失败

    update_ok = manager.update(mock_ml_pipeline, incrml_train_df, new_data)

    assert update_ok is False # 更新失败
    mock_ml_pipeline.train.assert_called_once() # 尝试了训练
    mock_sampler.update_state.assert_not_called() # 不应更新 sampler
    mock_metadata.add_version.assert_not_called() # 不应添加版本
    mock_metadata.save.assert_not_called() # 不应保存元数据
    mock_drift_detector._reset.assert_not_called() # 训练失败，不应重置检测器


# --- 结束 tests/incrml/test_manager.py --- 