# tests/incrml/test_data_sampling.py
# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring, redefined-outer-name, protected-access, unused-argument
"""
espml.incrml.data_sampling 模块的单元测试
验证 WindowSampler, ExemplarSampler 和 get_sampler 的功能
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import patch, MagicMock

# 导入被测模块和类
try:
    from espml.incrml.data_sampling import (
        BaseSampler, WindowSampler, ExemplarSampler, get_sampler
    )
    from espml.util import utils as common_utils # 需要文件操作
    from loguru import logger
    SAMPLING_MODULE_LOADED = True
except ImportError as e:
    SAMPLING_MODULE_LOADED = False
    pytest.skip(f"跳过 data_sampling 测试,因为导入失败: {e}", allow_module_level=True)

# --- Fixtures ---
@pytest.fixture
def sample_available_data() -> pd.DataFrame:
    """提供一个包含时间索引的可用数据 DataFrame"""
    index = pd.date_range('2024-01-01', '2024-01-20', freq='D', name='datetime')
    return pd.DataFrame({'feature': range(len(index)), 'target': range(len(index))}, index=index)

@pytest.fixture
def mock_logger() -> MagicMock:
    """创建一个模拟的 logger 对象"""
    mock = MagicMock()
    mock.bind.return_value = mock
    return mock

@pytest.fixture
def sample_icarl_config() -> Dict[str, Any]:
    """提供 iCaRL 方法的 IncrML 配置"""
    return {"IncrML": {"Method": "icarl", "DataSampling": {"MaxExemplarSetSize": 5, "ExemplarSelectionStrategy": "random"}}}

@pytest.fixture
def sample_window_config() -> Dict[str, Any]:
    """提供 window 方法的 IncrML 配置"""
    return {"IncrML": {"Method": "window", "DataSampling": {"WindowSize": "7D"}}}

@pytest.fixture
def dummy_exemplar_file(tmp_path: Path) -> Path:
    """创建一个临时的、有效的 feather 样本文件"""
    file_path = tmp_path / "dummy_exemplars.feather"
    index = pd.to_datetime(['2023-12-30', '2023-12-31'], name='datetime')
    df = pd.DataFrame({'feature': [-1, -2], 'target': [-1, -2]}, index=index)
    df.reset_index().to_feather(file_path)
    return file_path

# --- 测试 get_sampler (与之前实现一致) ---
def test_get_sampler_selects_correctly(sample_icarl_config, sample_window_config, mock_logger):
    sampler_icarl = get_sampler(sample_icarl_config, mock_logger)
    assert isinstance(sampler_icarl, ExemplarSampler)
    assert sampler_icarl.max_exemplar_set_size == 5

    sampler_window = get_sampler(sample_window_config, mock_logger)
    assert isinstance(sampler_window, WindowSampler)
    assert sampler_window.window_size == "7D"

    sampler_default = get_sampler({"IncrML": {}}, mock_logger)
    assert isinstance(sampler_default, WindowSampler)

# --- 测试 WindowSampler (与之前实现一致) ---
def test_window_sampler_select_data(sample_available_data: pd.DataFrame, mock_logger):
    config = {'WindowSize': '5D'}
    sampler = WindowSampler(config, mock_logger)
    selected = sampler.select_data(sample_available_data)
    assert len(selected) == 5
    assert selected.index.min() == pd.Timestamp('2024-01-16')
    assert selected.index.max() == pd.Timestamp('2024-01-20')

# --- 测试 ExemplarSampler ---
# 测试加载 (需要 mock common_utils.check_path_exists 和 pd.read_feather)
@patch('espml.incrml.data_sampling.common_utils.check_path_exists')
@patch('espml.incrml.data_sampling.pd.read_feather')
def test_exemplar_sampler_load_success(mock_read, mock_exists, dummy_exemplar_file: Path, mock_logger):
    """测试成功加载样本集"""
    mock_exists.return_value = True
    # 模拟 read_feather 返回 DataFrame
    dummy_df = pd.DataFrame({'index': pd.to_datetime(['2023-12-30', '2023-12-31']),
                             'feature': [-1, -2], 'target': [-1, -2]})
    mock_read.return_value = dummy_df

    config = {}
    sampler = ExemplarSampler(config, mock_logger)
    metadata = {'exemplar_set_path': str(dummy_exemplar_file)}
    sampler._load_exemplar_set(metadata)

    mock_exists.assert_called_once_with(dummy_exemplar_file, path_type='f')
    mock_read.assert_called_once_with(dummy_exemplar_file)
    assert sampler.exemplar_set is not None
    assert len(sampler.exemplar_set) == 2
    assert isinstance(sampler.exemplar_set.index, pd.DatetimeIndex) # 验证索引恢复

@patch('espml.incrml.data_sampling.common_utils.check_path_exists', return_value=False)
def test_exemplar_sampler_load_file_not_found(mock_exists, mock_logger, tmp_path):
    """测试加载时文件不存在"""
    sampler = ExemplarSampler({}, mock_logger)
    metadata = {'exemplar_set_path': str(tmp_path / "not_found.feather")}
    sampler._load_exemplar_set(metadata)
    assert sampler.exemplar_set is None

# 测试保存 (需要 mock df.to_feather 和 common_utils.mkdir_if_not_exist)
@patch('espml.incrml.data_sampling.pd.DataFrame.to_feather')
@patch('espml.incrml.data_sampling.common_utils.mkdir_if_not_exist', return_value=True)
def test_exemplar_sampler_save_success(mock_mkdir, mock_to_feather, mock_logger, sample_available_data: pd.DataFrame, tmp_path: Path):
    """测试成功保存样本集"""
    sampler = ExemplarSampler({}, mock_logger)
    sampler.exemplar_set = sample_available_data.head(3)
    metadata = {"version_id": "v_test_save", "model_path": str(tmp_path / "model_dir/m.pkl")}
    expected_path = tmp_path / "model_dir" / "exemplar_set_v_test_save.feather"

    save_path_str = sampler._save_exemplar_set(metadata)

    assert save_path_str == str(expected_path.as_posix())
    mock_mkdir.assert_called_once_with(expected_path.parent)
    # 验证 to_feather 是否用正确的 DataFrame (带重置的索引) 调用
    mock_to_feather.assert_called_once()
    call_df = mock_to_feather.call_args[0][0] # 获取传递给 to_feather 的 DataFrame
    assert 'index' in call_df.columns # 检查索引是否已重置

def test_exemplar_sampler_select_data_logic(mock_read_feather, mock_check_exists, # 依赖加载 mock
                                             dummy_exemplar_file: Path, sample_available_data: pd.DataFrame, mock_logger):
    """测试 select_data 的合并逻辑"""
    config = {}
    sampler = ExemplarSampler(config, mock_logger)
    metadata = {"exemplar_set_path": str(dummy_exemplar_file)}
    # 配置 mock read_feather
    prev_exemplars = pd.DataFrame({'feature': [-1, -2], 'target': [-1, -2]},
                                  index=pd.to_datetime(['2023-12-30', '2023-12-31'], name='datetime'))
    mock_read_feather.return_value = prev_exemplars.reset_index()
    mock_check_exists.return_value = True

    selected_data = sampler.select_data(sample_available_data, previous_model_metadata=metadata)
    assert len(selected_data) == len(sample_available_data) + len(prev_exemplars)
    assert pd.Timestamp('2023-12-30') in selected_data.index
    assert pd.Timestamp('2024-01-20') in selected_data.index

# 测试 update_state (需要 mock _reduce_exemplar_set 和 _save_exemplar_set)
@patch('espml.incrml.data_sampling.ExemplarSampler._reduce_exemplar_set')
@patch('espml.incrml.data_sampling.ExemplarSampler._save_exemplar_set')
def test_exemplar_sampler_update_state(mock_save, mock_reduce, mock_logger, sample_available_data: pd.DataFrame):
    """测试 update_state 的主要流程"""
    config = {"MaxExemplarSetSize": 8}
    sampler = ExemplarSampler(config, mock_logger)
    new_data = sample_available_data.tail(5)
    sampler.exemplar_set = sample_available_data.head(3) # 假设当前样本集
    # 模拟 _reduce_exemplar_set 返回缩减后的集合
    reduced_set_df = pd.concat([sampler.exemplar_set, new_data]).tail(8) # 模拟结果
    mock_reduce.return_value = reduced_set_df
    # 模拟 _save_exemplar_set 返回路径
    mock_save.return_value = "/saved/path.feather"
    # 模拟元数据
    new_metadata = {"version_id": "v_new", "model_path": "dummy"}

    result = sampler.update_state(new_data, pd.DataFrame(), new_metadata)

    # 验证 _reduce_exemplar_set 被调用
    mock_reduce.assert_called_once()
    # 验证传递给 _reduce 的是合并后的数据
    call_args_reduce, _ = mock_reduce.call_args
    assert len(call_args_reduce[0]) == 8 # 3 + 5

    # 验证 _save_exemplar_set 被调用
    mock_save.assert_called_once_with(new_metadata)
    # 验证最终的样本集是缩减后的结果
    assert_frame_equal(sampler.exemplar_set, reduced_set_df)
    # 验证返回值
    assert result == {"exemplar_set_path": "/saved/path.feather"}


# --- 结束 tests/incrml/test_data_sampling.py ---