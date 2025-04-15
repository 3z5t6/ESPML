# tests/incrml/test_metadata.py
# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring, redefined-outer-name, protected-access
"""
espml.incrml.metadata 模块的单元测试
验证 ModelVersionInfo 和 IncrmlMetadata 类的功能
"""

import pytest
import json
from pathlib import Path
import datetime
import os
from typing import Dict, Any
import time # 用于创建不同时间戳

# 导入被测类和相关模块
# 使用 try-except 保证测试文件本身可运行
try:
    from espml.incrml.metadata import ModelVersionInfo, IncrmlMetadata, EnhancedJSONEncoder
    from espml.util import const # 需要项目根目录
    from espml.util import utils as common_utils # 需要文件操作
    METADATA_MODULE_LOADED = True
except ImportError as e:
    METADATA_MODULE_LOADED = False
    pytest.skip(f"跳过 metadata 测试,因为导入失败: {e}", allow_module_level=True)

# --- Fixtures ---
@pytest.fixture
def sample_metadata_dir(tmp_path: Path) -> Path:
    """提供一个临时的元数据目录"""
    md_dir = tmp_path / "metadata_test_incrml"
    md_dir.mkdir()
    return md_dir

@pytest.fixture
def sample_version_info_data() -> Dict[str, Any]:
    """提供一个用于创建 ModelVersionInfo 的示例字典"""
    # 使用相对路径或占位符,因为绝对路径依赖环境
    base_path = "model_output/run_20240101T100000Z"
    return {
        "version_id": "20240101T100000Z",
        "timestamp": "2024-01-01T10:00:00+00:00",
        "model_path": f"{base_path}/model.joblib",
        "transformer_state_path": f"{base_path}/tf_state.joblib",
        "selected_features_path": f"{base_path}/features.json",
        "training_data_start": "2023-12-01T00:00:00+00:00",
        "training_data_end": "2024-01-01T09:45:00+00:00",
        "performance_metrics": {"rmse": 0.15, "mae": 0.1},
        "drift_status": False,
        "base_model_version": None,
        "exemplar_set_path": None,
        "misc_info": {"info": "Initial model", "feature_count": 50}
    }

@pytest.fixture
def sample_version_info(sample_version_info_data: Dict[str, Any]) -> ModelVersionInfo:
    """创建一个 ModelVersionInfo 实例"""
    return ModelVersionInfo.from_dict(sample_version_info_data)

# --- 测试 ModelVersionInfo ---
# (保持不变,已在先前步骤确认)

# --- 测试 IncrmlMetadata ---
def test_incrml_metadata_init_new(sample_metadata_dir: Path):
    """测试初始化（无现有文件）"""
    task_id = "TestTaskInitNewMeta"
    meta = IncrmlMetadata(task_id, sample_metadata_dir)
    assert meta.task_id == task_id
    assert meta.metadata_dir == sample_metadata_dir
    assert meta.versions == {}
    assert meta.current_version_id is None
    assert not meta.metadata_file.exists()

def test_incrml_metadata_save_load_cycle(sample_metadata_dir: Path, sample_version_info: ModelVersionInfo):
    """测试保存和加载元数据"""
    task_id = "TestTaskCycleMeta"
    meta_write1 = IncrmlMetadata(task_id, sample_metadata_dir)
    info1 = sample_version_info; info1.version_id = "v1_meta"
    meta_write1.add_version(info1, set_as_current=True)
    save_ok1 = meta_write1.save()
    assert save_ok1 is True
    assert meta_write1.metadata_file.exists()

    meta_read1 = IncrmlMetadata(task_id, sample_metadata_dir)
    assert len(meta_read1.versions) == 1
    assert meta_read1.current_version_id == "v1_meta"
    loaded_v1 = meta_read1.get_current_version()
    assert loaded_v1 is not None and loaded_v1.performance_metrics == info1.performance_metrics

    # 添加第二个版本
    info2_data = sample_version_info.to_dict(); info2_data['version_id'] = "v2_meta"
    info2_data['base_model_version'] = "v1_meta"
    info2_data['timestamp'] = (datetime.datetime.now(datetime.timezone.utc)).isoformat()
    info2 = ModelVersionInfo.from_dict(info2_data)
    meta_read1.add_version(info2, set_as_current=True)
    save_ok2 = meta_read1.save()
    assert save_ok2 is True

    meta_read2 = IncrmlMetadata(task_id, sample_metadata_dir)
    assert len(meta_read2.versions) == 2
    assert meta_read2.current_version_id == "v2_meta"
    loaded_v2 = meta_read2.get_current_version()
    assert loaded_v2 is not None and loaded_v2.base_model_version == "v1_meta"

    # 验证 JSON 内容
    with open(meta_read2.metadata_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    assert raw_data['current_version_id'] == "v2_meta"
    assert "v1_meta" in raw_data['versions']
    assert raw_data['versions']['v1_meta']['model_path'] == info1.model_path

def test_incrml_metadata_get_current_logic(sample_metadata_dir: Path, sample_version_info: ModelVersionInfo):
    """测试获取当前版本的逻辑（包括未设置时返回最新）"""
    task_id = "TestTaskGetCurrent"
    meta = IncrmlMetadata(task_id, sample_metadata_dir)
    info1 = sample_version_info; info1.version_id = "v1_old"; info1.timestamp = "2024-01-01T00:00:00Z"
    time.sleep(0.01) # 确保时间戳不同
    info2_data = sample_version_info.to_dict(); info2_data['version_id'] = "v2_new"
    info2_data['timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    info2 = ModelVersionInfo.from_dict(info2_data)

    meta.add_version(info1, set_as_current=False)
    meta.add_version(info2, set_as_current=False) # 都不设为当前
    meta.current_version_id = None

    current = meta.get_current_version()
    assert current is not None and current.version_id == "v2_new" # 应返回最新的 v2_new

    # 设置一个无效的 current_id
    meta.current_version_id = "v_non_existent"
    current = meta.get_current_version()
    assert current is not None and current.version_id == "v2_new" # 仍然返回最新的

    # 设置有效的 current_id
    meta.set_current_version("v1_old")
    current = meta.get_current_version()
    assert current is not None and current.version_id == "v1_old"

# --- 结束 tests/incrml/test_metadata.py ---