# tests/incrml/test_detect_drift.py
# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring, redefined-outer-name, protected-access
"""
espml.incrml.detect_drift 模块的单元测试
验证 DriftDetectorDDM 类和 get_drift_detector 函数
"""
import pytest
import numpy as np
from typing import Dict, Any, Optional, Literal

# 导入被测类和函数
try:
    from espml.incrml.detect_drift import DriftDetectorDDM, get_drift_detector, BaseDriftDetector
    from loguru import logger
    DRIFT_MODULE_LOADED = True
except ImportError as e:
    DRIFT_MODULE_LOADED = False
    pytest.skip(f"跳过 detect_drift 测试，因为导入失败: {e}", allow_module_level=True)

pytestmark = pytest.mark.skipif(not DRIFT_MODULE_LOADED, reason="espml.incrml.detect_drift 或其依赖项无法导入")

# --- Fixtures ---
@pytest.fixture
def ddm_config() -> Dict[str, Any]:
    """提供 DDM 的默认配置"""
    return {
        'MinNumInstances': 30,
        'WarningLevelFactor': 2.0,
        'DriftLevelFactor': 3.0
    }

@pytest.fixture
def drift_config_wrapper(ddm_config) -> Dict[str, Any]:
    """包装 DDM 配置以匹配 get_drift_detector 的输入"""
    return {"IncrML": {"DriftDetection": {"Enabled": True, "Method": "DDM", **ddm_config}}}

@pytest.fixture
def ddm_detector(ddm_config: Dict[str, Any]) -> DriftDetectorDDM:
    """创建一个 DDM 检测器实例"""
    # 实例化时传递配置字典本身
    return DriftDetectorDDM(config=ddm_config)

# --- 测试 DriftDetectorDDM (与之前实现一致) ---
def test_ddm_initialization(ddm_detector: DriftDetectorDDM):
    assert ddm_detector.n == 0
    assert ddm_detector.p == 1.0
    assert ddm_detector.s == 0.0
    assert ddm_detector.p_min == 1.0
    assert ddm_detector.s_min == 0.0
    assert ddm_detector.drift_state is None

def test_ddm_initialization_invalid_params():
    with pytest.raises(ValueError): DriftDetectorDDM({'MinNumInstances': 0})
    with pytest.raises(ValueError): DriftDetectorDDM({'WarningLevelFactor': -1.0})
    with pytest.raises(ValueError): DriftDetectorDDM({'DriftLevelFactor': 1.0, 'WarningLevelFactor': 2.0})

def test_ddm_reset(ddm_detector: DriftDetectorDDM):
    ddm_detector.n = 50; ddm_detector.p = 0.5; ddm_detector.drift_state = 'warning'
    ddm_detector._reset()
    assert ddm_detector.n == 0; assert ddm_detector.p == 1.0; assert ddm_detector.drift_state is None

def test_ddm_add_element_stats_update(ddm_detector: DriftDetectorDDM):
    # 添加错误 (error=1)
    ddm_detector.add_element(False)
    assert ddm_detector.n == 1
    assert np.isclose(ddm_detector.p, 1.0)
    assert np.isclose(ddm_detector.s, 0.0)
    # 添加正确 (error=0)
    ddm_detector.add_element(True)
    assert ddm_detector.n == 2
    assert np.isclose(ddm_detector.p, 0.5)
    assert np.isclose(ddm_detector.s, np.sqrt(0.5*0.5/2))
    # 添加正确 (error=0)
    ddm_detector.add_element(True)
    assert ddm_detector.n == 3
    assert np.isclose(ddm_detector.p, 1.0/3.0)
    assert np.isclose(ddm_detector.s, np.sqrt((1/3)*(2/3)/3))

def test_ddm_min_update_logic(ddm_detector: DriftDetectorDDM):
    # 填充到 min_instance_num
    for _ in range(ddm_detector.min_instance_num): ddm_detector.add_element(True)
    p_min1, s_min1 = ddm_detector.p_min, ddm_detector.s_min
    # 再添加一个正确，p+s 应该下降
    ddm_detector.add_element(True)
    assert ddm_detector.p + ddm_detector.s < p_min1 + s_min1
    assert np.isclose(ddm_detector.p_min, ddm_detector.p)
    assert np.isclose(ddm_detector.s_min, ddm_detector.s)
    p_min2, s_min2 = ddm_detector.p_min, ddm_detector.s_min
    # 添加一个错误，p+s 可能上升
    ddm_detector.add_element(False)
    assert np.isclose(ddm_detector.p_min, p_min2) # min 不应改变
    assert np.isclose(ddm_detector.s_min, s_min2)

def test_ddm_drift_states_transition(ddm_detector: DriftDetectorDDM):
    # 1. 稳定期
    for _ in range(ddm_detector.min_instance_num + 5): ddm_detector.add_element(True)
    ddm_detector.add_element(False) # 少量错误
    assert ddm_detector.get_drift_state() is None

    # 2. 触发警告
    p_min, s_min = ddm_detector.p_min, ddm_detector.s_min
    warning_thr = p_min + ddm_detector.warning_level_factor * s_min
    drift_thr = p_min + ddm_detector.drift_level_factor * s_min
    in_warning = False
    for _ in range(200): # 添加错误直到警告
        ddm_detector.add_element(False)
        if ddm_detector.p + ddm_detector.s >= warning_thr:
            in_warning = True; break
    assert in_warning, "未能触发警告"
    assert ddm_detector.detected_warning_zone() is True
    assert ddm_detector.detected_change() is False

    # 3. 触发漂移
    in_drift = False
    for _ in range(300): # 继续添加错误直到漂移
        ddm_detector.add_element(False)
        if ddm_detector.p + ddm_detector.s >= drift_thr:
            in_drift = True; break
    assert in_drift, "未能触发漂移"
    assert ddm_detector.detected_warning_zone() is False
    assert ddm_detector.detected_change() is True

    # 4. 漂移后状态保持
    ddm_detector.add_element(True)
    assert ddm_detector.detected_change() is True

    # 5. 重置后恢复
    ddm_detector._reset()
    assert ddm_detector.get_drift_state() is None

# --- 测试 get_drift_detector (与之前实现一致) ---
def test_get_drift_detector_factory(drift_config_wrapper: Dict[str, Any]):
    # 测试 DDM
    detector_ddm = get_drift_detector(drift_config_wrapper, logger)
    assert isinstance(detector_ddm, DriftDetectorDDM)
    # 测试禁用
    drift_config_wrapper['IncrML']['DriftDetection']['Enabled'] = False
    detector_disabled = get_drift_detector(drift_config_wrapper, logger)
    assert detector_disabled is None
    drift_config_wrapper['IncrML']['DriftDetection']['Enabled'] = True # 恢复
    # 测试未知方法
    drift_config_wrapper['IncrML']['DriftDetection']['Method'] = "UNKNOWN"
    detector_unknown = get_drift_detector(drift_config_wrapper, logger)
    assert detector_unknown is None
    drift_config_wrapper['IncrML']['DriftDetection']['Method'] = "DDM" # 恢复
    # 测试无漂移配置
    config_no_drift = {"IncrML": {}}
    detector_no_conf = get_drift_detector(config_no_drift, logger)
    assert detector_no_conf is None

# --- 结束 tests/incrml/test_detect_drift.py ---