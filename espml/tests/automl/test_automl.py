# tests/automl/test_automl.py
# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring, redefined-outer-name, protected-access, unused-argument
"""
espml.automl.automl 模块的单元测试
验证 FlamlAutomlWrapper 类的功能
需要 mock flaml 库
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import patch, MagicMock, ANY # 导入 ANY

# 导入被测类
try:
    from espml.automl.automl import FlamlAutomlWrapper, _calculate_metric
    from flaml import AutoML as FlamlAutoMLClass # 用于类型检查和 mock
    FLAML_INSTALLED = True
except ImportError:
    pytest.skip("跳过 automl 测试，因为无法导入 flaml 或 FlamlAutomlWrapper", allow_module_level=True)
    # 定义占位符以避免后续 NameError
    FlamlAutoMLClass = None
    FlamlAutomlWrapper = None
    _calculate_metric = None


# 导入其他依赖
from loguru import logger
import joblib
from espml.util import utils as common_utils

# --- Fixtures ---
@pytest.fixture
def sample_automl_config() -> Dict[str, Any]:
    """提供 AutoML 部分的基础配置"""
    return {
        "Method": "flaml",
        "TimeBudget": 60, # 缩短测试时间预算
        "flaml_settings": { # 用户特定设置
            "estimator_list": ["lgbm", "extra_tree"],
            "verbose": 0, # 测试时减少日志输出
            "eval_method": "holdout",
            "split_ratio": 0.3
        }
    }

@pytest.fixture
def sample_global_config(sample_automl_config: Dict[str, Any]) -> Dict[str, Any]:
    """提供包含 AutoML 部分的全局配置"""
    return {
        "Feature": {"TaskType": "regression", "Metric": "rmse", "RandomSeed": 42},
        "AutoML": sample_automl_config
    }

@pytest.fixture
def sample_train_val_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """提供简单的训练和验证数据"""
    X_tr = pd.DataFrame({'a': range(10), 'b': range(10, 20)})
    y_tr = pd.Series(range(0, 20, 2))
    X_va = pd.DataFrame({'a': range(5), 'b': range(5, 10)})
    y_va = pd.Series(range(1, 11, 2))
    return X_tr, y_tr, X_va, y_va

# --- Mock flaml.AutoML ---
# 将 mock 提升到 fixture，方便复用
@pytest.fixture
def mock_flaml(mocker) -> MagicMock:
    """创建一个模拟的 flaml.AutoML 类"""
    mock_instance = MagicMock(spec=FlamlAutoMLClass)
    # 模拟 fit 方法
    mock_instance.fit = MagicMock()
    # 模拟 predict/predict_proba
    mock_instance.predict = MagicMock(return_value=np.array([0.5]*5)) # 假设返回 5 个预测
    mock_instance.predict_proba = MagicMock(return_value=np.array([[0.8, 0.2]]*5)) # 假设返回 N x 2 概率
    # 模拟结果属性
    mock_model = MagicMock()
    mock_model.estimator = MagicMock() # 模拟最佳估计器
    mock_model.estimator.predict = mock_instance.predict # 让 estimator 的 predict 指向 mock
    mock_model.estimator.predict_proba = mock_instance.predict_proba
    mock_instance.model = mock_model
    mock_instance.best_config = {'learner': 'lgbm', 'n_estimators': 50}
    mock_instance.best_loss = 0.1 # FLAML 内部损失
    mock_instance.metric = 'rmse' # 假设内部 metric

    # patch flaml.AutoML 指向这个 mock 实例构造器
    # 需要 patch 被测模块中导入的 AutoML
    patcher = patch('espml.automl.automl.AutoML', return_value=mock_instance)
    mock_class = patcher.start()
    yield mock_class, mock_instance # 返回类 mock 和实例 mock
    patcher.stop() # 测试结束后停止 patch


# --- 测试 FlamlAutomlWrapper 初始化 ---
def test_automl_wrapper_init(sample_automl_config: Dict[str, Any], sample_global_config: Dict[str, Any]):
    """测试初始化和配置合并"""
    wrapper = FlamlAutomlWrapper(config=sample_automl_config, global_config=sample_global_config)
    assert wrapper.time_budget == 60
    assert wrapper.metric == 'rmse'
    assert wrapper.task_type == 'regression'
    assert wrapper.random_seed == 42
    assert wrapper.flaml_settings['task'] == 'regression'
    assert wrapper.flaml_settings['metric'] == 'rmse'
    assert wrapper.flaml_settings['time_budget'] == 60
    assert wrapper.flaml_settings['seed'] == 42
    assert wrapper.flaml_settings['estimator_list'] == ["lgbm", "extra_tree"] # 来自用户设置
    assert wrapper.flaml_settings['verbose'] == 0
    assert wrapper.flaml_settings['eval_method'] == 'holdout' # 来自用户设置

def test_automl_wrapper_init_no_global():
    """测试缺少全局配置的情况"""
    config = {"TimeBudget": 10} # 只有 AutoML 配置
    # 初始化时不应报错，应使用默认值
    wrapper = FlamlAutomlWrapper(config=config, global_config=None)
    assert wrapper.metric == 'rmse' # 默认值
    assert wrapper.task_type == 'regression'
    assert wrapper.random_seed is None # 默认值

# --- 测试 FlamlAutomlWrapper.fit ---
@patch('espml.automl.automl._calculate_metric', return_value=0.11) # Mock 内部指标计算
def test_automl_wrapper_fit_success(mock_calc_metric, mock_flaml: Tuple[MagicMock, MagicMock],
                                    sample_automl_config: Dict[str, Any],
                                    sample_global_config: Dict[str, Any],
                                    sample_train_val_data: Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
                                    tmp_path: Path):
    """测试成功的 fit 调用流程"""
    mock_automl_class, mock_automl_instance = mock_flaml
    X_tr, y_tr, X_va, y_va = sample_train_val_data
    log_dir = tmp_path / "automl_logs"
    experiment_name = "test_exp"

    wrapper = FlamlAutomlWrapper(config=sample_automl_config, global_config=sample_global_config)
    wrapper.fit(X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va,
                cat_features=['cat_col'], log_dir=str(log_dir), experiment_name=experiment_name)

    # 验证 AutoML() 被实例化
    mock_automl_class.assert_called_once()
    # 验证 automl_instance.fit 被调用
    mock_automl_instance.fit.assert_called_once()
    # 验证 fit 参数
    fit_call_args, fit_call_kwargs = mock_automl_instance.fit.call_args
    pd.testing.assert_frame_equal(fit_call_kwargs['X_train'], X_tr)
    pd.testing.assert_series_equal(fit_call_kwargs['y_train'], y_tr)
    pd.testing.assert_frame_equal(fit_call_kwargs['X_val'], X_va)
    pd.testing.assert_series_equal(fit_call_kwargs['y_val'], y_va)
    assert fit_call_kwargs['metric'] == 'rmse'
    assert fit_call_kwargs['task'] == 'regression'
    assert fit_call_kwargs['time_budget'] == 60
    assert fit_call_kwargs['estimator_list'] == ["lgbm", "extra_tree"]
    assert Path(fit_call_kwargs['log_file_name']).name == f"{experiment_name}_flaml.log"
    assert 'categorical_feature' in fit_call_kwargs # 分类特征参数
    # 验证结果被存储
    assert wrapper.automl_instance == mock_automl_instance
    assert wrapper.best_estimator is not None
    assert wrapper.best_config == {'learner': 'lgbm', 'n_estimators': 50}
    assert wrapper.best_loss == 0.1
    # 验证最终分数计算
    mock_calc_metric.assert_called_once()
    assert np.isclose(wrapper.final_val_score, 0.11)


@patch('espml.automl.automl.AutoML') # 直接 mock 类
def test_automl_wrapper_fit_fails(MockAutoML, sample_automl_config, sample_global_config, sample_train_val_data):
    """测试 fit 过程中 FLAML 失败"""
    X_tr, y_tr, _, _ = sample_train_val_data
    mock_instance = MagicMock()
    mock_instance.fit.side_effect = RuntimeError("FLAML failed to fit") # 模拟 fit 抛出异常
    MockAutoML.return_value = mock_instance

    wrapper = FlamlAutomlWrapper(config=sample_automl_config, global_config=sample_global_config)
    with pytest.raises(RuntimeError, match="FLAML fit 失败"):
        wrapper.fit(X_train=X_tr, y_train=y_tr)
    # 验证内部状态未被设置
    assert wrapper.automl_instance is None
    assert wrapper.best_estimator is None

# --- 测试 FlamlAutomlWrapper.predict / predict_proba ---
def test_automl_wrapper_predict(mock_flaml: Tuple[MagicMock, MagicMock], sample_automl_config, sample_global_config):
    """测试 predict 方法"""
    _, mock_automl_instance = mock_flaml
    wrapper = FlamlAutomlWrapper(config=sample_automl_config, global_config=sample_global_config)
    # 手动设置训练后的状态
    wrapper.automl_instance = mock_automl_instance
    wrapper.best_estimator = mock_automl_instance.model.estimator

    X_test = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    predictions = wrapper.predict(X_test)

    # 验证 best_estimator.predict 被调用
    wrapper.best_estimator.predict.assert_called_once_with(X_test)
    assert predictions is not None

def test_automl_wrapper_predict_proba_regression(mock_flaml, sample_automl_config, sample_global_config):
    """测试在回归任务上调用 predict_proba"""
    _, mock_automl_instance = mock_flaml
    # 确保任务是回归
    sample_global_config['Feature']['TaskType'] = 'regression'
    wrapper = FlamlAutomlWrapper(config=sample_automl_config, global_config=sample_global_config)
    wrapper.automl_instance = mock_automl_instance
    wrapper.best_estimator = mock_automl_instance.model.estimator

    X_test = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    result = wrapper.predict_proba(X_test)
    assert result is None # 回归任务应返回 None

def test_automl_wrapper_predict_proba_no_support(mock_flaml, sample_automl_config, sample_global_config):
    """测试当 best_estimator 不支持 predict_proba 时"""
    _, mock_automl_instance = mock_flaml
    sample_global_config['Feature']['TaskType'] = 'classification'
    wrapper = FlamlAutomlWrapper(config=sample_automl_config, global_config=sample_global_config)
    wrapper.automl_instance = mock_automl_instance
    # 移除 predict_proba 方法
    del mock_automl_instance.model.estimator.predict_proba
    wrapper.best_estimator = mock_automl_instance.model.estimator

    X_test = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    result = wrapper.predict_proba(X_test)
    assert result is None

# --- 测试 FlamlAutomlWrapper.save_model / load_model ---
@patch('espml.automl.automl.joblib.dump')
def test_automl_wrapper_save_full_instance(mock_dump, mock_flaml, sample_automl_config, sample_global_config, tmp_path):
    """测试保存完整的 AutoML 实例"""
    mock_automl_class, mock_automl_instance = mock_flaml
    wrapper = FlamlAutomlWrapper(config=sample_automl_config, global_config=sample_global_config)
    wrapper.automl_instance = mock_automl_instance # 设置实例
    save_path = tmp_path / "model.joblib"
    success = wrapper.save_model(str(save_path), only_estimator=False)
    assert success is True
    mock_dump.assert_called_once_with(mock_automl_instance, Path(save_path))

@patch('espml.automl.automl.joblib.dump')
def test_automl_wrapper_save_only_estimator(mock_dump, mock_flaml, sample_automl_config, sample_global_config, tmp_path):
    """测试只保存最佳估计器"""
    mock_automl_class, mock_automl_instance = mock_flaml
    wrapper = FlamlAutomlWrapper(config=sample_automl_config, global_config=sample_global_config)
    wrapper.automl_instance = mock_automl_instance # 假设已训练
    wrapper.best_estimator = mock_automl_instance.model.estimator
    save_path = tmp_path / "estimator.joblib"
    success = wrapper.save_model(str(save_path), only_estimator=True)
    assert success is True
    mock_dump.assert_called_once_with(wrapper.best_estimator, Path(save_path))


@patch('espml.automl.automl.joblib.load')
def test_automl_wrapper_load_full_instance(mock_load, mock_flaml, sample_automl_config, sample_global_config, tmp_path):
    """测试加载完整的 AutoML 实例"""
    mock_automl_class, mock_automl_instance_orig = mock_flaml
    # 配置 mock load 返回模拟的 AutoML 实例
    mock_load.return_value = mock_automl_instance_orig

    load_path = tmp_path / "model.joblib"
    # 模拟文件存在
    with patch('espml.automl.automl.common_utils.check_path_exists', return_value=True):
        loaded_wrapper = FlamlAutomlWrapper.load_model(
            str(load_path),
            config=sample_automl_config, # 传入加载时的配置
            global_config=sample_global_config
        )

    mock_load.assert_called_once_with(Path(load_path))
    assert loaded_wrapper is not None
    assert isinstance(loaded_wrapper, FlamlAutomlWrapper)
    assert loaded_wrapper.automl_instance == mock_automl_instance_orig
    assert loaded_wrapper.best_estimator is not None
    assert loaded_wrapper.best_config == mock_automl_instance_orig.best_config

@patch('espml.automl.automl.joblib.load')
def test_automl_wrapper_load_only_estimator(mock_load, sample_automl_config, sample_global_config, tmp_path):
    """测试加载仅估计器文件"""
    mock_estimator = MagicMock() # 模拟一个估计器
    mock_estimator.predict = MagicMock()
    mock_load.return_value = mock_estimator

    load_path = tmp_path / "estimator.joblib"
    with patch('espml.automl.automl.common_utils.check_path_exists', return_value=True):
        loaded_wrapper = FlamlAutomlWrapper.load_model(
            str(load_path),
            config=sample_automl_config,
            global_config=sample_global_config
        )

    assert loaded_wrapper is not None
    assert loaded_wrapper.automl_instance is None # 没有完整实例
    assert loaded_wrapper.best_estimator == mock_estimator
    assert loaded_wrapper.best_config is None # 无法恢复配置

# --- 结束 tests/automl/test_automl.py ---