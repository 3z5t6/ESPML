# tests/integration/test_training_flow.py
# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-argument
"""
ESPML 端到端训练流程的集成测试
"""

import pytest
import os
from pathlib import Path
import pandas as pd
import datetime

# 导入需要调用的类和函数
from espml.util.wind_incrml import WindTaskRunner
from espml.config.yaml_parser import load_yaml_config
from espml.util.state import save_state # 可能需要初始化状态

# 跳过测试如果核心模块无法导入
try:
    from espml.ml import MLPipeline
    INTEGRATION_TEST_READY = True
except ImportError:
    INTEGRATION_TEST_READY = False

pytestmark = pytest.mark.skipif(not INTEGRATION_TEST_READY, reason="核心 ESPML 模块无法导入,跳过集成测试")

def test_full_training_run(espml_test_project: Path, caplog):
    """
    测试运行一次完整的训练流程（非增量）
    Args:
        espml_test_project (Path): 由 fixture 创建的临时项目目录
        caplog: pytest 内建 fixture,捕获日志输出
    """
    project_dir = espml_test_project
    # 修改工作目录,以便配置文件中的相对路径能正确解析
    original_cwd = Path.cwd()
    os.chdir(project_dir)

    task_id_to_run = "SampleTask" # 使用 fixture config 中的任务 ID
    config_file = project_dir / "espml" / "project" / "WindPower" / "config.yaml"
    task_config_file = project_dir / "espml" / "project" / "WindPower" / "task_config.yaml"

    try:
        # --- 准备 ---
        # 加载配置
        full_config = load_yaml_config(config_file)
        task_config_content = load_yaml_config(task_config_file)
        full_config['tasks'] = task_config_content['tasks']

        # 确保任务已启用,且 IncrML 禁用（用于测试完全训练）
        task_found = False
        for task in full_config['tasks']:
             if task['task_id'] == task_id_to_run:
                  task['enabled'] = True
                  # 强制禁用增量学习以测试完整流程
                  task['config_override'] = task.get('config_override', {})
                  task['config_override']['IncrML'] = task['config_override'].get('IncrML', {})
                  task['config_override']['IncrML']['Enabled'] = False
                  task_found = True
                  break
        assert task_found, f"测试任务 '{task_id_to_run}' 未在配置中找到"

        # 初始化并运行 WindTaskRunner
        runner = WindTaskRunner(task_id=task_id_to_run, config=full_config)
        # 注意需要确保测试数据的时间范围与 runner 内部计算的训练窗口有交集
        # 可以 mock pd.Timestamp.now() 或传入 current_time_override
        # 此处假设测试数据足够覆盖默认配置下的训练窗口（例如过去30天）
        # 为了稳定,我们 mock 当前时间
        mock_current_time = pd.Timestamp("2024-01-15 10:00:00", tz=runner.DEFAULT_TIMEZONE)

        # --- 执行 ---
        with caplog.at_level(logging.INFO): # 捕获 INFO 及以上级别日志
             runner.run(current_time_override=mock_current_time)

        # --- 验证 ---
        # 1. 验证日志输出（关键步骤）
        assert f"=============== 开始执行任务: {task_id_to_run}" in caplog.text
        assert f"为任务 '{task_id_to_run}' 初始化 WindTaskRunner" in caplog.text
        assert "开始检查数据可用性" in caplog.text
        assert "make sure Data is available every 15 minutes in range" in caplog.text # 验证日志格式
        assert "数据可用性检查通过" in caplog.text
        assert f"{task_id_to_run} task start training." in caplog.text # 训练开始日志
        assert "步骤 1/5: 执行数据处理..." in caplog.text # MLPipeline 日志
        assert "步骤 2/5: 拆分训练集/验证集..." in caplog.text
        assert "步骤 3/5: 执行初始基线评估..." in caplog.text
        assert "origin data train val rmse:" in caplog.text
        assert "步骤 4/5: 执行自动特征工程 (AutoFE)..." in caplog.text
        # AutoFE 内部日志 (来自 algorithm.py)
        assert "开始 DFS 特征搜索" in caplog.text
        assert "dfs 1 algorithm search" in caplog.text # 假设至少跑一层
        assert "autofe finished, search" in caplog.text # MLPipeline 日志
        assert "步骤 5/5: 执行自动机器学习 (AutoML)..." in caplog.text
        # AutoML 内部日志 (来自 automl.py)
        assert "开始 FLAML AutoML 训练" in caplog.text
        assert "FLAML log file:" in caplog.text
        assert "automl finished, best model can achieve rmse:" in caplog.text # 或其他 metric
        assert "模型及状态保存完成" in caplog.text # MLPipeline 日志
        assert "finish training." in caplog.text # MLPipeline 日志
        assert f"{task_id_to_run} task finish training successfully." in caplog.text # WindTaskRunner 日志
        assert f"=============== 任务执行结束: {task_id_to_run}" in caplog.text

        # 2. 验证输出文件是否存在
        # 需要获取 runner.ml_pipeline.last_run_id
        last_run_id = runner.ml_pipeline.last_run_id
        assert last_run_id is not None
        model_path, tf_path, feat_path, _ = runner.ml_pipeline._get_run_specific_paths(last_run_id)
        assert Path(model_path).exists()
        assert Path(tf_path).exists()
        assert Path(feat_path).exists()
        # 检查 AutoML 日志文件
        automl_log_dir = Path(model_path).parent / "automl" / "logs"
        assert automl_log_dir.exists()
        # 检查是否有 flaml 日志文件生成 (文件名可能包含 run_id)
        assert len(list(automl_log_dir.glob("*flaml.log"))) > 0

    finally:
        # 切换回工作目录
        os.chdir(original_cwd)

# --- 可以添加更多集成测试,例如测试预测流程、增量流程、回测流程 ---
# 例如
def test_prediction_flow():
    """测试预测流程"""
    pass

def test_incremental_update_flow():
    """测试增量更新流程"""
    pass

def test_backtracking_run():
    """测试回溯流程"""
    pass

# --- 结束 tests/integration/test_training_flow.py ---