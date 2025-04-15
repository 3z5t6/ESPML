# tests/conftest.py
# -*- coding: utf-8 -*-
"""
Pytest 配置文件,定义共享的 fixtures
"""
from typing import Any, Dict
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import shutil

# 定义 fixture 作用域 (例如 'session' 或 'module')
FIXTURE_SCOPE = 'session'

@pytest.fixture(scope=FIXTURE_SCOPE)
def project_root_dir() -> Path:
    """返回项目根目录的 Path 对象"""
    return Path(__file__).parent.parent.resolve() # conftest.py 在 tests/ 下

@pytest.fixture(scope=FIXTURE_SCOPE)
def tests_root_dir(project_root_dir: Path) -> Path:
    """返回 tests 目录的 Path 对象"""
    return project_root_dir / "tests"

@pytest.fixture(scope=FIXTURE_SCOPE)
def fixtures_dir(tests_root_dir: Path) -> Path:
    """返回测试 fixtures 目录的 Path 对象"""
    return tests_root_dir / "fixtures"

@pytest.fixture(scope='function') # 每个测试函数都使用独立的临时目录
def temp_output_dir(tmp_path: Path) -> Path:
    """提供一个临时的输出目录路径 (基于 pytest 内建的 tmp_path)"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir

@pytest.fixture(scope=FIXTURE_SCOPE)
def sample_config_path(fixtures_dir: Path) -> Path:
    """返回示例配置文件的路径"""
    return fixtures_dir / "sample_config.yaml"

@pytest.fixture(scope=FIXTURE_SCOPE)
def sample_config_dict(sample_config_path: Path) -> Dict[str, Any]:
    """加载并返回示例配置字典"""
    with open(sample_config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

@pytest.fixture(scope=FIXTURE_SCOPE)
def sample_fans_path(fixtures_dir: Path) -> Path:
    """返回示例 fans.csv 路径"""
    return fixtures_dir / "sample_fans.csv"

@pytest.fixture(scope=FIXTURE_SCOPE)
def sample_tower_path(fixtures_dir: Path) -> Path:
    """返回示例 tower.csv 路径"""
    return fixtures_dir / "sample_tower.csv"

@pytest.fixture(scope=FIXTURE_SCOPE)
def sample_weather_path(fixtures_dir: Path) -> Path:
    """返回示例 weather.csv 路径"""
    return fixtures_dir / "sample_weather.csv"

@pytest.fixture(scope=FIXTURE_SCOPE)
def sample_fans_df(sample_fans_path: Path) -> pd.DataFrame:
    """加载示例 fans.csv 为 DataFrame"""
    return pd.read_csv(sample_fans_path, parse_dates=['时间'], index_col='时间')

@pytest.fixture(scope=FIXTURE_SCOPE)
def sample_tower_df(sample_tower_path: Path) -> pd.DataFrame:
    """加载示例 tower.csv 为 DataFrame"""
    return pd.read_csv(sample_tower_path, parse_dates=['时间'], index_col='时间')

@pytest.fixture(scope=FIXTURE_SCOPE)
def sample_weather_df(sample_weather_path: Path) -> pd.DataFrame:
    """加载示例 weather.csv 为 DataFrame"""
    return pd.read_csv(sample_weather_path, parse_dates=['时间', '预报时间'])

@pytest.fixture(scope='function') # 每个集成测试使用独立的项目目录
def espml_test_project(tmp_path: Path, fixtures_dir: Path, sample_config_dict: Dict[str, Any]) -> Path:
    """
    创建一个临时的、包含基本结构和配置的项目目录用于集成测试

    目录结构:
    tmp_path/
        espml_project/
            data/
                resource/ (复制 fixtures/*.csv)
                pred/
                model/
            espml/
                project/
                    WindPower/
                        config.yaml
                        task_config.yaml
            scripts/ (可选,如果需要运行脚本)
                main.py
                backtracking.py
                ...
    """
    project_dir = tmp_path / "espml_project"
    data_dir = project_dir / "data"
    resource_dir = data_dir / "resource"
    pred_dir = data_dir / "pred"
    model_dir = data_dir / "model"
    config_dir = project_dir / "espml" / "project" / "WindPower"
    scripts_dir = project_dir / "scripts" # 可选

    # 创建目录
    resource_dir.mkdir(parents=True)
    pred_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    scripts_dir.mkdir(parents=True) # 可选

    # 复制示例数据
    shutil.copy(fixtures_dir / "sample_fans.csv", resource_dir / "fans.csv")
    shutil.copy(fixtures_dir / "sample_tower.csv", resource_dir / "tower.csv")
    shutil.copy(fixtures_dir / "sample_weather.csv", resource_dir / "weather.csv")

    # 创建配置文件 (基于 fixture,但修改路径指向临时目录)
    config_data = sample_config_dict.copy()
    config_data['DataSource']['dir'] = str(resource_dir.relative_to(project_dir)) # 使用相对路径
    # 确保 IncrML 路径在临时目录下
    config_data['IncrML']['SaveModelPath'] = str(model_dir.relative_to(project_dir))
    # 任务配置中的路径也需要调整
    if 'tasks' in config_data:
        for task in config_data['tasks']:
             if 'config_override' in task and 'IncrML' in task['config_override']:
                  task_id = task.get('task_id', 'unknown')
                  task_model_path = model_dir / task_id
                  task['config_override']['IncrML']['SaveModelPath'] = str(task_model_path.relative_to(project_dir))

    # 写入配置文件
    config_file_path = config_dir / "config.yaml"
    task_config_file_path = config_dir / "task_config.yaml"
    with open(config_file_path, 'w', encoding='utf-8') as f:
        # 保存主配置（不含 tasks）
        tasks = config_data.pop('tasks', []) # 弹出 tasks
        yaml.dump(config_data, f, allow_unicode=True, indent=4, sort_keys=False)
        config_data['tasks'] = tasks # 加回去
    with open(task_config_file_path, 'w', encoding='utf-8') as f:
        # 保存任务配置
        yaml.dump({"tasks": config_data['tasks']}, f, allow_unicode=True, indent=4, sort_keys=False)

    # 可选复制脚本文件 (如果需要测试脚本入口)
    # shutil.copy(project_root_dir / "scripts" / "main.py", scripts_dir / "main.py")
    # shutil.copy(project_root_dir / "scripts" / "backtracking.py", scripts_dir / "backtracking.py")

    # 返回项目根目录
    return project_dir
