# tests/dataprocess/test_data_processor.py
# -*- coding: utf-8 -*-
# pylint: disable=protected-access, redefined-outer-name, too-many-lines, missing-function-docstring
"""
espml.dataprocess.data_processor 模块的单元测试

验证 DataProcessor 类的加载、合并、清洗、特征工程等功能
使用 pytest fixtures 提供测试数据和配置
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import shutil # 用于复制文件

# 导入被测类和相关模块
from espml.dataprocess.data_processor import DataProcessor, DataProcessingError
from espml.util import const
from espml.config.yaml_parser import ConfigError
from pandas.testing import assert_frame_equal, assert_series_equal, assert_index_equal

# --- Fixtures (从 conftest.py 导入或在此定义) ---
# 假设 conftest.py 提供了 fixtures_dir, sample_fans_path 等

@pytest.fixture(scope='module')
def full_valid_config_dict(fixtures_dir: Path) -> Dict[str, Any]:
    """提供一个完整的、用于测试 DataProcessor 的配置字典"""
    # 这个配置需要包含所有 DataProcessor 可能用到的键
    # 并且 DataSource.dir 应指向 fixtures_dir
    # (复用之前测试中定义的 minimal_valid_config 并补充完整)
    return {
        "DataSource": {
            "type": "local_dir", "dir": str(fixtures_dir), # 确保是字符串路径
            "FansTimeCol": "时间", "FansPowerCol": "实际功率",
            "TowerTimeCol": "时间",
            "TowerColRenameMap": {"风速": "Tower_WindSpeed", "风向": "Tower_WindDirection", "温度": "Tower_Temperature", "气压": "Tower_Pressure", "湿度": "Tower_Humidity"},
            "TowerHeight": 70.0,
            "WeatherIssueTimeCol": "时间", "WeatherForecastTimeCol": "预报时间",
            "WeatherColRenameMap": {"风速": "Weather_WindSpeed", "风向": "Weather_WindDirection", "温度": "Weather_Temperature", "气压": "Weather_Pressure", "湿度": "Weather_Humidity"},
            "WeatherWindSpeedHeight": 10.0
        },
        "Feature": {
            "TaskType": "regression", "TargetName": "label", "TimeIndex": "datetime",
            "TimeFormat": "%Y-%m-%d %H:%M:%S", "TimeFrequency": "15min",
            "CapacityKW": 1500.0, # 匹配 sample_fans.csv
            "IgnoreFeature": ["ID"], # ID 列不存在于示例数据中，但测试配置存在性
            "CategoricalFeature": [], # 示例数据无明确分类特征需要传递
            "ReferenceHeight": 70.0, # 设置参考高度，等于塔高，无需调整塔风速
            "TimeWindowLags": [1, 2, 4], # 测试滞后
            "TimeWindowRolling": { # 测试滚动
                 "label": {"windows": [4], "aggs": ["mean", "std"]},
                 "Tower_WindSpeed": {"windows": [4, 8], "aggs": ["mean"]}
            }
        },
        "FeatureEngineering": { # 测试特征工程配置
             "WindProfileMethod": "log_law", "RoughnessLength": 0.03, "PowerLawAlpha": 0.14,
             "Interactions": ["WindSpeedPower3", "AirDensity"], # 测试交互特征
             "DropOriginalWindDirection": True, # 测试删除风向
             "LagBaseColumns": ["label", "Tower_WindSpeed", "Weather_WindSpeed"] # 指定滞后基础列
        },
        "Cleaning": { # 测试清洗配置
             "InterpolateMethod": "linear", "InterpolateLimit": 3,
             "OutlierMethodPower": "PowerCurve",
             "PowerCurveParams": {"min_wind_speed": 3.0, "max_wind_speed": 25.0, "invalid_power_threshold_kw": 30.0, "overload_ratio": 1.1},
             "OutlierMethodWindSpeed": "Range",
             "WindSpeedRangeParams": {"min_value": 0.0, "max_value": 50.0},
             "StuckSensorWindow": 4, # 启用卡死检测
             "StuckSensorThreshold": 1e-6,
             "StdOutlierWindow": 8, # 启用滚动标准差检测
             "StdOutlierThreshold": 3.0,
             "EnableIcingDetection": False,
             "FinalNaNFillStrategy": "ffill_bfill_zero"
        },
        "Resource": {"EnableDataCache": False} # 禁用缓存确保每次测试独立加载
    }

@pytest.fixture
def processor(full_valid_config_dict: Dict[str, Any]) -> DataProcessor:
    """创建一个使用完整配置的 DataProcessor 实例"""
    # 每次测试都创建一个新实例
    return DataProcessor(config=full_valid_config_dict)

@pytest.fixture
def setup_temp_data_dir(tmp_path: Path, fixtures_dir: Path) -> Path:
    """在临时目录中创建 resource 子目录并复制所有示例 CSV 文件"""
    resource_dir = tmp_path / "resource"
    resource_dir.mkdir()
    shutil.copy(fixtures_dir / "sample_fans.csv", resource_dir / "fans.csv")
    shutil.copy(fixtures_dir / "sample_tower.csv", resource_dir / "tower.csv")
    shutil.copy(fixtures_dir / "sample_weather.csv", resource_dir / "weather.csv")
    return tmp_path # 返回包含 resource 的父目录

# --- 测试初始化 ---
def test_processor_init_success(full_valid_config_dict: Dict[str, Any]):
    """测试使用有效配置成功初始化"""
    try:
        processor = DataProcessor(config=full_valid_config_dict)
        assert processor.capacity_kw == 1500.0
        assert processor.internal_target_col == "label"
        assert processor.resource_dir == Path(full_valid_config_dict['DataSource']['dir'])
    except Exception as e:
        pytest.fail(f"DataProcessor 初始化失败: {e}")

def test_processor_init_invalid_config():
    """测试使用无效配置初始化时抛出异常"""
    with pytest.raises((ConfigError, ValueError, KeyError)):
        DataProcessor(config={}) # 空配置
    with pytest.raises((ConfigError, ValueError, KeyError)):
        DataProcessor(config={"DataSource": {"dir":"."}}) # 缺少 Feature 等

# --- 测试 _load_csv_robust ---
def test_load_csv_robust_standard(processor: DataProcessor, setup_temp_data_dir: Path):
    """测试加载标准 CSV 文件"""
    processor.resource_dir = setup_temp_data_dir / "resource"
    df = processor._load_csv_robust("fans.csv", processor.fans_time_col, time_format=processor.time_format)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == processor.internal_time_index
    assert processor.fans_power_col in df.columns

def test_load_csv_robust_usecols_rename(processor: DataProcessor, setup_temp_data_dir: Path):
    """测试 usecols 和 rename_map 参数"""
    processor.resource_dir = setup_temp_data_dir / "resource"
    tower_cols = [processor.tower_time_col, "风速", "风向"] # 只加载这几列
    tower_rename = {"风速": "WS", "风向": "WD"}
    df = processor._load_csv_robust("tower.csv", processor.tower_time_col,
                                    use_cols=tower_cols, rename_map=tower_rename,
                                    time_format=processor.time_format)
    assert list(df.columns) == ["WS", "WD"]
    assert "温度" not in df.columns

# --- 测试 _load_and_prepare_all_data ---
def test_load_and_prepare_all_success(processor: DataProcessor, setup_temp_data_dir: Path):
    """测试成功加载所有三个文件"""
    processor.resource_dir = setup_temp_data_dir / "resource"
    df_fans, df_tower, df_weather = processor._load_and_prepare_all_data()
    assert not df_fans.empty and processor.internal_target_col in df_fans.columns
    assert not df_tower.empty and const.INTERNAL_TOWER_WS_COL in df_tower.columns
    assert not df_weather.empty and const.INTERNAL_WEATHER_WS_COL in df_weather.columns
    # 检查天气预报去重
    assert len(df_weather.loc["2024-01-01 01:00:00"]) == 1 # 只有一行
    assert np.isclose(df_weather.loc["2024-01-01 01:00:00", const.INTERNAL_WEATHER_WS_COL], 6.5) # 保留的是 00:00 发布的

# --- 测试 _resample_and_align ---
def test_resample_and_align_standard(processor: DataProcessor, sample_fans_df: pd.DataFrame):
    """测试标准重采样"""
    # 准备输入 DataFrame
    df_in = sample_fans_df[[const.ACTUAL_POWER_COL]].rename(columns={const.ACTUAL_POWER_COL: 'label'})
    df_resampled = processor._resample_and_align(df_in, "Fans")
    expected_index = pd.date_range("2024-01-01 00:00:00", "2024-01-01 01:30:00", freq="15min", name=processor.internal_time_index)
    assert_index_equal(df_resampled.index, expected_index)
    assert df_resampled.loc["2024-01-01 00:00:00", 'label'] == 100.5
    #  NaN 位置仍然是 NaN (尚未插值)
    assert pd.isna(df_resampled.loc["2024-01-01 00:45:00", 'label'])

def test_resample_and_align_empty(processor: DataProcessor):
    """测试空 DataFrame 的重采样"""
    df_empty = pd.DataFrame()
    df_resampled = processor._resample_and_align(df_empty, "Empty")
    assert df_resampled.empty

# --- 测试 _merge_data_sources (包括高度调整) ---
def test_merge_data_sources_with_height_adjust(full_valid_config_dict: Dict[str, Any], setup_temp_data_dir: Path):
    """测试合并数据，特别是包含风速高度调整"""
    config = full_valid_config_dict.copy()
    # 修改配置，使塔高与参考高度不同，天气预报高度也不同
    config['DataSource']['TowerHeight'] = 10.0 # 塔高 10 米
    config['Feature']['ReferenceHeight'] = 70.0 # 参考高度 70 米
    config['DataSource']['WeatherWindSpeedHeight'] = 10.0 # 天气预报也是 10 米
    config['DataSource']['dir'] = str(setup_temp_data_dir / "resource")
    processor = DataProcessor(config=config)

    df_fans, df_tower, df_weather = processor._load_and_prepare_all_data()
    df_fans = df_fans[[processor.internal_target_col]] # 只保留目标列

    # 记录调整前风速
    ws_tower_before = df_tower.loc['2024-01-01 00:15:00', const.INTERNAL_TOWER_WS_COL] #  10m 风速 6.0
    ws_weather_before = df_weather.loc['2024-01-01 01:00:00', const.INTERNAL_WEATHER_WS_COL] #  10m 风速 6.5

    merged_df = processor._merge_data_sources(df_fans, df_tower, df_weather)

    # 检查合并结果
    assert not merged_df.empty
    assert const.INTERNAL_TOWER_WS_COL in merged_df.columns
    assert const.INTERNAL_WEATHER_WS_COL in merged_df.columns

    # 检查风速是否已被调整（应该变大）
    ws_tower_after = merged_df.loc['2024-01-01 00:15:00', const.INTERNAL_TOWER_WS_COL]
    ws_weather_after = merged_df.loc['2024-01-01 01:00:00', const.INTERNAL_WEATHER_WS_COL]
    assert ws_tower_after > ws_tower_before
    assert ws_weather_after > ws_weather_before
    # 可以用 wind_utils.adjust* 函数计算期望值进行精确断言
    expected_ws_tower = wind_utils.adjust_wind_speed_log_law(pd.Series([ws_tower_before]), 10.0, 70.0, 0.03).iloc[0]
    assert np.isclose(ws_tower_after, expected_ws_tower)


# --- 测试 _clean_merged_data (详细测试各种清洗步骤) ---
@pytest.fixture
def df_to_clean(processor: DataProcessor, setup_temp_data_dir: Path) -> pd.DataFrame:
    """提供一个包含各种待清洗问题的 DataFrame"""
    processor.resource_dir = setup_temp_data_dir / "resource"
    df_fans, df_tower, df_weather = processor._load_and_prepare_all_data()
    df_fans = df_fans[[processor.internal_target_col]]
    merged = processor._merge_data_sources(df_fans, df_tower, df_weather)
    # 手动制造问题
    merged.loc['2024-01-01 00:15:00', const.INTERNAL_TOWER_TEMP_COL] = np.nan # 插值点
    merged.loc['2024-01-01 00:30:00', const.INTERNAL_TOWER_WS_COL] = 60.0 # 范围异常
    merged.loc['2024-01-01 00:45:00', processor.internal_target_col] = np.nan # 功率 NaN
    merged.loc['2024-01-01 01:00:00', processor.internal_target_col] = 50.0 # 低风速高功率点
    merged.loc['2024-01-01 01:00:00', const.INTERNAL_TOWER_WS_COL] = 1.0
    merged.loc['2024-01-01 01:15:00', processor.internal_target_col] = -10.0 # 负功率
    # 制造卡死数据
    merged.loc['2024-01-01 00:45':'2024-01-01 01:30', const.INTERNAL_TOWER_HUM_COL] = 85.0
    return merged

def test_clean_data_pipeline(processor: DataProcessor, df_to_clean: pd.DataFrame):
    """测试完整的清洗流程"""
    df_cleaned = processor._clean_merged_data(df_to_clean)
    # 1. 检查 NaN 是否都被填充
    assert not df_cleaned.isna().any().any()
    # 2. 检查范围异常是否处理
    assert df_cleaned.loc['2024-01-01 00:30:00', const.INTERNAL_TOWER_WS_COL] <= 50.0
    assert df_cleaned.loc['2024-01-01 01:15:00', processor.internal_target_col] >= 0
    # 3. 检查功率曲线过滤（点 2024-01-01 01:00:00）
    # 由于后续插值，无法直接断言为 NaN，但值应该不是 50
    assert df_cleaned.loc['2024-01-01 01:00:00', processor.internal_target_col] != 50.0
    # 4. 检查卡死数据（湿度列在 00:45 到 01:30 之间）
    # filter_stuck_sensor 会产生 NaN，然后被插值
    humidity_slice = df_cleaned.loc['2024-01-01 00:45':'2024-01-01 01:30', const.INTERNAL_TOWER_HUM_COL]
    # 如果被标记为 NaN 并插值，值应该不再是恒定的 85.0（除非周围值恰好也是 85）
    assert not np.all(np.isclose(humidity_slice.values, 85.0)) # 验证值已改变

# --- 测试 _engineer_features (详细测试各种特征生成) ---
# 使用 processor fixture，它有完整的配置
def test_engineer_features_output(processor: DataProcessor, df_to_clean: pd.DataFrame):
    """测试特征工程的最终输出列和大致内容"""
    df_cleaned = processor._clean_merged_data(df_to_clean)
    df_featured = processor._engineer_features(df_cleaned)

    # 检查关键特征是否存在
    assert 'hour_cos' in df_featured.columns
    assert 'label_lag1' in df_featured.columns
    assert 'Tower_WindSpeed_roll4_mean' in df_featured.columns
    assert 'Tower_WindSpeed_roll8_mean' in df_featured.columns # 根据配置
    assert 'label_roll4_std' in df_featured.columns
    assert const.INTERNAL_TOWER_WDX_COL in df_featured.columns
    assert const.INTERNAL_WEATHER_WDY_COL in df_featured.columns
    assert 'AirDensity' in df_featured.columns # 交互特征
    assert f"{const.INTERNAL_TOWER_WS_COL}_pow3" in df_featured.columns # 交互特征

    # 检查风向是否被删除
    assert const.INTERNAL_TOWER_WD_COL not in df_featured.columns
    assert const.INTERNAL_WEATHER_WD_COL not in df_featured.columns

    # 检查结果中无 NaN/Inf
    assert not df_featured.isna().any().any()
    assert np.all(np.isfinite(df_featured.select_dtypes(include=np.number).values))

# --- 测试主 process 方法 (端到端) ---
def test_process_full_run(full_valid_config_dict: Dict[str, Any], setup_temp_data_dir: Path):
    """测试完整的 process 方法，从加载到特征工程"""
    config = full_valid_config_dict.copy()
    config['DataSource']['dir'] = str(setup_temp_data_dir / "resource")
    processor = DataProcessor(config=config)

    # 定义时间范围
    start_time = "2024-01-01 00:00:00"
    end_time = "2024-01-01 01:30:00"

    # 运行 process
    df_result = processor.process(start_time=start_time, end_time=end_time)

    # 基本检查
    assert isinstance(df_result, pd.DataFrame)
    expected_index = pd.date_range(start_time, end_time, freq="15min", name=processor.internal_time_index)
    assert_index_equal(df_result.index, expected_index)
    assert not df_result.isna().any().any()

    # 检查列是否存在
    assert processor.internal_target_col in df_result.columns
    assert 'hour' in df_result.columns
    assert 'label_lag1' in df_result.columns
    assert 'label_lag4' not in df_result.columns # lag_config 只配了 1, 2
    assert 'label_roll4_mean' in df_result.columns
    assert 'Tower_WindSpeed_roll8_mean' in df_result.columns
    assert const.INTERNAL_TOWER_WDX_COL in df_result.columns

    # 检查目标列名和索引名是否正确
    assert df_result.index.name == processor.internal_time_index
    # 目标列名在 process 内部已检查

def test_process_invalid_range(processor: DataProcessor, setup_temp_data_dir: Path):
    """测试当时间范围无效或无数据时 process 的行为"""
    processor.resource_dir = setup_temp_data_dir / "resource"
    # 时间范围无数据
    with pytest.raises(DataProcessingError, match="处理结果为空 DataFrame"): # 假设空结果会报错
         processor.process(start_time="2025-01-01", end_time="2025-01-02")


# --- 结束 tests/dataprocess/test_data_processor.py ---