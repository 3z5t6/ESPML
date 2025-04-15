# -*- coding: utf-8 -*-
"""
常量定义模块 (espml)
"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional

# --- 路径常量 ---
# 获取项目根目录 (假设此文件在 espml/util/ 下)
try:
    PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
except NameError:
    PROJECT_ROOT = Path.cwd() # Fallback for interactive use

# 从项目根目录构建其他路径
# 注意:  const.py 可能使用了相对路径或硬编码路径，这里统一为基于 PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
RESOURCE_DIR = DATA_DIR / "resource" # 数据目录
PRED_DIR = DATA_DIR / "pred"       # 预测结果目录
MODEL_DIR = DATA_DIR / "model"     # 模型保存目录
LOG_DIR = PROJECT_ROOT / "logs"      # 日志目录 (与 log_config.py 默认值一致)

# 项目特定配置目录 (假设与原项目结构一致)
PROJECT_CONFIG_DIR = PROJECT_ROOT / "espml" / "project" / "WindPower"

# --- 文件名常量 (来自示例数据和推断) ---
FANS_CSV = "fans.csv"        # 风机实际功率数据文件名
TOWER_CSV = "tower.csv"       # 测风塔数据文件名
WEATHER_CSV = "weather.csv"     # 天气预报数据文件名
META_CSV = "meta.csv"         # 预测/回测元数据文件名 (在 pred 目录下找到)

# --- 数据列名 (需要与 fans.csv, tower.csv, weather.csv 中的列名精确匹配) ---
# !!! 警告: 这些值必须与您 CSV 文件中的实际列标题完全一致 !!!
# 时间列
ORIG_TIME_COL = "时间" # 示例数据中的通用时间列名

# 风机数据列 (fans.csv)
ORIG_FANS_TIME_COL = ORIG_TIME_COL
ORIG_FANS_POWER_COL = "实际功率" # Target variable in fans.csv

# 测风塔数据列 (tower.csv)
# 示例 tower.csv: 时间,风向,风速,压力,湿度,温度 (根据日志推断，需要核对)
ORIG_TOWER_TIME_COL = ORIG_TIME_COL
ORIG_TOWER_WD_COL = "风向"
ORIG_TOWER_WS_COL = "风速"
ORIG_TOWER_PRESS_COL = "气压" # 日志示例中为 976.636, 976.858，看起来像气压 hPa
ORIG_TOWER_HUM_COL = "湿度"
ORIG_TOWER_TEMP_COL = "温度"
# 其他可能的测风塔列 (多层高度等)
# ORIG_TOWER_WS_80M_COL = "WindSpeed80m"
# ORIG_TOWER_WD_80M_COL = "WindDirection80m"

# 天气预报数据列 (weather.csv)
# 关键区分 '发布时间' 和 '预报目标时间'
ORIG_WEATHER_ISSUE_TIME_COL = "时间"  # 假设文件中的“时间”是发布时间
ORIG_WEATHER_FORECAST_TIME_COL = "预报时间" # 预报对应的未来时间点
ORIG_WEATHER_WD_COL = "风向"
ORIG_WEATHER_WS_COL = "风速"
ORIG_WEATHER_PRESS_COL = "气压"
ORIG_WEATHER_HUM_COL = "湿度"
ORIG_WEATHER_TEMP_COL = "温度"
# 其他可能的天气特征
# ORIG_WEATHER_CLOUD_COVER_COL = "CloudCover"
# ORIG_WEATHER_PRECIPITATION_COL = "Precipitation"

# --- 内部标准列名 (数据处理后统一使用的列名) ---
# 使用更明确的英文或英文缩写，避免中文在代码中可能带来的问题
# 并添加来源前缀以区分测风塔和天气预报特征

INTERNAL_TIME_INDEX = "datetime" # 标准时间索引名 (对应 config.yaml Feature.TimeIndex)
INTERNAL_TARGET_COL = "label"    # 标准目标变量名 (对应 config.yaml Feature.TargetName)

# 测风塔内部列名
INTERNAL_TOWER_WS_COL = "Tower_WindSpeed"
INTERNAL_TOWER_WD_COL = "Tower_WindDirection"
INTERNAL_TOWER_TEMP_COL = "Tower_Temperature"
INTERNAL_TOWER_PRESS_COL = "Tower_Pressure"
INTERNAL_TOWER_HUM_COL = "Tower_Humidity"
# 测风塔风向向量分量
INTERNAL_TOWER_WDX_COL = "Tower_WindDirection_X"
INTERNAL_TOWER_WDY_COL = "Tower_WindDirection_Y"

# 天气预报内部列名
INTERNAL_WEATHER_WS_COL = "Weather_WindSpeed"
INTERNAL_WEATHER_WD_COL = "Weather_WindDirection"
INTERNAL_WEATHER_TEMP_COL = "Weather_Temperature"
INTERNAL_WEATHER_PRESS_COL = "Weather_Pressure"
INTERNAL_WEATHER_HUM_COL = "Weather_Humidity"
# 天气预报风向向量分量
INTERNAL_WEATHER_WDX_COL = "Weather_WindDirection_X"
INTERNAL_WEATHER_WDY_COL = "Weather_WindDirection_Y"

# --- 预测任务相关常量 ---
FORECAST_PREFIX = "Forecast"      # 预测任务标识前缀
BACKTRACK_PREFIX = "backtrack"    # 回测任务标识前缀
TASK_TYPE_FORECAST = "forecast"
TASK_TYPE_BACKTRACK = "backtrack"

# 时间频率 (需要与 config.yaml Feature.TimeFrequency 保持一致)
DEFAULT_TIME_FREQ = "15min"

# 预测时间范围标识 -> 转换为 Pandas Freq String 或 Timedelta
# 日志和文件名表明存在多种预测 horizon
FORECAST_HORIZONS_PD = {
    "4Hour": pd.Timedelta(hours=4),
    "1Day": pd.Timedelta(days=1),
    "2Day": pd.Timedelta(days=2),
    "3Day": pd.Timedelta(days=3),
    "4Day": pd.Timedelta(days=4),
    "5Day": pd.Timedelta(days=5),
    "6Day": pd.Timedelta(days=6),
    "7Day": pd.Timedelta(days=7),
    "8Day": pd.Timedelta(days=8),
    "9Day": pd.Timedelta(days=9),
    "10Day": pd.Timedelta(days=10),
    "11Day": pd.Timedelta(days=11),
}
# 从 horizon 字符串 (如 "4Hour") 获取预测点数 (假设频率为 DEFAULT_TIME_FREQ)
def get_horizon_steps(horizon_key: str, freq: str = DEFAULT_TIME_FREQ) -> Optional[int]:
    """根据 Horizon Key 和时间频率计算预测步数"""
    timedelta = FORECAST_HORIZONS_PD.get(horizon_key)
    if timedelta:
        try:
            # 计算 timedelta 包含多少个 freq 时间间隔
            return int(timedelta / pd.Timedelta(freq))
        except ValueError:
            return None # 频率无法整除或无效
    return None

# --- ML/AutoML/AutoFE 常量 (默认值，会被 config.yaml 覆盖) ---
DEFAULT_METRIC = "rmse"       # 默认评估指标
DEFAULT_TASK_TYPE = "regression" # 默认任务类型
DEFAULT_RANDOM_SEED = 1024     # 默认全局随机种子

# --- 其他常量 ---
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S" # 标准日期时间格式 (用于解析和输出)
DATE_FORMAT = "%Y%m%d"             # 日期格式 (用于文件名等)
TIME_FORMAT = "%H%M"               # 时间格式 (用于文件名等)

# 风电场额定容量 (单位: KW) - 重要参数，应在 config.yaml 中配置
# DEFAULT_CAPACITY_KW = 100000.0 # 示例值 (100 MW)，必须由配置提供

# 物理常量
AIR_DENSITY_STANDARD = 1.225  # kg/m^3 at 15°C and 1 atm
GAS_CONSTANT_DRY_AIR = 287.058 # J/(kg·K)
ABSOLUTE_ZERO_CELSIUS = -273.15 # °C

# --- 状态管理常量 (如果使用了 state.py) ---
STATE_FILE_NAME = ".espml_state.json" # 状态文件名

# --- 可能存在于 const.py 但未明确的其他常量 ---
# 例如 API 端点、数据库表名、特定于项目的标识符等
# 需要仔细检查 const.py 文件以确保完全迁移