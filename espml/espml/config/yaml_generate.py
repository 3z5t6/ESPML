# -*- coding: utf-8 -*-
"""
ESPML 默认 YAML 配置文件生成器
用于生成 config.yaml 和 task_config.yaml 的模板文件
"""

import sys
import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Union

# 添加项目根目录到路径，以便导入 const
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent # config 目录的上一级是项目根目录
sys.path.insert(0, str(project_root))

try:
    from espml.util import const # 导入常量以获取默认路径
    from espml.util import utils as common_utils # 导入文件操作工具
except ImportError:
     print("错误无法导入 espml 模块请确保项目结构正确", file=sys.stderr)
     # 定义后备路径
     const = type('obj', (object,), {'PROJECT_CONFIG_DIR': Path('.') / 'espml' / 'project' / 'WindPower'})()
     common_utils = None


# --- 默认配置字典 ---

DEFAULT_PROJECT_CONFIG: Dict[str, Any] = {
    "AuthorName": "DefaultUser",
    "TaskName": "default_wind_farm",
    "JobID": None,
    "DataSource": {
        "type": "local_dir",
        "dir": "data/resource", # 相对于项目根目录
        "FansTimeCol": "时间",
        "FansPowerCol": "实际功率",
        "TowerTimeCol": "时间",
        "TowerColRenameMap": {
            "风速": "Tower_WindSpeed", "风向": "Tower_WindDirection",
            "温度": "Tower_Temperature", "气压": "Tower_Pressure", "湿度": "Tower_Humidity"
        },
        "TowerHeight": 70.0, # 假设测风塔高度
        "WeatherIssueTimeCol": "时间",
        "WeatherForecastTimeCol": "预报时间",
        "WeatherColRenameMap": {
            "风速": "Weather_WindSpeed", "风向": "Weather_WindDirection",
            "温度": "Weather_Temperature", "气压": "Weather_Pressure", "湿度": "Weather_Humidity"
        },
        "WeatherWindSpeedHeight": 10.0 # 假设预报风速高度
    },
    "Feature": {
        "TaskType": "regression",
        "TargetName": "label",
        "TimeIndex": "datetime",
        "TimeFormat": "%Y-%m-%d %H:%M:%S",
        "TimeFrequency": "15min",
        "CapacityKW": 100000.0, # 示例容量 100MW
        "IgnoreFeature": ["ID", "FARM_ID"],
        "CategoricalFeature": [],
        "GroupIndex": None,
        "Metric": "rmse",
        "SplitType": None,
        "TestSize": 0.2,
        "ReferenceHeight": 70.0, # 假设参考高度
        "TimeWindowLags": [1, 2, 4, 8, 12, 16, 24, 48, 96], # 示例滞后
        "TimeWindowRolling": {
            "label": {"windows": [4, 8, 16, 96], "aggs": ["mean", "std"]},
            "Tower_WindSpeed": {"windows": [4, 8, 16], "aggs": ["mean", "std"]},
            "Weather_WindSpeed": {"windows": [4, 8, 16], "aggs": ["mean", "std"]}
        }
    },
    "FeatureEngineering": { # 新增的特征工程详细配置
        "WindProfileMethod": "log_law", # 'log_law' or 'power_law'
        "RoughnessLength": 0.03,
        "PowerLawAlpha": 0.14,
        "Interactions": ["WindSpeedPower3", "AirDensity"], # 要生成的交互特征
        "DropOriginalWindDirection": True
    },
    "Cleaning": {
        "InterpolateMethod": "linear", # 线性插值
        "InterpolateLimit": None, # 不限制连续插值
        "OutlierMethodPower": "PowerCurve",
        "PowerCurveParams": {"min_wind_speed": 3.0, "max_wind_speed": 25.0, "invalid_power_threshold_kw": 10.0, "overload_ratio": 1.1},
        "OutlierMethodWindSpeed": "Range",
        "WindSpeedRangeParams": {"min_value": 0.0, "max_value": 50.0},
        "StuckSensorWindow": 4, # 启用卡死检测，窗口 1 小时
        "StuckSensorThreshold": 1e-6,
        "StdOutlierWindow": 96, # 启用滚动标准差检测，窗口 1 天
        "StdOutlierThreshold": 3.0,
        "EnableIcingDetection": False, # 默认不启用结冰检测
        "IcingParams": {"temp_threshold": 0.0, "rh_threshold": 90.0, "power_deviation_threshold": -0.15},
        "FinalNaNFillStrategy": "ffill_bfill_zero" # 最终填充策略
    },
    "AutoFE": {
        "Running": True, "Method": "DFS", "DFSLayers": 3, "maxTrialNum": 5, # DFS参数
        "RandomRatio": 0.25, "FeatImpThreshold": 0.001, "SaveFeatures": "OnlyGreater",
        "Operators": ["add", "sub", "mul", "div", "abs", "log"], # 示例算子
        "Transforms": [ # 示例转换
            {"name": "Lag", "params": {"periods": 1}},
            {"name": "Lag", "params": {"periods": 2}},
            {"name": "RollingMean", "params": {"window": 4}},
            {"name": "RollingStd", "params": {"window": 8}},
        ],
        "EvalValidationSize": 0.25, # AutoFE 内部评估验证集比例
        # "EvalModelParams": null, # 使用默认评估模型参数
        "NanThreshold": 0.95, "CorrThreshold": 0.98, "ConstantThreshold": 0.99,
        "MaxFeaturesPerLayer": 128,
        "port": None # port 默认不设置
    },
    "AutoML": {
        "Running": True, "Method": "flaml", "TimeBudget": 300,
        "flaml_settings": {
            # "estimator_list": ["lgbm", "extra_tree"], # 可以限制模型
        }
    },
    "IncrML": {
        "Enabled": True, # 默认启用增量学习
        "Method": "window", # 默认使用窗口方法
        "Trigger": "OnDataFileIncrease", # 默认按数据触发
        "SaveModelPath": "data/model/default_task", # 必须修改
        "DataSampling": { # 采样配置
             # WindowSampler params
             "WindowSize": "90D",
             # ExemplarSampler params (if Method=iCaRL)
             # "MaxExemplarSetSize": 10000,
             # "ExemplarSelectionStrategy": "random" # or 'herding'
        },
        "DriftDetection": { # 漂移检测配置
             "Enabled": False, # 默认禁用
             "Method": "DDM",
             # DDMParams or direct keys
             "MinNumInstances": 30,
             "WarningLevelFactor": 2.0,
             "DriftLevelFactor": 3.0,
             # "ErrorThreshold": 0.1 # For regression drift input
        },
        # "ScheduleCron": null # 如果 Trigger=Scheduled，需要设置
    },
    "Resource": {
        "trainingServicePlatform": "local",
        "MaxWorkers": -1, # 并行计算 worker 数量，-1 代表 CPU 核心数
        "EnableDataCache": True, # 是否启用数据读取缓存
        "DowncastDataTypes": False # 是否在特征工程后降低数据类型精度
    },
    "Project": { # 项目特定信息
        "Timezone": "Asia/Shanghai" # 时区配置
    }
}

DEFAULT_TASK_CONFIG: Dict[str, List[Dict[str, Any]]] = {
    "tasks": [
        {
            "task_id": "Forecast4Hour",
            "description": "短期4小时功率预测",
            "enabled": True,
            "type": "forecast",
            "forecast_horizon": "4H",
            "output_freq": "15min",
            "train_trigger_cron": "5 0,4,8,12,16,20 * * *",
            "predict_trigger_cron": "10 0,4,8,12,16,20 * * *",
            "data_fetch_lag": "15min",
            "train_start_offset": "-90D",
            "train_end_offset": "-1H",
            "predict_input_start_offset": "-2D",
            "predict_input_end_offset": "0H",
            "config_override": { # 任务特定覆盖
                "IncrML": {"SaveModelPath": "data/model/Forecast4Hour"},
                "AutoML": {"TimeBudget": 600}
            }
        },
        {
            "task_id": "Forecast1Day",
            "description": "日前1天功率预测",
            "enabled": True,
            "type": "forecast",
            "forecast_horizon": "1D",
            "output_freq": "15min",
            "train_trigger_cron": "15 7 * * *",
            "predict_trigger_cron": "20 7 * * *",
            "data_fetch_lag": "2H",
            "train_start_offset": "-180D",
            "train_end_offset": "-2H",
            "predict_input_start_offset": "-3D",
            "predict_input_end_offset": "0H",
            "config_override": {
                "IncrML": {"SaveModelPath": "data/model/Forecast1Day"},
                "AutoML": {"TimeBudget": 1800}
            }
        },
        {
            "task_id": "Backtrack1Day", # 回测任务示例
            "description": "日前1天功率预测回测",
            "enabled": False, # 默认不启用回测
            "type": "backtrack",
            "corresponding_forecast_task_id": "Forecast1Day", # 指向对应的预测任务
            "forecast_horizon": "1D",
            "output_freq": "15min",
            "backtrack_trigger_cron": "30 8 * * *", # 回测触发时间
            "backtrack_start_date": "2024-01-01", # 回测开始日期
            "backtrack_end_date": "2024-01-31",   # 回测结束日期
            "backtrack_time_of_day": "07:20:00", # 模拟预测时间点
            "backtrack_retrain": True, # 每次都重新训练
            # 回测的模型/状态保存路径模式 (可选，如果需要保存)
            # "backtrack_model_path_pattern": "data/backtrack_models/{task_id}/{date_YYYYMMDD}_{time_HHMMSS}",
            # 回测使用独立的窗口偏移量
            "backtrack_train_start_offset": "-180D",
            "backtrack_train_end_offset": "-2H",
            "backtrack_predict_input_start_offset": "-3D",
            "backtrack_predict_input_end_offset": "0H",
            "config_override": { # 回测特定覆盖
                 "AutoML": {"TimeBudget": 300} # 回测时减少预算
            }
        }
        # ... 可以添加更多预测和回测任务 ...
    ]
}


def generate_default_config(output_dir: Union[str, Path], force_overwrite: bool = False) -> None:
    """
    生成默认的 config.yaml 和 task_config.yaml 文件

    Args:
        output_dir (Union[str, Path]): 输出目录 (例如 espml/project/WindPower)
        force_overwrite (bool): 如果文件已存在，是否强制覆盖
    """
    output_path = Path(output_dir)
    # 使用 common_utils 创建目录
    if common_utils is None: # Fallback if import failed
         os.makedirs(output_path, exist_ok=True)
    elif not common_utils.mkdir_if_not_exist(output_path):
         print(f"错误无法创建输出目录 {output_path}", file=sys.stderr)
         return

    config_file = output_path / "config.yaml"
    task_config_file = output_path / "task_config.yaml"

    files_to_generate = {
        config_file: DEFAULT_PROJECT_CONFIG,
        task_config_file: DEFAULT_TASK_CONFIG
    }

    for file_path, default_data in files_to_generate.items():
        if file_path.exists() and not force_overwrite:
            print(f"文件已存在，跳过生成: {file_path}")
            continue
        print(f"正在生成默认配置文件: {file_path}")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # 使用 PyYAML 的 dump 函数，设置允许 Unicode 和缩进
                yaml.dump(default_data, f, allow_unicode=True, indent=4, sort_keys=False)
            print(f"成功生成: {file_path}")
        except Exception as e:
            print(f"错误生成文件 {file_path} 时失败: {e}", file=sys.stderr)

if __name__ == "__main__":
    # 设置默认输出目录
    default_output = const.PROJECT_CONFIG_DIR if const else Path('./default_config')
    # 可以添加命令行参数来指定输出目录和覆盖选项
    import argparse
    parser_gen = argparse.ArgumentParser(description="生成 ESPML 默认配置文件")
    parser_gen.add_argument("--output_dir", type=str, default=str(default_output), help="配置文件输出目录")
    parser_gen.add_argument("--force", action="store_true", help="如果文件已存在则覆盖")
    gen_args = parser_gen.parse_args()

    generate_default_config(gen_args.output_dir, gen_args.force)
