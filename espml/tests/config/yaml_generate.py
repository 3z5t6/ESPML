# espml/config/yaml_generate.py
# -*- coding: utf-8 -*-
# pylint: disable=all # 这是一个简单的工具脚本，禁用所有检查
"""
ESPML 默认 YAML 配置文件生成器
用于生成 config.yaml 和 task_config.yaml 的模板文件
"""

import yaml
import os
import sys # 需要 sys 来修改路径
from pathlib import Path
from typing import Dict, Any, List, Union
import argparse

# 添加项目根目录到路径，以便导入 const 和 utils
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent # config 目录的上一级是项目根目录
sys.path.insert(0, str(project_root))

# 尝试导入模块，如果失败则定义后备
try:
    from espml.util import const
    from espml.util import utils as common_utils
    YAML_DUMP_AVAILABLE = True
except ImportError:
     print("警告无法导入 espml 模块将使用基本路径和 yaml.dump", file=sys.stderr)
     # 定义后备路径和变量
     const = type('obj', (object,), {'PROJECT_CONFIG_DIR': Path('.') / 'espml' / 'project' / 'WindPower'})()
     common_utils = None
     YAML_DUMP_AVAILABLE = True # yaml 是标准库或很容易安装

# --- 默认配置字典 (与之前批次一致) ---
DEFAULT_PROJECT_CONFIG: Dict[str, Any] = {
    "AuthorName": "DefaultUser", "TaskName": "default_wind_farm", "JobID": None,
    "DataSource": {"type": "local_dir", "dir": "data/resource", "FansTimeCol": "时间", "FansPowerCol": "实际功率", "TowerTimeCol": "时间", "TowerColRenameMap": {"风速": "Tower_WindSpeed", "风向": "Tower_WindDirection", "温度": "Tower_Temperature", "气压": "Tower_Pressure", "湿度": "Tower_Humidity"}, "TowerHeight": 70.0, "WeatherIssueTimeCol": "时间", "WeatherForecastTimeCol": "预报时间", "WeatherColRenameMap": {"风速": "Weather_WindSpeed", "风向": "Weather_WindDirection", "温度": "Weather_Temperature", "气压": "Weather_Pressure", "湿度": "Weather_Humidity"}, "WeatherWindSpeedHeight": 10.0},
    "Feature": {"TaskType": "regression", "TargetName": "label", "TimeIndex": "datetime", "TimeFormat": "%Y-%m-%d %H:%M:%S", "TimeFrequency": "15min", "CapacityKW": 100000.0, "IgnoreFeature": ["ID", "FARM_ID"], "CategoricalFeature": [], "GroupIndex": None, "Metric": "rmse", "SplitType": None, "TestSize": 0.2, "ReferenceHeight": 70.0, "TimeWindowLags": [1, 2, 4, 8, 12, 16, 24, 48, 96], "TimeWindowRolling": {"label": {"windows": [4, 8, 16, 96], "aggs": ["mean", "std"]}, "Tower_WindSpeed": {"windows": [4, 8, 16], "aggs": ["mean", "std"]}, "Weather_WindSpeed": {"windows": [4, 8, 16], "aggs": ["mean", "std"]}}, "Plot": False, "RandomSeed": 1024 },
    "FeatureEngineering": {"WindProfileMethod": "log_law", "RoughnessLength": 0.03, "PowerLawAlpha": 0.14, "Interactions": ["WindSpeedPower3", "AirDensity"], "DropOriginalWindDirection": True},
    "Cleaning": {"InterpolateMethod": "linear", "InterpolateLimit": None, "OutlierMethodPower": "PowerCurve", "PowerCurveParams": {"min_wind_speed": 3.0, "max_wind_speed": 25.0, "invalid_power_threshold_kw": 10.0, "overload_ratio": 1.1}, "OutlierMethodWindSpeed": "Range", "WindSpeedRangeParams": {"min_value": 0.0, "max_value": 50.0}, "StuckSensorWindow": 4, "StuckSensorThreshold": 1e-06, "StdOutlierWindow": 96, "StdOutlierThreshold": 3.0, "EnableIcingDetection": False, "IcingParams": {"temp_threshold": 0.0, "rh_threshold": 90.0, "power_deviation_threshold": -0.15}, "FinalNaNFillStrategy": "ffill_bfill_zero"},
    "AutoFE": {"Running": True, "Method": "DFS", "DFSLayers": 3, "maxTrialNum": 5, "RandomRatio": 0.25, "FeatImpThreshold": 0.001, "SaveFeatures": "OnlyGreater", "Operators": ["add", "sub", "mul", "div", "abs", "log"], "Transforms": [{"name": "Lag", "params": {"periods": 1}}, {"name": "Lag", "params": {"periods": 2}}, {"name": "RollingMean", "params": {"window": 4}}, {"name": "RollingStd", "params": {"window": 8}}], "EvalValidationSize": 0.25, "NanThreshold": 0.95, "CorrThreshold": 0.98, "ConstantThreshold": 0.99, "MaxFeaturesPerLayer": 128, "port": None},
    "AutoML": {"Running": True, "Method": "flaml", "TimeBudget": 300, "flaml_settings": {}},
    "IncrML": {"Enabled": True, "Method": "window", "Trigger": "OnDataFileIncrease", "SaveModelPath": "data/model/default_task", "DataSampling": {"WindowSize": "90D"}, "DriftDetection": {"Enabled": False, "Method": "DDM", "MinNumInstances": 30, "WarningLevelFactor": 2.0, "DriftLevelFactor": 3.0}},
    "Resource": {"trainingServicePlatform": "local", "MaxWorkers": -1, "EnableDataCache": True, "DowncastDataTypes": False},
    "Project": {"Timezone": "Asia/Shanghai"}
}
DEFAULT_TASK_CONFIG: Dict[str, List[Dict[str, Any]]] = {
    "tasks": [
        {"task_id": "Forecast4Hour", "description": "短期4小时功率预测", "enabled": True, "type": "forecast", "forecast_horizon": "4H", "output_freq": "15min", "train_trigger_cron": "5 0,4,8,12,16,20 * * *", "predict_trigger_cron": "10 0,4,8,12,16,20 * * *", "data_fetch_lag": "15min", "train_start_offset": "-90D", "train_end_offset": "-1H", "predict_input_start_offset": "-2D", "predict_input_end_offset": "0H", "config_override": {"IncrML": {"SaveModelPath": "data/model/Forecast4Hour"}, "AutoML": {"TimeBudget": 600}}},
        {"task_id": "Forecast1Day", "description": "日前1天功率预测", "enabled": True, "type": "forecast", "forecast_horizon": "1D", "output_freq": "15min", "train_trigger_cron": "15 7 * * *", "predict_trigger_cron": "20 7 * * *", "data_fetch_lag": "2H", "train_start_offset": "-180D", "train_end_offset": "-2H", "predict_input_start_offset": "-3D", "predict_input_end_offset": "0H", "config_override": {"IncrML": {"SaveModelPath": "data/model/Forecast1Day"}, "AutoML": {"TimeBudget": 1800}}},
        {"task_id": "Backtrack1Day", "description": "日前1天功率预测回测", "enabled": False, "type": "backtrack", "corresponding_forecast_task_id": "Forecast1Day", "forecast_horizon": "1D", "output_freq": "15min", "backtrack_trigger_cron": "30 8 * * *", "backtrack_start_date": "2024-01-01", "backtrack_end_date": "2024-01-31", "backtrack_time_of_day": "07:20:00", "backtrack_retrain": True, "backtrack_train_start_offset": "-180D", "backtrack_train_end_offset": "-2H", "backtrack_predict_input_start_offset": "-3D", "backtrack_predict_input_end_offset": "0H", "config_override": {"AutoML": {"TimeBudget": 300}}}
    ]
}

def generate_default_config(output_dir: Union[str, Path], force_overwrite: bool = False) -> None:
    """生成默认的 config.yaml 和 task_config.yaml 文件"""
    if not YAML_DUMP_AVAILABLE:
        print("错误PyYAML 库未安装，无法生成 YAML 文件", file=sys.stderr)
        return

    output_path = Path(output_dir)
    # 创建目录
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
         print(f"错误无法创建输出目录 {output_path}: {e}", file=sys.stderr)
         return

    files_to_generate = {
        output_path / "config.yaml": DEFAULT_PROJECT_CONFIG,
        output_path / "task_config.yaml": DEFAULT_TASK_CONFIG
    }

    for file_path, default_data in files_to_generate.items():
        if file_path.exists() and not force_overwrite:
            print(f"文件已存在，跳过生成: {file_path}")
            continue
        print(f"正在生成默认配置文件: {file_path}")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_data, f, allow_unicode=True, indent=4, sort_keys=False, default_flow_style=False)
            print(f"成功生成: {file_path}")
        except Exception as e:
            print(f"错误生成文件 {file_path} 时失败: {e}", file=sys.stderr)

if __name__ == "__main__":
    # 脚本入口，允许通过命令行指定输出目录和覆盖选项
    default_output = const.PROJECT_CONFIG_DIR if const else Path('./default_config')
    parser_gen = argparse.ArgumentParser(description="生成 ESPML 默认配置文件")
    parser_gen.add_argument("--output_dir", type=str, default=str(default_output), help=f"配置文件输出目录 (默认: {default_output})")
    parser_gen.add_argument("--force", action="store_true", help="如果文件已存在则覆盖")
    gen_args = parser_gen.parse_args()

    generate_default_config(gen_args.output_dir, gen_args.force)

# --- 结束 espml/config/yaml_generate.py ---