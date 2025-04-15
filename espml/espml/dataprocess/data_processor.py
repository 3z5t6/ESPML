# -*- coding: utf-8 -*-
"""
数据处理模块 (espml)
负责加载、清洗、合并、特征工程
!! 利用了完整的 espml.util.wind_utils 和 espml.util.utils 模块 !!
"""

import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from loguru import logger

from espml.util import const, wind_utils, utils
from espml.config.yaml_parser import ConfigError

# --- 自定义数据处理错误 ---
class DataProcessingError(Exception):
    """在数据处理流程中发生特定错误时引发"""
    pass

class DataProcessor:
    """
    封装完整的数据处理流程,严格遵循项目逻辑

    从配置文件读取参数,执行加载、合并、清洗、特征工程等步骤
    包含对风速高度调整、高级数据过滤（卡死、标准差、结冰）、
    以及精细化特征工程（滞后、滚动、交互、风向向量）的支持
    """
    _raw_data_cache: Dict[str, pd.DataFrame] = {} # 缓存读取数据
    _cache_enabled: bool = True

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 DataProcessor

        Args:
            config (Dict[str, Any]): 从 config.yaml 加载的完整项目配置字典

        Raises:
            ConfigError: 如果关键配置项缺失或无效
        """
        logger.info("初始化 DataProcessor...")
        self.config = config
        # 校验并提取所有需要的配置项
        self._validate_and_extract_config()
        # 决定是否启用数据读取缓存
        self._cache_enabled = utils.safe_dict_get(config, 'Resource.EnableDataCache', True)
        logger.info("DataProcessor 初始化完成")

    def _validate_and_extract_config(self) -> None:
        """(私有) 校验配置的完整性,并提取常用配置项到实例属性以便访问"""
        logger.debug("开始校验和提取数据处理配置...")
        extracted_config = {}
        try:
            # --- 数据源配置 ---
            ds_config = self.config['DataSource']
            extracted_config['resource_dir'] = const.PROJECT_ROOT / ds_config['dir']
            extracted_config['fans_time_col'] = ds_config['FansTimeCol']
            extracted_config['fans_power_col'] = ds_config['FansPowerCol']
            extracted_config['tower_time_col'] = ds_config['TowerTimeCol']
            extracted_config['tower_rename_map'] = ds_config['TowerColRenameMap']
            extracted_config['tower_height'] = float(utils.safe_dict_get(ds_config, 'TowerHeight', 70.0))

            extracted_config['weather_issue_time_col'] = ds_config['WeatherIssueTimeCol']
            extracted_config['weather_forecast_time_col'] = ds_config['WeatherForecastTimeCol']
            extracted_config['weather_rename_map'] = ds_config['WeatherColRenameMap']
            extracted_config['weather_ws_height'] = float(utils.safe_dict_get(ds_config, 'WeatherWindSpeedHeight', 10.0))

            # --- 特征配置 ---
            feat_config = self.config['Feature']
            extracted_config['internal_target_col'] = feat_config['TargetName']
            extracted_config['internal_time_index'] = feat_config['TimeIndex']
            extracted_config['time_format'] = feat_config['TimeFormat']
            extracted_config['time_freq'] = feat_config['TimeFrequency']
            extracted_config['capacity_kw'] = float(feat_config['CapacityKW'])
            extracted_config['ignore_features'] = feat_config.get('IgnoreFeature', [])
            extracted_config['categorical_features'] = feat_config.get('CategoricalFeature', [])
            extracted_config['lag_config'] = feat_config.get('TimeWindowLags', []) # 假设直接是 lags 列表
            extracted_config['rolling_config'] = feat_config.get('TimeWindowRolling', {})
            extracted_config['reference_height'] = utils.safe_dict_get(feat_config, 'ReferenceHeight') # None if not present
            if extracted_config['reference_height'] is not None:
                extracted_config['reference_height'] = float(extracted_config['reference_height'])

            # --- 特征工程配置 ---
            fe_config = self.config.get('FeatureEngineering', {}) # FeatureEngineering section is optional
            extracted_config['wind_profile_method'] = fe_config.get('WindProfileMethod', 'log_law')
            extracted_config['roughness_length'] = float(fe_config.get('RoughnessLength', 0.03))
            extracted_config['power_law_alpha'] = float(fe_config.get('PowerLawAlpha', 0.14))
            extracted_config['interaction_config'] = fe_config.get('Interactions', [])
            extracted_config['drop_orig_wd'] = bool(fe_config.get('DropOriginalWindDirection', True))
            # 定义需要滞后的基础列 (可以配置化)
            extracted_config['lag_base_cols'] = fe_config.get('LagBaseColumns', [
                extracted_config['internal_target_col'], # Use extracted name
                const.INTERNAL_TOWER_WS_COL, const.INTERNAL_WEATHER_WS_COL,
                const.INTERNAL_TOWER_TEMP_COL, const.INTERNAL_WEATHER_TEMP_COL
            ])


            # --- 清洗配置 ---
            clean_config = self.config['Cleaning']
            extracted_config['interpolate_method'] = clean_config['InterpolateMethod']
            extracted_config['interpolate_limit'] = clean_config.get('InterpolateLimit')
            extracted_config['outlier_method_power'] = clean_config['OutlierMethodPower']
            extracted_config['power_curve_params'] = clean_config.get('PowerCurveParams', {})
            # 确保 power_curve_params 包含 overload_ratio 默认值
            extracted_config['power_curve_params'].setdefault('overload_ratio', 1.1)

            extracted_config['outlier_method_ws'] = clean_config.get('OutlierMethodWindSpeed')
            extracted_config['ws_range_params'] = clean_config.get('WindSpeedRangeParams', {})
            extracted_config['stuck_sensor_window'] = int(utils.safe_dict_get(clean_config, 'StuckSensorWindow', 0)) # 0 to disable
            extracted_config['stuck_sensor_threshold'] = float(utils.safe_dict_get(clean_config, 'StuckSensorThreshold', 1e-6))
            extracted_config['std_outlier_window'] = int(utils.safe_dict_get(clean_config, 'StdOutlierWindow', 0)) # 0 to disable
            extracted_config['std_outlier_threshold'] = float(utils.safe_dict_get(clean_config, 'StdOutlierThreshold', 3.0))
            extracted_config['enable_icing_detection'] = bool(utils.safe_dict_get(clean_config, 'EnableIcingDetection', False))
            extracted_config['icing_params'] = utils.safe_dict_get(clean_config, 'IcingParams', {})
            # 最终 NaN 填充策略 (需要添加到 config.yaml)
            extracted_config['final_nan_fill_strategy'] = clean_config.get('FinalNaNFillStrategy', 'ffill_bfill_zero') # 'zero', 'ffill_bfill_zero'

        except KeyError as e:
            raise ConfigError(f"配置文件缺少必要的键: {e}请检查 config.yaml") from e
        except (ValueError, TypeError) as e:
             raise ConfigError(f"配置项值错误: {e}") from e

        # 将提取的配置项设置到实例属性
        for key, value in extracted_config.items():
             setattr(self, key, value)

        # 额外的校验逻辑
        if self.capacity_kw <= 0:
            raise ConfigError("无效的风场额定容量配置 (必须 > 0)")
        if self.reference_height is not None and self.reference_height <= 0:
             raise ConfigError("无效的参考高度配置 (必须 > 0)")
        # 可以在这里添加更多校验...

        logger.debug("数据处理配置校验和提取完成")

    # --- _load_csv_robust ---
    # (保持之前的实现,确保它使用 self.resource_dir, self.internal_time_index, self.time_format 等实例属性)
    @utils.log_execution_time(level="DEBUG")
    def _load_csv_robust(self, file_name: str, time_col: str, use_cols: Optional[List[str]] = None, rename_map: Optional[Dict[str, str]] = None, is_forecast_time: bool = False) -> pd.DataFrame:
        cache_key = file_name
        if self._cache_enabled and cache_key in self._raw_data_cache:
            logger.info(f"从缓存加载数据: {file_name}")
            return self._raw_data_cache[cache_key].copy()

        file_path = self.resource_dir / file_name
        logger.info(f"开始加载数据文件: {file_path}")
        # 使用 utils.check_path_exists
        utils.check_path_exists(file_path, path_type='f', raise_error=True)

        try:
            # 可以在此处添加基于 utils.get_file_size 的预检查,如果文件过大则警告或分块读取
            # ...
            df = pd.read_csv(file_path, usecols=use_cols, encoding=const.DEFAULT_ENCODING)
            logger.debug(f"成功读取文件: {file_path}, 形状: {df.shape}")

            if time_col not in df.columns:
                raise DataProcessingError(f"时间列 '{time_col}' 在文件 {file_name} 中未找到")

            # 使用 utils.parse_datetime_flexible 增加解析鲁棒性
            # df[self.internal_time_index] = utils.parse_datetime_flexible(df[time_col], fmts=self.time_format) # 这不适用于 Series
            # 保持之前的 pandas 解析逻辑,但错误处理更清晰
            try:
                df[self.internal_time_index] = pd.to_datetime(df[time_col], format=self.time_format, errors='coerce')
            except ValueError: # If format string is invalid or doesn't match at all
                 logger.warning(f"配置的时间格式 '{self.time_format}' 可能无效或与 '{time_col}' 完全不符尝试自动推断")
                 df[self.internal_time_index] = pd.to_datetime(df[time_col], errors='coerce')
            except Exception as e:
                 logger.error(f"解析时间列 '{time_col}' 时发生未知错误: {e}")
                 raise DataProcessingError(f"无法解析时间列 '{time_col}'") from e

            original_count = len(df)
            df.dropna(subset=[self.internal_time_index], inplace=True)
            removed_count = original_count - len(df)
            if removed_count > 0:
                logger.warning(f"因时间列 '{time_col}' 解析失败或为空,移除了 {removed_count} 行")

            if df.empty:
                raise DataProcessingError(f"处理时间列后 DataFrame 为空: {file_name}")

            if not is_forecast_time:
                if df[self.internal_time_index].duplicated().any():
                    dup_count = df[self.internal_time_index].duplicated().sum()
                    logger.warning(f"文件 {file_name} 中发现 {dup_count} 个重复的时间戳,将保留第一个")
                    df = df[~df[self.internal_time_index].duplicated(keep='first')]

            df.set_index(self.internal_time_index, inplace=True)
            df.sort_index(inplace=True)

            if time_col != self.internal_time_index and time_col in df.columns:
                df.drop(columns=[time_col], inplace=True)

            if rename_map:
                valid_rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
                missing_orig_cols = set(rename_map.keys()) - set(df.columns)
                if missing_orig_cols:
                    logger.warning(f"在文件 {file_name} 中,配置的重命名列不存在: {missing_orig_cols}")
                df.rename(columns=valid_rename_map, inplace=True)
                # logger.debug(f"列已重命名: {valid_rename_map}")
                required_cols_after_rename = list(rename_map.values())
                missing_renamed_cols = set(required_cols_after_rename) - set(df.columns)
                if missing_renamed_cols:
                     raise DataProcessingError(f"文件 {file_name} 重命名后缺少关键列: {missing_renamed_cols}")

            # logger.info(f"完成加载和基础处理: {file_name}, 最终形状: {df.shape}")
            if self._cache_enabled:
                 self._raw_data_cache[cache_key] = df.copy()
            return df

        except FileNotFoundError: raise # 已检查,但以防万一
        except DataProcessingError: raise
        except Exception as e:
            logger.error(f"加载或处理文件 {file_path} 时发生未预料的错误: {e}", exc_info=True)
            raise DataProcessingError(f"加载文件 {file_path} 失败") from e


    # --- _load_and_prepare_all_data ---
    # (保持之前的实现,确保它使用 self.fans_time_col 等实例属性)
    @utils.log_execution_time(level="DEBUG")
    def _load_and_prepare_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("开始加载所有数据...")
        # --- Load Fans ---
        try:
            fans_cols_to_load = [self.fans_time_col, self.fans_power_col]
            fans_rename_map = {self.fans_power_col: self.internal_target_col}
            df_fans = self._load_csv_robust(const.FANS_CSV, self.fans_time_col, fans_cols_to_load, fans_rename_map)
            if self.internal_target_col not in df_fans.columns:
                raise DataProcessingError(f"加载风机数据后缺少目标列: {self.internal_target_col}")
        except FileNotFoundError:
            raise # Fans data is critical
        except DataProcessingError as e:
            raise # Critical error in fans data

        # --- Load Tower ---
        df_tower = pd.DataFrame() # Default to empty
        try:
            tower_cols_to_load = [self.tower_time_col] + list(self.tower_rename_map.keys())
            df_tower = self._load_csv_robust(const.TOWER_CSV, self.tower_time_col, tower_cols_to_load, self.tower_rename_map)
        except FileNotFoundError:
            logger.warning(f"{const.TOWER_CSV} 文件未找到,将不使用测风塔数据")
        except DataProcessingError as e:
            logger.error(f"加载或处理测风塔数据失败: {e}将不使用测风塔数据")

        # --- Load Weather ---
        df_weather = pd.DataFrame() # Default to empty
        try:
            weather_cols_to_load = [self.weather_forecast_time_col, self.weather_issue_time_col] + list(self.weather_rename_map.keys())
            df_weather_raw = self._load_csv_robust(const.WEATHER_CSV, self.weather_forecast_time_col, weather_cols_to_load, self.weather_rename_map, is_forecast_time=True)

            # Process duplicates based on issue time
            issue_time_col_in_df = self.weather_issue_time_col # Assuming it exists after loading
            if issue_time_col_in_df in df_weather_raw.columns:
                logger.info(f"根据发布时间 '{issue_time_col_in_df}' 处理重复预报,保留最新记录")
                try:
                    # Use flexible parsing for issue time as well
                    issue_time_dt = pd.to_datetime(df_weather_raw[issue_time_col_in_df], errors='coerce')
                    if issue_time_dt.isna().any():
                         logger.warning(f"解析天气预报发布时间列 '{issue_time_col_in_df}' 时有失败")
                    df_weather_raw = df_weather_raw.assign(issue_time_dt=issue_time_dt).dropna(subset=['issue_time_dt'])

                except Exception as e:
                     raise DataProcessingError(f"无法解析天气预报发布时间列 '{issue_time_col_in_df}'") from e

                df_weather_raw = df_weather_raw.sort_values(by=[df_weather_raw.index.name, 'issue_time_dt'], ascending=[True, False])
                df_weather = df_weather_raw[~df_weather_raw.index.duplicated(keep='first')]
                df_weather = df_weather.drop(columns=[issue_time_col_in_df, 'issue_time_dt'], errors='ignore')
                # logger.debug(f"天气预报去重完成,保留 {len(df_weather)} 条记录")
            else:
                logger.warning(f"天气预报数据中未找到发布时间列 '{issue_time_col_in_df}',无法按最新发布去重")
                df_weather = df_weather_raw[~df_weather_raw.index.duplicated(keep='first')]
            df_weather.sort_index(inplace=True)
        except FileNotFoundError:
            logger.warning(f"{const.WEATHER_CSV} 文件未找到,将不使用天气预报数据")
        except DataProcessingError as e:
            logger.error(f"加载或处理天气预报数据失败: {e}将不使用天气预报数据")

        logger.info("所有数据加载和初步准备完成")
        return df_fans, df_tower, df_weather

    # --- _resample_and_align ---
    # (保持之前的实现)
    @utils.log_execution_time(level="DEBUG")
    def _resample_and_align(self, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        if df.empty: return df
        if not isinstance(df.index, pd.DatetimeIndex): raise DataProcessingError(f"'{df_name}' 索引类型错误")
        # logger.debug(f"开始重采样数据帧 '{df_name}' 至频率: {self.time_freq}")
        try:
            if not df.index.is_monotonic_increasing: df = df.sort_index()
            if df.index.has_duplicates: df = df[~df.index.duplicated(keep='first')]
            df_resampled = df.resample(self.time_freq).asfreq()
            # logger.debug(f"'{df_name}' 重采样完成")
            return df_resampled
        except ValueError as e:
             raise DataProcessingError(f"'{df_name}' 重采样失败: {e}") from e

    # --- _merge_data_sources ---
    # (保持之前的实现,包含高度调整逻辑)
    @utils.log_execution_time(level="DEBUG")
    def _merge_data_sources(self, df_fans: pd.DataFrame, df_tower: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
        logger.info("开始合并已重采样的数据源 (含高度调整)...")

        df_fans_resampled = self._resample_and_align(df_fans, "Fans")
        df_tower_resampled = self._resample_and_align(df_tower, "Tower")
        df_weather_resampled = self._resample_and_align(df_weather, "Weather")

        if df_fans_resampled.empty:
             raise DataProcessingError("风机数据为空,无法合并")

        # --- 高度调整 ---
        if self.reference_height is not None:
            logger.info(f"将风速调整到参考高度: {self.reference_height}m...")
            # Adjust Tower WS
            if not df_tower_resampled.empty and const.INTERNAL_TOWER_WS_COL in df_tower_resampled.columns:
                if not np.isclose(self.tower_height, self.reference_height):
                     logger.debug(f"调整测风塔风速从 {self.tower_height}m 到 {self.reference_height}m (方法: {self.wind_profile_method})...")
                     adjust_func = wind_utils.adjust_wind_speed_log_law if self.wind_profile_method == 'log_law' else wind_utils.adjust_wind_speed_power_law
                     params = {'roughness_length': self.roughness_length} if self.wind_profile_method == 'log_law' else {'alpha': self.power_law_alpha}
                     try:
                         df_tower_resampled[const.INTERNAL_TOWER_WS_COL] = adjust_func(
                             df_tower_resampled[const.INTERNAL_TOWER_WS_COL], self.tower_height, self.reference_height, **params
                         )
                     except Exception as e:
                          logger.error(f"调整测风塔风速时出错: {e}", exc_info=True)
                          df_tower_resampled[const.INTERNAL_TOWER_WS_COL] = np.nan
                # else: logger.debug(f"测风塔高度 ({self.tower_height}m) 与参考高度相同")
            # Adjust Weather WS
            if not df_weather_resampled.empty and const.INTERNAL_WEATHER_WS_COL in df_weather_resampled.columns:
                if not np.isclose(self.weather_ws_height, self.reference_height):
                    logger.debug(f"调整天气预报风速从 {self.weather_ws_height}m 到 {self.reference_height}m (方法: {self.wind_profile_method})...")
                    adjust_func = wind_utils.adjust_wind_speed_log_law if self.wind_profile_method == 'log_law' else wind_utils.adjust_wind_speed_power_law
                    params = {'roughness_length': self.roughness_length} if self.wind_profile_method == 'log_law' else {'alpha': self.power_law_alpha}
                    try:
                         df_weather_resampled[const.INTERNAL_WEATHER_WS_COL] = adjust_func(
                             df_weather_resampled[const.INTERNAL_WEATHER_WS_COL], self.weather_ws_height, self.reference_height, **params
                         )
                    except Exception as e:
                         logger.error(f"调整天气预报风速时出错: {e}", exc_info=True)
                         df_weather_resampled[const.INTERNAL_WEATHER_WS_COL] = np.nan
                # else: logger.debug(f"天气预报风速高度 ({self.weather_ws_height}m) 与参考高度相同")
        # else: logger.debug("未配置参考高度,跳过风速高度调整")

        # --- 合并 ---
        df_merged = df_fans_resampled
        # 使用 pd.merge 避免潜在的索引不完全匹配问题 (虽然 resample 后理论上匹配)
        if not df_tower_resampled.empty:
            df_merged = pd.merge(df_merged, df_tower_resampled, left_index=True, right_index=True, how='left')
        if not df_weather_resampled.empty:
            df_merged = pd.merge(df_merged, df_weather_resampled, left_index=True, right_index=True, how='left')

        total_cells = df_merged.size
        nan_cells = df_merged.isna().sum().sum()
        nan_ratio = nan_cells / total_cells if total_cells > 0 else 0
        logger.info(f"数据源合并完成,形状: {df_merged.shape},NaN 比例: {nan_ratio:.2%}")
        return df_merged

    # --- _clean_merged_data (深度修订) ---
    @utils.log_execution_time(level="DEBUG")
    def _clean_merged_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """(深度修订) 清洗合并后的数据,严格按顺序执行配置的清洗步骤"""
        logger.info("开始数据清洗流程 (含高级过滤)...")
        df_cleaned = df.copy()
        numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()
        target_col = self.internal_target_col
        if target_col in numeric_cols: numeric_cols.remove(target_col)

        # --- 步骤 1: 类型处理 ---
        logger.debug("步骤 1/8: 数据类型检查与转换...")
        for col in df_cleaned.columns:
            if col == target_col or col in numeric_cols:
                if not pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
            elif col in self.categorical_features:
                 if not pd.api.types.is_categorical_dtype(df_cleaned[col]):
                     try: df_cleaned[col] = df_cleaned[col].astype('category')
                     except Exception: logger.warning(f"无法将列 '{col}' 转换为 category 类型")

        # --- 步骤 2: 处理卡死传感器 (如果启用) ---
        if self.stuck_sensor_window > 0:
            logger.info(f"步骤 2/8: 检测并处理卡死传感器数据 (window={self.stuck_sensor_window})...")
            cols_to_check_stuck = numeric_cols + ([target_col] if target_col in df_cleaned.columns else []) # 检查所有数值列
            for col in cols_to_check_stuck:
                 if col in df_cleaned.columns:
                      try:
                          df_cleaned[col] = wind_utils.filter_stuck_sensor(
                              df_cleaned[col], window=self.stuck_sensor_window,
                              threshold=self.stuck_sensor_threshold, flag_value=np.nan
                          )
                      except Exception as e: logger.error(f"处理列 '{col}' 卡死传感器数据失败: {e}")
        else: logger.debug("步骤 2/8: 跳过卡死传感器检测 (未启用或窗口<=0)")

        # --- 步骤 3: 处理滚动标准差异常值 (如果启用) ---
        if self.std_outlier_window > 0 and self.std_outlier_threshold > 0:
            logger.info(f"步骤 3/8: 检测并处理滚动标准差异常值 (window={self.std_outlier_window}, threshold={self.std_outlier_threshold}σ)...")
            cols_to_check_std = numeric_cols + ([target_col] if target_col in df_cleaned.columns else [])
            for col in cols_to_check_std:
                 if col in df_cleaned.columns:
                      try:
                          df_cleaned[col] = wind_utils.filter_outliers_stddev(
                              df_cleaned[col], window=self.std_outlier_window,
                              std_threshold=self.std_outlier_threshold, flag_value=np.nan
                          )
                      except Exception as e: logger.error(f"处理列 '{col}' 滚动标准差异常值失败: {e}")
        else: logger.debug("步骤 3/8: 跳过滚动标准差异常值检测 (未启用或参数无效)")

        # --- 步骤 4: 插值处理过滤产生的 NaN (除目标列) ---
        logger.info(f"步骤 4/8: 使用 '{self.interpolate_method}' 插值非目标数值列...")
        nan_before_interp = df_cleaned[numeric_cols].isna().sum().sum()
        try:
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].interpolate(
                method=self.interpolate_method, limit_direction='both', limit=self.interpolate_limit
            )
            interpolated_count = nan_before_interp - df_cleaned[numeric_cols].isna().sum().sum()
            logger.info(f"非目标数值列插值完成,填充了 {interpolated_count} 个 NaN")
        except Exception as e:
            logger.error(f"非目标数值列插值失败: {e}")

        # --- 步骤 5: 处理功率曲线异常值 ---
        logger.info("步骤 5/8: 处理功率曲线异常值...")
        if target_col in df_cleaned.columns:
            ws_col_pc = const.INTERNAL_TOWER_WS_COL if const.INTERNAL_TOWER_WS_COL in df_cleaned.columns else const.INTERNAL_WEATHER_WS_COL
            if ws_col_pc in df_cleaned.columns and self.outlier_method_power == "PowerCurve":
                params = self.power_curve_params
                # 先插值目标列,避免因 NaN 导致功率曲线判断失效
                nan_target_before_pc = df_cleaned[target_col].isna().sum()
                if nan_target_before_pc > 0:
                     logger.debug("功率曲线过滤前插值目标列...")
                     df_cleaned[target_col] = df_cleaned[target_col].interpolate(method=self.interpolate_method, limit_direction='both', limit=self.interpolate_limit)

                df_cleaned[target_col] = wind_utils.filter_by_power_curve(
                    power_kw=df_cleaned[target_col], wind_speed_ms=df_cleaned[ws_col_pc],
                    capacity_kw=self.capacity_kw,
                    min_wind_speed=utils.safe_dict_get(params, 'min_wind_speed', 3.0),
                    max_wind_speed=utils.safe_dict_get(params, 'max_wind_speed', 25.0),
                    invalid_power_threshold_kw=utils.safe_dict_get(params, 'invalid_power_threshold_kw', 10.0),
                    overload_ratio=utils.safe_dict_get(params, 'overload_ratio', 1.1)
                )
            elif self.outlier_method_power == "Range":
                 overload_ratio = utils.safe_dict_get(self.power_curve_params, 'overload_ratio', 1.1)
                 upper_limit = self.capacity_kw * overload_ratio
                 logger.info(f"应用基于范围 [0, {upper_limit:.2f}] 的功率过滤...")
                 df_cleaned[target_col] = df_cleaned[target_col].clip(0, upper_limit)
            else: logger.warning(f"未知的功率异常值处理方法: {self.outlier_method_power}")
        else: logger.warning(f"目标列 '{target_col}' 不在数据中,无法处理功率异常值")

        # --- 步骤 6: 再次插值目标列 (处理功率曲线过滤可能产生的 NaN) ---
        logger.debug("步骤 6/8: 再次插值目标列...")
        if target_col in df_cleaned.columns:
             nan_target_after_pc = df_cleaned[target_col].isna().sum()
             if nan_target_after_pc > 0:
                  df_cleaned[target_col] = df_cleaned[target_col].interpolate(method=self.interpolate_method, limit_direction='both', limit=self.interpolate_limit)
                  logger.debug(f"再次插值目标列,填充了 {nan_target_after_pc - df_cleaned[target_col].isna().sum()} 个 NaN")

        # --- 步骤 7: 处理风速范围异常值 ---
        logger.info("步骤 7/8: 处理风速范围异常值...")
        if self.outlier_method_ws == "Range":
             params = self.ws_range_params
             min_ws = utils.safe_dict_get(params, 'min_value', 0.0)
             max_ws = utils.safe_dict_get(params, 'max_value', 50.0)
             logger.debug(f"应用风速范围裁剪 [{min_ws}, {max_ws}]...")
             for col in [const.INTERNAL_TOWER_WS_COL, const.INTERNAL_WEATHER_WS_COL]:
                 if col in df_cleaned.columns: df_cleaned[col] = df_cleaned[col].clip(min_ws, max_ws)
        else: logger.debug(f"未配置或不支持的风速范围过滤方法: {self.outlier_method_ws}")

        # --- 步骤 8: 检测结冰条件 (如果启用,添加为特征) ---
        if self.enable_icing_detection:
             logger.info("步骤 8/8: 检测结冰条件...")
             temp_col = const.INTERNAL_TOWER_TEMP_COL if const.INTERNAL_TOWER_TEMP_COL in df_cleaned.columns else const.INTERNAL_WEATHER_TEMP_COL
             rh_col = const.INTERNAL_TOWER_HUM_COL if const.INTERNAL_TOWER_HUM_COL in df_cleaned.columns else const.INTERNAL_WEATHER_HUM_COL
             power_col = target_col if target_col in df_cleaned.columns else None
             predicted_power = None # 需要方法来获取预测功率才能进行偏差判断

             if temp_col in df_cleaned.columns:
                 try:
                     icing_mask = wind_utils.detect_icing_conditions(
                         temperature_c=df_cleaned[temp_col],
                         relative_humidity=df_cleaned.get(rh_col),
                         power_kw=df_cleaned.get(power_col),
                         predicted_power_kw=predicted_power, # 缺少预测值
                         **self.icing_params # 使用配置中的参数
                     )
                     df_cleaned['icing_condition_flag'] = icing_mask.astype(int)
                     logger.info("结冰条件检测完成,已添加 'icing_condition_flag' 特征")
                 except Exception as e: logger.error(f"检测结冰条件时失败: {e}")
             else: logger.warning("无法检测结冰条件,缺少温度列")
        else: logger.debug("步骤 8/8: 跳过结冰条件检测 (未启用)")

        # --- 最终 NaN 填充 ---
        # 清洗步骤完成后,理论上只有插值无法覆盖的开头/结尾可能有 NaN
        logger.info(f"开始最终 NaN 填充 (策略: {self.final_nan_fill_strategy})...")
        nan_counts = df_cleaned.isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if not nan_cols.empty:
             logger.warning(f"清洗后,在最终填充前,以下列仍包含 NaN:\n{nan_cols}")
             if self.final_nan_fill_strategy == 'ffill_bfill_zero':
                 df_cleaned.fillna(method='ffill', inplace=True)
                 df_cleaned.fillna(method='bfill', inplace=True)
                 df_cleaned.fillna(0, inplace=True)
                 logger.info("使用 ffill -> bfill -> 0 策略填充了剩余 NaN")
             elif self.final_nan_fill_strategy == 'zero':
                 df_cleaned.fillna(0, inplace=True)
                 logger.info("使用 0 策略填充了剩余 NaN")
             else: # 默认或未知策略,用 0 填充并警告
                 logger.warning(f"未知的最终 NaN 填充策略 '{self.final_nan_fill_strategy}',将使用 0 填充")
                 df_cleaned.fillna(0, inplace=True)
        else:
            logger.info("数据清洗后未发现需要最终填充的 NaN 值")

        logger.info("数据清洗流程完成")
        return df_cleaned

    # --- _engineer_features (深度修订) ---
    @utils.log_execution_time(level="DEBUG")
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """(深度修订) 执行特征工程,严格按照配置并调用相关工具函数"""
        logger.info("开始执行特征工程...")
        df_featured = df.copy()
        idx = df_featured.index

        # 1. 时间特征 (标准 + 周期)
        logger.debug("创建时间特征...")
        df_featured['hour'] = idx.hour; df_featured['dayofweek'] = idx.dayofweek
        df_featured['dayofyear'] = idx.dayofyear; df_featured['month'] = idx.month
        df_featured['weekofyear'] = idx.isocalendar().week.astype(int); df_featured['quarter'] = idx.quarter
        df_featured['year'] = idx.year
        # Sin/Cos
        df_featured['hour_sin'] = np.sin(2 * np.pi * df_featured['hour'] / 24.0); df_featured['hour_cos'] = np.cos(2 * np.pi * df_featured['hour'] / 24.0)
        df_featured['dayofweek_sin'] = np.sin(2 * np.pi * df_featured['dayofweek'] / 7.0); df_featured['dayofweek_cos'] = np.cos(2 * np.pi * df_featured['dayofweek'] / 7.0)
        df_featured['month_sin'] = np.sin(2 * np.pi * df_featured['month'] / 12.0); df_featured['month_cos'] = np.cos(2 * np.pi * df_featured['month'] / 12.0)

        # 2. 滞后特征 (根据 self.lag_config 和 self.lag_base_cols)
        logger.info(f"创建滞后特征 (Lags: {self.lag_config})...")
        if isinstance(self.lag_config, list) and self.lag_config:
            cols_to_lag = [col for col in self.lag_base_cols if col in df_featured.columns]
            logger.debug(f"将为以下列创建滞后特征: {cols_to_lag}")
            for col in cols_to_lag:
                for lag in self.lag_config:
                    if isinstance(lag, int) and lag > 0:
                        lag_col_name = f"{col}_lag{lag}"
                        df_featured[lag_col_name] = df_featured[col].shift(lag)
        else: logger.debug("未配置或配置无效,跳过滞后特征创建")

        # 3. 滚动窗口特征 (根据 self.rolling_config)
        logger.info(f"创建滚动窗口特征 (Config: {len(self.rolling_config)} 个列)...")
        if isinstance(self.rolling_config, dict):
            for col, config in self.rolling_config.items():
                if col in df_featured.columns and isinstance(config, dict):
                    windows = config.get('windows', []); aggs = config.get('aggs', [])
                    if not isinstance(windows, list) or not isinstance(aggs, list): continue
                    for window in windows:
                        if isinstance(window, int) and window > 1:
                            min_periods = max(1, window // 2) # 默认需要半个窗口数据
                            for agg in aggs:
                                roll_col_name = f"{col}_roll{window}_{agg}"
                                try:
                                    rolling_obj = df_featured[col].rolling(window=window, min_periods=min_periods)
                                    if hasattr(rolling_obj, agg):
                                        # 处理 'std' 可能产生的警告
                                        if agg == 'std':
                                             with warnings.catch_warnings():
                                                  warnings.simplefilter("ignore", category=RuntimeWarning)
                                                  df_featured[roll_col_name] = getattr(rolling_obj, agg)(ddof=1) # 样本标准差
                                        else:
                                             df_featured[roll_col_name] = getattr(rolling_obj, agg)()
                                    # else: logger.warning(f"无效聚合 '{agg}'")
                                except Exception as e: logger.warning(f"创建滚动特征 {roll_col_name} 失败: {e}")
                        # else: logger.warning(f"无效窗口 '{window}'")
        else: logger.warning(f"配置 'Feature.TimeWindowRolling' 格式不支持")

        # 4. 交互特征 (根据 self.interaction_config)
        logger.debug("创建交互特征...")
        if 'WindSpeedPower3' in self.interaction_config:
            for col in [const.INTERNAL_TOWER_WS_COL, const.INTERNAL_WEATHER_WS_COL]:
                if col in df_featured.columns: df_featured[f"{col}_pow3"] = df_featured[col] ** 3
        if 'AirDensity' in self.interaction_config or 'PowerDensity' in self.interaction_config:
            temp_col = const.INTERNAL_TOWER_TEMP_COL if const.INTERNAL_TOWER_TEMP_COL in df_featured.columns else const.INTERNAL_WEATHER_TEMP_COL
            press_col = const.INTERNAL_TOWER_PRESS_COL if const.INTERNAL_TOWER_PRESS_COL in df_featured.columns else const.INTERNAL_WEATHER_PRESS_COL
            if temp_col in df_featured.columns and press_col in df_featured.columns:
                 pressure_pa = df_featured[press_col] * 100 # 假设输入是hPa
                 try:
                     df_featured['AirDensity'] = wind_utils.calculate_air_density(df_featured[temp_col], pressure_pa)
                     logger.debug("交互特征 'AirDensity' 创建成功")
                     if 'PowerDensity' in self.interaction_config:
                          ws_col_pd = const.INTERNAL_TOWER_WS_COL if const.INTERNAL_TOWER_WS_COL in df_featured.columns else const.INTERNAL_WEATHER_WS_COL
                          if ws_col_pd in df_featured.columns:
                               df_featured['PowerDensity'] = wind_utils.calculate_air_energy_density(df_featured['AirDensity'], df_featured[ws_col_pd])
                               logger.debug("交互特征 'PowerDensity' 创建成功")
                 except Exception as e: logger.warning(f"创建空气/能量密度特征时出错: {e}")

        # 5. 风向向量化 (调用 wind_utils.degrees_to_vector)
        logger.debug("处理风向特征...")
        # 动态查找风向列的内部名称
        wd_cols_internal = [v for k, v in self.tower_rename_map.items() if '风向' in k] + \
                           [v for k, v in self.weather_rename_map.items() if '风向' in k]
        for col in wd_cols_internal:
             if col in df_featured.columns:
                 logger.debug(f"将风向列 '{col}' 转换为向量...")
                 try:
                     wd_vectors = wind_utils.degrees_to_vector(df_featured[col])
                     vector_rename_map = {'WD_X': f"{col}_X", 'WD_Y': f"{col}_Y"}
                     df_featured = pd.concat([df_featured, wd_vectors.rename(columns=vector_rename_map)], axis=1)
                     if self.drop_orig_wd: # 根据配置删除列
                          df_featured.drop(columns=[col], inplace=True, errors='ignore')
                 except Exception as e: logger.warning(f"转换风向列 '{col}' 失败: {e}")

        # 6. 删除忽略列
        logger.debug("删除忽略列...")
        actual_cols_to_drop = [c for c in self.ignore_features if c in df_featured.columns]
        if self.internal_target_col in actual_cols_to_drop: actual_cols_to_drop.remove(self.internal_target_col)
        if actual_cols_to_drop:
            logger.info(f"根据配置删除以下列: {actual_cols_to_drop}")
            df_featured.drop(columns=actual_cols_to_drop, inplace=True)

        # 7. 处理特征工程引入的 NaN (使用最终策略)
        logger.info(f"处理特征工程引入的 NaN (策略: {self.final_nan_fill_strategy})...")
        nan_counts_before_final = df_featured.isna().sum()
        nan_cols_before_final = nan_counts_before_final[nan_counts_before_final > 0]
        if not nan_cols_before_final.empty:
            logger.warning(f"特征工程后,在最终填充前,以下列包含 NaN:\n{nan_cols_before_final}")
            if self.final_nan_fill_strategy == 'ffill_bfill_zero':
                df_featured.fillna(method='ffill', inplace=True)
                df_featured.fillna(method='bfill', inplace=True)
                df_featured.fillna(0, inplace=True)
            elif self.final_nan_fill_strategy == 'zero':
                df_featured.fillna(0, inplace=True)
            else: # 默认或未知策略
                logger.warning(f"未知的最终 NaN 填充策略 '{self.final_nan_fill_strategy}',将使用 0 填充")
                df_featured.fillna(0, inplace=True)
            logger.info(f"特征工程 NaN 填充完成")
        else:
            logger.info("特征工程后未发现 NaN 值")

        # 8. (可选) 数据类型降级以节省内存
        if utils.safe_dict_get(self.config, 'Resource.DowncastDataTypes', False):
             df_featured = utils.downcast_dataframe_dtypes(df_featured, verbose=True)

        logger.info(f"特征工程完成,最终 DataFrame 形状: {df_featured.shape}")
        return df_featured


    # --- process 方法 ---
    # (保持不变,调用重写后的内部方法)
    @utils.log_execution_time(level="INFO") # 对整个流程计时
    def process(self, start_time: Optional[Union[str, pd.Timestamp]] = None, end_time: Optional[Union[str, pd.Timestamp]] = None) -> pd.DataFrame:
        """
        执行完整的数据处理流程加载 -> 合并 -> 清洗 -> 特征工程 -> 时间筛选
        """
        logger.info("启动完整数据处理流程...")
        logger.info(f"请求处理时间范围: [{start_time or '数据起点'} - {end_time or '数据终点'}]")
        try:
            # 使用 TimerContext 细分计时
            with utils.TimerContext("数据加载与准备", logger_instance=logger, level="INFO"):
                df_fans, df_tower, df_weather = self._load_and_prepare_all_data()
            with utils.TimerContext("数据合并与对齐 (含高度调整)", logger_instance=logger, level="INFO"):
                df_merged = self._merge_data_sources(df_fans, df_tower, df_weather)
            with utils.TimerContext("数据清洗 (含高级过滤)", logger_instance=logger, level="INFO"):
                df_cleaned = self._clean_merged_data(df_merged)
            with utils.TimerContext("特征工程", logger_instance=logger, level="INFO"):
                df_processed = self._engineer_features(df_cleaned)

            # 时间范围筛选
            initial_rows = len(df_processed)
            if start_time:
                df_processed = df_processed[df_processed.index >= pd.to_datetime(start_time)]
            if end_time:
                df_processed = df_processed[df_processed.index <= pd.to_datetime(end_time)]
            if len(df_processed) < initial_rows:
                 logger.info(f"时间筛选后,数据从 {initial_rows} 行减少到 {len(df_processed)} 行")

            # 最终校验
            if df_processed.empty: raise DataProcessingError("处理结果为空 DataFrame")
            if self.internal_target_col not in df_processed.columns: raise DataProcessingError(f"目标列 '{self.internal_target_col}' 处理后丢失")
            if df_processed.index.name != self.internal_time_index: df_processed.index.name = self.internal_time_index
            if df_processed.isna().any().any():
                 final_nan_counts = df_processed.isna().sum(); logger.error(f"严重警告最终结果包含 NaN!\n{final_nan_counts[final_nan_counts > 0]}")
                 # 生产环境可能需要报错: raise DataProcessingError("最终处理结果包含 NaN 值")

            final_start = df_processed.index.min(); final_end = df_processed.index.max()
            logger.success(f"数据处理流程成功完成最终数据范围: [{final_start} - {final_end}], 形状: {df_processed.shape}")
            return df_processed

        except (FileNotFoundError, ConfigError, DataProcessingError, Exception) as e:
            logger.critical(f"数据处理流程因错误而终止: {e}", exc_info=True)
            if isinstance(e, (FileNotFoundError, ConfigError, DataProcessingError)): raise
            else: raise DataProcessingError("数据处理流程中发生未捕获的严重错误") from e