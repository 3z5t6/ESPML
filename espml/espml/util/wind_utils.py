# -*- coding: utf-8 -*-
"""
风电项目特定工具函数模块 (espml)
包含风电相关的物理计算、数据转换、质量控制和风资源评估函数
"""

import math
import warnings
from typing import Tuple, Union, Optional, List, Dict, Any

import numpy as np
import pandas as pd
from loguru import logger
# 可能需要的额外库 (根据具体实现推断)
# from scipy import stats as sp_stats
# from scipy.optimize import curve_fit
# import pyproj # 如果需要坐标转换

# 导入本项目定义的常量
from espml.util import const


# --- 内部辅助函数 ---

def _safe_divide(numerator: pd.Series, denominator: pd.Series, default: float = np.nan) -> pd.Series:
    """安全除法,处理分母为零或 NaN 的情况"""
    with np.errstate(divide='ignore', invalid='ignore'): # 忽略除零和无效值警告
        result = numerator / denominator
    result[~np.isfinite(result)] = default # 将 inf, -inf, nan 替换为默认值
    # 或者更精细地处理
    # result = np.where(np.isclose(denominator, 0) | denominator.isna() | numerator.isna(), default, numerator / denominator)
    return pd.Series(result, index=numerator.index)

def _validate_series_input(*series: pd.Series, allow_empty: bool = False) -> None:
    """验证输入是否为 Pandas Series 且（可选）非空"""
    for i, s in enumerate(series):
        if not isinstance(s, pd.Series):
            raise TypeError(f"输入参数 {i+1} 必须是 pandas Series,但得到了 {type(s)}")
        if not allow_empty and s.empty:
             raise ValueError(f"输入 Series {i+1} 不能为空")

def _calculate_rolling_std(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    """计算滚动标准差,处理潜在的 ddof 问题和警告"""
    if min_periods is None:
        min_periods = max(1, window // 2) # 默认至少需要半个窗口
    # 使用 Pandas 内建函数,注意 ddof=1 (样本标准差) 是默认值
    # 忽略 RuntimeWarning: Degrees of freedom <= 0 for slice
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        rolling_std = series.rolling(window=window, min_periods=min_periods, center=False).std(ddof=1)
    return rolling_std


# --- 坐标与地理计算 (推断) ---

# def transform_coordinates(source_crs: str, target_crs: str, x_coords: Union[pd.Series, List[float]], y_coords: Union[pd.Series, List[float]]) -> Tuple[pd.Series, pd.Series]:
#     """
#     使用 pyproj 进行坐标转换
#
#     Args:
#         source_crs (str): 源坐标系统 (例如 'epsg:4326' for WGS84 Lon/Lat)
#         target_crs (str): 目标坐标系统 (例如 'epsg:32632' for UTM Zone 32N)
#         x_coords (Union[pd.Series, List[float]]): 源坐标 X 或经度
#         y_coords (Union[pd.Series, List[float]]): 源坐标 Y 或纬度
#
#     Returns:
#         Tuple[pd.Series, pd.Series]: 包含目标坐标 X 和 Y 的 Series 元组
#     """
#     try:
#         transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
#     except Exception as e:
#         logger.error(f"无法创建坐标转换器从 '{source_crs}' 到 '{target_crs}': {e}")
#         raise ValueError("无效的坐标参考系统") from e
#
#     if isinstance(x_coords, pd.Series):
#         x_in = x_coords.values
#         y_in = y_coords.values
#         index = x_coords.index
#     else:
#         x_in = np.array(x_coords)
#         y_in = np.array(y_coords)
#         index = None # 如果输入是列表,则没有索引
#
#     target_x, target_y = transformer.transform(x_in, y_in)
#
#     if index is not None:
#         return pd.Series(target_x, index=index), pd.Series(target_y, index=index)
#     else:
#          # 如果输入是列表,返回 Numpy 数组可能更合适,或者根据需要调整
#          return pd.Series(target_x), pd.Series(target_y)


# --- 风向处理 (确认存在) ---
def degrees_to_vector(degrees: pd.Series) -> pd.DataFrame:
    """
    将风向角度 (气象学定义) 转换为标准的数学单位圆向量 (x, y 分量)
    (实现同修正后的 Batch 2 - Step 1)
    """
    _validate_series_input(degrees, allow_empty=True)
    # logger.debug(f"开始转换风向角度 (共 {len(degrees)} 个) 为向量分量...")
    original_nan_count = degrees.isna().sum()
    degrees_processed = degrees.copy()
    valid_mask = degrees_processed.notna()
    degrees_processed[valid_mask] = degrees_processed[valid_mask] % 360
    radians = np.deg2rad((450 - degrees_processed[valid_mask]) % 360)
    wd_x = pd.Series(np.nan, index=degrees.index, dtype=float)
    wd_y = pd.Series(np.nan, index=degrees.index, dtype=float)
    wd_x[valid_mask] = np.cos(radians)
    wd_y[valid_mask] = np.sin(radians)
    result_df = pd.DataFrame({'WD_X': wd_x, 'WD_Y': wd_y}, index=degrees.index)
    final_nan_count_x = result_df['WD_X'].isna().sum()
    if final_nan_count_x != original_nan_count:
         logger.warning(f"风向转向量后 NaN 数量发生变化 (: {original_nan_count}, 结果: {final_nan_count_x}),请检查数据质量")
    # logger.debug("风向角度到向量分量转换完成")
    return result_df


# --- 大气物理与热力学计算 ---

def calculate_air_density(temperature_c: pd.Series, pressure_pa: pd.Series) -> pd.Series:
    """
    根据温度和绝对压力估算空气密度
    (实现同修正后的 Batch 2 - Step 1)
    """
    _validate_series_input(temperature_c, pressure_pa, allow_empty=True)
    # logger.debug(f"开始计算空气密度 (共 {len(temperature_c)} 个点)...")
    temp_c = temperature_c.copy()
    press_pa = pressure_pa.copy()
    invalid_temp_mask = temp_c < (const.ABSOLUTE_ZERO_CELSIUS - 1)
    if invalid_temp_mask.any():
        logger.warning(f"温度数据中发现 {invalid_temp_mask.sum()} 个低于绝对零度的值,将设为 NaN 处理")
        temp_c[invalid_temp_mask] = np.nan
    invalid_press_mask = press_pa < 0
    if invalid_press_mask.any():
        logger.warning(f"压力数据中发现 {invalid_press_mask.sum()} 个负值,将设为 NaN 处理")
        press_pa[invalid_press_mask] = np.nan
    temperature_k = temp_c + abs(const.ABSOLUTE_ZERO_CELSIUS)
    density = pd.Series(np.nan, index=temperature_c.index, dtype=float)
    valid_input_mask = temperature_k.notna() & press_pa.notna() & (temperature_k > 1.0)
    if valid_input_mask.any():
        try:
            density[valid_input_mask] = press_pa[valid_input_mask] / (const.GAS_CONSTANT_DRY_AIR * temperature_k[valid_input_mask])
        except Exception as e:
            logger.error(f"计算空气密度时发生错误: {e}", exc_info=True)
    density_clipped = density.clip(lower=0.8, upper=1.5)
    clipped_count = (density.dropna() != density_clipped.dropna()).sum()
    if clipped_count > 0 :
         logger.warning(f"计算出的空气密度中有 {clipped_count} 个值被限制在 [0.8, 1.5] 范围内")
    # logger.debug("空气密度计算完成")
    return density_clipped

def calculate_potential_temperature(temperature_c: pd.Series, pressure_pa: pd.Series, p0: float = 100000.0) -> pd.Series:
    """
    计算位温 (Potential Temperature)
    公式: theta = T * (p0 / p)^(R_d / c_p)
    其中 R_d 是干空气气体常数, c_p 是干空气定压比热 (约 1005 J/(kg·K))
    R_d / c_p 约等于 0.286

    Args:
        temperature_c (pd.Series): 温度 (摄氏度)
        pressure_pa (pd.Series): 气压 (帕斯卡)
        p0 (float): 参考气压 (通常为 100000 Pa)

    Returns:
        pd.Series: 位温 (开尔文 K)
    """
    _validate_series_input(temperature_c, pressure_pa, allow_empty=True)
    logger.debug("开始计算位温...")
    # 转换单位
    temperature_k = temperature_c + abs(const.ABSOLUTE_ZERO_CELSIUS)
    kappa = const.GAS_CONSTANT_DRY_AIR / 1005.0 # R_d / c_p

    # 计算位温
    # 使用安全除法处理潜在的 p=0 情况
    pressure_ratio = p0 / pressure_pa.clip(lower=1.0) # 限制最低压力防止除零或负数
    potential_temp_k = temperature_k * (pressure_ratio ** kappa)

    logger.debug("位温计算完成")
    return potential_temp_k

def calculate_saturation_vapor_pressure(temperature_c: pd.Series, method: str = 'Magnus') -> pd.Series:
    """
    计算饱和水汽压 (Saturation Vapor Pressure, es)

    Args:
        temperature_c (pd.Series): 温度 (摄氏度)
        method (str): 计算公式选择 ('Magnus' 或 'Bolton')Magnus-Tetens 公式较常用

    Returns:
        pd.Series: 饱和水汽压 (单位: 帕斯卡 Pa)
    """
    _validate_series_input(temperature_c, allow_empty=True)
    logger.debug(f"开始使用 '{method}' 方法计算饱和水汽压...")
    T = temperature_c.copy() # 摄氏度

    if method.lower() == 'magnus':
        # Magnus-Tetens approximation (适用于水面)
        # 参数来源: Alduchov, O.A. and R.E. Eskridge, 1996:
        # Improved Magnus Form Approximation of Saturation Vapor Pressure. J. Appl. Meteor., 35, 601–609.
        # es(T) = a1 * exp((a3*T)/(a4+T))  (T in Celsius, es in hPa)
        a1 = 6.1094 # hPa
        a3 = 17.625
        a4 = 243.04 # °C
        es_hpa = a1 * np.exp((a3 * T) / (a4 + T))
        es_pa = es_hpa * 100 # 转换为 Pa
    elif method.lower() == 'bolton':
        # Bolton (1980) formula, J. Appl. Meteor., 19, 1097-1101
        # es(T) = 6.112 * exp((17.67 * T) / (T + 243.5)) (T in Celsius, es in hPa)
        es_hpa = 6.112 * np.exp((17.67 * T) / (T + 243.5))
        es_pa = es_hpa * 100 # 转换为 Pa
    else:
        raise ValueError(f"不支持的饱和水汽压计算方法: {method}")

    # 对结果进行合理性检查 (水汽压不能为负)
    es_pa = es_pa.clip(lower=0)
    logger.debug("饱和水汽压计算完成")
    return es_pa

def calculate_actual_vapor_pressure(relative_humidity: pd.Series, saturation_vapor_pressure_pa: pd.Series) -> pd.Series:
    """
    根据相对湿度和饱和水汽压计算实际水汽压 (Actual Vapor Pressure, e)

    公式: e = (RH / 100) * es

    Args:
        relative_humidity (pd.Series): 相对湿度 (百分比, 0-100)
        saturation_vapor_pressure_pa (pd.Series): 饱和水汽压 (帕斯卡 Pa)

    Returns:
        pd.Series: 实际水汽压 (帕斯卡 Pa)
    """
    _validate_series_input(relative_humidity, saturation_vapor_pressure_pa, allow_empty=True)
    logger.debug("开始计算实际水汽压...")
    rh = relative_humidity.clip(0, 100) # 限制 RH 在 0-100
    es = saturation_vapor_pressure_pa.clip(lower=0) # 确保 es >= 0

    e_pa = (rh / 100.0) * es
    logger.debug("实际水汽压计算完成")
    return e_pa

def calculate_dew_point(actual_vapor_pressure_pa: pd.Series, method: str = 'Magnus') -> pd.Series:
    """
    根据实际水汽压反算露点温度 (Dew Point Temperature, Td)

    使用 Magnus 或 Bolton 公式的逆运算

    Args:
        actual_vapor_pressure_pa (pd.Series): 实际水汽压 (帕斯卡 Pa)
        method (str): 计算公式选择 ('Magnus' 或 'Bolton')应与计算 es 时使用的方法一致

    Returns:
        pd.Series: 露点温度 (摄氏度 °C)
    """
    _validate_series_input(actual_vapor_pressure_pa, allow_empty=True)
    logger.debug(f"开始使用 '{method}' 方法反算露点温度...")
    e_pa = actual_vapor_pressure_pa.clip(lower=0.1) # 避免 log(0),限制最低水汽压
    e_hpa = e_pa / 100.0 # 转换为 hPa

    if method.lower() == 'magnus':
        # Inverse Magnus-Tetens formula
        a1 = 6.1094
        a3 = 17.625
        a4 = 243.04
        # Td = (a4 * ln(e/a1)) / (a3 - ln(e/a1))
        log_term = np.log(e_hpa / a1)
        dew_point_c = (a4 * log_term) / (a3 - log_term)
    elif method.lower() == 'bolton':
        # Inverse Bolton formula
        # Td = (243.5 * ln(e/6.112)) / (17.67 - ln(e/6.112))
        log_term = np.log(e_hpa / 6.112)
        dew_point_c = (243.5 * log_term) / (17.67 - log_term)
    else:
        raise ValueError(f"不支持的露点温度反算方法: {method}")

    # 露点温度理论上不应高于当前温度,但这里不强制检查
    logger.debug("露点温度计算完成")
    return dew_point_c

def calculate_specific_humidity(pressure_pa: pd.Series, actual_vapor_pressure_pa: pd.Series) -> pd.Series:
    """
    计算比湿 (Specific Humidity, q)
    公式: q = (epsilon * e) / (p - (1 - epsilon) * e)
    其中 epsilon = R_d / R_v ≈ 0.622 (干空气与水汽的气体常数比)

    Args:
        pressure_pa (pd.Series): 总大气压 (帕斯卡 Pa)
        actual_vapor_pressure_pa (pd.Series): 实际水汽压 (帕斯卡 Pa)

    Returns:
        pd.Series: 比湿 (kg/kg)
    """
    _validate_series_input(pressure_pa, actual_vapor_pressure_pa, allow_empty=True)
    logger.debug("开始计算比湿...")
    epsilon = 0.622
    e_pa = actual_vapor_pressure_pa.clip(lower=0)
    p_pa = pressure_pa.clip(lower=e_pa + 1) # 确保 p > e

    # 使用安全除法处理分母
    denominator = p_pa - (1 - epsilon) * e_pa
    q = _safe_divide(epsilon * e_pa, denominator, default=0.0) # 比湿通常非负

    # 对结果进行合理性限制 [0, 0.05] (地球上常见范围)
    q = q.clip(0, 0.05)
    logger.debug("比湿计算完成")
    return q


# --- 风垂直廓线与湍流 (推断) ---

def adjust_wind_speed_power_law(
    wind_speed: Union[pd.Series, float],
    current_height: float,
    target_height: float,
    alpha: float = 0.14
    ) -> Union[pd.Series, float]:
    """
    使用风功率指数律 (Power Law) 根据风切变指数将风速从当前高度推算到目标高度

    公式: V(z2) = V(z1) * (z2 / z1) ^ alpha

    Args:
        wind_speed (Union[pd.Series, float]): 当前高度下的风速 (m/s)
        current_height (float): 当前风速测量的高度 (米)
        target_height (float): 需要推算风速的目标高度 (米)
        alpha (float): 风切变指数典型值范围 0.1 (平坦开阔地) 到 0.4 (复杂地形/城市)
                         默认为 0.14 (常用于近中性大气条件下的陆上)

    Returns:
        Union[pd.Series, float]: 目标高度处的推算风速 (m/s)

    Raises:
        ValueError: 如果高度或 alpha 参数无效
    """
    if not isinstance(current_height, (int, float)) or current_height <= 0:
        raise ValueError(f"当前高度 'current_height' 必须是正数,但得到 {current_height}")
    if not isinstance(target_height, (int, float)) or target_height <= 0:
        raise ValueError(f"目标高度 'target_height' 必须是正数,但得到 {target_height}")
    if not isinstance(alpha, (int, float)) or alpha < 0: # alpha 通常 >= 0
        raise ValueError(f"风切变指数 'alpha' 不能为负数,但得到 {alpha}")
    if current_height == target_height:
        logger.debug("当前高度与目标高度相同 (Power Law),无需调整风速")
        return wind_speed

    # logger.debug(f"开始使用 Power Law (alpha={alpha}) 调整风速从 {current_height}m 到 {target_height}m...")

    try:
        height_ratio = target_height / current_height
        adjustment_factor = height_ratio ** alpha
        adjusted_speed = wind_speed * adjustment_factor

        # 对结果进行合理性限制 (风速不能为负)
        if isinstance(adjusted_speed, pd.Series):
            adjusted_speed = adjusted_speed.clip(lower=0)
        else:
            adjusted_speed = max(0, adjusted_speed)

        # logger.debug("Power Law 风速高度调整完成")
        return adjusted_speed

    except Exception as e:
         logger.error(f"使用 Power Law 调整风速时发生错误: {e}", exc_info=True)
         if isinstance(wind_speed, pd.Series):
              return pd.Series(np.nan, index=wind_speed.index)
         else:
              return np.nan

def adjust_wind_speed_log_law(
    wind_speed: Union[pd.Series, float],
    current_height: float,
    target_height: float,
    roughness_length: float = 0.03
    ) -> Union[pd.Series, float]:
    """
    使用对数律根据地面粗糙度长度将风速从当前高度推算到目标高度
    (实现同修正后的 Batch 2 - Step 1)
    """
    if not isinstance(current_height, (int, float)) or current_height <= 0:
        raise ValueError(f"当前高度 'current_height' 必须是正数,但得到 {current_height}")
    if not isinstance(target_height, (int, float)) or target_height <= 0:
        raise ValueError(f"目标高度 'target_height' 必须是正数,但得到 {target_height}")
    if not isinstance(roughness_length, (int, float)) or roughness_length <= 0:
        raise ValueError(f"粗糙度长度 'roughness_length' 必须是正数,但得到 {roughness_length}")
    if current_height == target_height:
        # logger.debug("当前高度与目标高度相同 (Log Law),无需调整风速")
        return wind_speed
    # logger.debug(f"开始使用 Log Law (z0={roughness_length}m) 调整风速从 {current_height}m 到 {target_height}m...")
    try:
        log_term_target = np.log(target_height / roughness_length)
        log_term_current = np.log(current_height / roughness_length)
        if np.isclose(log_term_current, 0):
             logger.warning(f"计算风速高度调整(Log Law)时,ln(z1/z0) 接近 0,无法计算将返回 NaN")
             return pd.Series(np.nan, index=wind_speed.index) if isinstance(wind_speed, pd.Series) else np.nan
        adjustment_factor = log_term_target / log_term_current
        adjusted_speed = wind_speed * adjustment_factor
        if isinstance(adjusted_speed, pd.Series):
            adjusted_speed = adjusted_speed.clip(lower=0)
        else:
            adjusted_speed = max(0, adjusted_speed)
        # logger.debug("Log Law 风速高度调整完成")
        return adjusted_speed
    except Exception as e:
         logger.error(f"使用 Log Law 调整风速时发生错误: {e}", exc_info=True)
         return pd.Series(np.nan, index=wind_speed.index) if isinstance(wind_speed, pd.Series) else np.nan


def calculate_turbulence_intensity(wind_speed_series: pd.Series, window_size: Union[int, str] = '10min', center: bool = False) -> pd.Series:
    """
    计算湍流强度 (Turbulence Intensity, TI)

    TI = (滚动时间窗口内的风速标准差) / (滚动时间窗口内的平均风速)

    Args:
        wind_speed_series (pd.Series): 风速时间序列 (m/s)索引必须是 DatetimeIndex
        window_size (Union[int, str]): 滚动窗口的大小
                                       如果是整数,表示观测点的数量
                                       如果是字符串 (例如 '10min', '1H'),表示时间窗口大小 (需要 DatetimeIndex)
        center (bool): 滚动窗口是否居中False 表示窗口向后看

    Returns:
        pd.Series: 计算得到的湍流强度序列 (无量纲)

    Raises:
        TypeError: 如果输入不是 Series 或索引不是 DatetimeIndex (当 window_size 是 str 时)
        ValueError: 如果窗口大小无效
    """
    _validate_series_input(wind_speed_series)
    if isinstance(window_size, str):
        if not isinstance(wind_speed_series.index, pd.DatetimeIndex):
            raise TypeError("当 window_size 是时间字符串时,Series 索引必须是 DatetimeIndex")
        # 检查时间窗口字符串是否有效
        try:
             pd.Timedelta(window_size)
        except ValueError:
             raise ValueError(f"无效的时间窗口字符串: {window_size}")
    elif isinstance(window_size, int):
         if window_size <= 1:
              raise ValueError(f"窗口大小 (点数) 必须大于 1,但得到 {window_size}")
    else:
         raise TypeError(f"window_size 必须是整数或时间字符串,但得到 {type(window_size)}")

    logger.debug(f"开始计算湍流强度,窗口大小: {window_size}...")

    # 计算滚动标准差和滚动平均值
    rolling_std = _calculate_rolling_std(wind_speed_series, window=window_size, min_periods=2) # TI 至少需要2个点
    rolling_mean = wind_speed_series.rolling(window=window_size, min_periods=1, center=center).mean()

    # 计算 TI
    ti = _safe_divide(rolling_std, rolling_mean, default=np.nan) # 平均风速为0时 TI 无定义

    # 对 TI 进行合理性限制 (例如 0 到 0.5 或 1.0)
    ti_clipped = ti.clip(lower=0.0, upper=1.0) # 限制在 [0, 1]
    clipped_count = (ti.dropna() > 1.0).sum()
    if clipped_count > 0:
         logger.warning(f"计算出的湍流强度中有 {clipped_count} 个值大于 1.0,已被限制为 1.0")

    logger.debug("湍流强度计算完成")
    return ti_clipped


# --- 数据质量控制与过滤 (推断 & 确认) ---

def filter_by_power_curve(
    power_kw: pd.Series,
    wind_speed_ms: pd.Series,
    capacity_kw: float,
    min_wind_speed: float = 3.0,
    max_wind_speed: float = 25.0,
    invalid_power_threshold_kw: float = 10.0,
    overload_ratio: float = 1.1
    ) -> pd.Series:
    """
    根据简化的通用功率曲线规则过滤异常功率数据点
    (实现同修正后的 Batch 2 - Step 1)
    """
    _validate_series_input(power_kw, wind_speed_ms, allow_empty=True)
    if not isinstance(capacity_kw, (int, float)) or capacity_kw <= 0:
        raise ValueError(f"额定容量 'capacity_kw' 必须是正数,但得到 {capacity_kw}")
    if not isinstance(overload_ratio, (int, float)) or overload_ratio < 1.0:
         raise ValueError(f"过载比例 'overload_ratio' 必须大于等于 1.0,但得到 {overload_ratio}")
    # logger.debug(f"开始基于功率曲线规则过滤功率数据 (容量: {capacity_kw} kW)...")
    power_filtered = power_kw.copy()
    wind_speed_local = wind_speed_ms.copy()
    original_valid_count = power_filtered.count()
    upper_limit = capacity_kw * overload_ratio
    condition1 = ~power_filtered.between(0, upper_limit, inclusive='both')
    count1 = condition1.sum(); power_filtered[condition1] = np.nan
    condition2 = (wind_speed_local < min_wind_speed) & (power_filtered > invalid_power_threshold_kw)
    count2 = condition2.sum(); power_filtered[condition2] = np.nan
    condition3 = (wind_speed_local > max_wind_speed) & (power_filtered > invalid_power_threshold_kw)
    count3 = condition3.sum(); power_filtered[condition3] = np.nan
    final_valid_count = power_filtered.count()
    filtered_count = original_valid_count - final_valid_count
    if filtered_count > 0:
        logger.info(f"功率曲线过滤完成,标记了 {filtered_count} 个异常数据点为 NaN")
    # else:
    #     logger.debug("功率曲线过滤完成,未发现异常")
    return power_filtered

def filter_stuck_sensor(series: pd.Series, window: int, threshold: float = 1e-6, flag_value: Any = np.nan) -> pd.Series:
    """
    检测并标记传感器读数在滚动窗口内几乎不变的"卡死"数据

    Args:
        series (pd.Series): 需要检查的时间序列数据
        window (int): 滚动窗口大小 (观测点数量)
        threshold (float): 判断值是否"几乎不变"的标准差阈值
        flag_value (Any): 用于标记卡死数据的值 (例如 np.nan 或 True)

    Returns:
        pd.Series: 标记了卡死数据的 Series (如果 flag_value=np.nan) 或布尔标记 Series
    """
    _validate_series_input(series)
    if not isinstance(window, int) or window <= 1:
         raise ValueError(f"窗口大小 'window' 必须是大于 1 的整数,但得到 {window}")
    logger.debug(f"开始检测卡死传感器数据,窗口: {window}, 阈值: {threshold}...")

    series_filtered = series.copy()
    # 计算滚动标准差
    rolling_std = _calculate_rolling_std(series, window=window, min_periods=window) # 需要完整窗口

    # 标记标准差低于阈值的点
    stuck_mask = rolling_std < threshold
    stuck_count = stuck_mask.sum()

    if stuck_count > 0:
        logger.warning(f"检测到 {stuck_count} 个可能的卡死传感器数据点 (滚动标准差 < {threshold}),将使用 '{flag_value}'进行标记")
        if flag_value is True or flag_value is False: # 返回布尔掩码
             return stuck_mask
        else: # 用指定值替换
             series_filtered[stuck_mask] = flag_value
             return series_filtered
    else:
        logger.debug("未检测到卡死传感器数据")
        if flag_value is True or flag_value is False:
            return pd.Series(False, index=series.index) # 全是 False
        else:
            return series_filtered # 返回原数据

def filter_outliers_stddev(series: pd.Series, window: int, std_threshold: float = 3.0, center: bool = True, flag_value: Any = np.nan) -> pd.Series:
    """
    基于滚动标准差过滤异常值
    标记那些偏离滚动均值超过 N 倍滚动标准差的点

    Args:
        series (pd.Series): 需要检查的时间序列数据
        window (int): 滚动窗口大小 (观测点数量)
        std_threshold (float): 标准差倍数阈值
        center (bool): 滚动窗口是否居中
        flag_value (Any): 用于标记异常值的值 (例如 np.nan 或 True)

    Returns:
        pd.Series: 标记了异常值的 Series (如果 flag_value=np.nan) 或布尔标记 Series
    """
    _validate_series_input(series)
    if not isinstance(window, int) or window <= 1:
         raise ValueError(f"窗口大小 'window' 必须是大于 1 的整数,但得到 {window}")
    if not isinstance(std_threshold, (int, float)) or std_threshold <= 0:
        raise ValueError(f"标准差阈值 'std_threshold' 必须是正数,但得到 {std_threshold}")
    logger.debug(f"开始基于滚动标准差过滤异常值,窗口: {window}, 阈值: {std_threshold}σ...")

    series_filtered = series.copy()
    # 计算滚动均值和标准差
    rolling_mean = series.rolling(window=window, center=center, min_periods=1).mean()
    rolling_std = _calculate_rolling_std(series, window=window, min_periods=2) # 标准差至少需要2个点

    # 计算上下限
    upper_bound = rolling_mean + std_threshold * rolling_std
    lower_bound = rolling_mean - std_threshold * rolling_std

    # 标记超出上下限的点
    outlier_mask = (series < lower_bound) | (series > upper_bound)
    outlier_count = outlier_mask.sum()

    if outlier_count > 0:
        logger.warning(f"检测到 {outlier_count} 个基于滚动标准差的异常值 (>{std_threshold}σ),将使用 '{flag_value}' 进行标记")
        if flag_value is True or flag_value is False:
            return outlier_mask
        else:
            series_filtered[outlier_mask] = flag_value
            return series_filtered
    else:
        logger.debug("未检测到基于滚动标准差的异常值")
        if flag_value is True or flag_value is False:
            return pd.Series(False, index=series.index)
        else:
            return series_filtered

def detect_icing_conditions(
    temperature_c: pd.Series,
    relative_humidity: Optional[pd.Series] = None,
    power_kw: Optional[pd.Series] = None,
    predicted_power_kw: Optional[pd.Series] = None,
    temp_threshold: float = 0.0,
    rh_threshold: float = 90.0,
    power_deviation_threshold: float = -0.15 # 例如,功率比预测低 15%
    ) -> pd.Series:
    """
    基于环境条件和/或功率偏差检测潜在的叶片结冰条件

    这是一个简化的规则示例,实际可能需要更复杂的模型
    规则: (温度 <= 阈值) AND (相对湿度 >= 阈值 OR 实际功率显著低于预测功率)

    Args:
        temperature_c (pd.Series): 温度 (摄氏度)
        relative_humidity (Optional[pd.Series]): 相对湿度 (%)
        power_kw (Optional[pd.Series]): 实际功率 (kW)
        predicted_power_kw (Optional[pd.Series]): 模型预测或理论功率 (kW)
        temp_threshold (float): 判断结冰风险的温度上限 (°C)
        rh_threshold (float): 判断结冰风险的相对湿度下限 (%)
        power_deviation_threshold (float): 判断功率异常下降的相对偏差阈值
                                            (power_kw / predicted_power_kw - 1)

    Returns:
        pd.Series: 布尔序列,True 表示可能存在结冰条件
    """
    _validate_series_input(temperature_c, allow_empty=True)
    logger.debug("开始检测潜在的结冰条件...")

    # 初始化掩码
    icing_mask = pd.Series(False, index=temperature_c.index)

    # 条件 1: 温度低于或等于阈值
    temp_condition = temperature_c <= temp_threshold

    # 条件 2: 湿度高于阈值 或 功率显著下降
    humidity_condition = pd.Series(False, index=temperature_c.index)
    power_condition = pd.Series(False, index=temperature_c.index)

    if relative_humidity is not None:
        _validate_series_input(relative_humidity, allow_empty=True)
        if relative_humidity.index.equals(temperature_c.index):
            humidity_condition = relative_humidity >= rh_threshold
        else:
            logger.warning("结冰检测相对湿度索引与温度索引不匹配,将忽略湿度条件")

    if power_kw is not None and predicted_power_kw is not None:
        _validate_series_input(power_kw, predicted_power_kw, allow_empty=True)
        if power_kw.index.equals(temperature_c.index) and predicted_power_kw.index.equals(temperature_c.index):
             # 计算功率偏差,使用安全除法
             power_ratio = _safe_divide(power_kw, predicted_power_kw.clip(lower=1.0), default=1.0) # 预测功率为0时认为无偏差
             power_deviation = power_ratio - 1.0
             power_condition = power_deviation < power_deviation_threshold
        else:
             logger.warning("结冰检测功率或预测功率索引与温度索引不匹配,将忽略功率偏差条件")

    # 组合条件
    icing_mask = temp_condition & (humidity_condition | power_condition)
    icing_count = icing_mask.sum()

    if icing_count > 0:
        logger.info(f"检测到 {icing_count} 个时间点可能存在结冰条件")
    else:
        logger.debug("未检测到符合条件的结冰时间点")

    return icing_mask


# --- 风资源评估相关 (推断) ---

# def weibull_fit(wind_speed_data: pd.Series, method: str = 'MLE') -> Optional[Tuple[float, float]]:
#     """
#     对风速数据进行韦伯分布参数拟合
#
#     Args:
#         wind_speed_data (pd.Series): 风速数据 (m/s),应去除无效值和零值
#         method (str): 拟合方法 ('MLE' - 最大似然估计, 'MM' - 矩估计)
#
#     Returns:
#         Optional[Tuple[float, float]]: 返回韦伯分布的形状参数 (k) 和尺度参数 (A)
#                                        如果拟合失败或数据不足,返回 None
#     """
#     _validate_series_input(wind_speed_data)
#     ws_clean = wind_speed_data.dropna().clip(lower=0.01) # 去除 NaN 和 0
#     if len(ws_clean) < 10: # 样本太少无法可靠拟合
#         logger.warning(f"用于韦伯拟合的数据点过少 ({len(ws_clean)} < 10),无法进行拟合")
#         return None
#
#     logger.debug(f"开始使用 '{method}' 方法进行韦伯分布拟合...")
#     try:
#         if method.upper() == 'MLE':
#             # SciPy 提供 weibull_min 分布,对应双参数韦伯分布
#             # fit() 返回 (shape, loc, scale)对于双参数韦伯, loc=0
#             # k = shape, A = scale
#             k, loc, A = sp_stats.weibull_min.fit(ws_clean, floc=0) # floc=0 强制位置参数为0
#         elif method.upper() == 'MM':
#             # 矩估计法 (较少用,实现可能需要手动计算均值和标准差来估算 k 和 A)
#             # mean = ws_clean.mean()
#             # std = ws_clean.std()
#             # ... (复杂的迭代或查找表过程) ...
#             logger.error("韦伯分布的矩估计法 (MM) 暂未实现")
#             return None
#         else:
#             raise ValueError(f"不支持的韦伯拟合方法: {method}")
#
#         logger.info(f"韦伯分布拟合完成: k (shape) = {k:.4f}, A (scale) = {A:.4f}")
#         return k, A
#     except Exception as e:
#         logger.error(f"韦伯分布拟合失败: {e}", exc_info=True)
#         return None

# def weibull_pdf(x: Union[np.ndarray, float], k: float, A: float) -> Union[np.ndarray, float]:
#     """韦伯分布概率密度函数 (PDF)"""
#     if k <= 0 or A <= 0:
#         raise ValueError("韦伯参数 k 和 A 必须为正数")
#     # 使用 scipy.stats.weibull_min.pdf
#     return sp_stats.weibull_min.pdf(x, c=k, loc=0, scale=A)

# def weibull_cdf(x: Union[np.ndarray, float], k: float, A: float) -> Union[np.ndarray, float]:
#     """韦伯分布累积分布函数 (CDF)"""
#     if k <= 0 or A <= 0:
#         raise ValueError("韦伯参数 k 和 A 必须为正数")
#     # 使用 scipy.stats.weibull_min.cdf
#     return sp_stats.weibull_min.cdf(x, c=k, loc=0, scale=A)

def calculate_air_energy_density(air_density: Union[pd.Series, float], wind_speed: Union[pd.Series, float]) -> Union[pd.Series, float]:
    """
    计算空气能量密度 (Air Power Density)
    公式: P/Area = 0.5 * rho * V^3

    Args:
        air_density (Union[pd.Series, float]): 空气密度 (kg/m^3)
        wind_speed (Union[pd.Series, float]): 风速 (m/s)

    Returns:
        Union[pd.Series, float]: 空气能量密度 (W/m^2)
    """
    # logger.debug("开始计算空气能量密度...")
    if isinstance(air_density, pd.Series):
        _validate_series_input(air_density, allow_empty=True)
    if isinstance(wind_speed, pd.Series):
        _validate_series_input(wind_speed, allow_empty=True)

    rho = np.clip(air_density, 0, None) # 密度不能为负
    V = np.clip(wind_speed, 0, None)   # 风速不能为负

    power_density = 0.5 * rho * (V ** 3)
    # logger.debug("空气能量密度计算完成")
    return power_density

# --- 特定风机/风场逻辑 (推断) ---

# def load_power_curve_data(file_path: str, density_correction: bool = False) -> Optional[pd.DataFrame]:
#     """
#     从文件加载详细的功率曲线数据
#     文件格式约定: CSV, 包含列 'WindSpeed', 'PowerKW', 可选 'AirDensity' (如果进行密度修正)
#     """
#     logger.info(f"开始加载功率曲线数据: {file_path}")
#     try:
#         pc_data = pd.read_csv(file_path)
#         required_cols = ['WindSpeed', 'PowerKW']
#         if density_correction and 'AirDensity' not in pc_data.columns:
#              logger.error(f"功率曲线文件 {file_path} 缺少进行密度修正所需的 'AirDensity' 列")
#              return None
#         missing_cols = set(required_cols) - set(pc_data.columns)
#         if missing_cols:
#              logger.error(f"功率曲线文件 {file_path} 缺少必要列: {missing_cols}")
#              return None
#         # 排序并去重
#         pc_data = pc_data.sort_values(by='WindSpeed').drop_duplicates(subset=['WindSpeed'], keep='first')
#         logger.info(f"功率曲线数据加载成功,包含 {len(pc_data)} 个数据点")
#         return pc_data
#     except FileNotFoundError:
#          logger.error(f"功率曲线文件未找到: {file_path}")
#          return None
#     except Exception as e:
#          logger.error(f"加载功率曲线文件时出错: {e}", exc_info=True)
#          return None
#
# def interpolate_power_from_curve(
#     wind_speed_ms: pd.Series,
#     power_curve_data: pd.DataFrame,
#     air_density_kgm3: Optional[pd.Series] = None,
#     ref_density: float = const.AIR_DENSITY_STANDARD
#     ) -> pd.Series:
#     """
#     根据详细功率曲线数据插值计算理论功率,可选进行空气密度修正
#
#     Args:
#         wind_speed_ms (pd.Series): 需要计算功率的风速序列 (m/s)
#         power_curve_data (pd.DataFrame): 包含 'WindSpeed' 和 'PowerKW' 列的功率曲线数据
#                                          如果需要密度修正,还需包含 'AirDensity' 列或使用 ref_density
#         air_density_kgm3 (Optional[pd.Series]): 对应风速的实际空气密度 (kg/m^3)
#                                                 如果提供,将进行密度修正
#         ref_density (float): 功率曲线数据对应的参考空气密度 (kg/m^3)
#
#     Returns:
#         pd.Series: 插值计算得到的理论功率 (kW)
#     """
#     _validate_series_input(wind_speed_ms, allow_empty=True)
#     if not isinstance(power_curve_data, pd.DataFrame) or \
#        'WindSpeed' not in power_curve_data.columns or \
#        'PowerKW' not in power_curve_data.columns:
#         raise ValueError("无效的功率曲线数据 DataFrame")
#
#     logger.debug("开始根据功率曲线插值计算理论功率...")
#     ws_curve = power_curve_data['WindSpeed'].values
#     power_curve = power_curve_data['PowerKW'].values
#
#     # 使用 numpy.interp 进行线性插值
#     # interp 需要 x 坐标单调递增,power_curve_data 在加载时已排序
#     # left=0, right=0 表示超出曲线范围的风速对应功率为 0 (常见处理方式)
#     predicted_power = np.interp(wind_speed_ms, ws_curve, power_curve, left=0.0, right=0.0)
#     predicted_power_series = pd.Series(predicted_power, index=wind_speed_ms.index)
#
#     # 进行空气密度修正 (如果需要)
#     if air_density_kgm3 is not None:
#          _validate_series_input(air_density_kgm3, allow_empty=True)
#          if not air_density_kgm3.index.equals(wind_speed_ms.index):
#               logger.error("密度修正失败空气密度索引与风速索引不匹配")
#          else:
#               logger.debug(f"应用空气密度修正 (参考密度: {ref_density} kg/m^3)...")
#               # 修正公式: P_corrected = P_curve * (rho_actual / rho_ref)
#               density_ratio = _safe_divide(air_density_kgm3, pd.Series(ref_density, index=air_density_kgm3.index), default=1.0)
#               predicted_power_series = predicted_power_series * density_ratio
#               # 修正后功率不能超过额定容量 (需要 capacity 信息)
#               # capacity = power_curve.max() # 从功率曲线获取额定功率
#               # predicted_power_series = predicted_power_series.clip(upper=capacity)
#
#     logger.debug("理论功率插值计算完成")
#     return predicted_power_series.clip(lower=0) # 功率不能为负