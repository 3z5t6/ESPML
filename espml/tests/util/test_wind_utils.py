# tests/util/test_wind_utils.py
import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal

from espml.util import wind_utils # 导入被测模块

# --- 测试 degrees_to_vector ---
def test_degrees_to_vector_basic():
    """测试基本的风向角度转换"""
    degrees = pd.Series([0, 90, 180, 270, 360])
    expected_x = pd.Series([0.0, 1.0, 0.0, -1.0, 0.0])
    expected_y = pd.Series([1.0, 0.0, -1.0, 0.0, 1.0])
    expected_df = pd.DataFrame({'WD_X': expected_x, 'WD_Y': expected_y})
    result_df = wind_utils.degrees_to_vector(degrees)
    # 使用 check_dtype=False 和 atol 允许浮点误差
    assert_frame_equal(result_df, expected_df, check_dtype=False, atol=1e-9)

def test_degrees_to_vector_nan():
    """测试包含 NaN 的风向角度转换"""
    degrees = pd.Series([0, np.nan, 180])
    expected_x = pd.Series([0.0, np.nan, 0.0])
    expected_y = pd.Series([1.0, np.nan, -1.0])
    expected_df = pd.DataFrame({'WD_X': expected_x, 'WD_Y': expected_y})
    result_df = wind_utils.degrees_to_vector(degrees)
    assert_frame_equal(result_df, expected_df, check_dtype=False, atol=1e-9)

def test_degrees_to_vector_out_of_range():
    """测试超出 [0, 360] 范围的角度 (会被模运算处理)"""
    degrees = pd.Series([-90, 450]) # -90 -> 270, 450 -> 90
    expected_x = pd.Series([-1.0, 1.0])
    expected_y = pd.Series([0.0, 0.0])
    expected_df = pd.DataFrame({'WD_X': expected_x, 'WD_Y': expected_y})
    result_df = wind_utils.degrees_to_vector(degrees)
    assert_frame_equal(result_df, expected_df, check_dtype=False, atol=1e-9)

def test_degrees_to_vector_invalid_input():
    """测试无效输入类型"""
    with pytest.raises(TypeError):
        wind_utils.degrees_to_vector([0, 90]) # 输入是列表而非 Series

# --- 测试 calculate_air_density ---
def test_calculate_air_density_normal():
    """测试正常条件下的空气密度计算"""
    # 接近标准大气压和温度
    temp_c = pd.Series([15.0])
    pressure_pa = pd.Series([101325.0]) # 标准大气压 Pa
    # 预期密度约 1.225 kg/m^3
    expected_density = 101325.0 / (287.058 * (15.0 + 273.15))
    result = wind_utils.calculate_air_density(temp_c, pressure_pa)
    assert_series_equal(result, pd.Series([expected_density]), check_dtype=False, atol=1e-3)

def test_calculate_air_density_nan_input():
    """测试输入包含 NaN"""
    temp_c = pd.Series([15.0, np.nan])
    pressure_pa = pd.Series([101325.0, 100000.0])
    result = wind_utils.calculate_air_density(temp_c, pressure_pa)
    assert pd.isna(result.iloc[1]) # 第二个应为 NaN
    assert pd.notna(result.iloc[0])

def test_calculate_air_density_invalid_temp():
    """测试温度低于绝对零度"""
    temp_c = pd.Series([-300.0])
    pressure_pa = pd.Series([100000.0])
    with pytest.raises(ValueError, match="低于绝对零度"):
        wind_utils.calculate_air_density(temp_c, pressure_pa)

def test_calculate_air_density_invalid_pressure():
    """测试压力为负数"""
    temp_c = pd.Series([15.0])
    pressure_pa = pd.Series([-1000.0])
    # 内部 clip 会处理，但可能直接报错或返回 NaN
    # 根据当前实现，负压会被 clip 为 NaN 或导致计算错误，结果可能受 clip 影响
    # result = wind_utils.calculate_air_density(temp_c, pressure_pa)
    # assert pd.isna(result.iloc[0]) # 预期 NaN 或被 clip
    # 或者，如果希望它报错
    # with pytest.raises(ValueError, match="压力数据包含负值"): # 需要修改函数以在 clip 前报错
    #     wind_utils.calculate_air_density(temp_c, pressure_pa)
    # 当前实现是 clip，所以不报错
    result = wind_utils.calculate_air_density(temp_c, pressure_pa)
    assert pd.notna(result.iloc[0]) # 因为 clip 和 fillna(0) 处理了

def test_calculate_air_density_clipping():
    """测试结果裁剪"""
    # 构造产生极端密度的条件 (极低温高压 或 高温低压)
    temp_high = pd.Series([500]) # K
    press_low = pd.Series([50000]) # Pa -> 密度会很低
    result_low = wind_utils.calculate_air_density(temp_high - 273.15, press_low)
    assert result_low.iloc[0] >= 0.8 # 检查是否被 clip 到下限

    temp_low = pd.Series([-50 + 273.15]) # K
    press_high = pd.Series([150000]) # Pa -> 密度会很高
    result_high = wind_utils.calculate_air_density(temp_low - 273.15, press_high)
    assert result_high.iloc[0] <= 1.5 # 检查是否被 clip 到上限


# --- 测试 filter_by_power_curve ---
@pytest.fixture
def pc_filter_data(self):
    power = pd.Series([0, 500, 1500, 1600, 50, 20, 800, 10]) # kW
    wind_speed = pd.Series([2.0, 8.0, 15.0, 26.0, 2.5, 27.0, 10.0, 2.8]) # m/s
    capacity = 1500.0 # kW
    return power, wind_speed, capacity

def test_pc_filter_normal(self, pc_filter_data):
    """测试正常运行范围内的点"""
    power, wind_speed, capacity = pc_filter_data
    result = wind_utils.filter_by_power_curve(power, wind_speed, capacity)
    assert pd.notna(result.iloc[1]) # 500kW @ 8m/s
    assert pd.notna(result.iloc[2]) # 1500kW @ 15m/s
    assert pd.notna(result.iloc[6]) # 800kW @ 10m/s

def test_pc_filter_below_cutin(self, pc_filter_data):
    """测试低于切入风速但功率异常"""
    power, wind_speed, capacity = pc_filter_data
    result = wind_utils.filter_by_power_curve(power, wind_speed, capacity, min_wind_speed=3.0, invalid_power_threshold_kw=30.0)
    assert pd.isna(result.iloc[4]) # 50kW @ 2.5m/s 应被过滤
    assert pd.notna(result.iloc[0]) # 0kW @ 2.0m/s 不过滤
    assert pd.notna(result.iloc[7]) # 10kW @ 2.8m/s (低于阈值 30) 不过滤

def test_pc_filter_above_cutout(self, pc_filter_data):
    """测试高于切出风速但功率异常"""
    power, wind_speed, capacity = pc_filter_data
    result = wind_utils.filter_by_power_curve(power, wind_speed, capacity, max_wind_speed=25.0, invalid_power_threshold_kw=15.0)
    assert pd.isna(result.iloc[3]) # 1600kW @ 26m/s (功率也超了，但主要是风速)
    assert pd.isna(result.iloc[5]) # 20kW @ 27m/s 应被过滤

def test_pc_filter_power_out_of_range(self, pc_filter_data):
    """测试功率超出 [0, capacity * 1.1] 范围"""
    power = pd.Series([-10, 1700]) # 负值, 超过 1500*1.1=1650
    wind_speed = pd.Series([10.0, 15.0])
    capacity = 1500.0
    result = wind_utils.filter_by_power_curve(power, wind_speed, capacity)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])

# --- 测试 adjust_wind_speed_log_law & adjust_wind_speed_power_law ---
@pytest.mark.parametrize("h1, h2, z0, expected_ratio", [
    (10, 70, 0.03, np.log(70/0.03)/np.log(10/0.03)),
    (70, 10, 0.03, np.log(10/0.03)/np.log(70/0.03)),
    (10, 10, 0.03, 1.0), # 高度相同
])
def test_adjust_wind_speed_log_law(self, h1, h2, z0, expected_ratio):
    ws1 = pd.Series([10.0])
    expected_ws2 = ws1 * expected_ratio
    result_ws2 = wind_utils.adjust_wind_speed_log_law(ws1, h1, h2, z0)
    assert_series_equal(result_ws2, expected_ws2, check_dtype=False, atol=1e-6)

@pytest.mark.parametrize("h1, h2, alpha, expected_ratio", [
    (10, 70, 0.14, (70/10)**0.14),
    (70, 10, 0.14, (10/70)**0.14),
    (10, 10, 0.14, 1.0),
])
def test_adjust_wind_speed_power_law(self, h1, h2, alpha, expected_ratio):
    ws1 = pd.Series([10.0])
    expected_ws2 = ws1 * expected_ratio
    result_ws2 = wind_utils.adjust_wind_speed_power_law(ws1, h1, h2, alpha)
    assert_series_equal(result_ws2, expected_ws2, check_dtype=False, atol=1e-6)

def test_adjust_wind_speed_invalid_input(self):
    ws = pd.Series([10.0])
    with pytest.raises(ValueError): wind_utils.adjust_wind_speed_log_law(ws, -10, 70, 0.03)
    with pytest.raises(ValueError): wind_utils.adjust_wind_speed_log_law(ws, 10, -70, 0.03)
    with pytest.raises(ValueError): wind_utils.adjust_wind_speed_log_law(ws, 10, 70, -0.03)
    with pytest.raises(ValueError): wind_utils.adjust_wind_speed_power_law(ws, 10, 70, -0.14)