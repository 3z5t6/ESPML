# tests/autofe/test_operators.py
# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-argument
"""
espml.autofe.operators 模块的单元测试
验证各个算子函数的计算逻辑和边界情况处理
"""
import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal

# 导入所有被测函数
from espml.autofe.operators import (
    add, sub, mul, div, std as std_op, maximize, # std 重命名避免与 pd.std 冲突
    sine, cosine, pow as pow_op, log, # pow 重命名
    count, crosscount, nunique, combine,
    aggmean, aggmax, aggmin, aggstd,
    diff, delay, ts_mean, ts_std, ts_cov, ts_corr,
    ts_max, ts_min, ts_rank, ewm_mean, ewm_std,
    ewm_cov, ewm_corr
)

# --- Fixtures ---
@pytest.fixture
def series_a() -> pd.Series:
    return pd.Series([1.0, 2.0, 3.0, 4.0, np.nan], name='A')

@pytest.fixture
def series_b() -> pd.Series:
    return pd.Series([5.0, 0.0, -1.0, np.nan, 6.0], name='B')

@pytest.fixture
def series_c_cat() -> pd.Series:
    return pd.Series(['X', 'Y', 'X', 'Y', 'Z'], name='C')

@pytest.fixture
def series_d_cat() -> pd.Series:
    return pd.Series(['P', 'P', 'Q', 'Q', np.nan], name='D')

@pytest.fixture
def df_ab(series_a, series_b) -> pd.DataFrame:
    return pd.concat([series_a, series_b], axis=1)

@pytest.fixture
def df_ac(series_a, series_c_cat) -> pd.DataFrame:
    return pd.concat([series_a, series_c_cat], axis=1)

@pytest.fixture
def df_cd(series_c_cat, series_d_cat) -> pd.DataFrame:
    return pd.concat([series_c_cat, series_d_cat], axis=1)

@pytest.fixture
def series_ts() -> pd.Series:
    """带 DatetimeIndex 的时间序列"""
    index = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'])
    return pd.Series([10.0, 12.0, 11.0, 13.0, 12.5], index=index, name='TS')

@pytest.fixture
def df_ts_group(series_ts) -> pd.DataFrame:
    """带分组键的时间序列 DataFrame"""
    df = series_ts.to_frame()
    df['group'] = ['G1', 'G1', 'G2', 'G2', 'G1']
    # 交换列顺序，group 在前
    return df[['group', 'TS']]


# --- 测试二元数值算子 ('n', 'n') ---
def test_add(series_a, series_b, df_ab):
    expected = pd.Series([6.0, 2.0, 2.0, 0.0, 0.0], name='A') # NaN+4 -> NaN -> fillna(0)
    result = add(df_ab)
    assert_series_equal(result, expected, check_names=False)

def test_sub(series_a, series_b, df_ab):
    expected = pd.Series([-4.0, 2.0, 4.0, 0.0, 0.0], name='A') # 4-NaN -> NaN -> fillna(0)
    result = sub(df_ab)
    assert_series_equal(result, expected, check_names=False)

def test_mul(series_a, series_b, df_ab):
    expected = pd.Series([5.0, 0.0, -3.0, 0.0, 0.0], name='A') # NaN*4, 6*NaN -> NaN -> fillna(0)
    result = mul(df_ab)
    assert_series_equal(result, expected, check_names=False)

def test_div(series_a, series_b, df_ab):
    expected = pd.Series([1/5, 0.0, -3.0, 0.0, 0.0], name='A') # 2/0 -> NaN, 4/NaN -> NaN, NaN/6 -> NaN
    result = div(df_ab)
    assert_series_equal(result, expected, check_names=False)

def test_std_op(series_a, series_b, df_ab):
    # 计算 [1,5], [2,0], [3,-1], [4,NaN], [NaN,6] 的行标准差
    expected = pd.Series([np.std([1,5], ddof=1), np.std([2,0], ddof=1), np.std([3,-1], ddof=1), 0.0, 0.0])
    result = std_op(df_ab)
    assert_series_equal(result, expected, check_names=False, atol=1e-9)

def test_maximize(series_a, series_b, df_ab):
    expected = pd.Series([5.0, 2.0, 3.0, 0.0, 0.0], name='B') # max(4,NaN)->NaN->0, max(NaN,6)->NaN->0
    result = maximize(df_ab)
    assert_series_equal(result, expected, check_names=False)


# --- 测试一元数值算子 ('n',) ---
def test_sine(series_a):
    expected = np.sin(series_a).fillna(0)
    result = sine(series_a)
    assert_series_equal(result, expected)

def test_cosine(series_b):
    expected = np.cos(series_b).fillna(0)
    result = cosine(series_b)
    assert_series_equal(result, expected)

def test_pow_op(series_a):
    expected = series_a.pow(2.0).fillna(0) # 默认平方
    result = pow_op(series_a) # 使用默认 exponent=2.0
    assert_series_equal(result, expected)
    # 测试不同指数 (如果代码支持通过参数传递)
    # result_pow3 = pow_op(series_a, exponent=3.0)
    # expected_pow3 = series_a.pow(3.0).fillna(0)
    # assert_series_equal(result_pow3, expected_pow3)

def test_log(series_a): # series_a 都是正数
    expected = np.log1p(np.abs(series_a)).fillna(0)
    result = log(series_a)
    assert_series_equal(result, expected)

def test_log_non_positive(series_b): # series_b 包含 0 和负数
    expected = np.log1p(np.abs(series_b)).fillna(0) # log(1+0)=0, log(1+1)=log(2), log(1+6)=log(7)
    result = log(series_b)
    assert_series_equal(result, expected)


# --- 测试一元分类算子 ('c',) ---
def test_count(series_c_cat):
    expected_freq = series_c_cat.value_counts(normalize=True, dropna=False)
    expected = series_c_cat.map(expected_freq).fillna(0)
    result, intermediate = count(series_c_cat, intermediate=True)
    assert_series_equal(result, expected)
    assert isinstance(intermediate, pd.Series)
    assert intermediate.name == 'count'


# --- 测试二元分类算子 ('c', 'c') ---
def test_crosscount(df_cd):
    s1 = df_cd['C'].astype(str).fillna('__NaN__')
    s2 = df_cd['D'].astype_str().fillna('__NaN__')
    combined = s1 + "_&_" + s2
    expected_freq = combined.value_counts(normalize=True, dropna=False)
    expected = combined.map(expected_freq).fillna(0)
    result, intermediate = crosscount(df_cd, intermediate=True)
    assert_series_equal(result, expected)
    assert isinstance(intermediate, pd.Series) # Groupby size is Series

def test_nunique(df_cd):
    # 按 D 分组，计算 C 的 nunique
    # Group P: X, Y -> 2
    # Group Q: X, Y -> 2
    # Group NaN: Z -> 1
    expected = pd.Series([2, 2, 2, 2, 1], index=df_cd.index, dtype=float) # 返回 float
    result, intermediate = nunique(df_cd, intermediate=True)
    assert_series_equal(result, expected)
    assert isinstance(intermediate, pd.Series) # Groupby nunique is Series
    assert intermediate.loc['P'] == 2
    assert intermediate.loc['Q'] == 2
    assert intermediate.loc['__NaN__'] == 1 # 假设 fillna 内部实现

def test_combine(df_cd):
    s1_str = df_cd['C'].astype(str).fillna('__NaN__')
    s2_str = df_cd['D'].astype(str).fillna('__NaN__')
    expected_combined_str = s1_str + "_&_" + s2_str
    expected_codes = expected_combined_str.astype('category').cat.codes
    result, intermediate = combine(df_cd, intermediate=True)
    assert_series_equal(result, expected_codes, check_names=False)
    assert isinstance(intermediate, pd.Series) # 映射现在是 Series

# --- 测试数值按分类聚合算子 ('n', 'c') ---
def test_aggmean(df_ac):
    expected_map = df_ac.groupby('C')['A'].mean()
    expected = df_ac['C'].map(expected_map).fillna(0)
    result, intermediate = aggmean(df_ac, intermediate=True)
    assert_series_equal(result, expected)
    pd.testing.assert_series_equal(intermediate, expected_map, check_names=False)

def test_aggstd(df_ac):
    # Group X: [1, 3] -> std = np.std([1,3], ddof=1) = 1.414...
    # Group Y: [2, 4] -> std = np.std([2,4], ddof=1) = 1.414...
    # Group Z: [NaN] -> std = NaN -> fillna(0)
    group_std = df_ac.groupby('C')['A'].std(ddof=1)
    expected = df_ac['C'].map(group_std).fillna(0)
    result, intermediate = aggstd(df_ac, intermediate=True)
    assert_series_equal(result, expected)
    pd.testing.assert_series_equal(intermediate, group_std, check_names=False)

# --- 测试时间序列算子 ('n', 't') ---
def test_diff(series_ts):
    expected = series_ts.diff(periods=1).fillna(0) # d=1
    result = diff(series_ts, time=1)
    assert_series_equal(result, expected)
    expected = series_ts.diff(periods=2).fillna(0) # d=2
    result = diff(series_ts, time=2)
    assert_series_equal(result, expected)

def test_delay(series_ts):
    expected = series_ts.shift(periods=1).fillna(0) # d=1 (Lag)
    result = delay(series_ts, time=1)
    assert_series_equal(result, expected)
    expected = series_ts.shift(periods=-1).fillna(0) # d=-1 (Lead)
    result = delay(series_ts, time=-1)
    assert_series_equal(result, expected)

def test_ts_mean(series_ts):
    expected = series_ts.rolling(window=3, min_periods=1).mean().fillna(0)
    result = ts_mean(series_ts, time=3)
    assert_series_equal(result, expected)

def test_ts_std(series_ts):
    expected = series_ts.rolling(window=3, min_periods=2).std(ddof=1).fillna(0)
    result = ts_std(series_ts, time=3)
    assert_series_equal(result, expected)

def test_ewm_mean(series_ts):
    expected = series_ts.ewm(span=3, min_periods=1).mean().fillna(0)
    result = ewm_mean(series_ts, time=3)
    assert_series_equal(result, expected)

def test_ts_rank(series_ts):
    expected = series_ts.rolling(window=3, min_periods=1).rank(pct=True).fillna(0)
    result = ts_rank(series_ts, time=3)
    assert_series_equal(result, expected)

# 测试分组时间序列
def test_grouped_ts_mean(df_ts_group):
     expected = df_ts_group.groupby('group')['TS'].rolling(window=2, min_periods=1).mean()
     # rolling on grouped object creates multiindex, need to reset and align
     expected = expected.reset_index(level=0, drop=True).sort_index().fillna(0)
     result = ts_mean(df_ts_group, time=2) # _ts_op 内部处理分组
     assert_series_equal(result, expected)


# --- 结束 tests/autofe/test_operators.py ---