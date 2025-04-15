# tests/autofe/test_transform.py
# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring, redefined-outer-name, protected-access
"""
espml.autofe.transform 模块的单元测试
验证 Transform 类及其方法的正确性
"""
import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal

# 导入被测类和相关模块
from espml.autofe.transform import Transform, LabelEncoderWithOOV
from espml.autofe import utils as autofe_utils # 需要 split_features 等
from espml.autofe import operators # 需要导入算子函数以供 Transform 内部 eval
from espml.util import const

# --- Fixtures ---
@pytest.fixture(scope='module')
def transform_config() -> Dict[str, Any]:
    """提供用于初始化 Transform 的配置"""
    # 需要包含 Feature 部分
    return {
        "Feature": {
            "TaskType": "regression", "TargetName": "target",
            "TimeIndex": "time", "GroupIndex": "group", # 添加 group_index
            "CategoricalFeature": ["cat1", "cat2"], # 指定分类特征
        },
        # IncrML 和其他部分也可能被内部使用，但对 Transform 测试非必需
    }

@pytest.fixture
def transformer(transform_config: Dict[str, Any]) -> Transform:
    """创建一个 Transform 实例"""
    return Transform(**transform_config)

@pytest.fixture
def sample_df_for_transform() -> pd.DataFrame:
    """提供用于 Transform 测试的 DataFrame"""
    return pd.DataFrame({
        'A': [1.0, 2.0, 3.0, 4.0, 5.0],
        'B': [10.0, 20.0, np.nan, 40.0, 50.0],
        'C': ['X', 'Y', 'X', 'Y', 'Z'],
        'D': ['P', 'P', 'Q', np.nan, 'P'],
        'group': ['G1', 'G1', 'G2', 'G2', 'G1'],
        'target': [0, 1, 0, 1, 0] # 假设目标列
    }, index=pd.date_range('2024-01-01', periods=5, freq='D', name=const.INTERNAL_TIME_INDEX)) # 使用标准索引名

# --- 测试 LabelEncoderWithOOV ---
def test_label_encoder_oov_fit_transform():
    le = LabelEncoderWithOOV()
    data = pd.Series(['A', 'B', 'A', 'C', np.nan])
    le.fit(data)
    assert list(le.classes_) == ['A', 'B', 'C', '__NaN__'] # fit 时不包含 OOV
    transformed = le.transform(data)
    # A->0, B->1, C->2, NaN->3
    np.testing.assert_array_equal(transformed, [0, 1, 0, 2, 3])
    # 测试 OOV
    new_data = pd.Series(['A', 'D', np.nan, 'B'])
    transformed_new = le.transform(new_data)
    # A->0, D->OOV->4, NaN->3, B->1
    np.testing.assert_array_equal(transformed_new, [0, 4, 3, 1])

# --- 测试 Transform 初始化 ---
def test_transform_init(transform_config: Dict[str, Any]):
    """测试 Transform 类初始化"""
    transformer = Transform(**transform_config)
    assert transformer.target_name == "target"
    assert "cat1" in transformer.cat_features
    assert "add" in transformer.calculate_operators
    assert "count" in transformer.transform_operators

# --- 测试 Transform._is_integer, _is_combination_feature, _analysis_feature_space ---
def test_transform_internal_helpers(transformer: Transform, sample_df_for_transform):
    """测试内部辅助函数"""
    assert transformer._is_integer("5") is True
    assert transformer._is_integer("5.0") is False
    assert transformer._is_integer("abc") is False
    assert transformer._is_combination_feature("add###A|||B$$$") is True
    assert transformer._is_combination_feature("A") is False

    key = "aggmean###A|||C$$$"
    temp, op, fes, time_span, features = transformer._analysis_feature_space(sample_df_for_transform, key)
    assert op == "aggmean"
    assert fes == ['A', 'C'] # 列表形式
    assert time_span is None
    assert list(features.columns) == ['A', 'C'] # 提取了对应列

    key_ts = "ts_std###B|||3$$$"
    temp, op, fes, time_span, features = transformer._analysis_feature_space(sample_df_for_transform, key_ts)
    assert op == "ts_std"
    assert fes == 'B' # 单个特征是字符串
    assert time_span == 3
    assert list(features.columns) == ['B']

    # 测试带 group_index 的时间序列
    key_ts_group = "ts_mean###A|||2$$$"
    transformer.group_index = "group" # 设置 group_index
    temp, op, fes, time_span, features = transformer._analysis_feature_space(sample_df_for_transform, key_ts_group)
    assert op == "ts_mean"
    assert fes == ['group', 'A'] # group 被加在前面
    assert time_span == 2
    assert list(features.columns) == ['group', 'A']
    transformer.group_index = None # 恢复

# --- 测试 Transform._calculate 和 _transform (通过 _recursion 调用) ---
# 需要确保 operators.py 中的函数可用
def test_transform_calculate_simple(transformer: Transform, sample_df_for_transform):
    """测试计算简单算术特征"""
    feature_name = "add###A|||B$$$"
    result = transformer._recursion(sample_df_for_transform, feature_name, fit=False)
    assert isinstance(result, pd.Series)
    assert result.name == feature_name
    expected = (sample_df_for_transform['A'].astype(float) + sample_df_for_transform['B'].astype(float)).fillna(0)
    # 注意 _calculate 内部有 reset_index(drop=True)，索引会丢失
    # assert_series_equal(result, expected.rename(feature_name), check_index=False) # 不检查索引
    # 修正Transform 的 transform/fit_transform 会保证索引对齐
    assert_series_equal(result, expected.rename(feature_name))


def test_transform_transform_simple(transformer: Transform, sample_df_for_transform):
    """测试应用简单转换特征 (需要先 fit)"""
    feature_name = "count###C$$$"
    # 必须先 fit
    transformer.fit(sample_df_for_transform, [feature_name])
    assert feature_name in transformer._record # 检查 record 是否被填充
    # 再 transform
    result = transformer._recursion(sample_df_for_transform, feature_name, fit=False)
    assert isinstance(result, pd.Series)
    assert result.name == feature_name
    # 手动计算期望值
    mapping = sample_df_for_transform['C'].value_counts(normalize=True, dropna=False)
    expected = sample_df_for_transform['C'].map(mapping).fillna(0)
    assert_series_equal(result, expected.rename(feature_name))

def test_transform_calculate_nested(transformer: Transform, sample_df_for_transform):
    """测试计算嵌套特征"""
    # mul###add###A|||B$$$|||A$$$ -> (A+B)*A
    feature_name = "mul###add###A|||B$$$|||A$$$"
    result = transformer._recursion(sample_df_for_transform, feature_name, fit=False)
    assert isinstance(result, pd.Series)
    assert result.name == feature_name
    # 计算期望值
    s_a = sample_df_for_transform['A'].astype(float)
    s_b = sample_df_for_transform['B'].astype(float)
    expected = ((s_a + s_b) * s_a).fillna(0) # 假设内部填充0
    # 注意 _calculate 内部的 reset_index
    # assert_series_equal(result, expected.rename(feature_name), check_index=False)
    # 修正顶层调用会保持索引
    assert_series_equal(result, expected.rename(feature_name))


def test_transform_transform_nested_fit(transformer: Transform, sample_df_for_transform):
    """测试拟合嵌套的 transform 特征"""
    # aggstd###A|||combine###C|||D$$$$$$ -> 按 C&D 组合分组，计算 A 的 std
    feature_name = "aggstd###A|||combine###C|||D$$$$$$"
    # 调用 fit
    transformer.fit(sample_df_for_transform, [feature_name])
    # 检查 record 中是否存储了 combine 和 aggstd 的中间结果
    assert "combine###C|||D$$$" in transformer._record
    assert feature_name in transformer._record
    assert isinstance(transformer._record[feature_name], pd.Series) # aggstd 存储 Series

# --- 测试 fit, transform, fit_transform ---
def test_transform_fit_only(transformer: Transform, sample_df_for_transform):
    """测试只调用 fit 是否填充了 record"""
    features_to_fit = ["count###C$$$", "aggmean###A|||C$$$"]
    transformer.fit(sample_df_for_transform, features_to_fit)
    assert "count###C$$$" in transformer._record
    assert "aggmean###A|||C$$$" in transformer._record
    assert isinstance(transformer._record["count###C$$$"], pd.Series)
    assert isinstance(transformer._record["aggmean###A|||C$$$"], pd.Series)

def test_transform_transform_only_after_fit(transformer: Transform, sample_df_for_transform):
    """测试先 fit 再 transform"""
    features = ["count###C$$$", "add###A|||B$$$"]
    transformer.fit(sample_df_for_transform, features)
    df_transformed = transformer.transform(sample_df_for_transform, features)
    assert "count###C$$$" in df_transformed.columns
    assert "add###A|||B$$$" in df_transformed.columns
    assert not df_transformed["count###C$$$"].isna().any()
    assert not df_transformed["add###A|||B$$$"].isna().any()
    # 检查索引是否与输入一致
    assert_index_equal(df_transformed.index, sample_df_for_transform.index)

def test_transform_fit_transform(transformer: Transform, sample_df_for_transform):
    """测试 fit_transform 方法"""
    features = ["aggmin###B|||D$$$", "log###A$$$"]
    df_transformed = transformer.fit_transform(sample_df_for_transform, features)
    assert "aggmin###B|||D$$$" in df_transformed.columns
    assert "log###A$$$" in df_transformed.columns
    assert "aggmin###B|||D$$$" in transformer._record # 检查 fit 是否执行
    assert not df_transformed["aggmin###B|||D$$$"].isna().any()
    assert not df_transformed["log###A$$$"].isna().any()
    assert_index_equal(df_transformed.index, sample_df_for_transform.index)

def test_transform_label_encode(transformer: Transform, sample_df_for_transform):
    """测试带 LabelEncoding 的 fit_transform"""
    transformer.task_type = 'classification' # 设置为分类任务
    df_test = sample_df_for_transform.copy()
    # 制造需要编码的列target (数值但 < nunique), C (object)
    df_test['target'] = df_test['target'].astype(int) # 确保是整数
    df_test['C'] = df_test['C'].astype(str)

    df_encoded = transformer.fit_transform(df_test, [], labelencode=True)

    # 检查 target 列是否被编码
    assert pd.api.types.is_integer_dtype(df_encoded['target'])
    assert df_encoded['target'].min() >= 0
    assert f"__LABEL_ENCODER_{transformer.target_name}" in transformer._record

    # 检查 C 列是否被编码
    assert pd.api.types.is_integer_dtype(df_encoded['C'])
    assert df_encoded['C'].min() >= 0
    assert f"__LABEL_ENCODER_C" in transformer._record

    # 检查数值列 A, B 是否未被编码
    assert pd.api.types.is_float_dtype(df_encoded['A'])
    assert pd.api.types.is_float_dtype(df_encoded['B'])


# --- 结束 tests/autofe/test_transform.py ---