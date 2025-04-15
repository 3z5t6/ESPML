# tests/autofe/test_autofe_utils.py
# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring, redefined-outer-name, pointless-statement
"""
espml.autofe.utils 模块的单元测试
验证 AutoFE 相关工具函数的正确性
"""

import pytest
import pandas as pd
import numpy as np

# 导入被测模块和常量
from espml.autofe import utils as autofe_utils
from espml.autofe.utils import OPERATORCHAR, OPERATORTYPES, TIME_SPAN

# --- 测试常量 ---
def test_constants_definition():
    """检查常量是否正确定义"""
    assert isinstance(OPERATORCHAR.feature_separator, str)
    assert isinstance(OPERATORCHAR.head_character, str)
    assert isinstance(OPERATORCHAR.tail_character, str)
    assert isinstance(OPERATORTYPES, dict)
    assert isinstance(autofe_utils.OPERTORS, set) # 注意代码变量名 OPEATORS，已在utils中修正
    assert len(autofe_utils.OPERTORS) > 0
    assert isinstance(TIME_SPAN, list)

# --- 测试 split_num_cat_features ---
@pytest.fixture
def sample_df_for_split() -> pd.DataFrame:
    """提供用于特征分割测试的 DataFrame"""
    return pd.DataFrame({
        'numeric_high_card': np.random.rand(20) * 100,
        'numeric_low_card': np.random.randint(0, 3, 20), # 低基数数值
        'categorical_str': ['A', 'B'] * 10,
        'categorical_cat': pd.Categorical(['X', 'Y', 'X'] * 6 + ['X', 'X']),
        'boolean_col': [True, False] * 10,
        'object_col': [{'a': 1}, {'b': 2}] * 10, # 对象类型
        'ignored_col': range(20),
        'target': np.random.rand(20)
    })

def test_split_num_cat_features_basic(sample_df_for_split: pd.DataFrame):
    """测试基本的数值和分类特征分割"""
    cat_cols, num_cols = autofe_utils.split_num_cat_features(sample_df_for_split)
    assert 'numeric_high_card' in num_cols
    assert 'numeric_low_card' in cat_cols # 低基数数值被视为分类
    assert 'categorical_str' in cat_cols
    assert 'categorical_cat' in cat_cols
    assert 'boolean_col' in cat_cols
    assert 'object_col' in cat_cols
    assert 'ignored_col' in num_cols # 未忽略时视为数值
    assert 'target' in num_cols

def test_split_num_cat_features_with_ignore(sample_df_for_split: pd.DataFrame):
    """测试带忽略列的分割"""
    ignore = ['ignored_col', 'target', 'object_col']
    cat_cols, num_cols = autofe_utils.split_num_cat_features(sample_df_for_split, ignore_columns=ignore)
    assert 'numeric_high_card' in num_cols
    assert 'numeric_low_card' in cat_cols
    assert 'categorical_str' in cat_cols
    assert 'categorical_cat' in cat_cols
    assert 'boolean_col' in cat_cols
    assert 'object_col' not in cat_cols and 'object_col' not in num_cols
    assert 'ignored_col' not in cat_cols and 'ignored_col' not in num_cols
    assert 'target' not in cat_cols and 'target' not in num_cols

# --- 测试 split_features ---
@pytest.mark.parametrize("feature_name, expected_parts", [
    ("add###colA|||colB$$$", ['add', 'colA', 'colB']),
    ("log###colC$$$", ['log', 'colC']),
    ("ts_mean###colD|||5$$$", ['ts_mean', 'colD', '5']),
    ("aggstd###num_feat|||cat_feat$$$", ['aggstd', 'num_feat', 'cat_feat']),
    ("combine###cat1|||cat2$$$", ['combine', 'cat1', 'cat2']),
    ("delay###target|||-1$$$", ['delay', 'target', '-1']),
    ("single_feature", ["single_feature"]), # 非组合特征
    ("invalid###format", ["invalid###format"]), # 格式无效
    ("op###feature_with|||separator$$$", ['op', 'feature_with', 'separator']), # 包含分隔符但格式正确
    ("op###f1|||f2|||f3$$$", ['op', 'f1', 'f2', 'f3']), # 三个组件
])
def test_split_features(feature_name: str, expected_parts: List[str]):
    """测试各种特征名称字符串的解析"""
    assert autofe_utils.split_features(feature_name) == expected_parts

# --- 测试 is_combination_feature ---
@pytest.mark.parametrize("feature_name, expected_result", [
    ("add###colA|||colB$$$", True),
    ("log###colC$$$", True),
    ("ts_mean###colD|||5$$$", True),
    ("feature_a", False),
    ("no_headcolA|||colB$$$", False),
    ("add###no_tail", False),
    ("incomplete#$$", False),
    ("###$$$", True), # 极端情况，格式上是组合特征
])
def test_is_combination_feature(feature_name: str, expected_result: bool):
    """测试判断是否为组合特征的函数"""
    assert autofe_utils.is_combination_feature(feature_name) == expected_result

# --- 测试 feature_space ---
# 测试 feature_space 比较复杂，因为它依赖于大量的 OPERATORTYPES
# 这里只做基本的功能性测试，不验证所有可能的组合输出
@pytest.fixture
def df_for_feature_space() -> pd.DataFrame:
    """提供用于 feature_space 测试的 DataFrame"""
    return pd.DataFrame({
        'cat1': ['A', 'B', 'A', 'C'],
        'cat2': ['X', 'X', 'Y', 'Y'],
        'num1': [1, 2, 3, 4],
        'num2': [5, 6, 7, 8],
        'target': [0, 1, 0, 1],
        'time': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'])
    }).set_index('time')

def test_feature_space_basic(df_for_feature_space: pd.DataFrame):
    """测试基本的候选特征生成"""
    candidates = autofe_utils.feature_space(df_for_feature_space, target_name='target')
    assert isinstance(candidates, list)
    assert len(candidates) > 0
    # 检查是否包含一些预期的组合
    assert 'count###cat1$$$' in candidates
    assert 'crosscount###cat1|||cat2$$$' in candidates # 注意排序
    assert 'aggmean###num1|||cat1$$$' in candidates
    assert 'add###num1|||num2$$$' in candidates # 注意排序

def test_feature_space_with_time(df_for_feature_space: pd.DataFrame):
    """测试包含时间索引时的特征生成"""
    candidates = autofe_utils.feature_space(df_for_feature_space, target_name='target', time_index='time')
    assert 'diff###num1|||1$$$' in candidates
    assert 'delay###target|||-1$$$' in candidates
    assert 'ts_mean###num2|||3$$$' in candidates

def test_feature_space_already_selected(df_for_feature_space: pd.DataFrame):
    """测试 already_selected 参数是否生效"""
    existing = ['add###num1|||num2$$$', 'count###cat1$$$']
    candidates = autofe_utils.feature_space(df_for_feature_space, target_name='target', already_selected=existing)
    assert 'add###num1|||num2$$$' not in candidates
    assert 'count###cat1$$$' not in candidates
    assert 'sub###num1|||num2$$$' in candidates # 其他特征应该还在

def test_feature_space_max_candidates(df_for_feature_space: pd.DataFrame):
    """测试最大候选特征数限制"""
    max_num = 5
    candidates = autofe_utils.feature_space(df_for_feature_space, target_name='target', max_candidate_features=max_num)
    assert len(candidates) == max_num

# --- 测试 name2formula 和 feature2table ---
# name2formula 的测试比较繁琐，因为它依赖于 split_features 和递归
def test_name2formula_simple():
    """测试简单特征名的转换"""
    assert autofe_utils.name2formula("my_feature") == r"$my\_feature$"
    assert autofe_utils.name2formula("add###num1|||num2$$$") == r"$$(num1) + (num2)$$" # 假设 num1, num2 不需要转义
    assert autofe_utils.name2formula("log###feature_with_underscore$$$") == r"$$\log(feature\_with\_underscore)$$"
    assert autofe_utils.name2formula("div###f1|||f2$$$") == r"$$\frac{f1}{f2}$$"

def test_name2formula_nested():
    """测试嵌套特征名的转换"""
    nested_name = "mul###add###f1|||f2$$$|||log###f3$$$$$$"
    expected = r"$$( (f1) + (f2) ) \times (\log(f3))$$"
    assert autofe_utils.name2formula(nested_name) == expected

def test_feature2table(df_for_feature_space: pd.DataFrame):
    """测试 feature2table 的基本功能"""
    candidates = autofe_utils.feature_space(df_for_feature_space, target_name='target', max_candidate_features=5)
    table = autofe_utils.feature2table(candidates)
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == ['id', 'latex']
    assert len(table) == 5
    assert table['id'].tolist() == [1, 2, 3, 4, 5]
    assert table['latex'].iloc[0].startswith('$') and table['latex'].iloc[0].endswith('$')

# --- 测试 Gini 相关函数 ---
# 需要构造明确的真值和预测值
@pytest.fixture
def gini_data():
    """提供 Gini 测试数据"""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred_perfect = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7, 0.4, 0.6]) # 与真实值排序一致
    y_pred_random = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) # 无区分度
    y_pred_inverted = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3, 0.6, 0.4]) # 与真实值排序相反
    y_pred_good = np.array([0.2, 0.3, 0.8, 0.7, 0.1, 0.9, 0.4, 0.6]) # 较好的预测
    return y_true, y_pred_perfect, y_pred_random, y_pred_inverted, y_pred_good

def test_ginic(gini_data):
    """测试非标准化 Gini 计算"""
    y_true, y_pred_perfect, y_pred_random, y_pred_inverted, _ = gini_data
    assert np.isclose(autofe_utils.ginic(y_true, y_pred_perfect), autofe_utils.ginic(y_true, y_true)) # 完美预测等于自身 Gini
    assert np.isclose(autofe_utils.ginic(y_true, y_pred_random), 0.0) # 随机预测 Gini 接近 0
    assert autofe_utils.ginic(y_true, y_pred_inverted) < 0 # 反向预测 Gini 为负

def test_gini_normalizedc(gini_data):
    """测试标准化 Gini 计算"""
    y_true, y_pred_perfect, y_pred_random, y_pred_inverted, _ = gini_data
    assert np.isclose(autofe_utils.gini_normalizedc(y_true, y_pred_perfect), 1.0) # 完美预测标准化 Gini=1
    assert np.isclose(autofe_utils.gini_normalizedc(y_true, y_pred_random), 0.0) # 随机预测标准化 Gini=0
    assert np.isclose(autofe_utils.gini_normalizedc(y_true, y_pred_inverted), -1.0) # 完美反向预测标准化 Gini=-1

def test_gini_normalized(gini_data):
    """测试标准化 Gini 封装函数"""
    y_true, y_pred_perfect, y_pred_random, y_pred_inverted, y_pred_good = gini_data
    assert np.isclose(autofe_utils.gini_normalized(y_true, y_pred_perfect), 1.0)
    assert np.isclose(autofe_utils.gini_normalized(y_true, y_pred_random), 0.0)
    assert np.isclose(autofe_utils.gini_normalized(y_true, y_pred_inverted), 1.0) # 取了绝对值，所以是 1.0
    assert 0 < autofe_utils.gini_normalized(y_true, y_pred_good) < 1.0

def test_calc_ginis(gini_data):
    """测试为 DataFrame 多列计算 Gini"""
    y_true, _, y_pred_random, _, y_pred_good = gini_data
    df_pred = pd.DataFrame({'random': y_pred_random, 'good': y_pred_good})
    gini_scores = autofe_utils.calc_ginis(df_pred.to_numpy(), y_true)
    assert isinstance(gini_scores, np.ndarray)
    assert len(gini_scores) == 2
    assert np.isclose(gini_scores[0], 0.0) # random 列
    assert 0 < gini_scores[1] < 1.0 # good 列

# --- 测试 normalize_gini_select ---
def test_normalize_gini_select(gini_data):
    """测试基于 Gini 的特征筛选"""
    y_true, y_pred_perfect, y_pred_random, y_pred_inverted, y_pred_good = gini_data
    df = pd.DataFrame({
        'orig_feat': np.random.rand(len(y_true)),
        'new_perfect': y_pred_perfect,
        'new_good': y_pred_good,
        'new_random': y_pred_random,
        'new_inverted': y_pred_inverted # Gini < 0
    })
    y_s = pd.Series(y_true, name='target')
    ori_cols = ['orig_feat']
    top_n = 2

    selected, scores = autofe_utils.normalize_gini_select(df, y_s, ori_cols, top_n=top_n)

    assert len(selected) == top_n # 应该选出 top_n 个
    assert 'new_perfect' in selected # Gini 最高
    assert 'new_good' in selected # Gini 次高
    assert 'new_random' not in selected
    assert 'new_inverted' not in selected # Gini < 0 (如果筛选逻辑是 > 0)
    assert isinstance(scores, dict)
    assert 'new_perfect' in scores
    assert np.isclose(scores['new_perfect'], 1.0)

# --- 测试 update_time_span ---
@pytest.mark.parametrize("input_val, expected_output", [
    (None, None),
    (5, [5]),
    ([1, 3, 5], [1, 3, 5]),
    ("1, 2, 3, 4", [1, 2, 3, 4]),
    ("7", [7]),
    (" 1 , 5 ", [1, 5]),
    ("", None),
    (",", None),
    ([1, "a", 3], None), # 包含非整数
    ("1, b, 3", None),
    (1.0, None), # 类型错误
    ({"a":1}, None), # 类型错误
])
def test_update_time_span(input_val, expected_output):
    """测试时间窗口配置解析"""
    assert autofe_utils.update_time_span(input_val) == expected_output

# --- 结束 tests/autofe/test_autofe_utils.py ---