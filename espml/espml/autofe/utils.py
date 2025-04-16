# -*- coding: utf-8 -*-
"""
自动化特征工程 (AutoFE) 的工具模块 (espml)
包含用于特征工程流程的常量定义、特征名称解析、特征空间生成、
Gini 系数计算、基于 Gini 的筛选以及其他辅助函数
"""

import re # 用于更健壮的解析
import time  # 添加time模块导入
from typing import Union, Optional, Tuple, List, Dict, Any, Set # 确保导入所有需要的类型

# 添加全局LRU缓存
from functools import lru_cache

import numpy as np
import pandas as pd

# 导入日志记录器 (适配 espml 的日志系统)
from loguru import logger
# from module.util.log import get_logger # 导入,替换为 loguru
# logger = get_logger(__name__) # 获取 logger 方式
# 直接使用全局 logger 或创建子 logger
logger = logger.bind(name="autofe.utils") # 创建一个子 logger (可选)

# --- 常量定义 ---

class OPERATORCHAR:
    """定义特征名称中使用的特殊字符常量"""
    # pylint: disable=too-few-public-methods
    feature_separator: str = '|||' # 分隔特征名称
    head_character: str = '###'    # 标记特征组合的开始
    tail_character: str = '$$$'    # 标记特征组合的结束

# 操作符类型映射将参数类型元组映射到适用的操作符名称列表
OPERATORTYPES: Dict[Tuple[str, ...], List[str]] = {
    ('c',): ['count'],  # 单个分类特征的操作符
    ('n',): ['sine', 'cosine', 'pow', 'log'], # 单个数值特征的操作符 (pow 通常是 pow2 或 pow3)
    ('c', 'c'): ['crosscount', 'nunique', 'combine'], # 两个分类特征的操作符
    ('n', 'c'): ['aggmean', 'aggmax', 'aggmin', 'aggstd'], # 数值特征按分类特征聚合
    ('n', 'n'): ['add', 'sub', 'mul', 'div', 'std', 'maximize'], # 两个数值特征的操作符
    ('n', 't'): [ # 数值特征的时间序列操作符 (t 代表时间跨度)
        'diff', 'delay', 'ts_mean', 'ts_std', 'ts_cov', 'ts_corr',
        'ts_max', 'ts_min', 'ts_rank', 'ewm_mean', 'ewm_std',
        'ewm_cov', 'ewm_corr'
    ]
}

# 所有可用操作符的集合
# pylint: disable=invalid-name # 允许 OPERATORS 作为常量名
OPERATORS: Set[str] = set()
for operator_list in OPERATORTYPES.values():
    OPERATORS.update(operator_list)

# 默认的时间窗口跨度
# pylint: disable=invalid-name # 允许 TIME_SPAN 作为常量名
TIME_SPAN: List[int] = [1, 2, 3, 4, 5]

# --- 特征处理函数 ---

def split_num_cat_features(
    df: pd.DataFrame,
    ignore_columns: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    """
    将 DataFrame 的列分割为分类特征和数值特征列表

    严格遵循代码的逻辑进行判断

    Args:
        df (pd.DataFrame): 输入的数据帧
        ignore_columns (Optional[List[str]]): 需要忽略的列名列表

    Returns:
        Tuple[List[str], List[str]]: (分类特征列表, 数值特征列表)
    """
    if ignore_columns is None:
        ignore_columns = []
    ignore_set = set(ignore_columns)

    cat_columns: List[str] = []
    num_columns: List[str] = []

    logger.debug(f"开始分割特征类型,忽略列: {ignore_set}")
    for column in df.columns:
        if column in ignore_set:
            logger.trace(f"跳过忽略列: {column}")
            continue

        try:
            # 判断逻辑
            col_dtype = df[column].dtype
            col_type_name = col_dtype.name

            if col_type_name == 'bool':
                cat_columns.append(column)
                logger.trace(f"列 '{column}' (类型: {col_type_name}) 被识别为分类特征 (布尔型)")
            elif col_type_name in ['object', 'category', 'string']:
                cat_columns.append(column)
                logger.trace(f"列 '{column}' (类型: {col_type_name}) 被识别为分类特征 (类型判断)")
            elif pd.api.types.is_numeric_dtype(col_dtype):
                 nunique_val = df[column].nunique(dropna=False)
                 # 阈值判断
                 if nunique_val <= 10:
                     cat_columns.append(column)
                     logger.trace(f"列 '{column}' (类型: {col_type_name}, 唯一值: {nunique_val}) 被识别为分类特征 (唯一值 <= 10)")
                 else:
                     num_columns.append(column)
                     logger.trace(f"列 '{column}' (类型: {col_type_name}) 被识别为数值特征")
            else:
                logger.warning(f"列 '{column}' 的数据类型 '{col_type_name}' 未知或无法处理,已跳过")

        except Exception as e:
            logger.exception(f"处理列 '{column}' 时发生错误: {e}") # 使用 exception 记录完整堆栈

    logger.info(f"特征类型分割完成分类特征: {len(cat_columns)}个, 数值特征: {len(num_columns)}个")
    return cat_columns, num_columns

# 使用LRU缓存提高性能
@lru_cache(maxsize=1024)
def split_features(key: str) -> List[str]:
    """
    解析 AutoFE 生成的组合特征名称字符串

    严格按照代码的解析逻辑（基于头尾和分隔符）
    使用LRU缓存提高重复解析的性能

    Args:
        key (str): 组合特征名称字符串

    Returns:
        List[str]: 解析后的组件列表,第一个元素是操作符,后续是特征名或参数
                   如果解析失败或格式不符,返回仅包含键的列表
    """
    # 快速路径：如果key明显不是组合特征，直接返回
    if not (OPERATORCHAR.head_character in key and OPERATORCHAR.tail_character in key):
        return [key]
    
    # 使用正则表达式解析
    head_esc = re.escape(OPERATORCHAR.head_character)
    tail_esc = re.escape(OPERATORCHAR.tail_character)

    match = re.match(f"^(.*?){head_esc}(.*?){tail_esc}$", key)
    if match:
        operator = match.group(1)
        content = match.group(2)
        components = content.split(OPERATORCHAR.feature_separator)
        return [operator] + components
    else:
        return [key]

# 使用LRU缓存提高性能
@lru_cache(maxsize=1024)
def is_combination_feature(feature_name: str) -> bool:
    """
    检查给定的特征名称是否是 AutoFE 生成的组合特征
    使用缓存提高频繁调用的性能
    """
    return OPERATORCHAR.head_character in feature_name and \
           OPERATORCHAR.tail_character in feature_name

def feature_space(
    df: pd.DataFrame,
    target_name: Optional[str] = None,
    already_selected: Optional[List[str]] = None,
    time_index: Optional[str] = None,
    time_span: Optional[List[int]] = None,
    group_index: Optional[str] = None,
    max_candidate_features: int = 128
) -> List[str]:
    """
    生成候选组合特征的名称列表
    严格按照代码的生成逻辑和顺序

    Args:
        df (pd.DataFrame): 输入数据帧
        target_name (Optional[str]): 目标变量名称
        already_selected (Optional[List[str]]): 已选择特征列表
        time_index (Optional[str]): 时间索引列名
        time_span (Optional[List[int]]): 时间跨度列表
        group_index (Optional[str]): 分组列名
        max_candidate_features (int): 最大候选特征数

    Returns:
        List[str]: 生成的候选特征名称字符串列表
    """
    # 提前检查参数有效性
    if df.empty:
        logger.warning("输入DataFrame为空，无法生成特征空间")
        return []
        
    if max_candidate_features <= 0:
        logger.warning(f"无效的最大候选特征数 ({max_candidate_features})，使用默认值128")
        max_candidate_features = 128

    # 使用集合处理already_selected可以加速查找
    if already_selected is None:
        already_selected_set = set()
    else:
        already_selected_set = set(already_selected)

    # 准备需要忽略的列
    ignore_cols = [col for col in [time_index, group_index, target_name] if col is not None]
    
    # 获取特征列分类
    cat_columns, num_columns = split_num_cat_features(df, ignore_columns=ignore_cols)
    
    # 如果没有足够的列，提前返回
    if not cat_columns and not num_columns:
        logger.warning("没有有效的数值或分类特征列，无法生成特征空间")
        return []

    # 准备容器和常量
    candidate_feature: List[str] = []
    feature_separator = OPERATORCHAR.feature_separator
    head_character = OPERATORCHAR.head_character
    tail_character = OPERATORCHAR.tail_character

    logger.info(f"开始生成候选特征,最大限制: {max_candidate_features}")

    # 内部函数用于添加候选,并检查限制和重复
    def add_candidate(name: str) -> bool:
        nonlocal candidate_feature, already_selected_set
        if len(candidate_feature) < max_candidate_features and name not in already_selected_set:
            candidate_feature.append(name)
            already_selected_set.add(name)
            return True
        return False
        
    # 预先检查操作符类型映射
    op_c = OPERATORTYPES.get(('c',), [])
    op_cc = OPERATORTYPES.get(('c', 'c'), [])
    op_nc = OPERATORTYPES.get(('n', 'c'), [])
    op_n = OPERATORTYPES.get(('n',), [])
    op_nn = OPERATORTYPES.get(('n', 'n'), [])
    op_nt = OPERATORTYPES.get(('n', 't'), [])
    
    # 预先对分类特征排序以获得一致的特征生成顺序
    cat_columns = sorted(cat_columns)
    num_columns = sorted(num_columns)

    # --- 生成特征名称 ---

    # --- 分类特征 ('c',), ('c', 'c') ---
    if cat_columns:  # 只有当存在分类特征时才执行
        for i, c_col1 in enumerate(cat_columns):
            if len(candidate_feature) >= max_candidate_features: break
            
            # ('c',) - 单个分类特征操作
            for op_name in op_c:
                name = f"{op_name}{head_character}{c_col1}{tail_character}"
                if not add_candidate(name): break
                
            if len(candidate_feature) >= max_candidate_features: break
            
            # ('c', 'c') - 两个分类特征的组合
            for j in range(i + 1, len(cat_columns)):
                if len(candidate_feature) >= max_candidate_features: break
                c_col2 = cat_columns[j]
                
                # 标准化特征顺序避免重复 (a,b) vs (b,a)
                sorted_pair = sorted([c_col1, c_col2])
                pair_str = feature_separator.join(sorted_pair)
                
                for op_name in op_cc:
                    name = f"{op_name}{head_character}{pair_str}{tail_character}"
                    if not add_candidate(name): break

    # --- 数值与分类混合 ('n', 'c') ---
    if cat_columns and num_columns and len(candidate_feature) < max_candidate_features:
        # 可以限制组合数量以避免过多特征
        for c_col in cat_columns:
            if len(candidate_feature) >= max_candidate_features: break
            for n_col in num_columns:
                if len(candidate_feature) >= max_candidate_features: break
                for op_name in op_nc:
                    name = f"{op_name}{head_character}{n_col}{feature_separator}{c_col}{tail_character}"
                    if not add_candidate(name): break

    # --- 数值特征 ('n',), ('n', 'n') ---
    if num_columns and len(candidate_feature) < max_candidate_features:
        for i, n_col1 in enumerate(num_columns):
            if len(candidate_feature) >= max_candidate_features: break
            
            # ('n',) - 单个数值特征操作
            for op_name in op_n:
                name = f"{op_name}{head_character}{n_col1}{tail_character}"
                if not add_candidate(name): break
                
            if len(candidate_feature) >= max_candidate_features: break
            
            # ('n', 'n') - 两个数值特征的组合
            for j in range(i + 1, len(num_columns)):
                if len(candidate_feature) >= max_candidate_features: break
                n_col2 = num_columns[j]
                
                # 标准化特征顺序
                sorted_pair = sorted([n_col1, n_col2])
                pair_str = feature_separator.join(sorted_pair)
                
                for op_name in op_nn:
                    name = f"{op_name}{head_character}{pair_str}{tail_character}"
                    if not add_candidate(name): break

    # --- 时间序列特征 ('n', 't'), ('target', 't') ---
    if time_index is not None and len(candidate_feature) < max_candidate_features:
        # 使用配置的时间跨度或默认值
        effective_time_span = time_span if time_span is not None else TIME_SPAN
        logger.debug(f"生成时间序列特征,时间跨度: {effective_time_span}")
        
        # 检查时间跨度是否有效
        if not effective_time_span:
            logger.warning("时间跨度为空，跳过时间序列特征生成")
        else:
            # ('n', 't') - 数值特征的时间序列操作
            for n_col in num_columns:
                if len(candidate_feature) >= max_candidate_features: break
                for op_name in op_nt:
                    if len(candidate_feature) >= max_candidate_features: break
                    for d in effective_time_span:
                        name = f"{op_name}{head_character}{n_col}{feature_separator}{d}{tail_character}"
                        if not add_candidate(name): break
                        
            # 目标滞后特征 ('target', 't')
            if (target_name and target_name in df.columns and 
                len(candidate_feature) < max_candidate_features):
                for d in effective_time_span:
                    # delay操作使用负时间跨度
                    name = f"delay{head_character}{target_name}{feature_separator}{-d}{tail_character}"
                    if not add_candidate(name): break

    # 记录特征生成结果
    if not candidate_feature:
        logger.warning("未能生成任何候选特征")
    else:
        logger.info(f"候选特征生成完成,共生成 {len(candidate_feature)} 个特征")
        
    # 确保返回列表的稳定性
    return candidate_feature

def feature2table(names: List[str]) -> pd.DataFrame:
    """
    将特征名称列表转换为包含 ID 和 LaTeX 公式的 DataFrame
    """
    res = []
    logger.debug(f"开始将 {len(names)} 个特征名称转换为表格...")
    for i, name in enumerate(names):
        try:
            formula = name2formula(name)
            res.append([i + 1, formula])
        except Exception as e:
            logger.error(f"转换特征 '{name}' 为 LaTeX 公式时出错: {e}")
            res.append([i + 1, f"转换错误: {name}"])

    column_names = ['id', 'latex']
    df = pd.DataFrame(res, columns=column_names)
    logger.debug("特征名称到表格转换完成")
    return df

def name2formula(input_str: str) -> str:
    """
    将 AutoFE 特征名称递归地转换为 LaTeX 公式字符串
    """
    # logger.trace(f"转换特征名称为 LaTeX 公式: {input_str}")

    if not is_combination_feature(input_str):
        return input_str.replace('_', r'\_')

    parts = split_features(input_str)
    if not parts or len(parts) < 2:
        logger.warning(f"解析特征名称 '{input_str}' 失败或组件不足,无法转换为公式")
        return input_str.replace('_', r'\_')

    op_name = parts[0]
    sub_features = parts[1:]
    sub_formulas = [name2formula(feat) for feat in sub_features]

    formula = ""
    num_subs = len(sub_formulas)

    # LaTeX 格式化映射 (操作符名称 -> (LaTeX 模板, 参数数量))
    # 使用 {} 作为子公式的占位符
    op_latex_map = {
        # Binary Numeric
        'add': (r'({} + {})', 2), 'sub': (r'({} - {})', 2),
        'mul': (r'({} \times {})', 2), 'div': (r'\frac{{{}}}{{{}}}', 2),
        'std': (r'\text{{Std}}({}, {})', 2), 'maximize': (r'\max({}, {})', 2),
        'ts_cov': (r'\text{{TSCov}}({}, {})', 2), 'ts_corr': (r'\text{{TSCorr}}({}, {})', 2),
        'ewm_cov': (r'\text{{EWMCov}}({}, {})', 2), 'ewm_corr': (r'\text{{EWMCorr}}({}, {})', 2),
        # Unary Numeric
        'sine': (r'\sin({})', 1), 'cosine': (r'\cos({})', 1),
        'pow': (r'({})^2', 1), # 假设 pow 是平方
        'log': (r'\log({})', 1),
        # Categorical
        'count': (r'\text{{Count}}({})', 1),
        'crosscount': (r'\text{{CrossCount}}({}, {})', 2),
        'nunique': (r'\text{{NUnique}}({} \text{{ by }} {})', 2), # 假设第二个是分组依据
        'combine': (r'\text{{Combine}}({}, {})', 2),
        # Numeric by Categorical Aggregation
        'aggmean': (r'\text{{Mean}}({} \text{{ by }} {})', 2),
        'aggmax': (r'\text{{Max}}({} \text{{ by }} {})', 2),
        'aggmin': (r'\text{{Min}}({} \text{{ by }} {})', 2),
        'aggstd': (r'\text{{Std}}({} \text{{ by }} {})', 2),
        # Time Series (assuming second arg is time span 'd')
        'diff': (r'\text{{Diff}}({}, {})', 2),
        'delay': (r'\text{{Delay}}({}, {})', 2), # 注意 delay 的参数是 -d
        'ts_mean': (r'\text{{TSMean}}({}, {})', 2),
        'ts_std': (r'\text{{TSStd}}({}, {})', 2),
        'ts_max': (r'\text{{TSMax}}({}, {})', 2),
        'ts_min': (r'\text{{TSMin}}({}, {})', 2),
        'ts_rank': (r'\text{{TSRank}}({}, {})', 2),
        'ewm_mean': (r'\text{{EWMMean}}({}, {})', 2),
        'ewm_std': (r'\text{{EWMStd}}({}, {})', 2),
    }

    try:
        if op_name in op_latex_map:
            fmt_str, expected_args = op_latex_map[op_name]
            if num_subs == expected_args:
                formula = fmt_str.format(*sub_formulas) # 使用 * 解包
            else:
                raise ValueError(f"操作符 '{op_name}' 需要 {expected_args} 个参数,但得到 {num_subs} 个")
        else:
            logger.warning(f"未知操作符 '{op_name}',使用通用公式表示")
            op_name_escaped = op_name.replace('_', r'\_')
            sub_formulas_str = ', '.join(sub_formulas)
            formula = fr"\text{{{op_name_escaped}}}({sub_formulas_str})"
    except Exception as e:
        logger.exception(f"构建公式时发生错误Input: '{input_str}', Op: '{op_name}', Subs: {sub_formulas}, Error: {e}")
        input_escaped = input_str.replace('_', r'\_')
        formula = fr"\text{{Error}}({input_escaped})"

    return f"${formula}$"

# --- Gini 系数计算函数 ---
def ginic(actual: np.ndarray, pred: np.ndarray) -> float:
    """计算（非标准化的）Gini 系数,增强类型和数值检查"""
    try:
        # 如果输入已经是numpy数组且是float，跳过转换
        if (isinstance(actual, np.ndarray) and isinstance(pred, np.ndarray) and 
            np.issubdtype(actual.dtype, np.floating) and np.issubdtype(pred.dtype, np.floating)):
            actual_f = actual
            pred_f = pred
        else:
            # 强制转换为 float64 numpy array
            actual_f = np.asarray(actual, dtype=np.float64)
            pred_f = np.asarray(pred, dtype=np.float64)

        # 验证形状和维度
        if actual_f.shape != pred_f.shape or actual_f.ndim != 1 or len(actual_f) == 0:
             # logger.trace("Gini 输入形状不匹配或为空")
             return 0.0

        # 快速路径：如果数据完全有限且非空，直接计算
        if np.all(np.isfinite(actual_f)) and np.all(np.isfinite(pred_f)):
            n = len(actual_f)
            # 检查实际值是否全为零或接近零
            if np.all(np.isclose(actual_f, 0)):
                return 0.0
                
            # 高效计算Gini
            a_s = actual_f[np.argsort(pred_f)]
            a_c = np.cumsum(a_s)
            a_s_sum = np.sum(a_s)
            if np.isclose(a_s_sum, 0):
                return 0.0
                
            gini_sum = np.sum(a_c) / a_s_sum - (n + 1) / 2.0
            return float(gini_sum / n)
        
        # 处理含有NaN或Inf的情况
        valid_mask = np.isfinite(actual_f) & np.isfinite(pred_f)
        if not np.all(valid_mask):
            logger.warning("Gini 输入包含 NaN 或 Inf 值,将尝试移除")
            actual_f = actual_f[valid_mask]
            pred_f = pred_f[valid_mask]
            if len(actual_f) == 0: 
                return 0.0  # 如果移除后为空
        
        # 检查实际值是否全为零或接近零
        if np.all(np.isclose(actual_f, 0)):
             return 0.0

        n = len(actual_f)
        a_s = actual_f[np.argsort(pred_f)]
        a_c = np.cumsum(a_s)
        a_s_sum = np.sum(a_s)
        if np.isclose(a_s_sum, 0):
             return 0.0

        gini_sum = np.sum(a_c) / a_s_sum - (n + 1) / 2.0
        return float(gini_sum / n)
    except Exception as e:
        logger.exception(f"计算 Gini 系数时出错: {e}")
        return 0.0

def gini_normalizedc(a: np.ndarray, p: np.ndarray) -> float:
    """计算标准化的 Gini 系数"""
    gini_pred = ginic(a, p)
    gini_perfect = ginic(a, a) # 使用相同的健壮 ginic 函数
    if np.isclose(gini_perfect, 0):
        # logger.trace("标准化 Gini 计算失败完美 Gini 系数为零")
        return 0.0
    else:
        # 使用安全除法
        return float(gini_pred / gini_perfect) if gini_perfect != 0 else 0.0

def gini_normalized(y_true: np.ndarray, y_pred: np.ndarray, empty_slice: Optional[np.ndarray] = None) -> float:
    """
    计算标准化的 Gini 指数 (封装)
    
    Args:
        y_true (np.ndarray): 真实值数组
        y_pred (np.ndarray): 预测值数组
        empty_slice (Optional[np.ndarray]): 未使用的参数，仅为保持API兼容性
            这个参数可能在旧版本中有用，现在可以安全忽略
    
    Returns:
        float: 标准化的Gini指数值，范围通常在[0, 1]
    """
    # empty_slice参数未使用，保留是为了接口兼容性
    # 在代码重构时可以考虑完全移除此参数
    try:
        # 快速检查输入是否都为有限值(避免额外的数据转换)
        if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
            # 两个都是数组，检查是否都是float
            if np.issubdtype(y_true.dtype, np.floating) and np.issubdtype(y_pred.dtype, np.floating):
                # 已经是浮点数，直接检查是否有NaN/Inf
                if np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred)):
                    # 如果都是单列，无需转置直接计算
                    if y_true.ndim == 1 and y_pred.ndim == 1:
                        return gini_normalizedc(y_true, y_pred)
            
        # 常规路径：将输入转换为二维数组
        y_true_f = np.asarray(y_true, dtype=np.float64)
        y_pred_f = np.asarray(y_pred, dtype=np.float64)
    except ValueError:
         logger.warning("无法将 Gini 输入转换为 float64")
         return 0.0

    # 确保是二维数组
    if y_true_f.ndim == 1: 
        y_true_f = y_true_f[:, np.newaxis]
    if y_pred_f.ndim == 1: 
        y_pred_f = y_pred_f[:, np.newaxis]

    n_true_cols = y_true_f.shape[1]
    n_pred_cols = y_pred_f.shape[1]
    
    # 形状验证
    if n_true_cols != 1 and n_pred_cols != 1 and n_true_cols != n_pred_cols:
        raise AssertionError(f"形状不匹配: True {y_true_f.shape}, Pred {y_pred_f.shape}")

    # 优化：如果只有一列，直接计算避免循环
    if n_true_cols == 1 and n_pred_cols == 1:
        return gini_normalizedc(y_true_f[:, 0], y_pred_f[:, 0])

    # 多列处理
    ginis = []
    for i in range(n_true_cols):
        j = min(i, n_pred_cols - 1)
        yt = y_true_f[:, i]
        yp = y_pred_f[:, j]
        # ginic 内部会处理 NaN
        ginis.append(gini_normalizedc(yt, yp))

    if not ginis: 
        return 0.0
    
    # 计算平均绝对值
    mean_abs_gini = np.abs(np.array(ginis)).mean()
    return float(mean_abs_gini)

def calc_ginis(data: Union[pd.DataFrame, np.ndarray], target: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """为数据中的每一数值列计算标准化的 Gini 系数"""
    if isinstance(data, pd.DataFrame):
        numeric_cols = data.select_dtypes(include=np.number)
        if numeric_cols.empty:
            logger.warning("输入 DataFrame 没有数值列用于 Gini 计算")
            return np.array([], dtype=np.float32)
        data_np = numeric_cols.to_numpy()
        columns = numeric_cols.columns.tolist()
    elif isinstance(data, np.ndarray):
        if not pd.api.types.is_numeric_dtype(data.dtype):
             logger.warning("输入 Numpy 数组不是数值类型,无法计算 Gini")
             return np.array([], dtype=np.float32)
        data_np = data
        columns = [f"col_{i}" for i in range(data.shape[1])]
    else:
        raise TypeError("输入数据必须是 DataFrame 或 Numpy 数组")

    if isinstance(target, pd.Series): target_np = target.to_numpy()
    elif isinstance(target, np.ndarray): target_np = target
    else: raise TypeError("目标必须是 Series 或 Numpy 数组")

    if target_np.ndim > 1 and target_np.shape[1] != 1: target_np = target_np[:, 0]
    target_np = target_np.flatten()

    if data_np.shape[0] != target_np.shape[0]:
         raise ValueError(f"数据 ({data_np.shape[0]}) 和目标 ({target_np.shape[0]}) 的样本数不匹配")

    num_cols = data_np.shape[1]
    scores = np.zeros(num_cols, dtype=np.float32)
    
    # 预处理目标变量，处理NaN
    target_mask = ~np.isnan(target_np)
    if not np.all(target_mask):
        target_clean = target_np[target_mask]
        logger.warning(f"目标变量包含 {np.sum(~target_mask)} 个NaN值，已过滤")
    else:
        target_clean = target_np
    
    if len(target_clean) == 0:
        logger.error("过滤NaN后目标变量为空，无法计算Gini")
        return np.zeros(num_cols, dtype=np.float32)

    # 多线程计算Gini
    from concurrent.futures import ThreadPoolExecutor
    import os
    
    def calculate_column_gini(col_idx):
        try:
            col_data = data_np[:, col_idx]
            
            # 处理特征列的NaN
            col_mask = ~np.isnan(col_data)
            col_target_mask = target_mask & col_mask
            
            if np.sum(col_target_mask) < 10:  # 至少需要10个有效样本
                logger.warning(f"列 '{columns[col_idx]}' 有效样本不足，Gini设为0")
                return 0.0
                
            # 快速路径：当没有NaN时，避免不必要的数据复制
            if np.all(col_mask) and np.all(target_mask):
                return gini_normalized(target_np, col_data)
                
            col_clean = col_data[col_target_mask]
            target_for_col = target_clean[col_target_mask if not np.all(target_mask) else slice(None)]
            
            # 直接调用 gini_normalized (它内部处理 NaN)
            return gini_normalized(target_for_col, col_clean)
        except Exception as e:
            logger.error(f"计算列 '{columns[col_idx]}' (索引 {col_idx}) 的 Gini 分数时出错: {e}")
            return 0.0
    
    # 使用线程池并行计算
    # 根据特征数量动态调整线程数，避免为少量特征创建过多线程
    n_jobs = min(os.cpu_count() or 4, max(1, num_cols // 2))
    
    if num_cols <= 3:  # 对于少量列，直接串行计算
        results = [calculate_column_gini(i) for i in range(num_cols)]
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(calculate_column_gini, range(num_cols)))
    
    # 填充结果
    for i, score in enumerate(results):
        scores[i] = score

    return scores

# --- 基于 Gini 的特征筛选 ---
def normalize_gini_select(
    df: pd.DataFrame,
    y_train: pd.Series,
    ori_columns: List[str],
    top_n: int = 20
) -> Tuple[List[str], Dict[str, float]]:
    """使用标准化的 Gini 指数进行特征筛选"""
    logger.info(f"开始基于 Gini 的特征筛选,最多选择 {top_n} 个新特征...")
    best_features: List[str] = []
    gini_scores_dict: Dict[str, float] = {}

    if df.empty:
        logger.warning("输入的 DataFrame 为空,无法进行 Gini 筛选")
        return best_features, gini_scores_dict
    
    # 安全检查：确保y_train非空
    if y_train.empty:
        logger.warning("目标变量为空,无法进行 Gini 筛选")
        return best_features, gini_scores_dict

    # 确保ori_columns是有效的列表
    if ori_columns is None:
        ori_columns = []
    
    # 将列名和ori_columns都转换为字符串集合以提高查找效率
    ori_columns_set = set(map(str, ori_columns))
    
    # 高效选取数值列
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    # 快速路径：如果没有数值列则提前返回
    if len(numeric_cols) == 0:
        logger.warning("输入 DataFrame 没有数值列,无法进行 Gini 筛选")
        return best_features, gini_scores_dict
        
    # 仅保留含有足够非缺失值的数值列 (并行计算列的非缺失率)
    valid_columns = []
    missing_threshold = 0.5  # 允许最多50%的缺失值
    
    # 使用向量化操作一次性计算所有列的非缺失率
    non_missing_ratios = df[numeric_cols].notna().mean()
    
    # 基于阈值过滤列
    valid_numeric_cols = non_missing_ratios[non_missing_ratios >= missing_threshold].index
    
    if len(valid_numeric_cols) == 0:
        logger.warning("所有数值列的缺失值比例都超过阈值,无法进行 Gini 筛选")
        return best_features, gini_scores_dict
        
    # 记录过滤的列数
    filtered_count = len(numeric_cols) - len(valid_numeric_cols)
    if filtered_count > 0:
        logger.debug(f"由于缺失值过多跳过了 {filtered_count} 列进行Gini计算")
        
    # 只保留有效列
    df_numeric = df[valid_numeric_cols]
    
    # 确保列名都是字符串
    df_numeric.columns = df_numeric.columns.astype(str)

    try:
        # 计时
        start_time = time.perf_counter()
        
        # 转换为numpy加速计算
        y_train_np = y_train.to_numpy()
        
        # 计算Gini分数
        train_ginis = calc_ginis(df_numeric, y_train_np)
        
        # 构建分数字典
        gini_scores_dict = dict(zip(df_numeric.columns, map(float, train_ginis)))
        
        logger.debug(f"Gini计算完成，耗时: {time.perf_counter() - start_time:.2f}秒")

        # 先过滤无效值，再排序
        valid_scores = {
            feature: score 
            for feature, score in gini_scores_dict.items() 
            if pd.notna(score) and np.isfinite(score) and score > 1e-9 and feature not in ori_columns_set
        }
        
        # 如果没有有效特征，直接返回
        if not valid_scores:
            logger.warning("没有找到有效的特征（所有特征Gini分数过低或无效）")
            return best_features, gini_scores_dict
        
        # 按Gini分数降序排列
        sorted_scores = sorted(valid_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 动态调整取值数量，不超过有效特征数
        actual_top_n = min(top_n, len(sorted_scores))
        if actual_top_n < top_n:
            logger.debug(f"有效特征数 ({actual_top_n}) 少于请求的top_n ({top_n})")
        
        # 选取前top_n个特征
        best_features = [feature for feature, _ in sorted_scores[:actual_top_n]]
        
        # 记录最高和最低分数的特征，便于调试
        if sorted_scores:
            top_feature, top_score = sorted_scores[0]
            bottom_feature, bottom_score = sorted_scores[-1]
            logger.debug(f"最高分数特征: '{top_feature}' (分数: {top_score:.6f})")
            logger.debug(f"最低分数特征: '{bottom_feature}' (分数: {bottom_score:.6f})")

    except Exception as e:
        logger.exception(f"执行 Gini 特征筛选时出错: {e}")
        return [], gini_scores_dict # 返回空列表

    logger.info(f"基于 Gini 的特征筛选完成,选中 {len(best_features)}/{len(valid_scores)} 个新特征")
    return best_features, gini_scores_dict

# --- 时间窗口配置解析 ---
def update_time_span(config_value: Any) -> Optional[List[int]]:
    """解析时间窗口配置值,将其转换为整数列表"""
    # logger.trace(f"解析时间窗口配置值: {config_value} (类型: {type(config_value)})")
    if config_value is None: return None
    elif isinstance(config_value, int): return [config_value]
    elif isinstance(config_value, list):
        try: return [int(n) for n in config_value if int(n) > 0] # 确保是正整数
        except (ValueError, TypeError): logger.error(f"时间窗口列表含非正整数: {config_value}"); return None
    elif isinstance(config_value, str):
        try:
            int_list = [int(n.strip()) for n in config_value.split(',') if n.strip() and int(n.strip()) > 0]
            if not int_list:
                 try:
                     single_int = int(config_value.strip())
                     return [single_int] if single_int > 0 else None
                 except ValueError: logger.warning(f"时间窗口字符串无效: '{config_value}'"); return None
            return int_list
        except ValueError: logger.error(f"时间窗口字符串含无法转为正整数的部分: '{config_value}'"); return None
    else:
        logger.error(f"不支持的时间窗口配置类型: {type(config_value)}"); return None

logger.info("AutoFE 工具函数模块 (espml.autofe.utils) 加载完成")