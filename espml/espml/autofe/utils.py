# -*- coding: utf-8 -*-
"""
自动化特征工程 (AutoFE) 的工具模块 (espml)
包含用于特征工程流程的常量定义、特征名称解析、特征空间生成、
Gini 系数计算、基于 Gini 的筛选以及其他辅助函数
"""

import re # 用于更健壮的解析
from collections import Counter
from typing import Union, Optional, Tuple, List, Dict, Any, Set # 确保导入所有需要的类型

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
# pylint: disable=invalid-name # 允许 OPEATORS 作为常量名
OPERTORS: Set[str] = set()
for operator_list in OPERATORTYPES.values():
    OPERTORS.update(operator_list)

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

def split_features(key: str) -> List[str]:
    """
    解析 AutoFE 生成的组合特征名称字符串

    严格按照代码的解析逻辑（基于头尾和分隔符）

    Args:
        key (str): 组合特征名称字符串

    Returns:
        List[str]: 解析后的组件列表,第一个元素是操作符,后续是特征名或参数
                   如果解析失败或格式不符,返回仅包含键的列表
    """
    # 代码的解析逻辑 (优先使用更健壮的正则方法,如果确认代码使用它)
    # 此处使用之前确认的正则方法,因为它更可能健壮地处理各种情况
    head_esc = re.escape(OPERATORCHAR.head_character)
    tail_esc = re.escape(OPERATORCHAR.tail_character)
    # sep_esc = re.escape(OPERATORCHAR.feature_separator) # 分隔符通常不需要转义

    match = re.match(f"^(.*?){head_esc}(.*?){tail_esc}$", key)
    if match:
        operator = match.group(1)
        content = match.group(2)
        components = content.split(OPERATORCHAR.feature_separator)
        # 代码可能没有清理空字符串,
        # cleaned_components = [c for c in components if c] # 不清理
        # logger.trace(f"解析特征 '{key}' -> Op: '{operator}', Comp: {components}")
        return [operator] + components
    else:
        # logger.trace(f"特征名称 '{key}' 不符合组合特征格式,返回原名称")
        return [key]

def is_combination_feature(feature_name: str) -> bool:
    """
    检查给定的特征名称是否是 AutoFE 生成的组合特征
    严格按照代码的逻辑判断（检查头尾标记）
    """
    # 假设代码就是这样判断的
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
    if already_selected is None:
        already_selected_set = set()
    else:
        already_selected_set = set(already_selected)

    ignore_cols = [col for col in [time_index, group_index, target_name] if col is not None]
    cat_columns, num_columns = split_num_cat_features(df, ignore_columns=ignore_cols)

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

    # 生成顺序严格按照代码逻辑（此处假设是 C -> N+C -> N -> T 的顺序）

    # --- 分类特征 ('c',), ('c', 'c') ---
    for i, c_col1 in enumerate(cat_columns):
        if len(candidate_feature) >= max_candidate_features: break
        # ('c',)
        for op_name in OPERATORTYPES.get(('c',), []):
            name = f"{op_name}{head_character}{c_col1}{tail_character}"
            if not add_candidate(name): break # 如果达到上限,停止为 c_col1 生成
        if len(candidate_feature) >= max_candidate_features: break
        # ('c', 'c')
        for j in range(i + 1, len(cat_columns)): # 避免重复对 (a,b) vs (b,a)
             if len(candidate_feature) >= max_candidate_features: break
             c_col2 = cat_columns[j]
             sorted_pair = sorted([c_col1, c_col2]) # 标准化顺序
             pair_str = feature_separator.join(sorted_pair)
             for op_name in OPERATORTYPES.get(('c', 'c'), []):
                 name = f"{op_name}{head_character}{pair_str}{tail_character}"
                 if not add_candidate(name): break # 如果达到上限,停止为此对生成

    # --- 数值与分类混合 ('n', 'c') ---
    if len(candidate_feature) < max_candidate_features:
        for c_col in cat_columns:
            if len(candidate_feature) >= max_candidate_features: break
            for n_col in num_columns:
                if len(candidate_feature) >= max_candidate_features: break
                for op_name in OPERATORTYPES.get(('n', 'c'), []):
                    name = f"{op_name}{head_character}{n_col}{feature_separator}{c_col}{tail_character}"
                    if not add_candidate(name): break

    # --- 数值特征 ('n',), ('n', 'n') ---
    if len(candidate_feature) < max_candidate_features:
        for i, n_col1 in enumerate(num_columns):
            if len(candidate_feature) >= max_candidate_features: break
            # ('n',)
            for op_name in OPERATORTYPES.get(('n',), []):
                name = f"{op_name}{head_character}{n_col1}{tail_character}"
                if not add_candidate(name): break
            if len(candidate_feature) >= max_candidate_features: break
            # ('n', 'n')
            for j in range(i + 1, len(num_columns)):
                if len(candidate_feature) >= max_candidate_features: break
                n_col2 = num_columns[j]
                sorted_pair = sorted([n_col1, n_col2]) # 标准化顺序
                pair_str = feature_separator.join(sorted_pair)
                for op_name in OPERATORTYPES.get(('n', 'n'), []):
                    name = f"{op_name}{head_character}{pair_str}{tail_character}"
                    if not add_candidate(name): break

    # --- 时间序列特征 ('n', 't'), ('target', 't') ---
    if time_index is not None and len(candidate_feature) < max_candidate_features:
        effective_time_span = time_span if time_span is not None else TIME_SPAN
        logger.debug(f"生成时间序列特征,时间跨度: {effective_time_span}")
        # ('n', 't')
        for n_col in num_columns:
            if len(candidate_feature) >= max_candidate_features: break
            for op_name in OPERATORTYPES.get(('n', 't'), []):
                if len(candidate_feature) >= max_candidate_features: break
                for d in effective_time_span:
                    name = f"{op_name}{head_character}{n_col}{feature_separator}{d}{tail_character}"
                    if not add_candidate(name): break
        # target delay ('target', 't')
        if target_name and target_name in df.columns and len(candidate_feature) < max_candidate_features:
            for d in effective_time_span:
                # 使用 'delay' 操作符名称,参数是负的时间跨度 (与代码一致)
                name = f"delay{head_character}{target_name}{feature_separator}{-d}{tail_character}"
                if not add_candidate(name): break

    logger.info(f"候选特征生成完成,共生成 {len(candidate_feature)} 个特征 (已应用限制)")
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
        # 强制转换为 float64 numpy array
        actual_f = np.asarray(actual, dtype=np.float64)
        pred_f = np.asarray(pred, dtype=np.float64)

        if actual_f.shape != pred_f.shape or actual_f.ndim != 1 or len(actual_f) == 0:
             # logger.trace("Gini 输入形状不匹配或为空")
             return 0.0
        # 检查 NaN 或 Inf
        if not np.all(np.isfinite(actual_f)) or not np.all(np.isfinite(pred_f)):
             logger.warning("Gini 输入包含 NaN 或 Inf 值,将尝试移除")
             valid_mask = np.isfinite(actual_f) & np.isfinite(pred_f)
             actual_f = actual_f[valid_mask]
             pred_f = pred_f[valid_mask]
             if len(actual_f) == 0: return 0.0 # 如果移除后为空

        # 检查实际值是否全为零或接近零
        if np.all(np.isclose(actual_f, 0)):
             # logger.trace("Gini 计算的实际值全为零")
             return 0.0

        n = len(actual_f)
        a_s = actual_f[np.argsort(pred_f)]
        a_c = a_s.cumsum()
        a_s_sum = a_s.sum()
        if np.isclose(a_s_sum, 0):
             # logger.trace("Gini 计算的排序后实际值总和为零")
             return 0.0

        gini_sum = a_c.sum() / a_s_sum - (n + 1) / 2.0
        # 避免除以零 (虽然 n>0 时不可能)
        return float(gini_sum / n) if n > 0 else 0.0
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
    """计算标准化的 Gini 指数 (封装)"""
    try:
        y_true_f = np.asarray(y_true, dtype=np.float64)
        y_pred_f = np.asarray(y_pred, dtype=np.float64)
    except ValueError:
         logger.warning("无法将 Gini 输入转换为 float64")
         return 0.0

    if y_true_f.ndim == 1: y_true_f = y_true_f[:, np.newaxis]
    if y_pred_f.ndim == 1: y_pred_f = y_pred_f[:, np.newaxis]

    n_true_cols = y_true_f.shape[1]; n_pred_cols = y_pred_f.shape[1]
    if n_true_cols != 1 and n_pred_cols != 1 and n_true_cols != n_pred_cols:
        raise AssertionError(f"形状不匹配: True {y_true_f.shape}, Pred {y_pred_f.shape}")

    ginis = []
    for i in range(n_true_cols):
        j = min(i, n_pred_cols - 1)
        yt = y_true_f[:, i]
        yp = y_pred_f[:, j]
        # ginic 内部会处理 NaN
        ginis.append(gini_normalizedc(yt, yp))

    if not ginis: return 0.0
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

    for n in range(num_cols):
        try:
            # 直接调用 gini_normalized (它内部处理 NaN)
            scores[n] = gini_normalized(target_np, data_np[:, n])
        except Exception as e:
            logger.error(f"计算列 '{columns[n]}' (索引 {n}) 的 Gini 分数时出错: {e}")
            scores[n] = 0.0

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

    df_numeric = df.select_dtypes(include=np.number)
    if df_numeric.empty:
         logger.warning("DataFrame 中没有数值列可用于 Gini 计算")
         return best_features, gini_scores_dict

    # 确保列名和 ori_columns 都是字符串
    df_numeric.columns = df_numeric.columns.astype(str)
    ori_columns_set = set(map(str, ori_columns))

    try:
        y_train_np = y_train.to_numpy()
        train_ginis = calc_ginis(df_numeric, y_train_np)
        gini_scores_dict = dict(zip(df_numeric.columns, map(float, train_ginis)))

        dt_sorted = dict(sorted(
            gini_scores_dict.items(),
            key=lambda item: float('-inf') if pd.isna(item[1]) or np.isinf(item[1]) else item[1],
            reverse=True
        ))

        for feature, score in dt_sorted.items():
            if len(best_features) >= top_n: break
            # 确保分数有效且大于0 (Gini>0才有区分度)
            if feature not in ori_columns_set and pd.notna(score) and np.isfinite(score) and score > 1e-9: # 用一个小的正数阈值
                 best_features.append(feature)

    except Exception as e:
        logger.exception(f"执行 Gini 特征筛选时出错: {e}")
        return [], gini_scores_dict # 返回空列表

    logger.info(f"基于 Gini 的特征筛选完成,选中 {len(best_features)} 个新特征")
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