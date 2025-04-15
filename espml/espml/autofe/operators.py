# -*- coding: utf-8 -*-

"""
自动化特征工程 (AutoFE) 的操作符实现模块 (espml)
包含用于特征转换和生成的具体函数实现
"""

from collections import Counter
from typing import Union, Optional, Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import warnings
from loguru import logger

# 导入本项目定义的工具和常量
from espml.util import utils as common_utils
from espml.autofe import utils as autofe_utils # 可能需要导入常量,但函数本身不应依赖其他 autofe 模块

logger = logger.bind(name="autofe.operators") # 创建子 logger

# --- 内部辅助函数 (如果代码有) ---
# 例如,安全除法或类型检查,如果它们没有放在通用 utils 中
def _safe_divide_local(numerator: pd.Series, denominator: pd.Series, default: float = np.nan) -> pd.Series:
    """本地安全除法,防止 utils 模块循环依赖（如果 utils 导入了 operators）"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator.astype(float) / denominator.astype(float)
    result[~np.isfinite(result)] = default
    return result

def _get_numeric_series(features: Union[pd.Series, pd.DataFrame], index: int = 0) -> pd.Series:
    """从输入中获取指定索引的数值 Series"""
    s: pd.Series
    if isinstance(features, pd.DataFrame):
        if features.shape[1] <= index:
             raise IndexError(f"DataFrame 缺少索引为 {index} 的列")
        s = features.iloc[:, index]
    elif isinstance(features, pd.Series) and index == 0:
        s = features
    else:
        raise TypeError("输入必须是 Series 或 DataFrame")
    return pd.to_numeric(s, errors='coerce') # 转换为数值,无效值为 NaN


# --- 单特征操作符 ('n',) ---

def sine(features: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """计算特征的正弦值"""
    s = _get_numeric_series(features, 0)
    # logger.trace("Calculating sine...")
    # 代码可能在操作前填充 NaN,此处假设保留 NaN
    return np.sin(s) # NaN 输入 -> NaN 输出

def cosine(features: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """计算特征的余弦值"""
    s = _get_numeric_series(features, 0)
    # logger.trace("Calculating cosine...")
    return np.cos(s)

def pow(features: Union[pd.Series, pd.DataFrame], exponent: float = 2.0) -> pd.Series:
    """计算特征的幂次方（默认为平方）,并裁剪"""
    # 注意 Transform 类中的 eval 调用可能不直接支持传递 exponent 参数
    # 代码可能只实现了 pow2 或 pow3
    # 此处假设 exponent 是固定的或通过其他方式处理,暂时只实现平方
    # 如果需要支持参数,Transform 类中的 eval 调用需要修改
    logger.warning("Operator 'pow' 实现假定指数为 2.0,请检查代码逻辑")
    s = _get_numeric_series(features, 0)
    # logger.trace("Calculating pow (exponent=2.0)...")
    powered_feature = s.pow(2.0)
    # 裁剪值范围
    clip_lower, clip_upper = -1.0e9, 1.0e9
    return powered_feature.clip(lower=clip_lower, upper=clip_upper).fillna(0) # 填充 NaN

def log(features: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """计算特征的自然对数（log1p(|x|)）,并裁剪"""
    s = _get_numeric_series(features, 0)
    # logger.trace("Calculating log...")
    #  log(1 + |x|) 逻辑
    log_feature = np.log1p(np.abs(s)) # NaN 输入 -> NaN 输出
    clip_lower, clip_upper = -1.0e9, 1.0e9
    return log_feature.clip(lower=clip_lower, upper=clip_upper).fillna(0) # 填充 NaN


# --- 单特征操作符 ('c',) ---

def count(features: Union[pd.Series, pd.DataFrame], intermediate: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """计算分类特征的频率（计数值）"""
    s: pd.Series
    if isinstance(features, pd.DataFrame):
        if features.empty: return pd.Series(dtype=int) if not intermediate else (pd.Series(dtype=int), pd.Series(dtype=int))
        s = features.iloc[:, 0]
    elif isinstance(features, pd.Series):
        if features.empty: return pd.Series(dtype=int) if not intermediate else (pd.Series(dtype=int), pd.Series(dtype=int))
        s = features
    else: raise TypeError("输入 'features' 必须是 pandas Series 或 DataFrame")

    # logger.trace("Calculating count...")
    try:
        # dropna=False 使得 NaN 也被计数
        intermediate_stat = s.value_counts(dropna=False).rename('count')
    except Exception as e:
        logger.exception(f"计算值计数时出错: {e}")
        empty_series = pd.Series(np.nan, index=s.index).fillna(0).astype(int)
        return empty_series if not intermediate else (empty_series, pd.Series(dtype=int))

    feature_mapped = s.map(intermediate_stat).fillna(0).astype(int)

    if intermediate:
        return feature_mapped, intermediate_stat
    return feature_mapped


# --- 双特征操作符 ('c', 'c') ---

def crosscount(features: pd.DataFrame, intermediate: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """计算两个分类特征组合的频率（计数值）"""
    if not isinstance(features, pd.DataFrame) or features.shape[1] < 2: raise ValueError("CrossCount 输入必须是至少包含两列的 DataFrame")
    if features.empty: return pd.Series(dtype=int) if not intermediate else (pd.Series(dtype=int), pd.Series(dtype=int))

    col1, col2 = features.columns[0], features.columns[1]
    # logger.trace(f"Calculating crosscount between '{col1}' and '{col2}'...")
    try:
        # 使用 transform 计算分组大小
        intermediate_stat = features.groupby([col1, col2]).size().rename('crosscount')
        # 合并回原 DataFrame (保持索引)
        # 使用 merge 实现
        # 需要 reset_index 以便 merge
        # temp_df = features[[col1, col2]].reset_index()
        # merge_df = pd.merge(temp_df, intermediate_stat.reset_index(), on=[col1, col2], how='left')
        # merge_df = merge_df.set_index(temp_df.columns[0]) # 恢复索引名 (假设是'index')
        # feature_merged = merge_df['crosscount'].fillna(0).astype(int)

        # 或者使用 transform (如果不需要 intermediate_stat 的精确格式)
        # 使用 transform 更高效且直接保留索引
        feature_merged = features.groupby([col1, col2])[col1].transform('size').rename('crosscount')
        feature_merged = feature_merged.fillna(0).astype(int)

    except Exception as e:
        logger.exception(f"计算 crosscount 时出错: {e}")
        empty_series = pd.Series(np.nan, index=features.index).fillna(0).astype(int)
        return empty_series if not intermediate else (empty_series, pd.Series(dtype=int))

    if intermediate:
        # intermediate_stat 已经按 (col1, col2) 分组了
        return feature_merged, intermediate_stat
    return feature_merged

def nunique(features: pd.DataFrame, intermediate: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """计算第一个特征分组下第二个特征的唯一值数量"""
    if not isinstance(features, pd.DataFrame) or features.shape[1] < 2: raise ValueError("NUnique 输入必须是至少包含两列的 DataFrame")
    if features.empty: return pd.Series(dtype=int) if not intermediate else (pd.Series(dtype=int), pd.Series(dtype=int))

    group_col, value_col = features.columns[0], features.columns[1]
    # logger.trace(f"Calculating nunique of '{value_col}' grouped by '{group_col}'...")
    try:
        # 计算中间统计量
        intermediate_stat = features.groupby(group_col)[value_col].nunique(dropna=False).rename('nunique') # dropna=False
        # 使用 transform 广播结果
        # fillna(1) 对应代码的 Transform._apply 逻辑
        feature_merged = features.groupby(group_col)[value_col].transform(lambda x: x.nunique(dropna=False)).fillna(1).astype(float) # 保持 float
        feature_merged.rename('nunique', inplace=True) # 重命名 Series
    except Exception as e:
        logger.exception(f"计算 nunique 时出错: {e}")
        empty_series = pd.Series(np.nan, index=features.index).fillna(1).astype(float) # 填充 1
        return empty_series if not intermediate else (empty_series, pd.Series(dtype=float))

    if intermediate:
        return feature_merged, intermediate_stat
    return feature_merged

def combine(features: pd.DataFrame, intermediate: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]:
    """将两个特征组合成一个新的分类特征（整数编码）"""
    if not isinstance(features, pd.DataFrame) or features.shape[1] < 2: raise ValueError("Combine 输入必须是至少包含两列的 DataFrame")
    if features.empty: return pd.Series(dtype=int) if not intermediate else (pd.Series(dtype=int), pd.DataFrame())

    col1, col2 = features.columns[0], features.columns[1]
    # logger.trace(f"Combining features '{col1}' and '{col2}'...")
    try:
        # 拼接 -> category -> codes
        feature_combined = (features[col1].astype(str).fillna('__NaN__') + '_&_' +
                            features[col2].astype(str).fillna('__NaN__')).astype('category')
        feature_encoded = feature_combined.cat.codes.rename('combine')
        feature_encoded.index = features.index
    except Exception as e:
        logger.exception(f"执行 combine 操作时出错: {e}")
        empty_series = pd.Series(np.nan, index=features.index).fillna(-1).astype(int) # 填充 -1 表示错误
        return empty_series if not intermediate else (empty_series, pd.DataFrame())

    if intermediate:
        # 创建映射组合类别 -> 编码
        intermediate_stat_df = pd.DataFrame({
            'category': feature_combined.cat.categories,
            'code': range(len(feature_combined.cat.categories))
        }).set_index('category')
        # logger.trace(f"Combine intermediate map created with {len(intermediate_stat_df)} entries.")
        return feature_encoded, intermediate_stat_df # 返回 DataFrame 形式的映射
    return feature_encoded

# --- 数值按分类聚合操作符 ('n', 'c') ---

def _agg_op(op_name: str, features: pd.DataFrame, intermediate: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """(内部) 聚合操作辅助函数, transform 逻辑"""
    if not isinstance(features, pd.DataFrame) or features.shape[1] < 2:
        raise ValueError(f"聚合操作 '{op_name}' 需要至少包含两列（分组列,数值列）的 DataFrame")
    if features.empty: return pd.Series(dtype=float) if not intermediate else (pd.Series(dtype=float), pd.Series(dtype=float))

    group_col, value_col = features.columns[0], features.columns[1]
    agg_func_map = {'aggmean': 'mean', 'aggmax': 'max', 'aggmin': 'min', 'aggstd': 'std'}
    agg_func = agg_func_map.get(op_name)
    if agg_func is None: raise ValueError(f"未知的聚合操作: {op_name}")

    # logger.trace(f"Calculating {op_name} of '{value_col}' grouped by '{group_col}'...")
    try:
        # 确保值列是数值类型
        value_series = pd.to_numeric(features[value_col], errors='coerce')
        # 准备分组键,处理 NaN
        group_series = features[group_col].fillna('__NaN__') # 将 NaN 视为一个类别

        # 步骤 1: 计算中间统计量 (按组聚合)
        # intermediate_stat 的索引是 group_col 的唯一值
        intermediate_stat: pd.Series
        if agg_func == 'std':
             with warnings.catch_warnings():
                 warnings.simplefilter("ignore", category=RuntimeWarning)
                 intermediate_stat = value_series.groupby(group_series).std(ddof=1)
        else:
             intermediate_stat = value_series.groupby(group_series).agg(agg_func)
        intermediate_stat = intermediate_stat.rename(op_name)
        # logger.trace(f"{op_name} intermediate_stat calculated, size: {len(intermediate_stat)}")

        # 步骤 2: 将统计量映射回 DataFrame 索引
        # 使用 Series.map(dict) 或 merge
        # 使用 map 更高效
        # 创建映射字典
        stat_map = intermediate_stat.to_dict()
        feature_merged = group_series.map(stat_map)

        # 填充在 group_series 中但不在 stat_map 索引中的值（理论上不应发生）
        # 以及聚合结果为 NaN 的情况（例如 std 对单点组）
        # 代码使用 fillna(0)
        feature_merged = feature_merged.fillna(0).astype(float) # 确保是 float
        feature_merged.index = features.index # 确保索引正确

    except Exception as e:
        logger.exception(f"计算 {op_name} 时出错: {e}")
        empty_series = pd.Series(np.nan, index=features.index).fillna(0).astype(float)
        # 返回与预期类型匹配的空 Series
        return empty_series if not intermediate else (empty_series, pd.Series(dtype=float))

    if intermediate:
        # 返回映射后的 Series 和 中间统计 Series
        return feature_merged, intermediate_stat
    return feature_merged

# 具体聚合函数调用内部辅助函数
def aggmean(features: pd.DataFrame, intermediate: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """计算第一个特征分组下第二个数值特征的平均值"""
    return _agg_op('aggmean', features, intermediate)

def aggmin(features: pd.DataFrame, intermediate: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """计算第一个特征分组下第二个数值特征的最小值"""
    return _agg_op('aggmin', features, intermediate)

def aggmax(features: pd.DataFrame, intermediate: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """计算第一个特征分组下第二个数值特征的最大值"""
    return _agg_op('aggmax', features, intermediate)

def aggstd(features: pd.DataFrame, intermediate: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """计算第一个特征分组下第二个数值特征的标准差"""
    return _agg_op('aggstd', features, intermediate)


# --- 数值特征之间的基本运算 ('n', 'n') ---

def _arithmetic_op(op: str, features: pd.DataFrame) -> pd.Series:
    """(内部) 算术运算辅助函数,"""
    if not isinstance(features, pd.DataFrame) or features.shape[1] < 2:
        raise ValueError(f"算术操作 '{op}' 需要至少包含两列的 DataFrame")
    if features.empty: return pd.Series(dtype=float)

    col1, col2 = features.columns[0], features.columns[1]
    # logger.trace(f"Performing arithmetic op '{op}' on '{col1}' and '{col2}'")
    # 先转 float,再操作
    feat1 = features[col1].astype(float)
    feat2 = features[col2].astype(float)

    result: pd.Series
    try:
        if op == 'add': result = feat1 + feat2
        elif op == 'sub': result = feat1 - feat2
        elif op == 'mul': result = feat1 * feat2
        elif op == 'div':
            result = _safe_divide_local(feat1, feat2, default=np.nan) # 使用本地安全除法
        else: raise ValueError(f"未知的算术操作: {op}")

        # 代码对 mul 和 div 结果进行了裁剪
        if op in ['mul', 'div']:
            clip_lower, clip_upper = -1.0e9, 1.0e9
            result = result.clip(lower=clip_lower, upper=clip_upper)

        # 代码最后填充 NaN 为 0
        return result.fillna(0)
    except Exception as e:
         logger.exception(f"执行算术操作 '{op}' 时出错: {e}")
         return pd.Series(np.nan, index=features.index).fillna(0) # 返回填充0的Series

def add(features: pd.DataFrame) -> pd.Series: return _arithmetic_op('add', features)
def sub(features: pd.DataFrame) -> pd.Series: return _arithmetic_op('sub', features)
def mul(features: pd.DataFrame) -> pd.Series: return _arithmetic_op('mul', features)
def div(features: pd.DataFrame) -> pd.Series: return _arithmetic_op('div', features)

def std(features: pd.DataFrame) -> pd.Series: # 注意此 std 与 ('n','c') 的 aggstd 不同
    """计算两列数值特征逐行的标准差"""
    if not isinstance(features, pd.DataFrame) or features.shape[1] < 2:
        raise ValueError(f"行标准差 'std' 需要至少包含两列的 DataFrame")
    if features.empty: return pd.Series(dtype=float)
    # logger.trace(f"Calculating row-wise std for columns: {features.columns[:2].tolist()}")
    try:
        # 只取前两列,确保是数值
        feat1 = pd.to_numeric(features.iloc[:, 0], errors='coerce')
        feat2 = pd.to_numeric(features.iloc[:, 1], errors='coerce')
        temp_df = pd.concat([feat1, feat2], axis=1)
        # skipna=True, ddof=1
        return temp_df.std(axis=1, skipna=True, ddof=1).fillna(0) # 填充 NaN 为 0
    except Exception as e:
        logger.exception(f"计算行标准差 (std) 时出错: {e}")
        return pd.Series(np.nan, index=features.index).fillna(0)

def maximize(features: pd.DataFrame) -> pd.Series:
    """计算两列数值特征逐行的最大值"""
    if not isinstance(features, pd.DataFrame) or features.shape[1] < 2:
        raise ValueError(f"行最大值 'maximize' 需要至少包含两列的 DataFrame")
    if features.empty: return pd.Series(dtype=float)
    # logger.trace(f"Calculating row-wise max for columns: {features.columns[:2].tolist()}")
    try:
        feat1 = pd.to_numeric(features.iloc[:, 0], errors='coerce')
        feat2 = pd.to_numeric(features.iloc[:, 1], errors='coerce')
        # numpy.maximum 会传播 NaN,我们需要填充
        return np.maximum(feat1, feat2).fillna(0) # 假设填充 0
    except Exception as e:
        logger.exception(f"计算行最大值 (maximize) 时出错: {e}")
        return pd.Series(np.nan, index=features.index).fillna(0)


# --- 时间序列操作符 ('n', 't') ---

def _ts_op(op_type: str, features: Union[pd.Series, pd.DataFrame], time: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """(内部) 时间序列操作辅助函数,"""
    # 检查 time 参数
    if not isinstance(time, int) or time == 0:
         logger.error(f"时间序列操作 '{op_type}' 的时间跨度 'time' ({time}) 必须是非零整数")
         # 返回合适的空/NaN 结果
         if isinstance(features, pd.DataFrame): return pd.DataFrame(index=features.index)
         else: return pd.Series(np.nan, index=features.index)

    group_col: Optional[str] = None
    value_col: Optional[str] = None
    feature_series: pd.Series

    if isinstance(features, pd.Series):
        if features.empty: return pd.Series(dtype=float)
        feature_series = features
        target_col_name = features.name # 保留名称
    elif isinstance(features, pd.DataFrame):
        if features.empty: return pd.Series(dtype=float) # 假设非 cov/corr 时返回 Series
        if features.shape[1] == 1:
            feature_series = features.iloc[:, 0]
            target_col_name = features.columns[0]
        elif features.shape[1] >= 2 and op_type not in ['ts_cov', 'ts_corr', 'ewm_cov', 'ewm_corr']:
            # 假设对于非 cov/corr 的分组操作,第一列是分组键,第二列是值列
            group_col = features.columns[0]
            value_col = features.columns[1]
            feature_series = features # 在 groupby 内部处理
            target_col_name = value_col # 结果是值列的变换
            logger.trace(f"执行分组时间序列操作 '{op_type}', Group: '{group_col}', Value: '{value_col}'")
        elif features.shape[1] >= 2 and op_type in ['ts_cov', 'ts_corr', 'ewm_cov', 'ewm_corr']:
             # 对于协方差/相关性,需要两列数值
             # TODO: 代码如何处理这两列?是 DataFrame 的前两列吗?
             # 假设是前两列
             col1, col2 = features.columns[0], features.columns[1]
             logger.trace(f"执行滚动/EWM Cov/Corr 操作 '{op_type}' on '{col1}', '{col2}'")
             feature_series = features[[col1, col2]] # 传递包含两列的 DF
             target_col_name = None # 结果通常是单个 Series
        else:
             raise ValueError(f"时间序列操作 '{op_type}' 的输入 DataFrame 列数不足")
    else:
        raise TypeError("时间序列操作的输入必须是 pandas Series 或 DataFrame")

    try:
        result: Union[pd.Series, pd.DataFrame]
        # --- 分组时间序列操作 ---
        if group_col and value_col:
             grouped = features.groupby(group_col)[value_col]
             # 需要将 time (期数) 转换为滚动窗口大小
             window = abs(time) # 窗口大小用绝对值
             min_periods = max(1, window // 2) if op_type not in ['ts_std', 'ewm_std', 'ts_cov', 'ewm_cov'] else 2 # 确保 std/cov 至少2个点
             span = abs(time) # ewm 使用 span

             if op_type == 'diff': result = grouped.diff(periods=time)
             elif op_type == 'delay': result = grouped.shift(periods=time)
             elif op_type.startswith('ts_'):
                 rolling_obj = grouped.rolling(window=window, min_periods=min_periods)
                 op_func_name = op_type.split('_')[1]
                 if hasattr(rolling_obj, op_func_name):
                      if op_func_name == 'std': result = getattr(rolling_obj, op_func_name)(ddof=1, **kwargs)
                      else: result = getattr(rolling_obj, op_func_name)(**kwargs)
                 else: raise ValueError(f"未知的滚动操作: {op_func_name}")
             elif op_type.startswith('ewm_'):
                 ewm_obj = grouped.ewm(span=span, min_periods=min_periods, adjust=True) # adjust=True
                 op_func_name = op_type.split('_')[1]
                 if hasattr(ewm_obj, op_func_name):
                      if op_func_name == 'std': result = getattr(ewm_obj, op_func_name)(ddof=1, **kwargs)
                      else: result = getattr(ewm_obj, op_func_name)(**kwargs)
                 else: raise ValueError(f"未知的 EWM 操作: {op_func_name}")
             else: raise ValueError(f"未知的分组时间序列操作类型: {op_type}")
             # 恢复索引
             result = result.reindex(features.index)

        # --- 非分组时间序列操作 ---
        else:
             window = abs(time); span = abs(time)
             min_periods = max(1, window // 2) if op_type not in ['ts_std', 'ewm_std', 'ts_cov', 'ewm_cov', 'ts_corr', 'ewm_corr'] else 2

             if op_type == 'diff': result = feature_series.diff(periods=time)
             elif op_type == 'delay': result = feature_series.shift(periods=time)
             elif op_type.startswith('ts_'):
                 # 处理 cov/corr,它们需要两个 Series
                 if op_type in ['ts_cov', 'ts_corr']:
                      if not isinstance(feature_series, pd.DataFrame) or feature_series.shape[1] < 2:
                           logger.warning(f"操作 '{op_type}' 需要两列输入,但只收到一列或非 DataFrame返回 NaN")
                           result = pd.Series(np.nan, index=features.index)
                      else:
                           rolling_obj = feature_series.iloc[:, 0].rolling(window=window, min_periods=min_periods)
                           other_series = feature_series.iloc[:, 1]
                           op_func_name = op_type.split('_')[1]
                           if hasattr(rolling_obj, op_func_name):
                                result = getattr(rolling_obj, op_func_name)(other=other_series, **kwargs)
                           else: raise ValueError(f"未知的滚动操作: {op_func_name}")
                 else: # 其他 ts_ 操作
                      rolling_obj = feature_series.rolling(window=window, min_periods=min_periods)
                      op_func_name = op_type.split('_')[1]
                      if hasattr(rolling_obj, op_func_name):
                           if op_func_name == 'rank': result = getattr(rolling_obj, op_func_name)(pct=True, **kwargs) # Rank as pct
                           elif op_func_name == 'std': result = getattr(rolling_obj, op_func_name)(ddof=1, **kwargs)
                           else: result = getattr(rolling_obj, op_func_name)(**kwargs)
                      else: raise ValueError(f"未知的滚动操作: {op_func_name}")
             elif op_type.startswith('ewm_'):
                  # 处理 cov/corr
                 if op_type in ['ewm_cov', 'ewm_corr']:
                      if not isinstance(feature_series, pd.DataFrame) or feature_series.shape[1] < 2:
                           logger.warning(f"操作 '{op_type}' 需要两列输入返回 NaN")
                           result = pd.Series(np.nan, index=features.index)
                      else:
                           ewm_obj = feature_series.iloc[:, 0].ewm(span=span, min_periods=min_periods, adjust=True)
                           other_series = feature_series.iloc[:, 1]
                           op_func_name = op_type.split('_')[1]
                           if hasattr(ewm_obj, op_func_name):
                                result = getattr(ewm_obj, op_func_name)(other=other_series, **kwargs)
                           else: raise ValueError(f"未知的 EWM 操作: {op_func_name}")
                 else: # 其他 ewm_ 操作
                      ewm_obj = feature_series.ewm(span=span, min_periods=min_periods, adjust=True)
                      op_func_name = op_type.split('_')[1]
                      if hasattr(ewm_obj, op_func_name):
                           if op_func_name == 'std': result = getattr(ewm_obj, op_func_name)(ddof=1, **kwargs)
                           else: result = getattr(ewm_obj, op_func_name)(**kwargs)
                      else: raise ValueError(f"未知的 EWM 操作: {op_func_name}")
             else: raise ValueError(f"未知的时间序列操作类型: {op_type}")

        # 最终处理填充 NaN 为 0,恢复名称（如果输入是 Series）
        if isinstance(result, pd.Series):
             result = result.fillna(0)
             if target_col_name: result = result.rename(target_col_name)
        elif isinstance(result, pd.DataFrame):
             result = result.fillna(0)

        return result

    except Exception as e:
        logger.exception(f"执行时间序列操作 '{op_type}' 时出错: {e}")
        # 返回类型应与预期一致
        if op_type in ['ts_cov', 'ts_corr', 'ewm_cov', 'ewm_corr'] and isinstance(features, pd.DataFrame):
            return pd.DataFrame(0.0, index=features.index, columns=['error']) # 返回含0的 DataFrame
        else:
            return pd.Series(np.nan, index=features.index).fillna(0) # 返回含0的 Series

# --- 具体时间序列操作符的封装  ---
# 这些函数现在调用 _ts_op 内部辅助函数

def diff(features: Union[pd.Series, pd.DataFrame], time: int) -> pd.Series:
    """计算时间序列的差分"""
    return _ts_op('diff', features, time) # type: ignore

def delay(features: Union[pd.Series, pd.DataFrame], time: int) -> pd.Series:
    """计算时间序列的延迟（滞后）"""
    return _ts_op('delay', features, time) # type: ignore

def ts_mean(features: Union[pd.Series, pd.DataFrame], time: int) -> pd.Series:
    """计算时间序列的滚动平均值"""
    return _ts_op('ts_mean', features, time) # type: ignore

def ts_std(features: Union[pd.Series, pd.DataFrame], time: int) -> pd.Series:
    """计算时间序列的滚动标准差"""
    return _ts_op('ts_std', features, time) # type: ignore

def ts_cov(features: pd.DataFrame, time: int) -> pd.Series: # 返回 Series (两列计算结果)
    """计算时间序列的滚动协方差（需要 DataFrame 输入）"""
    # 注意返回类型改为 Series,因为 cov 通常计算两列之间的关系
    return _ts_op('ts_cov', features, time) # type: ignore

def ts_corr(features: pd.DataFrame, time: int) -> pd.Series: # 返回 Series
    """计算时间序列的滚动相关系数（需要 DataFrame 输入）"""
    return _ts_op('ts_corr', features, time) # type: ignore

def ts_max(features: Union[pd.Series, pd.DataFrame], time: int) -> pd.Series:
    """计算时间序列的滚动最大值"""
    return _ts_op('ts_max', features, time) # type: ignore

def ts_min(features: Union[pd.Series, pd.DataFrame], time: int) -> pd.Series:
    """计算时间序列的滚动最小值"""
    return _ts_op('ts_min', features, time) # type: ignore

def ts_rank(features: Union[pd.Series, pd.DataFrame], time: int) -> pd.Series:
    """计算时间序列的滚动排名（百分位数）"""
    return _ts_op('ts_rank', features, time, pct=True) # 传递 pct=True

def ewm_mean(features: Union[pd.Series, pd.DataFrame], time: int) -> pd.Series:
    """计算时间序列的指数加权移动平均值"""
    return _ts_op('ewm_mean', features, time) # type: ignore

def ewm_std(features: Union[pd.Series, pd.DataFrame], time: int) -> pd.Series:
    """计算时间序列的指数加权移动标准差"""
    return _ts_op('ewm_std', features, time) # type: ignore

def ewm_cov(features: pd.DataFrame, time: int) -> pd.Series: # 返回 Series
    """计算时间序列的指数加权移动协方差"""
    return _ts_op('ewm_cov', features, time) # type: ignore

def ewm_corr(features: pd.DataFrame, time: int) -> pd.Series: # 返回 Series
    """计算时间序列的指数加权移动相关系数"""
    return _ts_op('ewm_corr', features, time) # type: ignore
