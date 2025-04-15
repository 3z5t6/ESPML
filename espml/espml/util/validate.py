# -*- coding: utf-8 -*-
"""
数据验证工具函数模块 (espml)
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from loguru import logger

# 假设的 format_valid 函数实现
def format_valid(
    df: pd.DataFrame,
    column_formats: Optional[Dict[str, List[str]]] = None,
    datetime_format: str = '%Y-%m-%d %H:%M:%S',
    raise_error: bool = False
    ) -> bool:
    """
    验证 DataFrame 中的列是否符合指定的格式或类型要求
    (这是一个基于函数名和常见需求的推断实现，需要根据代码调整)

    Args:
        df (pd.DataFrame): 需要验证的 DataFrame
        column_formats (Optional[Dict[str, List[str]]]): 一个字典，键是列名，
            值是允许的数据类型字符串列表 (例如 ['int64', 'float64'], ['datetime64[ns]'], ['object', 'category'])
            如果为 None，则不进行严格的类型检查
        datetime_format (str): 用于尝试转换对象类型列为日期时间的格式
        raise_error (bool): 如果验证失败，是否引发 ValueError

    Returns:
        bool: 如果所有检查都通过，返回 True，否则返回 False (除非 raise_error=True)

    Raises:
        ValueError: 如果 raise_error=True 且验证失败
    """
    logger.debug(f"开始验证 DataFrame 格式 (共 {len(df.columns)} 列)...")
    is_valid = True
    error_messages = []

    for col in df.columns:
        col_dtype_str = str(df[col].dtype)
        logger.trace(f"检查列 '{col}', 类型: {col_dtype_str}")

        # 1. 检查 Inf/过大值 (仅对数值列)
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            if not np.all(np.isfinite(df[col].dropna().values)):
                 inf_count = np.isinf(df[col]).sum()
                 if inf_count > 0:
                     msg = f"列 '{col}' 包含 {inf_count} 个 Inf 值"
                     error_messages.append(msg)
                     logger.warning(msg)
                     is_valid = False

        # 2. 检查指定的列格式/类型
        if column_formats and col in column_formats:
            allowed_types = column_formats[col]
            if col_dtype_str not in allowed_types:
                # 尝试特殊处理如果允许 'datetime64[ns]' 但实际是 'object'
                # 则尝试按 datetime_format 转换，看是否成功
                is_compatible = False
                if 'datetime64[ns]' in allowed_types and col_dtype_str == 'object':
                     try:
                         _ = pd.to_datetime(df[col], format=datetime_format, errors='raise')
                         logger.trace(f"列 '{col}' (object) 可以成功解析为 datetime (format={datetime_format})")
                         is_compatible = True # 视为兼容
                     except (ValueError, TypeError):
                         pass # 解析失败，类型不匹配

                if not is_compatible:
                    msg = f"列 '{col}' 的类型 '{col_dtype_str}' 不在允许的类型 {allowed_types} 中"
                    error_messages.append(msg)
                    logger.warning(msg)
                    is_valid = False

    if not is_valid and raise_error:
        full_error_message = "DataFrame 格式验证失败:\n" + "\n".join(error_messages)
        raise ValueError(full_error_message)

    if is_valid:
        logger.debug("DataFrame 格式验证通过")
    else:
        logger.warning("DataFrame 格式验证未通过")

    return is_valid

logger.info("数据验证工具模块 (espml.util.validate) 加载完成")
