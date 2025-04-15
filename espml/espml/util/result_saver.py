# -*- coding: utf-8 -*-
"""
预测结果保存模块 (espml)
负责将预测和回测结果保存为 CSV 文件，并记录运行元数据
"""

import os
import datetime
from typing import Dict, Any, Optional, Union, List
import pandas as pd
import numpy as np # 需要 numpy 处理 Inf
from pathlib import Path
from loguru import logger

# 导入项目级 utils 和 const
from espml.util import utils as common_utils
from espml.util import const

# --- 默认配置  ---
DEFAULT_FLOAT_FORMAT = '%.6f' # 默认浮点数保存精度
PREDICTION_COLUMN_NAME = 'prediction' # 假设预测列名固定
TIMESTAMP_INDEX_NAME = const.INTERNAL_TIME_INDEX # 与 DataProcessor 保持一致
META_FILENAME = "meta.csv" # 元数据文件名

def _prepare_save_path(
    output_dir: Union[str, Path],
    task_id: str,
    pred_ref_time: Optional[datetime.datetime] = None,
    is_backtrack: bool = False
    ) -> Path:
    """
    (内部) 准备预测结果的保存路径和文件名命名逻辑

    Args:
        output_dir (Union[str, Path]): 结果保存的基础目录 (例如 data/pred)
        task_id (str): 任务 ID (例如 'Forecast4Hour')
        pred_ref_time (Optional[datetime.datetime]): 预测相关的参考时间（用于回测文件名）
        is_backtrack (bool): 是否为回测结果

    Returns:
        Path: 完整的保存文件路径

    Raises:
        ValueError: 如果回测模式缺少 pred_ref_time，或 task_id 无效
        OSError: 如果无法创建输出目录
    """
    if not task_id or not isinstance(task_id, str):
        raise ValueError("无效的任务 ID (task_id)")

    output_path = Path(output_dir)
    # 使用 common_utils 创建目录，并在失败时抛出异常
    if not common_utils.mkdir_if_not_exist(output_path):
         error_msg = f"无法创建结果保存目录: {output_path}"
         logger.critical(error_msg) # 使用 critical 级别
         raise OSError(error_msg)

    filename: str
    if is_backtrack:
        if pred_ref_time is None or not isinstance(pred_ref_time, datetime.datetime):
            raise ValueError("回测模式需要提供有效的 datetime 对象 'pred_ref_time'")
        # 文件名格式: backtrack_YYYYMMDD_HHMM_TaskID.csv
        time_str = pred_ref_time.strftime("%Y%m%d_%H%M") # 使用 YYYYMMDD_HHMM 格式
        filename = f"backtrack_{time_str}_{task_id}.csv"
    else:
        # 文件名格式: TaskID.csv (覆盖模式)
        filename = f"{task_id}.csv"

    return output_path / filename

def _validate_prediction_df(df: pd.DataFrame) -> bool:
    """(内部) 验证预测结果 DataFrame 的基本格式"""
    if not isinstance(df, pd.DataFrame):
        logger.error("结果保存验证失败输入不是有效的 Pandas DataFrame")
        return False
    if df.empty:
        logger.warning("正在尝试保存空的预测结果 DataFrame将创建空文件")
        # 允许保存空文件
    # 严格检查索引类型
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("结果保存验证失败DataFrame 索引必须是 DatetimeIndex")
        return False
    # 严格检查预测列是否存在
    if PREDICTION_COLUMN_NAME not in df.columns:
        logger.error(f"结果保存验证失败DataFrame 缺少必需的列 '{PREDICTION_COLUMN_NAME}'")
        return False
    # 检查 Inf 值
    if pd.api.types.is_numeric_dtype(df[PREDICTION_COLUMN_NAME].dtype):
         # 检查非 NaN 值中是否有 Inf
         if np.isinf(df[PREDICTION_COLUMN_NAME].dropna()).any():
              inf_count = np.isinf(df[PREDICTION_COLUMN_NAME]).sum()
              logger.warning(f"预测列 '{PREDICTION_COLUMN_NAME}' 包含 {inf_count} 个 Inf 值")
              # 代码可能在此处处理 Inf，时仅警告
    return True

# 使用计时器装饰（如果代码有）
# @common_utils.log_execution_time(level="DEBUG")
def save_prediction_result(
    predictions_df: pd.DataFrame,
    task_id: str,
    output_dir: Union[str, Path] = const.PRED_DIR, # 使用常量
    pred_ref_time: Optional[datetime.datetime] = None, # 用于区分普通和回测
    is_backtrack: bool = False,
    float_format: str = DEFAULT_FLOAT_FORMAT,
    **kwargs # 其他传递给 to_csv 的参数
    ) -> bool:
    """
    保存预测或回测结果到 CSV 文件
    代码逻辑，包括文件名生成和 CSV 参数

    Args:
        predictions_df (pd.DataFrame): 预测结果 DataFrame
        task_id (str): 任务 ID
        output_dir (Union[str, Path]): 输出目录
        pred_ref_time (Optional[datetime.datetime]): 预测参考时间（仅回测需要）
        is_backtrack (bool): 是否为回测结果
        float_format (str): 浮点数格式
        **kwargs: 其他 to_csv 参数

    Returns:
        bool: 保存是否成功
    """
    mode_str = "回测" if is_backtrack else "常规预测"
    logger.info(f"准备保存任务 '{task_id}' 的{mode_str}结果...")

    # 验证输入 DataFrame
    if not _validate_prediction_df(predictions_df):
        logger.error("输入 DataFrame 验证失败，取消保存")
        return False

    # 准备保存路径 (捕获可能的错误)
    try:
        save_path = _prepare_save_path(output_dir, task_id, pred_ref_time, is_backtrack)
        logger.info(f"将{mode_str}结果保存到: {save_path}")
    except (ValueError, OSError, Exception) as path_e:
         logger.error(f"准备保存路径失败: {path_e}", exc_info=True)
         return False

    # 准备 to_csv 参数
    csv_params = {
        'path_or_buf': str(save_path),
        'index': True,               # 确认保存时间索引
        'index_label': TIMESTAMP_INDEX_NAME, # 显式指定索引列名
        'header': True,              # 确认保存列名
        'encoding': 'utf-8',         # 确认编码
        'sep': ',',                  # 确认分隔符
        'float_format': float_format, # 确认浮点数格式
        # 'date_format': '%Y-%m-%d %H:%M:%S.%f' # 可以指定更精确的日期格式
    }
    # 合并用户传入的额外参数
    csv_params.update(kwargs)
    # logger.trace(f"to_csv 参数: {csv_params}")

    # 执行保存
    try:
        # 保存整个传入的 DataFrame
        predictions_df.to_csv(**csv_params)
        logger.success(f"任务 '{task_id}' 的{mode_str}结果成功保存到: {save_path}") # 使用 success 级别
        return True
    except Exception as e:
        logger.exception(f"保存{mode_str}结果到 '{save_path}' 时失败: {e}")
        # 尝试删除可能产生的损坏文件
        common_utils.safe_remove(save_path) # 调用通用工具
        return False

# 使用计时器装饰
# @common_utils.log_execution_time(level="DEBUG")
def append_run_metadata(
    metadata_dict: Dict[str, Any],
    output_dir: Union[str, Path] = const.PRED_DIR, # 使用常量
    filename: str = META_FILENAME # 使用常量
    ) -> bool:
    """
    将单次运行的元数据追加到 meta.csv 文件中
    追加逻辑和错误处理

    Args:
        metadata_dict (Dict[str, Any]): 包含单次运行元数据的字典
        output_dir (Union[str, Path]): 输出目录
        filename (str): 元数据 CSV 文件名

    Returns:
        bool: 追加是否成功
    """
    if not isinstance(metadata_dict, dict) or not metadata_dict:
        logger.error("追加元数据失败输入的 metadata_dict 无效或为空")
        return False

    meta_path = Path(output_dir) / filename
    logger.debug(f"准备追加运行元数据到: {meta_path}")

    # 使用 common_utils 确保目录存在
    if not common_utils.mkdir_if_not_exist(output_dir):
        return False # 目录创建失败

    # 检查文件是否存在以确定 header
    # 使用 os.path.exists (如果代码使用 os) 或 Path.exists()
    file_exists = meta_path.exists() and meta_path.is_file()

    try:
        # 将字典转换为单行 DataFrame
        meta_df = pd.DataFrame([metadata_dict])
        # 确保列顺序一致性（可选，但推荐）
        # if file_exists:
        #     try:
        #          existing_header = pd.read_csv(meta_path, nrows=0).columns.tolist()
        #          meta_df = meta_df.reindex(columns=existing_header) # 按现有文件头排序
        #     except Exception as read_e:
        #          logger.warning(f"读取现有 meta.csv 头失败: {read_e}，将按字典顺序写入")

        # logger.trace(f"追加的元数据: {metadata_dict}")
        # logger.trace(f"文件是否存在: {file_exists}, 是否写入 header: {not file_exists}")

        # 以追加模式写入 CSV
        meta_df.to_csv(
            str(meta_path),
            mode='a',
            header=not file_exists,
            index=False, # 确认不写索引
            encoding='utf-8',
            sep=','
        )
        logger.info(f"成功追加运行元数据到: {meta_path}")
        return True
    except Exception as e:
        logger.exception(f"追加元数据到 '{meta_path}' 时失败: {e}")
        return False

logger.info("结果保存模块 (espml.util.result_saver) 加载完成")