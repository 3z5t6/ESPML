# -*- coding: utf-8 -*-
"""
性能报告生成模块 (espml)
负责加载回测结果,计算性能指标,并生成汇总报告
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import glob # 使用 glob
from loguru import logger
# 导入明确需要的指标函数
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 导入项目级 utils 和 const
from espml.util import utils as common_utils
from espml.util import const

# --- 内部指标计算函数  ---
def calculate_regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Optional[float]]:
    """
    计算一组常用的回归性能指标

    Args:
        y_true (pd.Series): 真实值
        y_pred (pd.Series): 预测值

    Returns:
        Dict[str, Optional[float]]: 包含指标名称和值的字典 (RMSE, MAE, MAPE, R2)
                                     如果无法计算则值为 None 或 NaN
    """
    metrics: Dict[str, Optional[float]] = {'rmse': None, 'mae': None, 'mape': None, 'r2': None}
    # 对齐索引并移除 NaN/Inf 对
    df = pd.DataFrame({'true': y_true, 'pred': y_pred})
    # 确保是数值类型
    df['true'] = pd.to_numeric(df['true'], errors='coerce')
    df['pred'] = pd.to_numeric(df['pred'], errors='coerce')
    # 移除包含 NaN 或 Inf 的行
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if df.empty:
        logger.warning("计算指标时,有效数据为空所有指标将为 NaN")
        metrics = {k: np.nan for k in metrics} # 返回 NaN
        return metrics

    true = df['true']
    pred = df['pred']

    # 计算指标并处理潜在错误
    try: metrics['rmse'] = np.sqrt(mean_squared_error(true, pred))
    except Exception as e: logger.error(f"计算 RMSE 失败: {e}"); metrics['rmse'] = np.nan

    try: metrics['mae'] = mean_absolute_error(true, pred)
    except Exception as e: logger.error(f"计算 MAE 失败: {e}"); metrics['mae'] = np.nan

    try:
        # MAPE: 仅对 true != 0 的点计算,避免除零
        mask = true != 0
        if np.any(mask):
             mape = np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100
             metrics['mape'] = mape
        else: metrics['mape'] = np.nan # 无法计算
    except Exception as e: logger.error(f"计算 MAPE 失败: {e}"); metrics['mape'] = np.nan

    try: metrics['r2'] = r2_score(true, pred)
    except Exception as e: logger.error(f"计算 R2 Score 失败: {e}"); metrics['r2'] = np.nan

    return metrics

# --- 报告生成主函数  ---
@common_utils.log_execution_time(level="INFO") # 对整个报告生成计时
def generate_backtest_report(
    pred_dir: Union[str, Path] = const.PRED_DIR,
    raw_data_dir: Union[str, Path] = const.RESOURCE_DIR,
    task_id_pattern: str = "backtrack_*", # 匹配所有回测文件
    actual_power_col: str = const.ORIG_FANS_POWER_COL, # 从 const 获取
    time_col: str = const.ORIG_FANS_TIME_COL, # 从 const 获取
    output_report_file: Optional[str] = "backtest_summary_report.csv",
    generate_plots: bool = False # 是否生成图表
    ) -> Optional[pd.DataFrame]:
    """
    生成回测性能报告

    Args:
        pred_dir: 存放预测结果文件的目录
        raw_data_dir: 存放数据文件(fans.csv)的目录
        task_id_pattern: 用于 glob 查找回测结果文件的模式
        actual_power_col: fans.csv 中实际功率的列名
        time_col: fans.csv 中时间列的列名
        output_report_file: 输出汇总报告 CSV 的文件名None 则不保存
        generate_plots: 是否生成性能图表 (当前实现不包含绘图)

    Returns:
        Optional[pd.DataFrame]: 包含每个回测文件及其指标的 DataFrame,如果失败则返回 None
    """
    logger.info(f"开始生成回测性能报告 (模式: '{task_id_pattern}')...")
    pred_path = Path(pred_dir)
    raw_path = Path(raw_data_dir)
    fans_file = raw_path / const.FANS_CSV

    # 1. 检查并加载实际功率文件 
    if not common_utils.check_path_exists(fans_file, path_type='f'):
        logger.critical(f"无法生成报告找不到实际功率文件 {fans_file}") # 使用 critical
        return None
    logger.debug(f"加载实际功率数据从: {fans_file}")
    try:
        df_actual = pd.read_csv(
            fans_file,
            usecols=[time_col, actual_power_col],
            parse_dates=[time_col], # 解析日期
            index_col=time_col,     # 将时间设为索引
            infer_datetime_format=True # 尝试加速解析
        )
        # 重命名列并确保索引是 DatetimeIndex
        df_actual = df_actual.rename(columns={actual_power_col: 'actual'})
        if not isinstance(df_actual.index, pd.DatetimeIndex):
             raise ValueError("实际功率文件的时间列未能成功解析为 DatetimeIndex")
        df_actual = df_actual.sort_index() # 确保排序
        if df_actual.empty: raise ValueError("实际功率文件为空或无法解析")
        logger.info(f"实际功率数据加载完成,范围: [{df_actual.index.min()}, {df_actual.index.max()}]")
    except Exception as e:
        logger.exception(f"加载实际功率文件 '{fans_file}' 失败: {e}")
        return None

    # 2. 查找回测结果文件 (使用 glob)
    backtrack_files_pattern_full = str(pred_path / f"{task_id_pattern}.csv")
    # logger.debug(f"查找回测文件模式: {backtrack_files_pattern_full}")
    backtrack_files = glob.glob(backtrack_files_pattern_full)

    if not backtrack_files:
        logger.warning(f"在目录 '{pred_path}' 中未找到匹配模式 '{task_id_pattern}.csv' 的回测结果文件")
        return pd.DataFrame() # 返回空 DataFrame

    logger.info(f"找到 {len(backtrack_files)} 个回测结果文件进行处理")

    # 3. 循环处理文件并计算指标
    all_metrics: List[Dict[str, Any]] = []
    processed_files_count = 0
    for file_str in sorted(backtrack_files): # 对文件排序以保证报告顺序一致
        file_path = Path(file_str)
        logger.debug(f"处理回测文件: {file_path.name}...")
        try:
            # 加载预测结果 (假设时间是索引)
            df_pred = pd.read_csv(file_path, index_col=0, parse_dates=True)
            if not isinstance(df_pred.index, pd.DatetimeIndex):
                 df_pred.index = pd.to_datetime(df_pred.index, errors='coerce') # 尝试转换
                 df_pred = df_pred.dropna(axis=0, subset=[df_pred.index.name]) # 删除无效索引行
                 if not isinstance(df_pred.index, pd.DatetimeIndex):
                      raise ValueError("文件索引无法转换为 DatetimeIndex")
            df_pred = df_pred.sort_index() # 确保排序

            # 检查预测列
            if const.PREDICTION_COLUMN_NAME not in df_pred.columns:
                 logger.error(f"文件 '{file_path.name}' 缺少列 '{const.PREDICTION_COLUMN_NAME}',跳过")
                 continue

            # 提取时间范围
            pred_start, pred_end = df_pred.index.min(), df_pred.index.max()
            if pd.isna(pred_start) or pd.isna(pred_end): continue # 跳过无效索引

            # 合并实际值 (使用左连接)
            df_merged = df_pred[[const.PREDICTION_COLUMN_NAME]].merge(
                df_actual[['actual']], left_index=True, right_index=True, how='left'
            )
            df_merged = df_merged.rename(columns={const.PREDICTION_COLUMN_NAME: 'prediction'})

            # 检查合并后是否有有效数据对
            df_merged_valid = df_merged.dropna() # 移除任何包含 NaN 的行
            if df_merged_valid.empty:
                logger.warning(f"文件 '{file_path.name}' 在时间范围 [{pred_start}, {pred_end}] 内找不到对应的有效实际值/预测值对")
                continue

            # 计算指标
            metrics = calculate_regression_metrics(df_merged_valid['actual'], df_merged_valid['prediction'])

            # 记录结果 
            file_meta = {
                'filename': file_path.name,
                'start_time': pred_start.isoformat(),
                'end_time': pred_end.isoformat(),
                'num_points_pred': len(df_pred), # 预测点数
                'num_points_actual': len(df_merged['actual'].dropna()), # 有效实际值点数
                'num_points_eval': len(df_merged_valid), # 用于评估的点数
            }
            file_meta.update(metrics) # 合并指标
            all_metrics.append(file_meta)
            processed_files_count += 1

        except Exception as e:
            logger.error(f"处理回测文件 '{file_path.name}' 时失败: {e}", exc_info=True)

    # 4. 生成汇总报告
    if not all_metrics:
        logger.error("未能成功处理任何回测文件以生成报告")
        return None

    report_df = pd.DataFrame(all_metrics)
    # 计算总体平均指标 (忽略 NaN)
    avg_metrics = report_df[['rmse', 'mae', 'mape', 'r2']].mean(skipna=True) #.rename('Average')
    logger.info("\n------ 回测性能汇总 (每个文件) ------")
    try: logger.info(f"\n{report_df.to_string(index=False)}")
    except Exception: logger.info(f"{report_df}") # fallback to simple print
    logger.info("\n------ 平均指标 (所有文件) ------")
    try: logger.info(f"\n{avg_metrics.to_string()}")
    except Exception: logger.info(f"{avg_metrics}")
    logger.info("-------------------------------")

    # 5. 保存汇总报告 (如果需要)
    if output_report_file:
        report_save_path = pred_path / output_report_file
        logger.info(f"将汇总报告保存到: {report_save_path}")
        try:
             # 可以选择是否添加平均行
             report_df_to_save = report_df.copy()
             # 将平均值作为最后一行添加 (可选)
             # avg_metrics_df = avg_metrics.to_frame('Average').T
             # avg_metrics_df['filename'] = 'AVERAGE'
             # report_df_to_save = pd.concat([report_df, avg_metrics_df], ignore_index=True)

             report_df_to_save.to_csv(report_save_path, index=False, float_format='%.6f', encoding='utf-8')
             logger.info("汇总报告保存成功")
        except Exception as e:
             logger.exception(f"保存汇总报告到 '{report_save_path}' 时失败: {e}")

    # 6. 生成图表 (代码不包含则不实现)
    if generate_plots:
        logger.warning("图表生成功能 (generate_plots=True) ")

    logger.info("回测性能报告生成完毕")
    return report_df


logger.info("性能报告模块 (espml.util.report) 加载完成")