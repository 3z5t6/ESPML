# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
ESPML 项目回测执行脚本
负责根据配置执行指定任务在历史时间段内的回测
"""

import argparse
import sys
import os
import pandas as pd
import datetime
from typing import Dict, Any, List, Optional

from espml.scripts.main import load_and_merge_configs

# 添加项目根目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from loguru import logger


try:
    from espml.util.log_config import setup_logger, DEFAULT_LOG_DIR
    from espml.util.wind_incrml import WindTaskRunner, DEFAULT_TIMEZONE # 导入时区
    from espml.util import const
except ImportError as e:
    print(f"错误无法导入必需的 espml 模块: {e}", file=sys.stderr)
    sys.exit(1)

def parse_arguments() -> argparse.Namespace:
    """解析回测脚本的命令行参数"""
    parser = argparse.ArgumentParser(description="ESPML 风电功率预测回测程序")
    parser.add_argument("--config", type=str, default=str(const.PROJECT_CONFIG_DIR / "config.yaml"), help="主项目配置 YAML 文件路径")
    parser.add_argument("--task_config", type=str, default=str(const.PROJECT_CONFIG_DIR / "task_config.yaml"), help="任务配置 YAML 文件路径")
    parser.add_argument("--task_id", type=str, default="ALL", help="要执行回测的任务 ID (在任务配置中 type 为 backtrack)'ALL' 表示执行所有启用的回测任务")
    parser.add_argument("--start_date", type=str, required=True, help="回测起始日期 (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, required=True, help="回测结束日期 (YYYY-MM-DD)")
    parser.add_argument("--log_dir", type=str, default=None, help=f"日志文件输出目录 (默认: {DEFAULT_LOG_DIR})")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="日志记录级别 (默认: INFO)")
    args = parser.parse_args()
    return args

def main():
    """主回测执行函数"""
    args = parse_arguments()

    # 初始化日志 (使用回测特定的文件名)
    log_task_name = f"backtracking_{args.task_id}_{args.start_date}_{args.end_date}"
    log_directory = args.log_dir if args.log_dir else DEFAULT_LOG_DIR
    try:
        setup_logger(log_dir=log_directory, task_name=log_task_name, log_level=args.log_level.upper())
    except Exception as log_e:
        print(f"错误初始化日志系统失败: {log_e}", file=sys.stderr)
        logger.remove(); logger.add(sys.stderr, level=args.log_level.upper())
        logger.error("初始化文件日志失败,仅使用控制台输出")

    # 加载配置
    logger.info("开始加载配置...")
    full_config = load_and_merge_configs(args) # 复用 main.py 的函数
    if full_config is None: logger.critical("无法加载配置,程序终止"); sys.exit(1)
    logger.info("配置加载完成")

    # 解析回测日期范围
    try:
        backtrack_start_dt = pd.to_datetime(args.start_date).floor('D')
        backtrack_end_dt = pd.to_datetime(args.end_date).floor('D') # date_range 通常包含结束日期
        if backtrack_start_dt > backtrack_end_dt:
             raise ValueError("回测起始日期不能晚于结束日期")
        logger.info(f"回测日期范围: {backtrack_start_dt.date()} to {backtrack_end_dt.date()}")
    except Exception as date_e:
        logger.critical(f"解析回测日期失败: {date_e}"); sys.exit(1)

    # 筛选要执行的回测任务
    task_list: List[Dict[str, Any]] = full_config.get('tasks', [])
    all_backtrack_tasks: List[Dict[str, Any]] = []
    for task_conf in task_list:
        if isinstance(task_conf, dict) and task_conf.get('type') == 'backtrack':
             all_backtrack_tasks.append(task_conf)

    if not all_backtrack_tasks: logger.warning("未在任务配置中找到 type 为 'backtrack' 的任务"); return

    tasks_to_run_configs: List[Dict[str, Any]] = []
    if args.task_id.upper() == "ALL":
        tasks_to_run_configs = [t for t in all_backtrack_tasks if t.get('enabled', False)]
        logger.info(f"准备执行所有 {len(tasks_to_run_configs)} 个启用的回测任务")
    else:
        found_task = next((t for t in all_backtrack_tasks if t.get('task_id') == args.task_id), None)
        if found_task:
            if found_task.get('enabled', False):
                 tasks_to_run_configs = [found_task]
                 logger.info(f"准备执行指定的回测任务: {args.task_id}")
            else:
                 logger.error(f"指定的回测任务 '{args.task_id}' 已找到但未启用 (enabled=false)")
                 sys.exit(1)
        else:
            logger.error(f"指定的回测任务 ID '{args.task_id}' 未在配置中找到或其 type 不是 'backtrack'")
            sys.exit(1)

    # --- 执行回测循环 ---
    logger.info("====== 开始执行回测循环 ======")
    total_runs = 0
    successful_runs = 0

    # 按日期循环
    # 使用 pd.date_range 生成回测日期序列
    backtrack_date_range = pd.date_range(start=backtrack_start_dt, end=backtrack_end_dt, freq='D')

    for backtrack_task_config in tasks_to_run_configs:
        task_id_bt = backtrack_task_config['task_id']
        logger.info(f"--- 开始处理回测任务: {task_id_bt} ---")
        # 获取此回测任务对应的预测任务配置 (需要关联逻辑)
        # 假设预测任务 ID 可以从回测任务 ID 推断或在配置中显式指定
        forecast_task_id = backtrack_task_config.get('corresponding_forecast_task_id')
        if not forecast_task_id: # 尝试推断
             if task_id_bt.startswith('Backtrack'):
                  forecast_task_id = task_id_bt.replace('Backtrack', 'Forecast', 1)
             else:
                  logger.error(f"无法确定回测任务 '{task_id_bt}' 对应的预测任务 ID请在配置中添加 'corresponding_forecast_task_id'")
                  continue # 跳过此回测任务
        logger.info(f"回测任务 '{task_id_bt}' 将模拟运行预测任务 '{forecast_task_id}'")

        # 获取回测的每日模拟时间点
        try:
            backtrack_time = datetime.datetime.strptime(backtrack_task_config.get('backtrack_time_of_day', "00:00:00"), '%H:%M:%S').time()
        except ValueError:
             logger.error(f"任务 '{task_id_bt}' 的 backtrack_time_of_day 格式无效")
             continue

        # 实例化对应的 WindTaskRunner (用于预测任务)
        try:
            # 传递预测任务的 ID 和完整的配置
            runner = WindTaskRunner(task_id=forecast_task_id, config=full_config)
        except Exception as runner_init_e:
             logger.error(f"为任务 '{forecast_task_id}' 初始化 WindTaskRunner 失败: {runner_init_e}")
             continue # 跳过此回测任务

        # 按日期循环执行 run
        for current_backtrack_date in backtrack_date_range:
             total_runs += 1
             # 构造覆盖时间点
             simulated_time = pd.Timestamp.combine(current_backtrack_date.date(), backtrack_time)
             if DEFAULT_TIMEZONE: simulated_time = DEFAULT_TIMEZONE.localize(simulated_time)
             logger.info(f"--- 回测时间点: {simulated_time} (任务: {task_id_bt}) ---")

             try:
                 # 调用 runner.run 并传入覆盖时间
                 # WindTaskRunner 的 run 方法内部需要正确处理 current_time_override
                 # 并且能够根据 self.task_run_type == 'backtrack' 使用回测配置
                 # 为了传递回测特定配置,可能需要修改 run 方法或在这里修改配置副本
                 # 假设 WindTaskRunner.run 可以处理覆盖时间,但需要传递回测模式信息
                 # 或者直接修改 WindTaskRunner.__init__ 使其接收 task_config

                 # 简洁方案在 runner 内部判断是否为回测模式?不,入口处判断更清晰
                 # 此处直接调用 run,假设它能正确处理
                 # !!! 注意WindTaskRunner 的 run 方法目前没有区分 forecast 和 backtrack !!!
                 # !!! 需要修改 WindTaskRunner.run 或在此处传入特殊参数 !!!
                 # 临时解决方案修改传递给 runner 的配置副本,使其包含回测参数
                 # runner_config_override = runner.effective_config.copy() # 获取 runner 当前配置
                 # runner_config_override['RuntimeInfo'] = {'is_backtracking': True, 'backtrack_config': backtrack_task_config}
                 # runner.config = runner_config_override # 临时修改配置? 不安全

                 # 更可靠方案修改 WindTaskRunner.run 签名或添加 run_backtrack 方法
                 # 假设我们添加了 run_backtrack 方法
                 # runner.run_backtrack(backtrack_time=simulated_time, backtrack_config=backtrack_task_config)

                 # 当前实现依赖 WindTaskRunner.run 能处理 override 时间
                 # WindTaskRunner 需要根据 simulated_time 来计算回测窗口
                 # 并根据 task_config 中的 is_backtrack=True 来保存结果
                 # 因此 WindTaskRunner.__init__ 需要接收 task_config
                 # (之前初始化已传入 full_config,WindTaskRunner 可自行查找)
                 runner.run(current_time_override=simulated_time)
                 successful_runs += 1
             except KeyboardInterrupt:
                 logger.warning("用户中断回测")
                 return # 提前退出
             except Exception as run_e:
                  logger.exception(f"在回测时间点 {simulated_time} 运行任务 '{forecast_task_id}' 失败: {run_e}")
                  # 继续下一个时间点

        logger.info(f"--- 回测任务 {task_id_bt} 处理完毕 ---")

    logger.info("====== 回测循环结束 ======")
    logger.info(f"回测执行总结: 总运行次数={total_runs}, 成功次数={successful_runs} (注成功仅代表 runner.run 未抛出严重异常)")

    if total_runs == successful_runs and total_runs > 0:
        sys.exit(0)
    else:
        sys.exit(1) # 如果有失败或未运行则返回错误码


if __name__ == "__main__":
    logger.add(sys.stderr, level="INFO") # 基本日志
    main()
