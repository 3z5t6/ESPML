# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
ESPML 项目主入口脚本
负责解析命令行参数,加载配置,初始化日志,并根据任务配置驱动 WindTaskRunner 执行
"""

import argparse
import sys
import os
import time
from typing import Dict, Any, List, Optional

# 确保 espml 包在 Python 路径中
# (根据实际项目结构和运行方式调整)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # scripts 目录的上一级是项目根目录
sys.path.insert(0, project_root)

from loguru import logger # 使用 loguru

# 导入 espml 模块
try:
    from espml.config.yaml_parser import load_yaml_config, ConfigError
    from espml.util.log_config import setup_logger, LOG_DIR as DEFAULT_LOG_DIR # 导入默认日志目录
    from espml.util.wind_incrml import WindTaskRunner
    from espml.util import const
    from espml.util import utils as common_utils # 可能需要通用工具
except ImportError as e:
    print(f"错误无法导入必需的 espml 模块: {e}", file=sys.stderr)
    print("请确保 espml 包已正确安装或项目结构正确", file=sys.stderr)
    sys.exit(1)

def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ESPML 风电功率预测主程序")
    parser.add_argument(
        "--config",
        type=str,
        default=str(const.PROJECT_CONFIG_DIR / "config.yaml"), # 默认主配置文件路径
        help="主项目配置 YAML 文件路径"
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default=str(const.PROJECT_CONFIG_DIR / "task_config.yaml"), # 默认任务配置文件路径
        help="任务配置 YAML 文件路径"
    )
    parser.add_argument(
        "--task_id",
        type=str,
        default="ALL",
        help="要执行的特定任务 ID (来自任务配置文件)默认为 'ALL',执行所有启用的任务"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None, # 默认使用 log_config 中的设置
        help=f"日志文件输出目录 (默认: {DEFAULT_LOG_DIR})"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="日志记录级别 (默认: INFO)"
    )
    # 可以添加其他参数,例如 --run_mode (train/predict), --date (用于特定日期运行) 等
    # ...

    args = parser.parse_args()
    return args

def load_and_merge_configs(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """加载并合并主配置和任务配置"""
    logger.info(f"加载主配置文件: {args.config}")
    try:
        project_config = load_yaml_config(args.config)
    except (FileNotFoundError, ConfigError, Exception) as e:
        logger.critical(f"加载主配置文件失败: {e}", exc_info=True)
        return None

    logger.info(f"加载任务配置文件: {args.task_config}")
    try:
        # 任务配置直接加载,不提取 section
        # 假设 task_config.yaml 的顶层是一个包含 'tasks' 列表的字典
        task_config_content = load_yaml_config(args.task_config)
        if 'tasks' not in task_config_content or not isinstance(task_config_content['tasks'], list):
             raise ConfigError("任务配置文件必须包含一个名为 'tasks' 的列表")
        # 将 tasks 列表合并到主配置中
        # 使用深合并,避免覆盖其他顶级键
        # project_config = common_utils.merge_dictionaries(project_config, task_config_content, deep=True)
        # 或者简单地将 tasks 列表添加到主配置
        project_config['tasks'] = task_config_content['tasks']

    except (FileNotFoundError, ConfigError, Exception) as e:
        logger.critical(f"加载或合并任务配置文件失败: {e}", exc_info=True)
        return None

    return project_config


def main():
    """主执行函数"""
    # 1. 解析参数
    args = parse_arguments()

    # 2. 初始化日志
    # 使用 task_id 或 'main' 作为日志文件名
    log_task_name = f"main_{args.task_id}" if args.task_id != "ALL" else "main_all"
    log_directory = args.log_dir if args.log_dir else DEFAULT_LOG_DIR
    try:
        setup_logger(
            log_dir=log_directory,
            task_name=log_task_name,
            log_level=args.log_level.upper()
        )
        logger.info("日志系统初始化完成")
    except Exception as log_e:
        print(f"错误初始化日志系统失败: {log_e}", file=sys.stderr)
        # 即使日志失败,也尝试继续,但可能没有文件日志
        logger.remove() # 移除可能存在的处理器
        logger.add(sys.stderr, level=args.log_level.upper()) # 保证控制台输出
        logger.error("初始化文件日志失败,仅使用控制台输出")


    # 3. 加载配置
    logger.info("开始加载配置...")
    full_config = load_and_merge_configs(args)
    if full_config is None:
        logger.critical("无法加载配置,程序终止")
        sys.exit(1)
    logger.info("配置加载完成")

    # 4. 执行任务
    task_list: List[Dict[str, Any]] = full_config.get('tasks', [])
    if not task_list:
         logger.warning("任务配置列表为空,无任务执行")
         return

    # 获取所有任务的 ID
    all_task_ids = [str(t.get('task_id', '')) for t in task_list if isinstance(t, dict) and t.get('task_id')]

    tasks_to_run: List[str] = []
    if args.task_id.upper() == "ALL":
        tasks_to_run = all_task_ids
        logger.info(f"准备执行所有配置的任务 (共 {len(tasks_to_run)} 个)")
    elif args.task_id in all_task_ids:
        tasks_to_run = [args.task_id]
        logger.info(f"准备执行指定任务: {args.task_id}")
    else:
        logger.error(f"指定的任务 ID '{args.task_id}' 在任务配置中未找到可用任务: {all_task_ids}")
        sys.exit(1)

    # 循环执行选定的任务
    success_count = 0
    fail_count = 0
    skipped_count = 0

    for task_id_to_run in tasks_to_run:
        task_conf = next((t for t in task_list if t.get('task_id') == task_id_to_run), None)
        # 再次检查配置是否存在且启用
        if not task_conf:
            logger.error(f"内部错误无法找到任务 '{task_id_to_run}' 的配置（已在列表但未找到?）")
            fail_count += 1
            continue
        if not task_conf.get('enabled', False):
             logger.info(f"任务 '{task_id_to_run}' 已配置但未启用 (enabled=false),跳过")
             skipped_count += 1
             continue

        logger.info(f"****** 开始执行任务 '{task_id_to_run}' ******")
        task_start_time = time.time()
        try:
            # 实例化并运行任务驱动器
            runner = WindTaskRunner(task_id=task_id_to_run, config=full_config)
            runner.run() # WindTaskRunner 内部处理训练/预测逻辑和错误
            # 假设 run() 内部处理了所有预期的异常,如果 run() 成功结束就算成功
            success_count += 1
        except (ValueError, RuntimeError, Exception) as task_e:
             # 捕获初始化或运行期间的严重错误
             logger.error(f"执行任务 '{task_id_to_run}' 过程中发生严重错误!")
             logger.exception(task_e) # 记录完整堆栈
             fail_count += 1
        task_end_time = time.time()
        logger.info(f"****** 任务 '{task_id_to_run}' 执行结束 (耗时: {task_end_time - task_start_time:.2f} 秒) ******")
        # 添加分隔符以便区分不同任务的日志
        logger.info("-" * 80)


    # --- 结束报告 ---
    logger.info("所有请求的任务已执行完毕")
    logger.info(f"执行总结: 成功={success_count}, 失败={fail_count}, 跳过={skipped_count}")

    if fail_count > 0:
        sys.exit(1) # 有失败任务则以非零状态码退出
    else:
        sys.exit(0)

if __name__ == "__main__":
    # 设置基本的日志记录器,以防 setup_logger 失败
    logger.add(sys.stderr, level="INFO")
    main()