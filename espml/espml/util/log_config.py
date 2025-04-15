# -*- coding: utf-8 -*-
"""
日志配置模块 (espml)
使用 Loguru 实现,严格遵循项目要求的日志格式和行为
"""

import sys
import os
from pathlib import Path
from typing import Optional, Union
from loguru import logger

# --- 日志配置常量 ---
# 使用环境变量 ESPML_LOG_DIR 定义日志目录,否则默认为项目根目录下的 'logs' 文件夹
# 确保路径基于 log_config.py 文件位置计算项目根目录是可靠的
_DEFAULT_LOG_DIR_NAME = "logs"
try:
    # 假设 log_config.py 在 espml/util/ 下
    _PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
except NameError:
    # 在某些环境 (如交互式解释器) __file__ 可能未定义
    _PROJECT_ROOT = Path.cwd() # 使用当前工作目录作为备选

_DEFAULT_LOG_DIR = _PROJECT_ROOT / _DEFAULT_LOG_DIR_NAME
LOG_DIR = Path(os.getenv("ESPML_LOG_DIR", _DEFAULT_LOG_DIR))

# 日志格式,精确匹配示例: "时间戳 - 模块:函数:行号 - 级别 - 消息"
# Loguru 格式: <green>{time:YYYY-MM-DD HH:mm:ss,SSS}</green> - <cyan>{name}.{module}.{function}:{line}</cyan> - <level>{level: <8}</level> - <level>{message}</level>
# 注意: 示例中似乎混合了 logger name 和 module name,Loguru 的 {name} 通常是 logger add 时指定的,{module} 是文件名
# 为了匹配示例,我们可能需要调整 logger 的名称或者格式字符串# 或者更通用的: {time:YYYY-MM-DD HH:mm:ss,SSS} - {module}:{function}:{line} - {level} - {message}
# 最终决定使用最接近示例并且 Loguru 标准支持的格式
LOG_FORMAT_CONSOLE = (
    "<green>{time:YYYY-MM-DD HH:mm:ss,SSS}</green> - "
    "<cyan>{name}:{function}:{line}</cyan> - " # 使用 logger 名称
    "<level>{level: <8}</level> - "
    "<level>{message}</level>"
)
LOG_FORMAT_FILE = ( # 文件中通常不需要颜色
    "{time:YYYY-MM-DD HH:mm:ss,SSS} - "
    "{name}:{function}:{line} - "
    "{level: <8} - "
    "{message}"
)

# 日志文件轮转策略
ROTATION_POLICY = "100 MB"  # 每个日志文件最大 100MB
RETENTION_POLICY = "14 days" # 最多保留 14 天的日志文件
ENCODING = "utf-8"       # 日志文件编码
DEFAULT_LOG_LEVEL = "INFO" # 默认日志级别

# --- 全局日志句柄 ID ---
# 用于后续可能移除或修改 handler
_console_handler_id: Optional[int] = None
_file_handler_id: Optional[int] = None

# --- 配置函数 ---
def setup_logger(
    log_dir: Union[str, Path] = LOG_DIR,
    task_name: Optional[str] = None,
    log_level: str = DEFAULT_LOG_LEVEL,
    enable_console: bool = True,
    enable_file: bool = True,
    rotation: str = ROTATION_POLICY,
    retention: str = RETENTION_POLICY,
    encoding: str = ENCODING,
) -> None:
    """
    配置全局 Loguru 日志记录器

    此函数会移除所有现有的处理器,然后根据参数添加控制台和/或文件处理器

    Args:
        log_dir (Union[str, Path]): 日志文件存放目录默认为 LOG_DIR 常量
        task_name (Optional[str]): 当前任务的名称用于生成日志文件名 (例如 'Forecast4Hour_train.log')
                                    如果为 None,将使用通用日志文件 'espml_global.log'
        log_level (str): 最低日志记录级别 (例如 "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        enable_console (bool): 是否启用控制台 (stderr) 输出
        enable_file (bool): 是否启用文件输出
        rotation (str): 文件轮转策略 (例如 "100 MB", "1 week", "00:00")
        retention (str): 文件保留策略 (例如 "14 days", "1 month", 10)
        encoding (str): 日志文件编码

    Raises:
        ValueError: 如果 log_level 无效
        IOError: 如果无法创建日志目录或写入日志文件
    """
    global _console_handler_id, _file_handler_id

    # 验证日志级别
    log_level_upper = log_level.upper()
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "TRACE"]
    if log_level_upper not in valid_levels:
        raise ValueError(f"无效的日志级别: {log_level}有效级别为: {valid_levels}")

    # 移除所有旧的处理器,确保配置的幂等性
    logger.remove()
    _console_handler_id = None
    _file_handler_id = None

    # 配置控制台输出
    if enable_console:
        try:
            _console_handler_id = logger.add(
                sys.stderr,
                level=log_level_upper,
                format=LOG_FORMAT_CONSOLE,
                colorize=True,  # 在控制台启用颜色
                enqueue=True,   # 异步,避免阻塞主线程
                catch=True      # 捕获处理器内部异常
            )
            logger.info("控制台日志已启用")
        except Exception as e:
            # Fallback to basic print if logger setup fails critically
            print(f"CRITICAL: 无法配置控制台日志: {e}", file=sys.stderr)

    # 配置日志文件输出
    if enable_file:
        log_directory = Path(log_dir)
        try:
            log_directory.mkdir(parents=True, exist_ok=True) # 确保目录存在
            log_file_name = f"{task_name}.log" if task_name else "espml_global.log"
            log_file_path = log_directory / log_file_name

            _file_handler_id = logger.add(
                log_file_path,
                level=log_level_upper,
                format=LOG_FORMAT_FILE,
                rotation=rotation,
                retention=retention,
                encoding=encoding,
                enqueue=True,      # 异步写入
                backtrace=True,    # 发生异常时记录完整的堆栈跟踪
                diagnose=True,     # 发生异常时记录详细的变量诊断信息
                catch=True         # 捕获处理器内部异常
            )
            logger.info(f"文件日志已启用,将写入: {log_file_path}")
            # 打印 FLAML 日志示例中出现的日志文件路径信息
            # 注意: FLAML 的日志格式与我们自定义的不同,它会自己管理日志文件
            # 我们这里只记录我们自己的日志路径
            # 若要完全模拟,需要找到 FLAML 如何配置日志路径并传递参数

        except Exception as e:
            logger.error(f"无法配置日志文件写入: {log_directory}/{task_name or 'espml_global'}.log - {e}", exc_info=True)
            if not enable_console: # 如果文件失败且控制台未启用,强制启用控制台
                try:
                    _console_handler_id = logger.add(sys.stderr, level=log_level_upper, format=LOG_FORMAT_CONSOLE, colorize=True, enqueue=True, catch=True)
                    logger.warning("文件日志配置失败,已强制启用控制台日志")
                except Exception as fallback_e:
                    print(f"CRITICAL: 无法配置控制台日志作为后备: {fallback_e}", file=sys.stderr)

    # 记录初始化完成信息
    logger.info(f"日志系统为任务 '{task_name or 'global'}' 初始化完成,级别: {log_level_upper}")

# --- 获取配置好的 logger ---
# 其他模块应该直接 `from loguru import logger` 来使用配置好的实例
# logger.info("这是一个示例日志消息")

# --- (可选) 配置 FLAML 日志 ---
# FLAML 使用标准 logging 模块可以获取其 logger 并添加 handler,
# 或者设置其日志级别
# import logging
# flaml_logger = logging.getLogger("flaml.automl")
# flaml_logger.setLevel(logging.INFO) # 设置 FLAML 的日志级别
# # 如果需要将 FLAML 日志也输出到我们的文件:
# # flaml_logger.addHandler(your_loguru_compatible_handler)
# # 但这可能导致格式混乱,通常让 FLAML 自己管理其日志文件更好
# # 关键在于配置 FLAML 任务时指定 `log_file_name` 参数,如日志示例所示