# tests/util/test_log_config.py
import pytest
from pathlib import Path
from loguru import logger
import os
from espml.util.log_config import setup_logger, LOG_DIR # 导入被测函数和常量

# fixture 用于清理日志文件
@pytest.fixture(autouse=True)
def cleanup_logs(temp_output_dir: Path):
    # 测试前无需操作，测试后清理
    yield
    log_files = list(temp_output_dir.glob("*.log"))
    for f in log_files:
        try:
            f.unlink()
        except OSError:
            pass # 忽略删除错误

def test_setup_logger_defaults(temp_output_dir: Path):
    """测试使用默认参数设置 logger (INFO 级别，控制台+文件)"""
    logger.remove() # 清理现有 handlers
    task_name = "default_test"
    log_file = temp_output_dir / f"{task_name}.log"
    setup_logger(log_dir=temp_output_dir, task_name=task_name)

    assert len(logger._core.handlers) >= 2 # 至少有控制台和文件 handler
    assert log_file.exists()

    # 记录一条消息并检查文件内容
    logger.info("Test message for default setup.")
    log_content = log_file.read_text(encoding='utf-8')
    assert "Test message for default setup." in log_content
    assert "INFO" in log_content

def test_setup_logger_debug_level(temp_output_dir: Path):
    """测试设置 DEBUG 级别"""
    logger.remove()
    task_name = "debug_test"
    log_file = temp_output_dir / f"{task_name}.log"
    setup_logger(log_dir=temp_output_dir, task_name=task_name, log_level="DEBUG")

    logger.debug("This is a debug message.")
    log_content = log_file.read_text(encoding='utf-8')
    assert "This is a debug message." in log_content
    assert "DEBUG" in log_content

def test_setup_logger_no_file(temp_output_dir: Path):
    """测试禁用文件日志"""
    logger.remove()
    task_name = "no_file_test"
    log_file = temp_output_dir / f"{task_name}.log"
    setup_logger(log_dir=temp_output_dir, task_name=task_name, enable_file=False)

    assert not log_file.exists()
    assert len(logger._core.handlers) >= 1 # 应该还有控制台 handler

def test_setup_logger_no_console(temp_output_dir: Path):
    """测试禁用控制台日志"""
    logger.remove()
    task_name = "no_console_test"
    log_file = temp_output_dir / f"{task_name}.log"
    setup_logger(log_dir=temp_output_dir, task_name=task_name, enable_console=False)

    assert log_file.exists()
    # 验证 stderr 是否有输出比较困难，主要验证文件写入
    logger.info("Test message, no console.")
    log_content = log_file.read_text(encoding='utf-8')
    assert "Test message, no console." in log_content

def test_setup_logger_invalid_level(temp_output_dir: Path):
    """测试无效的日志级别"""
    logger.remove()
    with pytest.raises(ValueError, match="无效的日志级别"):
        setup_logger(log_dir=temp_output_dir, log_level="INVALID")

def test_setup_logger_task_name_in_file(temp_output_dir: Path):
    """测试 task_name 是否正确用于文件名"""
    logger.remove()
    custom_task = "MyCustomTask_123"
    setup_logger(log_dir=temp_output_dir, task_name=custom_task)
    expected_file = temp_output_dir / f"{custom_task}.log"
    assert expected_file.exists()

# 可以添加测试 rotation 和 retention 的场景，但这比较复杂，通常需要模拟时间和文件大小