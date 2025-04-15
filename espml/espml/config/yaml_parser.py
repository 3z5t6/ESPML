# -*- coding: utf-8 -*-
"""
YAML 配置文件解析器 (espml)
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from loguru import logger # 使用项目统一的 logger

# --- 自定义配置错误 ---
class ConfigError(Exception):
    """当加载或解析配置文件发生错误时引发"""
    pass

# --- YAML 加载函数 ---
def load_yaml_config(
    config_path: Union[str, Path],
    section: Optional[str] = None,
    encoding: str = 'utf-8'
) -> Dict[str, Any]:
    """
    加载指定的 YAML 配置文件

    可以加载整个文件或仅加载文件中的特定顶级部分

    Args:
        config_path (Union[str, Path]): YAML 配置文件的完整路径
        section (Optional[str]): (可选) 如果提供,则只返回 YAML 文件中以此为键的顶级部分
                                    如果为 None,则返回整个配置字典
        encoding (str): 读取文件时使用的编码,默认为 'utf-8'

    Returns:
        Dict[str, Any]: 解析后的配置字典如果指定了 section,则返回该部分的字典

    Raises:
        ConfigError: 如果文件不存在、不是有效的 YAML、顶层不是字典、
                     无法读取文件或指定的 section 不存在
        FileNotFoundError: 如果 config_path 指向的文件不存在 (由 Pathlib 引发)
        yaml.YAMLError: 如果 YAML 文件格式无效 (由 PyYAML 引发)
        IOError: 如果发生文件读取错误
    """
    config_file = Path(config_path)
    logger.debug(f"尝试加载配置文件: {config_file}, Section: {section}")

    if not config_file.is_file():
        logger.error(f"配置文件未找到: {config_file}")
        # 这里会由 Pathlib 引发 FileNotFoundError,无需手动引发
        # 但为了明确,可以在上层捕获 FileNotFoundError 并包装为 ConfigError
        raise FileNotFoundError(f"配置文件未找到: {config_file}")

    try:
        with open(config_file, 'r', encoding=encoding) as f:
            # 使用 safe_load 防止执行任意代码
            full_config = yaml.safe_load(f)

        if not isinstance(full_config, dict):
            logger.error(f"配置文件顶层结构必须是字典,但得到的是 {type(full_config)}: {config_file}")
            raise ConfigError(f"配置文件顶层结构必须是字典: {config_file}")

        logger.info(f"成功加载配置文件: {config_file}")

        if section:
            logger.debug(f"尝试提取配置部分: '{section}'")
            if section in full_config:
                section_config = full_config[section]
                if not isinstance(section_config, dict):
                     logger.warning(f"配置部分 '{section}' 的值不是字典 (类型: {type(section_config)}),将按原样返回")
                return section_config
            else:
                logger.error(f"在配置文件 {config_file} 中未找到指定的顶级部分: '{section}'")
                raise ConfigError(f"在配置文件 {config_file} 中未找到顶级部分: '{section}'")
        else:
            # 返回整个配置字典
            return full_config

    except yaml.YAMLError as e:
        logger.error(f"解析配置文件时发生 YAML 格式错误: {config_file}\n错误详情: {e}", exc_info=True)
        raise ConfigError(f"YAML 格式错误: {config_file}") from e
    except IOError as e:
        logger.error(f"读取配置文件时发生 IO 错误: {config_file}\n错误详情: {e}", exc_info=True)
        raise ConfigError(f"读取配置文件 IO 错误: {config_file}") from e
    except Exception as e:
        # 捕获其他潜在异常 (例如权限问题)
        logger.error(f"加载或解析配置文件时发生未知错误: {config_file}\n错误详情: {e}", exc_info=True)
        raise ConfigError(f"加载配置文件未知错误: {config_file}") from e