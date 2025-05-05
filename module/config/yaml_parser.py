# -*- coding: utf-8 -*-
"""
YAML Configuration File Parser (espml)
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from loguru import logger # Using project's unified logger

# --- Custom Configuration Error ---
class ConfigError(Exception):
    """Raised when an error occurs while loading or parsing configuration files"""
    pass

# --- YAML Loading Function ---
def load_yaml_config(
    config_path: Union[str, Path],
    section: Optional[str] = None,
    encoding: str = 'utf-8'
) -> Dict[str, Any]:
    """
    Load the specified YAML configuration file

    Can load the entire file or just a specific top-level section of the file

    Args:
        config_path (Union[str, Path]): Full path to the YAML configuration file
        section (Optional[str]): (Optional) If provided, only returns the top-level section of the YAML file with this key
                                If None, returns the entire configuration dictionary
        encoding (str): Encoding used when reading the file, defaults to 'utf-8'

    Returns:
        Dict[str, Any]: Parsed configuration dictionary; if section is specified, returns the dictionary for that section

    Raises:
        ConfigError: If the file doesn't exist, isn't valid YAML, top level isn't a dictionary,
                    can't read the file, or the specified section doesn't exist
        FileNotFoundError: If the file pointed to by config_path doesn't exist (raised by Pathlib)
        yaml.YAMLError: If the YAML file format is invalid (raised by PyYAML)
        IOError: If a file reading error occurs
    """
    config_file = Path(config_path)
    logger.debug(f"Attempting to load configuration file: {config_file}, Section: {section}")

    if not config_file.is_file():
        logger.error(f"Configuration file not found: {config_file}")
        # FileNotFoundError will be raised by Pathlib here, no need to manually raise
        # But for clarity, you can catch FileNotFoundError at a higher level and wrap it as ConfigError
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    try:
        with open(config_file, 'r', encoding=encoding) as f:
            # Use safe_load to prevent arbitrary code execution
            full_config = yaml.safe_load(f)

        if not isinstance(full_config, dict):
            logger.error(f"Configuration file top-level structure must be a dictionary, but got {type(full_config)}: {config_file}")
            raise ConfigError(f"Configuration file top-level structure must be a dictionary: {config_file}")

        logger.info(f"Successfully loaded configuration file: {config_file}")

        if section:
            logger.debug(f"Attempting to extract configuration section: '{section}'")
            if section in full_config:
                section_config = full_config[section]
                if not isinstance(section_config, dict):
                     logger.warning(f"Configuration section '{section}' value is not a dictionary (type: {type(section_config)}), will return as is")
                return section_config
            else:
                logger.error(f"Specified top-level section not found in configuration file {config_file}: '{section}'")
                raise ConfigError(f"Top-level section not found in configuration file {config_file}: '{section}'")
        else:
            # Return the entire configuration dictionary
            return full_config

    except yaml.YAMLError as e:
        logger.error(f"YAML format error occurred while parsing configuration file: {config_file}\nError details: {e}", exc_info=True)
        raise ConfigError(f"YAML format error: {config_file}") from e
    except IOError as e:
        logger.error(f"IO error occurred while reading configuration file: {config_file}\nError details: {e}", exc_info=True)
        raise ConfigError(f"IO error reading configuration file: {config_file}") from e
    except Exception as e:
        # Catch other potential exceptions (e.g., permission issues)
        logger.error(f"Unknown error occurred while loading or parsing configuration file: {config_file}\nError details: {e}", exc_info=True)
        raise ConfigError(f"Unknown error loading configuration file: {config_file}") from e