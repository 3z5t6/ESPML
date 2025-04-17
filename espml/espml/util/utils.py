# -*- coding: utf-8 -*-
"""
通用工具函数模块 (espml)
包含项目中多个模块可能共用的辅助函数
"""

import os
import sys
import json
import pickle
import time
import datetime
import pytz
import re
import logging
import hashlib
import warnings
import shutil
import glob
import tempfile
import subprocess
import pandas as pd
import functools
import gzip
import bz2
import zipfile
import tarfile
from pathlib import Path
from typing import (Any, Dict, List, Optional, Union, Callable, TypeVar,
                    Iterable, Sequence, Tuple, ContextManager, Generator)
from collections.abc import Mapping # 用于更准确的字典类型提示
# from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed # 如果需要并行处理

from loguru import logger # 统一使用 loguru

T = TypeVar('T') # 用于泛型类型提示

# --- 常量 ---
DEFAULT_ENCODING = 'utf-8'
DATE_FORMAT_ISO = '%Y-%m-%d'
DATETIME_FORMAT_ISO = '%Y-%m-%d %H:%M:%S'
DATETIME_FORMAT_FILENAME = '%Y%m%d_%H%M%S'

# --- 文件和目录操作 ---

def mkdir_if_not_exist(dir_path: Union[str, Path], mode: int = 0o775) -> bool:
    """
    如果目录不存在,则创建它 (包括父目录)

    Args:
        dir_path (Union[str, Path]): 需要创建的目录路径
        mode (int): 创建目录时设置的权限模式 (八进制)

    Returns:
        bool: 如果成功创建或目录已存在,返回 True；否则返回 False
    """
    path = Path(dir_path)
    if not path.exists():
        try:
            path.mkdir(parents=True, exist_ok=True, mode=mode)
            logger.debug(f"成功创建目录: {path}")
            # 尝试设置权限 (mkdir mode 参数在某些系统下可能不完全生效)
            try:
                os.chmod(path, mode)
            except OSError:
                logger.warning(f"设置目录权限失败 (可能由于系统限制): {path}")
            return True
        except Exception as e:
            logger.error(f"创建目录失败: {path} - {e}", exc_info=True)
            return False
    elif not path.is_dir():
        logger.error(f"路径已存在但不是目录: {path}")
        return False
    else:
        # logger.debug(f"目录已存在: {path}")
        return True

def check_path_exists(path: Union[str, Path], path_type: Optional[str] = None, raise_error: bool = False) -> bool:
    """
    检查路径是否存在,并可选地检查其类型（文件 'f' 或目录 'd'）

    Args:
        path (Union[str, Path]): 路径
        path_type (Optional[str]): 'f' 检查是否为文件, 'd' 检查是否为目录None 则只检查存在性
        raise_error (bool): 如果检查失败,是否引发异常

    Returns:
        bool: 如果检查通过,返回 True；否则返回 False (除非 raise_error=True)

    Raises:
        FileNotFoundError: 如果 raise_error=True 且路径不存在
        TypeError: 如果 raise_error=True 且路径类型不匹配
    """
    p = Path(path)
    exists = p.exists()

    if not exists:
        if raise_error:
            logger.error(f"路径检查失败路径不存在 {p}")
            raise FileNotFoundError(f"路径不存在: {p}")
        return False

    if path_type == 'f':
        is_correct_type = p.is_file()
        if not is_correct_type and raise_error:
            logger.error(f"路径检查失败路径不是文件 {p}")
            raise TypeError(f"路径不是文件: {p}")
        return is_correct_type
    elif path_type == 'd':
        is_correct_type = p.is_dir()
        if not is_correct_type and raise_error:
            logger.error(f"路径检查失败路径不是目录 {p}")
            raise TypeError(f"路径不是目录: {p}")
        return is_correct_type
    elif path_type is None:
        return True # 只检查存在性,已通过
    else:
         logger.warning(f"未知的路径类型检查: '{path_type}'")
         return False # 未知类型检查视为失败

def safe_remove(path: Union[str, Path], ignore_errors: bool = True) -> bool:
    """
    安全地删除文件或目录（递归删除）

    Args:
        path (Union[str, Path]): 要删除的路径
        ignore_errors (bool): 是否忽略删除过程中发生的错误

    Returns:
        bool: 如果成功删除或路径不存在,返回 True；否则返回 False
    """
    p = Path(path)
    try:
        if p.is_file() or p.is_symlink():
            p.unlink()
            logger.debug(f"已删除文件/链接: {p}")
        elif p.is_dir():
            shutil.rmtree(p)
            logger.debug(f"已递归删除目录: {p}")
        else:
            logger.debug(f"路径不存在,无需删除: {p}")
        return True
    except Exception as e:
        logger.error(f"删除路径时出错: {p} - {e}", exc_info=not ignore_errors)
        if not ignore_errors:
            raise
        return False

def list_files_in_dir(dir_path: Union[str, Path], pattern: str = '*', recursive: bool = False, full_path: bool = True) -> List[Path]:
    """
    列出目录中符合特定模式的文件

    Args:
        dir_path (Union[str, Path]): 目录路径
        pattern (str): 文件名匹配模式 (例如 '*.csv', 'data_*.txt')
        recursive (bool): 是否递归查找子目录
        full_path (bool): 是否返回文件的绝对路径

    Returns:
        List[Path]: 包含符合条件的文件路径对象的列表如果目录不存在则返回空列表
    """
    p = Path(dir_path)
    if not p.is_dir():
        logger.warning(f"尝试列出文件,但目录不存在或不是目录: {p}")
        return []

    try:
        if recursive:
            file_generator = p.rglob(pattern)
        else:
            file_generator = p.glob(pattern)

        files = [f for f in file_generator if f.is_file()]

        if full_path:
             # glob 默认返回相对路径（如果 p 是相对的）或绝对路径（如果 p 是绝对的）
             # 确保返回绝对路径
             files = [f.resolve() for f in files]

        logger.debug(f"在 '{p}' 中找到 {len(files)} 个匹配 '{pattern}' 的文件 (recursive={recursive})")
        return files
    except Exception as e:
         logger.error(f"列出目录文件时出错: {p} - {e}", exc_info=True)
         return []

def get_file_modification_time(file_path: Union[str, Path]) -> Optional[datetime.datetime]:
    """获取文件的最后修改时间"""
    p = Path(file_path)
    try:
        check_path_exists(p, path_type='f', raise_error=True)
        mtime_timestamp = p.stat().st_mtime
        return datetime.datetime.fromtimestamp(mtime_timestamp)
    except (FileNotFoundError, TypeError):
        return None
    except Exception as e:
         logger.error(f"获取文件修改时间失败: {p} - {e}")
         return None

def get_file_size(file_path: Union[str, Path], human_readable: bool = False) -> Union[int, str, None]:
    """获取文件大小"""
    p = Path(file_path)
    try:
        check_path_exists(p, path_type='f', raise_error=True)
        size_bytes = p.stat().st_size
        if human_readable:
            # 转换为易读格式 (KB, MB, GB)
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024**2:
                return f"{size_bytes / 1024:.2f} KB"
            elif size_bytes < 1024**3:
                return f"{size_bytes / (1024**2):.2f} MB"
            else:
                return f"{size_bytes / (1024**3):.2f} GB"
        else:
            return size_bytes
    except (FileNotFoundError, TypeError):
        return None
    except Exception as e:
        logger.error(f"获取文件大小失败: {p} - {e}")
        return None

# --- 压缩/解压文件 (推断) ---

def compress_file(input_path: Union[str, Path], output_path: Union[str, Path], compression: str = 'gzip', level: int = 9) -> bool:
    """压缩单个文件"""
    in_p = Path(input_path)
    out_p = Path(output_path)
    check_path_exists(in_p, path_type='f', raise_error=True)
    mkdir_if_not_exist(out_p.parent)

    comp_lower = compression.lower()
    logger.info(f"开始使用 '{comp_lower}' 压缩文件: {in_p} -> {out_p} (level={level})")

    try:
        if comp_lower == 'gzip':
            with open(in_p, 'rb') as f_in, gzip.open(out_p, 'wb', compresslevel=level) as f_out:
                shutil.copyfileobj(f_in, f_out)
        elif comp_lower == 'bz2':
            with open(in_p, 'rb') as f_in, bz2.open(out_p, 'wb', compresslevel=level) as f_out:
                shutil.copyfileobj(f_in, f_out)
        elif comp_lower == 'zip':
            with zipfile.ZipFile(out_p, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=level) as zf:
                 zf.write(in_p, arcname=in_p.name) # arcname 指定在 zip 文件中的名称
        else:
            logger.error(f"不支持的压缩格式: {compression}")
            return False
        logger.info(f"文件压缩成功: {out_p}")
        return True
    except Exception as e:
        logger.error(f"压缩文件时出错: {e}", exc_info=True)
        safe_remove(out_p) # 清理可能产生的损坏的输出文件
        return False

def decompress_file(input_path: Union[str, Path], output_dir: Union[str, Path], compression: Optional[str] = None) -> bool:
    """解压单个文件到指定目录"""
    in_p = Path(input_path)
    out_d = Path(output_dir)
    check_path_exists(in_p, path_type='f', raise_error=True)
    mkdir_if_not_exist(out_d)

    comp_lower = compression.lower() if compression else Path(input_path).suffix.lower().strip('.')
    logger.info(f"开始使用 '{comp_lower}' 解压文件: {in_p} -> {out_d}")

    output_filename = in_p.stem # 默认解压后的文件名 (去除压缩后缀)
    output_path = out_d / output_filename

    try:
        if comp_lower == 'gz' or comp_lower == 'gzip':
            with gzip.open(in_p, 'rb') as f_in, open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        elif comp_lower == 'bz2':
            with bz2.open(in_p, 'rb') as f_in, open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        elif comp_lower == 'zip':
            with zipfile.ZipFile(in_p, 'r') as zf:
                 zf.extractall(path=out_d) # 解压 zip 包内所有文件到目录
                 # 如果 zip 包只有一个文件且名称与压缩包不同,这里的 output_path 可能不准确
                 # 需要更复杂的逻辑来确定解压后的确切文件名
                 logger.warning(f"解压 ZIP 文件 '{in_p}' 到目录 '{out_d}',具体文件名取决于包内容")
                 # 此处简单假设解压成功即可
        elif comp_lower == 'tar':
             with tarfile.open(in_p, 'r') as tf:
                  tf.extractall(path=out_d)
        elif comp_lower == 'gz' and input_path.endswith('.tar.gz'): # 处理 .tar.gz
             with tarfile.open(in_p, 'r:gz') as tf:
                  tf.extractall(path=out_d)
        elif comp_lower == 'bz2' and input_path.endswith('.tar.bz2'): # 处理 .tar.bz2
             with tarfile.open(in_p, 'r:bz2') as tf:
                  tf.extractall(path=out_d)
        else:
            logger.error(f"不支持或无法推断的解压格式: {comp_lower} (来自文件 {in_p})")
            return False
        logger.info(f"文件解压成功: {in_p} -> {out_d}")
        return True
    except Exception as e:
        logger.error(f"解压文件时出错: {e}", exc_info=True)
        return False

# --- JSON / Pickle 操作 ---

def read_json_file(file_path: Union[str, Path], encoding: str = DEFAULT_ENCODING, **kwargs) -> Optional[Any]:
    """读取 JSON 文件"""
    if not check_path_exists(file_path, path_type='f'):
        logger.warning(f"尝试读取不存在的 JSON 文件: {file_path}")
        return None
    path = Path(file_path)
    try:
        with open(path, 'r', encoding=encoding) as f:
            data = json.load(f, **kwargs)
        # logger.debug(f"成功读取 JSON 文件: {path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"解析 JSON 文件失败: {path} - {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"读取 JSON 文件时发生错误: {path} - {e}", exc_info=True)
        return None

def write_json_file(data: Any, file_path: Union[str, Path], encoding: str = DEFAULT_ENCODING, indent: Optional[int] = 4, ensure_ascii: bool = False, **kwargs) -> bool:
    """将 Python 对象写入 JSON 文件"""
    path = Path(file_path)
    if not mkdir_if_not_exist(path.parent): return False
    try:
        with open(path, 'w', encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent, **kwargs)
        # logger.debug(f"成功写入 JSON 文件: {path}")
        return True
    except TypeError as e:
         logger.error(f"写入 JSON 失败数据包含无法序列化的类型 - {e}", exc_info=True)
         return False
    except Exception as e:
        logger.error(f"写入 JSON 文件时发生错误: {path} - {e}", exc_info=True)
        return False

def load_pickle(file_path: Union[str, Path], **kwargs) -> Optional[Any]:
    """安全地加载 Pickle 文件"""
    if not check_path_exists(file_path, path_type='f'):
        logger.warning(f"尝试加载不存在的 Pickle 文件: {file_path}")
        return None
    path = Path(file_path)
    try:
        with open(path, 'rb') as f:
            # 添加 encoding='latin1' 或 'bytes' 可能解决一些版本兼容问题
            # 但需要小心,最好确保保存和加载使用兼容的 Python 版本和库版本
            obj = pickle.load(f, **kwargs)
        # logger.debug(f"成功加载 Pickle 文件: {path}")
        return obj
    except pickle.UnpicklingError as e:
        logger.error(f"加载 Pickle 文件失败 (文件可能已损坏或不兼容): {path} - {e}", exc_info=True)
        return None
    except EOFError as e:
        logger.error(f"加载 Pickle 文件失败 (文件不完整): {path} - {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"加载 Pickle 文件时发生错误: {path} - {e}", exc_info=True)
        return None

def dump_pickle(obj: Any, file_path: Union[str, Path], protocol: int = pickle.HIGHEST_PROTOCOL, **kwargs) -> bool:
    """将 Python 对象保存为 Pickle 文件"""
    path = Path(file_path)
    if not mkdir_if_not_exist(path.parent): return False
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f, protocol=protocol, **kwargs)
        # logger.debug(f"成功保存 Pickle 文件: {path}")
        return True
    except pickle.PicklingError as e:
        logger.error(f"保存 Pickle 文件失败 (对象可能无法序列化): {path} - {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"保存 Pickle 文件时发生错误: {path} - {e}", exc_info=True)
        return False

# --- 字典操作 ---

def safe_dict_get(data: Optional[Mapping], key_path: Union[str, List[str]], default: Any = None) -> Any:
    """
    安全地从嵌套字典（或支持 get 方法的对象）中获取值
    """
    if data is None:
        return default

    if isinstance(key_path, str):
        keys = key_path.split('.')
    elif isinstance(key_path, list):
        keys = key_path
    else:
        logger.warning(f"safe_dict_get: 无效的 key_path 类型 ({type(key_path)})")
        return default

    current_level = data
    for key in keys:
        try:
            # 尝试使用 get 方法 (适用于 dict 和一些类 dict 对象)
            if hasattr(current_level, 'get') and callable(getattr(current_level, 'get')):
                 current_level = current_level.get(key)
            # 尝试使用索引 (适用于 list/tuple)
            elif isinstance(current_level, (list, tuple)) and isinstance(key, int) and -len(current_level) <= key < len(current_level):
                 current_level = current_level[key]
            # 尝试属性访问 (适用于对象)
            elif hasattr(current_level, key):
                  current_level = getattr(current_level, key, default) # 使用 getattr 以防属性不存在
                  # 注意属性访问可能不是预期行为,取决于 key 的内容
            else:
                # logger.debug(f"safe_dict_get: 在路径 {key_path} 中未找到键/索引 '{key}'")
                return default

            if current_level is None and key != keys[-1]: # 如果中间步骤为 None,则无法继续
                return default
        except (KeyError, IndexError, TypeError, AttributeError):
            # logger.debug(f"safe_dict_get: 访问路径 {key_path} 中的键/索引 '{key}' 时出错")
            return default
    return current_level if current_level is not None else default

def flatten_nested_dict(d: Mapping, parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """将嵌套字典扁平化"""
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + str(k) if parent_key else str(k)
        if isinstance(v, Mapping):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
             # 可选处理列表中的字典 (或者将列表视为最终值)
             # for i, item in enumerate(v):
             #     if isinstance(item, Mapping):
             #          items.extend(flatten_nested_dict(item, f"{new_key}{sep}{i}", sep=sep).items())
             #     else:
             #          items.append((f"{new_key}{sep}{i}", item))
             items.append((new_key, v)) # 默认不展开列表
        else:
            items.append((new_key, v))
    return dict(items)

def merge_dictionaries(dict1: Dict, dict2: Dict, deep: bool = True) -> Dict:
    """
    合并两个字典如果 deep=True,进行递归合并
    注意: dict2 中的值会覆盖 dict1 中的同名键
    """
    merged = dict1.copy()
    if deep:
        for key, value in dict2.items():
            if isinstance(value, Mapping) and key in merged and isinstance(merged[key], Mapping):
                merged[key] = merge_dictionaries(merged[key], value, deep=True)
            else:
                merged[key] = value
    else:
        merged.update(dict2)
    return merged

# --- 时间日期处理 ---

def parse_datetime_flexible(
    datetime_str: Optional[str],
    fmts: Optional[Union[str, List[str]]] = None,
    tz: Optional[Union[str, datetime.tzinfo]] = None,
    default: Optional[datetime.datetime] = None
    ) -> Optional[datetime.datetime]:
    """
    尝试使用多种格式解析日期时间字符串

    Args:
        datetime_str (Optional[str]): 输入的日期时间字符串
        fmts (Optional[Union[str, List[str]]]): 尝试解析的格式列表
                                                  如果为 None,使用一组常用默认格式
                                                  可以包含 '%Y%m%d%H%M%S' 等无分隔符格式
        tz (Optional[Union[str, datetime.tzinfo]]): 目标时区如果是字符串,将使用 pytz 查找
                                                      如果为 None,返回 naive datetime
        default (Optional[datetime.datetime]): 解析失败时返回的默认值

    Returns:
        Optional[datetime.datetime]: 解析后的 datetime 对象 (aware if tz is provided) 或默认值
    """
    if not datetime_str or not isinstance(datetime_str, str):
        return default

    default_fmts = [
        '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%Y%m%d%H%M%S',
        '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M', '%Y%m%d%H%M',
        '%Y-%m-%d %H', '%Y/%m/%d %H', '%Y%m%d%H',
        '%Y-%m-%d', '%Y/%m/%d', '%Y%m%d',
        '%Y-%m-%dT%H:%M:%S', # ISO 8601
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%SZ', # UTC ISO
        '%Y-%m-%dT%H:%M:%S%z', # ISO with offset
    ]

    formats_to_try = []
    if isinstance(fmts, str):
        formats_to_try.append(fmts)
    elif isinstance(fmts, list):
        formats_to_try.extend(fmts)
    formats_to_try.extend(default_fmts) # 总是尝试默认格式

    parsed_dt = None
    for fmt in formats_to_try:
        try:
            parsed_dt = datetime.datetime.strptime(datetime_str, fmt)
            break # 解析成功,跳出循环
        except ValueError:
            continue # 尝试下一种格式

    if parsed_dt is None:
        # 如果所有格式都失败,尝试让 pandas 自动推断（可能较慢）
        try:
            parsed_dt = pd.to_datetime(datetime_str)
            if isinstance(parsed_dt, pd.Timestamp):
                 parsed_dt = parsed_dt.to_pydatetime() # 转为标准 datetime
            else: # 如果解析结果不是单个时间戳 (例如解析为 NaT)
                 parsed_dt = None
        except Exception:
            logger.warning(f"无法使用任何已知格式或自动推断解析日期时间字符串: '{datetime_str}'")
            return default

    # 处理时区
    if tz and parsed_dt:
        target_tz = None
        if isinstance(tz, str):
            try:
                target_tz = pytz.timezone(tz)
            except pytz.UnknownTimeZoneError:
                logger.error(f"未知的时区字符串: {tz}")
                return default # 或抛出异常
        elif isinstance(tz, datetime.tzinfo):
            target_tz = tz
        else:
             logger.error(f"无效的时区类型: {type(tz)}")
             return default

        if parsed_dt.tzinfo is None:
            # 如果解析出的是 naive datetime,假定它是目标时区（或 UTC? 需要业务逻辑确定）
            # 普遍做法是先 localize 到某个默认时区（如本地或 UTC）,再转换
            # 简单处理直接附加目标时区 (不推荐,除非明确知道字符串就是目标时区)
            # parsed_dt = target_tz.localize(parsed_dt)
            # 更安全的做法是要求输入带时区信息或指定时区
            logger.warning(f"解析得到的 datetime 是 naive 的,但指定了目标时区 {tz}无法安全地进行时区转换返回 naive datetime")
            # return default
        else:
            # 如果解析出的是 aware datetime,转换到目标时区
            try:
                parsed_dt = parsed_dt.astimezone(target_tz)
            except Exception as e:
                 logger.error(f"时区转换失败从 {parsed_dt.tzinfo} 到 {target_tz}: {e}")
                 return default

    return parsed_dt


def format_datetime(
    dt: Optional[datetime.datetime],
    fmt: str = DATETIME_FORMAT_ISO,
    default: str = ""
    ) -> str:
    """格式化 datetime 对象为字符串"""
    if isinstance(dt, datetime.datetime):
        try:
            return dt.strftime(fmt)
        except ValueError: # 处理无效格式
             logger.error(f"无效的日期时间格式字符串: '{fmt}'")
             return default
    return default

def get_current_timestamp_str(fmt: str = DATETIME_FORMAT_FILENAME) -> str:
    """获取当前时间的格式化字符串"""
    return datetime.datetime.now().strftime(fmt)

def generate_time_range(
    start: Union[str, datetime.datetime],
    end: Union[str, datetime.datetime],
    freq: str = 'H',
    tz: Optional[str] = None,
    **kwargs
    ) -> pd.DatetimeIndex:
    """生成指定频率的时间范围索引"""
    try:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        # closed=None, 'left', 'right'
        dt_index = pd.date_range(start=start_dt, end=end_dt, freq=freq, tz=tz, **kwargs)
        return dt_index
    except Exception as e:
         logger.error(f"生成时间范围失败 (start='{start}', end='{end}', freq='{freq}'): {e}", exc_info=True)
         # 返回空索引或抛出异常
         return pd.DatetimeIndex([])


# --- 字符串处理 ---

def generate_unique_id(prefix: str = "", length: int = 8) -> str:
    """生成一个基于时间和随机性的（可能）唯一 ID"""
    timestamp_part = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    random_part = os.urandom(4).hex() # 8 个十六进制字符
    full_id = f"{prefix}{timestamp_part}{random_part}"
    return full_id[:length + len(prefix)] if length > 0 else full_id

def calculate_md5(file_path: Union[str, Path], chunk_size: int = 8192) -> Optional[str]:
    """计算文件的 MD5 哈希值"""
    if not check_path_exists(file_path, path_type='f'):
        logger.error(f"无法计算 MD5:文件不存在 {file_path}")
        return None
    path = Path(file_path)
    md5_hash = hashlib.md5()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except Exception as e:
        logger.error(f"计算文件 MD5 时出错: {path} - {e}", exc_info=True)
        return None

# --- 错误处理与重试 ---

def retry_on_exception(
    exceptions: Union[Tuple[Exception, ...], Exception] = (Exception,),
    tries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    logger_instance: Optional[logging.Logger] = logger # 使用 loguru
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    装饰器当函数抛出指定异常时自动重试

    Args:
        exceptions: 需要重试的异常类型（或元组）
        tries: 最大尝试次数（包括首次尝试）
        delay: 初始重试延迟时间（秒）
        backoff: 延迟时间的指数退避因子 (delay *= backoff)
        logger_instance: 用于记录重试信息的 logger 对象
    """
    if not isinstance(exceptions, tuple):
        exceptions_tuple = (exceptions,)
    else:
        exceptions_tuple = exceptions

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            _tries, _delay = tries, delay
            while _tries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions_tuple as e:
                    _tries -= 1
                    msg = (f"函数 '{func.__name__}' 发生异常: {e}"
                           f"将在 {_delay:.2f} 秒后重试 ({_tries} 次剩余)...")
                    if logger_instance:
                        logger_instance.warning(msg)
                    else:
                         print(msg, file=sys.stderr) # Fallback if no logger
                    time.sleep(_delay)
                    _delay *= backoff
            # 最后一次尝试（或首次尝试如果 tries=1）
            return func(*args, **kwargs)
        return wrapper
    return decorator

# --- 计时工具 (装饰器和上下文管理器) ---

def log_execution_time(logger_instance: logging.Logger = logger, level: str = "INFO") -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    装饰器记录函数执行时间
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            logger_instance.log(level, f"函数 '{func.__name__}' 执行耗时: {elapsed_time:.4f} 秒")
            return result
        return wrapper
    return decorator

class TimerContext(ContextManager[None]):
    """
    上下文管理器测量代码块执行时间
    """
    def __init__(self, name: str = "代码块", logger_instance: logging.Logger = logger, level: str = "INFO"):
        self.name = name
        self.logger = logger_instance
        self.level = level
        self._start_time: Optional[float] = None

    def __enter__(self) -> None:
        self._start_time = time.perf_counter()
        self.logger.log(self.level, f"开始计时: '{self.name}'...")

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._start_time is not None:
            end_time = time.perf_counter()
            elapsed_time = end_time - self._start_time
            self.logger.log(self.level, f"'{self.name}' 执行耗时: {elapsed_time:.4f} 秒")
        else:
            self.logger.warning(f"计时器 '{self.name}' 未正确启动")

# --- 并行处理 (示例,可能需要根据原代码调整) ---

# def parallel_map(func: Callable[[Any], Any], items: Iterable[Any], n_workers: int = -1, mode: str = 'process', chunksize: int = 1) -> List[Any]:
#     """
#     使用多进程或多线程并行执行函数
#
#     Args:
#         func (Callable): 需要并行执行的函数 (必须是可 pickle 的,如果 mode='process')
#         items (Iterable): 需要处理的输入项列表
#         n_workers (int): 工作进程/线程数-1 表示使用所有可用核心,1 表示串行执行
#         mode (str): 'process' 使用多进程, 'thread' 使用多线程
#         chunksize (int): 传递给 executor.map 的块大小
#
#     Returns:
#         List[Any]: 按输入顺序排列的结果列表
#     """
#     if n_workers == 1: # 串行执行
#         logger.debug("parallel_map: n_workers=1,执行串行映射")
#         return [func(item) for item in items]
#
#     items_list = list(items) # 需要具体列表才能获取长度
#     if not items_list:
#         return []
#
#     max_workers = os.cpu_count() if n_workers <= 0 or n_workers > os.cpu_count() else n_workers
#     results = [None] * len(items_list) # 预分配结果列表以保持顺序
#     submitted_futures = {}
#
#     logger.info(f"开始使用 {max_workers} 个 workers ({mode} 模式) 并行处理 {len(items_list)} 个项目...")
#
#     try:
#         Executor = ProcessPoolExecutor if mode == 'process' else ThreadPoolExecutor
#         with Executor(max_workers=max_workers) as executor:
#             # 使用 submit 保持顺序
#             for i, item in enumerate(items_list):
#                  future = executor.submit(func, item)
#                  submitted_futures[future] = i
#
#             # 获取结果并按顺序放入列表
#             for future in as_completed(submitted_futures):
#                  original_index = submitted_futures[future]
#                  try:
#                      result = future.result()
#                      results[original_index] = result
#                  except Exception as e:
#                      logger.error(f"并行任务 (索引 {original_index}) 执行失败: {e}", exc_info=True)
#                      # 可以选择将错误放入结果或抛出
#                      results[original_index] = e # 或者 None, 或者特定的错误标记
#
#     except Exception as e:
#         logger.error(f"并行映射执行期间发生错误: {e}", exc_info=True)
#         raise # 可能需要更具体的异常处理
#
#     logger.info(f"并行处理完成")
#     # 检查结果中是否有未处理的异常
#     if any(isinstance(r, Exception) for r in results):
#          logger.error("并行任务中至少有一个失败")
#          # 可以选择在这里抛出聚合异常
#
#     return results

# --- 其他可能的工具 ---

def is_running_in_notebook() -> bool:
    """检查代码是否在 Jupyter Notebook 或类似环境中运行"""
    try:
        # 这个检查在标准 Python 解释器中会失败
        shell = pd.get_option('shell')
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def suppress_warnings(category: Warning = Warning) -> Callable[[Callable[..., T]], Callable[..., T]]:
     """装饰器抑制特定类别的警告"""
     def decorator(func: Callable[..., T]) -> Callable[..., T]:
         @functools.wraps(func)
         def wrapper(*args: Any, **kwargs: Any) -> T:
             with warnings.catch_warnings():
                 warnings.simplefilter("ignore", category=category)
                 return func(*args, **kwargs)
         return wrapper
     return decorator

def downcast_dataframe_dtypes(df: pd.DataFrame, target_numeric: str = 'integer', target_float: str = 'float', verbose: bool = False) -> pd.DataFrame:
    """尝试将 DataFrame 的数值类型向下转型以减少内存占用"""
    df_out = df.copy()
    if verbose: logger.info(f"开始向下转型 DataFrame 数据类型 (当前内存: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB)...")
    for col in df_out.select_dtypes(include=['number']).columns:
        original_dtype = str(df_out[col].dtype)
        if 'int' in original_dtype:
             df_out[col] = pd.to_numeric(df_out[col], downcast=target_numeric)
        elif 'float' in original_dtype:
             df_out[col] = pd.to_numeric(df_out[col], downcast=target_float)
        new_dtype = str(df_out[col].dtype)
        if verbose and original_dtype != new_dtype:
             logger.debug(f"列 '{col}' 类型从 {original_dtype} 降级为 {new_dtype}")

    if verbose: logger.info(f"数据类型向下转型完成 (最终内存: {df_out.memory_usage(deep=True).sum() / 1024**2:.2f} MB)")
    return df_out