# -*- coding: utf-8 -*-
"""
应用程序状态管理模块 (espml)
使用 JSON 文件持久化简单的键值对状态信息，包含文件锁确保并发安全
"""

import json
import os
import time
import errno # 用于检查文件锁错误
from pathlib import Path
from typing import Dict, Any, Optional, Union
from loguru import logger

# 尝试导入 fcntl (仅限 Unix-like 系统)
try:
    import fcntl
    FCNTL_AVAILABLE = True
except ImportError:
    logger.warning("模块 'fcntl' 不可用 (可能在非 Unix 系统上)，文件锁功能将无法使用并发写入状态文件可能导致问题")
    fcntl = None # type: ignore
    FCNTL_AVAILABLE = False

# 导入项目级 utils 和 const
from espml.util import utils as common_utils
from espml.util import const

# 默认状态文件名 
DEFAULT_STATE_FILE = const.PROJECT_ROOT / ".espml_state.json" # 放在项目根目录下的隐藏文件

# 文件锁相关常量 
LOCK_TIMEOUT = 10.0 # 秒
LOCK_RETRY_INTERVAL = 0.1 # 秒

class StateManagerError(Exception):
    """状态管理特定错误"""
    pass

class FileLock:
    """简单的文件锁上下文管理器 """
    def __init__(self, lock_file_path: Path, timeout: float = LOCK_TIMEOUT):
        if not FCNTL_AVAILABLE:
            # 如果 fcntl 不可用，文件锁无法工作
            self.logger = logger.bind(name="FileLock_Disabled")
            self.logger.warning("fcntl 不可用，文件锁已被禁用")
            self.lock_file_path = None # 标记为不可用
            self._lock_file_handle = None
            return

        self.logger = logger.bind(name="FileLock")
        self.lock_file_path = lock_file_path
        self.timeout = timeout
        self._lock_file_handle: Optional[int] = None # 文件描述符是整数

    def __enter__(self) -> 'FileLock':
        if not self.lock_file_path: return self # 如果锁不可用，直接返回

        start_time = time.monotonic()
        while True:
            try:
                # 尝试创建并打开锁文件
                # 使用 os.open 获取文件描述符
                # O_EXCL: 如果文件已存在则失败
                self._lock_file_handle = os.open(self.lock_file_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                # 尝试获取独占锁 (非阻塞)
                fcntl.lockf(self._lock_file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.logger.trace(f"成功获取文件锁: {self.lock_file_path}")
                return self
            except OSError as e:
                 # 如果错误是 EEXIST (文件已存在) 或 EAGAIN (锁不可用)
                 if e.errno in (errno.EEXIST, errno.EAGAIN, errno.EACCES): # 添加 EACCES
                     if (time.monotonic() - start_time) >= self.timeout:
                         self.logger.error(f"获取文件锁超时 ({self.timeout}s): {self.lock_file_path}")
                         # 关闭可能已打开但未锁定的句柄 (如果 O_EXCL 失败但文件句柄仍被创建)
                         if self._lock_file_handle is not None:
                             os.close(self._lock_file_handle)
                             self._lock_file_handle = None
                         raise TimeoutError(f"无法在 {self.timeout} 秒内获取文件锁: {self.lock_file_path}")
                     # 等待后重试
                     # self.logger.trace(f"等待文件锁: {self.lock_file_path}")
                     time.sleep(LOCK_RETRY_INTERVAL)
                 else: # 其他 OSError
                      # 关闭可能已打开的句柄
                      if self._lock_file_handle is not None: os.close(self._lock_file_handle); self._lock_file_handle = None
                      self.logger.exception(f"获取文件锁时发生 OSError: {e}")
                      raise StateManagerError(f"获取文件锁时发生 OSError") from e
            except Exception as e:
                 # 关闭可能已打开的句柄
                 if self._lock_file_handle is not None: os.close(self._lock_file_handle); self._lock_file_handle = None
                 self.logger.exception(f"获取文件锁时发生未知错误: {e}")
                 raise StateManagerError(f"获取文件锁未知错误") from e

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self.lock_file_path or self._lock_file_handle is None: return # 锁未启用或未成功获取

        try:
            # 释放锁并关闭文件句柄
            fcntl.lockf(self._lock_file_handle, fcntl.LOCK_UN)
            os.close(self._lock_file_handle)
            # logger.trace(f"文件锁已释放: {self.lock_file_path}")
            # 删除锁文件
            try:
                self.lock_file_path.unlink(missing_ok=True)
            except OSError: pass # 忽略删除错误
        except Exception as e:
             # 即使释放失败，也要记录错误
             self.logger.error(f"释放文件锁或删除锁文件时出错: {e}")
        finally:
            self._lock_file_handle = None # 确保句柄被清理

def load_state(file_path: Union[str, Path] = DEFAULT_STATE_FILE) -> Dict[str, Any]:
    """
    从 JSON 文件加载应用程序状态
    """
    state_path = Path(file_path)
    lock_path = state_path.with_suffix(state_path.suffix + '.lock')
    state_data: Dict[str, Any] = {}
    logger.debug(f"尝试加载状态文件: {state_path}")

    try:
        # 使用文件锁确保读取一致性
        with FileLock(lock_path):
            # logger.trace(f"获取状态文件锁成功 (读取): {state_path}")
            if common_utils.check_path_exists(state_path, path_type='f'):
                loaded_data = common_utils.read_json_file(state_path)
                if isinstance(loaded_data, dict):
                    state_data = loaded_data
                    logger.debug(f"成功加载状态 ({len(state_data)} 条): {state_path}")
                elif loaded_data is not None:
                     logger.warning(f"状态文件内容无效 (非字典): {state_path}返回空状态")
                # else: logger.warning("状态文件为空或解析失败") # read_json_file 已记录
            # else: logger.debug(f"状态文件不存在: {state_path}返回空状态")
    except TimeoutError:
         logger.error(f"加载状态时获取文件锁超时: {state_path}返回空状态")
    except StateManagerError as file_lock_e: # 捕获文件锁内部错误
         logger.error(f"加载状态时文件锁操作失败: {file_lock_e}")
    except Exception as e:
         # 捕获 read_json_file 可能抛出的其他异常
         logger.exception(f"加载状态文件时发生未知错误: {state_path} - {e}")

    return state_data

def save_state(state_dict: Dict[str, Any], file_path: Union[str, Path] = DEFAULT_STATE_FILE) -> bool:
    """
    将应用程序状态字典保存到 JSON 文件
    """
    if not isinstance(state_dict, dict):
         logger.error("保存状态失败输入不是有效的字典")
         return False

    state_path = Path(file_path)
    lock_path = state_path.with_suffix(state_path.suffix + '.lock')
    logger.debug(f"尝试保存状态文件 ({len(state_dict)} 条): {state_path}")

    try:
        # 使用文件锁确保写入原子性
        with FileLock(lock_path):
             # logger.trace(f"获取状态文件锁成功 (写入): {state_path}")
             # 使用 common_utils 写入 JSON
             if common_utils.write_json_file(state_dict, state_path, indent=4): # 假设缩进为 4
                  logger.debug(f"成功保存状态到: {state_path}")
                  return True
             else:
                  # write_json_file 内部已记录错误
                  logger.error(f"保存状态失败 (写入函数返回 False): {state_path}")
                  return False
    except TimeoutError:
         logger.error(f"保存状态时获取文件锁超时: {state_path}")
         return False
    except StateManagerError as file_lock_e:
         logger.error(f"保存状态时文件锁操作失败: {file_lock_e}")
         return False
    except Exception as e:
         logger.exception(f"保存状态文件时发生未知错误: {state_path} - {e}")
         return False

def get_state_value(key: str, default: Any = None, file_path: Union[str, Path] = DEFAULT_STATE_FILE) -> Any:
    """获取指定键的状态值"""
    # logger.trace(f"获取状态值: Key='{key}', Path='{file_path}'")
    state = load_state(file_path)
    return state.get(key, default)

def set_state_value(key: str, value: Any, file_path: Union[str, Path] = DEFAULT_STATE_FILE) -> bool:
    """设置指定键的状态值并保存"""
    # logger.trace(f"设置状态值: Key='{key}', Value='{value}', Path='{file_path}'")
    # 直接调用 load 和 save，锁在内部处理
    current_state = load_state(file_path)
    current_state[key] = value
    return save_state(current_state, file_path)

logger.info("状态管理模块 (espml.util.state) 加载完成")
# --- 结束 espml/util/state.py ---