# -*- coding: utf-8 -*-
"""
增量学习数据采样模块 (espml)
包含用于增量学习的数据选择策略,例如 iCaRL 样本集管理或时间窗口采样
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod
from pathlib import Path
from loguru import logger

# 导入项目级 utils (用于文件操作等)
from espml.util import utils as common_utils

# --- 采样策略基类  ---
class BaseSampler(ABC):
    """数据采样器抽象基类"""
    def __init__(self, config: Dict[str, Any], logger_instance: Any):
        self.config = config
        self.logger = logger_instance.bind(name=f"Sampler_{self.__class__.__name__}")
        # self.logger.info(f"初始化 {self.__class__.__name__}...")

    @abstractmethod
    def select_data(self,
                    available_data: pd.DataFrame,
                    current_model_metadata: Optional[Dict] = None,
                    previous_model_metadata: Optional[Dict] = None
                    ) -> pd.DataFrame:
        """选择用于当前增量更新或训练的数据"""
        raise NotImplementedError

    @abstractmethod
    def update_state(self,
                     new_data_processed: pd.DataFrame,
                     combined_training_data: pd.DataFrame,
                     new_model_metadata: Dict
                     ) -> Optional[Dict]:
        """更新采样器内部状态并返回需保存的元数据"""
        raise NotImplementedError

# --- 时间窗口采样器  ---
class WindowSampler(BaseSampler):
    """基于滑动时间窗口选择数据"""
    def __init__(self, config: Dict[str, Any], logger_instance: Any):
        super().__init__(config, logger_instance)
        # 假设参数在 config 中,键名为 'WindowSize'
        self.window_size = self.config.get('WindowSize', '90D')
        # 验证窗口大小格式
        try:
             pd.Timedelta(self.window_size)
        except ValueError:
             raise ValueError(f"无效的时间窗口配置 'WindowSize': {self.window_size}")
        self.logger.info(f"WindowSampler 初始化,窗口大小: {self.window_size}")

    def select_data(self, available_data: pd.DataFrame, current_model_metadata: Optional[Dict] = None, previous_model_metadata: Optional[Dict] = None) -> pd.DataFrame:
        """选择指定时间窗口内的数据"""
        self.logger.info(f"使用时间窗口 '{self.window_size}' 选择数据...")
        if available_data.empty:
             self.logger.warning("可用数据为空,无法进行窗口采样")
             return available_data
        if not isinstance(available_data.index, pd.DatetimeIndex):
             raise ValueError("WindowSampler 需要 DataFrame 具有 DatetimeIndex")

        end_time = available_data.index.max()
        start_time = end_time - pd.Timedelta(self.window_size)

        selected_mask = (available_data.index >= start_time) & (available_data.index <= end_time)
        selected_data = available_data.loc[selected_mask] # 使用 .loc 避免 SettingWithCopyWarning

        if selected_data.empty:
            self.logger.warning(f"时间窗口 [{start_time}, {end_time}] 内没有数据")
        else:
            self.logger.info(f"窗口采样完成,选中数据范围: [{selected_data.index.min()}, {selected_data.index.max()}], 行数: {len(selected_data)} / {len(available_data)}")
        return selected_data

    def update_state(self, new_data_processed: pd.DataFrame, combined_training_data: pd.DataFrame, new_model_metadata: Dict) -> Optional[Dict]:
        """WindowSampler 是无状态的"""
        # self.logger.trace("WindowSampler 无状态,无需更新")
        return None


# --- iCaRL 样本集采样器  ---
class ExemplarSampler(BaseSampler):
    """基于样本集 (Exemplar Set) 的采样器 (iCaRL 策略)"""
    def __init__(self, config: Dict[str, Any], logger_instance: Any):
        super().__init__(config, logger_instance)
        self.max_exemplar_set_size = int(self.config.get('MaxExemplarSetSize', 10000))
        if self.max_exemplar_set_size <= 0: self.max_exemplar_set_size = 0
        self.selection_strategy = self.config.get('ExemplarSelectionStrategy', 'random').lower()
        self.exemplar_set: Optional[pd.DataFrame] = None
        self.logger.info(f"ExemplarSampler 初始化,最大样本数: {self.max_exemplar_set_size}, 选择策略: {self.selection_strategy}")

    def _load_exemplar_set(self, metadata: Optional[Dict]) -> None:
        """(内部) 从元数据路径加载样本集"""
        self.exemplar_set = None
        if not metadata or 'exemplar_set_path' not in metadata or not metadata['exemplar_set_path']:
            self.logger.info("元数据中未找到有效的样本集路径")
            return
        path_str = metadata['exemplar_set_path']
        path = Path(path_str)
        self.logger.info(f"尝试从路径加载样本集: {path}")
        if common_utils.check_path_exists(path, path_type='f'):
             try:
                  self.exemplar_set = pd.read_feather(path)
                  # 恢复索引 
                  if 'index' in self.exemplar_set.columns:
                       # 尝试恢复为 DatetimeIndex
                       datetime_index = pd.to_datetime(self.exemplar_set['index'], errors='coerce')
                       if datetime_index.notna().all():
                           self.exemplar_set = self.exemplar_set.set_index(datetime_index).drop(columns=['index'])
                       else: # 如果不是日期时间,作为普通索引
                            self.exemplar_set = self.exemplar_set.set_index('index')
                       # 检查索引类型
                       if not isinstance(self.exemplar_set.index, pd.DatetimeIndex):
                           self.logger.warning(f"恢复的样本集索引类型不是 DatetimeIndex ({self.exemplar_set.index.dtype})")

                  self.logger.info(f"成功加载样本集,形状: {self.exemplar_set.shape}")
             except ImportError: self.logger.error("加载 Feather 失败需安装 'pyarrow'"); self.exemplar_set = None
             except Exception as e: self.logger.exception(f"加载样本集文件 '{path}' 失败: {e}"); self.exemplar_set = None
        else: self.logger.error(f"样本集文件不存在: {path}")

    def _save_exemplar_set(self, metadata: Dict) -> Optional[str]:
        """(内部) 保存样本集到文件"""
        if self.exemplar_set is None or self.exemplar_set.empty: return None
        version_id = metadata.get('version_id', 'unknown')
        model_save_path = metadata.get('model_path')
        if not model_save_path: self.logger.error("无法确定样本集保存目录 (缺少 'model_path')"); return None
        save_dir = Path(model_save_path).parent
        save_path = save_dir / f"exemplar_set_{version_id}.feather"
        self.logger.info(f"开始保存样本集到: {save_path} (形状: {self.exemplar_set.shape})")
        try:
            common_utils.mkdir_if_not_exist(save_dir)
            self.exemplar_set.reset_index().to_feather(save_path) # 保存索引
            self.logger.info("样本集保存成功")
            return str(save_path.as_posix())
        except ImportError: self.logger.error("保存 Feather 失败需安装 'pyarrow'"); return None
        except Exception as e: self.logger.exception(f"保存样本集失败: {e}"); return None

    def _reduce_exemplar_set(self, combined_pool: pd.DataFrame) -> pd.DataFrame:
        """(内部) 根据策略缩减样本集大小（Herding 仍简化）"""
        pool_size = len(combined_pool)
        if pool_size <= self.max_exemplar_set_size: return combined_pool.sort_index()

        self.logger.info(f"样本池大小 ({pool_size}) 超过限制 ({self.max_exemplar_set_size}),执行缩减 (策略: {self.selection_strategy})...")
        if self.selection_strategy == 'random':
            indices_to_keep = np.random.choice(combined_pool.index, self.max_exemplar_set_size, replace=False)
            reduced_set = combined_pool.loc[indices_to_keep].sort_index()
        elif self.selection_strategy == 'herding':
            self.logger.warning("iCaRL Herding 样本选择策略需要模型信息,当前版本使用随机采样代替!")
            indices_to_keep = np.random.choice(combined_pool.index, self.max_exemplar_set_size, replace=False)
            reduced_set = combined_pool.loc[indices_to_keep].sort_index()
        else: # 默认或未知策略
             self.logger.warning(f"不支持的样本选择策略: {self.selection_strategy},使用随机采样")
             indices_to_keep = np.random.choice(combined_pool.index, self.max_exemplar_set_size, replace=False)
             reduced_set = combined_pool.loc[indices_to_keep].sort_index()
        self.logger.info(f"样本集缩减完成,最终大小: {reduced_set.shape}")
        return reduced_set

    def select_data(self, available_data: pd.DataFrame, current_model_metadata: Optional[Dict] = None, previous_model_metadata: Optional[Dict] = None) -> pd.DataFrame:
        """选择新数据和旧样本集的组合"""
        self.logger.info("使用样本集策略选择数据...")
        if self.max_exemplar_set_size <= 0: # 如果禁用样本集
             self.logger.info("样本集已禁用,只使用新数据")
             return available_data.copy()

        # 1. 加载旧样本集
        self._load_exemplar_set(previous_model_metadata)

        # 2. 合并
        if self.exemplar_set is not None and not self.exemplar_set.empty:
             self.logger.info(f"合并新数据 ({available_data.shape}) 和加载的样本集 ({self.exemplar_set.shape})...")
             try:
                  # 优先保留 available_data 的列
                  common_cols = available_data.columns.intersection(self.exemplar_set.columns)
                  if len(common_cols) < len(self.exemplar_set.columns):
                       logger.warning(f"样本集中的部分列在新数据中不存在,将被丢弃: {set(self.exemplar_set.columns) - set(common_cols)}")
                  if len(common_cols) < len(available_data.columns):
                      logger.warning(f"新数据中的部分列在样本集中不存在: {set(available_data.columns) - set(common_cols)}")

                  exemplars_aligned = self.exemplar_set[common_cols]
                  available_data_aligned = available_data[common_cols]

                  # 合并,保留新数据处理重复索引
                  selected_data = pd.concat([exemplars_aligned, available_data_aligned], ignore_index=False)
                  selected_data = selected_data[~selected_data.index.duplicated(keep='last')]
                  selected_data = selected_data.sort_index()
                  self.logger.info(f"合并后数据大小: {selected_data.shape}")
             except Exception as e:
                  self.logger.error(f"合并新数据和样本集失败: {e}将只使用新数据", exc_info=True)
                  selected_data = available_data.copy()
        else:
             self.logger.info("样本集为空或加载失败,只使用新数据")
             selected_data = available_data.copy()

        return selected_data

    def update_state(self, new_data_processed: pd.DataFrame, combined_training_data: pd.DataFrame, new_model_metadata: Dict) -> Optional[Dict]:
        """更新样本集状态"""
        if self.max_exemplar_set_size <= 0: return None # 禁用则不更新

        self.logger.info("开始更新样本集状态...")
        # 使用当前加载的旧样本集 (`self.exemplar_set`) 和新数据 (`new_data_processed`)
        pool_for_selection: pd.DataFrame
        if self.exemplar_set is not None and not self.exemplar_set.empty:
             # 确保列对齐
             common_cols = self.exemplar_set.columns.intersection(new_data_processed.columns)
             pool_for_selection = pd.concat(
                 [self.exemplar_set[common_cols], new_data_processed[common_cols]],
                 ignore_index=False
             )
             pool_for_selection = pool_for_selection[~pool_for_selection.index.duplicated(keep='last')]
        else:
             pool_for_selection = new_data_processed.copy() # 如果没有旧样本,直接用新数据

        # 执行缩减
        self.exemplar_set = self._reduce_exemplar_set(pool_for_selection)

        # 保存并返回路径
        exemplar_path = self._save_exemplar_set(new_model_metadata)
        return {'exemplar_set_path': exemplar_path} if exemplar_path else None


# --- 工厂函数  ---
def get_sampler(config: Dict[str, Any], logger_instance: Any) -> BaseSampler:
    """根据配置获取相应的采样器实例"""
    # 路径调整采样配置在 IncrML 下的 DataSampling 键中
    incrml_config = config.get('IncrML', {})
    sampling_config = incrml_config.get('DataSampling', {})
    # 方法也在 IncrML 下
    incrml_method = incrml_config.get('Method', 'window').lower()

    logger_instance.info(f"根据配置获取数据采样器 (IncrML Method: '{incrml_method}')...")

    if incrml_method == 'icarl':
        logger_instance.info("检测到 iCaRL 方法,使用 ExemplarSampler")
        return ExemplarSampler(sampling_config, logger_instance)
    elif incrml_method == 'window':
        logger_instance.info("检测到 window 方法或默认,使用 WindowSampler")
        return WindowSampler(sampling_config, logger_instance)
    else:
        logger_instance.warning(f"未知的增量学习方法 '{incrml_method}',将使用默认的 WindowSampler")
        return WindowSampler({}, logger_instance) # 使用空配置


logger.info("增量学习数据采样模块 (espml.incrml.data_sampling) 加载完成")