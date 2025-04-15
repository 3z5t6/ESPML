# -*- coding: utf-8 -*-
"""
概念漂移检测模块 (espml)
包含概念漂移检测算法的实现，例如 DDM
"""

import math
import numpy as np
from typing import Optional, Literal, Dict, Any
from abc import ABC, abstractmethod # 导入 ABC 用于基类
from loguru import logger

# --- 漂移检测器基类 (如果代码有) ---
# 如果代码没有基类，可以直接移除 BaseDriftDetector
class BaseDriftDetector(ABC):
    """漂移检测器抽象基类"""
    def __init__(self, config: Dict[str, Any]):
        """
        初始化基类

        Args:
            config (Dict[str, Any]): 检测器相关的配置
        """
        self.config = config
        # 创建子 logger
        self.logger = logger.bind(name=f"DriftDetector_{self.__class__.__name__}")
        self.drift_state: Optional[Literal['drift', 'warning']] = None # 存储当前状态
        # self.logger.info(f"初始化 {self.__class__.__name__}...")
        self._reset() # 调用重置方法

    @abstractmethod
    def _reset(self) -> None:
        """(抽象方法) 重置检测器的内部状态"""
        self.drift_state = None
        # logger.trace("BaseDriftDetector state reset.")

    @abstractmethod
    def add_element(self, prediction_is_correct: bool) -> None:
        """
        (抽象方法) 添加新的观测结果到检测器
        """
        raise NotImplementedError

    def detected_warning_zone(self) -> bool:
        """返回当前是否处于警告区域"""
        return self.drift_state == 'warning'

    def detected_change(self) -> bool:
        """返回当前是否检测到漂移"""
        return self.drift_state == 'drift'

    def get_drift_state(self) -> Optional[Literal['drift', 'warning']]:
        """获取当前的漂移状态"""
        return self.drift_state

# --- DDM 实现  ---
class DriftDetectorDDM(BaseDriftDetector):
    """
    Drift Detection Method (DDM) 实现
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 DDM 检测器

        Args:
            config (Dict[str, Any]): 包含 DDM 参数的配置字典预期键
                'MinNumInstances' (int): 最小样本数 (默认 30)
                'WarningLevelFactor' (float): 警告因子 (默认 2.0)
                'DriftLevelFactor' (float): 漂移因子 (默认 3.0)
        Raises:
            ValueError: 如果参数无效
        """
        # 先调用父类初始化 (设置 config, logger, drift_state=None, 调用 _reset)
        super().__init__(config)

        # 解析特定参数 (带类型转换和默认值)
        try:
            self.min_instance_num = int(self.config.get('MinNumInstances', 30))
            self.warning_level_factor = float(self.config.get('WarningLevelFactor', 2.0))
            self.drift_level_factor = float(self.config.get('DriftLevelFactor', 3.0))
        except (ValueError, TypeError) as e:
            self.logger.error(f"DDM 初始化参数类型错误: {e}")
            raise ValueError("DDM 初始化参数类型错误") from e

        # 参数有效性检查
        if self.min_instance_num <= 0: raise ValueError("MinNumInstances 必须 > 0")
        if self.warning_level_factor <= 0: raise ValueError("WarningLevelFactor 必须 > 0")
        if self.drift_level_factor <= 0: raise ValueError("DriftLevelFactor 必须 > 0")
        if self.warning_level_factor >= self.drift_level_factor:
            raise ValueError("DriftLevelFactor 必须大于 WarningLevelFactor")

        # 状态变量在 _reset 中初始化
        self.logger.info("DriftDetectorDDM 初始化完成")
        self.logger.debug(f"DDM 参数: min_instances={self.min_instance_num}, "
                          f"warning_factor={self.warning_level_factor}, drift_factor={self.drift_level_factor}")

    def _reset(self) -> None:
        """重置 DDM 状态变量"""
        super()._reset() # 调用父类重置 drift_state
        self.n: int = 0       # 样本计数器 (严格按 DDM 论文，从 0 开始，在 add_element 中 +1)
        self.p: float = 1.0   # 当前错误率估计值
        self.s: float = 0.0   # 当前错误率标准差估计值
        self.p_min: float = 1.0 # 观测到的最低错误率
        self.s_min: float = 0.0 # 最低错误率时的标准差
        # self.logger.trace("DDM 内部状态已重置")

    def add_element(self, prediction_is_correct: bool) -> None:
        """
        添加新的预测结果（True 正确, False 错误）并更新 DDM 状态
         DDM 算法流程
        """
        # 如果已检测到漂移，不再更新状态，需要外部 reset
        if self.drift_state == 'drift':
             # self.logger.trace("DDM 已检测到漂移，跳过状态更新")
             return

        # 1. 更新样本计数
        self.n += 1

        # 2. 更新错误率和标准差估计
        error_occurred = 1.0 if not prediction_is_correct else 0.0
        # 增量更新公式: p_new = p_old + (error - p_old) / n
        p_old = self.p
        self.p = p_old + (error_occurred - p_old) / float(self.n) # 确保是浮点数除法
        # 标准差公式: s = sqrt(p * (1-p) / n)
        # 确保 p 在 [0, 1] 范围内以计算标准差
        p_clipped = np.clip(self.p, 0.0, 1.0)
        # 确保 n > 0
        self.s = math.sqrt(p_clipped * (1.0 - p_clipped) / self.n) if self.n > 0 else 0.0

        # logger.trace(f"DDM Update: n={self.n}, error={error_occurred}, p={self.p:.6f}, s={self.s:.6f}")

        # 3. 检查是否更新 p_min 和 s_min
        # 仅在达到最小样本数后才开始更新和检测
        if self.n < self.min_instance_num:
            # logger.trace(f"样本数 ({self.n}) 未达到阈值 ({self.min_instance_num})，不更新 min 或检测漂移")
            return # 不更新也不检测

        current_p_s = self.p + self.s
        min_p_s = self.p_min + self.s_min

        # 使用一个小的 epsilon 防止浮点数比较问题
        epsilon = 1e-9
        if current_p_s < min_p_s - epsilon:
            self.p_min = self.p
            self.s_min = self.s
            # logger.trace(f"更新 p_min={self.p_min:.6f}, s_min={self.s_min:.6f} (p+s={current_p_s:.6f} < min_p+s={min_p_s:.6f})")
            # 当状态改善时，如果之前是警告状态，应重置
            if self.drift_state == 'warning':
                 self.logger.info("DDM 状态改善，离开警告区域")
                 self.drift_state = None
        # else: logger.trace(f"未更新 p_min/s_min (p+s={current_p_s:.6f} >= min_p+s={min_p_s:.6f})")

        # 4. 检查漂移和警告阈值
        # 确保 s_min > 0 以进行有效比较
        if self.s_min <= 0:
             # logger.trace("s_min 为 0 或负数，无法进行漂移检测")
             return

        drift_threshold = self.p_min + self.drift_level_factor * self.s_min
        warning_threshold = self.p_min + self.warning_level_factor * self.s_min

        # logger.trace(f"检查漂移: p+s={current_p_s:.6f}, 警告阈值={warning_threshold:.6f}, 漂移阈值={drift_threshold:.6f}")

        # 状态转换逻辑 
        if current_p_s >= drift_threshold:
            if self.drift_state != 'drift': # 仅在首次检测到时记录
                 self.logger.warning(f"DDM 检测到概念漂移！(p+s={current_p_s:.6f} >= drift_threshold={drift_threshold:.6f}) at n={self.n}")
            self.drift_state = 'drift'
            # 注意检测到漂移后，DDM 通常需要重置才能检测下一次漂移
            # 但这里只设置状态，重置由外部管理器决定
        elif current_p_s >= warning_threshold:
             if self.drift_state is None: # 仅在从稳定状态进入时记录
                  self.logger.warning(f"DDM 进入警告区域！(p+s={current_p_s:.6f} >= warning_threshold={warning_threshold:.6f}) at n={self.n}")
             # 即使之前是漂移状态，现在也降级为警告 (如果 p+s 下降了)
             # 或者如果之前是警告，则保持警告
             self.drift_state = 'warning'
        else: # 低于警告阈值
             if self.drift_state is not None: # 如果之前是警告或漂移
                  self.logger.info(f"DDM 恢复到稳定状态(p+s={current_p_s:.6f} < warning_threshold={warning_threshold:.6f}) at n={self.n}")
             self.drift_state = None # 恢复稳定


# --- 工厂函数  ---
def get_drift_detector(config: Dict[str, Any], logger_instance: Any) -> Optional[BaseDriftDetector]:
    """根据配置获取漂移检测器实例"""
    # 配置路径: config['IncrML']['DriftDetection']
    incrml_config = config.get('IncrML', {})
    drift_config = incrml_config.get('DriftDetection', {})
    drift_enabled = drift_config.get('Enabled', False)
    # 默认方法为 DDM 
    drift_method = drift_config.get('Method', 'DDM').upper()

    if not drift_enabled:
        logger_instance.info("概念漂移检测在配置中未启用")
        return None

    logger_instance.info(f"根据配置获取漂移检测器 (方法: {drift_method})...")
    if drift_method == 'DDM':
        # 将 DriftDetection 下的参数直接传递给 DDM
        # DriftDetectorDDM 内部会处理默认值
        try:
            return DriftDetectorDDM(drift_config, logger_instance)
        except ValueError as e:
             logger_instance.error(f"初始化 DriftDetectorDDM 失败: {e}将禁用漂移检测")
             return None
    # elif drift_method == 'EDDM':
        # logger_instance.info("尝试初始化 EDDM 检测器...")
        # return DriftDetectorEDDM(drift_config, logger_instance) # 需要实现 EDDM 类
    # elif drift_method == 'ADWIN':
        # logger_instance.info("尝试初始化 ADWIN 检测器...")
        # return DriftDetectorADWIN(drift_config, logger_instance) # 需要实现 ADWIN 类
    else:
        logger_instance.error(f"不支持的概念漂移检测方法: {drift_method}将禁用漂移检测")
        return None


logger.info("概念漂移检测模块 (espml.incrml.detect_drift) 加载完成")