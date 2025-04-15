# -*- coding: utf-8 -*-
"""
增量学习管理器模块 (espml)
负责协调元数据、数据采样、漂移检测，并根据触发条件执行增量更新流程
"""

import datetime
import os
from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

# 导入 IncrML 内部模块
from espml.incrml.metadata import IncrmlMetadata, ModelVersionInfo
from espml.incrml.data_sampling import BaseSampler, get_sampler
from espml.incrml.detect_drift import BaseDriftDetector, get_drift_detector
# 导入 ML 核心流程 (用于类型提示和调用)
# 使用 try-except 避免循环导入（如果 ml.py 也导入 manager.py）
try:
    from espml.ml import MLPipeline
except ImportError:
    MLPipeline = Any # type: ignore # 允许 Any 作为后备

# 导入项目级 utils 和 const
from espml.util import utils as common_utils
from espml.util import const

# 尝试导入 croniter
try:
    from croniter import croniter
    CRONITER_INSTALLED = True
except ImportError:
    logger.warning("库 'croniter' 未安装，基于 Cron 的增量学习触发器将不可用")
    croniter = None # type: ignore
    CRONITER_INSTALLED = False


class IncrmlManager:
    """
    增量学习管理器
    协调整个增量学习流程代码逻辑
    """
    def __init__(self, task_id: str, config: Dict[str, Any], logger_instance: Any):
        """
        初始化 IncrmlManager

        Args:
            task_id (str): 当前任务的 ID
            config (Dict[str, Any]): 完整的项目配置字典
            logger_instance (logger): loguru logger 实例

        Raises:
            ValueError: 如果 IncrML 配置无效
            KeyError: 如果缺少必要的配置项
        """
        self.task_id = task_id
        self.config = config
        self.logger = logger_instance.bind(name=f"IncrmlManager_{task_id}")
        self.logger.info(f"初始化 IncrmlManager for task '{task_id}'...")

        # 解析 IncrML 配置 
        self.incrml_config = self.config.get('IncrML', {})
        self.enabled = bool(self.incrml_config.get('Enabled', False))
        self.method = self.incrml_config.get('Method', 'window').lower()
        self.trigger = self.incrml_config.get('Trigger', 'OnDataFileIncrease')
        self.model_base_path = self.incrml_config.get('SaveModelPath')
        if not self.model_base_path:
            raise ValueError("IncrML 配置中缺少 'SaveModelPath'")
        # 确保路径是绝对或相对于项目根目录 (如果代码处理了相对路径)
        self.model_base_path = Path(self.model_base_path)
        if not self.model_base_path.is_absolute():
             self.model_base_path = const.PROJECT_ROOT / self.model_base_path
        self.metadata_dir = self.model_base_path / "metadata" # 元数据子目录

        # --- 初始化组件  ---
        # 1. 元数据管理器
        try:
            self.metadata = IncrmlMetadata(task_id=self.task_id, metadata_dir=self.metadata_dir)
        except Exception as e:
             self.logger.exception("初始化 IncrmlMetadata 失败！")
             raise RuntimeError("无法初始化元数据管理器") from e

        # 2. 数据采样器
        try:
            # 将整个 IncrML 配置传给 get_sampler
            self.sampler: Optional[BaseSampler] = get_sampler(self.incrml_config, self.logger)
            if self.sampler: self.logger.info(f"数据采样器已初始化: {type(self.sampler).__name__}")
            else: self.logger.warning("无法根据配置初始化数据采样器")
        except Exception as e:
             self.logger.exception("初始化数据采样器失败！")
             self.sampler = None # 初始化失败

        # 3. 漂移检测器 (仅当需要时初始化)
        self.drift_detector: Optional[BaseDriftDetector] = None
        if self.trigger == 'OnDriftDetected':
            drift_detection_config = self.incrml_config.get('DriftDetection', {})
            if drift_detection_config.get('Enabled', False):
                 try:
                     # 将整个项目配置传递给 get_drift_detector，因为它可能需要 IncrML 配置
                     self.drift_detector = get_drift_detector(self.config, self.logger)
                     if self.drift_detector: self.logger.info("漂移检测器已初始化")
                     else: self.logger.error("根据配置未能初始化漂移检测器！")
                 except Exception as e:
                      self.logger.exception("初始化漂移检测器失败！")
                      self.drift_detector = None
            else:
                 self.logger.warning("触发器配置为 'OnDriftDetected' 但漂移检测未启用")

        self.logger.info(f"IncrmlManager 初始化完成Enabled={self.enabled}, Method='{self.method}', Trigger='{self.trigger}'")

    def check_trigger(self,
                      latest_data_timestamp: Optional[pd.Timestamp] = None,
                      current_predictions: Optional[pd.Series] = None,
                      current_ground_truth: Optional[pd.Series] = None
                      ) -> bool:
        """
        检查是否满足增量学习的触发条件

        Args:
            latest_data_timestamp: 当前可用数据的最新时间戳 (用于 OnDataFileIncrease)
            current_predictions: 模型在最新数据上的预测结果 (用于 OnDriftDetected)
            current_ground_truth: 最新数据对应的真实值 (用于 OnDriftDetected)

        Returns:
            bool: 是否需要执行增量更新
        """
        if not self.enabled: return False # 未启用则不触发

        self.logger.debug(f"检查增量学习触发器: {self.trigger}...")
        triggered = False
        current_version_info = self.metadata.get_current_version()

        # --- OnDataFileIncrease 逻辑  ---
        if self.trigger == 'OnDataFileIncrease':
            if latest_data_timestamp is None:
                 self.logger.warning("'OnDataFileIncrease' 触发器需要 'latest_data_timestamp'")
                 return False # 无法判断
            if current_version_info and current_version_info.training_data_end:
                try:
                    last_train_end = pd.to_datetime(current_version_info.training_data_end)
                    # 严格比较新数据时间戳 > 上次训练结束时间戳
                    if latest_data_timestamp > last_train_end:
                        self.logger.info(f"触发 (OnDataFileIncrease): 最新数据 ({latest_data_timestamp}) > 上次训练 ({last_train_end})")
                        triggered = True
                    # else: self.logger.trace("无新数据")
                except Exception as e:
                     self.logger.error(f"解析上次训练时间戳失败: {e}假定需要更新")
                     triggered = True # 时间戳无效，触发更新
            else: # 没有历史记录，触发
                 self.logger.info("触发 (OnDataFileIncrease): 无历史记录，执行首次训练/更新")
                 triggered = True

        # --- Scheduled 逻辑  ---
        elif self.trigger == 'Scheduled':
            schedule_cron = self.incrml_config.get('ScheduleCron')
            if not schedule_cron:
                 self.logger.error("触发器 'Scheduled' 缺少 'ScheduleCron' 配置")
                 return False
            if not CRONITER_INSTALLED:
                 self.logger.error("'Scheduled' 触发器需要 'croniter' 库")
                 return False
            try:
                 # 需要上次运行时间来判断 Cron 是否到期
                 # 假设上次运行时间存储在元数据中或外部传入
                 last_run_time_str = current_version_info.timestamp if current_version_info else None
                 # 简化如果当前时间 >= Cron 表达式的下一个时间点（基于某个基准，如上次运行）
                 # 这种判断在 manager 内部不完美，最好由外部调度器完成
                 # 此处仅作演示，实际应依赖外部调度
                 # base_time = pd.to_datetime(last_run_time_str) if last_run_time_str else datetime.datetime.now() - datetime.timedelta(days=1) # 假设基准是昨天
                 # cron = croniter(schedule_cron, start_time=base_time)
                 # next_run = cron.get_next(datetime.datetime)
                 # if datetime.datetime.now() >= next_run: triggered = True
                 self.logger.warning("内部 Cron 触发检查逻辑不精确，依赖外部调度调用 update")
                 triggered = False # 假设由外部决定是否调用 update

            except Exception as e:
                 self.logger.error(f"评估 Cron 表达式 '{schedule_cron}' 失败: {e}")
                 triggered = False

        # --- OnDriftDetected 逻辑  ---
        elif self.trigger == 'OnDriftDetected':
            if self.drift_detector is None:
                 self.logger.warning("触发器 'OnDriftDetected' 但检测器未初始化")
                 return False
            if current_predictions is None or current_ground_truth is None:
                 self.logger.warning("'OnDriftDetected' 触发器需要当前预测和真实值")
                 return False
            if len(current_predictions) != len(current_ground_truth):
                 self.logger.error("漂移检测输入长度不匹配")
                 return False
            if len(current_predictions) == 0: return False # 无数据

            self.logger.debug(f"向漂移检测器添加 {len(current_predictions)} 个新观测结果...")
            try:
                # 假设需要计算预测是否正确 (需要知道任务类型和阈值)
                # : 此计算逻辑应该与代码一致
                errors = 0
                # 此处简化为直接比较（假设分类），实际应更复杂
                # 需要将 _calculate_metric 移到可调用的地方，或者在此处重新实现
                # 假设任务是回归，使用简单阈值判断
                if self.task_type == 'regression':
                    error_threshold = self.incrml_config.get('DriftDetection', {}).get('ErrorThreshold', 0.1)
                    absolute_error = np.abs(current_predictions - current_ground_truth)
                    relative_error = absolute_error / np.maximum(np.abs(current_ground_truth), 1e-6)
                    correct_predictions = relative_error <= error_threshold
                else: # classification
                    correct_predictions = (current_predictions == current_ground_truth)

                # 逐个添加
                for is_correct in correct_predictions:
                    self.drift_detector.add_element(is_correct)
                    if self.drift_detector.detected_change(): # 检查是否达到漂移状态
                        self.logger.info(f"触发 (OnDriftDetected): 检测到概念漂移！")
                        triggered = True
                        break # 已触发
                # 即使没触发漂移，也记录一下警告状态
                if not triggered and self.drift_detector.detected_warning_zone():
                     self.logger.info("漂移检测器处于警告状态")

            except Exception as e:
                 self.logger.exception(f"向漂移检测器添加元素或检查漂移时出错: {e}")
                 triggered = False # 出错时不触发

        else: # 未知触发器类型
            self.logger.error(f"未知的增量学习触发器类型: {self.trigger}")

        # logger.debug(f"触发器检查结果: {triggered}")
        return triggered

    def prepare_data(self, available_df: pd.DataFrame) -> pd.DataFrame:
        """
        准备用于增量更新的训练数据

        Args:
            available_df (pd.DataFrame): 当前所有可用的（已处理的）数据

        Returns:
            pd.DataFrame: 根据采样策略选择的数据子集
        """
        self.logger.info("开始准备增量学习数据...")
        if self.sampler is None:
            self.logger.warning("数据采样器未初始化，将使用所有可用数据")
            return available_df.copy()

        # 获取上一个版本的元数据字典
        previous_version_info = self.metadata.get_current_version()
        prev_meta_dict = previous_version_info.to_dict() if previous_version_info else None
        # 传递目标名称给 sampler (如果需要)
        if prev_meta_dict and self.target_name:
             prev_meta_dict['target_name'] = self.target_name

        self.logger.debug("调用采样器的 select_data 方法...")
        try:
            selected_data = self.sampler.select_data(
                available_data=available_df,
                previous_model_metadata=prev_meta_dict
            )
            if selected_data.empty:
                 self.logger.warning("采样器返回了空的数据集！")
            else:
                 self.logger.info(f"数据准备完成，选中数据形状: {selected_data.shape}")
            return selected_data
        except Exception as e:
             self.logger.exception("数据采样失败！将返回所有可用数据作为后备")
             return available_df.copy()

    # 使用计时器
    @common_utils.log_execution_time(level="INFO")
    def update(self,
               ml_pipeline: MLPipeline, # 明确接收 MLPipeline 实例
               incrml_train_df: pd.DataFrame, # 接收采样后的数据
               new_data_processed: pd.DataFrame # 本次使用的新数据
              ) -> bool:
        """
        执行一次增量模型更新（训练）

        Args:
            ml_pipeline (MLPipeline): 已初始化的 MLPipeline 实例
            incrml_train_df (pd.DataFrame): 由 prepare_data 选择出的用于本次更新的数据
            new_data_processed (pd.DataFrame): 本次增量中包含的新数据部分（用于 sampler 更新）

        Returns:
            bool: 增量更新是否成功
        """
        if not self.enabled: self.logger.error("尝试执行 update 但增量学习未启用！"); return False
        if not isinstance(ml_pipeline, MLPipeline): raise TypeError("'ml_pipeline' 必须是 MLPipeline 实例")
        if incrml_train_df.empty: self.logger.error("用于增量更新的数据为空！"); return False

        self.logger.info(f"开始执行增量模型更新 (任务: {self.task_id}, 方法: {self.method})...")
        self.logger.info(f"用于本次更新的数据形状: {incrml_train_df.shape}")

        # 1. 生成新的版本/运行 ID
        # 使用时间戳作为 ID
        new_version_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S%f')
        self.logger.info(f"新模型版本 ID: {new_version_id}")

        # 2. 执行训练 (调用 MLPipeline.train)
        # 直接调用 train，不修改配置（除非代码明确修改了）
        # iCaRL 的特殊逻辑（如蒸馏）需要 MLPipeline 或其内部组件支持
        if self.method == 'icarl':
             self.logger.warning("检测到 iCaRL 方法，但当前 update 实现未包含特定的 iCaRL 逻辑（如知识蒸馏），"
                               "假设此逻辑在 MLPipeline.train 或相关组件内部处理")
             # 可能需要传递旧模型信息给 train
             # old_model_info = self.metadata.get_current_version()
             # training_successful = ml_pipeline.train(..., incremental_params={'method':'icarl', 'old_model':old_model_info})

        training_successful = ml_pipeline.train(
            df_train_full=incrml_train_df, # 使用采样/准备好的数据
            run_id=new_version_id # 传递新 ID 用于保存
        )

        if not training_successful:
            self.logger.error("MLPipeline.train 执行失败，增量更新终止")
            return False

        # 3. 获取训练结果和路径 (严格依赖 ml_pipeline 的实现)
        self.logger.debug("尝试从 MLPipeline 获取训练结果...")
        # 假设 ml_pipeline 暴露了获取最新运行结果的方法或属性
        # 这是目前设计的一个潜在弱点，ml_pipeline 需要支持返回这些信息
        try:
            # 使用 ml_pipeline 内部方法获取路径 (更健壮)
            model_path, tf_path, feat_path, _ = ml_pipeline._get_run_specific_paths(new_version_id)
            # 从 automl_wrapper 获取性能
            performance = {}
            if hasattr(ml_pipeline, 'automl_wrapper') and ml_pipeline.automl_wrapper:
                if ml_pipeline.automl_wrapper.final_val_score is not None:
                    metric_name = ml_pipeline.automl_wrapper.metric
                    performance[metric_name] = ml_pipeline.automl_wrapper.final_val_score
                # 可以尝试获取更多指标
                if ml_pipeline.automl_wrapper.best_loss is not None:
                     performance['flaml_best_loss'] = ml_pipeline.automl_wrapper.best_loss
            else:
                 self.logger.warning("无法从 MLPipeline 获取 AutoML wrapper 或其性能分数")
            # 获取 AutoFE 选择的特征列表 (假设 ml_pipeline 保存了这个结果)
            # final_selected_features = ml_pipeline.last_run_selected_autofe_features
            # 或者从文件读取
            features_info = common_utils.read_json_file(feat_path)
            final_selected_features = features_info.get('selected_features', []) if features_info else []

            self.logger.info(f"获取训练结果成功模型: {model_path}, 性能: {performance}, 新特征数: {len(final_selected_features)}")

        except AttributeError:
            self.logger.error("无法从 MLPipeline 实例获取必要的训练结果（路径或性能），元数据将不完整！")
            # 根据代码决定是继续还是失败
            return False # 假设无法获取结果则失败
        except Exception as e:
             self.logger.exception(f"获取训练结果时发生未知错误: {e}")
             return False

        # 4. 创建新的元数据版本信息
        data_start = incrml_train_df.index.min().isoformat() if isinstance(incrml_train_df.index, pd.DatetimeIndex) else None
        data_end = incrml_train_df.index.max().isoformat() if isinstance(incrml_train_df.index, pd.DatetimeIndex) else None
        previous_version = self.metadata.get_current_version()
        new_version_info = ModelVersionInfo(
            version_id=new_version_id,
            # 保存相对路径（如果可能）
            model_path=str(Path(model_path).relative_to(const.PROJECT_ROOT).as_posix()) if Path(model_path).is_relative_to(const.PROJECT_ROOT) else model_path,
            transformer_state_path=str(Path(tf_path).relative_to(const.PROJECT_ROOT).as_posix()) if Path(tf_path).is_relative_to(const.PROJECT_ROOT) else tf_path,
            selected_features_path=str(Path(feat_path).relative_to(const.PROJECT_ROOT).as_posix()) if Path(feat_path).is_relative_to(const.PROJECT_ROOT) else feat_path,
            training_data_start=data_start,
            training_data_end=data_end,
            performance_metrics=performance,
            drift_status=self.drift_detector.detected_change() if self.drift_detector else False,
            base_model_version=previous_version.version_id if previous_version else None,
            misc_info={'selected_autofe_features_count': len(final_selected_features)} # 示例额外信息
        )

        # 5. 更新采样器状态 (例如样本集)
        if self.sampler:
             self.logger.info("更新数据采样器状态...")
             try:
                 sampler_state_update = self.sampler.update_state(
                     new_data_processed=new_data_processed, # 传递本次的新数据
                     combined_training_data=incrml_train_df, # 传递实际训练数据
                     new_model_metadata=new_version_info.to_dict() # 传递新元数据
                 )
                 if sampler_state_update:
                      self.logger.info(f"采样器返回状态更新: {sampler_state_update}")
                      # 更新到新版本的元数据中 
                      if 'exemplar_set_path' in sampler_state_update:
                           new_version_info.exemplar_set_path = sampler_state_update['exemplar_set_path']
                      # 可以将其他信息放入 misc_info
                      new_version_info.misc_info.update({k:v for k,v in sampler_state_update.items() if k != 'exemplar_set_path'})
             except Exception as sampler_e:
                  self.logger.exception(f"更新采样器状态时出错: {sampler_e}")
                  # 不应因此失败整个更新，但需记录
        else: self.logger.debug("采样器未初始化，跳过状态更新")

        # 6. 添加新版本元数据并保存
        try:
            self.metadata.add_version(new_version_info, set_as_current=True) # 添加并设为当前
            save_ok = self.metadata.save()
            if not save_ok: raise RuntimeError("元数据保存失败")
            self.logger.success(f"增量模型更新成功完成！新版本 ID: {new_version_id}")
        except Exception as meta_e:
             self.logger.exception(f"添加或保存新版本元数据时失败: {meta_e}")
             return False # 元数据失败是严重问题

        # 7. 重置漂移检测器 (假设漂移后需要重置)
        if self.drift_detector and self.drift_detector.detected_change():
             self.logger.info("检测到漂移，正在重置漂移检测器状态...")
             try:
                 self.drift_detector._reset()
             except Exception as reset_e:
                 self.logger.error(f"重置漂移检测器失败: {reset_e}")

        return True