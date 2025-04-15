# -*- coding: utf-8 -*-
"""
AutoML 模块封装 (espml)
主要负责与 AutoML 库 (FLAML) 进行交互，执行模型训练和预测
"""

import time
import os
import logging # 导入标准 logging 用于类型提示
from pathlib import Path  # 导入 Path 用于文件路径处理
from typing import Dict, Any, Optional, Tuple, Union, List, TYPE_CHECKING
import pandas as pd
import numpy as np
from loguru import logger
import joblib # 使用 joblib 进行序列化
try:
    from flaml import AutoML as FlamlAutoMLClass  # 导入并重命名 AutoML 类，用于运行时
    FLAML_INSTALLED = True
except ImportError:
    logger.error("FLAML 库未安装AutoML 功能将不可用请运行 'pip install flaml'")
    FlamlAutoMLClass = None # type: ignore
    FLAML_INSTALLED = False

# 导入项目级 utils
from espml.util import utils as common_utils
# 导入 AutoFE 的评估函数（用于在 Wrapper 内部重新评估）
try:
    from espml.autofe.model import _calculate_metric # 使用之前定义的内部函数
except ImportError:
    logger.error("无法从 espml.autofe.model 导入 _calculate_metric，评估功能受限")
    # 定义一个占位函数以防止导入失败
    def _calculate_metric(*args, **kwargs): return np.nan

# 正确引用loguru的Logger类型
if TYPE_CHECKING:
    from loguru import Logger as LoguruLogger

class FlamlAutomlWrapper:
    """
    对 FLAML AutoML 库的封装
    负责初始化、训练、预测和模型持久化
    代码逻辑
    """
    def __init__(self, config: Dict[str, Any], global_config: Optional[Dict[str, Any]] = None):
        """
        初始化 FlamlAutomlWrapper

        Args:
            config (Dict[str, Any]): AutoML 部分的配置 (project_config['AutoML'])
            global_config (Optional[Dict[str, Any]]): 完整的项目配置，用于获取关联设置

        Raises:
            ImportError: 如果 FLAML 未安装
            ValueError: 如果配置无效
            KeyError: 如果缺少必要配置
        """
        if not FLAML_INSTALLED:
             raise ImportError("FLAML 库未安装，无法初始化 FlamlAutomlWrapper")

        self.logger = logger.bind(name="FlamlAutomlWrapper")
        if not isinstance(config, dict): raise ValueError("AutoML 配置必须是字典")
        self.config = config
        self.global_config = global_config if global_config else {}
        self.logger.info("初始化 FlamlAutomlWrapper...")

        # --- 解析配置  ---
        self.method = self.config.get('Method', 'flaml')
        if self.method.lower() != 'flaml':
             raise ValueError(f"此 Wrapper 只支持 'flaml' 方法，但配置为 '{self.method}'")

        self.time_budget = self.config.get('TimeBudget', 300)
        user_flaml_settings = self.config.get('flaml_settings', {}) # 用户指定的 FLAML 设置

        # 获取全局配置
        self.task_type = common_utils.safe_dict_get(self.global_config, 'Feature.TaskType', 'regression')
        self.metric = common_utils.safe_dict_get(self.global_config, 'Feature.Metric', 'rmse')
        self.random_seed = common_utils.safe_dict_get(self.global_config, 'Feature.RandomSeed')

        # --- 准备 FLAML 参数 ---
        # 基础参数，会被 user_flaml_settings 覆盖
        self.flaml_settings: Dict[str, Any] = {
            'task': self.task_type,
            'metric': self.metric, # flaml 会自动映射或需要指定兼容指标
            'time_budget': self.time_budget,
            'seed': self.random_seed,
            'n_jobs': -1,
            'log_file_name': 'flaml.log', # 默认名称，会在 fit 时覆盖
            'verbose': 3, # 假设代码设置了详细程度
            # 'estimator_list': 'auto', # 默认自动选择
            # 'eval_method': 'auto',
            # 'split_ratio': 0.2,
            # 'n_splits': 5,
        }
        # 合并用户特定设置
        self.flaml_settings.update(user_flaml_settings)

        # 内部状态
        self.automl_instance: Optional[Any] = None  # 使用Any代替FlamlAutoMLClass
        self.best_estimator: Optional[Any] = None
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_loss: Optional[float] = None # FLAML 报告的最佳损失
        self.final_val_score: Optional[float] = None # 使用最佳模型在验证集上的最终得分

        self.logger.info("FlamlAutomlWrapper 初始化完成")
        self.logger.debug(f"最终 FLAML 配置: {self.flaml_settings}")

    @common_utils.log_execution_time(level="INFO") # 对 fit 方法计时
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        cat_features: Optional[List[str]] = None,
        log_dir: Optional[str] = None,
        experiment_name: str = "automl_experiment"
        ) -> None:
        """
        使用 FLAML 执行 AutoML 训练

        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征 (可选)
            y_val: 验证目标 (可选)
            cat_features: 分类特征列表 (可选)
            log_dir: FLAML 日志目录 (可选)
            experiment_name: 实验名称，用于日志文件名

        Raises:
            RuntimeError: 如果 FLAML 训练失败
            ValueError: 如果输入数据无效
        """
        self.logger.info(f"开始 FLAML AutoML 训练 (Budget: {self.time_budget}s, Metric: {self.metric})...")
        if not FLAML_INSTALLED: raise RuntimeError("FLAML 未安装")
        if X_train is None or y_train is None: raise ValueError("X_train 和 y_train 不能为空")
        # 确保 y_train 是 Series 或 1D array
        if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
            y_train = y_train.iloc[:, 0]
        elif isinstance(y_train, np.ndarray) and y_train.ndim > 1:
            y_train = y_train.flatten()

        # --- 准备 fit 参数 ---
        fit_kwargs = self.flaml_settings.copy()

        # 设置日志文件路径
        if log_dir:
             log_path = Path(log_dir)
             common_utils.mkdir_if_not_exist(log_path)
             log_file = log_path / f"{experiment_name}_flaml.log"
             fit_kwargs['log_file_name'] = str(log_file)
             self.logger.info(f"FLAML log file: {log_file}") # 严格匹配日志格式
        else:
             default_log_file = f"{experiment_name}_flaml.log"
             fit_kwargs['log_file_name'] = default_log_file
             self.logger.warning(f"未指定 FLAML 日志目录，将写入到当前目录: {default_log_file}")

        # 准备验证集 (如果提供)
        if X_val is not None and y_val is not None:
             self.logger.info("使用提供的验证集进行 FLAML 训练评估")
             # 确保 y_val 格式正确
             if isinstance(y_val, pd.DataFrame) and y_val.shape[1] == 1: y_val = y_val.iloc[:, 0]
             elif isinstance(y_val, np.ndarray) and y_val.ndim > 1: y_val = y_val.flatten()
             fit_kwargs['X_val'] = X_val
             fit_kwargs['y_val'] = y_val
             # 代码可能强制 'holdout'，或者依赖 FLAML 自动判断
             # fit_kwargs['eval_method'] = 'holdout'
        # else: self.logger.info("未提供验证集，FLAML 将使用内部方法评估")

        # 处理分类特征
        if cat_features:
             valid_cats = [c for c in cat_features if c in X_train.columns]
             if len(valid_cats) != len(cat_features):
                  logger.warning(f"提供的分类特征列表包含 X_train 中不存在的列，只传递存在的: {valid_cats}")
             # FLAML v1+ 推荐在 DataFrame 中设置类型，而不是传列表
             # 代码可能传递列表，
             fit_kwargs['categorical_feature'] = valid_cats
             logger.debug(f"传递 categorical_feature: {valid_cats}")
             # 如果使用 DataFrame 类型，则需要在此转换 X_train/X_val 类型
             # for col in valid_cats:
             #     X_train[col] = X_train[col].astype('category')
             #     if X_val is not None: X_val[col] = X_val[col].astype('category')

        # --- 执行训练 ---
        self.automl_instance = FlamlAutoMLClass() # 创建新实例，使用重命名的类
        try:
             self.logger.info("调用 flaml.AutoML().fit()...")
             # logger.debug(f"FLAML fit kwargs: {fit_kwargs}") # 可能过长
             self.automl_instance.fit(X_train=X_train, y_train=y_train, **fit_kwargs)

             # --- 获取结果 ---
             self.best_estimator = self.automl_instance.model.estimator
             self.best_config = self.automl_instance.best_config
             self.best_loss = self.automl_instance.best_loss # 这是 FLAML 内部验证的分数

             self.logger.info(f"FLAML AutoML 训练完成")
             # 严格匹配日志格式
             self.logger.info(f"best estimator = {self.automl_instance.best_estimator}, config = {self.best_config}, #training instances = {len(X_train)}")
             # self.logger.info(f"FLAML 内部最佳损失 ({self.metric}): {self.best_loss:.6f}") # 可以记录内部损失

             # --- 使用最佳模型在 *提供的* 验证集上重新评估 ---
             # 这是更可靠的最终分数
             if X_val is not None and y_val is not None and self.best_estimator is not None:
                  self.logger.info(f"使用最佳模型在提供的验证集上重新评估 (Metric: {self.metric})...")
                  try:
                       # 需要处理概率预测和标签预测
                       predict_proba_available = hasattr(self.best_estimator, 'predict_proba')
                       predict_needed = self.metric not in ['neg_log_loss'] # 非 logloss 指标需要 predict 结果
                       predict_proba_needed = self.metric == 'neg_log_loss' # logloss 需要 predict_proba

                       y_pred_val = None
                       if predict_needed:
                           y_pred_val = self.best_estimator.predict(X_val)
                       elif predict_proba_needed and predict_proba_available:
                            y_pred_val = self.best_estimator.predict_proba(X_val)
                       elif predict_proba_needed and not predict_proba_available:
                            logger.error(f"需要概率预测但最佳模型 {type(self.best_estimator)} 不支持 predict_proba无法计算 {self.metric}")
                            self.final_val_score = np.nan
                       else: # 不需要预测（例如，如果 metric 已经被 flaml 计算并返回）
                            self.final_val_score = self.best_loss # 假设 best_loss 就是最终分数 (不推荐)


                       if y_pred_val is not None:
                            # 调用评估函数计算最终分数
                            self.final_val_score = _calculate_metric(self.metric, y_val, y_pred_val)
                            self.logger.info(f"automl finished, best model can achieve {self.metric}: {self.final_val_score:.6f} on provided validation set.")
                       elif not np.isnan(self.final_val_score): # 处理不需要预测的情况
                            pass # final_val_score 已被设置
                       else: # 无法计算分数
                           self.final_val_score = self.best_loss # 回退到 best_loss
                           self.logger.warning(f"无法在验证集上计算最终指标 {self.metric}，使用 FLAML 内部最佳损失 {self.best_loss:.6f} 代替")

                  except Exception as eval_e:
                       self.logger.error(f"使用最佳模型在验证集上评估失败: {eval_e}", exc_info=True)
                       self.final_val_score = self.best_loss # 评估失败则使用 best_loss
                       self.logger.warning(f"评估失败，使用 FLAML 内部最佳损失 {self.best_loss:.6f}")
             else:
                  # 没有提供验证集，直接使用 best_loss
                  self.final_val_score = self.best_loss
                  self.logger.info(f"automl finished, best loss recorded by FLAML: {self.best_loss:.6f} (metric used by FLAML: {self.automl_instance.metric})")

        except Exception as e:
             self.logger.exception(f"FLAML AutoML 训练过程中发生严重错误: {e}")
             self.automl_instance = None # 标记训练失败
             self.best_estimator = None
             raise RuntimeError(f"FLAML fit 失败: {e}") from e

    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> Optional[np.ndarray]:
        """使用训练好的最佳模型进行预测"""
        self.logger.info(f"开始使用 AutoML 最佳模型进行预测 (数据形状: {X_test.shape})...")
        if self.best_estimator is None: # 检查 best_estimator 是否存在
             self.logger.error("预测失败AutoML 最佳模型不存在（训练可能未进行或失败）")
             return None
        try:
            predictions = self.best_estimator.predict(X_test)
            self.logger.info("AutoML 预测完成")
            return predictions
        except Exception as e:
             self.logger.exception(f"使用 AutoML 最佳模型预测时出错: {e}")
             return None

    def predict_proba(self, X_test: Union[pd.DataFrame, np.ndarray]) -> Optional[np.ndarray]:
        """使用训练好的最佳模型进行概率预测"""
        if self.task_type == 'regression':
            self.logger.error("概率预测仅适用于分类任务")
            return None
        self.logger.info(f"开始使用 AutoML 最佳模型进行概率预测 (数据形状: {X_test.shape})...")
        if self.best_estimator is None:
             self.logger.error("概率预测失败AutoML 最佳模型不存在")
             return None
        if not hasattr(self.best_estimator, 'predict_proba'):
             self.logger.error(f"最佳模型 {type(self.best_estimator)} 不支持 'predict_proba' 方法")
             return None
        try:
            probabilities = self.best_estimator.predict_proba(X_test)
            self.logger.info("AutoML 概率预测完成")
            return probabilities
        except Exception as e:
             self.logger.exception(f"使用 AutoML 最佳模型进行概率预测时出错: {e}")
             return None

    # --- 模型持久化  ---
    def save_model(self, file_path: str, only_estimator: bool = False) -> bool:
        """
        保存训练好的 AutoML 实例或仅保存最佳估计器
        假设代码可能保存整个 AutoML 实例

        Args:
            file_path (str): 模型保存路径
            only_estimator (bool): 是否只保存最佳估计器（减小体积）默认为 False

        Returns:
            bool: 是否保存成功
        """
        target_to_save: Optional[Any] = None
        if only_estimator:
            target_to_save = self.best_estimator
            if target_to_save is None:
                self.logger.error("无法保存模型最佳估计器不存在")
                return False
            log_msg = "最佳估计器"
        else:
            target_to_save = self.automl_instance
            if target_to_save is None:
                 self.logger.error("无法保存模型AutoML 实例不存在")
                 return False
            log_msg = "AutoML 实例"

        path = Path(file_path)
        if not common_utils.mkdir_if_not_exist(path.parent): return False

        self.logger.info(f"开始保存 {log_msg} 到: {file_path}")
        try:
             joblib.dump(target_to_save, path)
             self.logger.info(f"{log_msg} 保存成功")
             return True
        except Exception as e:
             self.logger.exception(f"保存 {log_msg} 到 '{file_path}' 时失败: {e}")
             return False

    @classmethod # 类方法加载
    def load_model(cls,
                   file_path: str,
                   logger_instance: Optional[Union[logging.Logger, 'LoguruLogger']] = None,  # 使用引号引用LoguruLogger
                   # 需要提供加载时的配置，因为无法完全从模型恢复
                   config: Optional[Dict[str, Any]] = None,
                   global_config: Optional[Dict[str, Any]] = None
                   ) -> Optional['FlamlAutomlWrapper']:
        """
        从文件加载之前保存的 AutoML 实例或最佳估计器
        假设保存的是 AutoML 实例

        Args:
            file_path (str): 模型文件路径
            logger_instance (Optional[Union[logging.Logger, 'LoguruLogger']]): 用于日志记录的 logger
            config (Optional[Dict[str, Any]]): 加载时使用的 AutoML 配置
            global_config (Optional[Dict[str, Any]]): 加载时使用的全局配置

        Returns:
            Optional[FlamlAutomlWrapper]: 加载并初始化的 Wrapper 实例，如果失败则返回 None
        """
        lg = logger_instance if logger_instance else logger
        # 绑定 logger 名称，如果 logger_instance 是 loguru logger
        if hasattr(lg, 'bind'): lg = lg.bind(name="FlamlAutomlWrapper")
        lg.info(f"开始从文件加载 AutoML 模型: {file_path}")

        if not common_utils.check_path_exists(file_path, path_type='f'):
             lg.error(f"加载模型失败文件不存在 {file_path}")
             return None
        if config is None:
            lg.error("加载模型失败需要提供 AutoML 配置 ('config')")
            return None

        try:
             # 加载对象
             loaded_obj = joblib.load(file_path)

             # 检查加载的是 AutoML 实例还是仅估计器
             if isinstance(loaded_obj, FlamlAutoMLClass):  # 使用重命名的类进行检查
                 loaded_automl_instance = loaded_obj
                 loaded_best_estimator = loaded_automl_instance.model.estimator
                 lg.info("成功加载完整的 AutoML 实例")
             elif hasattr(loaded_obj, 'predict'): # 假设加载的是估计器
                 loaded_automl_instance = None # 没有完整的 AutoML 实例
                 loaded_best_estimator = loaded_obj
                 lg.warning("加载的是最佳估计器而非完整 AutoML 实例，部分信息（如 best_config, best_loss）可能丢失")
             else:
                 lg.error(f"加载的文件 '{file_path}' 不是有效的 FLAML AutoML 实例或估计器 (类型: {type(loaded_obj)})")
                 return None

             # 使用 *传入的* 配置创建 Wrapper 实例
             wrapper = cls(config=config, global_config=global_config)
             wrapper.automl_instance = loaded_automl_instance
             wrapper.best_estimator = loaded_best_estimator
             # 尝试从加载的实例中恢复 best_config 和 best_loss (如果加载的是 AutoML)
             if loaded_automl_instance:
                  wrapper.best_config = loaded_automl_instance.best_config
                  wrapper.best_loss = loaded_automl_instance.best_loss

             lg.info("FlamlAutomlWrapper 实例已根据加载的模型初始化")
             return wrapper

        except Exception as e:
             lg.exception(f"加载 AutoML 模型或初始化 Wrapper 实例时出错: {file_path} - {e}")
             return None


logger.info("AutoML 模块 (espml.automl.automl) 加载完成")
