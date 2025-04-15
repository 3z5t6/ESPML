# -*- coding: utf-8 -*-
"""
核心机器学习流程模块 (espml)
负责协调数据处理、自动特征工程 (AutoFE) 和自动机器学习 (AutoML) 完成训练和预测任务
"""

import datetime
import os
from pathlib import Path
import time
from typing import Dict, Any, Optional, Tuple, List, Union
import pandas as pd
import numpy as np
from loguru import logger
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor # 用于初始评估
from sklearn.metrics import mean_squared_error # 用于初始评估

# 导入 espml 内部模块
from espml.dataprocess.data_processor import DataProcessor, DataProcessingError
from espml.autofe.autofe import AutoFE
from espml.autofe.transform import Transform # 需要 Transform 类用于加载状态
from espml.automl.automl import FlamlAutomlWrapper
from espml.incrml.metadata import ModelVersionInfo # 需要创建元数据对象
from espml.util import utils as common_utils
from espml.util import const
from espml.util import result_saver # 可能需要保存初始评估,

class MLPipeline:
    """
    机器学习流程协调器
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 MLPipeline

        Args:
            config (Dict[str, Any]): 完整的项目配置字典
        """
        self.config = config
        # 创建子 logger
        self.logger = logger.bind(name="MLPipeline")
        self.logger.info("初始化 MLPipeline...")

        # 提取配置
        self.feature_config = config.get('Feature', {})
        self.autofe_config = config.get('AutoFE', {})
        self.automl_config = config.get('AutoML', {})
        self.incrml_config = config.get('IncrML', {})
        self.resource_config = config.get('Resource', {})
        self.project_config = config.get('Project', {}) # 获取项目配置
        self.task_name = config.get('TaskName', 'default_task') # 获取任务名

        self.target_name = self.feature_config.get('TargetName')
        if not self.target_name: raise ValueError("配置中缺少 'Feature.TargetName'")
        self.random_seed = self.feature_config.get('RandomSeed', 1024)
        self.test_size = self.feature_config.get('TestSize', 0.25)
        self.metric = self.feature_config.get('Metric', 'rmse')
        self.task_type = self.feature_config.get('TaskType', 'regression')
        # 修正从 IncrML 配置获取模型保存基础路径
        self.base_model_save_path = self.incrml_config.get('SaveModelPath', f'data/model/{self.task_name}')

        # 内部状态,存储上次运行的结果路径等
        self.last_run_id: Optional[str] = None
        self.last_run_model_path: Optional[str] = None
        self.last_run_transformer_path: Optional[str] = None
        self.last_run_features_path: Optional[str] = None
        self.last_run_performance: Optional[Dict[str, float]] = None
        self.last_run_selected_autofe_features: Optional[List[str]] = None

        self.logger.info("MLPipeline 初始化完成")

    def _get_run_specific_paths(self, run_id: str) -> Tuple[str, str, str, str]:
        """(内部) 根据运行 ID 生成模型、转换器状态和特征列表的保存路径"""
        # 确保 run_id 是字符串
        run_id_str = str(run_id)
        # 保存路径基于 base_model_save_path / run_id
        run_dir = Path(self.base_model_save_path) / run_id_str
        common_utils.mkdir_if_not_exist(run_dir)

        model_path = str(run_dir / f"model_{run_id_str}.joblib")
        # utoFE 状态保存名称
        transformer_state_path = str(run_dir / f"transformer_state_{run_id_str}.joblib")
        feature_list_path = str(run_dir / f"selected_features_{run_id_str}.json")
        # AutoML 日志目录
        automl_log_dir = str(run_dir / "automl" / "logs")
        common_utils.mkdir_if_not_exist(Path(automl_log_dir).parent)
        common_utils.mkdir_if_not_exist(automl_log_dir)

        return model_path, transformer_state_path, feature_list_path, automl_log_dir

    # 使用计时器装饰整个训练过程
    @common_utils.log_execution_time(level="INFO")
    def train(self,
              df_train_full: pd.DataFrame,
              run_id: Optional[str] = None # 允许外部传入 run_id
             ) -> bool:
        """
        执行完整的模型训练流程流程和日志

        Args:
            df_train_full (pd.DataFrame): 包含特征和目标列的完整训练数据集
            run_id (Optional[str]): 本次运行的 ID如果为 None,则自动生成

        Returns:
            bool: 训练是否成功完成
        """
        # --- 初始化和日志 ---
        start_time_train = time.time()
        # 确定 run_id
        self.last_run_id = run_id if run_id else datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S%f')
        log_round = self.last_run_id # 使用 run_id 作为轮次标识
        self.logger.info(f"================ 开始训练 {log_round} round trail ================") # 严格匹配日志

        # 获取保存路径
        model_path, transformer_path, features_path, automl_log_dir = self._get_run_specific_paths(self.last_run_id)
        self.logger.info(f"本次运行 ID: {self.last_run_id}")
        self.logger.info(f"模型及状态将保存到: {os.path.dirname(model_path)}")

        # --- 步骤 1: 数据处理 ---
        # (MLPipeline 不直接调用 DP,而是接收已处理数据?)
        # (与 wind_incrml 分析矛盾,此处假设 MLPipeline 负责调用 DP)
        self.logger.info("步骤 1/5: 执行数据处理...")
        X_processed: Optional[pd.DataFrame] = None
        y_processed: Optional[pd.Series] = None
        initial_features: List[str] = []
        try:
            data_processor = DataProcessor(config=self.effective_config) # 使用有效配置
            df_processed = data_processor.process(df_train_full) # Process 整个传入的 DF
            if df_processed.empty or self.target_name not in df_processed.columns:
                 raise DataProcessingError("数据处理后结果为空或缺少目标列")
            y_processed = df_processed[self.target_name]
            X_processed = df_processed.drop(columns=[self.target_name])
            initial_features = list(X_processed.columns)
            self.logger.info(f"数据处理完成处理后特征数量: {len(initial_features)}")
        except Exception as e:
            self.logger.exception("数据处理阶段失败!训练终止")
            return False

        # --- 步骤 2: 训练/验证集拆分 ---
        self.logger.info("步骤 2/5: 拆分训练集/验证集...")
        X_train: pd.DataFrame; X_val: pd.DataFrame
        y_train: pd.Series; y_val: pd.Series
        try:
            val_size = max(0.01, min(0.99, self.test_size))
            stratify_opt = y_processed if self.task_type == 'classification' else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_processed, y_processed, test_size=val_size,
                random_state=self.random_seed, shuffle=True, stratify=stratify_opt
            )
            self.logger.info(f"拆分完成训练集大小: {X_train.shape}, 验证集大小: {X_val.shape}")
        except Exception as e:
            self.logger.exception("拆分训练/验证集失败!训练终止")
            return False

        # --- 步骤 3: (可选) 初始评估 ---
        #  "origin data train val rmse"
        self.logger.info("步骤 3/5: 执行初始基线评估...")
        try:
            # 使用简单快速的模型,例如 ExtraTreesRegressor
            initial_model = ExtraTreesRegressor(n_estimators=20, n_jobs=-1, random_state=self.random_seed)
            initial_model.fit(X_train.fillna(0), y_train) # 简单填充NaN
            initial_preds = initial_model.predict(X_val.fillna(0))
            # 计算 RMSE (假设 metric 是 rmse 或兼容)
            initial_rmse = np.sqrt(mean_squared_error(y_val, initial_preds))
            # 严格匹配日志格式
            self.logger.info(f"origin data train val rmse: {initial_rmse:.17f}") # 匹配日志中的高精度
        except Exception as e:
            self.logger.warning(f"初始基线评估失败: {e}")
            initial_rmse = np.inf # 设置为最差

        # --- 步骤 4: 自动特征工程 (AutoFE) ---
        self.logger.info("步骤 4/5: 执行自动特征工程 (AutoFE)...")
        final_selected_autofe_features: List[str] = []
        autofe_transformer_state: Optional[Dict] = None
        X_train_final = X_train # 如果 AutoFE 失败或禁用,使用拆分数据
        X_val_final = X_val     # 如果 AutoFE 失败或禁用,使用拆分数据
        try:
            # 实例化 AutoFE 引擎,传递完整配置
            autofe_instance = AutoFE(logger_instance=self.logger, **self.effective_config)
            if autofe_instance.running:
                 # 调用 fit,传入拆分后的数据
                 # autofe.fit 内部会打印 "Autofe Trial finished..." 等日志
                 # 返回增强后的 X_train, X_val 和选中的 *新* 特征列表
                 X_train_final, X_val_final, _, _, final_selected_autofe_features = autofe_instance.fit(
                     X_train, y_train, X_val, y_val
                 )
                 # 获取 transformer 状态
                 if autofe_instance.transformer and hasattr(autofe_instance.transformer, '_record'):
                      autofe_transformer_state = autofe_instance.transformer._record
                      self.logger.info(f"已获取 AutoFE Transformer 状态 ({len(autofe_transformer_state)} 条)")
                 # 严格匹配日志
                 self.logger.info(f"autofe finished, search {len(final_selected_autofe_features)} features")
            else:
                 self.logger.info("AutoFE 未启用,跳过此步骤")
                 final_selected_autofe_features = []
                 autofe_transformer_state = {} # 保存空状态

            # 保存 AutoFE 结果 (状态和特征列表)
            #  Transformer 状态
            if not common_utils.dump_pickle(autofe_transformer_state, transformer_path):
                 self.logger.error("保存 AutoFE Transformer 状态失败!后续预测可能出错")
                 # 是否算作训练失败?,此处假设继续但记录错误
            else:
                 self.logger.info(f"AutoFE Transformer 状态已保存到: {transformer_path}")
            # 保存选中的特征列表
            if not common_utils.write_json_file({"selected_features": final_selected_autofe_features}, features_path):
                 self.logger.error("保存选中的 AutoFE 特征列表失败!后续预测可能出错")
            else:
                 self.logger.info(f"选中的 AutoFE 特征列表已保存到: {features_path}")

        except Exception as e:
            self.logger.exception("自动特征工程 (AutoFE) 阶段失败!训练终止")
            return False

        # --- 步骤 5: 自动机器学习 (AutoML) ---
        self.logger.info("步骤 5/5: 执行自动机器学习 (AutoML)...")
        automl_wrapper: Optional[FlamlAutomlWrapper] = None
        training_successful = False
        try:
            automl_wrapper = FlamlAutomlWrapper(config=self.automl_config, global_config=self.effective_config)
            # 获取最终的分类特征列表
            final_cat_features = [f for f in self.feature_config.get('CategoricalFeature', []) if f in X_train_final.columns]
            newly_generated_cols = set(X_train_final.columns) - set(X_train.columns)
            for col in newly_generated_cols:
                 if pd.api.types.is_categorical_dtype(X_train_final[col]) or pd.api.types.is_object_dtype(X_train_final[col]):
                      if col not in final_cat_features: final_cat_features.append(col)

            # 严格匹配日志
            self.logger.info(f"start {self.task_name} fit...")
            automl_wrapper.fit(
                X_train=X_train_final, y_train=y_train,
                X_val=X_val_final, y_val=y_val,
                cat_features=final_cat_features,
                log_dir=automl_log_dir, # 传递 AutoML 日志目录
                experiment_name=f"{self.task_name}_run_{self.last_run_id}"
            )
            # FlamlAutomlWrapper 内部会打印 "automl finished..." 日志

            # 保存 AutoML 模型
            if not automl_wrapper.save_model(model_path):
                 raise RuntimeError("保存 AutoML 模型失败!")

            training_successful = True # 标记训练成功
            # 存储结果供外部获取（如果需要）
            self.last_run_model_path = model_path
            self.last_run_transformer_path = transformer_path
            self.last_run_features_path = features_path
            self.last_run_performance = {self.metric: automl_wrapper.final_val_score} if automl_wrapper.final_val_score is not None else {}
            self.last_run_selected_autofe_features = final_selected_autofe_features

        except Exception as e:
            self.logger.exception("自动机器学习 (AutoML) 阶段失败!")
            training_successful = False # 标记失败

        # --- 训练结束 ---
        train_end_time = time.time()
        if training_successful:
             self.logger.success("finish training.") # 严格匹配日志
             self.logger.info(f"================ 训练 {log_round} 结束 (耗时: {train_end_time - start_time_train:.2f} 秒) ================")
             return True
        else:
             self.logger.error("训练过程失败")
             self.logger.info(f"================ 训练 {log_round} 结束（失败） (耗时: {train_end_time - start_time_train:.2f} 秒) ================")
             return False

    # 使用计时器装饰
    @common_utils.log_execution_time(level="INFO")
    def predict(self,
                X_test: pd.DataFrame,
                run_id: str # 预测时必须指定要加载的模型对应的 run_id
               ) -> Optional[np.ndarray]:
        """
        执行预测流程

        Args:
            X_test (pd.DataFrame): 需要预测的输入特征数据
            run_id (str): 要加载的模型/状态对应的运行 ID

        Returns:
            Optional[np.ndarray]: 预测结果数组,如果失败则返回 None
        """
        self.logger.info(f"================ 开始预测 (使用 Run ID: {run_id}) ================")
        predict_start_time = time.time()

        # --- 获取加载路径 ---
        try:
            model_path, transformer_path, features_path, _ = self._get_run_specific_paths(run_id)
        except Exception as path_e:
             self.logger.error(f"无法确定 Run ID '{run_id}' 的加载路径: {path_e}")
             return None

        # --- 步骤 1: 加载 AutoFE 状态和特征列表 ---
        self.logger.info("步骤 1/5: 加载 AutoFE 状态和特征列表...")
        loaded_transformer_state: Optional[Dict] = common_utils.load_pickle(transformer_path)
        loaded_features_info: Optional[Dict] = common_utils.read_json_file(features_path)

        if loaded_transformer_state is None:
            self.logger.error(f"加载 AutoFE Transformer 状态失败: {transformer_path}")
            return None
        if loaded_features_info is None or "selected_features" not in loaded_features_info:
             self.logger.error(f"加载选中的 AutoFE 特征列表失败或格式错误: {features_path}")
             return None
        final_selected_features: List[str] = loaded_features_info["selected_features"]
        self.logger.info(f"加载成功: Transformer 状态 ({len(loaded_transformer_state)} 条), 选中特征列表 ({len(final_selected_features)} 个)")

        # --- 步骤 2: 加载 AutoML 模型 ---
        self.logger.info("步骤 2/5: 加载 AutoML 模型...")
        automl_wrapper = FlamlAutomlWrapper.load_model(
            model_path, logger_instance=self.logger,
            config=self.automl_config, global_config=self.config # 需要提供配置
        )
        if automl_wrapper is None or automl_wrapper.best_estimator is None:
            self.logger.error(f"加载 AutoML 模型失败: {model_path}")
            return None
        self.logger.info("AutoML 模型加载成功")

        # --- 步骤 3: 数据处理 ---
        self.logger.info("步骤 3/5: 对输入数据执行数据处理...")
        X_test_processed: Optional[pd.DataFrame] = None
        try:
            data_processor = DataProcessor(config=self.effective_config)
            # 预测时不需要目标列,但 process 可能需要完整 DF 结构
            # 假设 process 可以处理不含 target 的 DF
            if self.target_name in X_test.columns:
                 X_test_input = X_test.drop(columns=[self.target_name])
            else: X_test_input = X_test
            # Process 应该返回与训练时一致的特征集（去除 target 后）
            X_test_processed = data_processor.process(X_test_input)
            if X_test_processed.empty: raise DataProcessingError("数据处理后结果为空")
            # 提取训练时的初始特征列（需要保存或从 DataProcessor 获取）
            # 假设 DataProcessor 有方法获取处理后的列名
            if hasattr(data_processor, 'output_feature_names'):
                 initial_features_pred = [f for f in data_processor.output_feature_names if f != self.target_name]
            else: # 回退假设 X_test_processed 的列就是初始特征
                 initial_features_pred = list(X_test_processed.columns)
                 logger.warning("无法从 DataProcessor 获取确切的初始特征列表,将使用处理后的所有列")

            # 确保只包含初始特征
            X_test_processed = X_test_processed[initial_features_pred]
            self.logger.info(f"数据处理完成处理后特征数量: {len(initial_features_pred)}")
        except Exception as e:
            self.logger.exception("预测时数据处理阶段失败!")
            return None

        # --- 步骤 4: 应用 AutoFE 转换 ---
        X_test_final = X_test_processed.copy() # 从处理后的数据开始
        if final_selected_features: # 只有训练时选出了特征才需要生成
            self.logger.info(f"步骤 4/5: 应用 AutoFE 转换以生成 {len(final_selected_features)} 个特征...")
            try:
                 # 创建 Transform 实例并加载状态
                 transformer = Transform(logger=self.logger, **self.effective_config)
                 transformer._record = loaded_transformer_state # 加载状态
                 # 调用 transform 方法计算特征
                 X_test_final = transformer.transform(X_test_processed, final_selected_features)
                 # 检查并对齐列（与训练时使用的最终列对齐）
                 final_train_columns = initial_features_pred + final_selected_features # 训练时使用的完整列
                 missing_cols = set(final_train_columns) - set(X_test_final.columns)
                 if missing_cols: logger.error(f"预测时生成特征后缺少列: {missing_cols}")
                 # 按训练时的顺序和存在性选择列
                 X_test_final = X_test_final[[col for col in final_train_columns if col in X_test_final.columns]]
                 self.logger.info("AutoFE 转换应用完成")
            except Exception as e:
                self.logger.exception("预测时应用 AutoFE 转换失败!")
                return None
        else:
            self.logger.info("步骤 4/5: 训练时未选中 AutoFE 特征,跳过转换")

        # --- 步骤 5: 执行预测 ---
        self.logger.info("步骤 5/5: 使用加载的模型执行预测...")
        try:
            # 直接调用 predict
            predictions = automl_wrapper.predict(X_test_final)
            if predictions is None:
                 raise RuntimeError("AutoML wrapper predict 方法返回 None")
            self.logger.info(f"预测完成,共生成 {len(predictions)} 个预测结果")
        except Exception as e:
            self.logger.exception("执行预测失败!")
            return None

        predict_end_time = time.time()
        self.logger.success("预测流程成功完成")
        self.logger.info(f"================ 预测结束 (耗时: {predict_end_time - predict_start_time:.2f} 秒) ================")

        return predictions


logger.info("核心 ML 流程模块 (espml.ml) 加载完成")