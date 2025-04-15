# -*- coding: utf-8 -*-
"""
自动化特征工程 (AutoFE) 主引擎类 (espml)
负责根据配置初始化和运行 AutoFE 迭代流程
"""

import os
import pandas as pd
import numpy as np
# from logging import Logger # 使用 loguru
from loguru import logger # 适配 espml 日志
from loguru._logger import Logger as LoguruLoggerType
from typing import Optional, Tuple, List, Dict, Any, Set

# 导入所需的内部模块 (路径适配为 espml)
from espml.autofe.transform import Transform # 导入用户提供的 Transform 类
from espml.autofe import utils as autofe_utils # 导入整个utils模块作为autofe_utils
from espml.autofe.utils import feature_space, update_time_span # 导入 autofe.utils 函数
from espml.autofe.algorithm import ( # 导入 algorithm 中的核心函数
    model_features_select,
    max_threads_name2feature,
    threads_feature_select
)
# 导入项目级通用工具 (如果需要)
from espml.util import utils as common_utils

class AutoFE:
    """
    自动化特征工程类

    负责根据配置执行特征生成和选择的迭代过程
    版本

    属性:
        n (int): 特征工程的迭代轮数
        method (Optional[str]): 特征工程方法（保留，未使用）
        base_score (float): 用于比较的基线分数
        logger (logger): loguru logger 实例
        transformer (Transform): 特征转换器实例
        task_type (Optional[str]): 任务类型
        metric (Optional[str]): 评估指标
        seed (int): 随机种子
        cat_features (List[str]): 分类特征列表
        target_name (Optional[str]): 目标变量名称
        time_index (Optional[str]): 时间索引名称
        group_index (Optional[str]): 分组索引名称
        time_span (Optional[List[int]]): 时间跨度列表
        max_workers (int): 用于并行计算的最大工作进程/线程数
    """
    # pylint: disable=dangerous-default-value # 允许 kwargs
    def __init__(
        self,
        n: int = 2,
        method: Optional[str] = None,
        base_score: float = np.inf, # 初始化为最差分数 (假设越小越好)
        logger_instance: Optional[LoguruLoggerType] = logger, # 接收 logger 实例，避免与 logger 变量冲突
        transformer: Optional[Transform] = None,
        **kwargs: Any # 接收完整的项目配置字典或 AutoFE 部分
    ):
        """
        初始化 AutoFE 实例

        Args:
            n (int): 特征工程迭代轮数
            method (Optional[str]): 特征工程方法（保留）
            base_score (float): 基线分数 (应根据 metric 方向设置)
            logger_instance (Optional[logger]): loguru logger 实例
            transformer (Optional[Transform]): Transform 实例
            **kwargs (Any): 附加参数字典，预期包含 'Feature', 'AutoFE', 'Resource' 等键
        """
        # 使用传入或默认的 logger
        self.logger = logger_instance if logger_instance else logger.bind(name="AutoFE_default")
        if not logger_instance: self.logger.warning("未提供 logger 实例，使用默认绑定")
        # 绑定子 logger 名称
        self.logger = self.logger.bind(name="AutoFE")

        self.logger.info(f"初始化 AutoFE 引擎 (迭代轮数 n={n})...")
        self.n = int(n)
        self.method = method # 保留

        # 从 kwargs 安全地提取配置
        AUTOFE_CONFIG: Dict[str, Any] = kwargs.get('AutoFE', {})
        FEATURE_CONFIG: Dict[str, Any] = kwargs.get('Feature', {})
        RESOURCE_CONFIG: Dict[str, Any] = kwargs.get('Resource', {})

        # 提取特征相关配置
        self.task_type: Optional[str] = FEATURE_CONFIG.get('TaskType')
        self.metric: Optional[str] = FEATURE_CONFIG.get('Metric')
        # 处理代码可能的拼写错误 'metirc' -> 'Metric'
        if self.metric is None and 'metirc' in FEATURE_CONFIG:
             self.metric = FEATURE_CONFIG.get('metirc')
             self.logger.warning("配置键 'metirc' 被读取，建议更新为 'Metric'")
        if not self.metric: # 如果仍然没有，设置默认或报错
             self.logger.error("初始化 AutoFE 失败缺少评估指标 'Metric' 配置")
             raise ValueError("缺少 Metric 配置")

        self.seed: int = FEATURE_CONFIG.get('RandomSeed', 1024)
        self.cat_features: List[str] = list(FEATURE_CONFIG.get('CategoricalFeature', []))
        self.target_name: Optional[str] = FEATURE_CONFIG.get('TargetName')
        if not self.target_name:
            raise ValueError("初始化 AutoFE 失败缺少目标名称 'TargetName' 配置")
        self.time_index: Optional[str] = FEATURE_CONFIG.get('TimeIndex')
        self.group_index: Optional[str] = FEATURE_CONFIG.get('GroupIndex')
        time_window_config = FEATURE_CONFIG.get('TimeWindow') # 假设这个键包含时间跨度信息
        self.time_span: Optional[List[int]] = autofe_utils.update_time_span(time_window_config)

        # 设置基线分数（根据指标方向）
        higher_is_better_metrics = {'roc_auc', 'f1', 'accuracy', 'auc_mu', 'r2', 'ap'} # 示例
        self.higher_is_better = self.metric in higher_is_better_metrics
        # 如果传入的 base_score 是默认的 0.5，则根据 metric 调整
        if base_score == 0.5: # 代码的默认值
             self.base_score = -np.inf if self.higher_is_better else np.inf
        else:
             self.base_score = float(base_score)

        # 提取并行计算配置
        self.max_workers = RESOURCE_CONFIG.get('MaxWorkers', -1) # 默认使用所有核心？

        # 初始化 Transformer
        if transformer is not None:
             if not isinstance(transformer, Transform):
                  raise TypeError("'transformer' 参数必须是 Transform 类的实例")
             self.transformer = transformer
        else:
             self.logger.debug("未提供 Transformer 实例，将创建一个新的实例...")
             # 创建 Transform 实例，需要传递必要的配置给它
             transform_kwargs = {'Feature': FEATURE_CONFIG, 'logger': self.logger}
             try:
                  self.transformer = Transform(**transform_kwargs)
             except Exception as e:
                  self.logger.exception("初始化内部 Transform 实例失败")
                  raise RuntimeError("无法初始化 Transform") from e

        self.logger.info(f"AutoFE 初始化完成迭代轮数: {self.n}, 初始基线分数({self.metric}): {self.base_score:.6f}")
        # ... (其他配置日志)


    @common_utils.log_execution_time(level="INFO") # 对整个 fit 方法计时
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
        """
        执行自动化特征工程的拟合过程
        严格按照代码逻辑（迭代生成、筛选、模型选择）

        Args:
            X_train (pd.DataFrame): 训练集特征
            y_train (pd.Series): 训练集标签
            X_val (pd.DataFrame): 验证集特征
            y_val (pd.Series): 验证集标签

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
                增强后的 X_train, 增强后的 X_val, y_train, y_val, 最终选择的高级特征名称列表
        """
        if self.logger: self.logger.info("AutoFE fit 过程开始...")

        # --- 输入验证 ---
        if not all(isinstance(arg, (pd.DataFrame, pd.Series)) for arg in [X_train, y_train, X_val, y_val]):
             raise TypeError("fit 方法输入必须是 DataFrame 或 Series")
        if not X_train.index.equals(y_train.index) or not X_val.index.equals(y_val.index):
            raise ValueError("训练集或验证集的特征与标签索引不匹配")
        if not self.target_name: raise ValueError("目标变量名称 'target_name' 未设置")
        if self.target_name != y_train.name:
             self.logger.warning(f"输入 y_train 名称 '{y_train.name}' 与配置 '{self.target_name}' 不符，将重命名")
             y_train = y_train.rename(self.target_name)
             y_val = y_val.rename(self.target_name) # 同时重命名 y_val

        # --- 准备数据 ---
        df_train_iter = pd.concat([X_train.copy(), y_train], axis=1) # 在迭代中修改的训练数据
        df_val_iter = pd.concat([X_val.copy(), y_val], axis=1)   # 在迭代中修改的验证数据
        original_feature_cols = list(X_train.columns) # 初始特征列

        # --- 初始化状态 ---
        # already_selected 存储所有已存在或已生成的特征名，防止重复生成
        already_selected_features: Set[str] = set(original_feature_cols)
        if self.target_name: already_selected_features.add(self.target_name)
        if self.time_index: already_selected_features.add(self.time_index)
        if self.group_index: already_selected_features.add(self.group_index)

        best_score: float = self.base_score # 当前最佳分数
        pick_features: List[str] = [] # 最佳分数对应的 *新* 特征列表
        adv_features: Set[str] = set() # 所有迭代中累积的、模型认为有用的 *新* 特征

        # --- 特征生成和选择迭代 ---
        for i in range(self.n): # self.n 是迭代次数
            iteration_num = i + 1
            if self.logger: self.logger.info(f"--- 开始 AutoFE 迭代 {iteration_num}/{self.n} ---")

            # 1. 生成候选特征名称列表
            if self.logger: self.logger.debug("步骤 1: 生成候选特征名称...")
            candidate_feature_names: List[str] = []
            try:
                # 传递当前迭代 df_train_iter 的所有列名作为 already_selected
                # 确保 feature_space 不生成已存在的列
                candidate_feature_names = autofe_utils.feature_space(
                    df_train_iter, # 使用当前迭代的数据帧
                    target_name=self.target_name,
                    already_selected=list(df_train_iter.columns), # 传递当前所有列
                    time_span=self.time_span,
                    time_index=self.time_index,
                    group_index=self.group_index
                    # max_candidate_features 在 feature_space 内部处理
                )
                # 记录已尝试生成（防止无限循环或重复尝试）
                already_selected_features.update(candidate_feature_names)
            except Exception as e:
                 if self.logger: self.logger.exception(f"生成候选特征名称时出错: {e}")

            if not candidate_feature_names:
                if self.logger: self.logger.warning(f"迭代 {iteration_num}: 未生成新的候选特征名称结束 AutoFE 迭代")
                break # 没有新候选，无法继续

            if self.logger: self.logger.info(f"迭代 {iteration_num}: 生成了 {len(candidate_feature_names)} 个候选特征名称")

            # 2. 快速特征筛选 (基于 Gini 的并行筛选)
            if self.logger: self.logger.debug("步骤 2: 使用多线程快速筛选候选特征 (Gini)...")
            selected_features_thread: List[str] = []
            # 移除 new_df_thread，因为后续需要重新计算
            features_scores: Dict[str, float] = {}
            try:
                 # 严格传递参数，使用当前 df_train_iter
                 selected_features_thread, _, features_scores = threads_feature_select(
                     df=df_train_iter,
                     target_name=self.target_name,
                     candidate_feature=candidate_feature_names, # 传递本轮生成的候选
                     transformer=self.transformer,
                     logger=self.logger,
                     metric='gini', # 固定使用 Gini
                     return_socre=True,
                     n_jobs=self.max_workers
                 )
                 if self.logger: self.logger.info(f"迭代 {iteration_num}: 线程筛选 (Gini) 选中 {len(selected_features_thread)} 个特征")
            except Exception as e:
                 if self.logger: self.logger.exception(f"线程特征筛选过程中出错: {e}")
                 selected_features_thread = [] # 出错则无选中

            # 备选逻辑 
            if not selected_features_thread and iteration_num < self.n:
                if self.logger: self.logger.warning(f"迭代 {iteration_num}: 线程筛选未选中特征，尝试备选策略...")
                if features_scores:
                    sorted_scores = sorted(features_scores.items(), key=lambda item: item[1], reverse=True)
                    # 只从本轮生成的 candidate_feature_names 中选择
                    fallback_candidates = [name for name, score in sorted_scores if name in candidate_feature_names]
                    fallback_selected = fallback_candidates[:20] # 最多选 20 个
                    if fallback_selected:
                         if self.logger: self.logger.info(f"迭代 {iteration_num}: 备选策略选中 {len(fallback_selected)} 个特征")
                         selected_features_thread = fallback_selected
                    # else: if self.logger: self.logger.warning(f"迭代 {iteration_num}: 备选策略也未选中任何特征")
                # else: if self.logger: self.logger.warning(f"迭代 {iteration_num}: 无分数信息，无法执行备选策略")

            if not selected_features_thread:
                 if self.logger: self.logger.warning(f"迭代 {iteration_num}: 没有特征通过快速筛选，跳过本轮模型选择")
                 continue # 进入下一轮迭代

            # 3. 计算选中的特征值 (在训练集和验证集上)
            if self.logger: self.logger.debug(f"步骤 3: 在训练集和验证集上计算 {len(selected_features_thread)} 个选中特征...")
            try:
                 # 确保在当前迭代的数据帧上添加新列
                 df_train_iter = max_threads_name2feature(
                     df=df_train_iter, feature_names=selected_features_thread,
                     transformer=self.transformer, logger=self.logger, n_jobs=self.max_workers
                 )
                 df_val_iter = max_threads_name2feature(
                     df=df_val_iter, feature_names=selected_features_thread,
                     transformer=self.transformer, logger=self.logger, n_jobs=self.max_workers
                 )
                 # 确保验证集的列与训练集匹配
                 missing_val_cols = set(df_train_iter.columns) - set(df_val_iter.columns)
                 if missing_val_cols:
                      self.logger.warning(f"验证集缺少列: {missing_val_cols}，将用 NaN 填充")
                      for col in missing_val_cols: df_val_iter[col] = np.nan
                 # 保留训练集的列顺序和存在性
                 df_val_iter = df_val_iter.reindex(columns=df_train_iter.columns)
            except Exception as e:
                 if self.logger: self.logger.exception(f"为模型选择计算特征时出错: {e}")
                 continue # 计算失败，跳过本轮迭代

            # 4. 模型特征选择
            if self.logger: self.logger.debug("步骤 4: 执行模型特征选择...")
            selected_features_model: List[str] = []
            current_score: float = np.inf if not self.higher_is_better else -np.inf

            # 准备模型选择的输入数据
            X_train_select = df_train_iter.drop(columns=[self.target_name], errors='ignore')
            y_train_select = df_train_iter[self.target_name]
            X_val_select = df_val_iter.drop(columns=[self.target_name], errors='ignore')
            y_val_select = df_val_iter[self.target_name]

            try:
                # 调用 model_features_select 函数
                selected_features_model, current_score = model_features_select(
                    fes=(X_train_select, X_val_select, y_train_select, y_val_select),
                    baseline=best_score, # 使用当前最佳分数
                    metric=self.metric,
                    task_type=self.task_type,
                    logger=self.logger,
                    seed=self.seed,
                    cat_features=self.cat_features, # 传递分类特征列表
                    time_index=self.time_index # 传递时间索引
                )
                # 提取高级特征
                selected_advanced_features = [f for f in selected_features_model if f not in original_feature_cols]

                if self.logger:
                    logger_score = 1.0 - current_score if self.higher_is_better else current_score
                    self.logger.info(f"迭代 {iteration_num}: 模型选择完成分数({self.metric}): {logger_score:.6f} (: {current_score:.6f})")
                    self.logger.info(f"迭代 {iteration_num}: 模型选中了 {len(selected_advanced_features)} 个高级特征")

                # 更新最佳分数和 pick_features
                score_improved = (self.higher_is_better and current_score > best_score) or \
                                 (not self.higher_is_better and current_score < best_score)
                if score_improved:
                    if self.logger: self.logger.info(f"迭代 {iteration_num}: 找到更好的分数 {current_score:.6f} (优于 {best_score:.6f})更新 pick_features")
                    best_score = current_score
                    pick_features = selected_advanced_features # 更新最佳特征集

                # 累积 adv_features (本轮模型选出的前 20 个新特征)
                features_to_add_to_adv = selected_advanced_features[:20]
                newly_added_count = len(adv_features.union(features_to_add_to_adv)) - len(adv_features)
                adv_features.update(features_to_add_to_adv)
                if self.logger: self.logger.debug(f"迭代 {iteration_num}: 更新 adv_features, 新增 {newly_added_count} 个, 总计 {len(adv_features)} 个")

                # 为下轮迭代准备数据 (只保留初始特征 + 本轮加入 adv 的特征 + 目标)
                cols_to_keep_next = original_feature_cols + features_to_add_to_adv
                cols_to_keep_next = sorted(list(set(cols_to_keep_next))) # 去重并排序

                df_train_iter = df_train_iter[[col for col in cols_to_keep_next if col in df_train_iter.columns] + [self.target_name]]
                df_val_iter = df_val_iter[[col for col in cols_to_keep_next if col in df_val_iter.columns] + [self.target_name]]
                # 再次确保验证集与训练集列一致
                df_val_iter = df_val_iter.reindex(columns=df_train_iter.columns)
                if self.logger: self.logger.debug(f"迭代 {iteration_num}: 为下轮迭代准备数据，保留 {len(df_train_iter.columns)-1} 个特征")

            except Exception as e:
                 if self.logger: self.logger.exception(f"模型特征选择过程中出错: {e}")
                 logger.error("模型选择失败，将使用上一轮的特征集进行下一轮迭代（如果未完成）")
                 # 保持 df_train_iter, df_val_iter 不变，进入下一轮或结束

        # --- 迭代结束后处理 ---
        if self.logger:
            self.logger.info("--- AutoFE 迭代完成 ---")
            self.logger.info(f"累积的潜在高级特征 (adv_features): {len(adv_features)} 个")
            self.logger.info(f"最佳分数 ({self.metric}): {best_score:.6f}")
            self.logger.info(f"对应最佳分数的特征 (pick_features): {len(pick_features)} 个")

        final_selected_features: List[str] = []
        # 最终模型选择 
        if adv_features:
            if self.logger: self.logger.info(f"对累积的 {len(adv_features)} 个 adv_features 进行最终模型选择...")
            adv_features_list = sorted(list(adv_features))
            # 需要在数据 X_train, X_val 上生成这些特征
            df_train_final = X_train.copy()
            df_val_final = X_val.copy()
            try:
                 df_train_final = max_threads_name2feature(df_train_final, adv_features_list, self.transformer, self.logger, self.max_workers)
                 df_val_final = max_threads_name2feature(df_val_final, adv_features_list, self.transformer, self.logger, self.max_workers)
                 df_val_final = df_val_final.reindex(columns=df_train_final.columns) # 保证列一致
            except Exception as e:
                 if self.logger: self.logger.exception(f"为最终选择生成 adv_features 时出错: {e}")
                 logger.warning("无法生成 adv_features，将回退到 pick_features")
                 final_selected_features = pick_features
                 adv_features = set() # 标记为无法使用

            if adv_features: # 仅当特征成功生成时执行
                X_train_final_select = df_train_final.drop(columns=[self.target_name], errors='ignore')
                X_val_final_select = df_val_final.drop(columns=[self.target_name], errors='ignore')
                common_cols_final = X_train_final_select.columns.intersection(X_val_final_select.columns)
                X_train_final_select = X_train_final_select[common_cols_final]
                X_val_final_select = X_val_final_select[common_cols_final]

                try:
                    # 最终选择与初始基线比较
                    final_selected_model_cols, final_score = model_features_select(
                        fes=(X_train_final_select, X_val_final_select, y_train, y_val), # 使用 y
                        baseline=self.base_score, # 与初始基线比较
                        metric=self.metric, task_type=self.task_type,
                        logger=self.logger, seed=self.seed,
                        cat_features=self.cat_features, time_index=self.time_index
                    )
                    final_selected_features = [f for f in final_selected_model_cols if f not in original_feature_cols]
                    if self.logger:
                        logger_score_final = 1.0 - final_score if self.higher_is_better else final_score
                        self.logger.info(f"最终模型选择完成分数({self.metric}): {logger_score_final:.6f} (: {final_score:.6f})")
                        self.logger.info(f"最终选定 {len(final_selected_features)} 个高级特征")
                except Exception as e:
                    if self.logger: self.logger.exception(f"最终模型选择过程中出错: {e}")
                    final_selected_features = []

                # 回退逻辑
                if not final_selected_features and pick_features:
                    if self.logger: self.logger.warning("最终模型选择未选中任何高级特征回退到最佳迭代结果 (pick_features)")
                    final_selected_features = pick_features
        else:
             if self.logger: self.logger.warning("没有累积的高级特征 (adv_features) 可供最终选择")
             final_selected_features = []

        # --- 准备最终返回的数据 ---
        if self.logger:
            self.logger.info(f"最终确定的高级特征列表 (共 {len(final_selected_features)} 个): {final_selected_features}")
            self.logger.info("正在准备最终的输出数据帧...")

        # 在 X_train, X_val 上生成最终确认的特征
        final_X_train = X_train.copy()
        final_X_val = X_val.copy()

        if final_selected_features:
             try:
                 final_X_train = max_threads_name2feature(final_X_train, final_selected_features, self.transformer, self.logger, self.max_workers)
                 final_X_val = max_threads_name2feature(final_X_val, final_selected_features, self.transformer, self.logger, self.max_workers)
                 # 确保列一致性和顺序
                 final_cols_order = original_feature_cols + final_selected_features
                 final_X_train = final_X_train[[col for col in final_cols_order if col in final_X_train.columns]]
                 final_X_val = final_X_val.reindex(columns=final_X_train.columns) # 与训练集对齐
             except Exception as e:
                 if self.logger: self.logger.exception(f"为最终输出生成特征时出错: {e}")
                 final_X_train = X_train.copy()
                 final_X_val = X_val.copy()
                 final_selected_features = []
                 logger.error("无法为最终输出生成高级特征，将返回特征")

        # 打印最终日志（模拟 ml.py 中的日志）
        # 这个日志应该由调用 AutoFE 的地方打印，但为了完整性在此处模拟
        if self.logger:
            self.logger.info(f"autofe finished, search {len(final_selected_features)} features")


        if self.logger: self.logger.success("AutoFE fit 过程成功结束")

        # 严格按照返回格式
        return final_X_train, final_X_val, y_train, y_val, final_selected_features