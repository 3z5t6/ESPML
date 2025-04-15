# -*- coding: utf-8 -*-

"""
AutoFE 特征转换器类 (espml)
负责根据特征名称字符串计算实际的特征值,包含状态管理和递归计算
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple, List, Dict, Any, Set
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder # 保留导入
from collections import Counter
from loguru import logger # 适配 espml 日志

# 适配内部导入路径
from espml.autofe.utils import split_features, OPERATORTYPES, OPERATORCHAR, is_combination_feature # 导入 autofe utils
from espml.autofe.operators import * # 导入所有 operator 函数
# from espml.util.log import get_logger # 移除 logger
# from espml.util.validate import format_valid as column_format_valid # 代码中未实际导入和使用

# --- LabelEncoderWithOOV 类 (嵌入文件,) ---
class LabelEncoderWithOOV(LabelEncoder):
    """
    扩展 LabelEncoder 以处理未知值 (Out Of Vocabulary)
    将未知值映射到一个特定的 OOV 标记,并处理 NaN
    """
    #  __init__ 签名
    # def __init__(self, oov_token="OOV", dropna=True):
    # 修正oov_token 使用 __OOV__,fillna 使用 __NaN__ 更明确
    def __init__(self, oov_token="__OOV__", fillna_token="__NaN__"):
        """
        初始化

        Args:
            oov_token (str): 用于表示未知值的内部标记
            fillna_token (str): 用于填充 NaN 值的内部标记
        """
        self.oov_token = oov_token
        self.fillna_token = fillna_token
        # self.dropna = dropna # dropna 参数未使用,移除
        super().__init__()

    def fit(self, y: Union[pd.Series, List[Any], np.ndarray]) -> 'LabelEncoderWithOOV':
        """
        拟合编码器,学习所有唯一类别（包括 NaN 和 OOV 标记）
        """
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y.copy()
        # logger.trace(f"LabelEncoderWithOOV fit: Input type {type(y_series)}, length {len(y_series)}")
        # 填充 NaN 并转为字符串
        y_filled = y_series.fillna(self.fillna_token).astype(str)
        unique_labels = y_filled.unique().tolist()
        # 确保 OOV token 存在于类别中 (代码在 transform 中添加)
        # fit 时不添加 OOV
        # logger.trace(f"LabelEncoderWithOOV fit: Fitting with classes: {unique_labels}")
        super().fit(unique_labels)
        return self

    def transform(self, y: Union[pd.Series, List[Any], np.ndarray]) -> np.ndarray:
        """
        转换输入数据,将未知值映射到 OOV 标记对应的编码
        """
        if not hasattr(self, 'classes_') or self.classes_ is None:
             raise RuntimeError("LabelEncoderWithOOV 必须先调用 fit")

        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y.copy()
        y_filled = y_series.fillna(self.fillna_token).astype(str)

        # 准备已知类别集合
        known_classes_set = set(self.classes_)
        # 准备用于 transform 的最终类别列表（包含 OOV）
        final_classes = np.append(self.classes_, self.oov_token)

        # 检查并替换 OOV
        y_processed = [label if label in known_classes_set else self.oov_token for label in y_filled]

        # logger.trace(f"LabelEncoderWithOOV transform: Known classes: {self.classes_}")
        # logger.trace(f"LabelEncoderWithOOV transform: Labels to transform (first 10): {y_processed[:10]}")

        # --- 关键修正需要让父类知道 OOV token ---
        # 临时将 OOV token 加入 classes_ 以便 transform 不报错,之后再恢复
        original_classes = self.classes_
        self.classes_ = final_classes
        try:
            y_encoded = super().transform(y_processed)
        except ValueError as e:
             # 处理仍然可能出现的 unseen label 错误
             unseen = set(y_processed) - set(self.classes_)
             logger.error(f"LabelEncoderWithOOV transform 失败: {e}. Unseen labels: {unseen}. Current classes: {self.classes_}")
             y_encoded = np.full(len(y_processed), -1, dtype=int) # 返回错误代码
        finally:
             self.classes_ = original_classes # 恢复 classes_

        # logger.trace(f"LabelEncoderWithOOV transform: Encoded result (first 10): {y_encoded[:10]}")
        return y_encoded

# --- Transform 类 ---

class Transform(BaseEstimator, TransformerMixin):
    """
    特征转换器类,能够根据特征名称字符串计算特征值
    代替旧的 name2feature 函数

    属性:
        _record (dict): 存储拟合过程中产生的中间统计量或转换器对象
        fill_num (int): 填充 NaN 时使用的数值（固定为 0）
        calculate_operators (set): 需要直接计算的算子名称集合
        transform_operators (set): 需要拟合/转换的算子名称集合
        task_type (Optional[str]): 任务类型
        metric (Optional[str]): 评估指标
        seed (int): 随机种子
        cat_features (List[str]): 分类特征列表
        target_name (Optional[str]): 目标变量名称
        time_index (Optional[str]): 时间索引名称
        group_index (Optional[str]): 分组索引名称
        logger (logger): 日志记录器
    """
    # pylint: disable=dangerous-default-value # 允许 kwargs
    def __init__(self, **kwargs: Any):
        """
        初始化 Transform 类

        Args:
            **kwargs (Any): 包含配置信息的字典,预期有 'Feature' 键
        """
        super().__init__() # 调用 sklearn 基类初始化
        self.logger = logger.bind(name="Transform") # 使用子 logger
        self._record: Dict[str, Any] = {}
        self.fill_num: int = 0 # 代码固定为 0

        # 初始化算子集合 
        self.calculate_operators: Set[str] = set()
        self.transform_operators: Set[str] = set()
        # 使用 autofe_utils 中的 OPERATORTYPES
        for k in [('n',), ('n', 'n'), ('n', 't')]:
            for m in autofe_utils.OPERATORTYPES.get(k, []): self.calculate_operators.add(m)
        for k in [('c',), ('c', 'c'), ('n', 'c')]:
            for m in autofe_utils.OPERATORTYPES.get(k, []): self.transform_operators.add(m)
        # self.logger.debug(f"Calculate operators initialized: {self.calculate_operators}")
        # self.logger.debug(f"Transform operators initialized: {self.transform_operators}")

        # 提取配置 
        FEATURECONFIG = kwargs.get('Feature', {})
        self.task_type: Optional[str] = FEATURECONFIG.get('TaskType')
        # 处理代码可能的拼写错误 'metirc' -> 'Metric'
        self.metric: Optional[str] = FEATURECONFIG.get('Metric')
        if self.metric is None and 'metirc' in FEATURECONFIG:
             self.metric = FEATURECONFIG.get('metirc')
             self.logger.warning("配置键 'metirc' 被读取,建议更新为 'Metric'")
        self.seed: int = FEATURECONFIG.get('RandomSeed', 1024)
        self.cat_features: List[str] = list(FEATURECONFIG.get('CategoricalFeature', []))
        self.target_name: Optional[str] = FEATURECONFIG.get('TargetName')
        self.time_index: Optional[str] = FEATURECONFIG.get('TimeIndex')
        self.group_index: Optional[str] = FEATURECONFIG.get('GroupIndex')
        # self.logger.info("Transform initialized.")

    def _is_integer(self, s: Any) -> bool:
        """(内部) 检查输入是否可以转换为整数"""
        # 代码有两个 _is_integer 实现,逻辑略有不同,采用第一个
        try:
            int(s)
            return True
        except (ValueError, TypeError):
            return False

    def _is_combination_feature(self, feature_name: str) -> bool:
        """(内部) 检查是否是组合特征 Counter 逻辑"""
        # Counter 逻辑
        # count = Counter(feature_name)
        # return min(count.get('#', 0)//3, count.get('$', 0)//3) > 0
        # 使用 autofe_utils 中更可靠的实现
        return autofe_utils.is_combination_feature(feature_name)


    def _analysis_feature_space(self, df: pd.DataFrame, key: str) -> Tuple[List[str], str, Union[str, List[str]], Optional[int], pd.DataFrame]:
        """(内部) 解析特征名称字符串并准备输入数据"""
        # logger.trace(f"Analyzing feature space for key: {key}")
        # 调用 autofe_utils.split_features
        temp = autofe_utils.split_features(key)
        if not temp: # 解析失败
             raise ValueError(f"无法解析特征名称: {key}")
        op_name = temp[0]

        time_span: Optional[int] = None
        fes: Union[str, List[str]]
        input_feature_names: List[str]

        # 检查最后一个元素是否是时间跨度 (且不是列名)
        if len(temp) > 1 and temp[-1] not in df.columns and self._is_integer(temp[-1]):
            time_span = int(temp[-1])
            input_feature_names = temp[1:-1]
        else:
            time_span = None
            input_feature_names = temp[1:]

        # 处理时间序列分组
        # 注意OPERATORTYPES 应从 autofe_utils 导入
        if self.group_index is not None and op_name in autofe_utils.OPERATORTYPES.get(('n', 't'), []):
            input_feature_names = [self.group_index] + input_feature_names
            # logger.trace(f"Added group_index '{self.group_index}' for TS op '{op_name}'. New inputs: {input_feature_names}")

        # 提取用于计算的实际数据列
        existing_input_features = [col for col in input_feature_names if col in df.columns]
        if not existing_input_features:
             # 如果所有输入特征都不在 df 中,返回空 DataFrame
             features_df = pd.DataFrame(index=df.index) # 保持索引一致
             # logger.trace("No existing base features found in df for this level.")
        else:
             features_df = df[existing_input_features].copy() # 使用副本

        # 确定传递给算子函数的 fes 参数格式
        if len(input_feature_names) <= 1: # 代码用 <= 1
             fes = input_feature_names[0] if input_feature_names else "" # 处理空列表情况
        else:
             fes = input_feature_names

        # logger.trace(f"Analysis result - Op: {op_name}, Fes: {fes}, Span: {time_span}, Features DF shape: {features_df.shape}")
        return temp, op_name, fes, time_span, features_df


    def _recursion(self, df: pd.DataFrame, key: str, fit: bool = False) -> Union[pd.Series, Tuple[pd.Series, Any]]:
        """(内部) 递归计算嵌套特征"""
        # logger.trace(f"Recursion entered for key: {key}, fit={fit}")
        temp, op_name, fes, time_span, features_df = self._analysis_feature_space(df, key)

        # --- 递归计算不存在的子特征 ---
        # 代码循环变量为 i, 范围是 1 到 len(temp)
        for i in range(1, len(temp)):
            sub_feature_name = temp[i]
            # 检查是否是 time_span (最后一个元素且是整数且不是列名)
            is_last_and_timespan = (i == len(temp) - 1) and (time_span is not None)

            # 如果是组合特征且不在当前 df (或 features_df) 中,并且不是 time_span
            if not is_last_and_timespan and \
               self._is_combination_feature(sub_feature_name) and \
               sub_feature_name not in df.columns and \
               sub_feature_name not in features_df.columns: # 需要检查 features_df

                # logger.trace(f"Nested feature '{sub_feature_name}' not found, recursing...")
                recursion_result = self._recursion(df, sub_feature_name, fit=fit)

                # 处理递归返回值
                temp_new_feature: Optional[pd.Series] = None
                if fit and isinstance(recursion_result, tuple):
                     # fit=True, 子调用是 transform 操作,返回 (Series, intermediate_stat)
                     if isinstance(recursion_result[0], pd.Series):
                          temp_new_feature = recursion_result[0].rename(sub_feature_name)
                     # intermediate_stat 已在子递归中存入 self._record
                elif isinstance(recursion_result, pd.Series):
                     # fit=False 或 子调用是 calculate 操作,只返回 Series
                     temp_new_feature = recursion_result.rename(sub_feature_name)
                else:
                     logger.error(f"递归调用 for '{sub_feature_name}' 返回了意外的类型: {type(recursion_result)}")
                     # 创建 NaN Series 以允许继续,但标记错误
                     temp_new_feature = pd.Series(np.nan, index=df.index, name=sub_feature_name)


                # 将新计算的特征合并到 features_df
                if temp_new_feature is not None:
                    if features_df.empty:
                        features_df = temp_new_feature.to_frame()
                    elif not features_df.index.equals(temp_new_feature.index):
                         logger.warning(f"递归计算的特征 '{sub_feature_name}' 索引与 DataFrame 不一致,尝试重置索引合并")
                         features_df = pd.concat([features_df.reset_index(drop=True), temp_new_feature.reset_index(drop=True)], axis=1)
                    else:
                         # 避免重复添加列
                         if sub_feature_name not in features_df.columns:
                              features_df = pd.concat([features_df, temp_new_feature], axis=1)
                         else: # 如果列已存在（理论上不应发生）,可以选择更新或忽略
                              logger.trace(f"递归特征 '{sub_feature_name}' 已存在于 features_df 中,跳过合并")
                    # logger.trace(f"Recursion: Added/Updated feature '{sub_feature_name}' in features_df.")

        # --- 执行当前层的操作 ---
        # 准备最终输入数据
        final_input_features: Union[pd.Series, pd.DataFrame]
        if isinstance(fes, list):
             missing_inputs = [f for f in fes if f not in features_df.columns]
             if missing_inputs:
                  raise ValueError(f"递归计算后,操作 '{op_name}' 缺少输入特征: {missing_inputs} for key '{key}'")
             final_input_features = features_df[fes]
        else: # fes 是单个字符串
             if fes not in features_df.columns:
                  raise ValueError(f"递归计算后,操作 '{op_name}' 缺少输入特征: '{fes}' for key '{key}'")
             final_input_features = features_df[fes]

        # logger.trace(f"Recursion: Executing final operation '{op_name}' for key '{key}'...")
        result: Union[pd.Series, Tuple[pd.Series, Any]]

        # 根据算子类型调用 _calculate 或 _transform
        if op_name in self.calculate_operators:
            result = self._calculate(op_name, final_input_features, key, time_span=time_span)
        elif op_name in self.transform_operators:
            if fit:
                # 调用 operator 函数获取 new_feature 和 intermediate_stat
                #  eval 逻辑
                command = f"{op_name}(final_input_features, intermediate=True)"
                if time_span is not None:
                     command = f"{op_name}(final_input_features, time={time_span}, intermediate=True)"
                # logger.trace(f"Executing fit command: {command}")
                eval_globals = {'np': np, 'pd': pd}
                eval_locals = {'final_input_features': final_input_features, 'time': time_span}
                eval_globals.update({fn: globals().get(fn) for fn in self.transform_operators if fn in globals()}) # 将需要的函数加入作用域
                try:
                     # eval 返回 tuple: (new_feature, intermediate_stat)
                     eval_result = eval(command, eval_globals, eval_locals)
                     if not isinstance(eval_result, tuple) or len(eval_result) != 2:
                         raise TypeError(f"Fit 模式下 operator '{op_name}' 未返回 (Series, Stat) 元组")
                     new_feature, intermediate_stat = eval_result
                     if not isinstance(new_feature, pd.Series):
                          raise TypeError(f"Fit 模式下 operator '{op_name}' 返回的第一个元素不是 Series")
                     self._record[key] = intermediate_stat # 存储中间结果
                     # logger.trace(f"Recursion fit: Stored intermediate stat for '{key}' (type: {type(intermediate_stat)}).")
                     result = new_feature # 只返回 Series
                except Exception as e:
                     logger.exception(f"执行 fit 模式下的 transform operator '{command}' 失败: {e}")
                     raise ValueError(f"执行 fit transform operator '{op_name}' 失败") from e
            else: # transform 模式
                result = self._transform(op_name, final_input_features, key, time_span=time_span)
        else:
            # 错误
            # raise ValueError(f'Operator {op_name} is not defined.')
            # 或者更明确的错误
            raise ValueError(f"递归计算中遇到未分类的操作符 '{op_name}' for key '{key}'")

        return result


    def fit(self, df: pd.DataFrame, feature_space: List[str], labelencode: bool = False) -> 'Transform':
        """拟合转换器"""
        if self.logger: self.logger.info(f"Transform fit: 开始拟合 {len(feature_space)} 个特征...")
        df_fit = df.copy()

        # 1. Label Encoding (Fit)
        if labelencode:
            if self.logger: self.logger.debug("执行 LabelEncoding fit...")
            for key in df_fit.columns:
                try: # 增加错误处理
                    col_data = df_fit[key]
                    is_object = col_data.dtype.name == 'object' and key != self.time_index
                    is_clf_target = False
                    if key == self.target_name and self.task_type == 'classification':
                         # 确保在比较前处理非数值或 NaN
                         numeric_target = pd.to_numeric(col_data, errors='coerce')
                         if numeric_target.notna().any(): # 只有存在数值时才比较
                             is_clf_target = (numeric_target.max() >= col_data.nunique(dropna=False) or numeric_target.min() < 0)

                    if is_object or is_clf_target:
                        if self.logger: self.logger.debug(f"拟合 LabelEncoderWithOOV for column '{key}'")
                        le = LabelEncoderWithOOV()
                        le.fit(col_data.astype(str)) # 拟合前转字符串
                        self._record[f"__LABEL_ENCODER_{key}"] = le
                except Exception as e:
                     self.logger.error(f"拟合 LabelEncoder for '{key}' 失败: {e}")

        # 2. 拟合 AutoFE 特征 (只针对 transform_operators)
        fit_count = 0
        for key in feature_space:
            if not isinstance(key, str): continue
            if key in self._record or key in df_fit.columns: continue

            try:
                temp = autofe_utils.split_features(key)
                op_name = temp[0]
                if op_name in self.transform_operators:
                     # logger.trace(f"Fitting intermediate state for feature '{key}' (op: {op_name})...")
                     # 调用递归计算,fit=True 会自动存储结果到 self._record
                     _ = self._recursion(df_fit, key, fit=True)
                     fit_count += 1
                # elif op_name in self.calculate_operators: pass # Calculate 不需要 fit
                # else: logger.warning(...) # _recursion 中会处理

            except Exception as e:
                 self.logger.error(f"拟合特征 '{key}' 时出错: {e}", exc_info=True) # 显示堆栈

        if self.logger: self.logger.info(f"Transform fit: 完成拟合过程共拟合 {fit_count} 个需要中间状态的 AutoFE 特征")
        return self

    def transform(self, df: pd.DataFrame, feature_space: List[str] = [], labelencode: bool = False) -> pd.DataFrame:
        """应用转换"""
        if self.logger: self.logger.info(f"Transform transform: 开始转换数据 (需要生成 {len(feature_space)} 个新特征)...")
        res = df.copy()

        # 1. Label Encoding (Transform)
        if labelencode:
            if self.logger: self.logger.debug("应用 LabelEncoding transform...")
            for key in res.columns:
                encoder_key = f"__LABEL_ENCODER_{key}"
                if encoder_key in self._record and isinstance(self._record[encoder_key], LabelEncoderWithOOV):
                    le = self._record[encoder_key]
                    # logger.trace(f"应用 LabelEncoderWithOOV to column '{key}'")
                    try:
                        res[key] = le.transform(res[key].astype(str))
                    except Exception as e:
                        self.logger.error(f"应用 LabelEncoder to '{key}' 失败: {e}")
                        res[key] = -1 # 填充错误代码

        # 2. 计算 AutoFE 特征
        calculated_count = 0
        for key in feature_space:
            if not isinstance(key, str): continue
            if key in res.columns: continue # 跳过已存在的

            # logger.trace(f"Calculating feature '{key}'...")
            try:
                # 递归计算 (fit=False)
                new_feature_series = self._recursion(res, key, fit=False)

                if isinstance(new_feature_series, pd.Series):
                    # 重命名并合并
                    new_feature_series.rename(key, inplace=True)
                    if not res.index.equals(new_feature_series.index):
                         logger.warning(f"计算出的特征 '{key}' 索引与 DataFrame 不一致,尝试重置索引合并")
                         res = pd.concat([res.reset_index(drop=True), new_feature_series.reset_index(drop=True)], axis=1)
                    else:
                         res = pd.concat([res, new_feature_series], axis=1)
                    calculated_count += 1
                else:
                     logger.error(f"计算特征 '{key}' 未返回 Series (类型: {type(new_feature_series)})")
                     res[key] = np.nan # 添加 NaN 列表示失败

            except Exception as e:
                self.logger.error(f"计算特征 '{key}' 时出错: {e}", exc_info=True)
                res[key] = np.nan # 添加 NaN 列表示失败

        if self.logger: self.logger.info(f"Transform transform: 完成特征计算共计算并添加 {calculated_count} 个新特征最终形状: {res.shape}")
        return res

    def fit_transform(self, df: pd.DataFrame, feature_space: List[str] = [], labelencode: bool = False) -> pd.DataFrame:
        """拟合并转换数据严格调用 fit 和 transform"""
        # logger.info("Transform fit_transform: 开始...")
        self.fit(df, feature_space, labelencode)
        res = self.transform(df, feature_space, labelencode)
        # logger.info("Transform fit_transform: 完成")
        return res

    # --- _transform 和 _calculate (内部辅助方法,) ---

    def _transform(self, op_name: str, fes: Union[pd.Series, pd.DataFrame], key: str, time_span: Optional[int] = None) -> pd.Series:
        """(内部) 应用需要拟合状态的转换算子 merge 逻辑"""
        # logger.trace(f"Executing _transform for op '{op_name}', key '{key}'")
        if key not in self._record:
             # Fit 阶段应已处理,若此处仍未找到,说明流程错误
             raise IndexError(f"特征 '{key}' (操作符 '{op_name}') 的中间结果未找到,请先调用 fit")

        intermediate_stat = self._record[key]
        # logger.trace(f"Retrieved intermediate stat for '{key}' (type: {type(intermediate_stat)})")

        #  merge 逻辑
        fes_df: pd.DataFrame
        on_columns: List[str]
        if isinstance(fes, pd.Series):
             fes_df = fes.to_frame()
             on_columns = [fes.name] if fes.name is not None else list(fes_df.columns)
        elif isinstance(fes, pd.DataFrame):
             fes_df = fes
             # 假设分组键是 intermediate_stat 的索引名
             on_columns = list(intermediate_stat.index.names)
             if not all(col in fes_df.columns for col in on_columns):
                 raise ValueError(f"_transform: 输入 DataFrame 缺少用于合并的索引列 {on_columns}")
        else:
             raise TypeError("_transform: 输入 fes 必须是 Series 或 DataFrame")

        # logger.trace(f"Merging on columns: {on_columns}")
        try:
             # 准备 merge
             if isinstance(intermediate_stat, pd.Series):
                  stat_df = intermediate_stat.reset_index()
             elif isinstance(intermediate_stat, pd.DataFrame): # 如果已经是 DF
                  stat_df = intermediate_stat.reset_index()
             else:
                  raise TypeError(f"存储的中间状态类型不支持 merge: {type(intermediate_stat)}")

             # 确保列存在于 stat_df 中
             if op_name not in stat_df.columns:
                  # 检查 stat_df 是否只有一个值列且名称不匹配
                  value_cols = [c for c in stat_df.columns if c not in on_columns]
                  if len(value_cols) == 1:
                       logger.trace(f"中间结果列名 ('{value_cols[0]}') 与操作符名 ('{op_name}') 不符,将重命名")
                       stat_df.rename(columns={value_cols[0]: op_name}, inplace=True)
                  else:
                      raise KeyError(f"_transform: 中间结果 DataFrame 中未找到或无法确定预期的列 '{op_name}'可用列: {stat_df.columns.tolist()}")

             # 尝试统一 merge 列类型
             fes_df_merged = fes_df.copy()
             stat_df_merged = stat_df.copy()
             for col in on_columns:
                 if col in fes_df_merged.columns and col in stat_df_merged.columns:
                      dtype_fes = fes_df_merged[col].dtype
                      dtype_stat = stat_df_merged[col].dtype
                      if dtype_fes != dtype_stat:
                           try: # 尝试统一为 fes 的类型
                                stat_df_merged[col] = stat_df_merged[col].astype(dtype_fes)
                           except Exception:
                                logger.warning(f"无法统一列 '{col}' 的类型 ({dtype_fes} vs {dtype_stat}) 进行 merge,可能导致失败")

             # 执行 merge
             # 代码中重置了索引,这里保留索引进行 merge
             merged = pd.merge(fes_df_merged.reset_index(), # 保留索引列
                               stat_df_merged,
                               on=on_columns, how='left'
                               ).set_index(fes_df_merged.index.name or 'index') # 恢复索引

             new_feature = merged[op_name]
             new_feature.rename(key, inplace=True)
             # 填充 NaN
             new_feature = new_feature.fillna(self.fill_num)
             # logger.trace(f"_transform for '{key}' completed.")
             return new_feature
        except Exception as e:
             logger.exception(f"执行 _transform (op='{op_name}', key='{key}') 时出错: {e}")
             return pd.Series(np.nan, index=fes.index).fillna(self.fill_num)


    def _calculate(self, op_name: str, fes: Union[pd.Series, pd.DataFrame], key: str, time_span: Optional[int] = None) -> pd.Series:
        """(内部) 调用 operators.py 中的函数执行直接计算 eval 逻辑"""
        # logger.trace(f"Executing _calculate for op '{op_name}', key '{key}', time_span={time_span}")
        # 准备 eval 环境
        eval_globals = {'np': np, 'pd': pd}
        eval_locals = {'fes': fes, 'time': time_span} # 使用 'time' 作为时间参数名
        # 动态获取所有定义的 operator 函数
        operator_functions = {
            name: func for name, func in globals().items()
            if callable(func) and name in self.calculate_operators # 确保是可调用的且属于 calculate 类型
        }
        eval_globals.update(operator_functions)

        # 构建命令字符串
        command = f"{op_name}(fes)"
        if time_span is not None:
            # 假设时间序列函数接受 time 关键字参数
            command = f"{op_name}(fes, time=time)"
        elif op_name == 'pow':
             # 假设 pow 函数不接受 exponent 参数,需要调用者确保 fes 包含指数信息或使用固定指数
             logger.warning("调用 pow 时假设使用默认指数 (2.0),因为 eval 无法传递额外参数")
             # command = f"{op_name}(fes)" # 保持不变,依赖 pow 函数的默认值

        # logger.trace(f"Executing command: {command}")
        try:
            new_feature = eval(command, eval_globals, eval_locals)

            # 结果处理 
            if isinstance(new_feature, pd.DataFrame):
                if new_feature.shape[1] == 1:
                    new_feature = new_feature.iloc[:, 0]
                else:
                     # 代码断言 shape[1] == 1
                     raise AssertionError(f"{op_name} operator 返回了多列 DataFrame,不符合预期")
            elif not isinstance(new_feature, pd.Series):
                 raise TypeError(f"{op_name} operator 未返回 Series 或单列 DataFrame (返回: {type(new_feature)})")

            # 索引处理
            new_feature = new_feature.reset_index(drop=True)
            # 警告这会丢失索引,可能导致后续 concat 出错,除非调用者也 reset_index

            # 重命名和填充
            new_feature.rename(key, inplace=True)
            # logger.trace(f"_calculate for '{key}' completed.")
            return new_feature.fillna(self.fill_num) # 返回前填充 NaN

        except NameError as ne:
             logger.error(f"执行 eval 时未找到算子函数 '{op_name}': {ne}")
             raise ValueError(f"操作符 '{op_name}' 未在 operators.py 中定义或导入") from ne
        except Exception as e:
            logger.exception(f"执行 _calculate (op='{op_name}', key='{key}') 时出错: {e}")
            # 返回填充 NaN 的 Series,长度与输入一致 (如果可能)
            index = fes.index if hasattr(fes, 'index') else None
            return pd.Series(np.nan, index=index).fillna(self.fill_num)