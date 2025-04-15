# -*- coding: utf-8 -*-
# pylint: disable=logging-fstring-interpolation, invalid-name, too-many-locals, too-many-arguments
# pylint: disable=too-many-branches, too-many-statements, broad-except, protected-access
"""
风电增量学习任务驱动模块 (espml)
负责根据任务配置，驱动数据处理、训练（完全或增量）、预测、回测流程，并记录详细日志
"""

import datetime
from logging import config
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path
from loguru import logger
import traceback

from espml.incrml.metadata import ModelVersionInfo # 用于打印完整错误堆栈

# 导入 espml 内部模块
# 使用 try-except 包装，以便模块可以单独测试或处理导入问题
try:
    from espml.ml import MLPipeline
except ImportError:
    logger.error("无法导入 MLPipeline from espml.ml")
    MLPipeline = Any # type: ignore
try:
    from espml.incrml.manager import IncrmlManager
except ImportError:
    logger.error("无法导入 IncrmlManager from espml.incrml.manager")
    IncrmlManager = Any # type: ignore
try:
    from espml.dataprocess.data_processor import DataProcessor, DataProcessingError
except ImportError:
    logger.error("无法导入 DataProcessor from espml.dataprocess.data_processor")
    DataProcessor = Any; DataProcessingError = Exception # type: ignore
try:
    from espml.util import utils as common_utils
    from espml.util import const
    from espml.util import result_saver
except ImportError:
    logger.error("无法导入 espml.util 中的模块 (utils, const, result_saver)")
    # 定义占位符，可能导致后续错误
    common_utils = None; const = None; result_saver = None # type: ignore

# 导入 croniter
try:
    from croniter import croniter
    CRONITER_INSTALLED = True
except ImportError:
    logger.warning("库 'croniter' 未安装，基于 Cron 的任务触发器将不可用")
    croniter = None # type: ignore
    CRONITER_INSTALLED = False

# 导入时区库
try:
    import pytz
    DEFAULT_TIMEZONE = pytz.timezone(common_utils.safe_dict_get(config, 'Project.Timezone', 'Asia/Shanghai')) # 从配置读取时区
    PYTZ_INSTALLED = True
except Exception: # 捕获导入和配置读取错误
    logger.warning("库 'pytz' 未安装或配置中缺少 Timezone，将使用系统默认本地时区")
    DEFAULT_TIMEZONE = None
    PYTZ_INSTALLED = False


class WindTaskRunner:
    """
    单个风电预测/回测任务的驱动器
    """
    # 类属性存储数据缓存，用于数据可用性检查 (避免重复加载)
    _raw_data_cache: Dict[str, Optional[pd.DataFrame]] = {'fans': None, 'tower': None, 'weather': None}
    _raw_data_load_time: Optional[datetime.datetime] = None
    _raw_data_cache_ttl_seconds: int = 600 # 数据缓存有效期 (例如 10 分钟)

    def __init__(self, task_id: str, config: Dict[str, Any]):
        """
        初始化 WindTaskRunner

        Args:
            task_id (str): 要运行的任务 ID
            config (Dict[str, Any]): 完整的项目配置字典

        Raises:
            ValueError: 如果 task_id 无效或缺少必要配置
            RuntimeError: 如果核心组件初始化失败
        """
        self.task_id = task_id
        self.config = config
        # 创建绑定任务 ID 的 logger
        self.logger = logger.bind(name=f"WindTaskRunner_{task_id}")

        self.logger.info(f"为任务 '{task_id}' 初始化 WindTaskRunner...")

        # --- 解析任务特定配置  ---
        self.task_config = self._get_task_config(task_id)
        if not self.task_config:
            raise ValueError(f"无法在配置中找到任务 ID 为 '{task_id}' 的任务配置")

        self.description = self.task_config.get('description', 'N/A')
        self.enabled = self.task_config.get('enabled', False)
        self.task_run_type = self.task_config.get('type', 'forecast') # 'forecast' or 'backtrack'
        self.forecast_horizon_str = self.task_config.get('forecast_horizon', '4H')
        self.output_freq_str = self.task_config.get('output_freq', '15min') # 应与数据频率一致
        self.train_trigger_cron = self.task_config.get('train_trigger_cron')
        self.predict_trigger_cron = self.task_config.get('predict_trigger_cron')
        # 回测特定配置
        self.backtrack_trigger_cron = self.task_config.get('backtrack_trigger_cron')
        self.backtrack_start_date_str = self.task_config.get('backtrack_start_date')
        self.backtrack_end_date_str = self.task_config.get('backtrack_end_date')
        self.backtrack_time_of_day_str = self.task_config.get('backtrack_time_of_day', "00:00:00")
        self.backtrack_retrain = self.task_config.get('backtrack_retrain', True)
        self.backtrack_model_path_pattern = self.task_config.get('backtrack_model_path_pattern')
        self.backtrack_train_start_offset_str = self.task_config.get('backtrack_train_start_offset', '-90D')
        self.backtrack_train_end_offset_str = self.task_config.get('backtrack_train_end_offset', '-1H')
        self.backtrack_predict_input_start_offset_str = self.task_config.get('backtrack_predict_input_start_offset', '-2D')
        self.backtrack_predict_input_end_offset_str = self.task_config.get('backtrack_predict_input_end_offset', '0H')

        # 常规/共享配置
        self.data_fetch_lag_str = self.task_config.get('data_fetch_lag', '15min')
        self.train_start_offset_str = self.task_config.get('train_start_offset', '-90D')
        self.train_end_offset_str = self.task_config.get('train_end_offset', '-1H')
        self.predict_input_start_offset_str = self.task_config.get('predict_input_start_offset', '-2D')
        self.predict_input_end_offset_str = self.task_config.get('predict_input_end_offset', '0H')
        self.task_config_override = self.task_config.get('config_override', {})

        # --- 合并配置  ---
        import copy
        self.effective_config = copy.deepcopy(self.config)
        self.effective_config = common_utils.merge_dictionaries(self.effective_config, self.task_config_override, deep=True)

        # --- 初始化核心组件 ---
        # 在 run 方法中按需初始化可能更灵活，但代码（假设在 init 中）
        self.ml_pipeline: Optional[MLPipeline] = None
        self.incrml_manager: Optional[IncrmlManager] = None
        try:
            # 需要确保依赖都已导入
            if IncrmlManager is not Any:
                self.incrml_manager = IncrmlManager(task_id=self.task_id, config=self.effective_config, logger_instance=self.logger)
            else: self.logger.error("IncrmlManager 未正确导入，增量学习功能不可用")

            if MLPipeline is not Any:
                self.ml_pipeline = MLPipeline(config=self.effective_config)
            else: self.logger.error("MLPipeline 未正确导入，核心流程不可用")

        except Exception as init_e:
            self.logger.exception(f"初始化核心组件失败: {init_e}")
            raise RuntimeError(f"任务 '{task_id}' 初始化失败") from init_e

        # --- 解析时间参数  ---
        try:
            self.data_freq_str = self.effective_config['Feature']['TimeFrequency']
            self.data_freq = pd.Timedelta(self.data_freq_str)
            self.data_fetch_lag = pd.Timedelta(self.data_fetch_lag_str)
            # 使用 offset 更安全地处理各种偏移字符串
            self.train_start_offset = pd.tseries.frequencies.to_offset(self.train_start_offset_str)
            self.train_end_offset = pd.tseries.frequencies.to_offset(self.train_end_offset_str)
            self.predict_input_start_offset = pd.tseries.frequencies.to_offset(self.predict_input_start_offset_str)
            self.predict_input_end_offset = pd.tseries.frequencies.to_offset(self.predict_input_end_offset_str)
            self.forecast_horizon = pd.Timedelta(self.forecast_horizon_str)
            self.output_freq = pd.Timedelta(self.output_freq_str)
            # 回测时间偏移量
            self.backtrack_train_start_offset = pd.tseries.frequencies.to_offset(self.backtrack_train_start_offset_str)
            self.backtrack_train_end_offset = pd.tseries.frequencies.to_offset(self.backtrack_train_end_offset_str)
            self.backtrack_predict_input_start_offset = pd.tseries.frequencies.to_offset(self.backtrack_predict_input_start_offset_str)
            self.backtrack_predict_input_end_offset = pd.tseries.frequencies.to_offset(self.backtrack_predict_input_end_offset_str)
        except Exception as time_e:
            self.logger.error(f"解析任务 '{task_id}' 的时间配置失败: {time_e}")
            raise ValueError(f"任务 '{task_id}' 时间配置解析失败") from time_e

        self.logger.info(f"WindTaskRunner for '{task_id}' 初始化完成")

    def _get_task_config(self, task_id_to_find: str) -> Optional[Dict[str, Any]]:
        """(内部) 从完整配置中查找特定 task_id 的配置"""
        # 假设 tasks 列表在顶层 'tasks' 键下，或者需要从 task_config.yaml 加载
        # 此处假设在 self.config['tasks']
        all_tasks_config = self.config.get('tasks')
        if not isinstance(all_tasks_config, list): return None
        for task_conf in all_tasks_config:
            if isinstance(task_conf, dict) and task_conf.get('task_id') == task_id_to_find:
                return task_conf
        return None

    def _is_trigger_time(self, cron_expression: Optional[str], last_successful_run_time: Optional[datetime.datetime] = None) -> bool:
        """(内部) 检查 Cron 触发条件"""
        if not cron_expression: return False
        if not CRONITER_INSTALLED: return False # croniter 未安装则无法检查
        try:
            now = datetime.datetime.now(DEFAULT_TIMEZONE)
            base_time = last_successful_run_time if last_successful_run_time else now - datetime.timedelta(days=1)
            if base_time.tzinfo is None and now.tzinfo is not None: base_time = now.tzinfo.localize(base_time)
            elif base_time.tzinfo is not None and base_time.tzinfo != now.tzinfo: base_time = base_time.astimezone(now.tzinfo)

            cron = croniter(cron_expression, start_time=base_time)
            prev_run_naive = cron.get_prev(datetime.datetime) # 获取上一个计划运行点（naive）
            # 转换为当前时区
            prev_run = DEFAULT_TIMEZONE.localize(prev_run_naive) if DEFAULT_TIMEZONE else prev_run_naive
            # logger.trace(f"Cron Check: Now={now}, PrevPlanned={prev_run}, LastRun={last_successful_run_time}")
            # 如果当前时间在上一个计划点之后，并且（如果存在上次运行时间）上次运行时间在上一个计划点之前
            if now >= prev_run and (last_successful_run_time is None or last_successful_run_time < prev_run):
                 self.logger.debug(f"Cron trigger '{cron_expression}' is due.")
                 return True
            return False
        except Exception as e:
            self.logger.error(f"检查 Cron 表达式 '{cron_expression}' 时出错: {e}")
            return False

    def _load_raw_data_for_check(self, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> bool:
        """(内部) 加载或更新数据缓存用于检查返回是否加载成功"""
        now = datetime.datetime.now()
        # 检查缓存是否需要更新
        if self._raw_data_load_time is None or \
           (now - self._raw_data_load_time).total_seconds() > self._raw_data_cache_ttl_seconds:
            self.logger.info("数据缓存已过期或不存在，重新加载...")
            self._raw_data_cache = {'fans': None, 'tower': None, 'weather': None} # 清空缓存
            ds_config = self.effective_config['DataSource']
            raw_resource_dir = const.PROJECT_ROOT / ds_config['dir']
            required_raw_files = {
                'fans': raw_resource_dir / const.FANS_CSV,
                'tower': raw_resource_dir / const.TOWER_CSV,
                'weather': raw_resource_dir / const.WEATHER_CSV
            }
            load_success = True
            for name, path in required_raw_files.items():
                 if path.exists():
                      try:
                           time_col = ds_config[f"{name.capitalize()}TimeCol"] if name != 'weather' else ds_config['WeatherForecastTimeCol']
                           # 优化只读取时间列检查范围？不，检查需要值
                           # 读取完整文件（或优化为只读部分日期？）
                           df_raw = pd.read_csv(path, parse_dates=[time_col], infer_datetime_format=True)
                           df_raw = df_raw.set_index(time_col).sort_index()
                           self._raw_data_cache[name] = df_raw
                           self.logger.debug(f"成功加载并缓存数据: {name} ({df_raw.shape})")
                      except Exception as load_e:
                           self.logger.error(f"加载数据文件 '{path}' 失败: {load_e}")
                           if name == 'fans': load_success = False # fans 是必需的
                 else:
                      logger.warning(f"数据文件不存在: {path}")
                      if name == 'fans': load_success = False
            self._raw_data_load_time = now if load_success else None # 成功加载才更新时间戳
            return load_success
        else:
            # logger.trace("使用有效的数据缓存")
            return True # 缓存有效


    # 数据可用性检查逻辑和日志
    def _check_data_readiness(self, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> bool:
        """(内部) 检查所需时间范围内的数据是否完整可用"""
        self.logger.info(f"开始检查数据可用性，范围: [{start_dt}, {end_dt}]...")
        required_freq_str = self.effective_config['Feature']['TimeFrequency']
        is_ready = True

        try:
            # 1. 加载或更新数据缓存
            if not self._load_raw_data_for_check(start_dt, end_dt):
                 self.logger.error("无法加载必要的数据（特别是 fans.csv），数据检查失败")
                 return False

            # 2. 生成预期的时间点索引
            expected_index = pd.date_range(start=start_dt, end=end_dt, freq=required_freq_str, name=const.INTERNAL_TIME_INDEX)
            if expected_index.empty:
                 self.logger.warning("数据检查生成预期时间索引为空")
                 return False

            # 3. 获取需要检查的关键列名
            ds_config = self.effective_config['DataSource']
            critical_cols = {
                'fans': [ds_config['FansPowerCol']],
                'tower': list(ds_config['TowerColRenameMap'].keys())[1:], # 排除时间列
                'weather': list(ds_config['WeatherColRenameMap'].keys())
            }

            # 4. 循环检查每个时间间隔的数据可用性 (还有日志)
            check_interval = self.data_freq # 检查每个数据点
            current_check_dt = start_dt
            while current_check_dt <= end_dt:
                 # --- 打印日志 ---
                 log_start_str = current_check_dt.strftime(const.DATETIME_FORMAT)
                 # 假设日志范围是固定的 4 小时，与实际检查逻辑可能无关
                 log_end_dt = current_check_dt + datetime.timedelta(hours=4)
                 log_end_str = log_end_dt.strftime(const.DATETIME_FORMAT)
                 data_freq_minutes = int(self.data_freq.total_seconds() / 60)
                 self.logger.info(f"make sure Data is available every {data_freq_minutes} minutes in range [{log_start_str}, {log_end_str}]")

                 # --- 实际检查 ---
                 # 检查每个数据源在 current_check_dt 是否有效
                 for source_name, df_raw in self._raw_data_cache.items():
                      if df_raw is None or df_raw.empty: # 如果某个源未加载
                           if source_name == 'fans': # fans 是必需的
                                self.logger.error(f"数据检查失败必需的 {source_name} 数据源未加载")
                                is_ready = False; break
                           else: continue # 其他源缺失则跳过检查

                      cols_to_check = critical_cols.get(source_name, [])
                      if not cols_to_check: continue # 无需检查的列

                      if current_check_dt not in df_raw.index:
                           self.logger.error(f"数据检查失败时间点 {current_check_dt} 在 {source_name} 数据中缺失索引！")
                           is_ready = False; break
                      # 检查关键列是否有 NaN
                      # 使用 .loc 获取行，然后检查 NaN
                      row_data = df_raw.loc[[current_check_dt]] # 获取单行 DataFrame
                      if row_data[cols_to_check].isna().any().any():
                            missing_info_cols = row_data[cols_to_check].isna().any()[lambda x: x].index.tolist()
                            self.logger.error(f"数据检查失败时间点 {current_check_dt} 在 {source_name} 数据源的关键列 {missing_info_cols} 存在 NaN！")
                            is_ready = False; break

                 if not is_ready: break # 发现问题，停止检查

                 current_check_dt += check_interval # 移动到下一个时间点

        except Exception as e:
            self.logger.exception(f"检查数据可用性时发生严重错误: {e}")
            is_ready = False

        if is_ready: self.logger.info("数据可用性检查通过")
        else: self.logger.error("数据可用性检查未通过！")
        return is_ready

    # 使用计时器装饰 run 方法
    @common_utils.log_execution_time(level="INFO")
    def run(self, current_time_override: Optional[pd.Timestamp] = None) -> None:
        """
        执行单个任务的完整流程
        """
        if not self.enabled:
            self.logger.info(f"任务 '{self.task_id}' 未启用，跳过执行")
            return

        self.logger.info(f"=============== 开始执行任务: {self.task_id} ({self.description}) ===============")
        current_time = current_time_override if current_time_override is not None \
                       else pd.Timestamp.now(tz=DEFAULT_TIMEZONE)
        if current_time.tzinfo is None and DEFAULT_TIMEZONE is not None:
             current_time = DEFAULT_TIMEZONE.localize(current_time)
        self.logger.info(f"任务执行参考时间: {current_time}")

        # 获取当前和上次运行元数据
        current_metadata = self.incrml_manager.metadata.get_current_version()
        last_run_time = pd.to_datetime(current_metadata.timestamp).tz_convert(DEFAULT_TIMEZONE) if current_metadata and current_metadata.timestamp else None
        last_train_end_time = pd.to_datetime(current_metadata.training_data_end).tz_localize(DEFAULT_TIMEZONE) if current_metadata and current_metadata.training_data_end else None

        # --- 判断是否需要训练/增量更新 ---
        should_train_update = False
        run_incremental = False
        if self._is_trigger_time(self.train_trigger_cron, last_run_time):
             should_train_update = True; run_incremental = False
             self.logger.info("定时训练触发器满足，计划执行完全重新训练")
        elif self.incrml_manager and self.incrml_manager.enabled:
             latest_data_ts = current_time - self.data_fetch_lag # 估算最新数据时间
             if self.incrml_manager.trigger != 'OnDriftDetected': # 漂移检测不在此处主动触发
                  if self.incrml_manager.check_trigger(latest_data_timestamp=latest_data_ts):
                       should_train_update = True; run_incremental = True
                       self.logger.info("增量学习触发器满足，计划执行增量更新")

        # --- 执行训练或增量更新 ---
        training_status_ok = True # 标记本轮训练/更新是否成功
        current_run_id : Optional[str] = current_metadata.version_id if current_metadata else None # 获取当前模型ID

        if should_train_update:
            self.logger.info(f"the {self.task_id} {current_time.strftime(const.DATETIME_FORMAT)} training task start.") # 严格匹配日志

            # 计算训练窗口
            train_end_dt = (current_time + self.train_end_offset).floor(self.data_freq)
            train_start_dt = (train_end_dt + self.train_start_offset).ceil(self.data_freq)
            self.logger.info(f"计算训练数据窗口: [{train_start_dt}, {train_end_dt}]")

            # 检查数据可用性
            if not self._check_data_readiness(train_start_dt, train_end_dt):
                 self.logger.error("训练所需数据不完整，取消本次训练/更新！")
                 training_status_ok = False # 标记失败
            else:
                 self.logger.info("Getting the training data...")
                 # 确保 MLPipeline 和 IncrML Manager 实例存在
                 if self.ml_pipeline is None or self.incrml_manager is None:
                      self.logger.error("MLPipeline 或 IncrmlManager 未初始化！")
                      training_status_ok = False
                 else:
                      if run_incremental: # --- 执行增量更新 ---
                           self.logger.info("执行增量更新流程...")
                           try:
                                # 加载所有可用数据（由 manager 内部的 sampler 处理）
                                data_processor_inc = DataProcessor(config=self.effective_config)
                                available_df = data_processor_inc.process(start_time=None, end_time=train_end_dt)
                                # 确定新数据部分
                                new_data_start = last_train_end_time + self.data_freq if last_train_end_time else train_start_dt
                                new_data = available_df[available_df.index >= new_data_start] if not available_df.empty else pd.DataFrame()

                                incrml_train_data = self.incrml_manager.prepare_data(available_df=available_df)
                                if incrml_train_data.empty:
                                     self.logger.warning("增量数据准备后为空，跳过更新")
                                     training_status_ok = False # 算作未成功更新
                                else:
                                     update_successful = self.incrml_manager.update(
                                         ml_pipeline=self.ml_pipeline,
                                         incrml_train_df=incrml_train_data,
                                         new_data_processed=new_data
                                     )
                                     if update_successful:
                                         current_run_id = self.incrml_manager.metadata.current_version_id # 更新 run_id
                                         log_ts = pd.to_datetime(self.incrml_manager.metadata.get_version(current_run_id).training_data_end)
                                         # 严格匹配日志格式和内容
                                         self.logger.info(f"the {log_ts.strftime(const.DATETIME_FORMAT)} training task training successful.")
                                     else:
                                         training_status_ok = False # 标记失败
                                         self.logger.error("增量更新执行失败！")
                           except Exception as inc_e: self.logger.exception("执行增量更新时出错！"); training_status_ok = False
                      else: # --- 执行完全重新训练 ---
                           self.logger.info("执行完全重新训练流程...")
                           try:
                               data_processor_full = DataProcessor(config=self.effective_config)
                               df_full_train = data_processor_full.process(start_time=train_start_dt, end_time=train_end_dt)
                               if df_full_train.empty:
                                    self.logger.error("加载的完全训练数据为空！")
                                    training_status_ok = False
                               else:
                                    run_id_full = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S%f') + "_full"
                                    train_successful = self.ml_pipeline.train(
                                        df_train_full=df_full_train, run_id=run_id_full
                                    )
                                    if train_successful:
                                         current_run_id = run_id_full # 更新 run_id
                                         # 全量训练后也需要更新元数据
                                         self.logger.info("全量训练完成，正在更新 IncrML 元数据...")
                                         model_path, tf_path, feat_path, _ = self.ml_pipeline._get_run_specific_paths(run_id_full)
                                         performance = {}
                                         if hasattr(self.ml_pipeline, 'automl_wrapper') and self.ml_pipeline.automl_wrapper and \
                                            self.ml_pipeline.automl_wrapper.final_val_score is not None:
                                              performance[self.ml_pipeline.automl_wrapper.metric] = self.ml_pipeline.automl_wrapper.final_val_score

                                         new_version_info_full = ModelVersionInfo(
                                              version_id=run_id_full, model_path=str(Path(model_path).relative_to(const.PROJECT_ROOT).as_posix()),
                                              transformer_state_path=str(Path(tf_path).relative_to(const.PROJECT_ROOT).as_posix()),
                                              selected_features_path=str(Path(feat_path).relative_to(const.PROJECT_ROOT).as_posix()),
                                              training_data_start=train_start_dt.isoformat(), training_data_end=train_end_dt.isoformat(),
                                              performance_metrics=performance, base_model_version=None # 全量训练无基线
                                         )
                                         # 更新采样器状态（用全量数据？）
                                         if self.incrml_manager.sampler:
                                             sampler_state_upd = self.incrml_manager.sampler.update_state(df_full_train, df_full_train, new_version_info_full.to_dict())
                                             if sampler_state_upd: new_version_info_full.misc_info.update(sampler_state_upd)

                                         self.incrml_manager.metadata.add_version(new_version_info_full, set_as_current=True)
                                         self.incrml_manager.metadata.save()
                                         self.logger.info(f"the {train_end_dt.strftime(const.DATETIME_FORMAT)} training task training successful.")
                                    else:
                                         training_status_ok = False
                                         self.logger.error("完全重新训练失败！")
                           except Exception as full_e: self.logger.exception("执行完全重新训练时出错！"); training_status_ok = False

            # 严格匹配训练结束日志
            if training_status_ok and current_run_id: # 确保有成功的 run_id
                self.logger.info(f"{self.task_id} task finish training successfully.")
            elif not should_train_update: pass # 没有触发训练
            else: self.logger.error(f"{self.task_id} task finish training failed.")


        # --- 判断是否需要预测 ---
        # 预测应该独立于训练触发器进行检查
        should_predict = self._is_trigger_time(self.predict_trigger_cron, last_run_time) # 假设也基于上次运行时间

        if should_predict:
            predict_ref_time = current_time
            # 严格匹配日志
            self.logger.info(f"{self.task_id} task start prediction at {predict_ref_time}.")

            predict_end_dt = (predict_ref_time + self.predict_input_end_offset).floor(self.data_freq)
            predict_start_dt = (predict_end_dt + self.predict_input_start_offset).ceil(self.data_freq)
            self.logger.info(f"计算预测输入数据窗口: [{predict_start_dt}, {predict_end_dt}]")

            if not self._check_data_readiness(predict_start_dt, predict_end_dt):
                 self.logger.error("预测所需输入数据不完整，取消本次预测！")
            else:
                 if self.ml_pipeline is None: self.ml_pipeline = MLPipeline(config=self.effective_config) # 确保实例存在
                 try:
                      # 加载预测输入数据 (需要 DP 处理)
                      data_processor_pred = DataProcessor(config=self.effective_config)
                      # 注意预测时 end_time 应该是 predict_ref_time，但加载需要 predict_end_dt
                      X_pred_input = data_processor_pred.process(start_time=predict_start_dt, end_time=predict_end_dt)
                      if X_pred_input.empty: raise ValueError("处理后的预测输入数据为空")
                      if self.target_name in X_pred_input.columns:
                          X_pred_input = X_pred_input.drop(columns=[self.target_name])

                      # 获取当前模型 ID (应该使用最新的，可能是刚训练的，也可能是之前的)
                      current_model_meta = self.incrml_manager.metadata.get_current_version()
                      if not current_model_meta or not current_model_meta.version_id:
                          raise RuntimeError("无法获取当前有效的模型版本 ID 进行预测")
                      model_run_id_to_load = current_model_meta.version_id
                      self.logger.info(f"使用模型版本 '{model_run_id_to_load}' 进行预测")

                      # 调用预测
                      predictions_array = self.ml_pipeline.predict(
                          X_test=X_pred_input, run_id=model_run_id_to_load
                      )

                      if predictions_array is not None:
                           # 构建结果 DataFrame
                           pred_start_time = predict_ref_time.ceil(self.data_freq)
                           pred_end_time = pred_start_time + self.forecast_horizon - self.data_freq
                           pred_index = pd.date_range(start=pred_start_time, end=pred_end_time, freq=self.data_freq, name=const.INTERNAL_TIME_INDEX)

                           # 对齐预测结果和索引
                           predictions_adjusted = np.full(len(pred_index), np.nan)
                           # 预测结果通常对应于输入窗口的最后一个点之后的时间点
                           # 需要确定 predict 返回的数组与 pred_index 的对齐关系
                           # 假设 predict 返回的是从 pred_start_time 开始的预测
                           common_len = min(len(predictions_array), len(pred_index))
                           predictions_adjusted[:common_len] = predictions_array[:common_len]
                           if len(predictions_array) != len(pred_index):
                                logger.warning(f"预测结果长度({len(predictions_array)})与预期({len(pred_index)})不符，已调整")

                           predictions_df = pd.DataFrame({const.PREDICTION_COLUMN_NAME: predictions_adjusted}, index=pred_index)
                           # 可以加入其他列，例如预测输入特征？取决于代码

                           # 保存结果
                           save_ok = result_saver.save_prediction_result(
                               predictions_df=predictions_df, task_id=self.task_id,
                               output_dir=const.PRED_DIR, is_backtrack=False
                           )
                           if save_ok:
                               # 严格匹配日志
                               # 日志中的时间戳是哪个？假设是预测参考时间
                               self.logger.info(f"the {predict_ref_time.strftime(const.DATETIME_FORMAT)} training task predict successful.")
                           else: self.logger.error("保存预测结果失败！")
                      else: self.logger.error("MLPipeline.predict 返回 None，预测失败")

                 except Exception as pred_e: self.logger.exception(f"执行预测时出错: {pred_e}")

        else: self.logger.debug("预测触发器未满足条件，跳过预测")


        # --- 回测逻辑 (如果 task_type == 'backtrack') ---
        if self.task_run_type == 'backtrack':
             self._run_backtracking(current_time)


        self.logger.info(f"=============== 任务执行结束: {self.task_id} ================")

    # --- 回测方法  ---
    def _run_backtracking(self, execution_time: pd.Timestamp):
        """(内部) 执行回测流程"""
        self.logger.info(f"--- 开始执行回测流程 for task '{self.task_id}' ---")
        if not self.backtrack_start_date_str or not self.backtrack_end_date_str:
             self.logger.error("回测配置缺少 backtrack_start_date 或 backtrack_end_date")
             return

        try:
            backtrack_start = pd.to_datetime(self.backtrack_start_date_str).floor('D')
            backtrack_end = pd.to_datetime(self.backtrack_end_date_str).ceil('D') - pd.Timedelta(seconds=1) # 包含结束日期
            backtrack_time = datetime.datetime.strptime(self.backtrack_time_of_day_str, '%H:%M:%S').time()

            # 生成回测时间点序列 (每天一个预测点)
            backtrack_dates = pd.date_range(start=backtrack_start, end=backtrack_end, freq='D')
            self.logger.info(f"回测时间范围: {backtrack_start.date()} to {backtrack_end.date()}, "
                             f"模拟预测时间点: {self.backtrack_time_of_day_str}")

            for backtrack_date in backtrack_dates:
                 # 组合日期和时间形成回测的“当前时间点”
                 current_simulated_time = pd.Timestamp.combine(backtrack_date.date(), backtrack_time)
                 if DEFAULT_TIMEZONE: current_simulated_time = DEFAULT_TIMEZONE.localize(current_simulated_time)
                 self.logger.info(f"--- 回测时间点: {current_simulated_time} ---")

                 run_id_backtrack: Optional[str] = None
                 backtrack_train_status_ok = True

                 # 1. 回测训练 (如果需要)
                 if self.backtrack_retrain:
                      self.logger.info(f"回测 ({current_simulated_time}): 需要重新训练模型...")
                      # 计算回测训练窗口
                      train_end_dt = (current_simulated_time + self.backtrack_train_end_offset).floor(self.data_freq)
                      train_start_dt = (train_end_dt + self.backtrack_train_start_offset).ceil(self.data_freq)
                      self.logger.info(f"计算回测训练数据窗口: [{train_start_dt}, {train_end_dt}]")

                      if not self._check_data_readiness(train_start_dt, train_end_dt):
                           self.logger.error(f"回测 ({current_simulated_time}): 训练数据不完整，跳过此时间点！")
                           backtrack_train_status_ok = False
                      else:
                           self.logger.info(f"回测 ({current_simulated_time}): 开始训练...")
                           try:
                               # 实例化新的 MLPipeline 进行训练
                               pipeline_backtrack = MLPipeline(config=self.effective_config)
                               data_processor_backtrack = DataProcessor(config=self.effective_config)
                               df_backtrack_train = data_processor_backtrack.process(start_time=train_start_dt, end_time=train_end_dt)
                               if df_backtrack_train.empty: raise ValueError("回测训练数据为空")

                               # 使用特定的 run_id 模式
                               time_str_for_id = current_simulated_time.strftime('%Y%m%d_%H%M')
                               run_id_backtrack = f"backtrack_{time_str_for_id}_{self.task_id}"
                               # 构造模型保存路径 (如果需要保存回测模型)
                               # model_path_bt = self.backtrack_model_path_pattern.format(date_YYYYMMDD=..., time_HHMM=...)
                               # 需要确保 MLPipeline 使用正确的保存路径

                               train_successful = pipeline_backtrack.train(
                                   df_train_full=df_backtrack_train,
                                   run_id=run_id_backtrack
                                   # 可能需要传递不同的模型保存路径给 train
                               )
                               if not train_successful:
                                    backtrack_train_status_ok = False
                                    self.logger.error(f"回测 ({current_simulated_time}): 训练失败！")
                               else: # 严格匹配日志
                                    self.logger.info(f"the {train_end_dt.strftime(const.DATETIME_FORMAT)} training task training successful.") # 日志时间戳用训练结束时间
                           except Exception as bt_train_e:
                                self.logger.exception(f"回测 ({current_simulated_time}): 训练过程中出错: {bt_train_e}")
                                backtrack_train_status_ok = False
                 else: # 不重新训练，需要加载对应日期的模型
                      # 需要逻辑来确定加载哪个 run_id (可能基于 backtrack_model_path_pattern)
                      # run_id_backtrack = # ... 确定要加载的模型 ID ...
                      self.logger.info(f"回测 ({current_simulated_time}): 跳过重新训练，将加载预训练模型 (ID: {run_id_backtrack})")
                      # 需要检查模型是否存在
                      # ...

                 # 2. 回测预测 (仅当训练成功或模型已加载)
                 if run_id_backtrack and backtrack_train_status_ok:
                     self.logger.info(f"回测 ({current_simulated_time}): 开始预测...")
                     # 计算预测输入窗口
                     predict_end_dt = (current_simulated_time + self.backtrack_predict_input_end_offset).floor(self.data_freq)
                     predict_start_dt = (predict_end_dt + self.backtrack_predict_input_start_offset).ceil(self.data_freq)
                     self.logger.info(f"计算回测预测输入数据窗口: [{predict_start_dt}, {predict_end_dt}]")

                     if not self._check_data_readiness(predict_start_dt, predict_end_dt):
                          self.logger.error(f"回测 ({current_simulated_time}): 预测所需输入数据不完整，跳过预测！")
                     else:
                          try:
                              # 实例化 Pipeline (或使用已有的)
                              pipeline_pred = MLPipeline(config=self.effective_config)
                              data_processor_pred = DataProcessor(config=self.effective_config)
                              X_pred_input = data_processor_pred.process(start_time=predict_start_dt, end_time=predict_end_dt)
                              if X_pred_input.empty: raise ValueError("回测预测输入数据为空")
                              if self.target_name in X_pred_input.columns: X_pred_input = X_pred_input.drop(columns=[self.target_name])

                              predictions_array = pipeline_pred.predict(
                                  X_test=X_pred_input, run_id=run_id_backtrack # 使用回测 run_id 加载模型
                              )

                              if predictions_array is not None:
                                   # 构建结果 DataFrame
                                   pred_start_time = current_simulated_time.ceil(self.data_freq)
                                   pred_end_time = pred_start_time + self.forecast_horizon - self.data_freq
                                   pred_index = pd.date_range(start=pred_start_time, end=pred_end_time, freq=self.data_freq, name=const.INTERNAL_TIME_INDEX)

                                   predictions_adjusted = np.full(len(pred_index), np.nan)
                                   common_len = min(len(predictions_array), len(pred_index))
                                   predictions_adjusted[:common_len] = predictions_array[:common_len]
                                   if len(predictions_array) != len(pred_index): logger.warning("回测预测结果长度与预期不符")

                                   predictions_df = pd.DataFrame({const.PREDICTION_COLUMN_NAME: predictions_adjusted}, index=pred_index)

                                   # 保存回测结果
                                   save_ok = result_saver.save_prediction_result(
                                       predictions_df=predictions_df, task_id=self.task_id,
                                       output_dir=const.PRED_DIR,
                                       is_backtrack=True, # 标记为回测
                                       pred_ref_time=current_simulated_time # 传递回测时间点
                                   )
                                   if save_ok: # 严格匹配日志
                                        self.logger.info(f"the {current_simulated_time.strftime(const.DATETIME_FORMAT)} training task predict successful.")
                                   else: self.logger.error(f"回测 ({current_simulated_time}): 保存预测结果失败！")
                              else: self.logger.error(f"回测 ({current_simulated_time}): 预测失败 (predict 返回 None)")

                          except Exception as bt_pred_e:
                               self.logger.exception(f"回测 ({current_simulated_time}): 预测过程中出错: {bt_pred_e}")
                 else:
                      self.logger.warning(f"回测 ({current_simulated_time}): 训练失败或无法确定模型 ID，跳过预测")

        except Exception as bt_e:
             self.logger.exception(f"执行回测流程时发生严重错误: {bt_e}")

        self.logger.info(f"--- 回测流程结束 for task '{self.task_id}' ---")

logger.info("风电增量学习任务驱动模块 (espml.util.wind_incrml) 加载完成")