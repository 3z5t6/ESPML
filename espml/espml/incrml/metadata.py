# -*- coding: utf-8 -*-
"""
增量学习元数据管理模块 (espml)
负责定义、加载和保存与增量学习相关的模型元数据
"""

import datetime
import json
import os # 需要 os 来确保目录存在
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict, is_dataclass
from loguru import logger
import numpy as np

# 导入项目级 utils (用于文件操作)
from espml.util import utils as common_utils

class EnhancedJSONEncoder(json.JSONEncoder):
    """增强的 JSON 编码器,处理 dataclass 和 Path 对象"""
    def default(self, o: Any) -> Any:
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, Path):
            return str(o.as_posix())
        if isinstance(o, (datetime.date, datetime.datetime)):
             return o.isoformat()
        # 处理 numpy 类型 (如果需要)
        if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(o)
        elif isinstance(o, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(o)
        elif isinstance(o, (np.ndarray,)): # 处理 numpy 数组
            return o.tolist() # 转换为列表
        elif isinstance(o, (np.bool_)):
            return bool(o)
        elif isinstance(o, (np.void)): # 处理 numpy void 类型
             return None
        return super().default(o)

@dataclass
class ModelVersionInfo:
    """ 存储单个模型版本信息的类"""
    version_id: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    model_path: Optional[str] = None
    transformer_state_path: Optional[str] = None
    selected_features_path: Optional[str] = None
    training_data_start: Optional[str] = None
    training_data_end: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    drift_status: bool = False
    base_model_version: Optional[str] = None
    exemplar_set_path: Optional[str] = None
    misc_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersionInfo':
        """从字典创建实例"""
        valid_keys = cls.__annotations__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        try:
            # 可以添加类型转换逻辑,例如将 metrics 的值转为 float
            if 'performance_metrics' in filtered_data and isinstance(filtered_data['performance_metrics'], dict):
                filtered_data['performance_metrics'] = {k: float(v) for k, v in filtered_data['performance_metrics'].items()}
            return cls(**filtered_data)
        except (TypeError, ValueError) as e:
             logger.error(f"从字典创建 ModelVersionInfo 失败: {e}. Data: {data}")
             raise

class IncrmlMetadata:
    """ 管理特定任务的增量学习元数据"""
    def __init__(self, task_id: str, metadata_dir: Union[str, Path]):
        self.task_id = task_id
        self.metadata_dir = Path(metadata_dir)
        self.metadata_file = self.metadata_dir / f"{self.task_id}_incrml_metadata.json"
        self.versions: Dict[str, ModelVersionInfo] = {}
        self.current_version_id: Optional[str] = None
        self.logger = logger.bind(name=f"IncrmlMetadata_{task_id}")
        # 使用 common_utils 创建目录
        common_utils.mkdir_if_not_exist(self.metadata_dir)
        self._load()

    def _load(self) -> None:
        """(内部) 从 JSON 文件加载元数据"""
        self.logger.info(f"尝试从文件加载元数据: {self.metadata_file}")
        # 使用 common_utils 检查文件
        if common_utils.check_path_exists(self.metadata_file, path_type='f'):
            # 使用 common_utils 读取 JSON
            data = common_utils.read_json_file(self.metadata_file)
            if data and isinstance(data, dict): # 确保 data 是字典
                try:
                    self.current_version_id = data.get('current_version_id')
                    loaded_versions = data.get('versions', {})
                    if not isinstance(loaded_versions, dict):
                         raise TypeError("'versions' 字段必须是字典")
                    self.versions = {}
                    for v_id, v_data in loaded_versions.items():
                         if isinstance(v_data, dict):
                              try: self.versions[str(v_id)] = ModelVersionInfo.from_dict(v_data)
                              except Exception as model_e: self.logger.error(f"加载版本 '{v_id}' 数据失败: {model_e}跳过")
                         else: self.logger.warning(f"版本 '{v_id}' 数据非字典,跳过")
                    self.logger.info(f"成功加载 {len(self.versions)} 个版本元数据当前版本: {self.current_version_id}")
                except (TypeError, ValueError, KeyError) as e:
                    self.logger.error(f"元数据文件格式错误或内容无效: {e}初始化为空")
                    self.versions = {}; self.current_version_id = None
                except Exception as e:
                     self.logger.exception(f"加载元数据未知错误: {e}"); self.versions = {}; self.current_version_id = None
            else:
                self.logger.warning(f"元数据文件为空或解析失败: {self.metadata_file}初始化为空")
                self.versions = {}; self.current_version_id = None
        else:
            self.logger.info(f"元数据文件不存在: {self.metadata_file}初始化为空")
            self.versions = {}; self.current_version_id = None

    def save(self) -> bool:
        """将当前元数据保存到 JSON 文件"""
        self.logger.info(f"开始保存元数据到文件: {self.metadata_file}")
        data_to_save = {
            'task_id': self.task_id,
            'current_version_id': self.current_version_id,
            'versions': {v_id: v_info.to_dict() for v_id, v_info in self.versions.items()}
        }
        # 使用 common_utils 写入 JSON,传入自定义 Encoder
        success = common_utils.write_json_file(data_to_save, self.metadata_file, indent=4, cls=EnhancedJSONEncoder)
        if success: self.logger.info(f"元数据成功保存共 {len(self.versions)} 个版本")
        else: self.logger.error("保存元数据失败!")
        return success

    def add_version(self, version_info: ModelVersionInfo, set_as_current: bool = True) -> None:
        """添加一个新的模型版本信息"""
        if not isinstance(version_info, ModelVersionInfo): raise TypeError("version_info 必须是 ModelVersionInfo 实例")
        v_id = str(version_info.version_id);
        if not v_id: raise ValueError("版本 ID 不能为空")
        if v_id in self.versions: self.logger.warning(f"版本 ID '{v_id}' 已存在,将被覆盖")
        self.versions[v_id] = version_info
        self.logger.info(f"已添加/更新模型版本: {v_id}")
        if set_as_current: self.set_current_version(v_id)
        # self.save() # 代码决定是否自动保存

    def get_version(self, version_id: Optional[str]) -> Optional[ModelVersionInfo]:
        """获取指定版本的元数据"""
        return self.versions.get(str(version_id)) if version_id is not None else None

    def get_current_version(self) -> Optional[ModelVersionInfo]:
        """获取当前活动/最佳版本的元数据"""
        if self.current_version_id and self.current_version_id in self.versions:
            return self.versions[self.current_version_id]
        elif self.versions:
            try:
                latest_version = sorted(self.versions.values(), key=lambda v: v.timestamp, reverse=True)[0]
                self.logger.warning(f"当前版本 ID '{self.current_version_id}' 无效或未设置,返回最新版本: {latest_version.version_id}")
                return latest_version
            except (IndexError, Exception) as e: self.logger.error(f"查找最新版本时出错: {e}"); return None
        return None

    def set_current_version(self, version_id: str) -> bool:
        """设置当前活动/最佳版本的 ID"""
        v_id = str(version_id)
        if v_id not in self.versions:
            self.logger.error(f"无法设置版本 '{v_id}' 为当前,因其不存在")
            return False
        if self.current_version_id != v_id:
             self.current_version_id = v_id; self.logger.info(f"当前活动版本已设置为: {v_id}")
             # self.save()
        return True

    def get_all_versions(self) -> List[ModelVersionInfo]:
        """获取所有版本信息列表（按时间戳升序）"""
        return sorted(self.versions.values(), key=lambda v: v.timestamp)

    def remove_version(self, version_id: str) -> bool:
        """删除指定版本的元数据"""
        v_id = str(version_id)
        if v_id in self.versions:
            del self.versions[v_id]; self.logger.info(f"已删除版本: {v_id}")
            if self.current_version_id == v_id: self.current_version_id = None; self.logger.warning("当前活动版本已被删除")
            # self.save()
            return True
        else: self.logger.warning(f"尝试删除不存在的版本: {v_id}"); return False

logger.info("增量学习元数据模块 (espml.incrml.metadata) 加载完成")