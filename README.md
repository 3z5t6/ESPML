# ESPML - 智能能源预测学习系统

ESPML（Energy Smart Prediction and Machine Learning）是一个专注于能源领域的智能预测系统，基于机器学习技术实现对能源数据的智能分析和预测。系统支持增量学习能力，可以持续优化模型性能。

## 项目特点

- **增量学习**：持续从新数据中学习，优化模型性能
- **自动特征工程**：自动生成和选择有效特征
- **智能预测**：支持多种时间尺度的能源预测任务
- **数据漂移检测**：实时检测数据分布变化，保证模型可靠性
- **回溯测试**：支持历史数据回溯测试，验证模型性能

## 项目结构

```
ESPML/
├── config/                # 配置文件
├── data/                  # 数据文件
├── module/                # 模块文件
│   ├── autofe/            # 自动特征工程
│   ├── automl/            # 自动机器学习
│   ├── dataprocess/       # 数据处理
│   ├── incrml/            # 增量学习模块
│   ├── project/           # 项目相关模块
│   └── utils/             # 工具函数
├── scripts/               # 脚本文件
├── test/                  # 测试代码
├── backtracking.py        # 回溯测试程序
├── main.py                # 主程序
└── requirements.txt       # 依赖包列表
```

## 环境要求

- Python 3.10
- 依赖包见 `requirements.txt`

## 安装

1. 克隆仓库到本地
   ```
   git clone https://github.com/3z5t6/ESPML.git
   cd ESPML
   ```

2. 安装依赖
   ```
   pip install -r requirements.txt
   ```

## 使用方法示例

### 模型训练

```bash
python main.py --type train --task Forecast1Day --config config/taskconfig.yaml
```

### 模型预测

```bash
python main.py --type predict --task Forecast1Day --config config/taskconfig.yaml
```

### 回溯测试

```bash
python backtracking.py --task Forecast1Day --stime 2023-01-01 --etime 2023-01-31 --config config/taskconfig.yaml
```

## 参数说明

- `--type, -t`: 处理类型，可选 `train` 或 `predict`
- `--task`: 任务名称，如 `Forecast1Day`、`Forecast4Hour` 等
- `--config, -c`: 配置文件路径
- `--stime, -s`: 回溯测试起始时间（仅回溯测试使用）
- `--etime, -e`: 回溯测试结束时间（仅回溯测试使用）
- `--only_predict`: 仅执行预测（仅回溯测试使用）

## 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 许可证

[Apache 2.0 License](LICENSE)

项目链接：[https://github.com/3z5t6/ESPML](https://github.com/3z5t6/ESPML)