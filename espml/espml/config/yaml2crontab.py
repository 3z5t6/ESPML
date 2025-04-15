# -*- coding: utf-8 -*-
"""
ESPML 任务配置转 Crontab 命令生成器
读取 task_config.yaml 并生成对应的 crontab 计划任务行
"""

import yaml
import os
import sys
from pathlib import Path
import argparse
from typing import Dict, Any, List, Optional, Union

# 添加项目根目录到路径
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
from espml.util import utils as common_utils

# 导入所需模块
try:
    from espml.config.yaml_parser import load_yaml_config, ConfigError
    from espml.util import const
    # 导入 croniter 用于验证
    try:
        from croniter import croniter
        CRONITER_INSTALLED = True
    except ImportError:
        print("警告: 'croniter' 未安装,无法验证 Cron 表达式")
        croniter = None # type: ignore
        CRONITER_INSTALLED = False
except ImportError:
     print("错误无法导入 espml 模块请确保项目结构正确", file=sys.stderr)
     # 定义后备路径和变量
     const = type('obj', (object,), {'PROJECT_CONFIG_DIR': Path('.') / 'espml' / 'project' / 'WindPower'})()
     load_yaml_config = None; ConfigError = Exception # type: ignore
     croniter = None; CRONITER_INSTALLED = False

def generate_crontab_lines(
    task_config_path: Union[str, Path],
    python_executable: str = sys.executable, # 使用当前解释器
    main_script_path: Optional[str] = None,
    backtrack_script_path: Optional[str] = None,
    config_path_arg: Optional[str] = None # 允许传递主配置文件路径给脚本
    ) -> List[str]:
    """
    根据任务配置文件生成 Crontab 命令列表

    Args:
        task_config_path: 任务配置 YAML 文件路径
        python_executable: Python 解释器路径
        main_script_path: main.py 脚本的绝对路径 (可选, 自动推断)
        backtrack_script_path: backtracking.py 脚本的绝对路径 (可选, 自动推断)
        config_path_arg: 传递给脚本的 --config 参数值 (可选, 默认为任务配置同目录的 config.yaml)

    Returns:
        生成的 Crontab 命令字符串列表
    """
    crontab_lines: List[str] = []
    print(f"读取任务配置: {task_config_path}")

    if load_yaml_config is None:
        print("错误YAML 解析器未加载", file=sys.stderr)
        return []

    try:
        task_config_content = load_yaml_config(task_config_path)
        if task_config_content is None or 'tasks' not in task_config_content or not isinstance(task_config_content['tasks'], list):
             raise ConfigError("任务配置文件无效或缺少 'tasks' 列表")
        tasks: List[Dict[str, Any]] = task_config_content['tasks']
    except (FileNotFoundError, ConfigError, Exception) as e:
        print(f"错误加载或解析任务配置文件 '{task_config_path}' 失败: {e}", file=sys.stderr)
        return []

    # 确定脚本路径 (假设在 scripts/ 下)
    scripts_dir = project_root / "scripts"
    if main_script_path is None:
        main_script_path = str((scripts_dir / "main.py").resolve()) # 获取绝对路径
    if backtrack_script_path is None:
        backtrack_script_path = str((scripts_dir / "backtracking.py").resolve())

    # 确定传递给脚本的 --config 参数
    if config_path_arg is None:
        # 默认使用 task_config 同目录下的 config.yaml
        default_config_path = Path(task_config_path).parent / "config.yaml"
        config_path_arg = str(default_config_path.resolve())
    else:
         config_path_arg = str(Path(config_path_arg).resolve()) # 确保是绝对路径

    print(f"使用 Python: {python_executable}")
    print(f"主脚本: {main_script_path}")
    print(f"回测脚本: {backtrack_script_path}")
    print(f"传递给脚本的 --config: {config_path_arg}")
    print(f"传递给脚本的 --task_config: {str(Path(task_config_path).resolve())}") # 传递绝对路径

    # 生成 Crontab 行
    for task in tasks:
        if not isinstance(task, dict): continue

        task_id = task.get('task_id')
        is_enabled = task.get('enabled', False)
        task_type = task.get('type', 'forecast')

        if not task_id or not is_enabled: continue

        triggers: Dict[str, Optional[str]] = {
            'train': task.get('train_trigger_cron'),
            'predict': task.get('predict_trigger_cron'),
            'backtrack': task.get('backtrack_trigger_cron')
        }

        # 生成命令行基础部分
        base_command = f"{python_executable} {{script_path}} --config {config_path_arg} --task_config {str(Path(task_config_path).resolve())} --task_id {task_id}"

        # 根据类型生成命令
        if task_type == 'forecast':
            if triggers['train']:
                cron_expr = triggers['train']
                # 假设 main.py 通过 task_id 自行判断是训练还是预测,或者需要 --mode 参数
                command = base_command.format(script_path=main_script_path) # + " --mode train"
                if croniter and not croniter.is_valid(cron_expr): print(f"警告: 任务 '{task_id}' 训练 Cron 无效: '{cron_expr}'")
                crontab_lines.append(f"{cron_expr} {command} # ESPML Train: {task_id}")
            if triggers['predict']:
                cron_expr = triggers['predict']
                command = base_command.format(script_path=main_script_path) # + " --mode predict"
                if croniter and not croniter.is_valid(cron_expr): print(f"警告: 任务 '{task_id}' 预测 Cron 无效: '{cron_expr}'")
                crontab_lines.append(f"{cron_expr} {command} # ESPML Predict: {task_id}")

        elif task_type == 'backtrack':
            if triggers['backtrack']:
                cron_expr = triggers['backtrack']
                # 回测脚本需要指定日期范围,这通常不由 cron 提供
                # 生成的命令只负责按时启动回测脚本
                # 需要确保回测脚本能处理日期（例如默认跑昨天,或从状态文件读取）
                command = base_command.format(script_path=backtrack_script_path) # 回测脚本也需要 task_id
                # 移除 --task_id? 取决于回测脚本逻辑,假设它需要知道是哪个任务的回测配置
                # command = f"{python_executable} {backtrack_script_path} --config {config_path_arg} --task_config {str(Path(task_config_path).resolve())} --task_id {task_id}"

                if croniter and not croniter.is_valid(cron_expr): print(f"警告: 任务 '{task_id}' 回测 Cron 无效: '{cron_expr}'")
                crontab_lines.append(f"{cron_expr} {command} # ESPML Backtrack Trigger: {task_id}")


    return crontab_lines

if __name__ == "__main__":
    parser_cron = argparse.ArgumentParser(description="根据任务配置生成 Crontab 命令")
    default_task_config = const.PROJECT_CONFIG_DIR / "task_config.yaml" if const else Path('./default_config/task_config.yaml')
    parser_cron.add_argument("--task_config", type=str, default=str(default_task_config), help=f"任务配置 YAML 文件路径 (默认: {default_task_config})")
    parser_cron.add_argument("--python_exec", type=str, default=sys.executable, help=f"Python 解释器路径 (默认: {sys.executable})")
    parser_cron.add_argument("--main_script", type=str, default=None, help="main.py 脚本的绝对路径 (默认自动推断)")
    parser_cron.add_argument("--backtrack_script", type=str, default=None, help="backtracking.py 脚本的绝对路径 (默认自动推断)")
    parser_cron.add_argument("--config_arg", type=str, default=None, help="传递给脚本的 --config 参数值 (默认自动推断)")
    parser_cron.add_argument("--output_file", type=str, default=None, help="将 Crontab 行输出到指定文件")
    cron_args = parser_cron.parse_args()

    lines = generate_crontab_lines(
        task_config_path=cron_args.task_config,
        python_executable=cron_args.python_exec,
        main_script_path=cron_args.main_script,
        backtrack_script_path=cron_args.backtrack_script,
        config_path_arg=cron_args.config_arg
    )

    output_content = f"# ESPML Crontab Entries (Generated by {Path(__file__).name})\n"
    output_content += f"# Task Config: {Path(cron_args.task_config).resolve()}\n"
    output_content += "# PLEASE REVIEW AND ADD THESE LINES TO YOUR CRONTAB MANUALLY (e.g., using 'crontab -e')\n\n"
    output_content += "\n".join(lines) + "\n"

    if cron_args.output_file:
        try:
            output_file_path = Path(cron_args.output_file)
            # 使用 common_utils (如果导入成功)
            if common_utils: common_utils.mkdir_if_not_exist(output_file_path.parent)
            else: os.makedirs(output_file_path.parent, exist_ok=True) # Fallback
            with open(output_file_path, 'w', encoding='utf-8') as f: f.write(output_content)
            print(f"Crontab 命令已写入文件: {output_file_path}")
        except Exception as e:
            print(f"\n错误写入 Crontab 文件失败: {e}", file=sys.stderr)
            print("\n" + output_content) # 仍然打印到标准输出
    else:
        print(output_content) # 打印到标准输出
