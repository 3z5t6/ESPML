import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from module.utils.wind_incrml import WindIncrml
# from module.utils.wind_utils import incrml_predict, incrml_train, backtracking
from module.utils.yaml_parser import YamlParser
import argparse

def main():
    ### get & setting params
    config_path = r'config/taskconfig.yaml'
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default=config_path, 
                        help="任务配置文件路径")
    parser.add_argument("--type", "-t", default='predict', choices=['train', 'predict'],
                        help="Process type, train or predict. Default train")
    parser.add_argument("--task", default='Forecast1Day', help="Tasks for a single process")
    parser_config = parser.parse_args()

    ### init params
    processes_tyep  = parser_config.type
    taskname        = parser_config.task
    config          = YamlParser.parse(parser_config.config)
    
    task_incml = WindIncrml(config, taskname)
    if processes_tyep == 'train': 
        task_incml.incrml_train()
    elif processes_tyep == 'predict': 
        task_incml.incrml_predict()
    else: 
        raise("请输入正确的参数")

if __name__ == '__main__':
    main()