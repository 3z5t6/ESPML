import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import pandas as pd
from module.utils.wind_incrml import WindIncrml, YamlParser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Tasks config path.", default='taskconfig_jilin.yaml')
    parser.add_argument("--task", "-t", help="Tasks name.", default='Forecast4Hour')
    parser.add_argument("--stime", "-s",  help="The start time of the backtracking task.", default='2025-01-01') 
    parser.add_argument("--etime", "-e",  help="The end time of the backtracking task.", default='2025-01-31')
    parser.add_argument("--only_predict", help="Only predict.", action="store_true")
    parser_config = parser.parse_args()

    start_time = parser_config.stime
    end_time = parser_config.etime
    taskname = parser_config.task
    config = YamlParser().parse(parser_config.config)
    
    if taskname == 'all':
        tasknames = list(config['Task'].keys())
        for d in pd.date_range(start=start_time, end=end_time, freq='D'):
            for taskname in tasknames:
                if taskname.endswith("Day"):
                    wi = WindIncrml(config, taskname, limit_model_num=5)
                    wi.backtracking(start_time=str(d.date()), end_time=str(d.date()), only_predict=parser_config.only_predict)
    elif taskname == 'Forecast4Hour':
        for d in pd.date_range(start=start_time, end=end_time, freq='D'):
            wi = WindIncrml(config, taskname, limit_model_num=5) 
            wi.backtracking(start_time=str(d.date()), end_time=str(d.date()), only_predict=parser_config.only_predict)
    elif taskname == 'daily':
        for d in pd.date_range(start=start_time, end=end_time, freq='D'):
            for taskname in ['Forecast1Day', 'Forecast2Day']:
                wi = WindIncrml(config, taskname, limit_model_num=5)
                wi.backtracking(start_time=str(d.date()), end_time=str(d.date()), only_predict=parser_config.only_predict)
    else:
        wi = WindIncrml(config, taskname, limit_model_num=5)
        wi.backtracking(start_time=start_time, end_time=end_time, only_predict=parser_config.only_predict)