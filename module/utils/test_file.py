import glob
import os
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from typing import Union
from module.utils.yaml_parser import YamlParser
from module.dataprocess.data_processor import DataProcess
from module.utils.result_saver import ResultSaver

import warnings
warnings.filterwarnings('ignore')


model = None
seq = None
res_path = ""
adv_features = []

def check_and_update_model(config:dict, seq:int) -> None:
    res_path = os.path.join(
        os.getcwd(), 
        'results',
        str(config["AuthorName"]),
        str(config["TaskName"]),
        str(config["JobID"])
        ) if config['IncrML'].get('SaveModelPath', None) is None else config['IncrML']['SaveModelPath']
    if seq == -1:
        seq = max([int(num) for num in os.listdir(res_path)])
        while not os.path.exists(os.path.join(config['IncrML']['SaveModelPath'], str(seq), 'automl', 'val_res')) and seq > 0:
            seq -= 1
    current_res = os.path.join(res_path, str(seq))
    if seq <= 0: raise FileNotFoundError("Please make sure model result generated.")
    if res_path != current_res:
        # update model
        # res_path = current_res
        data_dict = ResultSaver.load_json(res_path, str(seq), 'autofe', 'result')
        adv_features = data_dict['combination_features']
        model = ResultSaver.load_pickle(res_path, str(seq), 'automl', 'result')
    return model, adv_features, str(seq)

def data_process(config:dict, 
                 input_obj:Union[str, pd.DataFrame], seq:int) \
                    -> tuple[pd.DataFrame, object, object, str, str]:
    """
    test data --> data process --> name2feature --> flaml --> predict --> save

    Args:
        input_file_path (str): _description_
    """
    model, adv_features, seq = check_and_update_model(config, seq)
    base_dir = os.path.join(
        os.getcwd(), 'results',  
        str(config["AuthorName"]),
        str(config["TaskName"]),
        str(config["JobID"]),
        seq
    ) if config['IncrML'].get('SaveModelPath', None) is None else os.path.join(config['IncrML']['SaveModelPath'], seq)
    test_res_path = os.path.join(base_dir, 'automl', 'test_res')
    if os.path.exists(test_res_path):
        os.remove(test_res_path)

    ### load data & result
    if isinstance(input_obj, str):
        df = pd.read_csv(f'{input_obj}', encoding="utf-8")
    else:
        df = input_obj
    # load labelencode dict
    transformer_path = os.path.join(base_dir, 'autofe', 'labelencoder.pkl')
    with open(transformer_path, 'rb') as f:
        transformer = pickle.load(f)

    ### init params
    time_index = colname_filter(config['Feature'].get('TimeIndex', None))
    target_name = colname_filter(config["Feature"]["TargetName"])
    group_id = colname_filter(config['Feature'].get('GroupIndex', None))
    ignore_columns = colname_filter(config['Feature'].get('IgnoreFeature', []))
    ignore_columns = [col for col in ignore_columns if col not in [time_index, group_id] and col in df.columns]
    category_feature = colname_filter(config['Feature'].get('CategoricalFeature', []))
    automl_running = config['AutoML'].get('Running', True)

    ### data processing
    # replace illegal characters
    df = colname_filter(df)

    # percent sign conversion
    df = percent_sign_conversion(df)

    # convert to category
    for f in category_feature:
        if f in df.columns: df[f] = df[f].astype('category')

    # transform time index to the pd.timedate
    if time_index is not None:
        df[time_index] = pd.to_datetime(df[time_index])
        df.sort_values(by=time_index, inplace=True)
        # df.drop(columns=[time_index], inplace=True)

    # dorp columns
    df = df.drop(columns=ignore_columns)

    # check high nan drop feature 
    features = model.feature_names_in_ if automl_running else model.feature_name()
    features = [fes for fes in df.columns if fes in features]
    if target_name in df.columns: features = features + [target_name]
    df = df[features]

    # feature combine
    df = transformer.transform(df, adv_features, labelencode=True)

    return df, model, transformer, target_name, base_dir

def predict_proba(config:dict, input_obj:Union[str, pd.DataFrame], 
                  save:bool=True) -> Union[pd.Series, pd.DataFrame]:
    ### config check
    assert config['Feature']['TaskType'] == 'classification', 'TaskType must be classification'
    
    ### data processing
    df, model, _, target_name, base_dir = data_process(config, input_obj)

    ### predict
    if target_name in df.columns:
        output_df = pd.DataFrame(model.predict_proba(df.drop(columns=[target_name]))).round(5)
        test_res = pd.concat([output_df.reset_index(drop=True), 
                              df.reset_index(drop=True)[target_name],
                              df.reset_index(drop=True).drop(columns=[target_name])], axis=1)
    else:
        output_df = pd.DataFrame(model.predict_proba(df)).round(5)
        test_res = pd.concat([output_df, df], axis=1)

    # save
    if save:
        ResultSaver.save_csv(test_res, False, base_dir, "automl", "test_proba_res")

    return output_df

def predict(config:dict, input_obj:Union[str, pd.DataFrame], save:bool=True, seq:int=-1) -> pd.Series:
    ### data processing
    df, model, transformer, target_name, base_dir = data_process(config, input_obj, seq)

    ### predict
    if target_name in df.columns:
        output_df = pd.DataFrame(model.predict(df.drop(columns=[target_name])), columns=["pred"])
        test_res = pd.concat([df.reset_index(drop=True)[target_name],
                                output_df.reset_index(drop=True),
                                df.reset_index(drop=True).drop(columns=[target_name])], axis=1)
        test_res.columns = [target_name] + ['ypred'] + df.drop(columns=[target_name]).columns.tolist()
    else:
        output_df = pd.DataFrame(model.predict(df))
        test_res = pd.concat([output_df.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
        test_res.columns = ['ypred'] + df.columns.tolist()

    # save
    if target_name in transformer._record.keys():
        test_res['ypred'] = transformer._record[target_name].inverse_transform(test_res['ypred'])
        try: test_res[target_name] = transformer._record[target_name].inverse_transform(test_res[target_name])
        except: pass
    if save:
        ResultSaver.save_csv(test_res, False, base_dir, "automl", "test_res")

    return test_res['ypred']

def colname_filter(obj:Union[list[str], str, pd.DataFrame]) -> Union[list[str], str, pd.DataFrame]:
    replace_str = [':', '[', ']', '（', '）', '！', '＠', '＃', '￥', '％', '…', '《', '》', '【', '】']
    if isinstance(obj, list):
        new_obj = [col.strip().replace(' ', '_') for col in obj]
        for s in replace_str: new_obj = [col.strip().replace(s, '_') for col in new_obj]
    elif isinstance(obj, str):
        new_obj = obj
        for s in replace_str: new_obj = new_obj.strip().replace(s, '_')
    elif isinstance(obj, pd.DataFrame):
        his_col = obj.columns.tolist()
        new_col = [col.strip().replace(' ', '_') for col in his_col]
        replace_str = [':', '[', ']', '（', '）', '！', '＠', '＃', '￥', '％', '…', '《', '》', '【', '】']
        for s in replace_str: new_col = [col.replace(s, '_') for col in new_col]
        obj.columns = new_col
        new_obj = obj
    else: new_obj = obj
    return new_obj

def percent_sign_conversion(df:pd.DataFrame):
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    result_df = pd.DataFrame()
    for column in df.columns:
        column_dtype = df[column].dtype
        if column_dtype == 'object' or column_dtype.name == 'category':
            column_data = df[column].astype(str)
            contains_percentage = column_data.str.endswith('%')
            if contains_percentage.any():
                converted_data = column_data.str.replace('%', '').astype(float) / 100
                result_df[column] = converted_data
            else:
                result_df[column] = df[column]
        else:
            result_df[column] = df[column]
    return result_df

if __name__ == "__main__":
    config_file = 'benchmark/0926/config.yaml'
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    # seq = sys.argv[2] if len(sys.argv) > 2 else None

    config = YamlParser.parse(config_file)
    predict(config, sys.argv[2])