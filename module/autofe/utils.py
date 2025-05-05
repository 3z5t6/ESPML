"""
Automated Feature Engineering Utilities Module

This module provides various utility functions and constants for automated feature engineering,
including feature classification, feature space generation, and feature selection.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set, Union, Optional, Any
from collections import Counter
from module.utils.log import get_logger

# Configure logger
logger = get_logger(__name__)

class OPERATORCHAR:
    """
    Feature operator string constant class
    """
    feature_separator = '|||'  # Feature separator
    head_character = '###'    # Feature head marker
    tail_character = '$$$'    # Feature tail marker


# Operator type definitions, classified by feature type (c=categorical, n=numerical, t=time parameter)
OPERATORTYPES: Dict[Tuple[str, ...], List[str]] = {
    ('c',): ['count'],  # Single categorical feature operations
    ('n',): ['sine', 'cosine', 'pow', 'log'],  # Single numerical feature operations
    ('c', 'c'): ['crosscount', 'nunique', 'combine'],  # Two categorical features operations
    ('n', 'c'): ['aggmean', 'aggmax', 'aggmin', 'aggstd'],  # Numerical and categorical feature operations
    ('n', 'n'): ['add', 'sub', 'mul', 'div', 'std', 'maximize'],  # Two numerical features operations
    ('n', 't'): [  # Numerical feature and time parameter operations
        'diff', 'delay', 'ts_mean', 'ts_std', 'ts_cov', 'ts_corr', 'ts_max', 'ts_min', 'ts_rank',
        'ewm_mean', 'ewm_std', 'ewm_cov', 'ewm_corr'
    ]
}

# Generate a set of all operators
OPERTORS: Set[str] = set()
for key, value in OPERATORTYPES.items():
    OPERTORS.update(value)

# Default time window sizes
TIME_SPAN = [1, 2, 3, 4, 5]


def split_num_cat_features(df: pd.DataFrame, ignore_columns: List[str] = None) -> Tuple[List[str], List[str]]:
    """
    Split data into categorical and numerical features
    
    Note: Features in the ignore list will not be classified, such as target variables, time indices, etc.

    Args:
        df: Input DataFrame
        ignore_columns: List of column names to ignore, defaults to None
        
    Returns:
        Tuple containing lists of categorical and numerical feature names
    """
    if ignore_columns is None:
        ignore_columns = []
        
    cat_columns = []
    num_columns = []
    
    for column in df.columns:
        # Skip columns that should be ignored
        if column in ignore_columns:
            continue
            
        # Determine feature type
        # If the column is a string type or has 2 or fewer unique values, classify as categorical
        if str(df[column].dtypes) in ['String', 'object', 'category'] or df[column].nunique() <= 2:
            cat_columns.append(column)
        else:
            num_columns.append(column)
            
    return cat_columns, num_columns


def split_features(key: str) -> List[str]:
    """
    Parse feature name string, decomposing compound feature names into operators and parameters

    Args:
        key: Feature name string

    Returns:
        Parsed list: [operator, feature1] or [operator, feature1, feature2] or 
        [operator, feature1, feature2, weight_ratio], depending on the feature structure
        :type key: str
    """
    left, right = 0, 0
    separator1, separator2 = -1, -1
    stack = []
    left_count = 0
    i = 0
    
    # Traverse the string and parse feature structure
    while i < len(key):
        # Detect head marker '###'
        if i + 2 < len(key) and key[i] == '#' and key[i + 1] == '#' and key[i + 2] == '#':
            left_count += 1
            if left_count == 1:
                left = i
            stack.append('(')
            i += 2
        # Detect tail marker '$$$'
        elif i + 2 < len(key) and key[i] == '$' and key[i + 1] == '$' and key[i + 2] == '$':
            if stack:  # Prevent pop from an empty stack
                stack.pop()
            if len(stack) == 0:
                right = i
            i += 2
        # Detect separator '|||'
        elif (i + 2 < len(key) and key[i] == '|' and key[i + 1] == '|' and 
              key[i + 2] == '|' and len(stack) == 1):
            if separator1 < 0:
                separator1 = i
            else:
                separator2 = i
        i += 1
    
    # Return different format lists based on parsing results
    if separator2 < 0:
        if separator1 != -1:
            # Form: operator###feature1|||feature2$$$
            return [key[:left], key[left + 3:separator1], key[separator1 + 3:right]]
        # Form: operator###feature1$$$
        return [key[:left], key[left + 3:right]]
    # Form: operator###feature1|||feature2|||weight_ratio$$$
    return [key[:left], key[left + 3:separator1], key[separator1 + 3:separator2], key[separator2 + 3:right]]


def is_combination_feature(feature_name: str) -> bool:
    """
    Determine if a feature name is a compound feature
    
    Compound features contain both head and tail markers
    
    Args:
        feature_name: Feature name string to check
        
    Returns:
        True if it's a compound feature, False otherwise
    """
    # Count special characters in the feature name
    count = Counter(feature_name)
    # Check if it contains both head and tail markers
    # Each marker consists of three identical characters, so divide by 3
    return min(count.get('#', 0) // 3, count.get('$', 0) // 3) > 0


def feature_space(
    df: pd.DataFrame, 
    target_name: Optional[str] = None, 
    already_selected: List[str] = None,
    time_index: Optional[str] = None,
    time_span: Optional[List[int]] = None, 
    group_index: Optional[str] = None
) -> List[str]:
    """
    Generate feature space containing various possible feature combinations

    Args:
        df: Input DataFrame
        target_name: Target variable name, defaults to None
        already_selected: List of already selected features, defaults to None
        time_index: Time index column name, defaults to None
        time_span: List of time window sizes, defaults to None
        group_index: Group index column name, defaults to None

    Returns:
        List of candidate features
    """
    # Initialize parameters
    if already_selected is None:
        already_selected = []
        
    # Use strings from the constant class
    feature_separator = OPERATORCHAR.feature_separator
    head_character = OPERATORCHAR.head_character
    tail_character = OPERATORCHAR.tail_character
    
    # Separate categorical and numerical features, ignoring target variable, time index and group index
    ignore_columns = [col for col in [time_index, group_index, target_name] if col is not None]
    cat_columns, num_columns = split_num_cat_features(df, ignore_columns)
    
    # Initialize a candidate feature list and maximum length
    candidate_feature = []
    candidate_maxlenth = 128
    
    # Set time window sizes
    if time_span is None:
        time_span = TIME_SPAN
    
    # Generate features related to categorical features
    for i in range(len(cat_columns)):
        # Single categorical feature operations
        for op_name in OPERATORTYPES[('c',)]:
            # Generate feature name
            name = (op_name + head_character + '{}' + tail_character).format(
                feature_separator.join([cat_columns[i]])
            )
            # Check if the feature name is valid and not already selected
            if (name not in already_selected and 
                    min(Counter(name).get('#', 0) // 3, Counter(name).get('$', 0) // 3) < 3 and 
                    len(candidate_feature) < candidate_maxlenth):
                candidate_feature.append(name)
                
        # Combinations of two categorical features
        for j in range(i + 1, len(cat_columns)):
            for op_name in OPERATORTYPES[('c', 'c')]:
                name = (op_name + head_character + '{}' + feature_separator + '{}' + 
                       tail_character).format(cat_columns[i], cat_columns[j])
                if (name not in already_selected and 
                        min(Counter(name).get('#', 0) // 3, Counter(name).get('$', 0) // 3) < 3 and 
                        len(candidate_feature) < candidate_maxlenth):
                    candidate_feature.append(name)
    
    # Generate features related to numerical features
    for i in range(len(num_columns)):
        # Single numerical feature operations
        for op_name in OPERATORTYPES[('n',)]:
            name = (op_name + head_character + '{}' + tail_character).format(
                feature_separator.join([num_columns[i]])
            )
            if (name not in already_selected and 
                    min(Counter(name).get('#', 0) // 3, Counter(name).get('$', 0) // 3) < 3 and 
                    len(candidate_feature) < candidate_maxlenth):
                candidate_feature.append(name)
                
        # Combinations of numerical and categorical features
        for j in range(len(cat_columns)):
            for op_name in OPERATORTYPES[('n', 'c')]:
                name = (op_name + head_character + '{}' + feature_separator + '{}' + 
                       tail_character).format(num_columns[i], cat_columns[j])
                if (name not in already_selected and 
                        min(Counter(name).get('#', 0) // 3, Counter(name).get('$', 0) // 3) < 3 and 
                        len(candidate_feature) < candidate_maxlenth):
                    candidate_feature.append(name)
                    
        # Combinations of two numerical features
        for j in range(i + 1, len(num_columns)):
            for op_name in OPERATORTYPES[('n', 'n')]:
                name = (op_name + head_character + '{}' + feature_separator + '{}' + 
                       tail_character).format(num_columns[i], num_columns[j])
                if (name not in already_selected and 
                        min(Counter(name).get('#', 0) // 3, Counter(name).get('$', 0) // 3) < 3 and 
                        len(candidate_feature) < candidate_maxlenth):
                    candidate_feature.append(name)
                    
        # Generate time-related features if the time index exists
        if time_index is not None:
            for op_name in OPERATORTYPES[('n', 't')]:
                for span in time_span:
                    name = (op_name + head_character + '{}' + feature_separator + 
                           '{}' + feature_separator + str(span) + tail_character).format(
                        num_columns[i], time_index
                    )
                    if (name not in already_selected and 
                            min(Counter(name).get('#', 0) // 3, Counter(name).get('$', 0) // 3) < 3 and 
                            len(candidate_feature) < candidate_maxlenth):
                        candidate_feature.append(name)
    
    return candidate_feature


def feature2table(names: List[str]) -> pd.DataFrame:
    """
    Convert a list of feature names to a feature table
    
    Args:
        names: List of feature names
        
    Returns:
        DataFrame containing feature IDs and LaTeX formulas
    """
    res = []
    for i, name in enumerate(names):
        temp = [i + 1]
        new_name = name2formula(name)
        temp.append(new_name)
        res.append(temp)
    column_names = ['id', 'latex']
    df = pd.DataFrame(res, columns=column_names)
    return df


def name2formula(input_str: str) -> str:
    """
    Convert feature names to corresponding calculation formulas

    Args:
        input_str: Feature name string

    Returns:
        Calculation formula string
    """
    # If not a compound feature, return the original string
    if not is_combination_feature(input_str):
        return input_str
        
    # Parse feature name
    temp = split_features(input_str)
    
    # Handle underscore escaping
    for i in range(1, len(temp)):
        j = 0
        while not is_combination_feature(temp[i]) and j < len(temp[i]):
            if temp[i][j] == '_':
                temp[i] = temp[i][:j] + '\\' + temp[i][j:]
                j += 1
            j += 1
    
    # 根据操作符类型生成公式
    formula = ''
    
    # 加法操作
    if temp[0] == 'add':
        if len(temp) >= 3:
            # Process first operand
            if is_combination_feature(temp[1]):
                formula += '(' + name2formula(temp[1]) + ')' + '+'
            else:
                formula += temp[1] + '+'
            
            # Process second operand
            if is_combination_feature(temp[2]):
                formula += '(' + name2formula(temp[2]) + ')'
            else:
                formula += temp[2]
            return formula
    
    # 减法操作
    elif temp[0] == 'sub':
        if len(temp) >= 3:
            # Process first operand
            if is_combination_feature(temp[1]):
                formula += '(' + name2formula(temp[1]) + ')' + '-'
            else:
                formula += temp[1] + '-'
            
            # Process second operand
            if is_combination_feature(temp[2]):
                formula += '(' + name2formula(temp[2]) + ')'
            else:
                formula += temp[2]
            return formula
    
    # 乘法操作
    elif temp[0] == 'mul':
        if len(temp) >= 3:
            # Process first operand
            if is_combination_feature(temp[1]):
                formula += '(' + name2formula(temp[1]) + ')' + '\\times '
            else:
                formula += temp[1] + '\\times '
            
            # Process second operand
            if is_combination_feature(temp[2]):
                formula += '(' + name2formula(temp[2]) + ')'
            else:
                formula += temp[2]
            return formula
    
    # 除法操作
    elif temp[0] == 'div':
        if len(temp) >= 3:
            # Process first operand
            if is_combination_feature(temp[1]):
                formula += '(' + name2formula(temp[1]) + ')' + '\\div '
            else:
                formula += temp[1] + '\\div '
            
            # Process second operand
            if is_combination_feature(temp[2]):
                formula += '(' + name2formula(temp[2]) + ')'
            else:
                formula += temp[2]
            return formula
    
    # 平方操作
    elif temp[0] == 'pow':
        if len(temp) >= 2:
            # Process operand
            if is_combination_feature(temp[1]):
                formula += '(' + name2formula(temp[1]) + ')' + '^2'
            else:
                formula += temp[1] + '^2'
            return formula
    
    # 其他操作符
    elif temp[0] in OPERTORS - {'add', 'mul', 'sub', 'div', 'pow'}:
        # 将特征名转换为函数调用格式
        formula = input_str.replace(OPERATORCHAR.head_character, '(').replace(
            OPERATORCHAR.tail_character, ')').replace(OPERATORCHAR.feature_separator, ',')
        return formula
    
    # 如果没有匹配的操作符，返回原始字符串
    return input_str


def ginic(actual: np.ndarray, pred: np.ndarray) -> float:
    """Denormalized gini calculation.

    Args:
        actual: True values.
        pred: Predicted values.

    Returns:
        Metric value.

    """
    actual = np.asarray(actual)
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    gini_sum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return gini_sum / n


def gini_normalizedc(a: np.ndarray, p: np.ndarray) -> float:
    """Calculated normalized gini.

    Args:
        a: True values.
        p: Predicted values.

    Returns:
        Metric value.

    """
    return ginic(a, p) / ginic(a, a)


def gini_normalized(y_true: np.ndarray, y_pred: np.ndarray, empty_slice: Optional[np.ndarray] = None) -> float:
    """Calculate normalized gini index.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        empty_slice: Mask.

    Returns:
        Gini value.

    """
    if not np.issubdtype(y_pred.dtype, np.number):
        return 0.0
    if empty_slice is None:
        empty_slice = np.isnan(y_pred)
    elif empty_slice.all():
        return 0.0
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    if empty_slice.ndim > 1:
        empty_slice = empty_slice[:, 0]
    sl = ~empty_slice
    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]
    assert y_pred.shape[1] == 1 or y_true.shape[1] == y_pred.shape[
        1], 'Shape missmatch. Only calculate NxM vs NxM or Nx1 vs NxM'
    outp_size = y_true.shape[1]
    ginis = np.zeros((outp_size,), dtype=np.float32)
    for i in range(outp_size):
        j = min(i, y_pred.shape[1] - 1)
        yt = y_true[:, i][sl]
        yp = y_pred[:, j][sl]
        ginis[i] = gini_normalizedc(yt, yp)
    return np.abs(ginis).mean()


def calc_ginis(data: np.ndarray, target: np.ndarray, empty_slice: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate ginis for an array of preditions.

    Args:
        data: np.ndarray, 2D array of features.
        target: np.ndarray, 1D, or 2D array of target values.
        empty_slice: np.ndarray, should be 2D array with shape matching data if provided.

    Returns:
        np.ndarray: Array of gini scores for each feature.

    """
    scores = np.zeros(data.shape[1])
    for n in range(data.shape[1]):
        sl = None
        if empty_slice is not None:
            # 确保empty_slice是二维数组，防止索引错误
            if empty_slice.ndim < 2:
                raise ValueError("empty_slice must be a 2D array with shape matching data")
            sl = empty_slice[:, n]
        scores[n] = gini_normalized(target, data[:, n], empty_slice=sl)
    return scores


def normalize_gini_select(df: Union[pd.DataFrame, pd.Series], y_train: pd.Series, ori_columns: List[str]) -> Tuple[List[str], Dict[str, float]]:
    """
    Select best features based on Gini coefficient
    
    Calculate Gini coefficient for each feature and select top 10 best features
    
    Args:
        df: Input feature DataFrame or Series
        y_train: Target variable
        ori_columns: List of original column names to exclude existing features
        
    Returns:
        Tuple containing list of best features and dictionary of feature Gini coefficients
    """
    best_feature = []
    # Calculate Gini coefficients for all features, explicitly passing None as empty_slice parameter
    train_ginis = calc_ginis(df.to_numpy(), target=y_train.to_numpy(), empty_slice=None)
    # Create dictionary of feature names and Gini coefficients
    dt = dict(zip(df.columns, train_ginis))
    # Sort by Gini coefficient in descending order
    dt_sorted = dict(sorted(dt.items(), key=lambda x: x[1], reverse=True))
    
    # Select top 10 best features (excluding original features)
    for item in dt_sorted.items():
        if len(best_feature) >= 10:
            break
        if item[0] not in ori_columns:
            best_feature.append(item[0])
            
    return best_feature, dt


def update_time_span(a: Any) -> Optional[List[int]]:
    """
    Convert different types of time window parameters to a list of integers
    
    Args:
        a: Time window parameter, can be an integer, list, or string
        
    Returns:
        Converted list of integers or None
    """
    # If parameter is None, return None directly
    if a is None:
        return None
        
    # If it's an integer, convert to a single-element list
    if isinstance(a, int):
        return [a]
        
    # If it's a list, convert each element to integer
    if isinstance(a, list):
        return [int(n) for n in a]
        
    # If it's a string
    if isinstance(a, str):
        # If it contains commas, split and convert to list of integers
        if ',' in a:
            return [int(n) for n in a.split(',')]
            
        # Try to convert the string to a single integer
        try:
            return [int(a)]
        except Exception as e:
            # Record error and return original value if conversion fails
            print(f"Time window parameter conversion error: {str(e)}")
            return a
            
    # If it's another type, return the original value
    return a