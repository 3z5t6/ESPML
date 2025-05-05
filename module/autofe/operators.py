import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Any
from sklearn.base import BaseEstimator, TransformerMixin
from module.utils.log import get_logger

# Configure logger
logger = get_logger(__name__)

def split_num_cat_features(df: pd.DataFrame, target_name: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """
    Split data into categorical and numerical features
    
    Note: The target variable does not belong to any feature type

    Args:
        df: Input DataFrame
        target_name: Target variable name, default is None
        
    Returns:
        Tuple of lists: (categorical features, numerical features)
    """
    cat_columns = []
    num_columns = []
    
    for column in df.columns:
        if column == target_name:
            continue
            
        # Determine feature type: string type or unique value count less than or equal to 2 is categorical feature
        if str(df[column].dtype) in ['String', 'object', 'category'] or len(df[column].unique()) <= 2:
            cat_columns.append(column)
        else:
            num_columns.append(column)
            
    return cat_columns, num_columns


def split_features(key: str) -> List[str]:
    """
    Parse feature name

    Args:
        key: Feature name string

    Returns:
        List: [operator, feature1, feature2, weight]
    """
    left, right = 0, 0
    separator1, separator2 = -1, -1
    stack = []
    left_count = 0
    i = 0
    
    # Parse feature string
    while i < len(key):
        # Process start marker '###'
        if i + 2 < len(key) and key[i] == '#' and key[i + 1] == '#' and key[i + 2] == '#':
            left_count += 1
            if left_count == 1:
                left = i
            stack.append('(')
            i += 2
        # Process end marker '$$$'
        elif i + 2 < len(key) and key[i] == '$' and key[i + 1] == '$' and key[i + 2] == '$':
            stack.pop()
            if len(stack) == 0:
                right = i
            i += 2
        # Process separator '|||'
        elif (i + 2 < len(key) and key[i] == '|' and key[i + 1] == '|' and 
              key[i + 2] == '|' and len(stack) == 1):
            if separator1 < 0:
                separator1 = i
            else:
                separator2 = i
        i += 1
    
    # Return different formats based on the number of separators
    if separator2 < 0:
        if separator1 != -1:
            return [key[:left], key[left + 3:separator1], key[separator1 + 3:right]]
        return [key[:left], key[left + 3:right]]
    return [key[:left], key[left + 3:separator1], key[separator1 + 3:separator2], key[separator2 + 3:right]]


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Frequency encoder, converts categorical features to their frequency in the training set
    
    This encoder inherits from sklearn's BaseEstimator and TransformerMixin, and can be used in feature engineering pipelines
    """

    def __init__(self, columns: Optional[List[str]] = None, normalize: bool = False):
        """
        Initialize frequency encoder
        
        Args:
            columns: List of columns to encode, default is None (process all columns)
            normalize: Whether to normalize frequency, default is False
        """
        self.columns = columns
        self.frequencies_: Optional[Dict[str, Dict]] = None
        self.normalize = normalize

    def fit(self, X: Union[pd.DataFrame, pd.Series], y: Optional[Any] = None) -> 'FrequencyEncoder':
        """
        Fit the encoder, calculate the frequency of each categorical value
        
        Args:
            X: Input data, can be DataFrame or Series
            y: Target variable, not used but kept for compatibility with sklearn API
            
        Returns:
            Fitted encoder instance
        """
        if isinstance(X, pd.DataFrame):
            # If no columns specified, use all columns
            if self.columns is None:
                self.columns = list(X.columns)
                
            # Calculate the frequency of each value in each column
            self.frequencies_ = {}
            for col in self.columns:
                freqs = X[col].value_counts(normalize=self.normalize).to_dict()
                self.frequencies_[col] = freqs
        elif isinstance(X, pd.Series):
            # Handle Series case
            self.frequencies_ = X.value_counts(normalize=self.normalize).to_dict()
            
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Convert categorical features to their frequency
        
        Args:
            X: Input data, can be DataFrame or Series
            
        Returns:
            Transformed data
        """
        if isinstance(X, pd.DataFrame):
            x_encoded = X.copy()
            for col, freqs in self.frequencies_.items():
                x_encoded[col] = x_encoded[col].map(freqs).fillna(0)
        elif isinstance(X, pd.Series):
            x_encoded = X.map(self.frequencies_).fillna(0)
            
        # Ensure only specified columns are returned
        if isinstance(x_encoded, pd.DataFrame) and self.columns is not None:
            return x_encoded[self.columns]
        return x_encoded


def count(features: Union[pd.Series, pd.DataFrame], intermediate: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """
    Calculate the frequency of feature values
    
    Args:
        features: Input features, can be Series or DataFrame
        intermediate: Whether to return intermediate statistics, default is False
        
    Returns:
        If intermediate is False, return count feature
        If intermediate is True, return (count feature, intermediate statistics)
    """
    # If input is DataFrame, take the first column
    if isinstance(features, pd.DataFrame):
        features = features.iloc[:, 0]
        
    # Calculate the frequency of each value
    intermediate_stat = features.value_counts().rename('count')
    
    # Merge the count result with the original feature
    feature = pd.DataFrame(features).merge(intermediate_stat, on=features.name, how='left')['count']
    
    if intermediate:
        return feature, intermediate_stat
    return feature


def sine(feature: pd.Series) -> pd.Series:
    """
    Calculate the sine of the feature
    
    Args:
        feature: Input feature
        
    Returns:
        Sine transformed feature
    """
    return pd.Series(np.sin(feature), index=feature.index)


def cosine(feature: pd.Series) -> pd.Series:
    """
    Calculate the cosine of the feature
    
    Args:
        feature: Input feature
        
    Returns:
        Cosine transformed feature
    """
    return pd.Series(np.cos(feature), index=feature.index)


def softmax(feature: pd.Series) -> pd.Series:
    """
    Apply softmax function to the feature
    
    Args:
        feature: Input feature
        
    Returns:
        Softmax transformed feature
    """
    exp_values = np.exp(feature)
    return pd.Series(exp_values / exp_values.sum(), index=feature.index)


def sigmoid(feature: pd.Series) -> pd.Series:
    """
    Apply sigmoid function to the feature
    
    Args:
        feature: Input feature
        
    Returns:
        Sigmoid transformed feature
    """
    return pd.Series(1 / (1 + np.exp(-feature)), index=feature.index)


def relu(feature: pd.Series) -> pd.Series:
    """
    Apply ReLU function to the feature
    
    Args:
        feature: Input feature
        
    Returns:
        ReLU transformed feature
    """
    return pd.Series(np.maximum(feature, 0), index=feature.index)


def tanh(feature: pd.Series) -> pd.Series:
    """
    Apply hyperbolic tangent function to the feature
    
    Args:
        feature: Input feature
        
    Returns:
        tanh transformed feature
    """
    return pd.Series(np.tanh(feature), index=feature.index)


def pow(feature: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate the square of the feature
    
    Args:
        feature: Input feature
        
    Returns:
        Square transformed feature, with clipping to prevent overflow
    """
    feature = feature.pow(2)
    # Clip values to prevent overflow
    feature = feature.clip(lower=-10000000.0, upper=10000000.0)
    return feature


def log(feature: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """
    Calculate the natural logarithm of the feature
    
    Args:
        feature: Input feature
        
    Returns:
        Log transformed feature, with clipping to prevent overflow
    """
    # Take absolute value to prevent negative log
    feature = pd.Series(np.log(np.abs(feature)))
    # Clip values to prevent overflow
    feature = feature.clip(lower=-10000000.0, upper=10000000.0)
    return feature


def crosscount(features: pd.DataFrame, intermediate: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """
    Calculate the frequency of feature combinations
    
    Args:
        features: Input feature DataFrame, containing multiple columns
        intermediate: Whether to return intermediate statistics, default is False
        
    Returns:
        If intermediate is False, return count feature
        If intermediate is True, return (count feature, intermediate statistics)
    """
    feature_names = list(features.columns)
    
    # Calculate the frequency of each feature combination
    intermediate_stat = features.groupby(feature_names).size().rename('crosscount')
    
    # Merge the count result with the original feature
    feature = features.merge(intermediate_stat, on=list(feature_names), how='left')['crosscount']
    
    if intermediate:
        return feature, intermediate_stat
    return feature


def nunique(features: pd.DataFrame, intermediate: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """
    Calculate the number of unique values in the second column grouped by the first column
    
    Args:
        features: Input feature DataFrame, containing at least 2 columns
        intermediate: Whether to return intermediate statistics, default is False
        
    Returns:
        If intermediate is False, return count feature
        If intermediate is True, return (count feature, intermediate statistics)
    """
    feature_names = list(features.columns)
    
    # Calculate the number of unique values in the second column grouped by the first column
    intermediate_stat = features.groupby(feature_names[0])[feature_names[1]].nunique().rename('nunique')
    
    # Merge the count result with the original feature
    feature = features.merge(intermediate_stat, on=feature_names[0], how='left')['nunique']
    
    if intermediate:
        return feature, intermediate_stat
    return feature


def combine(features: pd.DataFrame, intermediate: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]:
    """
    Combine two columns of features into a new categorical feature
    
    Args:
        features: Input feature DataFrame, containing at least 2 columns
        intermediate: Whether to return intermediate statistics, default is False
        
    Returns:
        If intermediate is False, return combined feature
        If intermediate is True, return (combined feature, intermediate statistics)
    """
    # Convert two columns to strings and connect with an underscore
    feature_combined = (features.iloc[:, 0].astype(str) + '_' + features.iloc[:, 1].astype(str)).astype('category')
    
    # Convert the categorical encoding to a numeric value
    feature = feature_combined.cat.codes.rename('combine')
    
    if intermediate:
        # Create intermediate statistics, containing original features and combined result
        intermediate_stat = pd.concat([features, feature], axis=1).set_index(list(features.columns)).drop_duplicates()
        return feature, intermediate_stat
    return feature


def aggmean(features: pd.DataFrame, intermediate: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """
    Calculate the mean of the second column grouped by the first column
    
    Args:
        features: Input feature DataFrame, containing at least 2 columns
        intermediate: Whether to return intermediate statistics, default is False
        
    Returns:
        If intermediate is False, return aggregated mean feature
        If intermediate is True, return (aggregated mean feature, intermediate statistics)
    """
    feature_name = list(features.columns)
    
    # Calculate the mean of the second column grouped by the first column
    intermediate_stat = features.groupby(feature_name[0]).agg('mean')[feature_name[1]].rename('aggmean')
    
    # Merge the count result with the original feature
    feature = features.merge(intermediate_stat, on=feature_name[0], how='left')['aggmean']
    
    if intermediate:
        return feature, intermediate_stat
    return feature


def aggmin(features: pd.DataFrame, intermediate: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """
    Calculate the minimum of the second column grouped by the first column
    
    Args:
        features: Input feature DataFrame, containing at least 2 columns
        intermediate: Whether to return intermediate statistics, default is False
        
    Returns:
        If intermediate is False, return aggregated minimum feature
        If intermediate is True, return (aggregated minimum feature, intermediate statistics)
    """
    feature_name = list(features.columns)
    
    # Calculate the minimum of the second column grouped by the first column
    intermediate_stat = features.groupby(feature_name[0]).agg('min')[feature_name[1]].rename('aggmin')
    
    # Merge the count result with the original feature
    feature = features.merge(intermediate_stat, on=feature_name[0], how='left')['aggmin']
    
    if intermediate:
        return feature, intermediate_stat
    return feature


def aggmax(features: pd.DataFrame, intermediate: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """
    Calculate the maximum of the second column grouped by the first column
    
    Args:
        features: Input feature DataFrame, containing at least 2 columns
        intermediate: Whether to return intermediate statistics, default is False
        
    Returns:
        If intermediate is False, return aggregated maximum feature
        If intermediate is True, return (aggregated maximum feature, intermediate statistics)
    """
    feature_name = list(features.columns)
    
    # Calculate the maximum of the second column grouped by the first column
    intermediate_stat = features.groupby(feature_name[0]).agg('max')[feature_name[1]].rename('aggmax')
    
    # 将统计结果合并到原始特征
    feature = features.merge(intermediate_stat, on=feature_name[0], how='left')['aggmax']
    
    if intermediate:
        return feature, intermediate_stat
    return feature


def aggstd(features: pd.DataFrame, intermediate: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """
    Calculate the standard deviation of the second column grouped by the first column
    
    Args:
        features: Input feature DataFrame, containing at least 2 columns
        intermediate: Whether to return intermediate statistics, default is False
        
    Returns:
        If intermediate is False, return aggregated standard deviation feature
        If intermediate is True, return (aggregated standard deviation feature, intermediate statistics)
    """
    feature_name = list(features.columns)
    
    # Calculate the standard deviation of the second column grouped by the first column
    intermediate_stat = features.groupby(feature_name[0]).agg('std')[feature_name[1]].rename('aggstd')
    
    # Merge the count result with the original feature
    feature = features.merge(intermediate_stat, on=feature_name[0], how='left')['aggstd']
    
    if intermediate:
        return feature, intermediate_stat
    return feature


def add(features: pd.DataFrame) -> pd.Series:
    """
    Calculate the sum of two columns of features
    
    Args:
        features: Input feature DataFrame, containing at least 2 columns
        
    Returns:
        The sum of two columns of features
    """
    feature_name = list(features.columns)
    return features[feature_name[0]] + features[feature_name[1]]


def sub(features: pd.DataFrame) -> pd.Series:
    """
    Calculate the difference between two columns of features
    
    Args:
        features: Input feature DataFrame, containing at least 2 columns
        
    Returns:
        The difference between the first column and the second column
    """
    feature_name = list(features.columns)
    return features[feature_name[0]] - features[feature_name[1]]


def mul(features: pd.DataFrame) -> pd.Series:
    """
    Calculate the product of two columns of features
    
    Args:
        features: Input feature DataFrame, containing at least 2 columns
        
    Returns:
        The product of two columns of features, with clipping to prevent overflow
    """
    feature_name = list(features.columns)
    # Calculate the product of two columns of features
    feature = features[feature_name[0]] * features[feature_name[1]]
    # Clip the value range to prevent overflow
    feature = feature.clip(lower=-10000000.0, upper=10000000.0)
    return feature


def div(features: pd.DataFrame) -> pd.Series:
    """
    Calculate the division result of two columns of features
    
    Args:
        features: Input feature DataFrame, containing at least 2 columns
        
    Returns:
        The division result of the first column divided by the second column, with clipping to prevent overflow
    """
    feature_name = list(features.columns)
    # Calculate the division result of two columns of features
    feature = features[feature_name[0]] / features[feature_name[1]]
    # Clip the value range to prevent overflow
    feature = feature.clip(lower=-10000000.0, upper=10000000.0)
    return feature


def std(features: pd.DataFrame) -> pd.Series:
    """
    Calculate the standard deviation of each row of features
    
    Args:
        features: Input feature DataFrame
        
    Returns:
        The standard deviation of each row of features
    """
    return features.std(axis=1)


def maximize(features: pd.DataFrame) -> pd.Series:
    """
    Calculate the maximum value of each row of features
    
    Args:
        features: Input feature DataFrame
        
    Returns:
        The maximum value of each row of features
    """
    return features.max(axis=1)


def diff(features: Union[pd.DataFrame, pd.Series], time: int) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculate the difference of features
    
    Args:
        features: Input feature DataFrame or Series
        time: Time step for difference
        
    Returns:
        The difference of features
    """
    # If it is a DataFrame, calculate the difference by the first column
    if len(features.shape) > 1:
        feature_name = list(features.columns)
        return features.groupby(feature_name[0]).diff(time)
    # If it is a Series, directly calculate the difference
    return features.diff(time)


def delay(features: Union[pd.DataFrame, pd.Series], time: int) -> Union[pd.DataFrame, pd.Series]:
    """
    Move the feature backward by a specified time step
    
    Args:
        features: Input feature DataFrame or Series
        time: Time step for delay
        
    Returns:
        The delayed feature
    """
    # If it is a DataFrame, calculate the delay by the first column
    if len(features.shape) > 1:
        feature_name = list(features.columns)
        return features.groupby(feature_name[0]).shift(time)
    #  If it is a Series, directly move
    return features.shift(time)


def ts_mean(features: Union[pd.DataFrame, pd.Series], time: int) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculate the rolling mean of features
    
    Args:
        features: Input feature DataFrame or Series
        time: Rolling window size
        
    Returns:
        The rolling mean of features
    """
    # If it is a DataFrame, calculate the rolling mean by the first column
    if len(features.shape) > 1:
        feature_name = list(features.columns)
        return features.groupby(feature_name[0]).rolling(time).mean()
    # If it is a Series, directly calculate the rolling mean
    return features.rolling(time).mean()


def ts_std(features: Union[pd.DataFrame, pd.Series], time: int) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculate the rolling standard deviation of features
    
    Args:
        features: Input feature DataFrame or Series
        time: Rolling window size
        
    Returns:
        The rolling standard deviation of features
    """
    # If it is a DataFrame, calculate the rolling standard deviation by the first column
    if len(features.shape) > 1:
        feature_name = list(features.columns)
        return features.groupby(feature_name[0]).rolling(time).std()
    # If it is a Series, directly calculate the rolling standard deviation
    return features.rolling(time).std()


def ts_cov(features: Union[pd.DataFrame, pd.Series], time: int) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculate the rolling covariance of features
    
    Args:
        features: Input feature DataFrame or Series
        time: Rolling window size
        
    Returns:
        The rolling covariance of features
    """
    # If it is a DataFrame, calculate the rolling covariance by the first column
    if len(features.shape) > 1:
        feature_name = list(features.columns)
        return features.groupby(feature_name[0]).rolling(time).cov()
    # If it is a Series, directly calculate the rolling covariance
    return features.rolling(time).cov()


def ts_corr(features: Union[pd.DataFrame, pd.Series], time: int) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculate the rolling correlation of features
    
    Args:
        features: Input feature DataFrame or Series
        time: Rolling window size
        
    Returns:
        The rolling correlation of features
    """
    # If it is a DataFrame, calculate the rolling correlation by the first column
    if len(features.shape) > 1:
        feature_name = list(features.columns)
        return features.groupby(feature_name[0]).rolling(time).corr()
    # If it is a Series, directly calculate the rolling correlation
    return features.rolling(time).corr()


def ts_max(features: Union[pd.DataFrame, pd.Series], time: int) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculate the rolling maximum of features
    
    Args:
        features: Input feature DataFrame or Series
        time: Rolling window size
        
    Returns:
        The rolling maximum of features
    """
    # If it is a DataFrame, calculate the rolling maximum by the first column
    if len(features.shape) > 1:
        feature_name = list(features.columns)
        return features.groupby(feature_name[0]).rolling(time).max()
    # If it is a Series, directly calculate the rolling maximum
    return features.rolling(time).max()


def ts_min(features: Union[pd.DataFrame, pd.Series], time: int) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculate the rolling minimum of features
    
    Args:
        features: Input feature DataFrame or Series
        time: Rolling window size
        
    Returns:
        The rolling minimum of features
    """
    # If it is a DataFrame, calculate the rolling minimum by the first column
    if len(features.shape) > 1:
        feature_name = list(features.columns)
        return features.groupby(feature_name[0]).rolling(time).min()
    # If it is a Series, directly calculate the rolling minimum
    return features.rolling(time).min()


def ts_rank(features: Union[pd.DataFrame, pd.Series], time: int) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculate the rolling rank of features
    
    Args:
        features: Input feature DataFrame or Series
        time: Rolling window size
        
    Returns:
        The rolling rank of features
    """
    # If it is a DataFrame, calculate the rolling rank by the first column
    if len(features.shape) > 1:
        feature_name = list(features.columns)
        return features.groupby(feature_name[0]).rolling(time).rank()
    # If it is a Series, directly calculate the rolling rank
    return features.rolling(time).rank()


def ewm_mean(features: Union[pd.DataFrame, pd.Series], time: int, alpha: float = 0.5) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculate the exponential weighted moving average of features
    
    Args:
        features: Input feature DataFrame or Series
        time: Exponential weighted span parameter
        alpha: Smoothing coefficient, default is 0.5
        
    Returns:
        The exponential weighted moving average of features
    """
    # If it is a DataFrame, calculate the exponential weighted moving average by the first column
    if len(features.shape) > 1:
        feature_name = list(features.columns)
        return features.groupby(feature_name[0]).ewm(span=time).mean()
    # If it is a Series, directly calculate the exponential weighted moving average
    return features.ewm(span=time).mean()


def ewm_std(features: Union[pd.DataFrame, pd.Series], time: int, alpha: float = 0.5) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculate the exponential weighted moving standard deviation of features
    
    Args:
        features: Input feature DataFrame or Series
        time: Exponential weighted span parameter
        alpha: Smoothing coefficient, default is 0.5
        
    Returns:
        The exponential weighted moving standard deviation of features
    """
    # If it is a DataFrame, calculate the exponential weighted moving standard deviation by the first column
    if len(features.shape) > 1:
        feature_name = list(features.columns)
        return features.groupby(feature_name[0]).ewm(span=time).std()
    # If it is a Series, directly calculate the exponential weighted moving standard deviation
    return features.ewm(span=time).std()


def ewm_cov(features: Union[pd.DataFrame, pd.Series], time: int, alpha: float = 0.5) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculate the exponential weighted moving covariance of features
    
    Args:
        features: Input feature DataFrame or Series
        time: Exponential weighted span parameter
        alpha: Smoothing coefficient, default is 0.5
        
    Returns:
        The exponential weighted moving covariance of features
    """
    # If it is a DataFrame, calculate the exponential weighted moving covariance by the first column
    if len(features.shape) > 1:
        feature_name = list(features.columns)
        return features.groupby(feature_name[0]).ewm(span=time).cov()
    # If it is a Series, directly calculate the exponential weighted moving covariance
    return features.ewm(span=time).cov()


def ewm_corr(features: Union[pd.DataFrame, pd.Series], time: int, alpha: float = 0.5) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculate the exponential weighted moving correlation of features
    
    Args:
        features: Input feature DataFrame or Series
        time: Exponential weighted span parameter
        alpha: Smoothing coefficient, default is 0.5
        
    Returns:
        The exponential weighted moving correlation of features
    """
    # If it is a DataFrame, calculate the exponential weighted moving correlation by the first column
    if len(features.shape) > 1:
        feature_name = list(features.columns)
        return features.groupby(feature_name[0]).ewm(span=time).corr()
    # If it is a Series, directly calculate the exponential weighted moving correlation
    return features.ewm(span=time).corr()