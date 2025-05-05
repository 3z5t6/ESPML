import pandas as pd
import numpy as np
from typing import Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from module.autofe.utils import split_features, OPERATORTYPES
from module.autofe.operators import *
from module.utils.log import get_logger

# Configure logger
logger = get_logger(__name__)

class Transform(BaseEstimator, TransformerMixin):
    """Replace name2feature

    Args:
        BaseEstimator: Base class for all estimators in scikit-learn
        TransformerMixin: Mixin class for all transformers in scikit-learn
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._record = {}
        self.fill_num = 0
        self.calculate_operators = set()
        self.transform_operators = set()
        for k in [('n',), ('n', 'n'), ('n', 't')]:
            for m in OPERATORTYPES[k]: self.calculate_operators.add(m)

        for k in [('c',), ('c', 'c'), ('n', 'c')]:
            for m in OPERATORTYPES[k]: self.transform_operators.add(m)

        FEATURECONFIG = kwargs.get('Feature', {})
        self.task_type = FEATURECONFIG.get('TaskType', None)
        self.metirc = FEATURECONFIG.get('Metric', None)
        self.seed = FEATURECONFIG.get('RandomSeed', 1024)
        self.cat_features = FEATURECONFIG.get('CategoricalFeature', [])
        self.target_name = FEATURECONFIG.get('TargetName', None)
        self.time_index = FEATURECONFIG.get('TimeIndex', None)
        self.group_index = FEATURECONFIG.get('GroupIndex', None)

    def fit(self, df: pd.DataFrame, feature_space: Union[list, list[str]], labelencode: bool = False):
        """Calculate transformers for the same DataFrame
    
        Args:
            df (pd.DataFrame): Input DataFrame
            feature_space (Union[list, list[str]]): List of feature names or feature combinations
            labelencode (bool, optional): Whether to perform label encoding. Defaults to False.
    
        Raises:
            ValueError: If the operator is not supported
    
        Returns:
            Transform: The fitted transformer
        """
        if labelencode:
            for key in df.columns:
                if df[key].dtype.name == 'object' and key != self.time_index or \
                        (key == self.target_name and self.task_type == 'classification' and (
                                df[key].max() >= len(df[key].unique()) or df[key].min() < 0)):
                    le = LabelEncoderWithOOV()
                    le.fit(df[key].astype(str))
                    self._record[key] = le
        for key in feature_space:
            # get the operator and feature name from feature space
            temp = split_features(key)
            op_name = temp[0]
            if temp[-1] not in df.columns and self._is_integer(temp[-1]):
                time_span = int(temp[-1])
            else:
                time_span = None
            fes = temp[1:] if time_span is None else temp[1:-1]
            if self.group_index is not None and op_name in OPERATORTYPES[('n', 't')]: fes = [self.group_index] + fes
            if len(fes) <= 1: fes = fes[0]

            # already calculate or not
            if key in self._record: continue

            # fit recursion
            if any([(self._is_combination_feature(x) and x not in df.columns) for x in temp[1:]]):
                self._recursion(df, key, fit=True)

            # calculate need intermediate operator
            elif op_name in self.transform_operators:
                command = op_name + '(df[fes], intermediate=True)' if time_span is None \
                    else op_name + '(df[fes], time=time_span, intermediate=True)'
                _, intermediate_stat = eval(command)
                self._record[key] = intermediate_stat
            elif op_name in self.calculate_operators:
                continue
            else:
                raise ValueError('Operator not supported')
        return self

    def transform(self, df: pd.DataFrame, feature_space: list = [], labelencode: bool = False) -> pd.DataFrame:
        """Transform input DataFrame with calculated features
    
        Args:
            df (pd.DataFrame): Input DataFrame
            feature_space (list, optional): List of features to transform. Defaults to [].
            labelencode (bool, optional): Whether to perform label encoding. Defaults to False.
    
        Returns:
            pd.DataFrame: Transformed DataFrame with new features
        """
        if labelencode:
            for key in df.columns:
                # Filter keys that exist in df, have object type or are classification targets starting from 0, and value is a transformer
                if key in self._record and df[key].dtype.name == 'object' and key != self.time_index or \
                        (key == self.target_name and self.task_type == 'classification' and (
                                df[key].max() >= len(df[key].unique()) or df[key].min() < 0)) and \
                        isinstance(self._record[key], LabelEncoderWithOOV):
                    df[key] = self._record[key].transform(df[key].astype(str))
        res = df.copy()
        for key in feature_space:
            # Skip features that already exist
            if key in res.columns: continue

            # init feature space information
            temp = split_features(key)
            op_name = temp[0]
            if temp[-1] not in df.columns and self._is_integer(temp[-1]):
                time_span = int(temp[-1])
            else:
                time_span = None
            fes = temp[1:] if time_span is None else temp[1:-1]
            if self.group_index is not None and op_name in OPERATORTYPES[('n', 't')]: fes = [self.group_index] + fes
            if len(fes) <= 1: fes = fes[0]

            # if combination feature need recursion
            if any([(self._is_combination_feature(x) and x not in df.columns) for x in temp[1:]]):
                new_feature = self._recursion(res, key)
            elif op_name in self.transform_operators:
                new_feature = self._transform(op_name, res[fes], key, time_span)
            elif op_name in self.calculate_operators:
                new_feature = self._calculate(op_name, res[fes].astype(float), key, time_span)
            else:
                raise IndexError('The New Feature Need Fit First.')

            # concat new feature
            res = pd.concat([res.reset_index(drop=True), new_feature.reset_index(drop=True)], axis=1)
        return res

    def fit_transform(self, df: pd.DataFrame, feature_space: list = [], labelencode: bool = False) -> pd.DataFrame:
        """Fit to data, then transform it
    
        Args:
            df (pd.DataFrame): Input DataFrame
            feature_space (list, optional): List of features to transform. Defaults to [].
            labelencode (bool, optional): Whether to perform label encoding. Defaults to False.
    
        Returns:
            pd.DataFrame: Transformed DataFrame with new features
        """
        self.fit(df, feature_space, labelencode)
        res = self.transform(df, feature_space, labelencode)
        return res

    def _transform(self, op_name: str, fes: Union[pd.DataFrame, pd.Series], key: str, time_span: int = None) -> pd.Series:
        """Transform features using the operator
    
        Args:
            op_name (str): Operator name
            fes (Union[pd.DataFrame, pd.Series]): Features to transform
            key (str): Key for storing the result
            time_span (int, optional): Time span for time series operators. Defaults to None.
    
        Returns:
            pd.Series: Transformed feature
        """
        if isinstance(fes, pd.Series): fes = fes.to_frame()
        new_feature = fes.merge(self._record[key], on=list(self._record[key].index.names), how='left')[op_name]
        new_feature.rename(key, inplace=True)
        return new_feature

    def _calculate(self, op_name: str, fes: Union[pd.DataFrame, pd.Series], key: str, time_span: int = None) -> pd.Series:
        """Calculate features using the operator
    
        Args:
            op_name (str): Operator name
            fes (Union[pd.DataFrame, pd.Series]): Features to calculate
            key (str): Key for storing the result
            time_span (int, optional): Time span for time series operators. Defaults to None.
    
        Returns:
            pd.Series: Calculated feature
        """
        command = op_name + '(fes)' if time_span is None else op_name + '(fes, time=time_span)'
        new_feature = eval(command)
        new_feature = new_feature.reset_index(drop=True)
        if isinstance(new_feature, pd.DataFrame):
            assert new_feature.shape[1] == 1, f"{op_name} operator do not meet the required format."
            new_feature = new_feature.iloc[:, 0]
        new_feature.rename(key, inplace=True)
        return new_feature

    def _recursion(self, df: pd.DataFrame, key: str, fit: bool = False) -> pd.Series:
        """Recursively process complex feature combinations
    
        Args:
            df (pd.DataFrame): Input DataFrame
            key (str): Feature key
            fit (bool, optional): Whether in fit mode. Defaults to False.
    
        Raises:
            ValueError: If the operator is not defined
    
        Returns:
            pd.Series: Calculated or transformed feature
        """
        temp, op_name, fes, time_span, features = self._analysis_feature_space(df, key)
        for i in range(1, len(temp)):
            if self._is_combination_feature(temp[i]) and temp[i] not in df.columns:
                temp_new_feature = self._recursion(df, temp[i], fit=fit).rename(temp[i])
                features = pd.concat([features.reset_index(drop=True), temp_new_feature.reset_index(drop=True)], axis=1)
        features = features[fes]

        if op_name in self.calculate_operators:
            new_feature = self._calculate(op_name, features.astype(float), key, time_span=time_span)
        elif op_name in self.transform_operators and not fit:
            new_feature = self._transform(op_name, features, key, time_span=time_span)
        elif op_name in self.transform_operators and fit:
            command = op_name + '(features, intermediate=True)' if time_span is None \
                else op_name + '(features, time=time_span, intermediate=True)'
            new_feature, intermediate_stat = eval(command)
            self._record[key] = intermediate_stat
        else:
            raise ValueError(f'Operator {op_name} is not defined.')

        return new_feature

    def _analysis_feature_space(self, df: pd.DataFrame, key: str) -> tuple:
        """Analyze feature space to extract operator, features, and time span
    
        Args:
            df (pd.DataFrame): Input DataFrame
            key (str): Feature key
    
        Returns:
            tuple: Contains temp (split features), op_name, fes (feature names), 
                  time_span (if applicable), and features DataFrame
        """
        temp = split_features(key)
        op_name = temp[0]
        if temp[-1] not in df.columns and self._is_integer(temp[-1]):
            time_span = int(temp[-1])
        else:
            time_span = None
        fes = temp[1:] if time_span is None else temp[1:-1]
        features = df[[col for col in fes if col in df.columns]] if set(fes) & set(list(df.columns)) else pd.DataFrame(
            [])
        if len(fes) <= 1: fes = fes[0]
        return temp, op_name, fes, time_span, features

    def _is_integer(self, x: str) -> bool:
        """Check if a string can be converted to an integer
    
        Args:
            x (str): String to check
    
        Returns:
            bool: True if can be converted to integer, False otherwise
        """
        try:
            int(x)
            return True
        except ValueError:
            return False
    
    def _is_combination_feature(self, feature_name: str) -> bool:
        """Check if a feature name represents a combination feature
    
        Args:
            feature_name (str): Feature name to check
    
        Returns:
            bool: True if it's a combination feature, False otherwise
        """
        count = Counter(feature_name)
        return min(count.get('#', 0) // 3, count.get('$', 0) // 3) > 0


class LabelEncoderWithOOV(LabelEncoder):
    """Label Encoder that handles Out-of-Vocabulary (OOV) values

    This encoder extends the standard LabelEncoder to handle values not seen during training
    by assigning them to a special OOV token.
    """
    def __init__(self, oov_token: str = "OOV", dropna: bool = True):
        """Initialize LabelEncoderWithOOV

        Args:
            oov_token (str, optional): Token to use for out-of-vocabulary values. Defaults to "OOV".
            dropna (bool, optional): Whether to handle NaN values specially. Defaults to True.
        """
        self.oov_token = oov_token
        self.dropna = dropna
        super().__init__()

    def fit(self, y: pd.Series) -> 'LabelEncoderWithOOV':
        """Fit label encoder to data

        Args:
            y (pd.Series): Target values to encode

        Returns:
            LabelEncoderWithOOV: Fitted encoder
        """
        if self.dropna: y = y.fillna("__NAN__")
        super().fit(y)
        return self

    def transform(self, y: pd.Series) -> np.ndarray:
        """Transform labels to encoded values

        Args:
            y (pd.Series): Target values to encode

        Returns:
            np.ndarray: Encoded labels
        """
        if self.dropna: y = y.fillna("__NAN__")
        if self.oov_token is not None:
            self.classes_ = np.append(self.classes_, self.oov_token)
        y_encoded = super().transform([label if label in self.classes_ else self.oov_token for label in y])
        return y_encoded