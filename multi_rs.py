#!/usr/bin/env python
# coding: utf-8

"""
Multi-System Synthesis Condition Prediction System

Embeds and vectorizes synthesis parameters using a factor matrix, and predicts the success/failure
of multi-component synthesis conditions via various regression methods.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide INFO, WARNING, ERROR logs from TensorFlow
import sys
import math
import shutil
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import itertools
from fractions import Fraction
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
import optuna
import matplotlib as mpl
mpl.use('Agg')  # If not using a GUI, specify this backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import comb
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, precision_recall_curve, accuracy_score,
    average_precision_score, f1_score, matthews_corrcoef, cohen_kappa_score,
    mean_squared_error, r2_score, ndcg_score, log_loss, confusion_matrix
)
import umap
from scipy.special import softmax
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass


############################
# Utility Functions
############################

def move_files_to_folder(folder_name, file_list):
    """
    Moves specified files into a target folder (creates it if it does not exist).

    Args:
        folder_name (str): The name of the target folder.
        file_list (list): List of filenames to move.
    """
    os.makedirs(folder_name, exist_ok=True)
    for file_name in file_list:
        if os.path.isfile(file_name):
            if os.path.isfile(os.path.join(folder_name, file_name)):
                print(f"File {file_name} already exists in the destination folder. Skipped.")
            else:
                shutil.move(file_name, folder_name)
        else:
            print(f"File {file_name} does not exist.")


def load_csv_to_dataframe(file_path, column_names=[]):
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.
        column_names (list of str, optional): Column names for the DataFrame.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file cannot be found.
        ValueError: If data loading or processing fails.
    """
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline().strip().split()

        if 'score' in first_line:
            df = pd.read_csv(file_path, sep=" ")
            column_names = df.columns
        else:
            if not column_names:
                raise ValueError("Column names are required.")
            df = pd.read_csv(file_path, names=column_names, sep=" ")

        for column_name in column_names:
            df[column_name] = df[column_name].astype(str)
            if "__" in df[column_name].iloc[0]:
                split_data = df[column_name].str.split("__", expand=True)
                split_data.columns = [f"{column_name}_{i}" for i in range(split_data.shape[1])]
                df = pd.concat([df, split_data], axis=1)

        return df

    except FileNotFoundError as e:
        raise FileNotFoundError(f"CSV file {file_path} not found: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load CSV file: {e}")


def get_random_rows(df, N):
    """
    Returns N random rows from a DataFrame, and also the rows that were not selected.
    If N is a float <= 1, treat it as a fraction; if N is an int >= 1, treat it as a row count.

    Args:
        df (pd.DataFrame): Input DataFrame.
        N (int or float): Number of rows to sample or a fraction.

    Returns:
        tuple of (pd.DataFrame, pd.DataFrame): (selected rows, non-selected rows).
    """
    if isinstance(N, float) and 0 < N <= 1:
        selected_rows = df.sample(frac=N)
    elif N == 1:
        selected_rows = df.sample(n=1)
    elif isinstance(N, int) and N > 1:
        selected_rows = df.sample(n=N)
    else:
        raise ValueError("N must be a positive integer or a float between 0 and 1.")

    not_selected_rows = df.drop(selected_rows.index)
    return selected_rows, not_selected_rows


def get_unique_values(df, column_names, split=False, delimiter="__", keyword=None):
    """
    Returns a dictionary of unique values for specified columns. If split=True, split on the delimiter.

    Args:
        df (pd.DataFrame): Target DataFrame.
        column_names (list): Columns to extract unique values from.
        split (bool): Whether to split on the delimiter before extracting. Defaults to False.
        delimiter (str): Delimiter for splitting. Defaults to "__".
        keyword (str): If provided, only process columns containing this keyword.

    Returns:
        dict: A dictionary of unique values for each column/keyword.
    """
    try:
        unique_dict = {}

        if keyword:
            keyword_columns = [col for col in column_names if keyword in col]
            combined_values = []
            for column_name in keyword_columns:
                if split:
                    combined_values.extend(df[column_name].str.split(delimiter).explode().unique().tolist())
                else:
                    combined_values.extend(df[column_name].unique().tolist())
            unique_dict[keyword] = list(set(combined_values))

        for column_name in column_names:
            if column_name not in unique_dict:
                if split:
                    unique_dict[column_name] = df[column_name].str.split(delimiter).explode().unique().tolist()
                else:
                    unique_dict[column_name] = df[column_name].unique().tolist()

        return unique_dict

    except Exception as e:
        raise ValueError(f"Failed to retrieve unique values: {e}")


def replace_ion_items(d, keyword):
    """
    Replaces items of 'keyword1', 'keyword2', ... in dict `d` with items from 'keyword'.

    Args:
        d (dict): Dictionary containing 'keyword' and 'keywordN' as keys.
        keyword (str): Base keyword to look for.

    Returns:
        dict: Updated dictionary where 'keywordN' items are replaced with 'keyword' items.
    """
    if keyword not in d:
        raise KeyError(f"Key '{keyword}' not found in dictionary.")

    ion_items = d[keyword]

    for key in d.keys():
        if key.startswith(keyword) and key != keyword:
            d[key] = ion_items

    return d


def add_next_key_to_dict(d, keyword, add_item):
    """
    Adds a new key 'keyword{i+1}' to the dictionary if existing keys start with 'keyword'.

    Args:
        d (dict): The original dictionary.
        keyword (str): The keyword to look for in keys.
        add_item: The value to assign to the new key.

    Returns:
        dict: A new dictionary with the additional key.
    """
    new_dict = d.copy()
    key_count = sum(1 for key in new_dict if key.startswith(keyword))
    new_key = f"{keyword}{key_count + 1}"
    new_dict[new_key] = add_item
    return new_dict


def replace_key_lists_with_first_key_list(d, keyword):
    """
    Replaces the values of all dict keys starting with 'keyword' with the value
    from the first key that starts with 'keyword'.

    Args:
        d (dict): The dictionary to modify.
        keyword (str): Keyword to search for in keys.

    Returns:
        dict: Updated dictionary.
    """
    first_key = next((key for key in d if key.startswith(keyword)), None)
    if first_key is None:
        raise ValueError(f"No key found starting with '{keyword}'")

    first_value = d[first_key]
    if isinstance(first_value, pd.DataFrame):
        if first_value.empty:
            raise ValueError(f"The DataFrame of key '{first_key}' is empty.")
    elif isinstance(first_value, list):
        if not first_value:
            raise ValueError(f"The list value of key '{first_key}' is empty.")
    else:
        value_type = type(first_value)
        raise ValueError(f"The value of key '{first_key}' is not a list or DataFrame. It is type {value_type}.")

    key_list = [key for key in d if key.startswith(keyword)]
    for key in key_list:
        d[key] = first_value
    return d


def get_factor_axis_files(directory="."):
    """
    Returns a list of file paths in the specified directory that start with 'factor_axis_'.

    Args:
        directory (str): Directory to search.

    Returns:
        list: List of file paths matching 'factor_axis_*'.
    """
    files = os.listdir(directory)
    factor_axis_files = [os.path.join(directory, f) for f in files if f.startswith("factor_axis_")]
    return factor_axis_files


def load_factors(file_path, delimiter=' ', header=False, column_names=None):
    """
    Loads a factor matrix from a file into a DataFrame.

    Args:
        file_path (str): Path to the factor matrix file.
        delimiter (str): Delimiter in the file (default is space).
        header (bool): Whether the file contains a header row (default is False).
        column_names (list of str, optional): Names for the columns. If not provided, columns are unnamed.

    Returns:
        pd.DataFrame: The loaded factor matrix as a DataFrame, or None if an error occurred.
    """
    try:
        if header:
            df = pd.read_csv(file_path, delimiter=delimiter, index_col=0)
        else:
            df = pd.read_csv(file_path, delimiter=delimiter, header=None, index_col=0)

        if column_names is not None:
            df.columns = column_names

        return df

    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    except pd.errors.ParserError:
        print(f"Failed to parse {file_path}. Check delimiter and format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def extract_keys_containing_strings(d, strings):
    """
    Returns a list of keys in dict `d` that contain all strings in `strings`.

    Args:
        d (dict): The dictionary to search.
        strings (list of str): List of substrings that must appear in the key.

    Returns:
        list: Keys matching all substrings.
    """
    return [key for key in d.keys() if all(s in key for s in strings)]


def factor_axis_to_dict(list_df_factors, list_axis_name):
    """
    Creates a dictionary that maps axis names to factor matrices.

    Args:
        list_df_factors (list): List of factor-matrix DataFrames.
        list_axis_name (list): List of axis names.

    Returns:
        dict: Dictionary of {axis_name: factor_matrix} pairs.
    """
    factor_dict = {}
    num_factors = len(list_df_factors)
    num_axes = len(list_axis_name)
    if num_factors == num_axes:
        for i in range(num_axes):
            factor_dict[list_axis_name[i]] = list_df_factors[i]
    else:
        print("Number of factor matrices and axis names do not match.")
        print(f"Number of factor matrices: {num_factors}")
        print(f"Number of axis names: {num_axes}")
    return factor_dict


def remove_interchangeable(df, target_cols, ratio_col, exclude_cols=[]):
    """
    Removes duplicate rows from the dataframe when combinations of `target_cols` are interchangeable.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_cols (list): Columns that are interchangeable.
        ratio_col (str): Column name containing ratios in 'a:b' or 'a:b:c' form.
        exclude_cols (list): Columns to exclude from deduplication.

    Returns:
        pd.DataFrame: DataFrame after removing duplicates based on sorted items.
    """
    _df = df.copy()
    sorted_item_col = "sorted_item"
    original_cols = _df.columns
    nontarget_cols = [col for col in original_cols if col not in target_cols + [ratio_col] + exclude_cols] + [sorted_item_col]
    _df[ratio_col] = _df[ratio_col].astype(str)

    sorted_items = []
    for row in tqdm(_df.itertuples(index=False), total=len(_df), desc="Sorting"):
        ratio_values = getattr(row, ratio_col).split(':')
        try:
            ratio_values = [float(value) for value in ratio_values]
        except ValueError:
            raise ValueError(f"Invalid ratio format: {getattr(row, ratio_col)}")

        sorted_item = "_".join(
            sorted([
                f"{getattr(row, tc)}@{ratio_values[i]}" for i, tc in enumerate(target_cols)
            ])
        )
        sorted_items.append(sorted_item)

    _df[sorted_item_col] = sorted_items
    _df = _df.drop_duplicates(subset=nontarget_cols).drop(sorted_item_col, axis=1).reset_index(drop=True)
    return _df


def duplicated_items_score_change(df, target_cols, same_score):
    """
    If a row has duplicate items among `target_cols`, change the 'score' to `same_score`.

    Args:
        df (pd.DataFrame): Target DataFrame.
        target_cols (list): Columns to check duplicates within a row.
        same_score (float or int): Score to assign if duplicates exist.

    Returns:
        pd.DataFrame: Updated DataFrame with scores changed for duplicate rows.
    """
    def check_duplicates(row):
        vals = row[target_cols].tolist()
        return len(vals) != len(set(vals))

    duplicates_mask = df.apply(check_duplicates, axis=1)
    df.loc[duplicates_mask, 'score'] = same_score
    return df


def create_duplicated_combination(d, mode, score_col_name="score", temporary_score=1):
    """
    Creates a DataFrame of all combinations from dictionary `d` but only keeps rows
    where there is at least one duplicated element in that row.

    Args:
        d (dict): {column_name: list_of_values} pairs.
        mode (int or str): If an integer, sample that many rows; otherwise use all.
        score_col_name (str): Name of the score column. Defaults to 'score'.
        temporary_score (int): Score to assign. Default is 1.

    Returns:
        pd.DataFrame: DataFrame of duplicated-element rows.
    """
    total_combinations = 1
    for v in d.values():
        total_combinations *= len(v)

    combinations = list(tqdm(itertools.product(*d.values()), desc="Generating all combinations", total=total_combinations))
    df = pd.DataFrame(combinations, columns=d.keys())

    def check_duplicates(row):
        seen = set()
        for item in row:
            if isinstance(item, list):
                item = frozenset(item)
            if item in seen:
                return True
            seen.add(item)
        return False

    mask = np.array([check_duplicates(row) for row in tqdm(df.values, desc="Filtering duplicates")])
    df = df[mask]
    print(f"df_mask: {len(df)}")

    if not isinstance(mode, str):
        print(f"Randomly sampling {mode} combinations...")
        df, _ = get_random_rows(df, mode)

    if score_col_name in df.columns:
        df[score_col_name] = temporary_score
    else:
        df.insert(0, score_col_name, temporary_score)

    print("Processing complete.")
    return df


def create_hypothetical_combination(d, mode, score_col_name="score", temporary_score=1):
    """
    Creates a DataFrame of all combinations from dictionary `d`, but only keeps rows
    where all elements in a row are distinct.

    Args:
        d (dict): {column_name: list_of_values} pairs.
        mode (int or str): If an integer, sample that many rows; otherwise use all.
        score_col_name (str): Name of the score column. Defaults to 'score'.
        temporary_score (int): Score to assign. Default is 1.

    Returns:
        pd.DataFrame: DataFrame containing rows with all distinct elements.
    """
    total_combinations = 1
    for v in d.values():
        total_combinations *= len(v)

    print("Generating all combinations...")
    combinations = list(tqdm(itertools.product(*d.values()), desc="Combining", total=total_combinations))
    df = pd.DataFrame(combinations, columns=d.keys())

    def check_unique(row):
        seen = set()
        for item in row:
            if isinstance(item, list):
                item = frozenset(item)
            if item in seen:
                return False
            seen.add(item)
        return True

    print("Filtering unique rows...")
    mask = np.array([check_unique(row) for row in tqdm(df.values, desc="Checking uniqueness")])
    df = df[mask]

    if not isinstance(mode, str):
        print(f"Randomly sampling {mode} combinations...")
        df, _ = get_random_rows(df, mode)

    if score_col_name in df.columns:
        df[score_col_name] = temporary_score
    else:
        df.insert(0, score_col_name, temporary_score)

    print("Processing complete.")
    return df


def generate_weights(value, dimensions, round_num=5, start_value=0, end_value=1, min_non_zero_elements=2):
    """
    Generates combinations of weights where each dimension sums to 1, and the number of non-zero elements
    is at least `min_non_zero_elements`.

    Args:
        value (float): Step size of weights.
        dimensions (int): Number of dimensions (length of weight vectors).
        round_num (int): Rounding precision for weights. Defaults to 5.
        start_value (float): Start of the weight range. Default 0.
        end_value (float): End of the weight range. Default 1.
        min_non_zero_elements (int): Minimum number of non-zero elements in each weight vector. Default 2.

    Returns:
        np.ndarray: Array of valid weight combinations.
    """
    range_values = np.round(np.arange(start_value, end_value, value), round_num)
    combinations = np.array(np.meshgrid(*[range_values] * dimensions)).T.reshape(-1, dimensions)
    valid_combinations = combinations[np.isclose(combinations.sum(axis=1), 1.0)]
    valid_combinations = valid_combinations[valid_combinations.astype(bool).sum(axis=1) >= min_non_zero_elements]
    return valid_combinations


def filter_non_zero_ratios(df, ratio_col='ratio'):
    """
    Filters rows so that ratio 'a:b:c' do not contain zero elements.

    Args:
        df (pd.DataFrame): Input DataFrame.
        ratio_col (str): Column containing 'a:b:c' format ratio.

    Returns:
        pd.DataFrame: Filtered DataFrame with non-zero a,b,c.
    """
    ratio_values = df[ratio_col].values
    for idx, value in enumerate(ratio_values):
        if not isinstance(value, str):
            print(f"Row {idx} has non-string ratio: {value} (type: {type(value)})")

    try:
        ratio_values = np.array([str(ratio) for ratio in ratio_values])
    except Exception as e:
        raise DataProcessingError(f"Error converting ratio to string: {e}")

    try:
        split_ratios = np.core.defchararray.split(ratio_values, ':')
    except TypeError as e:
        print(f"Error splitting string ratio at row {idx}: {value}")
        raise DataProcessingError(f"Split failure: {e}")

    valid_mask = np.array([len(ratios) == 3 for ratios in split_ratios])
    valid_ratios = [ratios for i, ratios in enumerate(split_ratios) if valid_mask[i]]

    if len(valid_ratios) == 0:
        return df.iloc[0:0]

    try:
        a, b, c = zip(*[(float(x[0]), float(x[1]), float(x[2])) for x in valid_ratios])
    except Exception as e:
        print(f"Error converting ratio to float at row {idx}: {value}")
        raise DataProcessingError(f"Float conversion error: {e}")

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    non_zero_mask = (a != 0) & (b != 0) & (c != 0)
    final_mask = valid_mask.copy()
    final_mask[valid_mask] = non_zero_mask
    filtered_df = df[final_mask]
    return filtered_df


def generate_col_names(dim_df_vec, moments):
    """
    Generates column names based on vector dimension and the list of moments.

    Args:
        dim_df_vec (int): Vector dimension.
        moments (list): Moments to compute (e.g., ['mean','std','cov']).

    Returns:
        list: Generated column names.
    """
    col_names = []
    for moment in moments:
        if moment == "cov":
            col_names += [f"{moment}{i}" for i in range(comb(dim_df_vec, 2, exact=True))]
        else:
            col_names += [f"{moment}{i}" for i in range(dim_df_vec)]
    return col_names


def embedding_file_to_df(filename, use_col, score_col_name="score"):
    """
    Loads an embedding file and extracts only specified columns (plus a score column).

    Args:
        filename (str): File to load.
        use_col (list): Columns of interest.
        score_col_name (str): Score column name.

    Returns:
        pd.DataFrame: DataFrame containing only specified columns.
    """
    df = pd.read_csv(filename)
    selected_columns = [col for col in df.columns if any(s in col for s in [score_col_name] + use_col)]
    return df[selected_columns]


def get_vector_cached(factor_dict, factor, value, cache_dict):
    """
    Retrieves a vector from factor_dict, using a cache to speed up repeated lookups.

    Args:
        factor_dict (dict): Dictionary mapping factor names to DataFrames.
        factor (str): Factor (key in factor_dict).
        value (str): Index in the factor DataFrame.
        cache_dict (dict): Dictionary for caching results.

    Returns:
        np.array: The vector for the specified factor and value.
    """
    cache_key = (factor, value)
    if cache_key not in cache_dict:
        cache_dict[cache_key] = factor_dict[factor].loc[value].values
    return cache_dict[cache_key]


def calc_weighted_vectors_optimized(vectors, weights):
    """
    Computes weighted mean, std, kurtosis, skewness, and pairwise covariances for a set of vectors.

    Args:
        vectors (np.array): Array of shape (n_items, dim).
        weights (np.array): Weights for each item.

    Returns:
        dict: Weighted stats for the input vectors.
    """
    weights_sum = np.sum(weights)
    weighted_mean_vector = np.dot(weights, vectors) / weights_sum
    diff = vectors - weighted_mean_vector

    weighted_var_vector = np.dot(weights, diff ** 2) / weights_sum
    weighted_std_vector = np.sqrt(weighted_var_vector)

    with np.errstate(divide='ignore', invalid='ignore'):
        weighted_kurtosis_vector = np.where(
            weighted_std_vector != 0,
            np.dot(weights, diff ** 4) / (weights_sum * weighted_std_vector ** 4) - 3,
            np.nan
        )
        weighted_skewness_vector = np.where(
            weighted_std_vector != 0,
            np.dot(weights, diff ** 3) / (weights_sum * weighted_std_vector ** 3),
            np.nan
        )

    weighted_cov_matrix = np.einsum('ij,ik,i->jk', diff, diff, weights) / weights_sum
    nondiag_cov = weighted_cov_matrix[np.triu_indices_from(weighted_cov_matrix, k=1)]

    return {
        "mean": weighted_mean_vector,
        "std": weighted_std_vector,
        "kurt": weighted_kurtosis_vector,
        "skew": weighted_skewness_vector,
        "cov": nondiag_cov
    }


def chain_vectors(vector_list):
    """
    Concatenates a list of vectors into one flat list.

    Args:
        vector_list (list of np.array): List of 1D arrays.

    Returns:
        list: Flattened list of all elements.
    """
    return list(itertools.chain.from_iterable(vector_list))


def calculate_moments(df_stat, stat_columns, correspond_factors, factor_dict, use_moment, ratio_column="ratio"):
    """
    Calculates weighted statistical moments (mean, std, cov, kurt, skew) for each row
    in df_stat, using factor_dict to retrieve vectors.

    Args:
        df_stat (pd.DataFrame): DataFrame containing the columns to compute stats for.
        stat_columns (list): Columns to be used (e.g., ['ion1','ion2']).
        correspond_factors (list): Factor keys in factor_dict for each stat_column.
        factor_dict (dict): {factor_name: DataFrame of factor vectors}.
        use_moment (list): Which moments to compute (e.g., ['mean','std','cov']).
        ratio_column (str): Column name containing ratio like 'a:b' or 'a:b:c'.

    Returns:
        list of lists: Each element is the concatenated moments for one row.
    """
    cache_dict = {}
    all_vectors = {
        col: np.array([get_vector_cached(factor_dict, correspond_factors[j], val, cache_dict) for val in df_stat[col]])
        for j, col in enumerate(stat_columns)
    }

    moments = []
    for _, row in tqdm(df_stat.iterrows(), position=0, leave=True, desc="Moments calculating"):
        vectors = np.array([all_vectors[col][row.name] for col in stat_columns])
        weights = np.array([float(i) for i in str(row[ratio_column]).split(":")])
        weighted_vectors = calc_weighted_vectors_optimized(vectors, weights)
        # gather requested stats
        moments.append(chain_vectors([weighted_vectors[vec] for vec in use_moment]))

    return moments


def add_factor_columns(df_score, add_columns, factor_dict):
    """
    Retrieves factor vectors for each row in df_score[add_columns] and appends them as new columns.

    Args:
        df_score (pd.DataFrame): DataFrame containing the item to be factorized.
        add_columns (list): Columns that correspond to factor_dict keys.
        factor_dict (dict): {factor_name: DataFrame} mapping.

    Returns:
        pd.DataFrame: DataFrame of appended factor columns.
    """
    df_moment2 = pd.DataFrame()
    for add in add_columns:
        values = [
            factor_dict[add].loc[row].values.tolist()
            for row in tqdm(df_score[add], position=0, leave=True, desc="Adding factors")
        ]
        col_name = [f"{add}_{i}" for i in factor_dict[add].columns]
        df_moment2 = pd.concat([df_moment2, pd.DataFrame(values, columns=col_name)], axis=1)
    return df_moment2


def create_embedding(
    df_score, use_moment, stat_columns, correspond_factors,
    add_columns, factor_dict, ratio_column="ratio", score_col_name="score",
    decimal_places=5
):
    """
    Creates an embedding representation using factor matrices for specified columns.

    Args:
        df_score (pd.DataFrame): The DataFrame with score and columns to embed.
        use_moment (list): Moments to compute (e.g., ['mean','std','cov']).
        stat_columns (list): Columns that will have their vectors combined (e.g., ['ion1','ion2']).
        correspond_factors (list): Factor keys matching stat_columns.
        add_columns (list): Additional columns to embed directly.
        factor_dict (dict): {factor_name: DataFrame of factor vectors}.
        ratio_column (str): Name of the ratio column. Default 'ratio'.
        score_col_name (str): Name of the score column. Default 'score'.
        decimal_places (int): Decimal rounding. Default 5.

    Returns:
        pd.DataFrame: Embedded representation of df_score.
    """
    if len(stat_columns) > 1:
        df_stat = df_score[stat_columns + [ratio_column]]
        moments = calculate_moments(df_stat, stat_columns, correspond_factors, factor_dict, use_moment)
        col_name1 = generate_col_names(len(factor_dict[correspond_factors[0]].columns), use_moment)
        df_moment1 = pd.DataFrame(moments, columns=col_name1)
    else:
        df_moment1 = pd.DataFrame()

    if len(add_columns) > 0:
        df_moment2 = add_factor_columns(df_score, add_columns, factor_dict)
    else:
        df_moment2 = pd.DataFrame()

    df_s = df_score[[score_col_name]]

    if not df_moment1.empty and not df_moment2.empty:
        df_moment = pd.concat([df_s, df_moment1, df_moment2], axis=1)
    elif not df_moment1.empty:
        df_moment = pd.concat([df_s, df_moment1], axis=1)
    elif not df_moment2.empty:
        df_moment = pd.concat([df_s, df_moment2], axis=1)
    else:
        df_moment = df_s

    return df_moment.round(decimal_places)


def dimension_reduction(list_df, method, dim, score_col_name="score", decimal_places=5):
    """
    Performs dimensionality reduction (PCA, SVD, t-SNE, or UMAP) on a list of DataFrames (concatenated).

    Args:
        list_df (list of pd.DataFrame): List of DataFrames to reduce.
        method (str): Reduction method ('PCA','SVD','t-SNE','UMAP').
        dim (int): Target dimension.
        score_col_name (str): Score column name to exclude from transformation.
        decimal_places (int): Rounding precision for results.

    Returns:
        list of pd.DataFrame: The dimension-reduced DataFrames, split back to original sizes.
    """
    df_combined = pd.concat(list_df)
    df_no_score = df_combined.drop(score_col_name, axis=1)

    if method == "PCA":
        pca_model = PCA(n_components=dim)
        vectors = pca_model.fit_transform(df_no_score)
        col_name = [f"pca_{i}" for i in range(dim)]
        print("Contribution rates", pca_model.explained_variance_ratio_)
        print("Sum of contribution rates", sum(pca_model.explained_variance_ratio_))
    elif method == "SVD":
        svd_model = TruncatedSVD(n_components=dim)
        vectors = svd_model.fit_transform(df_no_score)
        col_name = [f"svd_{i}" for i in range(dim)]
    elif method == "UMAP":
        umap_model = umap.UMAP(n_components=dim, n_neighbors=2, metric="correlation")
        vectors = umap_model.fit_transform(df_no_score)
        col_name = [f"umap_{i}" for i in range(dim)]
    elif method == "t-SNE":
        tsne_model = TSNE(
            n_components=dim,
            perplexity=10,
            method='exact' if dim > 4 else 'barnes_hut',
            init='pca'
        )
        vectors = tsne_model.fit_transform(df_no_score)
        col_name = [f"tsne_{i}" for i in range(dim)]
    else:
        raise ValueError(f"Unsupported dimensionality reduction method: {method}")

    df_reduced = pd.DataFrame(vectors, columns=col_name).round(decimal_places)
    df_reduced.insert(0, score_col_name, df_combined[score_col_name].reset_index(drop=True))

    list_df_lengths = [len(df) for df in list_df]
    list_reduced_dfs = []
    start = 0
    for length in list_df_lengths:
        list_reduced_dfs.append(df_reduced.iloc[start:start+length])
        start += length

    return list_reduced_dfs


def map_to_binary_labels(y_true):
    """
    Maps an array of labels to binary (0/1). Requires exactly 2 unique values in y_true.

    Args:
        y_true (array-like): Array of labels.

    Returns:
        np.array: Binary labels.

    Raises:
        ValueError: If y_true has more than 2 unique values.
    """
    unique_values = np.unique(y_true)
    if len(unique_values) != 2:
        raise ValueError("y_true must contain exactly two unique values for binary classification.")
    return np.where(y_true == unique_values[0], 0, 1)


def standardize_columns(df, columns):
    """
    Standardizes specified columns in a DataFrame, returning the updated DataFrame and the means/stds.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): Columns to standardize.

    Returns:
        tuple: (standardized_df, dict_of_means, dict_of_stds).
    """
    df_standardized = df.copy()
    means = df[columns].mean()
    stds = df[columns].std()
    df_standardized[columns] = (df[columns] - means) / stds
    return df_standardized, means.to_dict(), stds.to_dict()


def find_optimal_threshold(y_true, probabilities, metric='f1', top_n=None):
    """
    Finds the optimal decision threshold for predicted probabilities that maximizes the given metric.

    Args:
        y_true (array-like): True labels (must be binary).
        probabilities (array-like): Predicted probabilities.
        metric (str): Metric to optimize ('f1','mcc','sensitivity','specificity','kappa').
        top_n (int, optional): If set, only use top_n predictions in evaluation.

    Returns:
        tuple: (best_threshold, best_metric_value).
    """
    thresholds = np.linspace(np.min(probabilities), np.max(probabilities), 100)
    best_threshold = thresholds[0]
    best_metric_value = 0
    y_true_mapped = map_to_binary_labels(y_true)

    for threshold in thresholds:
        if top_n:
            indices = np.argsort(probabilities)[-top_n:]
            y_true_top_n = y_true_mapped[indices]
            probabilities_top_n = probabilities[indices]
            predictions = (probabilities_top_n > threshold).astype(int)
        else:
            predictions = (probabilities > threshold).astype(int)
            y_true_top_n = y_true_mapped

        if metric == 'f1':
            metric_value = f1_score(y_true_top_n, predictions)
        elif metric == 'mcc':
            metric_value = matthews_corrcoef(y_true_top_n, predictions)
            metric_value = np.clip(metric_value, -1, 1)
        elif metric in ['sensitivity', 'specificity']:
            tn, fp, fn, tp = confusion_matrix(y_true_top_n, predictions).ravel()
            if metric == 'sensitivity':
                metric_value = tp / (tp + fn) if (tp + fn) != 0 else 0
            else:
                metric_value = tn / (tn + fp) if (tn + fp) != 0 else 0
        elif metric == 'kappa':
            metric_value = cohen_kappa_score(y_true_top_n, predictions)
        else:
            raise ValueError(f"Unsupported metric '{metric}'.")

        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_threshold = threshold

    return best_threshold, best_metric_value


def compute_regression_metrics(
    df_result,
    df_val,
    match_columns,
    original_score_col="original_score",
    predicted_score_col="predicted_score",
    positive_label=2
):
    """
    Computes regression metrics (treated as binary classification under the hood) like ROC-AUC, log loss, etc.

    Args:
        df_result (pd.DataFrame): Prediction results with columns [original_score, predicted_score].
        df_val (pd.DataFrame): Validation data.
        match_columns (list): Columns to match/merge on. (Not used here in code, but left for template.)
        original_score_col (str): Column containing true scores. Default 'original_score'.
        predicted_score_col (str): Column containing predicted scores. Default 'predicted_score'.
        positive_label (int): Label considered "positive" in the data. Default 2.

    Returns:
        tuple: (roc_auc, logloss, rmse, avg_precision, f1, mcc, kappa, true_values, predicted_probabilities).
    """
    true_values = df_result[original_score_col]
    predicted_scores = df_result[predicted_score_col].values
    true_values_binary = (true_values == positive_label).astype(int)

    if len(predicted_scores.shape) == 1:
        predicted_probabilities = softmax(np.vstack([predicted_scores, 1 - predicted_scores]).T, axis=1)[:, 0]
    else:
        predicted_probabilities = softmax(predicted_scores, axis=1)

    rmse = math.sqrt(mean_squared_error(true_values_binary, predicted_probabilities))
    roc_auc = roc_auc_score(true_values_binary, predicted_probabilities)
    logloss = log_loss(true_values_binary, predicted_probabilities)
    avg_precision = average_precision_score(true_values_binary, predicted_probabilities)

    _, f1 = find_optimal_threshold(true_values_binary, predicted_probabilities, metric="f1")
    _, mcc = find_optimal_threshold(true_values_binary, predicted_probabilities, metric="mcc")
    _, kappa = find_optimal_threshold(true_values_binary, predicted_probabilities, metric="kappa")

    return roc_auc, logloss, rmse, avg_precision, f1, mcc, kappa, true_values_binary, predicted_probabilities


def compute_classification_metrics(
    df_result,
    original_class_col="original_score",
    predicted_class_col="predicted_class",
    predicted_proba_col="predicted_proba",
    positive_label=2
):
    """
    Computes classification metrics (ROC-AUC, log loss, etc.) given predicted classes/probabilities.

    Args:
        df_result (pd.DataFrame): DataFrame with columns [original_score, predicted_class, predicted_proba].
        original_class_col (str): Name of column with ground truth classes. Default 'original_score'.
        predicted_class_col (str): Name of column with predicted classes. Default 'predicted_class'.
        predicted_proba_col (str): Name of column with predicted probabilities. Default 'predicted_proba'.
        positive_label (int): Label considered positive. Default 2.

    Returns:
        tuple: (roc_auc, logloss, rmse, avg_precision, f1, mcc, kappa, true_values_binary, predicted_probabilities).
    """
    true_values = df_result[original_class_col]
    predicted_probabilities = df_result[predicted_proba_col].values
    true_values_binary = (true_values == positive_label).astype(int)

    roc_auc = roc_auc_score(true_values_binary, predicted_probabilities)
    logloss = log_loss(true_values_binary, predicted_probabilities)
    rmse = math.sqrt(mean_squared_error(true_values_binary, predicted_probabilities))
    avg_precision = average_precision_score(true_values_binary, predicted_probabilities)

    _, f1 = find_optimal_threshold(true_values_binary, predicted_probabilities, metric="f1")
    _, mcc = find_optimal_threshold(true_values_binary, predicted_probabilities, metric="mcc")
    _, kappa = find_optimal_threshold(true_values_binary, predicted_probabilities, metric="kappa")

    return roc_auc, logloss, rmse, avg_precision, f1, mcc, kappa, true_values_binary, predicted_probabilities


def perform_random_forest_classification(df_train, df_test, params, score_col_idx=0):
    """
    Trains a RandomForestClassifier on df_train and predicts on df_test. Appends predictions.

    Args:
        df_train (pd.DataFrame): Training data.
        df_test (pd.DataFrame): Test data.
        params (dict): Hyperparameters for RandomForestClassifier.
        score_col_idx (int): Column index of the score label in df_train.

    Returns:
        pd.DataFrame: df_test with added 'predicted_class' and 'predicted_proba' columns.
    """
    score_col_name = df_train.columns[score_col_idx]
    X_train = df_train.drop(columns=[score_col_name])
    y_train = df_train[score_col_name]
    X_test = df_test.drop(columns=[score_col_name])

    rf = RandomForestClassifier(**params, n_jobs=-1)
    rf.fit(X_train, y_train)

    df_test['predicted_class'] = rf.predict(X_test)
    df_test['predicted_proba'] = rf.predict_proba(X_test)[:, 1]
    return df_test


def perform_lightgbm_classification(df_train, df_test, params, score_col_idx=0):
    """
    Trains a LightGBM classifier on df_train and predicts on df_test.

    Args:
        df_train (pd.DataFrame): Training data.
        df_test (pd.DataFrame): Test data.
        params (dict): Hyperparameters for LGBMClassifier.
        score_col_idx (int): Column index of the score label in df_train.

    Returns:
        pd.DataFrame: df_test with added 'predicted_class' and 'predicted_proba' columns.
    """
    score_col_name = df_train.columns[score_col_idx]
    X_train = df_train.drop(columns=[score_col_name])
    y_train = df_train[score_col_name]
    X_test = df_test.drop(columns=[score_col_name])

    lgb_model = lgb.LGBMClassifier(**params)
    lgb_model.fit(X_train, y_train)

    df_test['predicted_class'] = lgb_model.predict(X_test)
    df_test['predicted_proba'] = lgb_model.predict_proba(X_test)[:, 1]
    return df_test


def perform_autoencoder_classification(df_train, df_test, params, score_col_idx=0, encoding_dim=64):
    """
    Uses an Autoencoder for anomaly detection, based on reconstruction error.

    Args:
        df_train (pd.DataFrame): Training data.
        df_test (pd.DataFrame): Test data.
        params (dict): Hyperparameters (latent_dim, learning_rate, batch_size, epochs).
        score_col_idx (int): Column index of the score label in df_train.
        encoding_dim (int): Dimension of the encoding. Default 64.

    Returns:
        pd.DataFrame: df_test with 'anomaly_score' and 'predicted_proba'.
    """
    score_col_name = df_train.columns[score_col_idx]
    X_train = df_train.drop(columns=[score_col_name])
    X_test = df_test.drop(columns=[score_col_name])

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    input_dim = X_train_scaled.shape[1]
    encoding_dim = params.get('latent_dim', 64)
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    epochs = params.get('epochs', 50)
    batch_size = params.get('batch_size', 32)
    autoencoder.fit(
        X_train_scaled, X_train_scaled,
        epochs=epochs, batch_size=batch_size, shuffle=True,
        validation_data=(X_test_scaled, X_test_scaled),
        callbacks=[early_stopping]
    )

    X_test_pred = autoencoder.predict(X_test_scaled)
    reconstruction_error = np.mean(np.square(X_test_scaled - X_test_pred), axis=1)
    df_test['anomaly_score'] = reconstruction_error
    df_test['predicted_proba'] = df_test['anomaly_score']
    return df_test


def perform_gaussian_process_classification(df_train, df_test, params, score_col_idx=0):
    """
    Trains a GaussianProcessClassifier on df_train and predicts on df_test.

    Args:
        df_train (pd.DataFrame): Training data.
        df_test (pd.DataFrame): Test data.
        params (dict): Hyperparameters for GaussianProcessClassifier.
        score_col_idx (int): Column index of the score label in df_train.

    Returns:
        pd.DataFrame: df_test with 'predicted_class' and 'predicted_proba' columns.
    """
    score_col_name = df_train.columns[score_col_idx]
    X_train = df_train.drop(columns=[score_col_name])
    y_train = df_train[score_col_name]
    X_test = df_test.drop(columns=[score_col_name])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    kernel = params.get('kernel', 1.0 * RBF(1.0))
    n_restarts_optimizer = params.get('n_restarts_optimizer', 0)

    gpc = GaussianProcessClassifier(
        kernel=kernel,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=42
    )
    gpc.fit(X_train_scaled, y_train)

    df_test['predicted_class'] = gpc.predict(X_test_scaled)
    df_test['predicted_proba'] = gpc.predict_proba(X_test_scaled)[:, 1]
    return df_test


def cross_validate_model(folds, params, columns, column_score, pred_model="rf"):
    """
    Performs cross-validation using one of four models:
    RandomForestClassifier, LightGBM, Autoencoder, or GaussianProcessClassifier.

    Args:
        folds: Cross-validation splits.
        params: Model hyperparameters.
        columns: Columns to use for evaluation.
        column_score: Score column name.
        pred_model: Which model to use ('rf','lgbm','ae','gpc').

    Returns:
        dict: Dictionary of mean metric values: roc_auc, logloss, rmse, etc.
    """
    eval_scores = {
        "roc_auc": [],
        "logloss": [],
        "rmse": [],
        "avg_precision": [],
        "f1": [],
        "mcc": [],
        "kappa": []
    }

    for fold_id, (df_train, df_test) in enumerate(folds):
        if pred_model == "rf":
            df_result = perform_random_forest_classification(df_train, df_test, params)
        elif pred_model == "lgbm":
            df_result = perform_lightgbm_classification(df_train, df_test, params)
        elif pred_model == "ae":
            df_result = perform_autoencoder_classification(df_train, df_test, params)
        elif pred_model == "gpc":
            df_result = perform_gaussian_process_classification(df_train, df_test, params)
        else:
            raise ValueError(f"Invalid pred_model: {pred_model}. Must be 'rf','lgbm','ae','gpc'.")

        roc_auc, logloss, rmse, avg_precision, f1, mcc, kappa, _, _ = compute_classification_metrics(
            df_result,
            original_class_col=column_score,
            predicted_class_col="predicted_class",
            predicted_proba_col="predicted_proba",
            positive_label=2
        )

        eval_scores["roc_auc"].append(roc_auc)
        eval_scores["logloss"].append(logloss)
        eval_scores["rmse"].append(rmse)
        eval_scores["avg_precision"].append(avg_precision)
        eval_scores["f1"].append(f1)
        eval_scores["mcc"].append(mcc)
        eval_scores["kappa"].append(kappa)

    return {metric: np.mean(scores) for metric, scores in eval_scores.items()}


def stratified_k_fold(df, n_folds, score_col_name='score', shuffle=False, random_state=None):
    """
    Performs stratified K-fold split on a DataFrame.

    Args:
        df (pd.DataFrame): Input data.
        n_folds (int): Number of folds.
        score_col_name (str): Name of the score column. Default 'score'.
        shuffle (bool): Whether to shuffle. Default False.
        random_state (int): Random seed.

    Returns:
        generator: Yields (train_df, val_df) for each fold.
    """
    df = df.reset_index(drop=True)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    for train_idx, val_idx in folds.split(df, df[score_col_name]):
        yield df.loc[train_idx], df.loc[val_idx]


def k_fold(df, n_folds, score_col_name='score', shuffle=False, random_state=None):
    """
    Performs standard K-fold split on a DataFrame.

    Args:
        df (pd.DataFrame): Input data.
        n_folds (int): Number of folds.
        score_col_name (str): Name of the score column. Default 'score'.
        shuffle (bool): Whether to shuffle. Default False.
        random_state (int): Random seed.

    Returns:
        generator: Yields (train_df, val_df) for each fold.
    """
    df = df.reset_index(drop=True)
    folds = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    for train_idx, val_idx in folds.split(df):
        yield df.loc[train_idx], df.loc[val_idx]


def is_discrete(series):
    """
    Checks if a pandas Series is discrete (integer dtype or <=10 unique values).

    Args:
        series (pd.Series): The Series to check.

    Returns:
        bool: True if discrete, else False.
    """
    return pd.api.types.is_integer_dtype(series) or series.nunique() <= 10


def objective(df, config, trial):
    """
    Optuna objective function that trains one of four models and returns an evaluation metric.

    Args:
        df (pd.DataFrame): Input data.
        config (dict): Config with hyperparameter search ranges.
        trial (optuna.trial.Trial): Optuna Trial object.

    Returns:
        float: The mean evaluation score for the selected metric.
    """
    columns_wo_score = [col for col in df.columns if 'score' not in col]
    column_score = [col for col in df.columns if 'score' in col][0]
    pred_model = config["pred_model"]

    if pred_model == "rf":
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 5, 20)
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': 42
        }
    elif pred_model == "lgbm":
        num_leaves = trial.suggest_int('num_leaves', 20, 150)
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1, log=True)
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        min_child_samples = trial.suggest_int('min_child_samples', 10, 100)
        params = {
            'boosting_type': 'gbdt',
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'min_child_samples': min_child_samples,
            'random_state': 42
        }
    elif pred_model == "ae":
        latent_dim = trial.suggest_int('latent_dim', 3, 60)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        params = {
            'latent_dim': latent_dim,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': 100,
            'random_state': 42
        }
    elif pred_model == "gpc":
        kernel_type = trial.suggest_categorical('kernel', ['RBF_1', 'RBF_0.5'])
        kernel = 1.0 * RBF(1.0) if kernel_type == 'RBF_1' else 1.0 * RBF(0.5)
        params = {
            'kernel': kernel,
            'n_restarts_optimizer': trial.suggest_int('n_restarts_optimizer', 0, 10),
            'random_state': 42
        }
    else:
        raise ValueError(f"Unsupported model {pred_model}")

    n_fold = config["n_fold"]
    if is_discrete(df[column_score]):
        folds = stratified_k_fold(df, n_fold, score_col_name=column_score)
    else:
        folds = k_fold(df, n_fold, score_col_name=column_score)

    eval_scores = cross_validate_model(folds, params, columns_wo_score, column_score, pred_model=pred_model)
    eval_metrics = config["eval_metrics"]
    mean_eval_score = eval_scores[eval_metrics]

    output_data = {
        "trial_number": trial.number,
        "eval_metric": eval_metrics,
        "mean_eval_score": mean_eval_score,
        **params,
        **{f"mean_{metric}": scores for metric, scores in eval_scores.items()}
    }
    df_output = pd.DataFrame([output_data])
    csv_file = f'trial_results_{eval_metrics}_{config["eval_top"]}.csv'
    df_output.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
    return mean_eval_score


def best_trial_callback(study, trial):
    """
    Optuna callback: prints info whenever a new best trial is found.

    Args:
        study (optuna.study.Study): The current study object.
        trial (optuna.trial.FrozenTrial): The current trial.
    """
    if study.best_trial.number == trial.number:
        print(f'New best value: {trial.value:.4f} achieved at trial {trial.number}.')
        print(f'Best parameters: {trial.params}')


def run_optuna_optimization(config, df):
    """
    Runs Optuna optimization with the specified config and DataFrame.

    Args:
        config (dict): Configuration parameters.
        df (pd.DataFrame): Input DataFrame for model training.

    Returns:
        tuple: (study, best_param, best_score)
    """
    def delete_incomplete_study(study_name, storage, n_trials):
        study = optuna.load_study(study_name=study_name, storage=storage)
        if len(study.trials) < n_trials:
            print(f"Deleting incomplete study '{study_name}' with only {len(study.trials)} trials.")
            os.remove(config['optuna_db_path'])
            return False
        return True

    if os.path.exists(config['optuna_db_path']):
        if delete_incomplete_study(config['study_name'], config['optuna_storage'], config['n_trials']):
            study = optuna.load_study(study_name=config['study_name'], storage=config['optuna_storage'])
        else:
            study = optuna.create_study(direction='maximize', study_name=config['study_name'], storage=config['optuna_storage'])
    else:
        study = optuna.create_study(direction='maximize', study_name=config['study_name'], storage=config['optuna_storage'])

    remaining_trials = max(0, config['n_trials'] - len(study.trials))
    if remaining_trials > 0:
        partial_objective = partial(objective, df, config)
        study.optimize(partial_objective, n_trials=remaining_trials, callbacks=[best_trial_callback])
    else:
        print(f"Study '{config['study_name']}' already completed the required {config['n_trials']} trials.")

    if len(study.trials) == 0:
        raise ValueError("No trials were completed in the study.")

    best_trial = study.best_trial
    best_param = best_trial.params.copy()
    best_score = best_trial.value
    return study, best_param, best_score


def feature_importance(df, optuna_name, best_params, score_col_name="score"):
    """
    Computes feature importance for the best model and plots it.

    Args:
        df (pd.DataFrame): Input data.
        optuna_name (str): Model name (e.g., 'rf', 'lgbm').
        best_params (dict): Best hyperparameters from the optimization.
        score_col_name (str): Score column name.
    """
    X = df.drop(score_col_name, axis=1)
    y = df[score_col_name]

    if 'lgbm' in optuna_name:
        dtrain = lgb.Dataset(X, label=y)
        model = lgb.train(best_params, dtrain)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importance(importance_type='gain')
        })
    elif 'rf' in optuna_name:
        model = RandomForestClassifier(**best_params, random_state=42)
        model.fit(X, y)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        })
    else:
        return

    feature_importance.sort_values(by='importance', ascending=False, inplace=True)
    plt.figure(figsize=(10, 6))
    plt.title('Feature importance')
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.gca().invert_yaxis()
    plt.savefig(f'feature_importance_{optuna_name}.png')
    move_files_to_folder(optuna_name, [f'feature_importance_{optuna_name}.png'])


def pred_with_best_param(df_train, df_test, best_params, optuna_name, score_col_name_train="score", score_col_name_test="score"):
    """
    Trains a model with the best parameters and predicts on df_test.

    Args:
        df_train (pd.DataFrame): Training DataFrame.
        df_test (pd.DataFrame): Test DataFrame.
        best_params (dict): Best hyperparameters from optimization.
        optuna_name (str): Model name.
        score_col_name_train (str): Score column in training data.
        score_col_name_test (str): Score column in test data.

    Returns:
        np.ndarray: Predictions on df_test.
    """
    model = None
    if 'rf' in optuna_name:
        model = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            random_state=42
        )
    elif 'lgbm' in optuna_name:
        model = lgb.LGBMClassifier(
            boosting_type=best_params['boosting_type'],
            num_leaves=best_params['num_leaves'],
            learning_rate=best_params['learning_rate'],
            n_estimators=best_params['n_estimators']
        )
    elif 'gpr' in optuna_name:
        kernel = best_params.get('kernel', 1.0 * RBF(1.0))
        alpha = best_params.get('alpha', 1e-10)
        n_restarts_optimizer = best_params.get('n_restarts_optimizer', 0)
        model = GaussianProcessClassifier(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=42
        )
    elif 'ae' in optuna_name:
        encoding_dim = best_params.get('latent_dim', 64)
        epochs = best_params.get('epochs', 50)
        batch_size = best_params.get('batch_size', 32)

        X_train = df_train.drop(columns=[score_col_name_train])
        X_test = df_test.drop(columns=[score_col_name_test])

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        input_dim = X_train_scaled.shape[1]
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        autoencoder.fit(
            X_train_scaled, X_train_scaled,
            epochs=epochs, batch_size=batch_size, shuffle=True,
            validation_data=(X_test_scaled, X_test_scaled),
            callbacks=[early_stopping]
        )
        X_test_pred = autoencoder.predict(X_test_scaled)
        reconstruction_error = np.mean(np.square(X_test_scaled - X_test_pred), axis=1)
        return reconstruction_error
    else:
        raise ValueError(f"Unknown model name: {optuna_name}")

    if model is not None:
        X_train = df_train.drop(score_col_name_train, axis=1)
        y_train = df_train[score_col_name_train]
        X_test = df_test.drop(score_col_name_test, axis=1)
        model.fit(X_train, y_train)
        if isinstance(model, GaussianProcessRegressor):
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
        return y_pred
    else:
        raise ValueError(f"Failed to instantiate model for {optuna_name}")


def plot_roc_curve(y_true, y_pred, comment, pos_label=2):
    """
    Plots the ROC curve and saves it as 'roc_curve_{comment}.png'.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted scores or probabilities.
        comment (str): A suffix to use in the filename.
        pos_label (int): The positive class label (default 2).
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC (AUC = %0.2f)' % roc_auc)
    plt.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax = plt.gca()
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_aspect('equal')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve, AUC: {np.round(roc_auc, 5)}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'roc_curve_{comment}.png')


def plot_pr_curve(y_true, y_pred, comment):
    """
    Plots Precision-Recall curve and saves it as 'pr_curve_{comment}.png'.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted scores or probabilities.
        comment (str): A suffix to use in the filename.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=2)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label='PR (AUC = %0.2f)' % pr_auc)
    plt.fill_between(recall, precision, alpha=0.2, color='blue')
    ax = plt.gca()
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR curve, AUC: {np.round(pr_auc, 5)}')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f'pr_curve_{comment}.png')


def violin_plot(df_pred, pred_score_col_name="y_pred", score_col_name="score", filename=None):
    """
    Creates a violin plot showing the distribution of predicted scores grouped by actual score.

    Args:
        df_pred (pd.DataFrame): DataFrame containing predicted and actual scores.
        pred_score_col_name (str): Column with predicted scores. Default 'y_pred'.
        score_col_name (str): Column with actual scores. Default 'score'.
        filename (str): If specified, the plot is saved as 'violin_plot_{filename}.png'.
    """
    plt.style.use('default')
    sns.set_style('white')

    fig, ax = plt.subplots()
    list_original_score = sorted(set(df_pred[score_col_name].values.tolist()))
    list_x = [df_pred[df_pred[score_col_name] == sc][pred_score_col_name].values.tolist() for sc in list_original_score]

    parts = ax.violinplot(list_x)
    cmap = plt.get_cmap("coolwarm")
    for i, pc in enumerate(parts['bodies']):
        color = cmap(i / (len(list_original_score) - 1))
        pc.set_facecolor(color)

    ax.set_xticks(range(1, len(list_original_score) + 1))
    ax.set_xticklabels(list_original_score)
    ax.set_xlabel('Original Score')
    ax.set_ylabel('Predicted Score')
    min_val, max_val = df_pred[pred_score_col_name].min(), df_pred[pred_score_col_name].max()
    ax.set_ylim(int(min_val), int(max_val) + 1)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(f'violin_plot_{filename}.png')


def map_binary_labels(y_true):
    """
    Maps labels to binary (0/1) by assigning min -> 0 and max -> 1.
    Raises an error if more than two unique values exist.

    Args:
        y_true (array-like): True labels.

    Returns:
        np.ndarray: Mapped binary labels.

    Raises:
        ValueError: If y_true has more than two unique values.
    """
    unique_values = np.unique(y_true)
    if len(unique_values) != 2:
        raise ValueError("y_true should only contain two unique values for binary classification.")
    return np.where(y_true == unique_values[0], 0, 1)


def find_best_threshold(y_true, probabilities, metric='mcc', top_n=None):
    """
    Finds the best threshold for a given metric ('f1','mcc','sensitivity','specificity','kappa').

    Args:
        y_true (array-like): True labels.
        probabilities (array-like): Predicted probabilities.
        metric (str): Metric to optimize. Default 'mcc'.
        top_n (int, optional): If set, only top_n predictions are considered.

    Returns:
        tuple: (best_threshold, best_metric_value).
    """
    min_pred = np.min(probabilities)
    max_pred = np.max(probabilities)
    thresholds = np.linspace(min_pred, max_pred, 100)

    best_threshold = min_pred
    best_metric_value = 0
    y_true_map = map_binary_labels(y_true)

    for threshold in thresholds:
        if top_n:
            indices = np.argsort(probabilities)[-top_n:]
            y_true_top_n = y_true_map[indices]
            probabilities_top_n = probabilities[indices]
            predictions = (probabilities_top_n > threshold).astype(int)
        else:
            predictions = (probabilities > threshold).astype(int)
            y_true_top_n = y_true_map

        if metric == 'f1':
            metric_value = f1_score(y_true_top_n, predictions)
        elif metric == 'mcc':
            metric_value = matthews_corrcoef(y_true_top_n, predictions)
        elif metric in ['sensitivity', 'specificity']:
            tn, fp, fn, tp = confusion_matrix(y_true_top_n, predictions).ravel()
            if metric == 'sensitivity':
                metric_value = tp / (tp + fn) if (tp + fn) != 0 else 0
            else:
                metric_value = tn / (tn + fp) if (tn + fp) != 0 else 0
        elif metric == 'kappa':
            metric_value = cohen_kappa_score(y_true_top_n, predictions)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_threshold = threshold

    return best_threshold, best_metric_value


###########################
# Main Workflow Functions
###########################

def load_ps2data_and_initialize(config):
    """
    Loads CSV data, extracts columns for recommendation, and retrieves unique items.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        tuple: (df_exp, ax_dict)
    """
    try:
        df_csv = load_csv_to_dataframe(config["ps2_filename"], config["ps2_columns"])
    except FileNotFoundError as e:
        raise FileNotFoundError(f"CSV file {config['ps2_filename']} not found: {e}")

    try:
        df_exp = df_csv[config["ps2_columns_use"]].copy()
    except KeyError as e:
        raise ValueError(f"Specified columns {config['ps2_columns_use']} not found in DataFrame: {e}")

    try:
        ax_dict = get_unique_values(df_exp, df_exp.columns, keyword=config["keyword"])
    except Exception as e:
        raise ValueError(f"Failed to retrieve unique items: {e}")

    ax_dict = replace_ion_items(ax_dict, config["keyword"])
    del ax_dict[config["keyword"]]
    return df_exp, ax_dict


def load_factors_data(ax_dict, factor_axis_file, config):
    """
    Loads factor matrices into dictionaries for binary and ternary systems.

    Args:
        ax_dict (dict): Dictionary of columns -> unique items.
        factor_axis_file (list): List of factor-axis filenames.
        config (dict): Configuration dictionary.

    Returns:
        tuple: (dict_factors_ps2, dict_factors_ps3, ax_dict_ps3)
    """
    keyword = config.get("keyword", None)
    if keyword is None:
        raise ValueError("Config is missing 'keyword' key.")

    use_factor_axes = extract_keys_containing_strings(ax_dict, [keyword])
    if not use_factor_axes:
        raise ValueError(f"No axis names containing '{keyword}' found in dict keys.")

    relevant_files = [filename for filename in factor_axis_file if any(kwd in filename for kwd in use_factor_axes)]
    if not relevant_files:
        raise ValueError(f"No relevant factor files found for keyword '{keyword}'.")

    factors = [load_factors(file) for file in relevant_files]
    dict_factors_ps2 = factor_axis_to_dict(factors, use_factor_axes)
    dict_factors_ps2 = replace_key_lists_with_first_key_list(dict_factors_ps2, keyword)

    dict_factors_ps3 = add_next_key_to_dict(dict_factors_ps2, keyword, factors[0])
    dict_factors_ps3 = replace_key_lists_with_first_key_list(dict_factors_ps3, keyword)

    ax_dict_ps3 = add_next_key_to_dict(ax_dict, keyword, ax_dict[use_factor_axes[0]])
    ax_dict_ps3 = replace_key_lists_with_first_key_list(ax_dict_ps3, keyword)

    ratio_ps3 = []
    for weights in generate_weights(value=0.1, dimensions=3, round_num=1, start_value=0, end_value=1, min_non_zero_elements=2):
        ratio_ps3.append(list(weights))
    del ax_dict_ps3["ratio"]
    ax_dict_ps3["ratio"] = [":".join(map(str, sublist)) for sublist in ratio_ps3]

    return dict_factors_ps2, dict_factors_ps3, ax_dict_ps3


def process_interchangeable_data(df, target_columns, ratio_column, interchange_filename):
    """
    Removes interchangeable duplicates and writes to file if file does not exist.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_columns (list): Columns that are interchangeable.
        ratio_column (str): Name of ratio column.
        interchange_filename (str): Filename to store result.

    Returns:
        pd.DataFrame: DataFrame after removing interchangeable duplicates.
    """
    if not os.path.exists(interchange_filename):
        print(f'----Now making {interchange_filename}----')
        df_rem = remove_interchangeable(df, target_columns, ratio_column)
        df_rem = duplicated_items_score_change(df_rem, target_columns, 1)
        df_rem.to_csv(interchange_filename, index=False)
    else:
        df_rem = pd.read_csv(interchange_filename)
    return df_rem


def create_combinations(ax_dict, dup_filename, hypo_filename, target_cols, ratio_col="ratio", score_col="score"):
    """
    Creates DataFrames for duplicated and hypothetical combinations, removes interchangeable items,
    and writes them to files if not found.

    Args:
        ax_dict (dict): Dictionary of columns -> unique items.
        dup_filename (str): Filename for duplicated combos.
        hypo_filename (str): Filename for hypothetical combos.
        target_cols (list): Columns that are interchangeable.
        ratio_col (str): Ratio column. Default 'ratio'.
        score_col (str): Score column. Default 'score'.

    Returns:
        tuple: (df_dup_rem, df_hypo_rem)
    """
    if not os.path.exists(dup_filename):
        print(f'----Now making {dup_filename}----')
        df_dup = create_duplicated_combination(ax_dict, mode="all", score_col_name=score_col, temporary_score=1)
        df_dup_rem = remove_interchangeable(df_dup, target_cols, ratio_col)
        df_dup_rem.to_csv(dup_filename, index=False)
    else:
        df_dup_rem = pd.read_csv(dup_filename)

    if not os.path.exists(hypo_filename):
        print(f'----Now making {hypo_filename}----')
        df_hypo = create_hypothetical_combination(ax_dict, mode="all", score_col_name=score_col, temporary_score=-1)
        df_hypo_rem = remove_interchangeable(df_hypo, target_cols, ratio_col)
        df_hypo_rem.to_csv(hypo_filename, index=False)
    else:
        df_hypo_rem = pd.read_csv(hypo_filename)

    return df_dup_rem, df_hypo_rem


def embedding_process(df, dict_factors, use_moment, target_cols, add_cols, embedding_filename, score_col_name="score"):
    """
    Performs embedding and saves/loads from a file.

    Args:
        df (pd.DataFrame): Input DataFrame.
        dict_factors (dict): Factor dictionary.
        use_moment (list): Moments to compute (e.g., ['mean','std','cov']).
        target_cols (list): Columns for moment calculations.
        add_cols (list): Additional columns to embed.
        embedding_filename (str): Filename to store/load embedding.
        score_col_name (str): Score column name.

    Returns:
        pd.DataFrame: The embedding result.
    """
    if not os.path.exists(embedding_filename):
        print(f'----Now making {embedding_filename}----')
        df_emb = create_embedding(df, use_moment, target_cols, target_cols, add_cols, dict_factors, score_col_name=score_col_name, decimal_places=5)
        df_emb.to_csv(embedding_filename, index=False)
    else:
        df_emb = embedding_file_to_df(embedding_filename, use_moment, score_col_name=score_col_name)
    return df_emb


def perform_dimension_reduction(df_list, method, dim, score_col_name="score", decimal_places=5):
    """
    Performs dimension reduction on a list of DataFrames using a specified method.

    Args:
        df_list (list): List of DataFrames.
        method (str): Dimensionality reduction method ('PCA','SVD','t-SNE','UMAP').
        dim (int): Target dimension.
        score_col_name (str): Name of the score column.
        decimal_places (int): Decimal rounding.

    Returns:
        list: List of dimension-reduced DataFrames.
    """
    return dimension_reduction(df_list, method, dim, score_col_name=score_col_name, decimal_places=decimal_places)


def perform_optimization(df, score_col_name="score", n_dim=None, method=None):
    """
    Optionally performs dimension reduction on df, then runs model hyperparameter optimization.

    Args:
        df (pd.DataFrame): Input data for training.
        score_col_name (str): Score column name.
        n_dim (int, optional): If provided, reduce to n_dim dimensions first.
        method (str, optional): Dimensionality reduction method.

    Returns:
        tuple: (study, best_param, best_score)
    """
    if n_dim is not None and method is not None:
        print(f"----START dimension reduction by {method}, n_dim={n_dim}----")
        df, = perform_dimension_reduction([df], method, n_dim, score_col_name=score_col_name)
    study, best_param, best_score = run_optuna_optimization(config, df)
    return study, best_param, best_score


def save_and_plot_results(df, y_pred, score_col_name, comment):
    """
    Saves predictions to df, plots ROC/PR curves, and creates a violin plot.

    Args:
        df (pd.DataFrame): Input DataFrame with actual score.
        y_pred (np.ndarray): Predicted scores or probabilities.
        score_col_name (str): Column with actual scores.
        comment (str): Suffix for plot filenames.
    """
    df.insert(0, 'y_pred', y_pred)
    df = filter_non_zero_ratios(df, ratio_col='ratio')

    unique_classes = df[score_col_name].nunique()
    if unique_classes > 2:
        roc_auc = roc_auc_score(df[score_col_name], df['y_pred'], multi_class='ovr')
    else:
        roc_auc = roc_auc_score(df[score_col_name], df['y_pred'])

    y_true_transformed = df[score_col_name].map({1: 0, 2: 1})
    f1_threth, f1_best = find_best_threshold(y_true_transformed, df['y_pred'], metric='f1')
    mcc_threth, mcc_best = find_best_threshold(y_true_transformed, df['y_pred'], metric='mcc')

    print(f"ROC-AUC: {roc_auc}")
    print(f"F1: {f1_best} @ threshold : {f1_threth}")
    print(f"MCC: {mcc_best} @ threshold : {mcc_threth}")

    plot_roc_curve(df[score_col_name], df['y_pred'], comment)
    plot_pr_curve(df[score_col_name], df['y_pred'], comment)
    violin_plot(df, pred_score_col_name="y_pred", score_col_name=score_col_name, filename=comment)


def main(config):
    """
    Main function integrating all data loading, factor embedding, model optimization, and final predictions.
    """
    try:
        logger.info("Starting data processing...")

        # Load and initialize
        df_ps2, ax_dict_ps2 = load_ps2data_and_initialize(config)
        ax_dict_ps2 = replace_key_lists_with_first_key_list(ax_dict_ps2, config["keyword"])
        factor_axis_files = get_factor_axis_files()
        dict_factors_ps2, dict_factors_ps3, ax_dict_ps3 = load_factors_data(ax_dict_ps2, factor_axis_files, config)

        # Binary system data
        combi_columns_ps2 = [key for key in ax_dict_ps2.keys() if config["keyword"] in key]
        ratio_column_ps2 = [key for key in df_ps2.columns if "ratio" in key][0]
        df_ps2_rem = process_interchangeable_data(
            df=df_ps2,
            target_columns=combi_columns_ps2,
            ratio_column=ratio_column_ps2,
            interchange_filename=config["interchange_filename_ps2"]
        )

        df_dup_ps2_rem, df_hypo_ps2_rem = create_combinations(
            ax_dict=ax_dict_ps2,
            dup_filename=config["dup_ps2_filename"],
            hypo_filename=config["hypo_ps2_filename"],
            target_cols=combi_columns_ps2
        )

        df_ps2_rem_emb = embedding_process(
            df=df_ps2_rem,
            dict_factors=dict_factors_ps2,
            use_moment=config["use_moment"],
            target_cols=combi_columns_ps2,
            add_cols=[],
            embedding_filename=config["embedding_filenames"]["ps2_rem"]
        )

        df_dup_ps2_rem_emb = embedding_process(
            df=df_dup_ps2_rem,
            dict_factors=dict_factors_ps2,
            use_moment=config["use_moment"],
            target_cols=combi_columns_ps2,
            add_cols=[],
            embedding_filename=config["embedding_filenames"]["dup_ps2_rem"]
        )

        df_hypo_ps2_rem_emb = embedding_process(
            df=df_hypo_ps2_rem,
            dict_factors=dict_factors_ps2,
            use_moment=config["use_moment"],
            target_cols=combi_columns_ps2,
            add_cols=[],
            embedding_filename=config["embedding_filenames"]["hypo_ps2_rem"]
        )

        # Ternary system data
        combi_columns_ps3 = [key for key in ax_dict_ps3.keys() if config["keyword"] in key]
        df_dup_ps3_rem, df_hypo_ps3_rem = create_combinations(
            ax_dict=ax_dict_ps3,
            dup_filename=config["dup_ps3_filename"],
            hypo_filename=config["hypo_ps3_filename"],
            target_cols=combi_columns_ps3
        )
        df_dup_ps3_rem_emb = embedding_process(
            df=df_dup_ps3_rem,
            dict_factors=dict_factors_ps3,
            use_moment=config["use_moment"],
            target_cols=combi_columns_ps3,
            add_cols=[],
            embedding_filename=config["embedding_filenames"]["dup_ps3_rem"]
        )
        df_hypo_ps3_rem_emb = embedding_process(
            df=df_hypo_ps3_rem,
            dict_factors=dict_factors_ps3,
            use_moment=config["use_moment"],
            target_cols=combi_columns_ps3,
            add_cols=[],
            embedding_filename=config["embedding_filenames"]["hypo_ps3_rem"]
        )

        # If a ternary CSV is specified
        if config["ps3_filename"]:
            logger.info("Processing three-component known data...")
            df_csv_ps3 = load_csv_to_dataframe(config["ps3_filename"])
            column_names_ps3 = config["ps3_columns_use"]
            df_ps3 = df_csv_ps3[column_names_ps3].copy()
            ratio_column_ps3 = [key for key in df_ps3.columns if "ratio" in key][0]
            df_ps3_rem = process_interchangeable_data(
                df=df_ps3,
                target_columns=combi_columns_ps3,
                ratio_column=ratio_column_ps3,
                interchange_filename=config['interchange_filename_ps3']
            )
            df_ps3_rem_emb = embedding_process(
                df=df_ps3_rem,
                dict_factors=dict_factors_ps3,
                use_moment=config["use_moment"],
                target_cols=combi_columns_ps3,
                add_cols=[],
                embedding_filename=config["embedding_filenames"]["ps3_rem"]
            )

        # Prepare training data (binary system). Optionally add hypothetical data.
        df_hypo_ps2_rem["score"] = 1
        df_hypo_ps2_rem_emb["score"] = 1
        df_hypo_ps2_rem_emb_frac, df_hypo_ps2_rem_emb_rest = get_random_rows(df_hypo_ps2_rem_emb, config["num_ps2_hypo_for_train"])
        df_hypo_ps2_rem_rest = df_hypo_ps2_rem.loc[df_hypo_ps2_rem_emb_rest.index]

        df_train_frac_ps2_emb = pd.concat(
            [df_ps2_rem_emb, df_dup_ps2_rem_emb, df_hypo_ps2_rem_emb_frac]
        ).reset_index(drop=True)
        df_train_frac_ps2_emb.to_csv("train_ps2.csv", index=False)

        # Model optimization
        logger.info("Starting model optimization...")
        study, best_params, best_score = perform_optimization(df=df_train_frac_ps2_emb)

        # Predict on leftover hypothetical data
        logger.info("Making predictions for hypothetical ps2 data (not used for training)...")
        df_hypo_ps2_rem_emb_rest["score"] = 1
        y_pred_hypo_ps2 = pred_with_best_param(
            df_train=df_train_frac_ps2_emb,
            df_test=df_hypo_ps2_rem_emb_rest,
            best_params=best_params,
            optuna_name=config["optuna_name"]
        )
        df_hypo_ps2_rem_emb_rest["y_pred"] = y_pred_hypo_ps2
        df_pred_all_ps2 = pd.concat([df_hypo_ps2_rem_rest, df_hypo_ps2_rem_emb_rest], axis=1)
        df_pred_all_ps2.to_csv(config["pred_only_hypo_ps2_rest_filename"])

        # If ps3_filename is empty, we consider hypothetical ps3 data
        if config["ps3_filename"] == "":
            logger.info("Making predictions for hypothetical ps3 data...")
            df_hypo_ps3_rem_emb["score"] = 1
            y_pred_hypo = pred_with_best_param(
                df_train=df_train_frac_ps2_emb,
                df_test=df_hypo_ps3_rem_emb,
                best_params=best_params,
                optuna_name=config["optuna_name"]
            )
            df_hypo_ps3_rem_emb["y_pred"] = y_pred_hypo
            df_pred_all_ps3 = pd.concat([df_hypo_ps3_rem, df_hypo_ps3_rem_emb], axis=1)
            df_pred_all_ps3.to_csv(config["pred_only_hypo_ps3_filename"])
        else:
            logger.info("Making predictions for experimental data...")
            df_hypo_ps3_rem["score"] = 1
            df_hypo_ps3_rem_emb["score"] = 1
            df_test_all_ps3 = pd.concat([df_ps3_rem, df_hypo_ps3_rem]).reset_index(drop=True)
            df_hypo_ps3_rem_emb = df_hypo_ps3_rem_emb.drop('score', axis=1)
            df_test_all_ps3_emb = pd.concat([df_ps3_rem_emb, df_hypo_ps3_rem_emb]).reset_index(drop=True)
            y_pred_ps3 = pred_with_best_param(
                df_train=df_train_frac_ps2_emb,
                df_test=df_test_all_ps3_emb,
                best_params=best_params,
                optuna_name=config["optuna_name"]
            )
            df_test_all_ps3_emb = df_test_all_ps3_emb.drop('score', axis=1)
            df_pred_all_ps3 = pd.concat([df_test_all_ps3, df_test_all_ps3_emb], axis=1)
            save_and_plot_results(
                df=df_pred_all_ps3,
                y_pred=y_pred_ps3,
                score_col_name="score",
                comment="ps3"
            )
            df_pred_all_ps3.to_csv(config["pred_ps3_filename"], index=False)

        logger.info("Data processing completed successfully.")

    except Exception as e:
        logger.error("An error occurred during data processing.", exc_info=True)
        raise DataProcessingError(f"Data processing failed: {e}")


if __name__ == "__main__":
    argvs = sys.argv
    eval_metrics = str(argvs[1])
    if argvs[2] == "all":
        eval_top = None
    else:
        eval_top = int(argvs[2])
    pred_model = argvs[3]

    config = {
        "ps2_filename": "data_ps2",
        "ps2_columns": ["score", "ion1", "ion2", "ratio"],
        "ps2_columns_use": ["score", "ion1", "ion2", "ratio"],
        "ps3_columns_use": ["score", "ion1", "ion2", "ion3", "ratio"],
        "ps3_filename": "data_ps3",
        "pred_ps3_filename": "pred_ps3_all.csv",
        "pred_only_hypo_ps3_filename": "pred_only_hypo.csv",
        "pred_only_hypo_ps2_rest_filename": "pred_only_hypo_ps2_rest.csv",
        "eval_metrics": eval_metrics,
        "eval_top": eval_top,
        "pred_model": pred_model,
        "keyword": "ion",
        "interchange_filename_ps2": "df_ps2_rem.csv",
        "interchange_filename_ps3": "df_ps3_rem.csv",
        "dup_ps2_filename": "df_dup_ps2_rem.csv",
        "hypo_ps2_filename": "df_hypo_ps2_rem.csv",
        "dup_ps3_filename": "df_dup_ps3_rem.csv",
        "hypo_ps3_filename": "df_hypo_ps3_rem.csv",
        "embedding_filenames": {
            "ps2_rem": "df_ps2_rem_emb.csv",
            "ps3_rem": "df_ps3_rem_emb.csv",
            "dup_ps2_rem": "df_dup_ps2_rem_emb.csv",
            "hypo_ps2_rem": "df_hypo_ps2_rem_emb.csv",
            "dup_ps3_rem": "df_dup_ps3_rem_emb.csv",
            "hypo_ps3_rem": "df_hypo_ps3_rem_emb.csv"
        },
        "use_moment": ["mean", "std", "cov"],  # "kurt", "skew" can be added if needed
        "optuna_name": f'multi_{pred_model}_{eval_metrics}_{eval_top}',
        'optuna_db_path': f'multi_{pred_model}_{eval_metrics}_{eval_top}_optuna.db',
        'optuna_storage': f'sqlite:///multi_{pred_model}_{eval_metrics}_{eval_top}_optuna.db',
        'study_name': f'multi_{pred_model}_{eval_metrics}_{eval_top}',
        'n_trials': 100,
        'n_fold': 10,
        'num_ps2_hypo_for_train': 0.1,
        "hypo_yes": "yes"
    }

    main(config)
