#!/usr/bin/env python
# coding: utf-8

import os
import sys
import itertools
import glob
import shutil

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')  # If not using a GUI, specify this backend
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

from typing import List, Tuple

import optuna
from optuna.logging import get_logger

from functools import partial
import math

from tensorly import tucker_tensor
from tensorly.decomposition import tucker, parafac, non_negative_tucker

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    mean_squared_error, roc_auc_score, f1_score, log_loss, confusion_matrix,
    roc_curve, auc, average_precision_score, matthews_corrcoef, cohen_kappa_score,
    precision_recall_curve, pairwise_distances
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS, TSNE

from scipy.spatial.distance import pdist, squareform
from scipy.special import softmax
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import linkage, dendrogram
# from sklearn.preprocessing import StandardScaler  # (Commented out, used if needed)

########################################
# Functions in use
########################################

def get_unique_values(df, column_names, split=False, delimiter="__", keyword=None):
    """
    Returns a dictionary of unique values from specified columns in a dataframe.
    If split=True, split values by a delimiter before getting unique values.

    Args:
        df (pd.DataFrame): The target dataframe.
        column_names (list): List of column names from which to extract unique values.
        split (bool): Whether to split each value by the specified delimiter. Defaults to False.
        delimiter (str): Delimiter for splitting values. Defaults to '__'.
        keyword (str): Keyword used to filter the column names if needed. Defaults to None.

    Returns:
        dict: Dictionary where keys are column names (or a keyword) and values are lists of unique values.
    """
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
        if column_name not in unique_dict:  # Skip columns already processed by keyword
            if split:
                split_values = df[column_name].str.split(delimiter).explode().unique().tolist()
                unique_dict[column_name] = split_values
            else:
                unique_dict[column_name] = df[column_name].unique().tolist()

    return unique_dict


def permute_columns_in_dataframe(df, columns_to_permute, ratio_col_name, score_col_name):
    """
    Creates new rows for all permutations of specified columns in the dataframe.

    Args:
        df (pd.DataFrame): The dataframe to permute.
        columns_to_permute (str): Common substring to identify columns to be permuted.
        ratio_col_name (str): Column name that holds the ratio information to permute.
        score_col_name (str): Column name holding the score.

    Returns:
        pd.DataFrame: A new dataframe with permuted columns.
    """
    valid_columns = df.columns
    permute_columns = [col for col in df.columns if columns_to_permute in col]
    dim = len(permute_columns)

    # Split ratio_col_name by ":" into new columns, then drop the original ratio_col_name
    df[[f"{ratio_col_name}_{i}" for i in range(dim)]] = df[ratio_col_name].str.split(":", expand=True)
    df = df.drop(columns=[ratio_col_name])

    # Generate permutations
    permutations = list(itertools.permutations(range(dim)))

    new_rows = []
    for _, row in df.iterrows():
        for perm in permutations:
            new_row = []
            for i in perm:
                new_row.append(row[permute_columns[i]])
            for i in perm:
                new_row.append(row[f"{ratio_col_name}_{i}"])
            new_row.append(row[score_col_name])
            new_row.extend(row[col] for col in df.columns if col not in permute_columns
                           + [f"{ratio_col_name}_{i}" for i in range(dim)] + [score_col_name])
            new_rows.append(new_row)

    df_perm = pd.DataFrame(
        new_rows,
        columns=permute_columns
        + [f"{ratio_col_name}_{i}" for i in range(dim)]
        + [score_col_name]
        + [
            col
            for col in df.columns
            if col not in permute_columns + [f"{ratio_col_name}_{i}" for i in range(dim)] + [score_col_name]
        ]
    )

    # Reconstruct the ratio column
    df_perm[[f"{ratio_col_name}_{i}" for i in range(dim)]] = df_perm[
        [f"{ratio_col_name}_{i}" for i in range(dim)]
    ].astype(str)
    df_perm[ratio_col_name] = df_perm[
        [f"{ratio_col_name}_{i}" for i in range(dim)]
    ].agg(':'.join, axis=1)
    df_perm = df_perm.drop(columns=[f"{ratio_col_name}_{i}" for i in range(dim)])

    # Drop duplicates and reorder columns to the original order
    df_perm = df_perm.drop_duplicates()
    df_perm = df_perm[valid_columns]

    return df_perm


def create_duplicated_combination(d, mode, score_col_name="score", temporary_score=1):
    """
    Generates a dataframe containing all combinations of elements in a dictionary,
    then filters only rows that contain duplicated elements within each row.

    Args:
        d (dict): Keys are column names, values are lists of possible elements.
        mode (int or str): If an integer, randomly sample that many rows. 
                           If any other string, use all combinations.
        score_col_name (str): Name of the score column to be added. Default is 'score'.
        temporary_score (int): Score value to assign. Default is 1.

    Returns:
        pd.DataFrame: A dataframe with duplicated-element rows.
    """
    combinations = list(itertools.product(*d.values()))
    df = pd.DataFrame(combinations, columns=d.keys())

    # Keep only rows where there are duplicate elements in the row
    df = df[df.apply(lambda x: len(x) != len(set(x)), axis=1)]

    if isinstance(mode, int):
        df = df.sample(n=mode)

    df.insert(0, score_col_name, temporary_score)
    return df


def label_encode_columns(df, output_filename, score_name="score", drop_score=0):
    """
    Label-encodes columns in a dataframe except for a specified score column.
    If existing encoding files are present, they will be used. Otherwise, new ones are created.

    Args:
        df (pd.DataFrame): Dataframe to encode.
        output_filename (str): Filename prefix for the encoded output.
        score_name (str): Score column name. Default is 'score'.
        drop_score (int or float): Exclude rows with this score value. Default is 0.

    Returns:
        pd.DataFrame: Label-encoded dataframe.
    """
    df = df[df[score_name] != drop_score]
    columns = df.drop(score_name, axis=1).columns

    for col in columns:
        le = LabelEncoder()
        axis_file = f'axis_{col}'

        if not os.path.exists(axis_file):
            le.fit(df[col])
            classes = pd.DataFrame({col: le.classes_})
            classes.to_csv(axis_file, header=False, index=False)
        else:
            classes = pd.read_csv(axis_file, header=None).iloc[:, 0]
            le.classes_ = classes.to_numpy()

        df[col] = le.transform(df[col])

    output_path = f'index_{output_filename}'
    df.to_csv(output_path, index=False, sep=",")
    return df


def perform_tucker_decomposition(df_idx, rank, score_col_idx=0, output_result_file=True, output_factor_file=True):
    """
    Performs Tucker decomposition at a specified rank, reconstructs the tensor,
    and optionally outputs the result and factor matrices to files.

    Args:
        df_idx (pd.DataFrame): The label-encoded dataframe.
        rank (list): The ranks for Tucker decomposition, e.g., [2, 3, 2].
        score_col_idx (int): The column index holding the score. Default is 0.
        output_result_file (bool): Whether to output the reconstruction result to a file. Default is True.
        output_factor_file (bool): Whether to output the factor matrices to files. Default is True.

    Returns:
        df_result (pd.DataFrame): Dataframe containing the coordinates, original score, and reconstructed score.
        factors (list of np.ndarray): List of factor matrices.
        core (np.ndarray): The core tensor from Tucker decomposition.
    """
    score_col_name = df_idx.columns[score_col_idx]
    ave_score = df_idx[score_col_name].mean()
    stddev_score = df_idx[score_col_name].std()

    # Exclude the score column and build axis filename pattern
    col_name = df_idx.drop(columns=[score_col_name]).columns
    axis_filename = "axis_"

    len_list = []
    for ln in col_name:
        _l = np.genfromtxt(axis_filename + str(ln), delimiter=",")
        len_list.append(len(_l))

    # Build a zero tensor with mean score
    zeromat = np.zeros(len_list, dtype='float64') + ave_score

    # Fill the tensor with actual scores
    nonzero = np.empty((len(rank), 0), dtype='int')
    idxlist = []
    scores = []
    scorelist = sorted(df_idx[score_col_name].unique().tolist())

    for sc in scorelist:
        idx = np.array(df_idx[df_idx[score_col_name] == sc].iloc[:, 1:])
        for i in idx:
            zeromat.itemset(tuple(i), sc)
        _id = np.where(np.array(zeromat) == sc)
        _ones = np.zeros(np.array(_id).shape[1]) + sc
        scores.extend(_ones)
        idxlist.append(_id)
        nonzero = np.concatenate((nonzero, _id), axis=1)

    # Standardize the tensor
    zeromat = (zeromat - ave_score) / stddev_score

    # Perform Tucker decomposition
    data_tensor = zeromat
    S = tucker(data_tensor, rank=rank, n_iter_max=20000, init="svd", tol=1e-5)
    core, factors = S
    score = tucker_tensor.tucker_to_tensor(S)

    # Prepare results
    indices = [list(idx) for idx in itertools.product(*[np.arange(i) for i in len_list])]
    original_scores = (np.array(data_tensor) * stddev_score + ave_score).flatten()
    processed_scores = (score.flatten() * stddev_score + ave_score).reshape(-1, 1)
    result_columns = list(col_name) + ["original_score", "predicted_score"]
    result = np.hstack((indices, original_scores.reshape(-1, 1), processed_scores))

    rank_str = '_'.join(map(str, rank))
    fmt = ["%d"] * len(rank) + ["%.6g", "%.5f"]
    result_file_name = f'result_tucker_{rank_str}'
    header_str = ' '.join(result_columns)

    if output_result_file:
        with open(result_file_name, 'w') as f:
            f.write(header_str + '\n')
        with open(result_file_name, 'a') as f:
            np.savetxt(f, result, delimiter=' ', fmt=fmt)

    df_result = pd.DataFrame(result, columns=result_columns)

    # Optionally output factor matrices
    if output_factor_file:
        for i, column in enumerate(col_name):
            elements = pd.read_csv(f'{axis_filename}{column}', header=None, sep=" ", dtype=str)[0].values
            fmt_factor = ['%s'] + ['%.3g'] * factors[i].shape[1]
            output_df = pd.DataFrame(
                np.vstack((elements, factors[i].T)).T,
                columns=[0] + list(range(1, factors[i].shape[1] + 1))
            ).set_index(0).astype(float)
            output_df.to_csv(f'factor_axis_{column}_{rank_str}', header=False, sep=" ", float_format='%.3g')

    return df_result, factors, core


def stratified_k_fold(df, n_folds, score_col_name='score', shuffle=False, random_state=None):
    """
    Splits a dataframe into stratified folds.

    Args:
        df (pd.DataFrame): The dataset to split.
        n_folds (int): Number of folds.
        score_col_name (str): Name of the score column. Default is 'score'.
        shuffle (bool): Whether to shuffle before splitting. Default is False.
        random_state (int): Random seed. Default is None.

    Returns:
        generator: Yields tuples (train_df, val_df) for each fold.
    """
    df = df.reset_index(drop=True)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    for train_idx, val_idx in folds.split(df, df[score_col_name]):
        yield df.loc[train_idx], df.loc[val_idx]


def map_to_binary_labels(y_true):
    """
    Maps label array to binary (0, 1). Assumes exactly two unique values in y_true,
    mapping the minimum to 0 and the maximum to 1.

    Args:
        y_true (array-like): True labels.

    Returns:
        np.array: Binary labels (0, 1).
    """
    unique_values = np.unique(y_true)

    if len(unique_values) != 2:
        raise ValueError("y_true must contain exactly two unique values for binary classification.")

    return np.where(y_true == unique_values[0], 0, 1)


def find_optimal_threshold(y_true, probabilities, metric='f1', top_n=None):
    """
    Finds the optimal threshold for predicted probabilities to maximize a given metric.

    Args:
        y_true (array-like): True labels.
        probabilities (array-like): Predicted probabilities.
        metric (str): Metric to optimize ('f1', 'mcc', 'sensitivity', 'specificity', 'kappa').
        top_n (int or None): Use only the top_n predictions if specified.

    Returns:
        best_threshold (float): The optimal threshold.
        best_metric_value (float): The metric value at the best threshold.
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
            raise ValueError(f"Unsupported metric '{metric}' provided.")

        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_threshold = threshold

    return best_threshold, best_metric_value


def compute_evaluation_metrics(
    df_result,
    df_val,
    match_columns,
    original_score_col="original_score",
    predicted_score_col="predicted_score",
    positive_label=2
):
    """
    Computes several evaluation metrics by comparing model predictions with validation data.

    Args:
        df_result (pd.DataFrame): Model prediction results (contains original_score and predicted_score).
        df_val (pd.DataFrame): Validation dataframe.
        match_columns (list): Columns to use for merging the two dataframes.
        original_score_col (str): Name of the column containing true scores. Default is 'original_score'.
        predicted_score_col (str): Name of the column containing predicted scores. Default is 'predicted_score'.
        positive_label (int): Label considered positive for binary classification (e.g., 2).

    Returns:
        tuple: Contains values (roc_auc, logloss, rmse, avg_precision, f1, mcc, kappa, true_values_binary, predicted_probabilities).
    """
    df_merged = pd.merge(df_result, df_val, on=match_columns, how='inner')
    true_values = df_merged[original_score_col]
    predicted_scores = df_merged[predicted_score_col].values

    # Convert to binary
    true_values_binary = (true_values == positive_label).astype(int)

    # Apply softmax if needed
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


def cross_validate_model(folds, rank_list, columns, column_score, core_tensor=None):
    """
    Performs cross-validation using Tucker decomposition and evaluates multiple metrics.

    Args:
        folds (generator): Fold splits from stratified_k_fold.
        rank_list (list): Ranks for Tucker decomposition.
        columns (list): Columns to match on for metrics comparison.
        column_score (str): Name of the score column.
        core_tensor (optional): If provided, can be used for direct decomposition logic.

    Returns:
        dict: A dictionary containing mean values of each evaluation metric.
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
        df_result, _, _ = perform_tucker_decomposition(
            df_train,
            rank_list,
            output_result_file=False,
            output_factor_file=False
        )

        roc_auc, logloss, rmse, avg_precision, f1, mcc, kappa, _, _ = compute_evaluation_metrics(
            df_result,
            df_test,
            columns,
            original_score_col=column_score,
            predicted_score_col=f"predicted_{column_score}"
        )

        eval_scores["roc_auc"].append(roc_auc)
        eval_scores["logloss"].append(logloss)
        eval_scores["rmse"].append(rmse)
        eval_scores["avg_precision"].append(avg_precision)
        eval_scores["f1"].append(f1)
        eval_scores["mcc"].append(mcc)
        eval_scores["kappa"].append(kappa)

    return {metric: np.mean(scores) for metric, scores in eval_scores.items()}


def objective(df_idx, config, trial):
    """
    Objective function for Optuna optimization. Trains a model with sampled hyperparameters
    and evaluates performance using the specified metric.

    Args:
        df_idx (pd.DataFrame): Label-encoded input dataframe.
        config (dict): Dictionary containing configuration parameters (e.g., eval_metrics, rank limits).
        trial (optuna.trial.Trial): Optuna Trial object.

    Returns:
        float: Mean evaluation score for the specified metric.
    """
    columns_wo_score = [col for col in df_idx.columns if 'score' not in col]
    column_score = [col for col in df_idx.columns if 'score' in col][0]

    grouped_columns = {}
    for col in columns_wo_score:
        if config["keyword"] in col:
            if config["keyword"] not in grouped_columns:
                grouped_columns[config["keyword"]] = []
            grouped_columns[config["keyword"]].append(col)
        else:
            grouped_columns[col] = [col]

    rank_params = {}
    for key, cols in grouped_columns.items():
        unique_count = df_idx[cols[0]].nunique() if key == config["keyword"] else df_idx[key].nunique()
        max_value = min(unique_count, config.get(f"rank_{key}_max", unique_count))
        rank_params[key] = trial.suggest_int(f"rank_{key}", 1, max_value)

    rank_list = [rank_params[key] for key, cols in grouped_columns.items() for _ in cols]

    folds = stratified_k_fold(df_idx, 10, score_col_name=column_score)
    eval_scores = cross_validate_model(folds, rank_list, columns_wo_score, column_score)
    eval_metrics = config["eval_metrics"]
    mean_eval_score = eval_scores[eval_metrics]

    output_data = {
        "trial_number": trial.number,
        "eval_metric": eval_metrics,
        "mean_eval_score": mean_eval_score,
        **rank_params,
        **{f"mean_{metric}": scores for metric, scores in eval_scores.items()}
    }

    eval_top = config["eval_top"]
    csv_file = f'trial_results_{eval_metrics}_{eval_top}.csv'
    df_output = pd.DataFrame([output_data])
    if not os.path.isfile(csv_file):
        df_output.to_csv(csv_file, index=False)
    else:
        df_output.to_csv(csv_file, mode='a', header=False, index=False)

    return mean_eval_score


def generate_subplot_layout(n_folds: int, max_cols: int = 5):
    """
    Generates subplot layout based on the number of folds and a maximum number of columns.

    Args:
        n_folds (int): Number of folds.
        max_cols (int): Maximum columns to use.

    Returns:
        tuple: (n_rows, n_cols) for subplot layout.
    """
    n_rows = (n_folds + max_cols - 1) // max_cols
    return n_rows, min(n_folds, max_cols)


def plot_roc_curve_cv(df, n_folds, rank):
    """
    Plots ROC curves for cross-validation folds.

    Args:
        df (pd.DataFrame): The full dataframe.
        n_folds (int): Number of folds.
        rank (list): Ranks for the Tucker decomposition.
    """
    rank_str = '_'.join(map(str, rank))
    fprs, tprs, roc_aucs = [], [], []

    folds = stratified_k_fold(df, n_folds, shuffle=True, score_col_name='score')
    for fold_id, (df_train, df_test) in enumerate(folds):
        df_result, _, _ = perform_tucker_decomposition(
            df_train, rank, output_result_file=False, output_factor_file=False
        )
        roc_auc, _, _, _, _, _, _, true_y, pred_y = compute_evaluation_metrics(
            df_result,
            df_test,
            [col for col in df.columns if 'score' not in col],
            original_score_col="score",
            predicted_score_col="predicted_score"
        )

        fpr, tpr, _ = roc_curve(true_y, pred_y)
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(roc_auc)

    n_rows, n_cols = generate_subplot_layout(n_folds)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for i, (fpr, tpr, roc_auc) in enumerate(zip(fprs, tprs, roc_aucs)):
        ax = axes[i // n_cols, i % n_cols]
        ax.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
        ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
        ax.set_aspect('equal')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'Receiver Operating Characteristic Fold {i}')
        ax.legend(loc="lower right")

    fig.tight_layout()
    plt.savefig(f'roc_curve_cv_{n_folds}_rank_{rank_str}.png', dpi=300)
    plt.close()


def plot_pr_curve_cv(df, n_folds, rank):
    """
    Plots Precision-Recall curves for cross-validation folds.

    Args:
        df (pd.DataFrame): The full dataframe.
        n_folds (int): Number of folds.
        rank (list): Ranks for the Tucker decomposition.
    """
    rank_str = '_'.join(map(str, rank))
    precisions, recalls, avg_precisions = [], [], []

    folds = stratified_k_fold(df, n_folds, shuffle=True, score_col_name='score')
    for fold_id, (df_train, df_test) in enumerate(folds):
        df_result, _, _ = perform_tucker_decomposition(
            df_train, rank, output_result_file=False, output_factor_file=False
        )
        _, _, _, avg_precision, _, _, _, true_y, pred_y = compute_evaluation_metrics(
            df_result,
            df_test,
            [col for col in df.columns if 'score' not in col],
            original_score_col="score",
            predicted_score_col="predicted_score"
        )

        precision, recall, _ = precision_recall_curve(true_y, pred_y)
        precisions.append(precision)
        recalls.append(recall)
        avg_precisions.append(avg_precision)

    n_rows, n_cols = generate_subplot_layout(n_folds)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for i, (precision, recall, avg_precision) in enumerate(zip(precisions, recalls, avg_precisions)):
        ax = axes[i // n_cols, i % n_cols]
        ax.plot(recall, precision, color='blue', label=f'PR curve (AP = {avg_precision:.2f})')
        ax.fill_between(recall, precision, alpha=0.2, color='blue')
        ax.set_aspect('equal')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Fold {i}')
        ax.legend(loc="lower left")

    fig.tight_layout()
    plt.savefig(f'pr_curve_cv_{n_folds}_rank_{rank_str}.png', dpi=300)
    plt.close()


def plot_clustered_heatmap(filename, figsize=(3, 2), metric='cosine', method='complete', col_cluster=False):
    """
    Reads a file, computes distances or similarities, and plots a clustered heatmap.

    Args:
        filename (str): Path to the data file.
        figsize (tuple): Figure size. Default is (3, 2).
        metric (str): Distance metric for clustering. Default is 'cosine'.
        method (str): Linkage method. Default is 'complete'.
        col_cluster (bool): Whether to cluster columns. Default is False.
    """
    df = pd.read_csv(filename, sep=' ', header=None)
    df.set_index(0, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')

    if metric == 'cosine':
        dists = pdist(df.values, metric=metric)
    else:
        dists = pdist(df.values, metric=metric)

    linkage_matrix = hierarchy.linkage(dists, method=method)

    plt.figure(figsize=figsize)
    num_labels = len(df.index)
    font_scale = max(0.3, 1.0 - num_labels / 100.0)
    sns.set_theme(font_scale=font_scale)

    g = sns.clustermap(df, cmap='coolwarm',
                       row_linkage=linkage_matrix, col_cluster=col_cluster,
                       xticklabels=True, yticklabels=df.index)

    plt.savefig('clustered_heatmap_' + filename + '.png', dpi=300)
    plt.close()


def plot_mds_2d(
    filename,
    metric='cosine',
    font_size=10,
    label_offset=(0.02, 0.02),
    tick_size=8,
    axis_line_width=1.5,
    figsize=(10, 8),
    show_labels=True
):
    """
    Reads data from a file and uses Multidimensional Scaling (MDS) to project each row into 2D.

    Args:
        filename (str): Path to the data file.
        metric (str): Distance metric for pairwise distances. Defaults to 'cosine'.
        font_size (int): Font size for labels.
        label_offset (tuple): Offset for label placement.
        tick_size (int): Font size for axis ticks.
        axis_line_width (float): Line width for axes.
        figsize (tuple): Figure size.
        show_labels (bool): Whether to show text labels.
    """
    df = pd.read_csv(filename, sep='\s+', index_col=0)

    dist_matrix = pairwise_distances(df, metric=metric)
    print("Distance matrix stats: Min =", np.min(dist_matrix), "Max =", np.max(dist_matrix), "Mean =", np.mean(dist_matrix))

    mds = MDS(n_components=2, dissimilarity='precomputed', normalized_stress=False, random_state=42, n_init=10)
    mds_result = mds.fit_transform(dist_matrix)

    stress_value = mds.stress_

    plt.figure(figsize=figsize)
    ax = plt.gca()

    ax.set_facecolor('white')
    ax.spines['top'].set_linewidth(axis_line_width)
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_linewidth(axis_line_width)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(axis_line_width)
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_linewidth(axis_line_width)
    ax.spines['right'].set_color('black')

    plt.scatter(mds_result[:, 0], mds_result[:, 1], c='blue', s=50)

    if show_labels:
        texts = []
        for i, label in enumerate(df.index):
            text_x = mds_result[i, 0] + label_offset[0]
            text_y = mds_result[i, 1] + label_offset[1]
            texts.append(plt.text(text_x, text_y, label, fontsize=font_size))

        adjust_text(
            texts,
            only_move={'text': 'xy'},
            expand_text=(1.2, 3),
            expand_objects=(2, 3)
        )

        for i, label in enumerate(df.index):
            bbox = texts[i].get_window_extent(renderer=plt.gcf().canvas.get_renderer())
            bbox = bbox.transformed(ax.transData.inverted())

            if mds_result[i, 0] > bbox.x0 + (bbox.width / 2):
                label_x = bbox.x0 + bbox.width
            else:
                label_x = bbox.x0

            if mds_result[i, 1] > bbox.y0 + (bbox.height / 2):
                label_y = bbox.y0 + bbox.height
            else:
                label_y = bbox.y0

            plt.annotate(
                '',
                xy=(mds_result[i, 0], mds_result[i, 1]),
                xytext=(label_x, label_y),
                arrowprops=dict(arrowstyle='-', color='red')
            )

    plt.title(f'MDS 2D Plot (Stress: {stress_value:.4f})', fontsize=font_size + 2)
    plt.xlabel('Component 1', fontsize=font_size)
    plt.ylabel('Component 2', fontsize=font_size)

    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    plt.savefig('MDS_2D_' + filename + '.png', dpi=300)
    plt.close()


def knn_preservation(df, tsne_result, n_neighbors=5):
    """
    Computes k-Nearest Neighbors (kNN) preservation score between high-dimensional data (df) and low-dimensional embeddings (tsne_result).

    Args:
        df (pd.DataFrame): High-dimensional data.
        tsne_result (np.ndarray): Low-dimensional embedding result (2D or otherwise).
        n_neighbors (int): Number of neighbors to consider. Default is 5.

    Returns:
        float: kNN preservation score (ratio of neighbors preserved).
    """
    knn_high = NearestNeighbors(n_neighbors=n_neighbors).fit(df)
    high_neighbors = knn_high.kneighbors(df, return_distance=False)

    knn_low = NearestNeighbors(n_neighbors=n_neighbors).fit(tsne_result)
    low_neighbors = knn_low.kneighbors(tsne_result, return_distance=False)

    matching = 0
    for i in range(df.shape[0]):
        matching += len(set(high_neighbors[i]).intersection(low_neighbors[i]))

    return matching / (df.shape[0] * n_neighbors)


def plot_tsne_2d(
    filename,
    metric='euclidean',
    font_size=10,
    label_offset=(0.02, 0.02),
    tick_size=8,
    axis_line_width=1.5,
    figsize=(10, 8),
    show_labels=True,
    n_neighbors=5
):
    """
    Reads a file, computes pairwise distances, and applies t-SNE for 2D projection.
    Plots the resulting points and displays the kNN preservation score in the title.

    Args:
        filename (str): Path to the data file.
        metric (str): Distance metric for pairwise distances. Default is 'euclidean'.
        font_size (int): Font size for labels.
        label_offset (tuple): Offset for label placement.
        tick_size (int): Font size for axis ticks.
        axis_line_width (float): Line width for axes.
        figsize (tuple): Figure size.
        show_labels (bool): Whether to show text labels near points.
        n_neighbors (int): Number of neighbors for kNN preservation score.
    """
    df = pd.read_csv(filename, sep='\s+', index_col=0)
    dist_matrix = pairwise_distances(df, metric=metric)

    n_samples = df.shape[0]
    perplexity = min(30, n_samples - 1)

    tsne = TSNE(n_components=2, metric=metric, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(dist_matrix)

    knn_score = knn_preservation(df, tsne_result, n_neighbors=n_neighbors)

    plt.figure(figsize=figsize)
    ax = plt.gca()

    ax.set_facecolor('white')
    ax.spines['top'].set_linewidth(axis_line_width)
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_linewidth(axis_line_width)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(axis_line_width)
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_linewidth(axis_line_width)
    ax.spines['right'].set_color('black')

    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c='blue', s=50)

    if show_labels:
        texts = []
        for i, label in enumerate(df.index):
            text_x = tsne_result[i, 0] + label_offset[0]
            text_y = tsne_result[i, 1] + label_offset[1]
            texts.append(plt.text(text_x, text_y, label, fontsize=font_size))

        adjust_text(texts, only_move={'text': 'xy'}, expand_text=(1.2, 3), expand_objects=(2, 3))

    for i, label in enumerate(df.index):
        bbox = texts[i].get_window_extent(renderer=plt.gcf().canvas.get_renderer())
        bbox = bbox.transformed(ax.transData.inverted())

        if tsne_result[i, 0] > bbox.x0 + (bbox.width / 2):
            label_x = bbox.x0 + bbox.width
        else:
            label_x = bbox.x0

        if tsne_result[i, 1] > bbox.y0 + (bbox.height / 2):
            label_y = bbox.y0 + bbox.height
        else:
            label_y = bbox.y0

        plt.annotate(
            '',
            xy=(tsne_result[i, 0], tsne_result[i, 1]),
            xytext=(label_x, label_y),
            arrowprops=dict(arrowstyle='-', color='red')
        )

    plt.title(f't-SNE 2D Plot (kNN Preservation: {knn_score:.2f})', fontsize=font_size + 2)
    plt.xlabel('Component 1', fontsize=font_size)
    plt.ylabel('Component 2', fontsize=font_size)

    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    plt.savefig('tSNE_2D_' + filename + '.png', dpi=300)
    plt.close()


def plot_heatmap_and_dendrogram(
    filename,
    metric='cosine',
    method='complete',
    figsize_heatmap=(10, 8),
    figsize_dendrogram=(5, 10)
):
    """
    Reads data from a file and plots both a similarity/distance heatmap and a hierarchical clustering dendrogram.

    Args:
        filename (str): Path to the data file.
        metric (str): Metric for distance/similarity. Default is 'cosine'.
        method (str): Clustering method. Default is 'complete'.
        figsize_heatmap (tuple): Figure size for the heatmap. Default is (10, 8).
        figsize_dendrogram (tuple): Figure size for the dendrogram. Default is (5, 10).
    """
    df = pd.read_csv(filename, sep=' ', header=None, index_col=0)

    if metric == 'cosine':
        similarity_matrix = cosine_similarity(df)
        dist_mat = 1 - similarity_matrix
    else:
        dist_mat = pdist(df.values, metric=metric)
        similarity_matrix = None

    num_labels = len(df.index)
    font_scale = max(0.3, 1.0 - num_labels / 100.0)
    sns.set_theme(font_scale=font_scale)

    plt.figure(figsize=figsize_heatmap)
    if similarity_matrix is not None:
        sns.heatmap(similarity_matrix, annot=False, xticklabels=df.index, yticklabels=df.index, cmap='coolwarm')
        plt.title('Similarity Heatmap')
        plt.savefig('heatmap_' + filename + '.png', dpi=600)

    fontsize_dendro = max(3, 12 - num_labels // 10)
    linkage_matrix = linkage(dist_mat, method=method)
    plt.figure(figsize=figsize_dendrogram)
    dendrogram(linkage_matrix, labels=list(df.index), orientation='left')
    plt.setp(plt.gca().get_ymajorticklabels(), fontsize=fontsize_dendro)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.savefig('dendrogram_' + filename + '.png', dpi=300)
    plt.close()


def best_trial_callback(study, trial):
    """
    Callback for Optuna. Prints out new best value and parameters whenever a new best is found.

    Args:
        study (optuna.study.Study): The Optuna study.
        trial (optuna.trial.FrozenTrial): The current trial.
    """
    if study.best_trial.number == trial.number:
        print(f'New best value: {trial.value:.4f} achieved at trial {trial.number}.')
        print(f'Best parameters: {trial.params}')


def parse_data(input_str: str) -> Tuple[List[str], List[int]]:
    """
    Parses an input string, extracting item names and corresponding integer values.

    Args:
        input_str (str): The input string to parse.

    Returns:
        Tuple[List[str], List[int]]: A list of item names and a list of their associated integer values.
    """
    items = input_str.split('_')
    strings = []
    values = []
    for i in range(0, len(items), 2):
        if items[i] != 'O-2.00':
            strings.append(items[i])
            values.append(int(items[i + 1]))
    return strings, values


def gcd_of_list(numbers: List[int]) -> int:
    """
    Computes the greatest common divisor (GCD) for a list of integers.

    Args:
        numbers (List[int]): List of integers.

    Returns:
        int: The GCD of all integers in the list.
    """
    x = numbers[0]
    for element in numbers[1:]:
        x = math.gcd(x, element)
    return x


def process_data(data_list: List[str], score_function=None) -> List[List]:
    """
    Processes a list of data strings. Each string is parsed, values are normalized by GCD,
    and an optional score function is applied.

    Args:
        data_list (List[str]): List of data strings to process.
        score_function (callable, optional): A function taking (strings, values) and returning a score.

    Returns:
        List[List]: Each element is [score, "items, normalized_values"].
    """
    result = []
    for data in data_list:
        strings, values = parse_data(data)
        gcd_value = gcd_of_list(values)
        normalized_values = [v // gcd_value for v in values]

        if score_function:
            score = score_function(strings, values)
        else:
            score = 2  # Default score

        result.append([score, ', '.join(strings) + ', ' + ':'.join(map(str, normalized_values))])
    return result


def plot_score_histograms(df, core_tensor_rank, bin_width=0.05, font_size=12):
    """
    Plots separate histograms of predicted scores for each original_score in the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with columns 'original_score' and 'predicted_score'.
        core_tensor_rank (list): Rank of the core tensor (used for filename).
        bin_width (float): Bin width for histograms.
        font_size (int): Font size for plot elements.
    """
    plt.rcParams.update({'font.size': font_size})
    unique_scores = df['original_score'].unique()
    fig, axes = plt.subplots(len(unique_scores), 1, figsize=(10, len(unique_scores) * 5), sharex=True)

    for i, score in enumerate(unique_scores):
        ax = axes[i]
        score_data = df[df['original_score'] == score]

        color = 'gray'
        if score == 2:
            color = 'orange'
        elif score == 1:
            color = 'blue'

        sns.histplot(score_data['predicted_score'], binwidth=bin_width, ax=ax, color=color)
        ax.set_title(f'Original Score: {score}')
        ax.set_xlabel('Predicted Score')
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    core_tensor_rank_str = "_".join(map(str, core_tensor_rank))
    save_path = f"histgram_score_{core_tensor_rank_str}.png"
    plt.savefig(save_path)


def load_and_prepare_data(config):
    """
    Reads a CSV file and prepares a dataframe for Tucker decomposition.

    Args:
        config (dict): Configuration dictionary containing file paths and parameters.

    Returns:
        (pd.DataFrame, dict): Label-encoded dataframe and a dictionary of unique values.
    """
    _df = pd.read_csv(config['csv_file_path'], sep=None, engine='python')

    if _df.shape[1] == 1:
        # If there's only one column (raw data), process_data is applied
        df = process_data(config['csv_file_path'], config['csv_column_names'])
    else:
        df = _df[config['csv_column_names']].copy()

    unique_values_dict = get_unique_values(df, df.columns, keyword=config["keyword"])
    column_names_wo_score = [item for item in config['csv_column_names'] if 'score' not in item]
    column_name_score = [item for item in config['csv_column_names'] if 'score' in item][0]
    column_name_ratio = [item for item in config['csv_column_names'] if 'ratio' in item][0]

    dict_axis = {}
    for i in column_names_wo_score:
        if config["keyword"] in i:
            dict_axis[i] = unique_values_dict[config["keyword"]]
        else:
            dict_axis[i] = unique_values_dict[i]

    index_file_path = f"index_{config['csv_file_path']}"
    if not os.path.exists(index_file_path):
        print(f"{index_file_path} does not exist.")

        # Create duplicated rows with repeated materials
        df_only_dup = create_duplicated_combination(dict_axis, mode="all", score_col_name=column_name_score, temporary_score=1)

        df_base = pd.concat([df, df_only_dup]).reset_index(drop=True)
        df_base.to_csv(config['base_csv_output_path'], index=None, sep=" ")

        df_perm = permute_columns_in_dataframe(df_base, config["keyword"], column_name_ratio, column_name_score)
        df_perm_idx = label_encode_columns(df_perm, config["csv_file_path"], score_name=column_name_score)
    else:
        df_perm_idx = pd.read_csv(index_file_path)

    return df_perm_idx, dict_axis


def run_optuna_optimization(config, df_perm_idx):
    """
    Runs Optuna hyperparameter optimization. If an existing study has fewer trials than required,
    the old database is removed, and the study restarts.

    Args:
        config (dict): Configuration dictionary with parameters.
        df_perm_idx (pd.DataFrame): Label-encoded dataframe.

    Returns:
        (optuna.study.Study, list): The Optuna study and the best rank found.
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
        partial_objective = partial(objective, df_perm_idx, config)
        study.optimize(partial_objective, n_trials=remaining_trials, callbacks=[best_trial_callback])
    else:
        print(f"Study '{config['study_name']}' already completed the required {config['n_trials']} trials.")

    if len(study.trials) == 0:
        raise ValueError("No trials were completed in the study.")

    best_trial = study.best_trial

    # Construct best_rank from trial parameters
    best_rank = []
    for key in best_trial.params:
        if key.startswith("rank_"):
            best_rank.extend(
                [best_trial.params[key]]
                * len([col for col in df_perm_idx.columns if key.split('_')[1] in col])
            )

    return study, best_rank


def plot_results(config, df_perm_idx, df_result, core_tensor_rank):
    """
    Generates various plots based on the final results.

    Args:
        config (dict): Configuration dictionary.
        df_perm_idx (pd.DataFrame): Label-encoded dataframe.
        df_result (pd.DataFrame): Decomposition results.
        core_tensor_rank (list): The best or user-provided ranks for Tucker decomposition.
    """
    plot_pr_curve_cv(df_perm_idx, n_folds=5, rank=core_tensor_rank)
    plot_pr_curve_cv(df_perm_idx, n_folds=10, rank=core_tensor_rank)

    plot_roc_curve_cv(df_perm_idx, n_folds=5, rank=core_tensor_rank)
    plot_roc_curve_cv(df_perm_idx, n_folds=10, rank=core_tensor_rank)

    factor_axis_files = glob.glob('factor_axis_*')
    for file in factor_axis_files:
        plot_clustered_heatmap(file, (3, 2))
        if config['plot_ok']:
            plt.show()

        plot_heatmap_and_dendrogram(file, metric='cosine', method='complete', figsize_heatmap=(10, 8), figsize_dendrogram=(5, 10))
        if config['plot_ok']:
            plt.show()

        plot_mds_2d(file, metric="cosine")
        if config['plot_ok']:
            plt.show()

        plot_tsne_2d(file, metric="cosine")
        if config['plot_ok']:
            plt.show()

    plot_score_histograms(df_result, core_tensor_rank)


def main():
    """
    Main function that integrates all steps: data loading, optional hyperparameter optimization,
    Tucker decomposition, and result visualization.
    """
    argvs = sys.argv
    eval_metrics = str(argvs[1])
    if argvs[2] == "all":
        eval_top = None
    else:
        eval_top = int(argvs[2])
    if argvs[3] == "plot":
        plot_ok = True
    else:
        plot_ok = False

    config = {
        'keyword': "ion",
        'csv_file_path': "data_ps2",
        'csv_column_names': ["score", "ion1", "ion2", "ratio"],
        'rank_ion_max': 30,
        'base_csv_output_path': "df_base.csv",
        'optuna_db_path': f'tucker_{eval_metrics}_{eval_top}_optuna.db',
        'optuna_storage': f'sqlite:///tucker_{eval_metrics}_{eval_top}_optuna.db',
        'study_name': f'tucker_{eval_metrics}_{eval_top}',
        'n_trials': 100,
        'eval_metrics': eval_metrics,
        'eval_top': eval_top,
        'plot_ok': plot_ok,
        'core_tensor': [int(argvs[4]), int(argvs[4]), int(argvs[5])] if len(argvs) > 4 else []
    }

    df_perm_idx, ax_dict = load_and_prepare_data(config)

    if len(config['core_tensor']) > 0:
        core_tensor = config['core_tensor']
        print(f"core tensor : {core_tensor}")
        columns_wo_score = [col for col in df_perm_idx.columns if 'score' not in col]
        column_score = [col for col in df_perm_idx.columns if 'score' in col][0]
        folds = stratified_k_fold(df_perm_idx, 10, score_col_name=column_score)
        eval_scores = cross_validate_model(folds, core_tensor, columns_wo_score, column_score)
        df_evals = pd.DataFrame([eval_scores])
        print(df_evals)
        df_result, factors, core = perform_tucker_decomposition(df_perm_idx, core_tensor, score_col_idx=0, output_result_file=True, output_factor_file=True)
        plot_results(config, df_perm_idx, df_result, core_tensor)

    else:
        study, best_rank = run_optuna_optimization(config, df_perm_idx)
        df_result, factors, core = perform_tucker_decomposition(df_perm_idx, best_rank, score_col_idx=0, output_result_file=True, output_factor_file=True)
        plot_results(config, df_perm_idx, df_result, best_rank)

        print(f"Best trial number: {study.best_trial.number}")
        print(f"Best value {config['eval_metrics']}: {study.best_trial.value}")
        print(f"Best trial parameters: {study.best_trial.params}")


if __name__ == '__main__':
    main()
