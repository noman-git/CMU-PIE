import pytest
import pandas as pd
import numpy as np
import sys


sys.path.insert(0, '.')

from src.data_processing import create_df, normalize_data, create_data_splits, pre_process_data, extract_features_and_labels, prepare_data

def test_create_df():
    df = pd.DataFrame(np.random.rand(1700, 10))
    num_labels = 10
    truncated_df, labels = create_df(df, num_labels)
    assert isinstance(truncated_df, pd.DataFrame)
    assert isinstance(labels, np.ndarray)
    assert truncated_df.shape == (num_labels*170, 10)
    assert labels.shape == (num_labels*170,)

def test_normalize_data():
    df = pd.DataFrame(np.random.rand(1700, 10))
    normalized_df = normalize_data(df)
    assert isinstance(normalized_df, pd.DataFrame)
    assert normalized_df.shape == df.shape
    assert np.allclose(normalized_df.apply(np.linalg.norm, axis=1), 1)

def test_create_data_splits():
    df = pd.DataFrame(np.random.rand(1700, 11))
    df.columns = list(range(10)) + ['label']
    df['label'] = np.repeat(np.arange(10), 170)
    num_labels = 10
    train_rows = 100
    test_rows = 70
    num_splits = 5
    train_dfs_list, test_dfs_list = create_data_splits(df, num_labels, train_rows, test_rows, num_splits)
    assert isinstance(train_dfs_list, list)
    assert isinstance(test_dfs_list, list)
    assert len(train_dfs_list) == num_splits
    assert len(test_dfs_list) == num_splits

def test_pre_process_data():
    df = pd.DataFrame(np.random.rand(1700, 10))
    num_labels = 10
    train_rows = 100
    test_rows = 70
    train_dfs_list, test_dfs_list = pre_process_data(df, num_labels, train_rows, test_rows)
    assert isinstance(train_dfs_list, list)
    assert isinstance(test_dfs_list, list)

def test_extract_features_and_labels():
    df = pd.DataFrame(np.random.rand(1700, 11))
    df.columns = list(range(10)) + ['label']
    df['label'] = np.repeat(np.arange(10), 170)
    dfs_list = [df]*5
    X, y = extract_features_and_labels(dfs_list)
    assert isinstance(X, list)
    assert isinstance(y, list)
    assert len(X) == len(dfs_list)
    assert len(y) == len(dfs_list)

def test_prepare_data():
    df = pd.DataFrame(np.random.rand(1700, 10))
    labels_list = [10, 7, 5]
    train_test_pairs = [(150, 20), (100, 70)]
    X_train_all, y_train_all, X_test_all, y_test_all = prepare_data(df, labels_list, train_test_pairs)
    assert isinstance(X_train_all, tuple)
    assert isinstance(y_train_all, tuple)
    assert isinstance(X_test_all, tuple)
    assert isinstance(y_test_all, tuple)
