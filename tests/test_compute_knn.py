import pytest
import numpy as np
import pandas as pd
import sys


sys.path.insert(0, '.')

from src.compute_knn import run_single_split, run_all_splits, run_knn_model
from src.knn import KNNClassifier

def test_run_single_split():
    # Create a simple dataset
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])
    X_test = np.array([[2, 3]])

    # Initialize a KNNClassifier
    knn = KNNClassifier(k=1, distance='euclidean')

    # Test the run_single_split function
    accuracy = run_single_split(knn, X_train, y_train, X_test, np.array([0]))
    assert accuracy == 1.0

def test_run_all_splits():
    # Create a simple dataset
    X_train = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    y_train = np.array([[0, 1], [1, 0]])
    X_test = np.array([[[2, 3]], [[6, 7]]])
    y_test = np.array([[0], [1]])

    # Initialize a KNNClassifier
    knn = KNNClassifier(k=1, distance='euclidean')

    # Test the run_all_splits function
    accuracy_list = run_all_splits(knn, X_train, y_train, X_test, y_test)
    assert np.array_equal(accuracy_list, np.array([1.0, 1.0]))

def test_run_knn_model():
    # Create a simple dataset
    X_train_all = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])
    y_train_all = np.array([[[0, 1], [1, 0]]])
    X_test_all = np.array([[[[2, 3]], [[6, 7]]]])
    y_test_all = np.array([[[0], [1]]])
    labels_list = ['label1']
    train_test_pairs = [(2, 1)]
    current_k = 1
    distance = 'euclidean'
    column_list = ['k_value', 'distance_algo', 'number_of_labels', 'training_test_pair', 'average_accuracy', 'std_accuracy', 'computation_time']

    # Test the run_knn_model function
    results = run_knn_model(X_train_all, y_train_all, X_test_all, y_test_all, labels_list, train_test_pairs, current_k, distance, column_list)
    assert isinstance(results, pd.DataFrame)
    assert list(results.columns) == column_list
    assert len(results) == 1
    assert results['k_value'].iloc[0] == current_k
    assert results['distance_algo'].iloc[0] == distance
    assert results['number_of_labels'].iloc[0] == labels_list[0]
    assert results['training_test_pair'].iloc[0] == f"{train_test_pairs[0][0]}_{train_test_pairs[0][1]}"
    assert isinstance(results['average_accuracy'].iloc[0], float)
    assert isinstance(results['std_accuracy'].iloc[0], float)
    assert isinstance(results['computation_time'].iloc[0], float)
