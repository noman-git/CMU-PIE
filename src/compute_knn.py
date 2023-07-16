import time
import numpy as np
import pandas as pd
from .knn import KNNClassifier

def run_single_split(knn_model, X_train, y_train, X_test, y_test):
    """Run the KNN model for a single split of the data."""
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    return accuracy

def run_all_splits(knn_model, X_train_splits, y_train_splits, X_test_splits, y_test_splits):
    """Run the KNN model for all splits of the data."""
    accuracy_list = []
    for X_train, y_train, X_test, y_test in zip(X_train_splits, y_train_splits, X_test_splits, y_test_splits):
        accuracy = run_single_split(knn_model, X_train, y_train, X_test, y_test)
        accuracy_list.append(accuracy)
    return accuracy_list

def run_knn_model(X_train_all, y_train_all, X_test_all, y_test_all, labels_list, train_test_pairs, current_k, distance, column_list):
    """Run the KNN model for a given k, distance measure, labels, training examples and test cases"""
    knn_model = KNNClassifier(k=current_k, distance=distance)
    print(f'Results using k = {current_k} and distance measure of {distance}:')
    count = 0
    results_df = pd.DataFrame(columns=["k_value", "distance_algo", "number_of_labels", "training_test_pair", "average_accuracy", "std_accuracy", "computation_time"])
    for X_train_splits, y_train_splits, X_test_splits, y_test_splits in zip(X_train_all, y_train_all, X_test_all, y_test_all):
        if count % 2 == 0:
            current_label = labels_list[count//2]
        current_training_examples, current_testing_values = train_test_pairs[count%2]
        count += 1
        start_time = time.time()
        
        accuracy_list = run_all_splits(knn_model, X_train_splits, y_train_splits, X_test_splits, y_test_splits)
            
        end_time = time.time()
        computation_time = end_time - start_time
        print(computation_time)
        average_accuracy = np.mean(accuracy_list)
        std_accuracy = np.std(accuracy_list)
        print(f'Using {current_label} labels, {current_training_examples} training examples and {current_testing_values} test cases PER SUBJECT over 5 random splits:')
        print(f'Average accuracy: {average_accuracy:.3f}                           Standard deviation over accuracy: {std_accuracy:.3f}')
        print('------------------------------------------------------------------------------------------')
        
        results_to_append = pd.DataFrame([[current_k, distance, current_label, f"{current_training_examples}_{current_testing_values}", average_accuracy, std_accuracy, computation_time]], columns=column_list) 
        results_df = pd.concat([results_df, results_to_append], ignore_index=True) 
    print('==========================================================================================')
    return results_df