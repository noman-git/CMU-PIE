import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Constants
examples_per_label = 170
num_splits = 5

def create_df(dataframe, num_labels):
    """Create a dataframe with a specified number of labels."""
    labels = np.repeat(np.arange(num_labels), examples_per_label)
    dataframe = dataframe.iloc[:num_labels*examples_per_label].copy()
    dataframe['label'] = labels
    return dataframe

def normalize_data(raw_dataframe):
    """Normalize the data in the dataframe."""
    normalized_data_array = raw_dataframe.values / np.linalg.norm(raw_dataframe.values, axis=1, keepdims=True)
    return pd.DataFrame(normalized_data_array, columns=raw_dataframe.columns)

def create_data_splits(normalized_data_dataframe, num_labels, train_rows, test_rows, num_splits):
    """Create data splits for training and testing."""
    train_dfs_list = []
    test_dfs_list = []
    for i in range(num_splits):
        train_data = []
        test_data = []
        for label in range(num_labels):
            label_data = normalized_data_dataframe[normalized_data_dataframe['label'] == label].copy()
            train_label_data, test_label_data = train_test_split(label_data,
                                                                 train_size=train_rows,
                                                                 test_size=test_rows,
                                                                 random_state=i)
            train_data.append(train_label_data)
            test_data.append(test_label_data)
        train_dfs_list.append(pd.concat(train_data).sample(frac=1.0, random_state=i).reset_index(drop=True))
        test_dfs_list.append(pd.concat(test_data).sample(frac=1.0, random_state=i).reset_index(drop=True))
    return train_dfs_list, test_dfs_list

def pre_process_data(data, num_labels, train_rows, test_rows):
    """Run the pre-processing steps."""
    truncated_df = create_df(data, num_labels) #create the df
    return create_data_splits(truncated_df, num_labels, train_rows, test_rows, num_splits) #create the data splits

def extract_features_and_labels(dfs_list):
    """Create X and y from a list of dataframes."""
    X = [df.drop('label', axis=1).values for df in dfs_list]
    y = [df['label'].values for df in dfs_list]
    return X, y

def prepare_data(data, labels_list, train_test_pairs):
    """Prepare the data for training and testing using all the func"""
    train_test_dfs = [pre_process_data(data, num_labels, train_rows, test_rows)
                      for num_labels in labels_list
                      for train_rows, test_rows in train_test_pairs]
    train_dfs_all, test_dfs_all = zip(*train_test_dfs)
    X_train_all, y_train_all = zip(*[extract_features_and_labels(train_dfs_list)
                                     for train_dfs_list in train_dfs_all])
    X_test_all, y_test_all = zip(*[extract_features_and_labels(test_dfs_list)
                                   for test_dfs_list in test_dfs_all])
    return X_train_all, y_train_all, X_test_all, y_test_all