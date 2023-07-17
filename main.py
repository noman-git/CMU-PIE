import pandas as pd
from src.data_processing import normalize_data, prepare_data
from src.compute_knn import run_knn_model
from src.result_viz import create_pivot_table, plot_pivot_tables


data = pd.read_csv('data/fea.csv', header=None)
data = normalize_data(data)

# defining different parameters to test
labels_list = [10,7,5]
train_test_pairs = [(150, 20), (100, 70)]
k_values = [3,5,7,9,11]
distance_algos = ['euclidean', 'manhattan', 'cosine']

# creating a dataframe to store the results
results_df = pd.DataFrame(columns=["k_value", "distance_algo", "number_of_labels", "training_test_pair",
                                   "average_accuracy", "std_accuracy", "computation_time"])

# Set to True if you want to use PCA
pca = True
# Set to True if you want to visualize the covariance matrices
pca_visualize = True

if pca:
    from src.pca_data import calc_n_components, create_pca_df # importing only if needed 
    
    if pca_visualize:
        from src.pca_data import analyze_correlations, plot_cov_matrix # importing only if needed      
        cov_matrix = analyze_correlations(data)
        plot_cov_matrix(cov_matrix, title='Covariance Matrix Heatmap before PCA')
        
        n_components_99_percent = calc_n_components(data)
        data = create_pca_df(data, n_components_99_percent)
        
        # a bit of redundancy here, but creating a new function just for this makes things more complicated
        cov_matrix = analyze_correlations(data) 
        plot_cov_matrix(cov_matrix, title='Covariance Matrix Heatmap after PCA')
    else:
        n_components_99_percent = calc_n_components(data)
        data = create_pca_df(data, n_components_99_percent)

# prepare the data
X_train_all, y_train_all, X_test_all, y_test_all = prepare_data(data, labels_list, train_test_pairs)

# run the knn function for all the different parameters
for current_k in k_values:
    for distance in distance_algos:
        results_to_append = run_knn_model(X_train_all, y_train_all, X_test_all, y_test_all, labels_list, train_test_pairs, current_k, distance, results_df.columns)
        results_df = pd.concat([results_df, results_to_append], ignore_index=True)
    print('==============================================================================================================')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('==============================================================================================================')

#columns to create pivot tables
group_by_columns = results_df.columns[:4]
values_columns = results_df.columns[4:]

#create pivot tables and plot them
pivot_tables = create_pivot_table(results_df, group_by_columns, values_columns)
plot_pivot_tables(pivot_tables, group_by_columns, values_columns)