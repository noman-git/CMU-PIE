from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_correlations(data):
    """Analyze the correlations between the features."""
    cov_matrix = np.cov(data, rowvar=False)

    # Get the off-diagonal elements
    off_diagonal = cov_matrix[~np.eye(cov_matrix.shape[0], dtype=bool)]

    # Check for correlations
    uncorrelated = np.isclose(off_diagonal, 0, atol=0.01).sum()
    positively_correlated = (off_diagonal > 0.01).sum()
    negatively_correlated = (off_diagonal < -0.01).sum()

    print(f"There are {uncorrelated} pairs of variables that are uncorrelated.")
    print(f"There are {positively_correlated} pairs of variables that are positively correlated.")
    print(f"There are {negatively_correlated} pairs of variables that are negatively correlated.")
    print("\n")
    return cov_matrix

def plot_cov_matrix(cov_matrix, title='Covariance Matrix Heatmap'):
    """Plot the covariance matrix heatmap."""
    plt.figure(figsize=(10,10))
    sns.heatmap(cov_matrix, cmap='coolwarm', center=0)
    plt.title(title)
    plt.show()
    
def calc_n_components(data, variance=0.99):
    """Calculate the number of components that explain a given variance."""
    pca = PCA(n_components=variance)
    pca.fit(data)
    print(f"{pca.n_components_} components explain {variance} of the variance.")
    print("\n")
    return pca.n_components_

def create_pca_df(data, n_components):
    """Create a dataframe with the PCA components."""
    pca = PCA(n_components=n_components)
    
    # Apply PCA to data
    data_array = pca.fit_transform(data)
    return pd.DataFrame(data_array)