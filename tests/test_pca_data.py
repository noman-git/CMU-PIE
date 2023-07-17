import pandas as pd
import numpy as np
import sys

# go to the parent directory
sys.path.insert(0, '.')

from src.pca_data import analyze_correlations, calc_n_components, create_pca_df

def test_analyze_correlations() -> None:
    data = np.random.rand(100, 10)
    cov_matrix = analyze_correlations(data)
    assert isinstance(cov_matrix, np.ndarray)
    assert cov_matrix.shape == (data.shape[1], data.shape[1])

def test_calc_n_components() -> None:
    data = np.random.rand(100, 10)
    n_components = calc_n_components(data)
    assert isinstance(n_components, np.int64)
    assert 0 < n_components <= data.shape[1]

def test_create_pca_df() -> None:
    data = np.random.rand(100, 10)
    n_components = 5
    pca_df = create_pca_df(data, n_components)
    assert isinstance(pca_df, pd.DataFrame)
    assert pca_df.shape == (data.shape[0], n_components)