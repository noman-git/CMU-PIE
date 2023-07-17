import numpy as np
import sys

# go to the parent directory
sys.path.insert(0, '.')

from src.knn import KNNClassifier

def test_fit_method() -> None:
    # Create a simple dataset
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([0, 1, 0])

    # Initialize a KNNClassifier
    knn = KNNClassifier(k=2, distance='euclidean')

    # Test the fit method
    knn.fit(X_train, y_train)
    assert np.array_equal(knn.X_train, X_train)
    assert np.array_equal(knn.y_train, y_train)

def test_predict_method() -> None:
    # Create a simple dataset
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([0, 1, 0])
    X_test = np.array([[2, 3], [4, 5]])

    # Initialize a KNNClassifier
    knn = KNNClassifier(k=2, distance='euclidean')
    knn.fit(X_train, y_train)

    # Test the predict method
    y_pred = knn.predict(X_test)
    assert np.array_equal(y_pred, np.array([0, 0]))

def test_euclidean_distance() -> None:
    # Initialize a KNNClassifier
    knn = KNNClassifier(k=2, distance='euclidean')

    # Test the _euclidean_distance method
    X_train = np.array([[1, 2]])
    X_test = np.array([[2, 3]])
    assert np.allclose(knn._euclidean_distance(X_train, X_test), np.array([[1.41421356]]), atol=1e-5)

def test_manhattan_distance() -> None:
    # Initialize a KNNClassifier
    knn = KNNClassifier(k=2, distance='euclidean')

    # Test the _manhattan_distance method
    X_train = np.array([[1, 2]])
    X_test = np.array([[2, 3]])
    assert np.array_equal(knn._manhattan_distance(X_train, X_test), np.array([[2]]))

def test_cosine_distance() -> None:
    # Initialize a KNNClassifier
    knn = KNNClassifier(k=2, distance='euclidean')

    # Test the _cosine_distance method
    X_train = np.array([[1, 0]])
    X_test = np.array([[0, 1]])
    assert np.array_equal(knn._cosine_distance(X_train, X_test), np.array([[1]]))