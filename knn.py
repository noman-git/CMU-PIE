import numpy as np


class KNNClassifier:
    """Class implementing the k-nearest neighbors vote."""
    def __init__(self, k=3, distance='euclidean'):
        self.k = k
        self.distance = self._get_distance_method(distance)

    def _get_distance_method(self, distance):
        """Choose the distance function based on the distance parameter."""
        match distance:
            case 'euclidean':
                return self._euclidean_distance
            case 'manhattan':
                return self._manhattan_distance
            case 'cosine':
                return self._cosine_distance

    def fit(self, X, y):
        """Store the training data. KNNClassifier does not do any training."""
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Compute the k-nearest neighbors vote and return the most common class label."""
        distances = self.distance(self.X_train, X)
        k_indices = np.argsort(distances, axis=0)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        most_common = np.argmax(np.apply_along_axis(np.bincount, 0, k_nearest_labels, minlength=self.y_train.max() + 1), axis=0)
        return most_common

    def _euclidean_distance(self, a, b):
        """Calculate the Euclidean distance between two matrices."""
        return np.sqrt(np.sum((a[:, np.newaxis] - b) ** 2, axis=2))

    def _manhattan_distance(self, a, b):
        """Calculate the Manhattan distance between two matrices."""
        return np.sum(np.abs(a[:, np.newaxis] - b), axis=2)

    def _cosine_distance(self, a, b):
        """Calculate the cosine distance between two matrices."""
        dot_product = np.einsum('ij,kj->ik', a, b)
        norm_a = np.linalg.norm(a, axis=1)[:, np.newaxis]
        norm_b = np.linalg.norm(b, axis=1)
        cosine_similarity = dot_product / (norm_a * norm_b)
        return 1 - cosine_similarity
