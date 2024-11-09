import numpy as np

class Node:
    def __init__(self, prediction=None):
        self.t = None
        self.feature_index = None
        self.left = None
        self.right = None
        self.prediction = prediction

class Binary_Decision_Tree:
    def __init__(self, L, N, X=None):
        self.L = L
        self.N = N
        if X is not None:
            self.X = np.array(X)
        else:
            self.X = np.zeros((N, L))
        self.tree = None

    def optimal_y(self, Y):
        return np.mean(Y)

    def MSE(self, Y):
        return np.mean((Y - self.optimal_y(Y)) ** 2)

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.N, self.L = X.shape

        self.sorted_indices = {}
        for j in range(self.L):
            self.sorted_indices[j] = np.argsort(self.X[:, j])

        full_range = {j: (0, self.N) for j in range(self.L)}
        indices = np.arange(self.N)
        self.tree = self.build_tree(indices, full_range)

    def build_tree(self, indices, sorted_ranges):
        Y = self.y[indices]
        node = Node(prediction=self.optimal_y(Y))

        if len(indices) <= 1:
            return node

        min_loss = float('inf')
        optimal_t = None
        optimal_j = None
        best_left_indices = None
        best_right_indices = None
        best_left_ranges = None
        best_right_ranges = None

        total_samples = len(indices)

        for j in range(self.L):
            sorted_idx = self.sorted_indices[j]
            start, end = sorted_ranges[j]
            sorted_idx = sorted_idx[start:end]
            X_j_sorted = self.X[sorted_idx, j]
            y_sorted = self.y[sorted_idx]

            left_count = 0
            left_sum = 0.0
            right_count = len(y_sorted)
            right_sum = np.sum(y_sorted)

            for i in range(1, len(y_sorted)):
                xi_prev, yi_prev = X_j_sorted[i - 1], y_sorted[i - 1]
                xi, yi = X_j_sorted[i], y_sorted[i]
                left_count += 1
                left_sum += yi_prev
                right_count -= 1
                right_sum -= yi_prev

                if xi == xi_prev:
                    continue

                left_mean = left_sum / left_count
                right_mean = right_sum / right_count

                left_mse = np.sum((y_sorted[:i] - left_mean) ** 2)
                right_mse = np.sum((y_sorted[i:] - right_mean) ** 2)

                loss = (left_mse + right_mse) / total_samples

                if loss < min_loss:
                    min_loss = loss
                    optimal_j = j
                    optimal_t = (xi + xi_prev) / 2
                    best_left_indices = sorted_idx[:i]
                    best_right_indices = sorted_idx[i:]
                    best_left_ranges = sorted_ranges.copy()
                    best_right_ranges = sorted_ranges.copy()
                    best_left_ranges[j] = (start, start + i)
                    best_right_ranges[j] = (start + i, end)

        if optimal_j is not None:
            node.feature_index = optimal_j
            node.t = optimal_t
            node.left = self.build_tree(best_left_indices, best_left_ranges)
            node.right = self.build_tree(best_right_indices, best_right_ranges)
        else:
            node.prediction = self.optimal_y(Y)

        return node

    def predict_single(self, x, node):
        if node.left is None and node.right is None:
            return node.prediction

        j = node.feature_index
        t = node.t

        if x[j] <= t:
            return self.predict_single(x, node.left)
        else:
            return self.predict_single(x, node.right)

    def predict(self, X):
        return np.array([self.predict_single(x, self.tree) for x in X])

# Example
X = np.array([[2, 3],
              [10, 15],
              [7, 8],
              [5, 7],
              [3, 5],
              [8, 9],
              [6, 6],
              [1, 4],
              [9, 14],
              [4, 5],
              [5, 8],
              [11, 16],
              [2, 9],
              [6, 12],
              [8, 3]])

y = np.array([3, 10, 7, 5, 4, 8, 6, 3, 9, 5, 6, 11, 5, 8, 7])

permutation = np.random.permutation(len(X))
X = X[permutation]
y = y[permutation]

tree = Binary_Decision_Tree(L=2, N=15)
tree.fit(X, y)

predictions = tree.predict(X)
print("Prediction:", predictions)
print("Correct values:", y)
