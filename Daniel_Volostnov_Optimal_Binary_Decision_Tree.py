import numpy as np
import matplotlib.pyplot as plt

class Binary_Decision_Tree:
    def __init__(self, L, N, X=None):
        self.L = L
        self.N = N
        if X is not None:
            self.X = np.array(X)
        else:
            self.X = np.zeros((N, L))

    def optimal_y(self, Y):
        return np.mean(Y)

    def MSE(self, Y):
        return np.mean((Y - self.optimal_y(Y)) ** 2)

    def calculate_loss(self, t, j, X, y):
        left_indices = X[:, j] <= t
        right_indices = ~left_indices

        if sum(left_indices) == 0 or sum(right_indices) == 0:
            return float('inf')

        left_loss = self.MSE(y[left_indices])
        right_loss = self.MSE(y[right_indices])

        total_loss = (
                (sum(left_indices) / len(y)) * left_loss +
                (sum(right_indices) / len(y)) * right_loss
        )
        return total_loss

    def fit(self, X, y):
        min_loss = float('inf')
        optimal_border = None
        total_samples = len(y)

        for j in range(self.L):
            sorted_indices = np.argsort(X[:, j])
            sorted_X = X[sorted_indices, j]
            sorted_y = y[sorted_indices]

            left_sum = 0
            left_count = 0
            right_sum = np.sum(sorted_y)
            right_count = total_samples

            for i in range(1, total_samples):
                left_sum += sorted_y[i - 1]
                left_count += 1
                right_sum -= sorted_y[i - 1]
                right_count -= 1

                if sorted_X[i] == sorted_X[i - 1]:
                    continue

                left_mean = left_sum / left_count
                right_mean = right_sum / right_count
                left_mse = np.mean((sorted_y[:i] - left_mean) ** 2)
                right_mse = np.mean((sorted_y[i:] - right_mean) ** 2)

                loss = (left_count / total_samples) * left_mse + (right_count / total_samples) * right_mse

                if loss < min_loss:
                    min_loss = loss
                    optimal_border = (j, (sorted_X[i] + sorted_X[i - 1]) / 2)

        self.optimal_border = optimal_border
        self.min_loss = min_loss

    def plot_decision_boundary(self, X, y):
        if not hasattr(self, 'optimal_border') or self.optimal_border is None:
            print("Fit the model first to find the optimal border.")
            return

        j, t = self.optimal_border
        left_indices = X[:, j] <= t
        right_indices = ~left_indices

        plt.figure(figsize=(8, 6))
        plt.scatter(X[left_indices, 0], X[left_indices, 1], color='blue', label='Left split', alpha=0.7)
        plt.scatter(X[right_indices, 0], X[right_indices, 1], color='red', label='Right split', alpha=0.7)

        if j == 0:
            plt.axvline(x=t, color='green', linestyle='--', label=f'Split at {t} (Feature {j})')
        else:
            plt.axhline(y=t, color='green', linestyle='--', label=f'Split at {t} (Feature {j})')

        plt.xlabel('Feature 0')
        plt.ylabel('Feature 1')
        plt.title('Decision Boundary Visualization')
        plt.legend()
        plt.show()


# Test the Binary_Decision_Tree
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

tree = Binary_Decision_Tree(L=2, N=15)
tree.fit(X, y)
tree.plot_decision_boundary(X, y)
