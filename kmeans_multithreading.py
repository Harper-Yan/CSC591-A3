import numpy as np
import matplotlib.pyplot as plt
from time import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

def initialize_centroids(data, k):
    # Randomly select k data points as initial centroids
    indices = np.random.choice(len(data), k, replace=False)
    centroids = data[indices]
    return centroids


def calculate_distances(data, centroids, i):

    result = np.linalg.norm(data[i] - centroids, axis=1)

    return result

def assign_to_clusters(data, centroids):

    num_data_points = data.shape[0]
    num_centroids = centroids.shape[0]

    distances = np.zeros((num_data_points, num_centroids))

    if __name__ == '__main__':
        with ThreadPoolExecutor() as executor:
            # Use functools.partial to fix the first two arguments of calculate_distances
            partial_calc_distances = partial(calculate_distances, data, centroids)
            
            # Use executor.map with the partial function
            results = list(executor.map(partial_calc_distances, range(num_data_points)))

            for i, result in enumerate(results):
                distances[i, :] = result

    for i, result in enumerate(results):
        distances[i, :] = result

    labels = np.argmin(distances, axis=1)

    return labels

def update_centroids(data, labels, k):
    # Update centroids based on the mean of data points in each cluster
    centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
    return centroids

def k_means_clustering(data, k, max_iterations=100,log_file=None):

    with open(log_file, 'w') as f:
        f.write("K-Means Execution Log\n")

    centroids = initialize_centroids(data, k)
    
    for _ in range(max_iterations):

        t_iteration_start = time()
        labels = assign_to_clusters(data, centroids)
        
        # Update centroids
        new_centroids = update_centroids(data, labels, k)

        t_iteration_end = time()
        iteration_time = t_iteration_end - t_iteration_start

        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"Iteration {_ + 1}: {iteration_time} seconds\n")

        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return labels, centroids

# Generate 500 random 2-dimensional points
np.random.seed(42)
data = np.random.randn(1000, 2)

# Specify the log file name
log_file_name = 'kmeans_multithreading_execution_log.txt'

# Apply k-means clustering with k=10
k = 50
labels, centroids = k_means_clustering(data, k, log_file=log_file_name)

# Plot the original data and the cluster centers
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='k')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100, label='Centroids')
plt.title('K-Means Clustering Multitheading')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.savefig("K-Means Clustering Multithreading.png")

plt.show()
