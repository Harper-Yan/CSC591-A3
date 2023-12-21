import numpy as np
import matplotlib.pyplot as plt
from time import time
import cupy as cp

def initialize_centroids(data, k):
    indices = np.random.choice(len(data), k, replace=False)
    centroids = data[indices]
    return centroids

compute_distances_kernel = cp.RawKernel(r'''
extern "C" 
__global__ void compute_distances(float* data,float* centroids_gpu, float* distances)
{
                                        
  extern __shared__ float shared_centroids[2];
                                        
  int i = blockDim.x * threadIdx.y + threadIdx.x;
                                        
  int j = blockIdx.x;

  if(i==0&&j<50)
  {                                 
    shared_centroids[0] = centroids_gpu[j * 2];
    shared_centroids[1] = centroids_gpu[j * 2 + 1];
  }  
                                        
  __syncthreads();            
                                                          
  if(i < 1000 && j < 50){
    float d=0;
    d+=(data[i*2]-shared_centroids[0])*(data[i*2]-shared_centroids[0]);
    d+=(data[i*2+1]-shared_centroids[1])*(data[i*2+1]-shared_centroids[1]);    
    d=sqrt(d);                     
    distances[i*50+j]=d;                                      
  }

                                       
}
''', 'compute_distances')


def assign_to_clusters(data, centroids, block_dim=(32, 32, 1)):
    num_data_points = data.shape[0]
    num_centroids = centroids.shape[0]

    # Transfer data to GPU memory
    data_gpu = cp.asarray(data)
    centroids_gpu = cp.asarray(centroids)

    # Allocate GPU memory for distances
    distances_gpu = cp.zeros((num_data_points, num_centroids)).astype(np.float32)

    # Launch the CUDA kernels
    compute_distances_kernel((num_data_points, 1, 1), block_dim, (data_gpu, centroids_gpu,distances_gpu))

    distances = cp.asnumpy(distances_gpu)

    # Find the index of the minimum distance on the GPU
    labels_gpu = cp.argmin(distances_gpu, axis=1)

    # Transfer labels back to CPU memory
    labels = cp.asnumpy(labels_gpu)

    return labels



def update_centroids(data, labels, k):
    centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
    return centroids

def k_means_clustering(data, k, max_iterations=100, log_file=None):

    with open(log_file, 'w') as f:
        f.write("K-Means Execution Log\n")

    centroids = initialize_centroids(data, k)
    
    for iteration in range(max_iterations):
        t_iteration_start = time()
        
        labels = assign_to_clusters(data, centroids)
        new_centroids = update_centroids(data, labels, k)
        
        t_iteration_end = time()
        iteration_time = t_iteration_end - t_iteration_start

        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"Iteration {iteration + 1}: {iteration_time} seconds\n")

        if np.linalg.norm(new_centroids - centroids)<0.1:
            break
        
        centroids = new_centroids

    return labels, centroids

# Generate 500 random 2-dimensional points
np.random.seed(42)
data = np.random.randn(1000, 2).astype(np.float32)

# Specify the log file name
log_file_name = 'kmeans_cuda_execution_shared_log.txt'

# Apply k-means clustering with k=10
k = 50
labels, centroids = k_means_clustering(data, k, log_file=log_file_name)

plt.figure()
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='k')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100, label='Centroids')
plt.title('K-Means Clustering Using Shared Memory')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.savefig("K-Means Clustering Shared Cuda.png")
plt.show()

