# CSC591-A3
CUDA-Parallel K-means clustering

This repo stores the py files, visualized results and excution logs that sotres the running time for k-means clustering algorithm implemented by four different methods:

To summerize, all four programs works correctly, generating similar results under the same set of parameters. As for performance, GPU parallel programs outperform the CPU multithreading baseline by roughly 10* times, however, the explicit using of shared memory didn't bring significant performance boost.

1. CPU sequential method.
   
   Files:   kmeans_sequential.py;   K-Means Clustering Sequential.png;   kmeans_sequential_execution_log.txt

   This method is outperformed by all other methods by at least 10* times.

2. GPU multithreading method.
   
   Files:   kmeans_multithreading.py;   K-Means Clustering Multithreading.png;   kmeans_multithreading_execution_log.txt

   This method is outperformed by both of GPU methods by at least 10* times.
   
3. GPU multithreading without using shared memory.
   
   Files:   kmeans_cuda_without_s.py;   K-Means Clustering CUDA.png;   kmeans_cuda_execution_log.txt

   This method has basically the same performance with the version using shared memory.
   
4. GPU multithreading using shared memory.

   Files:   kmeans_cuda.py;   K-means Clustering Shared Cuda.png;   Kmeans_cuda_execution_shared_log.txt

   This method has the same performance with the version not using shared memory.

   
