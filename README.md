# CSC591-A3
CUDA-Parallel K-means clustering

This repo stores the py files, visualized results and excution logs that sotres the running time for k-means clustering algorithm implemented by four different methods:

1. CPU sequential method.
   
   Files:   kmeans_sequential.py;   K-Means Clustering Sequential.png;   kmeans_sequential_excution_log.txt

   This method is outperformed by all other methods by at least 10* times.

3. GPU multithreading method.
4. 
   Files:   kmeans_multithreading.py;   K-Means Clustering Multithreading.png;   kmeans_multithreading_excution_log.txt

   This method is outperformed by both of GPU methods by at least 10* times.
   
6. GPU multithreading without using shared memory.
7. GPU multithreading using shared memory.

   
