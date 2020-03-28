#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <omp.h>
#include <limits.h> 

#define n_node 500

void init_graph(int **matrix_distance, int seed) {
  for (int i = 0; i < n_node; i++) {
    for (int j = 0; j < n_node; j++) {
      if (i == j) {
        matrix_distance[i][j] = 0;
      } else if (i < j) {
        int parity = rand() % seed;
        if (parity % 2 == 0) {
          matrix_distance[i][j] = -1;
          matrix_distance[j][i] = -1;
        } else {
          matrix_distance[i][j] = parity;
          matrix_distance[j][i] = parity;
        }
      }
    }
  }
}

void print_matrix_to_file(int **final_matrix_distance) {
  FILE * fp;
  /* open the file for writing*/
  fp = fopen ("../output/cuda_500.txt","w");

  for (int i = 0; i < n_node; i++) {
    for (int j = 0; j < n_node; j++) {
      fprintf(fp, "%d ", final_matrix_distance[i][j]);
    }
    fprintf(fp, "\n");
  }

  /* close the file*/
  fclose (fp);
}

int minDistance(int dist[], bool sptSet[]) { 
  int min = INT_MAX, min_index; 

  for (int v = 0; v < n_node; v++) {
    if (sptSet[v] == false && dist[v] <= min) {
      min = dist[v], min_index = v; 
    }
  }

  return min_index; 
} 

void dijkstra(int src, int dist[n_node], int **matrix_distance) { 
  bool sptSet[n_node];

  for (int i = 0; i < n_node; i++) {
    dist[i] = INT_MAX, sptSet[i] = false; 
  }

  dist[src] = 0; 

  for (int count = 0; count < n_node - 1; count++) { 
    int u = minDistance(dist, sptSet); 

    sptSet[u] = true; 

    for (int v = 0; v < n_node; v++) {
      if (!sptSet[v] && matrix_distance[u][v] && dist[u] != INT_MAX 
        && dist[u] + matrix_distance[u][v] < dist[v]) {
        dist[v] = dist[u] + matrix_distance[u][v]; 
      }
    }
  } 
}

__global__
void cuda_dijkstra(int **matrix_distance, int **final_matrix_distance) {
  // CUDA PARALLEL DIJKSTRA EXECUTION FROM EACH SOURCE NODE
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int itr = index; itr < n_node; itr += stride) {
    dijkstra(itr, final_matrix_distance[itr], matrix_distance);
    printf("Cuda | Node %d out of %d\n", itr+1, n_node);
  }
}

int main(int argc, char** argv[]) {
  int **matrix_distance, **final_matrix_distance;

  cudaMallocManaged(&matrix_distance, n_node * sizeof(int *));
  cudaMallocManaged(&final_matrix_distance, n_node * sizeof(int *));

  for (int i = 0; i < n_node; i++) {
    cudaMallocManaged(&matrix_distance[i], n_node * sizeof(int));
    cudaMallocManaged(&final_matrix_distance[i], n_node * sizeof(int));
  } 

  // seed from 13517080
  int seed = 80;

  // Matrix initialization for graph
  init_graph(matrix_distance, seed);

  // TODO: Thread count using input from argument (this use all available computing resources on the GPU)
  int block_size = 256;
  int n_block = (n_node + block_size - 1) / block_size;

  cuda_dijkstra<<<n_block, block_size>>>(matrix_distance, final_matrix_distance);

  cudaDeviceSynchronize();

  print_matrix_to_file(final_matrix_distance);

  for (int i = 0; i < n_node; i++) {
    cudaFree(matrix_distance[i]);
    cudaFree(final_matrix_distance[i]);
  }

  cudaFree(matrix_distance);
  cudaFree(final_matrix_distance);

  return 0;
}