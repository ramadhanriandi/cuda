#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <omp.h>
#include <limits.h> 

#define n_node 3000

int matrix_distance[n_node][n_node];
int final_matrix_distance[n_node][n_node];

int main(int argc, char** argv[]) {
  // number of thread, bool of serial, source node, iterator
  int source, itr;

  // seed from 13517080
  int seed = 80;

  // time for serial
  clock_t t_serial;

  // Matrix initialization for graph
  init_graph(seed);

  // START SERIAL DIJKSTRA ALGORITHM
  t_serial = clock();

  for (itr = 0; itr < n_node; itr++) {  
    dijkstra(itr, final_matrix_distance[itr]);
    printf("Serial | Node %d out of %d\n", itr+1, n_node);
  }

  t_serial = clock() - t_serial;

  double time_taken_serial = ((double)t_serial * 1000000) / (CLOCKS_PER_SEC);

  print_matrix_to_file();

  // PRINT RESULT OF SERIAL DIJKSTRA ALGORITHM
  printf("\n%s%2.f%s\n", "Time elapsed for serial dijkstra algorithm: ", time_taken_serial, " microsecond");

  // END OF SERIAL DIJKSTRA ALGORITHM

  return 0;
}

void init_graph(int seed) {
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

void print_matrix_to_file() {
  FILE * fp;
  /* open the file for writing*/
  fp = fopen ("../output/serial_3000.txt","w");

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

// DIJKSTRA SERIAL
void dijkstra(int src, int dist[n_node]) { 
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