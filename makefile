all:
	gcc -g -Wall -o ./output/dijkstra_omp ./src/dijkstra_omp.c -fopenmp