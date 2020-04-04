all: dijkstra dijkstra_serial

dijkstra: ./src/dijkstra.cu
	nvcc ./src/dijkstra.cu -o ./src/dijkstra

dijkstra_serial: ./src/dijkstra_serial.c
	gcc ./src/dijkstra_serial.c -o ./src/dijkstra_serial