# CUDA

## Petunjuk Penggunaan Program
1. Kompilasi program serial dan parallel yang dibuat dengan menjalankan perintah `make`.
2. Jalankan hasil kompilasi pada folder `src` dengan perintah `./dijkstra` untuk parallel dan `./dijkstra_serial` untuk serial.
    
## Pembagian Tugas
Berikut adalah pembagian tugas yang diterapkan.

| NIM | Nama | Pembagian Tugas | Kontribusi |
| ------ | ------ | ------ | ------ | 
| 13517029 | Reyhan Naufal Hakim | picking up thread, cuda dijkstra, main | 50% |
| 13517080 | Mgs. Muhammad Riandi Ramadhan | makefile, serial and cuda separation, cuda memory allocation, cuda record event, function type qualifier, main | 50% | 

## Laporan Pengerjaan
Berikut adalah laporan pengerjaan dari tugas ini.

### Deskripsi Solusi Parallel
Pada pengerjaan tugas ini, dilakukan proses pencarian jarak terpendek dari setiap node ke semua node lainnya.
Untuk mencari jarak terpendek dengan suatu source node tertentu, kita menggunakan algoritma dijkstra untuk melakukan penelusuran yang menghasilkan array yang berisi jarak terpendak source node tersebut ke semua node lainnya.
Untuk mendapatkan matriks yang berisi jarak terpendek dari setiap node ke semua node lainnya, algoritma dijkstra yang diterapkan pada setiap node dijalankan secara parallel dan setiap hasil array jarak terpendek dari algoritma tersebut akan disatukan menjadi sebuah matriks.

Kita menggunakan CUDA untuk melakukan pemrograman paralel dengan NVIDIA GPU. Dengan CUDA, dapat dilakukan pemrosesan algoritma dalam bahasa C++ dengan memanfaatkan ribuan thread paralel yang berjalan pada NVIDIA GPU(s), sebuah pendekatan pemrograman yang dikenal sebagai GPGPU (General-Purpose computing on Graphics Processing Units).
Platform CUDA sendiri adalah sebuah layer perangkat lunak yang dapat memberikan akses langsung terhadap virtual instruction set dan elemen komputasi paralel, untuk melakukan eksekusi terhadap compute kernels. 

Setelah graph berhasil di-generate menggunakan seed yang diberikan, dilakukan perhitungan waktu proses untuk serial dan parallel.
Serial menggunakan fungsi ```clock() ``` sedangkan parallel menggunakan fungsi berikut:
```c++
cudaEvent_t start_time, end_time;
cudaEventCreate(&start_time);
cudaEventCreate(&end_time);

cudaEventRecord(start_time);

// Cuda Djikstra Parallel Algorithm execution here

cudaEventRecord(end_time);
cudaDeviceSynchronize();

float elapsed_time = 0.0;
cudaEventElapsedTime(&elapsed_time, start_time, end_time);
```
Agar program dapat bekerja secara parallel, diberikan sebuah jumlah block (```n_block```) dan ukuran block (```block_size```) yang digunakan dalam pemrosesan algoritma dijkstra:
```c++
cuda_dijkstra<<<n_block, block_size>>>(matrix_distance, final_matrix_distance);
```
### Analisis Solusi yang Diberikan
Berdasarkan solusi yang digunakan, kita menghasilkan kinerja yang lebih ringan dengan melakukan parallelisasi data untuk ditangani secara multi-threaded pada GPU.
Hal ini dibuktikan dengan hasil pengujian di mana ketika dijkstra dijalankan secara parallel memiliki elapsed-time yang lebih kecil dibandingkan serial, dengan
asumsi actual work yang dilakukan oleh setiap thread secara signifikan lebih besar dibandingkan overhead yang dihasilkan dari paralelisasi dengan GPU berbasis CUDA.
Untuk menghasilkan solusi yang lebih baik, dapat digunakan hardware GPU yang lebih baik karena program telah memanfaatkan semua thread yang dapat digunakan pada GPU.

### Jumlah Thread yang Digunakan
Jumlah thread yang digunakan adalah sejumlah _parallel processors_ yang terdapat pada GPU CUDA dari sistem uji. _Parallel processors_ ini 
dikelompokkan ke dalam _Streaming Multiprocessors_, atau SMs. Setiap SM dapat menjalankan beberapa blok thread bersamaan. Sebagai contoh, GPU
GTX 1080 Ti yang digunakan pada sistem uji memiliki 28 SM, masing-masing memiliki 128 CUDA core. Dengan algoritma di bawah, berarti digunakan 28 SM x 128 CUDA cores = **3584 CUDA cores sebagai thread**.

```c++
int block_size = 256;
int n_block = (n_node + block_size - 1) / block_size;

cuda_dijkstra<<<n_block, block_size>>>(matrix_distance, final_matrix_distance);
```

Pada prosedur cuda_dijkstra:
```c++
__global__
void cuda_dijkstra(int **matrix_distance, int **final_matrix_distance) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int itr = index; itr < n_node; itr += stride) {
    dijkstra(itr, final_matrix_distance[itr], matrix_distance);
    printf("Cuda | Node %d out of %d\n", itr+1, n_node);
  }
}
```

### Pengukuran Kinerja untuk Tiap Kasus Uji Dibandingkan dengan Dijkstra Algorithm Serial (dalam microseconds)
Hasil pengukuran kinerja juga dapat dilihat dalam folder `screenshots`.


| Banyak Node | Tes Serial ke-1 | Tes Serial ke-2 | Tes Serial ke-3 | Tes Parallel ke-1 | Tes Parallel ke-2 | Tes Parallel ke-3 | Rata-rata Tes Serial | Rata-rata Tes Parallel |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 100 | 23.784 | 20.785 | 25.054 | 6.380 | 7.708 | 6.711 | 23.208 | 6.933 |
| 500 | 1.877.301 | 1.975.650 | 1.871.909 | 149.734 | 135.774 | 150.216 | 1.908.287 | 145.241 | 
| 1000 | 13.134.749 | 13.155.631 | 12.872.469 | 629.758 | 631.748 | 592.944 | 13.054.283 | 618.150 | 
| 3000 | 345.747.573 | 345.804.669 | 347.799.601 | 6.315.039 | 6.287.546 | 6.290.903 | 346.450.614 | 6.297.829 | 

### Analisis Perbandingan Kinerja Serial dan Parallel
Berdasarkan pengukuran kinerja yang dilakukan untuk setiap kasus uji, hasil analisis pada sistem uji adalah sebagai berikut.

* Untuk setiap kasus uji, program yang dijalankan secara parallel dengan CUDA GPU menghasilkan elapsed time yang lebih sedikit. Selain karena jumlah thread (_CUDA cores_) yang digunakan banyak, GPU sangat baik dalam melakukan pemrosesan dalam jumlah banyak namun sederhana seperti pemrosesan matriks .
* Semakin banyak jumlah data yang digunakan, maka perbedaan kecepatan elapsed time parallel dan serial akan semakin besar. 