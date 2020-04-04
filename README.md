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
```
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
```
cuda_dijkstra<<<n_block, block_size>>>(matrix_distance, final_matrix_distance);
```
### Analisis Solusi yang Diberikan


### Jumlah Thread yang Digunakan


### Pengukuran Kinerja untuk Tiap Kasus Uji Dibandingkan dengan Dijkstra Algorithm Serial (dalam microseconds)
Hasil pengukuran kinerja juga dapat dilihat dalam folder `screenshots`.


| Banyak Node | Tes Serial ke-1 | Tes Serial ke-2 | Tes Serial ke-3 | Tes Parallel ke-1 | Tes Parallel ke-2 | Tes Parallel ke-3 | Rata-rata Tes Serial | Rata-rata Tes Parallel |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 100 | 23.784 | 20.785 | 25.054 | 6.380 | 7.708 | 6.711 | 23.208 | 6.933 |
| 500 | 1.877.301 | 1.975.650 | 1.871.909 | 149.734 | 135.774 | 150.216 | 1.908.287 | 145.241 | 
| 1000 | 13.134.749 | 13.155.631 | 12.872.469 | 629.758 | 631.748 | 592.944 | 13.054.283 | 618.150 | 
| 3000 | 345.747.573 | 345.804.669 | 347.799.601 | 6.315.039 | 6.287.546 | 6.290.903 | 346.450.614 | 6.297.829 | 

### Analisis Perbandingan Kinerja Serial dan Parallel
