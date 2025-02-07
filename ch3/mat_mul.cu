#include <stdio.h>
#include <cuda.h>
#include <time.h>

__global__
void mat_mul(float *A, float *B, float *C, int m, int n, int k, int l){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < l){
        float sum = 0.0f;
        for (int i = 0; i < n; i++){
            sum += A[row * n + i] * B[i * l + col];
        }
        C[row * l + col] = sum;
    }
    
    
}

__global__
void row_output(float *A, float *B, float *C, int m, int n, int k, int l, int row) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < l) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * l + col];
        }
        C[row * l + col] = sum;
    }
}

__global__
void col_output(float *A, float *B, float *C, int m, int n, int k, int l, int col) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * l + col];
        }
        C[row * l + col] = sum;
    }
}

int main(int argc, char **argv)
{

    float elapsed_time;
    cudaEvent_t start, stop;
    clock_t cpu_start, cpu_end;

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    if (argc != 5){
        printf("Usage: ./a.out <m> <n> <k> <l>\n");
        return 1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    int l = atoi(argv[4]);

    printf("Matrix size: %d x %d\n", m, n);
    printf("Matrix size: %d x %d\n", k, l);
    printf("----------------------------------------\n");   

    // Allocate memory on host
    h_A = (float *)malloc(m * n * sizeof(float));
    h_B = (float *)malloc(k * l * sizeof(float));
    h_C = (float *)malloc(m * l * sizeof(float));

    // Initialize host memory   
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            h_A[i * n + j] = (float)(rand() % 100);
        }
    }
    
    // Calculate total GPU memory required
    size_t total_gpu_memory = (m * n + k * l + m * l) * sizeof(float);
    printf("Total GPU memory required: %.2f MB\n", total_gpu_memory / (1024.0 * 1024.0));

    // Allocate memory on device
    cudaError_t err;
    err = cudaMalloc((void **)&d_A, m * n * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for d_A (%.2f MB) - %s\n", 
                (m * n * sizeof(float)) / (1024.0 * 1024.0),
                cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc((void **)&d_B, k * l * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for d_B (%.2f MB) - %s\n",
                (k * l * sizeof(float)) / (1024.0 * 1024.0),
                cudaGetErrorString(err));
        cudaFree(d_A);
        return 1;
    }

    err = cudaMalloc((void **)&d_C, m * l * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for d_C (%.2f MB) - %s\n",
                (m * l * sizeof(float)) / (1024.0 * 1024.0),
                cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return 1;
    }

    // Copy host memory to device memory
    cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * l * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing 
    cpu_start = clock();
    cudaEventRecord(start);

    // Launch the kernel
    dim3 dimBlock(32, 32);  
    dim3 dimGrid((m + dimBlock.x - 1) / dimBlock.x, 
                 (l + dimBlock.y - 1) / dimBlock.y);
    mat_mul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n, k, l);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cpu_end = clock();

    // Copy device memory to host memory
    cudaMemcpy(h_C, d_C, m * l * sizeof(float), cudaMemcpyDeviceToHost);

    // Print timing information
    printf("----------------------------------------\n");
    printf("Performance Results:\n");
    printf("GPU Kernel Time: %f ms\n", elapsed_time);
    printf("Total CPU Time: %f ms\n", 1000.0 * (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC);

    // Cleanup timing events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("\n=== Performance Comparison ===\n");

    // 1. Full Matrix Multiplication
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 fullBlock(32, 32);
    dim3 fullGrid((m + fullBlock.x - 1) / fullBlock.x, 
                  (l + fullBlock.y - 1) / fullBlock.y);
    mat_mul<<<fullGrid, fullBlock>>>(d_A, d_B, d_C, m, n, k, l);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float full_time;
    cudaEventElapsedTime(&full_time, start, stop);
    printf("Full matrix multiplication time: %.3f ms\n", full_time);

    // 2. Row-by-row Multiplication
    cudaEventRecord(start);
    
    dim3 rowBlock(256);
    for(int i = 0; i < m; i++) {
        dim3 rowGrid((l + rowBlock.x - 1) / rowBlock.x);
        row_output<<<rowGrid, rowBlock>>>(d_A, d_B, d_C, m, n, k, l, i);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float row_time;
    cudaEventElapsedTime(&row_time, start, stop);
    printf("Row-by-row multiplication time: %.3f ms\n", row_time);

    // 3. Column-by-column Multiplication
    cudaEventRecord(start);
    
    dim3 colBlock(256);
    for(int j = 0; j < l; j++) {
        dim3 colGrid((m + colBlock.x - 1) / colBlock.x);
        col_output<<<colGrid, colBlock>>>(d_A, d_B, d_C, m, n, k, l, j);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float col_time;
    cudaEventElapsedTime(&col_time, start, stop);
    printf("Column-by-column multiplication time: %.3f ms\n", col_time);

    // Print speedup comparisons
    printf("\nSpeedup Analysis:\n");
    printf("Row-by-row vs Full: %.2fx slower\n", row_time / full_time);
    printf("Column-by-column vs Full: %.2fx slower\n", col_time / full_time);
    printf("Column vs Row: %.2fx %s\n", 
           fabs(col_time / row_time), 
           col_time > row_time ? "slower" : "faster");

    return 0;
}