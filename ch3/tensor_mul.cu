//------------------------------------------------------------------------------
// Refactored tensor_mul.cu
//------------------------------------------------------------------------------
// This file is refactored to organize every test into separate functions and 
// to reduce overall file length. The tests include:
//  0. Naive GPU Implementation
//  1. CPU Implementation (OpenMP)
//  2. Shared Memory Implementation
//  4. Vectorized Implementation
//  5. Warp-Optimized Implementation
//  6. Double-Buffered Implementation
//  7. Tensor Core Implementation
//------------------------------------------------------------------------------

// Standard and CUDA includes:
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <omp.h>
#include <quadmath.h>  // For __float128 (quad precision)
#include <cuda_fp16.h> // For half precision

//------------------------------------------------------------------------------
// Macros and architecture constants
//------------------------------------------------------------------------------
#define TILE_SIZE      32
#define BLOCK_SIZE     32
#define MIN(a,b)       ((a) < (b) ? (a) : (b))

// CPU blocking constants for cache-blocked multiplication:
#define BLOCK_SIZE_M   32
#define BLOCK_SIZE_N   32
#define BLOCK_SIZE_L   32

//------------------------------------------------------------------------------
// Structure to hold performance metrics for a single test
//------------------------------------------------------------------------------
struct PerfMetrics {
    float transferTime;  // Host-to-device time (ms)
    float kernelTime;    // Kernel execution time (ms)
    float d2hTime;       // Device-to-host time (ms)
    float totalTime;     // Total measured time (ms)
    float gflops;        // GFLOPS computed from kernel time
};

//------------------------------------------------------------------------------
// Forward declarations of GPU kernel functions
//------------------------------------------------------------------------------
// Test 0: Naive GPU kernel (already implemented)
__global__ void tensor_mul(const float* A, const float* B, float* C,
                           int batch_size, int m, int n, int k, int l);
// Test 2: Shared Memory implementation (assumed implemented)
__global__ void tensor_mul_shared(const float* A, const float* B, float* C,
                                  int batch_size, int m, int n, int k, int l);
// Test 4: Vectorized Implementation (using regular floats)
__global__ void tensor_mul_vectorized(const float* A, const float* B, float* C,
                                      int batch_size, int m, int n, int k, int l);
// Test 5: Warp-Optimized Implementation
__global__ void tensor_mul_warp_optimized(const float* A, const float* B, float* C,
                                          int batch_size, int m, int n, int k, int l);
// Test 6: Double-Buffered Implementation
__global__ void tensor_mul_double_buffered(const float* A, const float* B, float* C,
                                           int batch_size, int m, int n, int k, int l);
// Test 7: Tensor Core Implementation (using half precision for A and B)
__global__ void tensor_mul_tensorcore(const half* A, const half* B, float* C,
                                      int batch_size, int m, int n, int k, int l);

//------------------------------------------------------------------------------
// Utility function: Initialize matrices A and B with random floats
//------------------------------------------------------------------------------
void initMatrices(float* A, float* B, int batch_size, int m, int n, int k, int l) {
    for (int b = 0; b < batch_size; b++){
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                A[b * m * n + i * n + j] = (float)(rand() % 100) / 100.0f;
            }
        }
        for (int i = 0; i < k; i++){
            for (int j = 0; j < l; j++){
                B[b * k * l + i * l + j] = (float)(rand() % 100) / 100.0f;
            }
        }
    }
}

//------------------------------------------------------------------------------
// Utility function: Check results against baseline with tolerance tol
// Returns the maximum difference found.
//------------------------------------------------------------------------------
float checkResults(const float* baseline, const float* test, int total_elements, float tol) {
    float max_diff = 0.0f;
    for (int i = 0; i < total_elements; i++){
        float diff = fabs(baseline[i] - test[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    return max_diff;
}

//------------------------------------------------------------------------------
// CPU Implementation using OpenMP and __float128 (quad precision) for improved accuracy.
//------------------------------------------------------------------------------
void cpu_matrix_multiply(float* A, float* B, float* C,
                         int batch_size, int m, int n, int k, int l) {
    // Initialize C to zero.
    memset(C, 0, batch_size * m * l * sizeof(float));
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch_size; b++){
        for (int i_start = 0; i_start < m; i_start += BLOCK_SIZE_M){
            for (int j_start = 0; j_start < l; j_start += BLOCK_SIZE_L){
                int imax = MIN(i_start + BLOCK_SIZE_M, m);
                int jmax = MIN(j_start + BLOCK_SIZE_L, l);
                for (int i = i_start; i < imax; i++){
                    for (int j = j_start; j < jmax; j++){
                        __float128 sum = 0.0Q;
                        size_t base_a = (size_t)b * m * n + (size_t)i * n;
                        size_t base_b = (size_t)b * k * l + (size_t)j;
                        for (int p = 0; p < n; p++){
                            __float128 a_val = A[base_a + p];
                            __float128 b_val = B[base_b + p * l];
                            sum += a_val * b_val;
                        }
                        size_t idx = (size_t)b * m * l + (size_t)i * l + j;
                        #pragma omp atomic
                        C[idx] += (float)sum;
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// A generic GPU test runner which performs H2D transfer, kernel launch, and D2H transfer.
//------------------------------------------------------------------------------
PerfMetrics runGpuTest(const char* testName,
    void(*kernel)(const float*, const float*, float*, int, int, int, int, int),
    const float* h_A, const float* h_B, float* h_C,
    int batch_size, int m, int n, int k, int l,
    dim3 grid, dim3 block) {

    PerfMetrics pm = {0};
    float elapsed;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate device memory.
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, batch_size * m * n * sizeof(float));
    cudaMalloc((void**)&d_B, batch_size * k * l * sizeof(float));
    cudaMalloc((void**)&d_C, batch_size * m * l * sizeof(float));

    // Perform Host-to-Device transfer.
    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, batch_size * m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, batch_size * k * l * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    pm.transferTime = elapsed;

    // Kernel launch.
    cudaEventRecord(start);
    kernel<<<grid, block>>>(d_A, d_B, d_C, batch_size, m, n, k, l);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    pm.kernelTime = elapsed;

    // Device-to-Host transfer.
    cudaEventRecord(start);
    cudaMemcpy(h_C, d_C, batch_size * m * l * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    pm.d2hTime = elapsed;

    pm.totalTime = pm.transferTime + pm.kernelTime + pm.d2hTime;
    pm.gflops = (2.0f * batch_size * m * n * l) / (pm.kernelTime * 1e6f);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("%s:\n", testName);
    printf("   H2D: %.3f ms, Kernel: %.3f ms, D2H: %.3f ms, Total: %.3f ms, GFLOPS: %.2f\n",
           pm.transferTime, pm.kernelTime, pm.d2hTime, pm.totalTime, pm.gflops);
    return pm;
}

//------------------------------------------------------------------------------
// Test 0: Naive GPU Implementation
//------------------------------------------------------------------------------
PerfMetrics runTestNaive(const float* h_A, const float* h_B, float* h_C,
                         int batch_size, int m, int n, int k, int l) {
    dim3 block(16, 16);
    dim3 grid((m + 15) / 16, (l + 15) / 16, batch_size);
    return runGpuTest("Test 0: Naive GPU Implementation", tensor_mul,
                      h_A, h_B, h_C, batch_size, m, n, k, l, grid, block);
}

//------------------------------------------------------------------------------
// Test 2: Shared Memory Implementation
//------------------------------------------------------------------------------
PerfMetrics runTestSharedMemory(const float* h_A, const float* h_B, float* h_C,
                                int batch_size, int m, int n, int k, int l) {
    dim3 block(16, 16);
    dim3 grid((m + 15) / 16, (l + 15) / 16, batch_size);
    return runGpuTest("Test 2: Shared Memory Implementation", tensor_mul_shared,
                      h_A, h_B, h_C, batch_size, m, n, k, l, grid, block);
}

//------------------------------------------------------------------------------
// Test 3: cuBLAS Implementation (Placeholder)
//------------------------------------------------------------------------------
PerfMetrics runTestCublas(const float* h_A, const float* h_B, float* h_C,
                           int batch_size, int m, int n, int k, int l) {
    PerfMetrics pm = {0};
    cublasHandle_t handle;
    cublasCreate(&handle);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, batch_size * m * n * sizeof(float));
    cudaMalloc((void**)&d_B, batch_size * n * l * sizeof(float));
    cudaMalloc((void**)&d_C, batch_size * m * l * sizeof(float));

    cudaMemcpy(d_A, h_A, batch_size * m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, batch_size * n * l * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;
    // cuBLAS uses column-major ordering. This call transposes the operation so that
    // our row-major data is correctly multiplied.
    int lda = n, ldb = l, ldc = l;  // Leading dimensions for column-major format
    long long strideA = m * n;
    long long strideB = n * l;
    long long strideC = m * l;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              l, m, n,  // Swap m and l for column-major
                              &alpha,
                              d_B, ldb, strideB,  // Swap A and B
                              d_A, lda, strideA,
                              &beta,
                              d_C, ldc, strideC, batch_size);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    pm.kernelTime = elapsed;

    cudaMemcpy(h_C, d_C, batch_size * m * l * sizeof(float), cudaMemcpyDeviceToHost);

    pm.totalTime = pm.kernelTime; // excluding transfers for simplicity
    pm.transferTime = 0;
    pm.d2hTime = 0;
    pm.gflops = (2.0f * batch_size * m * n * l) / (pm.kernelTime * 1e6f);

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    printf("Test 3: cuBLAS Implementation:\n");
    printf("   Kernel: %.3f ms, GFLOPS: %.2f\n", pm.kernelTime, pm.gflops);

    return pm;
}

//------------------------------------------------------------------------------
// Test 4: Vectorized Implementation (using regular floats)
//------------------------------------------------------------------------------
PerfMetrics runTestVectorized(const float* h_A, const float* h_B, float* h_C,
                              int batch_size, int m, int n, int k, int l) {
    dim3 block(16, 16);
    dim3 grid((m + 15) / 16, (l + 15) / 16, batch_size);
    
    return runGpuTest("Test 4: Vectorized Implementation",
                      tensor_mul_vectorized,
                      h_A, h_B, h_C, batch_size, m, n, k, l, grid, block);
}

//------------------------------------------------------------------------------
// Test 5: Warp-Optimized Implementation
//------------------------------------------------------------------------------
PerfMetrics runTestWarpOptimized(const float* h_A, const float* h_B, float* h_C,
                                 int batch_size, int m, int n, int k, int l) {
    dim3 block(16, 16);
    dim3 grid((m + 15) / 16, (l + 15) / 16, batch_size);
    return runGpuTest("Test 5: Warp-Optimized Implementation", tensor_mul_warp_optimized,
                      h_A, h_B, h_C, batch_size, m, n, k, l, grid, block);
}

//------------------------------------------------------------------------------
// Test 6: Double-Buffered Implementation
//------------------------------------------------------------------------------
PerfMetrics runTestDoubleBuffered(const float* h_A, const float* h_B, float* h_C,
                                  int batch_size, int m, int n, int k, int l) {
    dim3 block(16, 16);
    dim3 grid((m + 15) / 16, (l + 15) / 16, batch_size);
    return runGpuTest("Test 6: Double-Buffered Implementation", tensor_mul_double_buffered,
                      h_A, h_B, h_C, batch_size, m, n, k, l, grid, block);
}

//------------------------------------------------------------------------------
// Test 7: Tensor Core Implementation (requires half precision)
//------------------------------------------------------------------------------
PerfMetrics runTestTensorCore(const float* h_A, const float* h_B, float* h_C,
                              int batch_size, int m, int n, int k, int l) {
    PerfMetrics pm = {0};
    // Tensor Core requires dimensions be multiples of 16.
    if (m % 16 != 0 || n % 16 != 0 || l % 16 != 0) {
        printf("Test 7: Tensor Core Implementation: Skipped (dimensions must be multiples of 16).\n");
        return pm;
    }
    
    size_t totalA = batch_size * m * n;
    size_t totalB = batch_size * k * l;
    // Allocate host memory for half-precision arrays.
    half* h_A_half = (half*)malloc(totalA * sizeof(half));
    half* h_B_half = (half*)malloc(totalB * sizeof(half));
    for (size_t i = 0; i < totalA; i++){
        h_A_half[i] = __float2half(h_A[i]);
    }
    for (size_t i = 0; i < totalB; i++){
        h_B_half[i] = __float2half(h_B[i]);
    }
    
    // Allocate device memory for half precision.
    half *d_A_half, *d_B_half;
    cudaMalloc((void**)&d_A_half, totalA * sizeof(half));
    cudaMalloc((void**)&d_B_half, totalB * sizeof(half));
    cudaMemcpy(d_A_half, h_A_half, totalA * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_half, h_B_half, totalB * sizeof(half), cudaMemcpyHostToDevice);
    
    // Allocate device memory for result (reuse d_C allocated in runGpuTest pattern).
    float* d_C;
    cudaMalloc((void**)&d_C, batch_size * m * l * sizeof(float));
    
    // Launch Tensor Core kernel.
    dim3 block(16, 16, 1);
    dim3 grid((m + 15) / 16, (l + 15) / 16, batch_size);
    float elapsed;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    tensor_mul_tensorcore<<<grid, block>>>(d_A_half, d_B_half, d_C, batch_size, m, n, k, l);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    pm.kernelTime = elapsed;
    
    // Copy result back to host.
    cudaMemcpy(h_C, d_C, batch_size * m * l * sizeof(float), cudaMemcpyDeviceToHost);
    pm.totalTime = pm.kernelTime;  // For simplicity, not timing transfers here.
    pm.transferTime = 0;
    pm.d2hTime = 0;
    pm.gflops = (2.0f * batch_size * m * n * l) / (pm.kernelTime * 1e6f);
    
    free(h_A_half);
    free(h_B_half);
    cudaFree(d_A_half);
    cudaFree(d_B_half);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return pm;
}

//------------------------------------------------------------------------------
// Main: Orchestrate all tests and print a summary.
//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc != 6) {
        printf("Usage: ./tensor_mul <batch_size> <m> <n> <k> <l>\n");
        return 1;
    }
    int batch_size = atoi(argv[1]);
    int m = atoi(argv[2]);
    int n = atoi(argv[3]);
    int k = atoi(argv[4]);
    int l = atoi(argv[5]);
    if (batch_size <= 0 || m <= 0 || n <= 0 || k <= 0 || l <= 0) {
        printf("Error: All dimensions must be positive integers.\n");
        return 1;
    }
    
    size_t total_elements_A = batch_size * m * n;
    size_t total_elements_B = batch_size * k * l;
    size_t total_elements_C = batch_size * m * l;
    
    // Allocate host matrices.
    float *h_A = (float*)malloc(total_elements_A * sizeof(float));
    float *h_B = (float*)malloc(total_elements_B * sizeof(float));
    // For baseline from GPU test 0.
    float *h_C_baseline = (float*)malloc(total_elements_C * sizeof(float));
    // Temporary output array for other tests.
    float *h_C_temp = (float*)malloc(total_elements_C * sizeof(float));
    if (!h_A || !h_B || !h_C_baseline || !h_C_temp) {
        printf("Error: Host memory allocation failed.\n");
        return 1;
    }
    srand(time(NULL));
    initMatrices(h_A, h_B, batch_size, m, n, k, l);
    
    // Test 0: Naive GPU Implementation (baseline)
    PerfMetrics pm0 = runTestNaive(h_A, h_B, h_C_baseline, batch_size, m, n, k, l);
    
    // Test 1: CPU Implementation using OpenMP.
    double t_cpu_start = clock();
    cpu_matrix_multiply(h_A, h_B, h_C_temp, batch_size, m, n, k, l);
    double cpu_time = (clock() - t_cpu_start) / (double)CLOCKS_PER_SEC * 1000.0;
    printf("Test 1: CPU Implementation (OpenMP):\n   Computation: %.3f ms\n", cpu_time);
    float tol = 1e-4f;  // Tolerance set to 1e-4 for CPU check.
    float max_diff_cpu = checkResults(h_C_baseline, h_C_temp, total_elements_C, tol);
    if (max_diff_cpu <= tol)
        printf("   Accuracy Check: PASSED (max diff: %e)\n", max_diff_cpu);
    else
        printf("   Accuracy Check: FAILED (max diff: %e)\n", max_diff_cpu);
    
    // Test 2: Shared Memory Implementation.
    PerfMetrics pm2 = runTestSharedMemory(h_A, h_B, h_C_temp, batch_size, m, n, k, l);
    float diff2 = checkResults(h_C_baseline, h_C_temp, total_elements_C, 1e-5f);
    printf("Test 2: Shared Memory Implementation: Accuracy (max diff: %e)\n", diff2);
    
    // Test 3: cuBLAS Implementation.
    PerfMetrics pm3 = runTestCublas(h_A, h_B, h_C_temp, batch_size, m, n, k, l);
    float diff3 = checkResults(h_C_baseline, h_C_temp, total_elements_C, 1e-5f);
    printf("Test 3: cuBLAS Implementation: Accuracy (max diff: %e)\n", diff3);
    
    // Test 4: Vectorized Implementation.
    PerfMetrics pm4 = runTestVectorized(h_A, h_B, h_C_temp, batch_size, m, n, k, l);
    float diff4 = checkResults(h_C_baseline, h_C_temp, total_elements_C, 1e-5f);
    printf("Test 4: Vectorized Implementation: Accuracy (max diff: %e)\n", diff4);
    
    // Test 5: Warp-Optimized Implementation.
    PerfMetrics pm5 = runTestWarpOptimized(h_A, h_B, h_C_temp, batch_size, m, n, k, l);
    float diff5 = checkResults(h_C_baseline, h_C_temp, total_elements_C, 1e-5f);
    printf("Test 5: Warp-Optimized Implementation: Accuracy (max diff: %e)\n", diff5);
    
    // Test 6: Double-Buffered Implementation.
    PerfMetrics pm6 = runTestDoubleBuffered(h_A, h_B, h_C_temp, batch_size, m, n, k, l);
    float diff6 = checkResults(h_C_baseline, h_C_temp, total_elements_C, 1e-5f);
    printf("Test 6: Double-Buffered Implementation: Accuracy (max diff: %e)\n", diff6);
    
    // Test 7: Tensor Core Implementation.
    PerfMetrics pm7 = runTestTensorCore(h_A, h_B, h_C_temp, batch_size, m, n, k, l);
    float diff7 = checkResults(h_C_baseline, h_C_temp, total_elements_C, 2e-2f);
    printf("Test 7: Tensor Core Implementation: Accuracy (max diff: %e)\n", diff7);
    
    //------------------------------------------------------------------------------    
    // Performance Summary.
    //------------------------------------------------------------------------------
    printf("\n=== Performance Summary ===\n");
    printf("Implementation      Time (ms)    GFLOPS    vs Naive    vs CPU\n");
    printf("----------------------------------------------------------------\n");
    printf("0. Naive GPU       %8.3f    %7.2f    %6.2fx    %6.2fx\n", 
           pm0.totalTime, pm0.gflops, 1.0f, cpu_time/pm0.totalTime);
    printf("1. CPU (OpenMP)    %8.3f    %7.2f    %6.2fx    %6.2fx\n", 
           cpu_time, (2.0f * batch_size * m * n * l) / (cpu_time * 1e6f), 
           pm0.totalTime/cpu_time, 1.0f);
    printf("2. Shared Memory   %8.3f    %7.2f    %6.2fx    %6.2fx\n", 
           pm2.totalTime, pm2.gflops, pm0.totalTime/pm2.totalTime, 
           cpu_time/pm2.totalTime);
    printf("3. cuBLAS          %8.3f    %7.2f    %6.2fx    %6.2fx\n", 
           pm3.totalTime, pm3.gflops, pm0.totalTime/pm3.totalTime, 
           cpu_time/pm3.totalTime);
    printf("4. Vectorized      %8.3f    %7.2f    %6.2fx    %6.2fx\n", 
           pm4.totalTime, pm4.gflops, pm0.totalTime/pm4.totalTime, 
           cpu_time/pm4.totalTime);
    printf("5. Warp-Optimized  %8.3f    %7.2f    %6.2fx    %6.2fx\n", 
           pm5.totalTime, pm5.gflops, pm0.totalTime/pm5.totalTime, 
           cpu_time/pm5.totalTime);
    printf("6. Double-Buffered %8.3f    %7.2f    %6.2fx    %6.2fx\n", 
           pm6.totalTime, pm6.gflops, pm0.totalTime/pm6.totalTime, 
           cpu_time/pm6.totalTime);
    printf("7. Tensor Core     %8.3f    %7.2f    %6.2fx    %6.2fx\n", 
           pm7.totalTime, pm7.gflops, pm0.totalTime/pm7.totalTime, 
           cpu_time/pm7.totalTime);
    
    // Cleanup.
    free(h_A);
    free(h_B);
    free(h_C_baseline);
    free(h_C_temp);
    cudaFree(0);  // Reset device.
    
    return 0;
}

// Test 0: Naive GPU kernel implementation
__global__ void tensor_mul(const float* A, const float* B, float* C,
                           int batch_size, int m, int n, int k, int l) {
    int batch = blockIdx.z;
    int row   = blockIdx.x * blockDim.x + threadIdx.x;
    int col   = blockIdx.y * blockDim.y + threadIdx.y;
    if (batch < batch_size && row < m && col < l) {
        float sum = 0.0f;
        for (int p = 0; p < n; p++) {
            sum += A[batch * m * n + row * n + p] *
                   B[batch * k * l + p * l + col];
        }
        C[batch * m * l + row * l + col] = sum;
    }
}

// Device helper function to perform naive multiplication.
// This function can be called from inside __global__ kernels.
__device__ void tensor_mul_device(const float* A, const float* B, float* C,
                                  int batch_size, int m, int n, int k, int l) {
    int batch = blockIdx.z;
    int row   = blockIdx.x * blockDim.x + threadIdx.x;
    int col   = blockIdx.y * blockDim.y + threadIdx.y;
    if (batch < batch_size && row < m && col < l) {
        float sum = 0.0f;
        for (int p = 0; p < n; p++) {
            sum += A[batch * m * n + row * n + p] *
                   B[batch * k * l + p * l + col];
        }
        C[batch * m * l + row * l + col] = sum;
    }
}

// Test 2: Shared Memory Implementation (Placeholder)
__global__ void tensor_mul_shared(const float* A, const float* B, float* C,
                                  int batch_size, int m, int n, int k, int l) {
    tensor_mul_device(A, B, C, batch_size, m, n, k, l);
}

// Test 4: Vectorized Implementation (using regular floats)
__global__ void tensor_mul_vectorized(const float* A, const float* B, float* C,
                                      int batch_size, int m, int n, int k, int l) {
    int batch = blockIdx.z;
    int row   = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch < batch_size && row < m && col < l) {
        float sum = 0.0f;
        
        // Vectorized loop: process 4 elements at a time
        int p;
        #pragma unroll
        for (p = 0; p < n-3; p += 4) {
            sum += A[batch * m * n + row * n + p] * B[batch * k * l + p * l + col]
                 + A[batch * m * n + row * n + p + 1] * B[batch * k * l + (p + 1) * l + col]
                 + A[batch * m * n + row * n + p + 2] * B[batch * k * l + (p + 2) * l + col]
                 + A[batch * m * n + row * n + p + 3] * B[batch * k * l + (p + 3) * l + col];
        }
        
        // Handle remaining elements
        for (; p < n; p++) {
            sum += A[batch * m * n + row * n + p] * B[batch * k * l + p * l + col];
        }
        
        C[batch * m * l + row * l + col] = sum;
    }
}

// Test 5: Warp-Optimized Implementation (Placeholder)
__global__ void tensor_mul_warp_optimized(const float* A, const float* B, float* C,
                                          int batch_size, int m, int n, int k, int l) {
    tensor_mul_device(A, B, C, batch_size, m, n, k, l);
}

// Test 6: Double-Buffered Implementation (Placeholder)
__global__ void tensor_mul_double_buffered(const float* A, const float* B, float* C,
                                           int batch_size, int m, int n, int k, int l) {
    tensor_mul_device(A, B, C, batch_size, m, n, k, l);
}

// Test 7: Tensor Core Implementation (using half precision) (Placeholder)
__global__ void tensor_mul_tensorcore(const half* A, const half* B, float* C,
                                      int batch_size, int m, int n, int k, int l) {
    int batch = blockIdx.z;
    int row   = blockIdx.x * blockDim.x + threadIdx.x;
    int col   = blockIdx.y * blockDim.y + threadIdx.y;
    if (batch < batch_size && row < m && col < l) {
        float sum = 0.0f;
        for (int p = 0; p < n; p++) {
            float a = __half2float(A[batch * m * n + row * n + p]);
            float b = __half2float(B[batch * k * l + p * l + col]);
            sum += a * b;
        }
        C[batch * m * l + row * l + col] = sum;
    }
}
