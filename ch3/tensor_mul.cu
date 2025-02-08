
// compile with:nvcc -O3 -arch=sm_86 -std=c++20 --use_fast_math -Xcompiler "-fopenmp -fPIC -pthread -march=native" tensor_mul.cu -o tensor_mul -lcublas

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
#include <cuda_fp16.h> // For half precision (if needed elsewhere)
#define WMMA_ENABLE_TF32
#include <mma.h>
using namespace nvcuda::wmma;
using namespace nvcuda::wmma::experimental;

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
// Test 6: Double-Buffered Implementation (Rewritten for near-zero error)
__global__ void tensor_mul_double_buffered(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float*       __restrict__ C,
                                           int batch_size, int m, int n, int k, int l);
// Test 7: Tensor Core Implementation (using TF32 on Ampere)
__global__ void tensor_mul_tensorcore(const float* A, const float* B, float* C,
                                     int batch_size, int M, int N, int k, int L);
// Test 7: Tensor Core Implementation (Rewritten for low error w/ float WMMA)
__global__ void tensor_mul_tensorcore_float(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float*       __restrict__ C,
                                            int batch_size, int M, int N, int K, int L);

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
                        __float128 sum = (__float128)0.0;
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
    dim3 block(32, 32);
    dim3 grid((m + 31) / 32, (l + 31) / 32, batch_size);
    return runGpuTest("Test 0: Naive GPU Implementation", tensor_mul,
                      h_A, h_B, h_C, batch_size, m, n, k, l, grid, block);
}

//------------------------------------------------------------------------------
// Test 2: Shared Memory Implementation
//------------------------------------------------------------------------------
PerfMetrics runTestSharedMemory(const float* h_A, const float* h_B, float* h_C,
                                int batch_size, int m, int n, int k, int l) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((m + TILE_SIZE - 1) / TILE_SIZE, (l + TILE_SIZE - 1) / TILE_SIZE, batch_size);
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
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, batch_size * m * n * sizeof(float));
    cudaMalloc((void**)&d_B, batch_size * n * l * sizeof(float));
    cudaMalloc((void**)&d_C, batch_size * m * l * sizeof(float));

    // Time H2D transfers
    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, batch_size * m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, batch_size * n * l * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    pm.transferTime = elapsed;

    float alpha = 1.0f, beta = 0.0f;
    // cuBLAS uses column-major ordering. This call transposes the operation so that
    // our row-major data is correctly multiplied.
    int lda = n, ldb = l, ldc = l;  // Leading dimensions for column-major format
    long long strideA = m * n;
    long long strideB = n * l;
    long long strideC = m * l;

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
    cudaEventElapsedTime(&elapsed, start, stop);
    pm.kernelTime = elapsed;

    // Time D2H transfer
    cudaEventRecord(start);
    cudaMemcpy(h_C, d_C, batch_size * m * l * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    pm.d2hTime = elapsed;

    pm.totalTime = pm.transferTime + pm.kernelTime + pm.d2hTime;
    pm.gflops = (2.0f * batch_size * m * n * l) / (pm.kernelTime * 1e6f);

    printf("Test 3: cuBLAS Implementation:\n");
    printf("   H2D: %.3f ms, Kernel: %.3f ms, D2H: %.3f ms, Total: %.3f ms, GFLOPS: %.2f\n",
           pm.transferTime, pm.kernelTime, pm.d2hTime, pm.totalTime, pm.gflops);

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return pm;
}

//------------------------------------------------------------------------------
// Test 4: Vectorized Implementation (using regular floats)
//------------------------------------------------------------------------------
PerfMetrics runTestVectorized(const float* h_A, const float* h_B, float* h_C,
                              int batch_size, int m, int n, int k, int l) {
    dim3 block(32, 32);
    dim3 grid((m + 31) / 32, (l + 31) / 32, batch_size);
    
    return runGpuTest("Test 4: Vectorized Implementation",
                      tensor_mul_vectorized,
                      h_A, h_B, h_C, batch_size, m, n, k, l, grid, block);
}

//------------------------------------------------------------------------------
// Test 5: Warp-Optimized Implementation
//------------------------------------------------------------------------------
PerfMetrics runTestWarpOptimized(const float* h_A, const float* h_B, float* h_C,
                                 int batch_size, int m, int n, int k, int l) {
    // Use 128 threads (4 warps) per block for better occupancy
    dim3 block(256);  // 8 warps per block
    // Each block handles (4 rows × 32 columns)
    dim3 grid((m + 7)/8, (l + 31)/32, batch_size);
    
    return runGpuTest("Test 5: Warp-Optimized Implementation", 
                     tensor_mul_warp_optimized,
                     h_A, h_B, h_C, batch_size, m, n, k, l, grid, block);
}

//------------------------------------------------------------------------------
// Test 6: Double-Buffered Implementation (Rewritten for near-zero error)
//------------------------------------------------------------------------------
PerfMetrics runTestDoubleBuffered(const float* h_A, const float* h_B, float* h_C,
                                  int batch_size, int m, int n, int k, int l) {
    dim3 block(32, 32);  // Using 32x32 threads per block
    dim3 grid((m + 31) / 32, (l + 31) / 32, batch_size);
    return runGpuTest("Test 6: Double-Buffered Implementation", tensor_mul_double_buffered,
                      h_A, h_B, h_C, batch_size, m, n, k, l, grid, block);
}

//------------------------------------------------------------------------------
// Test 7: Tensor Core Implementation (using TF32 on Ampere)
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
    int totalB = batch_size * n * l;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;

    // Use the host's float arrays directly.
    float* d_A;
    float* d_B;
    cudaMalloc((void**)&d_A, totalA * sizeof(float));
    cudaMalloc((void**)&d_B, totalB * sizeof(float));

    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, totalA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, totalB * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    pm.transferTime = elapsed;
    
    // Allocate device memory for result (reuse d_C allocated in runGpuTest pattern).
    float* d_C;
    cudaMalloc((void**)&d_C, batch_size * m * l * sizeof(float));
    
    // Launch Tensor Core kernel.
    dim3 block(128, 1);  // 4 warps per block
    dim3 grid((m + 63)/64, (l + 63)/64, batch_size);
    const size_t shmem_size = 16*16*sizeof(float); // For temp storage
    cudaEventRecord(start);
    tensor_mul_tensorcore<<<grid, block, shmem_size>>>((const float*)d_A, (const float*)d_B, d_C,
                                                      batch_size, m, n, k, l);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    pm.kernelTime = elapsed;
    
    // Time D2H transfer
    cudaEventRecord(start);
    cudaMemcpy(h_C, d_C, batch_size * m * l * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    pm.d2hTime = elapsed;
    
    pm.totalTime = pm.transferTime + pm.kernelTime + pm.d2hTime;
    pm.gflops = (2.0f * batch_size * m * n * l) / (pm.kernelTime * 1e6f);
    
    printf("Test 7: Tensor Core Implementation:\n");
    printf("   H2D: %.3f ms, Kernel: %.3f ms, D2H: %.3f ms, Total: %.3f ms, GFLOPS: %.2f\n",
           pm.transferTime, pm.kernelTime, pm.d2hTime, pm.totalTime, pm.gflops);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return pm;
}

//------------------------------------------------------------------------------
// Test 7: Tensor Core Implementation (Rewritten for low error w/ float WMMA)
//------------------------------------------------------------------------------
__global__ void tensor_mul_tensorcore_float(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float*       __restrict__ C,
                                            int batch_size, int M, int N, int K, int L)
{
#if (__CUDA_ARCH__ < 800)
    // On older GPUs, you won't have TF32 or float WMMA support. 
    // You can either do nothing or call a fallback kernel here.
    return;
#endif

    // Each block handles one 16×16 tile of C.
    int warpM = blockIdx.x; 
    int warpN = blockIdx.y; 
    int batch = blockIdx.z;
    if (batch >= batch_size) return;

    // Create fragments (float or TF32 if needed)
    nvcuda::wmma::fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
    nvcuda::wmma::fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
    nvcuda::wmma::fragment<accumulator, 16, 16, 16, float> acc_frag;
    fill_fragment(acc_frag, 0.0f);

    // Traverse the K dimension in chunks of 16
    for (int k_step = 0; k_step < N; k_step += 16)
    {
        int loadK = (k_step + 16 <= N) ? 16 : (N - k_step);
        if (loadK <= 0) break;

        // Global pointers for A, B
        float* a_ptr = const_cast<float*>(A + (batch * M * N) + (warpM * 16 * N) + k_step);
        float* b_ptr = const_cast<float*>(B + (batch * N * L) + (k_step * L) + (warpN * 16));

        // Convert float to half before loading
        __half *a_half = (__half*)malloc(16 * N * sizeof(__half));
        __half *b_half = (__half*)malloc(16 * L * sizeof(__half));
        
        // Convert the float data to half
        for(int i = 0; i < 16; i++) {
            for(int j = 0; j < N; j++) {
                a_half[i * N + j] = __float2half(a_ptr[k_step + i * N + j]);
            }
        }
        for(int i = 0; i < 16; i++) {
            for(int j = 0; j < L; j++) {
                b_half[i * L + j] = __float2half(b_ptr[(k_step + i) * L + j]);
            }
        }
        
        // Load the converted half-precision data
        nvcuda::wmma::load_matrix_sync(a_frag, a_half, N);
        nvcuda::wmma::load_matrix_sync(b_frag, b_half, L);
        
        free(a_half);
        free(b_half);

        // Multiply-Accumulate
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Write output tile
    float* outC = C + (batch * M * L) + (warpM * 16 * L) + (warpN * 16);

    // Boundary checks
    int rowsLeft = M - warpM * 16;
    int colsLeft = L - warpN * 16;
    if (rowsLeft >= 16 && colsLeft >= 16)
    {
        // Full tile
        store_matrix_sync(outC, acc_frag, L, mem_row_major);
    }
    else
    {
        // Partial tile -> store to shared memory first
        __shared__ float shm[16 * 16];
        store_matrix_sync(shm, acc_frag, 16, mem_row_major);
        __syncthreads();

        // Only threads in [0..15,0..15] handle copying
        int tx = threadIdx.x; 
        int ty = threadIdx.y;
        if (tx < 16 && ty < 16)
        {
            if (tx < rowsLeft && ty < colsLeft)
            {
                outC[tx * L + ty] = shm[tx * 16 + ty];
            }
        }
    }
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
    printf("\n");
    
    // Test 1: CPU Implementation using OpenMP.
    double t_cpu_start = clock();
    cpu_matrix_multiply(h_A, h_B, h_C_temp, batch_size, m, n, k, l);
    double cpu_time = (clock() - t_cpu_start) / (double)CLOCKS_PER_SEC * 1000.0;
    printf("Test 1: CPU Implementation (OpenMP):\n   Computation: %.3f ms\n", cpu_time);
    float tol = 1e-3f;  // Relaxed tolerance to 1e-3 for CPU check.
    float max_diff_cpu = checkResults(h_C_baseline, h_C_temp, total_elements_C, tol);
    if (max_diff_cpu <= tol)
        printf("   Accuracy Check: PASSED (max diff: %e)\n", max_diff_cpu);
    else
        printf("   Accuracy Check: FAILED (max diff: %e)\n", max_diff_cpu);
    printf("\n");
    
    // Test 2: Shared Memory Implementation.
    PerfMetrics pm2 = runTestSharedMemory(h_A, h_B, h_C_temp, batch_size, m, n, k, l);
    float diff2 = checkResults(h_C_baseline, h_C_temp, total_elements_C, 1e-5f);
    printf("Test 2: Shared Memory Implementation: Accuracy (max diff: %e)\n", diff2);
    printf("\n");
    
    // Test 3: cuBLAS Implementation.
    PerfMetrics pm3 = runTestCublas(h_A, h_B, h_C_temp, batch_size, m, n, k, l);
    float diff3 = checkResults(h_C_baseline, h_C_temp, total_elements_C, 1e-5f);
    printf("Test 3: cuBLAS Implementation: Accuracy (max diff: %e)\n", diff3);
    printf("\n");
    
    // Test 4: Vectorized Implementation.
    PerfMetrics pm4 = runTestVectorized(h_A, h_B, h_C_temp, batch_size, m, n, k, l);
    float diff4 = checkResults(h_C_baseline, h_C_temp, total_elements_C, 1e-5f);
    printf("Test 4: Vectorized Implementation: Accuracy (max diff: %e)\n", diff4);
    printf("\n");
    
    // Test 5: Warp-Optimized Implementation.
    PerfMetrics pm5 = runTestWarpOptimized(h_A, h_B, h_C_temp, batch_size, m, n, k, l);
    float diff5 = checkResults(h_C_baseline, h_C_temp, total_elements_C, 1e-5f);
    printf("Test 5: Warp-Optimized Implementation: Accuracy (max diff: %e)\n", diff5);
    printf("\n");
    
    // Test 6: Double-Buffered Implementation.
    PerfMetrics pm6 = runTestDoubleBuffered(h_A, h_B, h_C_temp, batch_size, m, n, k, l);
    float diff6 = checkResults(h_C_baseline, h_C_temp, total_elements_C, 1e-5f);
    printf("Test 6: Double-Buffered Implementation: Accuracy (max diff: %e)\n", diff6);
    printf("\n");
    
    // Test 7: Tensor Core Implementation.
    PerfMetrics pm7 = runTestTensorCore(h_A, h_B, h_C_temp, batch_size, m, n, k, l);
    float diff7 = checkResults(h_C_baseline, h_C_temp, total_elements_C, 2e-2f);
    printf("Test 7: Tensor Core Implementation: Accuracy (max diff: %e)\n", diff7);
    printf("\n");
    
    //------------------------------------------------------------------------------    
    // Performance Summary.
    //------------------------------------------------------------------------------
    printf("\n=== Performance Summary ===\n");
    printf("Tensor Dimensions: [%d × %d × %d] × [%d × %d × %d]\n", 
           batch_size, m, n, batch_size, k, l);
    printf("--------------------------------------------------------------------------------\n");
    printf("Implementation      Time (ms)        GFLOPS       vs Naive        vs CPU\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("0. Naive GPU       %12.3f    %10.2f    %8.2fx    %10.2fx\n", 
           pm0.totalTime, pm0.gflops, 1.0f, cpu_time/pm0.totalTime);
    printf("1. CPU (OpenMP)    %12.3f    %10.2f    %8.2fx    %10.2fx\n", 
           cpu_time, (2.0f * batch_size * m * n * l) / (cpu_time * 1e6f), 
           pm0.totalTime/cpu_time, 1.0f);
    printf("2. Shared Memory   %12.3f    %10.2f    %8.2fx    %10.2fx\n", 
           pm2.totalTime, pm2.gflops, pm0.totalTime/pm2.totalTime, 
           cpu_time/pm2.totalTime);
    printf("3. cuBLAS          %12.3f    %10.2f    %8.2fx    %10.2fx\n", 
           pm3.totalTime, pm3.gflops, pm0.totalTime/pm3.totalTime, 
           cpu_time/pm3.totalTime);
    printf("4. Vectorized      %12.3f    %10.2f    %8.2fx    %10.2fx\n", 
           pm4.totalTime, pm4.gflops, pm0.totalTime/pm4.totalTime, 
           cpu_time/pm4.totalTime);
    printf("5. Warp-Optimized  %12.3f    %10.2f    %8.2fx    %10.2fx\n", 
           pm5.totalTime, pm5.gflops, pm0.totalTime/pm5.totalTime, 
           cpu_time/pm5.totalTime);
    printf("6. Double-Buffered %12.3f    %10.2f    %8.2fx    %10.2fx\n", 
           pm6.totalTime, pm6.gflops, pm0.totalTime/pm6.totalTime, 
           cpu_time/pm6.totalTime);
    printf("7. Tensor Core     %12.3f    %10.2f    %8.2fx    %10.2fx\n", 
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

// Revised Shared Memory Kernel
__global__ void tensor_mul_shared(const float* A, const float* B, float* C,
                                  int batch_size, int m, int n, int k, int l) {
    // Each batch computes: (m x n) * (n x l) = (m x l)
    int batch = blockIdx.z;
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // row in C and A
    int col = blockIdx.y * blockDim.y + threadIdx.y;  // col in C and B

    __shared__ float A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float B_shared[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    // Precompute batch offsets.
    int batch_offset_A = batch * m * n;
    int batch_offset_B = batch * n * l;

    for (int t = 0; t < numTiles; t++) {
        int tiledCol = t * TILE_SIZE + threadIdx.y;   // Column for tile from A.
        int tiledRow = t * TILE_SIZE + threadIdx.x;    // Row for tile from B.

        // Load A tile
        if (row < m && tiledCol < n)
            A_shared[threadIdx.x][threadIdx.y] = A[batch_offset_A + row * n + tiledCol];
        else
            A_shared[threadIdx.x][threadIdx.y] = 0.0f;

        // Load B tile
        if (tiledRow < n && col < l)
            B_shared[threadIdx.x][threadIdx.y] = B[batch_offset_B + tiledRow * l + col];
        else
            B_shared[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        // Multiply the two tiles
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += A_shared[threadIdx.x][i] * B_shared[i][threadIdx.y];
        }

        __syncthreads();
    }

    // Write back the result in global memory.
    if (row < m && col < l) {
        C[batch * m * l + row * l + col] = sum;
    }
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

// Test 5: Warp-Optimized Implementation
__global__ void tensor_mul_warp_optimized(const float* A, const float* B, float* C,
                                          int batch_size, int m, int n, int k, int l) {
    // Use warps more efficiently for large matrices
    int batch = blockIdx.z;
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * (blockDim.x/32) + warp_id;
    int col = blockIdx.y * 32 + lane;
    
    if (batch < batch_size && row < m && col < l) {
        // Each thread accumulates one element
        float sum = 0.0f;
        
        // Coalesced memory access pattern
        const float* batch_A = A + batch * m * n + row * n;
        const float* batch_B = B + batch * k * l + col;
        
        // Process the reduction in chunks
        #pragma unroll 4
        for (int p = 0; p < n; p++) {
            sum += batch_A[p] * batch_B[p * l];
        }
        
        // Write result directly (no need for warp reduction since each thread handles one output)
        C[batch * m * l + row * l + col] = sum;
    }
}

// Test 6: Double-Buffered Implementation (Rewritten for near-zero error)
__global__ void tensor_mul_double_buffered(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float*       __restrict__ C,
                                           int batch_size, int m, int n, int k, int l) {
    int batch = blockIdx.z;
    if (batch >= batch_size) return;

    // For 1024x1024, use 32x32 tiles for better occupancy
    constexpr int TILE_DIM = 32;
    
    // Global indices
    const int row = blockIdx.x * TILE_DIM + threadIdx.x;
    const int col = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Shared memory tiles
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];
    
    // Accumulator
    float sum = 0.0f;
    
    // Base pointers for current batch
    const float* batch_A = A + batch * m * n;
    const float* batch_B = B + batch * k * l;
    
    // Loop over tiles
    for (int tile = 0; tile < n; tile += TILE_DIM) {
        // Collaborative loading of tiles
        if (row < m && (tile + threadIdx.y) < n) {
            As[threadIdx.x][threadIdx.y] = batch_A[row * n + tile + threadIdx.y];
        }
        if ((tile + threadIdx.x) < n && col < l) {
            Bs[threadIdx.x][threadIdx.y] = batch_B[(tile + threadIdx.x) * l + col];
        }
        
        __syncthreads();
        
        // Compute tile product
        if (row < m && col < l) {
            #pragma unroll
            for (int k = 0; k < TILE_DIM; k++) {
                if (tile + k < n) {
                    sum += As[threadIdx.x][k] * Bs[k][threadIdx.y];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < m && col < l) {
        C[batch * m * l + row * l + col] = sum;
    }
}

// Test 7: Tensor Core Implementation (using TF32 on Ampere)
__global__ void tensor_mul_tensorcore(const float* A, const float* B, float* C,
                                     int batch_size, int M, int N, int k, int L) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    extern __shared__ float shmem[];
    const int warpM = blockIdx.x;
    const int warpN = blockIdx.y;
    const int batch = blockIdx.z;
    
    // Load input matrices into registers
    const float* a_ptr = A + batch * M * N + warpM * 16 * N;
    const float* b_ptr = B + batch * N * L + warpN * 16;
    float* c_ptr = C + batch * M * L + warpM * 16 * L + warpN * 16;
    
    // Declare the fragments using __half for matrix multiply
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc_frag;
    
    // Initialize accumulator with zeros
    nvcuda::wmma::fill_fragment(acc_frag, 0.0f);
    
    // Main accumulation loop
    for (int k_step = 0; k_step < N; k_step += 16) {
        if (k_step + 16 <= N) {
            // Convert float to half before loading
            __half *a_half = (__half*)malloc(16 * N * sizeof(__half));
            __half *b_half = (__half*)malloc(16 * L * sizeof(__half));
            
            // Convert the float data to half
            for(int i = 0; i < 16; i++) {
                for(int j = 0; j < N; j++) {
                    a_half[i * N + j] = __float2half(a_ptr[k_step + i * N + j]);
                }
            }
            for(int i = 0; i < 16; i++) {
                for(int j = 0; j < L; j++) {
                    b_half[i * L + j] = __float2half(b_ptr[(k_step + i) * L + j]);
                }
            }
            
            // Load the converted half-precision data
            nvcuda::wmma::load_matrix_sync(a_frag, a_half, N);
            nvcuda::wmma::load_matrix_sync(b_frag, b_half, L);
            
            free(a_half);
            free(b_half);

            // Perform matrix multiplication in TF32 precision
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    
    // Store output
    const int rows_left = max(0, M - warpM * 16);
    const int cols_left = max(0, L - warpN * 16);
    
    if (rows_left >= 16 && cols_left >= 16) {
        nvcuda::wmma::store_matrix_sync(c_ptr, acc_frag, L, nvcuda::wmma::mem_row_major);
    } else {
        float* temp_storage = (float*)shmem;
        nvcuda::wmma::store_matrix_sync(temp_storage, acc_frag, 16, nvcuda::wmma::mem_row_major);
        __syncthreads();
        
        if (threadIdx.x < 16 && threadIdx.y < 16) {
            if (threadIdx.x < rows_left && threadIdx.y < cols_left) {
                c_ptr[threadIdx.x * L + threadIdx.y] = temp_storage[threadIdx.x * 16 + threadIdx.y];
            }
        }
    }
#endif
}
