#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <mma.h>  // Add this for tensor core operations
#include <cuda_runtime.h>
#include <cublas_v2.h>

// === CUDA Program Structure ===
// This file demonstrates a complete CUDA application with:
// 1. Host code (runs on CPU)
// 2. Device code (runs on GPU)
// 3. Memory management between CPU and GPU
// 4. Performance optimization techniques

// === CUDA Architecture Constants ===
// These values are optimized for the RTX 3080 Ti (Ampere Architecture)
#define TILE_SIZE 32    // Size of shared memory tiles (32x32 = 1024 threads, maximum for Ampere)
#define BLOCK_SIZE 32   // Thread block dimensions (optimal for memory coalescing)

// === GPU Memory Hierarchy (from largest/slowest to smallest/fastest) ===
// 1. Global Memory (VRAM):
//    - Several GB, high latency (400-800 cycles)
//    - Accessible by all threads
//    - Persists across kernel launches

// 2. Shared Memory:
//    - 48KB per SM, low latency (20-30 cycles)
//    - Shared within thread block
//    - Must be explicitly managed

// 3. L2 Cache:
//    - Automatic caching
//    - Shared by all SMs

// 4. L1 Cache/Registers:
//    - Fastest (1-2 cycles)
//    - Per-thread storage
//    - Managed by compiler

/**
 * Optimized CUDA kernel for tensor multiplication using shared memory tiling
 * Computes C = A × B for multiple batches of matrices in parallel
 * 
 * === Thread Hierarchy in CUDA ===
 * 1. Thread: Individual execution unit
 *    - Each thread computes one output element
 *    - Has private registers and local memory
 * 
 * 2. Warp: Group of 32 threads
 *    - Executes in SIMT fashion
 *    - All threads in warp execute same instruction
 * 
 * 3. Block: Group of warps
 *    - Shares fast shared memory
 *    - Can synchronize threads using __syncthreads()
 * 
 * 4. Grid: Array of blocks
 *    - Processes entire dataset
 *    - Blocks execute independently
 * 
 * Memory Hierarchy (from slowest to fastest):
 * 1. Global Memory (GPU VRAM)
 *    - Large but high latency
 *    - Persists across kernel launches
 * 
 * 2. Shared Memory (On-chip)
 *    - Low latency but limited size (48KB per SM on 3080 Ti)
 *    - Shared between threads in a block
 *    - Used here for tiling optimization
 * 
 * 3. Registers (On-chip)
 *    - Fastest memory
 *    - Per-thread variables stored here
 * 
 * @param A [in] Input matrix A [batch_size × m × n] in global memory
 * @param B [in] Input matrix B [batch_size × k × l] in global memory
 * @param C [out] Output matrix C [batch_size × m × l] in global memory
 * @param batch_size Number of matrix multiplications to perform
 * @param m Number of rows in matrix A and C
 * @param n Number of columns in A and rows in B
 * @param k Number of columns in B (must equal n)
 * @param l Number of columns in output matrix C
 */
__global__
void tensor_mul_optimized(float *A, float *B, float *C, int batch_size, int m, int n, int k, int l) {
    // Declare shared memory for tiles
    // Each block loads two tiles: one from A and one from B
    // Size is TILE_SIZE × TILE_SIZE for each tile
    __shared__ float As[TILE_SIZE][TILE_SIZE];  // Tile for matrix A
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];  // Tile for matrix B
    
    // Calculate thread and block indices
    // blockIdx: which block this thread belongs to
    // threadIdx: position of thread within its block
    int batch = blockIdx.z;    // Which batch this thread works on
    int bx = blockIdx.x;       // Block index in x dimension (row)
    int by = blockIdx.y;       // Block index in y dimension (col)
    int tx = threadIdx.x;      // Thread index in x dimension
    int ty = threadIdx.y;      // Thread index in y dimension
    
    // Calculate global indices for this thread
    // Each thread computes one element of the output matrix
    int row = bx * TILE_SIZE + tx;  // Global row index
    int col = by * TILE_SIZE + ty;  // Global column index
    
    // Calculate memory offsets for each batch
    // Using size_t to handle large matrices (>2^31 elements)
    // Each batch starts at batch * matrix_size offset
    size_t batch_offset_A = (size_t)batch * m * n;  // Offset for current batch in A
    size_t batch_offset_B = (size_t)batch * k * l;  // Offset for current batch in B
    size_t batch_offset_C = (size_t)batch * m * l;  // Offset for current batch in C
    
    // Accumulator for dot product
    float sum = 0.0f;
    
    // Process the matrix multiplication tile by tile
    // Each iteration processes one TILE_SIZE × TILE_SIZE portion
    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile from matrix A into shared memory
        // Check bounds to handle matrices not divisible by TILE_SIZE
        if (row < m && (tile * TILE_SIZE + ty) < n) {
            // Load element from global memory to shared memory
            // Coalesced access: adjacent threads read adjacent memory
            As[tx][ty] = A[batch_offset_A + row * n + tile * TILE_SIZE + ty];
        } else {
            As[tx][ty] = 0.0f;  // Zero padding for boundary conditions
        }
        
        // Load tile from matrix B into shared memory
        if ((tile * TILE_SIZE + tx) < k && col < l) {
            // Load element from global memory to shared memory
            // Coalesced access pattern
            Bs[tx][ty] = B[batch_offset_B + (tile * TILE_SIZE + tx) * l + col];
        } else {
            Bs[tx][ty] = 0.0f;  // Zero padding for boundary conditions
        }
        
        // Synchronize to make sure the tiles are loaded
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll  // Hint to compiler to unroll this loop
        for (int k = 0; k < TILE_SIZE; k++) {
            // Multiply and accumulate (MAC) operation
            // Each thread accesses one row of As and one column of Bs
            sum += As[tx][k] * Bs[k][ty];
        }
        
        // Synchronize before loading the next tile
        // Ensures all threads are done using the current tile
        __syncthreads();
    }
    
    // Write final result to global memory
    // Check bounds to handle edge cases
    if (row < m && col < l && batch < batch_size) {
        C[batch_offset_C + row * l + col] = sum;
    }
}

/**
 * Naive CUDA kernel for tensor multiplication (baseline implementation)
 * Simple implementation without optimizations for performance comparison
 * 
 * Memory Access Pattern:
 * - Direct global memory access for all operations
 * - No use of shared memory or caching
 * - High memory bandwidth usage due to repeated access
 *
 * Thread Organization:
 * - One thread per output element
 * - 3D grid for handling multiple batches:
 *   * x-dimension: rows
 *   * y-dimension: columns
 *   * z-dimension: batches
 *
 * Performance Characteristics:
 * - High global memory traffic
 * - No data reuse
 * - Memory access not coalesced
 * - Used as baseline for performance comparison
 *
 * @param A [in] Input matrix A [batch_size × m × n]
 * @param B [in] Input matrix B [batch_size × k × l]
 * @param C [out] Output matrix C [batch_size × m × l]
 * @param batch_size Number of matrix multiplications to perform
 * @param m Number of rows in matrix A and C
 * @param n Number of columns in A and rows in B
 * @param k Number of columns in B (must equal n)
 * @param l Number of columns in output matrix C
 */
__global__
void tensor_mul(float *A, float *B, float *C, int batch_size, int m, int n, int k, int l) {
    // Calculate global thread indices
    // Each thread computes one element of the output
    int batch = blockIdx.z;    // Batch index
    // Global row and column indices
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // Global row in output
    int col = blockIdx.y * blockDim.y + threadIdx.y;  // Global column in output

    // Early exit if thread is out of bounds
    // Prevents invalid memory access and ensures correctness
    if (row >= m || col >= l || batch >= batch_size) return;

    // Calculate batch offsets for each matrix
    // Using size_t to handle large matrices
    size_t batch_offset_A = (size_t)batch * m * n;  // Offset in A
    size_t batch_offset_B = (size_t)batch * k * l;  // Offset in B
    size_t batch_offset_C = (size_t)batch * m * l;  // Offset in C

    // Compute dot product directly from global memory
    // This is inefficient due to:
    // 1. Repeated global memory access
    // 2. No data reuse
    // 3. Non-coalesced memory access pattern
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        // Load elements from A and B
        // A[batch][row][i] * B[batch][i][col]
        float a = A[batch_offset_A + row * n + i];         // Element from A
        float b = B[batch_offset_B + i * l + col];         // Element from B
        sum += a * b;  // Multiply and accumulate
    }

    // Write result to global memory
    // C[batch][row][col] = sum
    C[batch_offset_C + row * l + col] = sum;
}

/**
 * Main function to demonstrate and compare different tensor multiplication implementations
 * 
 * This program demonstrates three key CUDA concepts:
 * 1. Basic CUDA Programming:
 *    - Memory allocation (CPU vs GPU)
 *    - Data transfer (Host to Device, Device to Host)
 *    - Kernel launches
 * 
 * 2. CUDA Optimization Techniques:
 *    - Shared memory usage
 *    - Memory coalescing
 *    - Stream processing
 * 
 * 3. Professional GPU Programming:
 *    - cuBLAS library usage
 *    - Performance measurement
 *    - Error handling
 */
int main(int argc, char **argv) {
    // === Command Line Arguments ===
    // Matrix dimensions must be provided at runtime for flexibility
    // Format: batch_size m n k l
    // Example: ./tensor_mul 8 1024 1024 1024 1024
    if (argc != 6) {
        printf("Usage: ./a.out <batch_size> <m> <n> <k> <l>\n");
        printf("For tensor multiplication: (batch_size × m × n) * (batch_size × k × l)\n");
        return 1;
    }

    // Parse dimensions from command line
    // These determine the size of our computation
    int batch_size = atoi(argv[1]);  // Number of matrix multiplications
    int m = atoi(argv[2]);           // Rows in matrix A
    int n = atoi(argv[3]);           // Cols in A, must equal rows in B
    int k = atoi(argv[4]);           // Cols in matrix B
    int l = atoi(argv[5]);           // Cols in output matrix

    // === Memory Size Calculations ===
    // Calculate total elements needed for each matrix
    // Using size_t to handle large matrices (>2GB)
    size_t total_elements_A = batch_size * m * n;
    size_t total_elements_B = batch_size * k * l;
    size_t total_elements_C = batch_size * m * l;
    size_t total_bytes = (total_elements_A + total_elements_B + total_elements_C) * sizeof(float);

    // === Grid and Block Configuration ===
    // CUDA kernels are launched with a grid of thread blocks
    // Each thread processes one output element
    
    // For naive kernel: 16x16 thread blocks (256 threads)
    dim3 originalBlock(16, 16);
    dim3 originalGrid(
        (m + originalBlock.x - 1) / originalBlock.x,  // Ceil(m/16) blocks in x
        (l + originalBlock.y - 1) / originalBlock.y,  // Ceil(l/16) blocks in y
        batch_size                                    // One z-block per batch
    );

    // For optimized kernel: 32x32 thread blocks (1024 threads)
    // Maximum threads per block on RTX 3080 Ti
    dim3 optimizedBlock(32, 32);
    dim3 optimizedGrid(
        (m + 31) / 32,  // Ceil(m/32) blocks in x
        (l + 31) / 32,  // Ceil(l/32) blocks in y
        batch_size      // One z-block per batch
    );

    // === cuBLAS Setup ===
    // cuBLAS is NVIDIA's optimized BLAS library
    // It provides highly optimized matrix operations
    cublasHandle_t handle;
    cublasCreate(&handle);
    // Use maximum precision mode for accuracy
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

    // === CUDA Event Creation ===
    // Events are used for precise GPU timing
    // They are more accurate than CPU timers for GPU operations
    cudaEvent_t start, stop;
    float original_time, optimized_time, tc_time;

    // === Memory Pointers and Allocation ===
    float *h_A, *h_B, *h_C, *h_C_original;  // Host pointers
    float *d_A, *d_B, *d_C;                 // Device pointers

    // Allocate host memory with error checking
    h_A = (float *)malloc(total_elements_A * sizeof(float));
    h_B = (float *)malloc(total_elements_B * sizeof(float));
    h_C = (float *)malloc(total_elements_C * sizeof(float));
    h_C_original = (float *)malloc(total_elements_C * sizeof(float));

    // Check all host allocations
    if (h_A == NULL || h_B == NULL || h_C == NULL || h_C_original == NULL) {
        printf("Error: Failed to allocate host memory\n");
        // Clean up any successful allocations
        if (h_A) free(h_A);
        if (h_B) free(h_B);
        if (h_C) free(h_C);
        if (h_C_original) free(h_C_original);
        return 1;
    }

    // Initialize matrices with random values
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                h_A[b * m * n + i * n + j] = (float)(rand() % 100) / 100.0f;
            }
        }
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < l; j++) {
                h_B[b * k * l + i * l + j] = (float)(rand() % 100) / 100.0f;
            }
        }
    }

    // Validate input parameters
    if (batch_size <= 0 || m <= 0 || n <= 0 || k <= 0 || l <= 0) {
        printf("Error: All dimensions must be positive integers\n");
        return 1;
    }

    // Check if we have enough system memory
    size_t total_host_memory = (total_elements_A + total_elements_B + 
                               total_elements_C * 2) * sizeof(float);  // *2 for original copy
    printf("\n=== Memory Requirements ===\n");
    printf("Host Memory Required: %.2f MB\n", total_host_memory / (1024.0 * 1024.0));
    printf("GPU Memory Required: %.2f MB\n", total_bytes / (1024.0 * 1024.0));

    if (total_host_memory > 16ULL * 1024 * 1024 * 1024) {  // 16GB limit check
        printf("Warning: Required host memory (%.2f GB) might exceed system memory\n",
               total_host_memory / (1024.0 * 1024.0 * 1024.0));
        printf("Continue? (y/n): ");
        char response;
        if (scanf(" %c", &response) != 1) {
            printf("Error reading response\n");
            return 1;
        }
        if (response != 'y' && response != 'Y') {
            return 0;
        }
    }

    // Allocate GPU memory with error checking
    cudaError_t err;
    err = cudaMalloc((void **)&d_A, total_elements_A * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for A: %s\n", 
                cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc((void **)&d_B, total_elements_B * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for B: %s\n", 
                cudaGetErrorString(err));
        cudaFree(d_A);
        return 1;
    }

    err = cudaMalloc((void **)&d_C, total_elements_C * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for C: %s\n", 
                cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return 1;
    }

    // Configure cache for optimal performance
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 4 * 1024 * 1024);  // 4MB L2 cache
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    // === Copy Data to GPU and Initial Processing ===
    // First, copy matrices A and B to GPU
    cudaMemcpy(d_A, h_A, total_elements_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, total_elements_B * sizeof(float), cudaMemcpyHostToDevice);

    // === CUDA Streams Setup ===
    // CUDA streams allow overlapping operations (compute + memory transfers)
    const int NUM_STREAMS = 4;  // Using 4 streams for parallelism
    cudaStream_t streams[NUM_STREAMS];
    // Create streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Calculate how many batches each stream should handle
    int batches_per_stream = (batch_size + NUM_STREAMS - 1) / NUM_STREAMS;
    
    // === Process Initial Batches Using Streams ===
    for (int i = 0; i < NUM_STREAMS; i++) {
        // Calculate batch range for this stream
        int stream_batch_start = i * batches_per_stream;
        int stream_batch_end = min((i + 1) * batches_per_stream, batch_size);
        int stream_batches = stream_batch_end - stream_batch_start;
        
        if (stream_batches <= 0) continue;  // Skip if no work for this stream
        
        // Calculate memory offsets for this stream's batch
        size_t stream_offset = stream_batch_start * m * n;
        size_t stream_size = stream_batches * m * n * sizeof(float);
        
        // Copy data asynchronously (non-blocking)
        cudaMemcpyAsync(d_A + stream_offset, h_A + stream_offset,
                       stream_size, cudaMemcpyHostToDevice, streams[i]);
        
        // Configure grid for this stream's batch
        dim3 streamGrid(
            (m + BLOCK_SIZE - 1) / BLOCK_SIZE,  // Ceil(m/BLOCK_SIZE)
            (l + BLOCK_SIZE - 1) / BLOCK_SIZE,  // Ceil(l/BLOCK_SIZE)
            stream_batches                       // Number of matrices for this stream
        );
        
        // Launch kernel in this stream
        // Each stream processes its portion of batches independently
        tensor_mul_optimized<<<streamGrid, optimizedBlock, 0, streams[i]>>>(
            d_A + stream_offset,    // Start of this stream's input A
            d_B + stream_offset,    // Start of this stream's input B
            d_C + stream_offset,    // Where this stream should write output
            stream_batches,         // How many matrices this stream processes
            m, n, k, l             // Matrix dimensions remain the same
        );
    }
    
    // Wait for all streams to finish
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // === Performance Testing Section ===
    printf("\n=== Performance Comparison ===\n");
    printf("Matrix Size per Batch: %d x %d\n", m, n);
    printf("Number of Batches: %d\n", batch_size);
    printf("----------------------------------------\n");

    // === Test 1: Naive Implementation ===
    // This is our baseline implementation
    // - Each thread reads directly from global memory
    // - No optimization techniques used
    // - Helps us understand the importance of optimizations
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // === Memory Transfer (Host to Device) ===
    // Copy input matrices from CPU to GPU
    // This is a blocking operation - CPU waits for completion
    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, total_elements_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, total_elements_B * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float transfer_time;
    cudaEventElapsedTime(&transfer_time, start, stop);
    printf("1. Naive Implementation:\n");
    printf("   Memory Transfer Time (H2D): %.3f ms\n", transfer_time);
    
    // === Kernel Execution ===
    // Launch naive kernel with 16x16 thread blocks
    // Each thread computes one element of output matrix
    cudaEventRecord(start);
    tensor_mul<<<originalGrid, originalBlock>>>(d_A, d_B, d_C, batch_size, m, n, k, l);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&original_time, start, stop);
    printf("   Computation Time: %.3f ms\n", original_time);
    
    // === Memory Transfer (Device to Host) ===
    // Copy results back to CPU for verification
    cudaEventRecord(start);
    cudaMemcpy(h_C, d_C, total_elements_C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float result_time;
    cudaEventElapsedTime(&result_time, start, stop);
    printf("   Memory Transfer Time (D2H): %.3f ms\n", result_time);
    printf("   Total Time: %.3f ms\n", transfer_time + original_time + result_time);
    // Calculate TFLOPS (Trillion Floating Point Operations per Second)
    // Each multiply-add is 2 operations, hence the 2.0 multiplier
    printf("   TFLOPS: %.2f\n", (2.0 * batch_size * m * n * l) / (original_time * 1000000000.0));
    
    // Save results for accuracy comparison
    memcpy(h_C_original, h_C, total_elements_C * sizeof(float));
    
    // === Test 2: Shared Memory Implementation ===
    // This version uses shared memory to reduce global memory access
    // Each thread block loads a tile of input matrices into shared memory
    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, total_elements_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, total_elements_B * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&transfer_time, start, stop);
    printf("\n2. Shared Memory Implementation:\n");
    printf("   Memory Transfer Time (H2D): %.3f ms\n", transfer_time);
    
    // Launch optimized kernel with 32x32 thread blocks
    // Uses shared memory for better performance
    cudaEventRecord(start);
    tensor_mul_optimized<<<optimizedGrid, optimizedBlock>>>(d_A, d_B, d_C, batch_size, m, n, k, l);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&optimized_time, start, stop);
    printf("   Computation Time: %.3f ms\n", optimized_time);
    
    // Time result transfer and check accuracy
    cudaEventRecord(start);
    cudaMemcpy(h_C, d_C, total_elements_C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result_time, start, stop);
    printf("   Memory Transfer Time (D2H): %.3f ms\n", result_time);
    printf("   Total Time: %.3f ms\n", transfer_time + optimized_time + result_time);
    printf("   TFLOPS: %.2f\n", (2.0 * batch_size * m * n * l) / (optimized_time * 1000000000.0));
    printf("   Speedup vs Naive: %.2fx\n", original_time / optimized_time);
    
    // Check accuracy against naive implementation
    bool shared_mem_matches = true;
    float shared_mem_max_diff = 0.0f;
    for (size_t i = 0; i < total_elements_C; i++) {
        float diff = fabs(h_C[i] - h_C_original[i]);
        shared_mem_max_diff = max(shared_mem_max_diff, diff);
        if (diff > 1e-5) {
            shared_mem_matches = false;
            break;
        }
    }
    printf("   Accuracy Check: %s (max diff: %e)\n", 
           shared_mem_matches ? "PASSED" : "FAILED", shared_mem_max_diff);
    
    // === Test 3: cuBLAS Implementation ===
    // cuBLAS provides highly optimized implementations of BLAS operations
    // Advantages:
    // 1. Automatically uses hardware features (Tensor Cores)
    // 2. Optimized memory access patterns
    // 3. Tuned for specific GPU architectures
    
    // Time memory transfer
    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, total_elements_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, total_elements_B * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&transfer_time, start, stop);
    printf("\n3. cuBLAS Implementation:\n");
    printf("   Memory Transfer Time (H2D): %.3f ms\n", transfer_time);
    
    // Time computation
    cudaEventRecord(start);
    
    // === Maximum Accuracy Configuration ===
    // Use default math mode but with correct matrix layout
    cublasStatus_t status;
    status = cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Failed to set math mode: %d\n", status);
        return 1;
    }
    
    // 2. Force device synchronization before computation
    cudaDeviceSynchronize();

    // Batch parameters for strided GEMM
    long long int strideA = (long long int)m * n;
    long long int strideB = (long long int)k * l;
    long long int strideC = (long long int)m * l;

    // === Maximum Precision GEMM Parameters ===
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    status = cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,                // No transpose for A
        CUBLAS_OP_N,                // No transpose for B
        l, m, k,                    // Dimensions swapped for column-major
        &alpha,
        d_B, l,                     // Matrix B first
        strideB,
        d_A, k,                     // Matrix A second
        strideA,
        &beta,
        d_C, m,
        strideC,
        batch_size
    );
    
    // 3. Force synchronization after computation
    cudaDeviceSynchronize();

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Failed to execute batched GEMM: %d\n", status);
        return 1;
    }

    // 3. Force synchronization after computation
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tc_time, start, stop);
    printf("   Computation Time: %.3f ms\n", tc_time);
    
    // Time result transfer and check accuracy
    cudaEventRecord(start);
    cudaMemcpy(h_C, d_C, total_elements_C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&result_time, start, stop);
    printf("   Memory Transfer Time (D2H): %.3f ms\n", result_time);
    printf("   Total Time: %.3f ms\n", transfer_time + tc_time + result_time);
    printf("   TFLOPS: %.2f\n", (2.0 * batch_size * m * n * l) / (tc_time * 1000000000.0));
    printf("   Speedup vs Naive: %.2fx\n", original_time / tc_time);
    printf("   Speedup vs Shared Memory: %.2fx\n", optimized_time / tc_time);

    // Check accuracy against naive implementation
    bool cublas_matches = true;
    float cublas_max_diff = 0.0f;
    for (size_t i = 0; i < total_elements_C; i++) {
        float diff = fabs(h_C[i] - h_C_original[i]);
        cublas_max_diff = max(cublas_max_diff, diff);
        if (diff > 1e-5) {
            cublas_matches = false;
            break;
        }
    }
    printf("   Accuracy Check: %s (max diff: %e)\n", 
           cublas_matches ? "PASSED" : "FAILED", cublas_max_diff);
    
    // === Resource Cleanup Strategy ===
    // CUDA requires explicit cleanup of all allocated resources
    // Order matters: destroy dependent resources first
    
    // 1. Streams: Allow concurrent operations
    // Must be destroyed after all operations complete
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }

    // 2. CUDA Events: Used for performance timing
    // Can be destroyed after timing is complete
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // 3. GPU Memory: Must be explicitly freed
    // Failure to free GPU memory leads to memory leaks
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // 4. CPU Memory: Standard C cleanup
    // Free host memory last
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_original);

    return 0;
} 