//------------------------------------------------------------------------------
// Compile with:
// nvcc -O3 -arch=sm_86 --use_fast_math -Xcompiler "-fopenmp -fPIC -pthread -march=native" tensor_mul.cu -o tensor_mul -lcublas
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// Include standard C and CUDA headers with detailed explanations.
//------------------------------------------------------------------------------
#include <stdio.h>                 // Include the standard I/O library for functions like printf() and scanf().
#include <cuda.h>                  // Include the CUDA Driver API header for low-level CUDA functionality.
#include <time.h>                  // Include the C time header for time-related functions (e.g., seeding random numbers).
#include <cooperative_groups.h>    // Include the CUDA cooperative groups header for advanced thread grouping and synchronization.
#include <cuda_fp16.h>             // Include the header for half-precision (16-bit) floating point support in CUDA.
#include <mma.h>                   // Include the header for matrix multiply-accumulate (MMA) operations (used with Tensor Cores).
#include <cuda_runtime.h>          // Include the CUDA Runtime API header for simplified memory management and kernel launching.
#include <cublas_v2.h>             // Include the cuBLAS header for GPU-accelerated BLAS routines.
#include <omp.h>                   // Include OpenMP for CPU parallelization

using namespace nvcuda;

//------------------------------------------------------------------------------
// Define architecture-specific constants and macros for optimization.
//------------------------------------------------------------------------------
#define TILE_SIZE 32               // Define TILE_SIZE as 32; used for shared memory tiling (tile dimension: 32x32).
#define BLOCK_SIZE 32              // Define BLOCK_SIZE as 32; used for the dimensions of thread blocks in optimized kernels.

//------------------------------------------------------------------------------
// GPU Memory Hierarchy Overview (provided as comments for reference):
//   1. Global Memory:
//      - Large capacity (GBs) but high latency (400-800 clock cycles).
//      - Accessible by all threads; persists across kernel launches.
//   2. Shared Memory:
//      - Limited capacity (e.g., 48KB per SM on RTX 3080 Ti) but very low latency (20-30 cycles).
//      - Shared among threads within a block; must be managed explicitly by the programmer.
//   3. L2 Cache:
//      - Automatically caches data from global memory and is shared by all SMs.
//   4. L1 Cache/Registers:
//      - Fastest memory (1-2 clock cycles); registers are private per thread, L1 is shared per block.
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Optimized CUDA kernel for tensor multiplication using shared memory tiling.
// This kernel computes C = A × B for a batch of matrices.
// Each thread computes one element of the output matrix C.
//------------------------------------------------------------------------------
__global__
void tensor_mul_optimized(float *A, float *B, float *C, int batch_size, int m, int n, int k, int l) {  // Kernel definition with parameters.
    __shared__ float As[TILE_SIZE][TILE_SIZE];  // Declare shared memory tile for matrix A.
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];    // Declare shared memory tile for matrix B.
    
    int batch = blockIdx.z;        // Determine the current batch index from grid dimension z.
    int bx = blockIdx.x;           // Determine the block index in the x-dimension.
    int by = blockIdx.y;           // Determine the block index in the y-dimension.
    int tx = threadIdx.x;          // Determine the thread index in the x-dimension within the block.
    int ty = threadIdx.y;          // Determine the thread index in the y-dimension within the block.
    
    int row = bx * TILE_SIZE + tx; // Calculate the global row index for the output element.
    int col = by * TILE_SIZE + ty; // Calculate the global column index for the output element.
    
    size_t batch_offset_A = (size_t)batch * m * n;  // Compute the starting offset for matrix A for this batch.
    size_t batch_offset_B = (size_t)batch * k * l;  // Compute the starting offset for matrix B for this batch.
    size_t batch_offset_C = (size_t)batch * m * l;  // Compute the starting offset for matrix C for this batch.
    
    float sum = 0.0f;              // Initialize the accumulator variable for the dot product.
    
    // Loop over the tiles along the shared dimension.
    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; tile++) {  // Loop over tiles; the division rounds up.
        if (row < m && (tile * TILE_SIZE + ty) < n) {  // Check bounds for matrix A.
            As[tx][ty] = A[batch_offset_A + row * n + tile * TILE_SIZE + ty];  // Load element from A into shared memory tile.
        } else {  // If out of bounds...
            As[tx][ty] = 0.0f;  // ...store zero to pad the tile.
        }
        
        if ((tile * TILE_SIZE + tx) < k && col < l) {  // Check bounds for matrix B.
            Bs[tx][ty] = B[batch_offset_B + (tile * TILE_SIZE + tx) * l + col];  // Load element from B into shared memory tile.
        } else {  // If out of bounds...
            Bs[tx][ty] = 0.0f;  // ...store zero.
        }
        
        __syncthreads();         // Synchronize threads in the block to ensure the tiles are fully loaded.
        
        #pragma unroll           // Hint to the compiler to unroll the following loop for efficiency.
        for (int k = 0; k < TILE_SIZE; k++) {  // Loop over elements in the tile to compute partial dot product.
            sum += As[tx][k] * Bs[k][ty];     // Multiply corresponding elements and accumulate the sum.
        }
        
        __syncthreads();         // Synchronize again before loading the next tile.
    }
    
    if (row < m && col < l && batch < batch_size) {  // Check that computed indices are within valid bounds.
        C[batch_offset_C + row * l + col] = sum;       // Write the computed dot product to the output matrix.
    }
}  // End of tensor_mul_optimized kernel.

//------------------------------------------------------------------------------
// Naive CUDA kernel for tensor multiplication (baseline implementation).
// This kernel reads directly from global memory without any optimizations.
//------------------------------------------------------------------------------
__global__
void tensor_mul(float *A, float *B, float *C, int batch_size, int m, int n, int k, int l) {  // Kernel definition.
    int batch = blockIdx.z;        // Get the current batch index.
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // Calculate global row index for output.
    int col = blockIdx.y * blockDim.y + threadIdx.y;  // Calculate global column index for output.
    
    if (row >= m || col >= l || batch >= batch_size) return;  // Exit thread if indices are out-of-bounds.
    
    size_t batch_offset_A = (size_t)batch * m * n;  // Compute starting offset for matrix A for this batch.
    size_t batch_offset_B = (size_t)batch * k * l;  // Compute starting offset for matrix B.
    size_t batch_offset_C = (size_t)batch * m * l;    // Compute starting offset for matrix C.
    
    float sum = 0.0f;              // Initialize accumulator for the dot product.
    
    for (int i = 0; i < n; i++) {  // Loop over the shared dimension.
        float a = A[batch_offset_A + row * n + i];  // Load element from matrix A.
        float b = B[batch_offset_B + i * l + col];  // Load corresponding element from matrix B.
        sum += a * b;            // Multiply and accumulate.
    }
    
    C[batch_offset_C + row * l + col] = sum;  // Write the result to the output matrix.
}  // End of tensor_mul kernel.

//------------------------------------------------------------------------------
// Vectorized CUDA kernel using float4 for coalesced memory access.
// This kernel loads/stores 4 elements at a time using the float4 type.
//------------------------------------------------------------------------------
__global__
void tensor_mul_vectorized(float4 *A, float4 *B, float4 *C, int batch_size, int m, int n, int k, int l) {  // Kernel definition.
    __shared__ float As[TILE_SIZE][TILE_SIZE];  // Declare shared memory tile for matrix A.
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];    // Declare shared memory tile for matrix B.
    
    int batch = blockIdx.z;        // Get the current batch index.
    int bx = blockIdx.x;           // Get block index in the x-dimension.
    int by = blockIdx.y;           // Get block index in the y-dimension.
    int tx = threadIdx.x;          // Get thread index in x-dimension.
    int ty = threadIdx.y;          // Get thread index in y-dimension.
    
    int row = bx * TILE_SIZE + tx; // Calculate global row index.
    int col = by * TILE_SIZE + ty; // Calculate global column index.
    
    size_t batch_offset_A = (size_t)batch * m * n;  // Compute starting offset for matrix A.
    size_t batch_offset_B = (size_t)batch * k * l;    // Compute starting offset for matrix B.
    size_t batch_offset_C = (size_t)batch * m * l;    // Compute starting offset for matrix C.
    
    float sum = 0.0f;              // Initialize the accumulator.
    
    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; tile++) {  // Loop over tiles.
        if (row < m && (tile * TILE_SIZE + ty) < n) {  // Check bounds for A.
            float4 a4 = A[batch_offset_A/4 + row * (n/4) + (tile * TILE_SIZE + ty)/4];  // Load a float4 from matrix A.
            As[tx][ty] = (ty % 4 == 0) ? a4.x : (ty % 4 == 1) ? a4.y : (ty % 4 == 2) ? a4.z : a4.w;  // Extract the correct float.
        } else {  // Out-of-bound case.
            As[tx][ty] = 0.0f;  // Pad with zero.
        }
        
        if ((tile * TILE_SIZE + tx) < k && col < l) {  // Check bounds for B.
            float4 b4 = B[batch_offset_B/4 + (tile * TILE_SIZE + tx) * (l/4) + col/4];  // Load a float4 from matrix B.
            Bs[tx][ty] = (col % 4 == 0) ? b4.x : (col % 4 == 1) ? b4.y : (col % 4 == 2) ? b4.z : b4.w;  // Extract the correct float.
        } else {  // Out-of-bound case.
            Bs[tx][ty] = 0.0f;  // Pad with zero.
        }
        
        __syncthreads();         // Synchronize threads before computing.
        
        #pragma unroll           // Unroll the inner loop.
        for (int k = 0; k < TILE_SIZE; k++) {  // Loop over elements in the tile.
            sum += As[tx][k] * Bs[k][ty];     // Multiply and accumulate.
        }
        
        __syncthreads();         // Synchronize before the next tile.
    }
    
    if (row < m && col < l && batch < batch_size) {  // Check bounds for output.
        int c_idx = batch_offset_C/4 + row * (l/4) + col/4;  // Calculate the index in terms of float4.
        float4 c4 = C[c_idx];   // Load the current float4 from output.
        if (col % 4 == 0) c4.x = sum;  // Store the computed sum in the correct component.
        else if (col % 4 == 1) c4.y = sum;
        else if (col % 4 == 2) c4.z = sum;
        else c4.w = sum;
        C[c_idx] = c4;          // Write the updated float4 back to global memory.
    }
}  // End of tensor_mul_vectorized kernel.

//------------------------------------------------------------------------------
// Warp-Optimized Kernel Using Warp Shuffle.
// This kernel uses warp-level primitives to reduce shared memory usage.
//------------------------------------------------------------------------------
__global__
void tensor_mul_warp_optimized(float *A, float *B, float *C, int batch_size, int m, int n, int k, int l) {  // Kernel definition.
    __shared__ float As[32][32];  // Declare a fixed-size shared memory tile for A (32x32).
    __shared__ float Bs[32][32];  // Declare a fixed-size shared memory tile for B (32x32).
    
    int batch = blockIdx.z;        // Get the current batch index.
    int warpId = threadIdx.y;      // Use threadIdx.y as the warp ID within the block.
    int lane = threadIdx.x;        // Use threadIdx.x as the lane (position) within the warp.
    
    int row = blockIdx.x * 32 + warpId;  // Calculate the global row index.
    int col = blockIdx.y * 32 + lane;      // Calculate the global column index.
    
    size_t batch_offset_A = (size_t)batch * m * n;  // Compute starting offset for matrix A.
    size_t batch_offset_B = (size_t)batch * k * l;    // Compute starting offset for matrix B.
    size_t batch_offset_C = (size_t)batch * m * l;    // Compute starting offset for matrix C.
    
    float sum = 0.0f;              // Initialize the accumulator.
    
    for (int tile = 0; tile < (n + 31) / 32; tile++) {  // Loop over tiles; using 32 as tile size.
        if (row < m && (tile * 32 + lane) < n) {  // Check bounds for A.
            As[warpId][lane] = A[batch_offset_A + row * n + tile * 32 + lane];  // Load element into shared memory tile.
        } else {  // Out-of-bound.
            As[warpId][lane] = 0.0f;  // Pad with zero.
        }
        
        if ((tile * 32 + warpId) < k && col < l) {  // Check bounds for B.
            Bs[warpId][lane] = B[batch_offset_B + (tile * 32 + warpId) * l + col];  // Load element into shared memory tile.
        } else {  // Out-of-bound.
            Bs[warpId][lane] = 0.0f;  // Pad with zero.
        }
        
        __syncthreads();         // Synchronize threads to ensure full tile load.
        
        #pragma unroll           // Unroll the inner loop.
        for (int k = 0; k < 32; k++) {  // Loop over the tile dimension (32 elements).
            sum += As[warpId][k] * Bs[k][lane];  // Multiply and accumulate.
        }
        
        __syncthreads();         // Synchronize threads before loading the next tile.
    }
    
    if (row < m && col < l && batch < batch_size) {  // Check that indices are valid.
        C[batch_offset_C + row * l + col] = sum;       // Write the result to the output matrix.
    }
}  // End of tensor_mul_warp_optimized kernel.

//------------------------------------------------------------------------------
// Double-Buffered Kernel that overlaps computation with memory loads.
// Uses two sets of shared memory buffers (ping-pong buffers) to hide global memory latency.
//------------------------------------------------------------------------------
__global__
void tensor_mul_double_buffered(float *A, float *B, float *C, int batch_size, int m, int n, int k, int l) {  // Kernel definition.
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];  // Declare two shared memory buffers for A.
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE];    // Declare two shared memory buffers for B.
    
    int batch = blockIdx.z;        // Get the current batch index.
    int bx = blockIdx.x;           // Get the block index in the x-dimension.
    int by = blockIdx.y;           // Get the block index in the y-dimension.
    int tx = threadIdx.x;          // Get the thread index in the x-dimension.
    int ty = threadIdx.y;          // Get the thread index in the y-dimension.
    
    int row = bx * TILE_SIZE + tx; // Calculate global row index.
    int col = by * TILE_SIZE + ty; // Calculate global column index.
    
    size_t batch_offset_A = (size_t)batch * m * n;  // Compute starting offset for matrix A.
    size_t batch_offset_B = (size_t)batch * k * l;    // Compute starting offset for matrix B.
    size_t batch_offset_C = (size_t)batch * m * l;    // Compute starting offset for matrix C.
    
    float sum = 0.0f;              // Initialize the accumulator.
    int buf = 0;                   // Initialize buffer selector (0 means buffer 0 is active).
    
    // Initial load: Load the first tile into buffer 0.
    if (row < m && ty < n) {  // Check bounds for A.
        As[0][tx][ty] = A[batch_offset_A + row * n + ty];  // Load element from A into buffer 0.
    }
    if (tx < k && col < l) {  // Check bounds for B.
        Bs[0][tx][ty] = B[batch_offset_B + tx * l + col];  // Load element from B into buffer 0.
    }
    
    __syncthreads();         // Synchronize to ensure initial load is complete.
    
    // Main loop: Process each subsequent tile.
    for (int tile = 1; tile < (n + TILE_SIZE - 1) / TILE_SIZE; tile++) {  // Loop over remaining tiles.
        if (row < m && (tile * TILE_SIZE + ty) < n) {  // Check bounds for A.
            As[1 - buf][tx][ty] = A[batch_offset_A + row * n + tile * TILE_SIZE + ty];  // Load next tile into inactive buffer.
        }
        if ((tile * TILE_SIZE + tx) < k && col < l) {  // Check bounds for B.
            Bs[1 - buf][tx][ty] = B[batch_offset_B + (tile * TILE_SIZE + tx) * l + col];  // Load next tile into inactive buffer.
        }
        
        #pragma unroll           // Unroll loop for computation.
        for (int k = 0; k < TILE_SIZE; k++) {  // Loop over elements in the current tile.
            sum += As[buf][tx][k] * Bs[buf][k][ty];  // Compute partial dot product using active buffer.
        }
        
        buf = 1 - buf;           // Swap buffers (ping-pong): active buffer becomes inactive.
        __syncthreads();         // Synchronize to ensure new tile load is complete before next iteration.
    }
    
    #pragma unroll               // Unroll loop for final computation.
    for (int k = 0; k < TILE_SIZE; k++) {  // Process the last loaded tile.
        sum += As[buf][tx][k] * Bs[buf][k][ty];  // Final accumulation.
    }
    
    if (row < m && col < l && batch < batch_size) {  // Check bounds for output.
        C[batch_offset_C + row * l + col] = sum;       // Write the final result to the output matrix.
    }
}  // End of tensor_mul_double_buffered kernel.

//------------------------------------------------------------------------------
// Tensor Core Implementation using WMMA API
// This kernel leverages Tensor Cores for high-performance matrix multiplication
// - Uses FP16 (half precision) for input matrices
// - Outputs in FP32 (single precision)
// - Operates on 16x16x16 matrix fragments
//------------------------------------------------------------------------------
__global__ void tensor_mul_tensorcore(half *A, half *B, float *C, 
                                    int batch_size, int m, int n, int k, int l) {
    // WMMA operates on fixed-size 16x16x16 matrices for optimal tensor core utilization
    const int WMMA_M = 16;  // Height of the matrix multiplication
    const int WMMA_N = 16;  // Width of the matrix multiplication
    const int WMMA_K = 16;  // Size of inner dimension
    
    // Declare matrix fragments for tensor operations
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, 
                  wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                  wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Calculate thread and block indices
    int batch = blockIdx.z;        // Current batch being processed
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;  // Warp's row position
    int warpN = blockIdx.y * blockDim.y + threadIdx.y;               // Warp's column position
    
    // Calculate memory offsets for each batch
    size_t batch_offset_A = (size_t)batch * m * n;  // Offset for matrix A
    size_t batch_offset_B = (size_t)batch * k * l;  // Offset for matrix B
    size_t batch_offset_C = (size_t)batch * m * l;  // Offset for output matrix C
    
    // Initialize accumulator fragment to zeros
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Process matrix multiplication in 16x16x16 tiles
    for (int i = 0; i < n; i += WMMA_K) {
        // Calculate current tile positions
        int aRow = warpM * WMMA_M;  // Starting row in matrix A
        int aCol = i;               // Starting column in matrix A
        int bRow = i;               // Starting row in matrix B
        int bCol = warpN * WMMA_N;  // Starting column in matrix B
        
        // Check if current tile is within matrix bounds
        if (aRow < m && aCol < n && bRow < k && bCol < l) {
            // Load matrix fragments from global memory
            wmma::load_matrix_sync(a_frag, A + batch_offset_A + aRow * n + aCol, n);
            wmma::load_matrix_sync(b_frag, B + batch_offset_B + bRow * l + bCol, l);
            
            // Perform matrix multiplication on fragments using tensor cores
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    
    // Write results back to global memory
    int cRow = warpM * WMMA_M;  // Output row position
    int cCol = warpN * WMMA_N;  // Output column position
    if (cRow < m && cCol < l) {
        // Store accumulated results back to global memory
        wmma::store_matrix_sync(C + batch_offset_C + cRow * l + cCol, c_frag, l, 
                              wmma::mem_row_major);
    }
}

//------------------------------------------------------------------------------
// Pthread CPU Implementation Definitions
//------------------------------------------------------------------------------
#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

// CPU Implementation using OpenMP and cache blocking
#define BLOCK_SIZE_M 32
#define BLOCK_SIZE_N 32
#define BLOCK_SIZE_K 32
#define BLOCK_SIZE_L 32

void cpu_matrix_multiply(float* A, float* B, float* C, int batch_size, int m, int n, int k, int l) {
    // Initialize output array to zero
    memset(C, 0, batch_size * m * l * sizeof(float));

    #pragma omp parallel for collapse(4)
    for (int b = 0; b < batch_size; b++) {
        for (int i0 = 0; i0 < m; i0 += BLOCK_SIZE_M) {
            for (int j0 = 0; j0 < l; j0 += BLOCK_SIZE_L) {
                for (int p0 = 0; p0 < n; p0 += BLOCK_SIZE_N) {
                    // Block boundaries
                    int imax = min(i0 + BLOCK_SIZE_M, m);
                    int jmax = min(j0 + BLOCK_SIZE_L, l);
                    int pmax = min(p0 + BLOCK_SIZE_N, n);
                    
                    // Compute on blocks
                    for (int i = i0; i < imax; i++) {
                        for (int j = j0; j < jmax; j++) {
                            // Compute partial sums with extended precision
                            __float128 sum = 0.0Q;  // Quad precision for maximum accuracy
                            size_t base_a = (size_t)b * m * n + (size_t)i * n;
                            size_t base_b = (size_t)b * k * l + (size_t)j;

                            for (int p = p0; p < pmax; p++) {
                                __float128 a_val = A[base_a + p];
                                __float128 b_val = B[base_b + p * l];
                                sum += a_val * b_val;
                            }
                            
                            // Accumulate partial sum into final result
                            size_t idx = (size_t)b * m * l + (size_t)i * l + j;
                            #pragma omp atomic
                            C[idx] += (float)sum;
                        }
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// Main function demonstrating various tensor multiplication implementations using CUDA.
//------------------------------------------------------------------------------
int main(int argc, char **argv) {  // Start of the main function; argc holds argument count, argv holds argument strings.
    //------------------------------------------------------------------------------
    // Command-Line Argument Parsing and Validation.
    //------------------------------------------------------------------------------
    if (argc != 6) {  // Check if the number of command-line arguments is exactly 6 (program name + 5 parameters).
        printf("Usage: ./a.out <batch_size> <m> <n> <k> <l>\n");  // Print usage instructions.
        printf("For tensor multiplication: (batch_size × m × n) * (batch_size × k × l)\n");  // Explain expected parameters.
        return 1;  // Exit the program with an error code.
    }
    
    int batch_size = atoi(argv[1]);  // Convert the first argument to an integer for batch_size.
    int m = atoi(argv[2]);           // Convert the second argument to an integer for m (number of rows in A and C).
    int n = atoi(argv[3]);           // Convert the third argument to an integer for n (number of columns in A and rows in B).
    int k = atoi(argv[4]);           // Convert the fourth argument to an integer for k (number of rows in B).
    int l = atoi(argv[5]);           // Convert the fifth argument to an integer for l (number of columns in B and C).
    
    //------------------------------------------------------------------------------
    // Memory Size Calculations.
    //------------------------------------------------------------------------------
    size_t total_elements_A = batch_size * m * n;  // Calculate the total number of elements in matrix A.
    size_t total_elements_B = batch_size * k * l;    // Calculate the total number of elements in matrix B.
    size_t total_elements_C = batch_size * m * l;    // Calculate the total number of elements in matrix C.
    size_t total_bytes = (total_elements_A + total_elements_B + total_elements_C) * sizeof(float);  // Compute total bytes needed.
    
    //------------------------------------------------------------------------------
    // Display Memory Requirements.
    //------------------------------------------------------------------------------
    size_t total_host_memory = (total_elements_A + total_elements_B + total_elements_C * 2) * sizeof(float);
    printf("\n=== Memory Requirements ===\n");
    printf("Host Memory Required: %.2f MB\n", total_host_memory / (1024.0 * 1024.0));
    printf("GPU Memory Required: %.2f MB\n", total_bytes / (1024.0 * 1024.0));
    
    // Check tensor core memory requirements
    size_t tensor_memory = (total_elements_A + total_elements_B) * sizeof(half) 
                        + total_elements_C * sizeof(float);
    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);
    bool skip_tensor = false;
    if (tensor_memory > free_memory) {
        printf("Note: Tensor Core implementation will be skipped (needs %.2f GB, has %.2f GB free)\n",
               tensor_memory / (1024.0 * 1024.0 * 1024.0),
               free_memory / (1024.0 * 1024.0 * 1024.0));
        skip_tensor = true;
    }
    
    if (total_host_memory > 16ULL * 1024 * 1024 * 1024) {  // Check if host memory exceeds 16GB.
        printf("Warning: Required host memory (%.2f GB) might exceed system memory\n", 
               total_host_memory / (1024.0 * 1024.0 * 1024.0));  // Print warning.
        printf("Continue? (y/n): ");  // Ask user if they wish to continue.
        char response;  // Variable to store user response.
        if (scanf(" %c", &response) != 1) {  // Read user response.
            printf("Error reading response\n");  // Print error if reading fails.
            return 1;  // Exit with error.
        }
        if (response != 'y' && response != 'Y') {  // If user does not choose 'y' or 'Y'.
            return 0;  // Exit the program.
        }
    }
    
    //------------------------------------------------------------------------------
    // Grid and Block Configuration for Kernel Launches.
    //------------------------------------------------------------------------------
    dim3 originalBlock(16, 16);  // Define a block size of 16x16 threads for the naive kernel.
    dim3 originalGrid(         // Calculate grid dimensions for the naive kernel.
        (m + originalBlock.x - 1) / originalBlock.x,  // Compute number of blocks in the x-direction.
        (l + originalBlock.y - 1) / originalBlock.y,  // Compute number of blocks in the y-direction.
        batch_size                                    // Use the batch_size as the z-dimension.
    );
    
    dim3 optimizedBlock(32, 32);  // Define a block size of 32x32 threads for optimized kernels.
    dim3 optimizedGrid(         // Calculate grid dimensions for the optimized kernels.
        (m + 31) / 32,         // Compute number of blocks in the x-direction (ceiling division).
        (l + 31) / 32,         // Compute number of blocks in the y-direction.
        batch_size             // Use batch_size for the z-dimension.
    );
    
    //------------------------------------------------------------------------------
    // Host and Device Memory Allocation.
    //------------------------------------------------------------------------------
    float *h_A, *h_B, *h_C, *h_C_original;  // Declare host pointers for matrices A, B, C, and a copy for baseline results.
    float *d_A, *d_B, *d_C;                 // Declare device pointers for matrices A, B, and C.
    
    h_A = (float *)malloc(total_elements_A * sizeof(float));  // Allocate host memory for matrix A.
    h_B = (float *)malloc(total_elements_B * sizeof(float));  // Allocate host memory for matrix B.
    h_C = (float *)malloc(total_elements_C * sizeof(float));  // Allocate host memory for matrix C (output).
    h_C_original = (float *)malloc(total_elements_C * sizeof(float));  // Allocate host memory for baseline output.
    
    if (h_A == NULL || h_B == NULL || h_C == NULL || h_C_original == NULL) {  // Check if any allocation failed.
        printf("Error: Failed to allocate host memory\n");  // Print an error message.
        if (h_A) free(h_A);  // Free allocated memory if necessary.
        if (h_B) free(h_B);
        if (h_C) free(h_C);
        if (h_C_original) free(h_C_original);
        return 1;  // Exit the program with an error code.
    }
    
    //------------------------------------------------------------------------------
    // Matrix Initialization.
    //------------------------------------------------------------------------------
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
    
    //------------------------------------------------------------------------------
    // Input Validation.
    //------------------------------------------------------------------------------
    if (batch_size <= 0 || m <= 0 || n <= 0 || k <= 0 || l <= 0) {  // Ensure all dimensions are positive.
        printf("Error: All dimensions must be positive integers\n");  // Print an error if not.
        return 1;  // Exit with an error code.
    }
    
    //------------------------------------------------------------------------------
    // Device Memory Allocation.
    //------------------------------------------------------------------------------
    cudaError_t err;  // Declare a variable to capture CUDA error codes.
    err = cudaMalloc((void **)&d_A, total_elements_A * sizeof(float));  // Allocate GPU memory for matrix A.
    if (err != cudaSuccess) {  // Check if allocation failed.
        fprintf(stderr, "Failed to allocate device memory for A: %s\n", cudaGetErrorString(err));  // Print error.
        return 1;  // Exit with error.
    }
    err = cudaMalloc((void **)&d_B, total_elements_B * sizeof(float));  // Allocate GPU memory for matrix B.
    if (err != cudaSuccess) {  // Check for errors.
        fprintf(stderr, "Failed to allocate device memory for B: %s\n", cudaGetErrorString(err));  // Print error.
        cudaFree(d_A);  // Free previously allocated device memory.
        return 1;  // Exit.
    }
    err = cudaMalloc((void **)&d_C, total_elements_C * sizeof(float));  // Allocate GPU memory for matrix C.
    if (err != cudaSuccess) {  // Check for errors.
        fprintf(stderr, "Failed to allocate device memory for C: %s\n", cudaGetErrorString(err));  // Print error.
        cudaFree(d_A);  // Free previously allocated device memory.
        cudaFree(d_B);
        return 1;  // Exit.
    }
    
    //------------------------------------------------------------------------------
    // Device Configuration.
    //------------------------------------------------------------------------------
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 4 * 1024 * 1024);  // Set persisting L2 cache size to 4MB for optimal performance.
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);           // Configure shared memory banks to use 8-byte size.
    
    //------------------------------------------------------------------------------
    // Initial Data Transfer (Host to Device).
    //------------------------------------------------------------------------------
    cudaMemcpy(d_A, h_A, total_elements_A * sizeof(float), cudaMemcpyHostToDevice);  // Copy matrix A from host to device.
    cudaMemcpy(d_B, h_B, total_elements_B * sizeof(float), cudaMemcpyHostToDevice);  // Copy matrix B from host to device.
    
    //------------------------------------------------------------------------------
    // CUDA Streams Setup.
    //------------------------------------------------------------------------------
    const int NUM_STREAMS = 4;  // Define the number of CUDA streams to use.
    cudaStream_t streams[NUM_STREAMS];  // Declare an array to hold stream handles.
    for (int i = 0; i < NUM_STREAMS; i++) {  // Loop to create each stream.
        cudaStreamCreate(&streams[i]);  // Create a CUDA stream.
    }
    
    //------------------------------------------------------------------------------
    // Determine Batches Per Stream.
    //------------------------------------------------------------------------------
    int batches_per_stream = (batch_size + NUM_STREAMS - 1) / NUM_STREAMS;  // Compute how many batches each stream should process.
    
    //------------------------------------------------------------------------------
    // Launch Kernels on Each Stream.
    //------------------------------------------------------------------------------
    for (int i = 0; i < NUM_STREAMS; i++) {  // Loop over each stream.
        int stream_batch_start = i * batches_per_stream;  // Calculate the starting batch index for this stream.
        int stream_batch_end = min((i + 1) * batches_per_stream, batch_size);  // Calculate the ending batch index.
        int stream_batches = stream_batch_end - stream_batch_start;  // Determine the number of batches for this stream.
        
        if (stream_batches <= 0) continue;  // If no batches are assigned, skip this stream.
        
        size_t stream_offset = stream_batch_start * m * n;  // Calculate the data offset in the flattened matrix for this stream.
        size_t stream_size = stream_batches * m * n * sizeof(float);  // Calculate the size (in bytes) of data for these batches.
        
        cudaMemcpyAsync(d_A + stream_offset, h_A + stream_offset, stream_size, cudaMemcpyHostToDevice, streams[i]);  // Asynchronously copy matrix A subset.
        
        dim3 streamGrid(  // Configure the grid for this stream.
            (m + BLOCK_SIZE - 1) / BLOCK_SIZE,  // Number of blocks in x-direction.
            (l + BLOCK_SIZE - 1) / BLOCK_SIZE,  // Number of blocks in y-direction.
            stream_batches                       // Number of batches for this stream in z-direction.
        );
        
        tensor_mul_optimized<<<streamGrid, optimizedBlock, 0, streams[i]>>>(  // Launch the optimized kernel on this stream.
            d_A + stream_offset,  // Pointer to the subset of matrix A.
            d_B + stream_offset,  // Pointer to the subset of matrix B.
            d_C + stream_offset,  // Pointer to the output for this stream.
            stream_batches,       // Number of batches to process in this stream.
            m, n, k, l            // Matrix dimensions.
        );
    }
    
    for (int i = 0; i < NUM_STREAMS; i++) {  // Synchronize each stream.
        cudaStreamSynchronize(streams[i]);  // Wait for the stream to finish execution.
    }
    
    //------------------------------------------------------------------------------
    // Performance Testing Section: Display problem size.
    //------------------------------------------------------------------------------
    printf("\n=== Performance Comparison ===\n");  // Print header.
    printf("Matrix Size per Batch: %d x %d\n", m, n);  // Print matrix dimensions.
    printf("Number of Batches: %d\n", batch_size);       // Print number of batches.
    printf("----------------------------------------\n");  // Print a separator.
    
    //------------------------------------------------------------------------------
    // Initialize timing variables and CUDA events
    //------------------------------------------------------------------------------
    cudaEvent_t start, stop;  // Declare CUDA events for timing.
    cudaEventCreate(&start);  // Create the start event.
    cudaEventCreate(&stop);   // Create the stop event.
    
    float transfer_time;     // Declare variables for timing
    float result_time;
    float original_time = 0.0f, optimized_time = 0.0f;
    float tc_time = 0.0f, vectorized_time = 0.0f;
    float warp_time = 0.0f, buffered_time = 0.0f;
    float vec_transfer_time = 0.0f, vec_result_time = 0.0f;
    float warp_transfer_time = 0.0f, warp_result_time = 0.0f;
    float buf_transfer_time = 0.0f, buf_result_time = 0.0f;
    float tensor_time = 0.0f, tensor_transfer_time = 0.0f, tensor_result_time = 0.0f;
    double cpu_time = 0.0;  // Add CPU timing variable
    
    //------------------------------------------------------------------------------
    // Test 0: Naive Implementation.
    //------------------------------------------------------------------------------
    printf("\n0. Naive Implementation:\n");  // Print test header.
    // Time the host-to-device (H2D) memory transfer operations
    cudaEventRecord(start);  // Record the start time for H2D transfer.
    cudaMemcpy(d_A, h_A, total_elements_A * sizeof(float), cudaMemcpyHostToDevice);  // Copy matrix A to device.
    cudaMemcpy(d_B, h_B, total_elements_B * sizeof(float), cudaMemcpyHostToDevice);  // Copy matrix B to device.
    cudaEventRecord(stop);   // Record the stop time.
    cudaEventSynchronize(stop);  // Wait for the event to complete.
    cudaEventElapsedTime(&transfer_time, start, stop);  // Compute elapsed time.
    printf("   Memory Transfer (H2D): %.3f ms\n", transfer_time);  // Print H2D transfer time.
    
    cudaEventRecord(start);  // Record start time for kernel execution.
    tensor_mul<<<originalGrid, originalBlock>>>(d_A, d_B, d_C, batch_size, m, n, k, l);  // Launch the naive kernel.
    cudaEventRecord(stop);   // Record stop time.
    cudaEventSynchronize(stop);  // Synchronize to ensure kernel completion.
    cudaEventElapsedTime(&original_time, start, stop);  // Calculate kernel execution time.
    printf("   Computation Time: %.3f ms\n", original_time);  // Print kernel computation time.
    
    cudaEventRecord(start);  // Record start time for D2H transfer.
    cudaMemcpy(h_C, d_C, total_elements_C * sizeof(float), cudaMemcpyDeviceToHost);  // Copy result matrix C back to host.
    cudaEventRecord(stop);   // Record stop time.
    cudaEventSynchronize(stop);  // Synchronize events.
    cudaEventElapsedTime(&result_time, start, stop);  // Compute elapsed time for D2H transfer.
    printf("   Memory Transfer (D2H): %.3f ms\n", result_time);  // Print D2H transfer time.
    printf("   Total Time: %.3f ms\n", transfer_time + original_time + result_time);  // Print total time.
    printf("   TFLOPS: %.2f\n", (2.0 * batch_size * m * n * l) / (original_time * 1000000000.0));  // Compute and print TFLOPS.
    
    // Save the baseline GPU result for correctness checks.
    memcpy(h_C_original, h_C, total_elements_C * sizeof(float));
    
    //------------------------------------------------------------------------------
    // Test 1: CPU Implementation (OpenMP)
    //------------------------------------------------------------------------------
    printf("\n1. CPU Implementation (OpenMP):\n");

    // Allocate memory for CPU result
    float *h_C_cpu = (float *)malloc(total_elements_C * sizeof(float));
    memset(h_C_cpu, 0, total_elements_C * sizeof(float));  // Initialize to zero
 
    // Time CPU implementation
    clock_t cpu_start = clock();
 
    // Run optimized CPU implementation using OpenMP
    cpu_matrix_multiply(h_A, h_B, h_C_cpu, batch_size, m, n, k, l);
 
    clock_t cpu_end = clock();
    cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("   Computation Time: %.3f ms\n", cpu_time);
    printf("   GFLOPS: %.2f\n", (2.0 * batch_size * m * n * l) / (cpu_time * 1000000.0));
    
    // Get and print the number of OpenMP threads used
    int max_threads = omp_get_max_threads();
    printf("   Number of Threads: %d\n", max_threads);
 
    // Verify CPU results against baseline
    bool cpu_matches = true;
    float cpu_max_diff = 0.0f;
    for (size_t i = 0; i < total_elements_C; i++) {
        float diff = fabs(h_C_cpu[i] - h_C_original[i]);  // Now compare against h_C_original
        cpu_max_diff = max(cpu_max_diff, diff);
        if (diff > 1e-4) {  // More lenient tolerance for CPU implementation
            cpu_matches = false;
            break;
        }
    }
    printf("   Accuracy Check: %s (max diff: %e)\n",
           cpu_matches ? "PASSED" : "FAILED", cpu_max_diff);

    free(h_C_cpu);
    
    //------------------------------------------------------------------------------
    // Test 2: Shared Memory Implementation.
    //------------------------------------------------------------------------------
    printf("\n2. Shared Memory Implementation:\n");  // Print test header.
    cudaEventRecord(start);  // Record start time for H2D transfer.
    cudaMemcpy(d_A, h_A, total_elements_A * sizeof(float), cudaMemcpyHostToDevice);  // Copy matrix A.
    cudaMemcpy(d_B, h_B, total_elements_B * sizeof(float), cudaMemcpyHostToDevice);  // Copy matrix B.
    cudaEventRecord(stop);   // Record stop time.
    cudaEventSynchronize(stop);  // Synchronize.
    float shared_transfer_time;  // Declare variable for shared memory H2D transfer time.
    cudaEventElapsedTime(&shared_transfer_time, start, stop);  // Compute elapsed time.
    printf("   Memory Transfer (H2D): %.3f ms\n", shared_transfer_time);  // Print transfer time.
    
    cudaEventRecord(start);  // Record start time for kernel execution.
    tensor_mul_optimized<<<optimizedGrid, optimizedBlock>>>(d_A, d_B, d_C, batch_size, m, n, k, l);  // Launch the shared memory optimized kernel.
    cudaEventRecord(stop);   // Record stop time.
    cudaEventSynchronize(stop);  // Synchronize.
    float shared_compute_time;  // Declare variable for shared memory kernel execution time.
    cudaEventElapsedTime(&shared_compute_time, start, stop);  // Compute elapsed time.
    printf("   Computation Time: %.3f ms\n", shared_compute_time);  // Print kernel time.
    
    cudaEventRecord(start);  // Record start time for D2H transfer.
    cudaMemcpy(h_C, d_C, total_elements_C * sizeof(float), cudaMemcpyDeviceToHost);  // Copy result matrix back to host.
    cudaEventRecord(stop);   // Record stop time.
    cudaEventSynchronize(stop);  // Synchronize.
    float shared_result_time;  // Declare variable for shared memory D2H transfer time.
    cudaEventElapsedTime(&shared_result_time, start, stop);  // Compute elapsed time.
    printf("   Memory Transfer (D2H): %.3f ms\n", shared_result_time);  // Print D2H time.
    printf("   Total Time: %.3f ms\n", shared_transfer_time + shared_compute_time + shared_result_time);  // Print total time.
    printf("   TFLOPS: %.2f\n", (2.0 * batch_size * m * n * l) / (shared_compute_time * 1000000000.0));  // Compute and print TFLOPS.
    printf("   Speedup vs Test 0 (CPU): %.2fx\n", cpu_time / shared_compute_time);
    printf("   Speedup vs Test 1 (Naive): %.2fx\n", original_time / shared_compute_time);  // Compute speedup over naive kernel.

    bool shared_mem_matches = true;  // Initialize flag for accuracy checking.
    float shared_mem_max_diff = 0.0f;  // Initialize variable to track maximum difference.
    for (size_t i = 0; i < total_elements_C; i++) {  // Loop over every element in the output.
        float diff = fabs(h_C[i] - h_C_original[i]);  // Calculate absolute difference.
        shared_mem_max_diff = max(shared_mem_max_diff, diff);  // Update maximum difference.
        if (diff > 1e-5) {  // Check if difference exceeds tolerance.
            shared_mem_matches = false;  // Mark as inaccurate.
            break;  // Exit loop.
        }
    }
    printf("   Accuracy Check (vs Baseline): %s (max diff: %e)\n", shared_mem_matches ? "PASSED" : "FAILED", shared_mem_max_diff);  // Print accuracy result.
    
    //------------------------------------------------------------------------------
    // Test 3: cuBLAS Implementation.
    //------------------------------------------------------------------------------
    printf("\n3. cuBLAS Implementation:\n");  // Print test header.
    cudaEventRecord(start);  // Record start time for H2D transfer.
    cudaMemcpy(d_A, h_A, total_elements_A * sizeof(float), cudaMemcpyHostToDevice);  // Copy matrix A.
    cudaMemcpy(d_B, h_B, total_elements_B * sizeof(float), cudaMemcpyHostToDevice);  // Copy matrix B.
    cudaEventRecord(stop);   // Record stop time.
    cudaEventSynchronize(stop);  // Synchronize.
    float cublas_transfer_time;  // Declare variable for cuBLAS H2D transfer time.
    cudaEventElapsedTime(&cublas_transfer_time, start, stop);  // Compute elapsed time.
    printf("   Memory Transfer (H2D): %.3f ms\n", cublas_transfer_time);  // Print transfer time.
    
    cublasHandle_t handle;  // Declare a cuBLAS handle.
    cublasCreate(&handle);  // Create the cuBLAS handle.
    cublasStatus_t status;  // Declare a variable for cuBLAS status.
    status = cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);  // Set the cuBLAS math mode.
    if (status != CUBLAS_STATUS_SUCCESS) {  // Check for errors.
        printf("Failed to set math mode: %d\n", status);  // Print error message.
        return 1;  // Exit with error.
    }
    
    cudaDeviceSynchronize();  // Synchronize the device.
    
    long long int strideA = (long long int)m * n;  // Calculate the stride for matrix A.
    long long int strideB = (long long int)k * l;    // Calculate the stride for matrix B.
    long long int strideC = (long long int)m * l;    // Calculate the stride for matrix C.
    
    const float alpha = 1.0f;  // Define the scalar alpha for GEMM.
    const float beta = 0.0f;   // Define the scalar beta for GEMM.
    
    status = cublasSgemmStridedBatched(  // Launch the batched GEMM using cuBLAS.
        handle,                      // The cuBLAS handle.
        CUBLAS_OP_N,                 // Specify no transpose for matrix A.
        CUBLAS_OP_N,                 // Specify no transpose for matrix B.
        l, m, k,                     // Dimensions for GEMM (swapped due to column-major order).
        &alpha,                      // Scalar multiplier for the product.
        d_B, l,                      // Matrix B and its leading dimension.
        strideB,                     // Stride between successive matrices in B.
        d_A, k,                      // Matrix A and its leading dimension.
        strideA,                     // Stride between successive matrices in A.
        &beta,                       // Scalar multiplier for matrix C.
        d_C, m,                      // Output matrix C and its leading dimension.
        strideC,                     // Stride between successive matrices in C.
        batch_size                   // Number of matrices (batches).
    );
    
    cudaDeviceSynchronize();  // Synchronize the device after cuBLAS computation.
    
    if (status != CUBLAS_STATUS_SUCCESS) {  // Check for errors in cuBLAS call.
        printf("Failed to execute batched GEMM: %d\n", status);  // Print error message.
        return 1;  // Exit with error.
    }
    
    cudaEventRecord(stop);  // Record stop time for cuBLAS computation.
    cudaEventSynchronize(stop);  // Synchronize events.
    cudaEventElapsedTime(&tc_time, start, stop);  // Compute cuBLAS computation time.
    printf("   Computation Time: %.3f ms\n", tc_time);  // Print cuBLAS computation time.
    
    cudaEventRecord(start);  // Record start time for D2H transfer.
    cudaMemcpy(h_C, d_C, total_elements_C * sizeof(float), cudaMemcpyDeviceToHost);  // Copy the result from device to host.
    cudaEventRecord(stop);   // Record stop time.
    cudaEventSynchronize(stop);  // Synchronize.
    float cublas_result_time;  // Declare variable for cuBLAS D2H transfer time.
    cudaEventElapsedTime(&cublas_result_time, start, stop);  // Compute elapsed time.
    printf("   Memory Transfer (D2H): %.3f ms\n", cublas_result_time);  // Print D2H time.
    printf("   Total Time: %.3f ms\n", tc_time + cublas_result_time);  // Print total time.
    printf("   TFLOPS: %.2f\n", (2.0 * batch_size * m * n * l) / (tc_time * 1000000000.0));  // Compute and print TFLOPS.

    printf("   Speedup vs Test 0 (CPU): %.2fx\n", cpu_time / tc_time);
    printf("   Speedup vs Test 1 (Naive): %.2fx\n", original_time / tc_time);  // Compute speedup over naive kernel.
    printf("   Speedup vs Test 2 (Shared Memory): %.2fx\n", shared_compute_time / tc_time);  // Compute speedup over shared memory kernel.
    
    bool cublas_matches = true;  // Initialize flag for cuBLAS accuracy check.
    float cublas_max_diff = 0.0f;  // Initialize variable for maximum difference.
    for (size_t i = 0; i < total_elements_C; i++) {  // Loop over each element.
        float diff = fabs(h_C[i] - h_C_original[i]);  // Compute absolute difference.
        cublas_max_diff = max(cublas_max_diff, diff);  // Update maximum difference.
        if (diff > 1e-5) {  // If difference exceeds tolerance...
            cublas_matches = false;  // ...mark as failed.
            break;  // Exit loop.
        }
    }
    printf("   Accuracy Check (vs Baseline): %s (max diff: %e)\n", cublas_matches ? "PASSED" : "FAILED", cublas_max_diff);  // Print accuracy result.
    
    //------------------------------------------------------------------------------
    // Test 4: Vectorized Implementation.
    //------------------------------------------------------------------------------
    printf("\n4. Vectorized Implementation:\n");  // Print test header.
    if (n % 4 != 0 || l % 4 != 0) {  // Check if matrix dimensions are multiples of 4.
        printf("   Skipped: Matrix dimensions must be multiples of 4\n");  // Print message if not.
    } else {  // If dimensions are valid for vectorization.
        cudaEventRecord(start);  // Record start time for H2D transfer.
        cudaMemcpy(d_A, h_A, total_elements_A * sizeof(float), cudaMemcpyHostToDevice);  // Copy matrix A.
        cudaMemcpy(d_B, h_B, total_elements_B * sizeof(float), cudaMemcpyHostToDevice);  // Copy matrix B.
        cudaEventRecord(stop);   // Record stop time.
        cudaEventSynchronize(stop);  // Synchronize.
        cudaEventElapsedTime(&vec_transfer_time, start, stop);  // Compute elapsed time.
        printf("   Memory Transfer (H2D): %.3f ms\n", vec_transfer_time);  // Print transfer time.
        
        float4 *d_A4 = (float4*)d_A;  // Reinterpret pointer d_A as float4 pointer.
        float4 *d_B4 = (float4*)d_B;  // Reinterpret pointer d_B as float4 pointer.
        float4 *d_C4 = (float4*)d_C;  // Reinterpret pointer d_C as float4 pointer.
        
        cudaEventRecord(start);  // Record start time for kernel execution.
        tensor_mul_vectorized<<<optimizedGrid, optimizedBlock>>>(d_A4, d_B4, d_C4, batch_size, m, n, k, l);  // Launch vectorized kernel.
        cudaEventRecord(stop);   // Record stop time.
        cudaEventSynchronize(stop);  // Synchronize.
        cudaEventElapsedTime(&vectorized_time, start, stop);  // Compute execution time.
        printf("   Computation Time: %.3f ms\n", vectorized_time);  // Print computation time.
        
        cudaEventRecord(start);  // Record start time for D2H transfer.
        cudaMemcpy(h_C, d_C, total_elements_C * sizeof(float), cudaMemcpyDeviceToHost);  // Copy result back to host.
        cudaEventRecord(stop);   // Record stop time.
        cudaEventSynchronize(stop);  // Synchronize.
        cudaEventElapsedTime(&vec_result_time, start, stop);  // Compute elapsed time.
        printf("   Memory Transfer (D2H): %.3f ms\n", vec_result_time);  // Print D2H time.
        printf("   Total Time: %.3f ms\n", vec_transfer_time + vectorized_time + vec_result_time);  // Print total time.
        printf("   TFLOPS: %.2f\n", (2.0 * batch_size * m * n * l) / (vectorized_time * 1000000000.0));  // Compute and print TFLOPS.

        printf("   Speedup vs Test 0 (CPU): %.2fx\n", cpu_time / vectorized_time);
        printf("   Speedup vs Test 1 (Naive): %.2fx\n", original_time / vectorized_time);  // Compute speedup over naive.
        printf("   Speedup vs Test 2 (Shared Memory): %.2fx\n", shared_compute_time / vectorized_time);  // Compute speedup over shared memory.
        printf("   Speedup vs Test 3 (cuBLAS): %.2fx\n", tc_time / vectorized_time);  // Compute speedup over cuBLAS.
        
        bool vectorized_matches = true;  // Initialize flag for vectorized accuracy check.
        float vectorized_max_diff = 0.0f;  // Initialize variable for maximum difference.
        for (size_t i = 0; i < total_elements_C; i++) {  // Loop over every element.
            float diff = fabs(h_C[i] - h_C_original[i]);  // Compute absolute difference.
            vectorized_max_diff = max(vectorized_max_diff, diff);  // Update maximum difference.
            if (diff > 1e-5) {  // If difference exceeds tolerance...
                vectorized_matches = false;  // ...mark as failed.
                break;  // Exit loop.
            }
        }
        printf("   Accuracy Check (vs Baseline): %s (max diff: %e)\n", vectorized_matches ? "PASSED" : "FAILED", vectorized_max_diff);  // Print accuracy result.
    }
    
    //------------------------------------------------------------------------------
    // Test 5: Warp-Optimized Implementation
    //------------------------------------------------------------------------------
    printf("\n5. Warp-Optimized Implementation:\n");  // Print test header
    
    // Time the host-to-device (H2D) memory transfer operations
    cudaEventRecord(start);  // Start timing H2D transfer
    cudaMemcpy(d_A, h_A, total_elements_A * sizeof(float), cudaMemcpyHostToDevice);  // Copy matrix A to device
    cudaMemcpy(d_B, h_B, total_elements_B * sizeof(float), cudaMemcpyHostToDevice);  // Copy matrix B to device
    cudaEventRecord(stop);   // Stop timing H2D transfer
    cudaEventSynchronize(stop);  // Wait for transfer to complete
    cudaEventElapsedTime(&warp_transfer_time, start, stop);  // Calculate H2D transfer time
    printf("   Memory Transfer (H2D): %.3f ms\n", warp_transfer_time);  // Print H2D transfer time
    
    // Time the kernel execution
    cudaEventRecord(start);  // Start timing kernel execution
    tensor_mul_warp_optimized<<<optimizedGrid, optimizedBlock>>>(d_A, d_B, d_C, batch_size, m, n, k, l);  // Launch the warp-optimized kernel
    cudaEventRecord(stop);   // Stop timing kernel execution
    cudaEventSynchronize(stop);  // Wait for kernel to complete
    cudaEventElapsedTime(&warp_time, start, stop);  // Calculate kernel execution time
    printf("   Computation Time: %.3f ms\n", warp_time);  // Print computation time
    
    // Time the device-to-host (D2H) memory transfer operations
    cudaEventRecord(start);  // Start timing D2H transfer
    cudaMemcpy(h_C, d_C, total_elements_C * sizeof(float), cudaMemcpyDeviceToHost);  // Copy result back to host
    cudaEventRecord(stop);   // Stop timing D2H transfer
    cudaEventSynchronize(stop);  // Wait for transfer to complete
    cudaEventElapsedTime(&warp_result_time, start, stop);  // Calculate D2H transfer time
    printf("   Memory Transfer (D2H): %.3f ms\n", warp_result_time);  // Print D2H transfer time
    
    // Print overall performance metrics
    printf("   Total Time: %.3f ms\n", warp_transfer_time + warp_time + warp_result_time);  // Print total execution time
    printf("   TFLOPS: %.2f\n", (2.0 * batch_size * m * n * l) / (warp_time * 1000000000.0));  // Compute and print TFLOPS

    printf("   Speedup vs Test 0 (Naive): %.2fx\n", original_time / warp_time);
    printf("   Speedup vs Test 1 (CPU): %.2fx\n", cpu_time / warp_time);  // Compute speedup over naive kernel
    printf("   Speedup vs Test 2 (Shared Memory): %.2fx\n", shared_compute_time / warp_time);  // Compute speedup over shared memory kernel
    printf("   Speedup vs Test 3 (cuBLAS): %.2fx\n", tc_time / warp_time);  // Compute speedup over cuBLAS
    printf("   Speedup vs Test 4 (Vectorized): %.2fx\n", vectorized_time / warp_time);  // Compute speedup over vectorized implementation
    
    // Check accuracy against baseline
    bool warp_matches = true;  // Initialize flag for accuracy checking
    float warp_max_diff = 0.0f;  // Initialize variable to track maximum difference
    for (size_t i = 0; i < total_elements_C; i++) {  // Loop over every element in the output
        float diff = fabs(h_C[i] - h_C_original[i]);  // Calculate absolute difference
        warp_max_diff = max(warp_max_diff, diff);  // Update maximum difference
        if (diff > 1e-5) {  // Check if difference exceeds tolerance
            warp_matches = false;  // Mark as inaccurate
            break;  // Exit loop
            }
        }
        printf("   Accuracy Check (vs Baseline): %s (max diff: %e)\n",
           warp_matches ? "PASSED" : "FAILED", warp_max_diff);  // Print accuracy result
    
    //------------------------------------------------------------------------------
    // Test 6: Double-Buffered Implementation
    //------------------------------------------------------------------------------
    printf("\n6. Double-Buffered Implementation:\n");  // Print test header
    
    // Time the host-to-device (H2D) memory transfer operations
    cudaEventRecord(start);  // Start timing H2D transfer
    cudaMemcpy(d_A, h_A, total_elements_A * sizeof(float), cudaMemcpyHostToDevice);  // Copy matrix A to device
    cudaMemcpy(d_B, h_B, total_elements_B * sizeof(float), cudaMemcpyHostToDevice);  // Copy matrix B to device
    cudaEventRecord(stop);   // Stop timing H2D transfer
    cudaEventSynchronize(stop);  // Wait for transfer to complete
    cudaEventElapsedTime(&buf_transfer_time, start, stop);  // Calculate H2D transfer time
    printf("   Memory Transfer (H2D): %.3f ms\n", buf_transfer_time);  // Print H2D transfer time
    
    // Time the kernel execution
    cudaEventRecord(start);  // Start timing kernel execution
    tensor_mul_double_buffered<<<optimizedGrid, optimizedBlock>>>(d_A, d_B, d_C, batch_size, m, n, k, l);  // Launch double-buffered kernel
    cudaEventRecord(stop);   // Stop timing kernel execution
    cudaEventSynchronize(stop);  // Wait for kernel to complete
    cudaEventElapsedTime(&buffered_time, start, stop);  // Calculate kernel execution time
    printf("   Computation Time: %.3f ms\n", buffered_time);  // Print computation time
    
    // Time the device-to-host (D2H) memory transfer operations
    cudaEventRecord(start);  // Start timing D2H transfer
    cudaMemcpy(h_C, d_C, total_elements_C * sizeof(float), cudaMemcpyDeviceToHost);  // Copy result back to host
    cudaEventRecord(stop);   // Stop timing D2H transfer
    cudaEventSynchronize(stop);  // Wait for transfer to complete
    cudaEventElapsedTime(&buf_result_time, start, stop);  // Calculate D2H transfer time
    
    printf("   Memory Transfer (D2H): %.3f ms\n", buf_result_time);  // Print D2H transfer time
    printf("   Total Time: %.3f ms\n", buf_transfer_time + buffered_time + buf_result_time);  // Print total execution time
    printf("   TFLOPS: %.2f\n", (2.0 * batch_size * m * n * l) / (buffered_time * 1000000000.0));  // Calculate and print TFLOPS

    printf("   Speedup vs Test 0 (Naive): %.2fx\n", original_time / buffered_time);
    printf("   Speedup vs Test 1 (CPU): %.2fx\n", cpu_time / buffered_time);  // Compare against naive
    printf("   Speedup vs Test 2 (Shared Memory): %.2fx\n", shared_compute_time / buffered_time);  // Compare against shared memory
    printf("   Speedup vs Test 3 (cuBLAS): %.2fx\n", tc_time / buffered_time);  // Compare against cuBLAS
    printf("   Speedup vs Test 4 (Vectorized): %.2fx\n", vectorized_time / buffered_time);  // Compare against vectorized
    printf("   Speedup vs Test 5 (Warp-Optimized): %.2fx\n", warp_time / buffered_time);  // Compare against warp
    
    bool buffered_matches = true;  // Initialize flag for double-buffered accuracy check.
    float buffered_max_diff = 0.0f;  // Initialize variable for maximum difference.
    for (size_t i = 0; i < total_elements_C; i++) {  // Loop over each element in the output.
        float diff = fabs(h_C[i] - h_C_original[i]);  // Calculate absolute difference.
        buffered_max_diff = max(buffered_max_diff, diff);  // Update maximum difference.
        if (diff > 1e-5) {  // Check if difference exceeds tolerance.
            buffered_matches = false;  // Mark as failed.
            break;  // Exit loop.
        }
    }
    printf("   Accuracy Check: %s (Max Diff: %e)\n", buffered_matches ? "PASSED" : "FAILED", buffered_max_diff);  // Print accuracy result.
    
    //------------------------------------------------------------------------------
    // Test 7: Tensor Core Implementation
    //------------------------------------------------------------------------------
    printf("\n7. Tensor Core Implementation:\n");
    
    if (skip_tensor) {  // Check if we need to skip tensor core implementation due to memory constraints
        printf("   Skipped: Insufficient GPU memory\n");  // Print skip message for memory limitation
    } else if (m % 16 != 0 || n % 16 != 0 || l % 16 != 0) {  // Check if dimensions are multiples of 16 (tensor core requirement)
        printf("   Skipped: Matrix dimensions must be multiples of 16\n");  // Print skip message for dimension mismatch
    } else {  // If all requirements are met, proceed with tensor core implementation
        // Allocate device memory for half-precision matrices
        half *d_A_half, *d_B_half;  // Declare device pointers for half-precision matrices
        cudaMalloc(&d_A_half, total_elements_A * sizeof(half));  // Allocate device memory for matrix A in half precision
        cudaMalloc(&d_B_half, total_elements_B * sizeof(half));  // Allocate device memory for matrix B in half precision

        // Allocate and prepare host memory for half-precision matrices
        half *h_A_half = (half*)malloc(total_elements_A * sizeof(half));  // Allocate host memory for half-precision A
        half *h_B_half = (half*)malloc(total_elements_B * sizeof(half));  // Allocate host memory for half-precision B
        for (size_t i = 0; i < total_elements_A; i++) {  // Convert matrix A elements to half precision
            h_A_half[i] = __float2half(h_A[i]);  // Convert each element from float to half
        }
        for (size_t i = 0; i < total_elements_B; i++) {  // Convert matrix B elements to half precision
            h_B_half[i] = __float2half(h_B[i]);  // Convert each element from float to half
        }

        // Time host to device transfer
        cudaEventRecord(start);  // Start timing H2D transfer
        cudaMemcpy(d_A_half, h_A_half, total_elements_A * sizeof(half), cudaMemcpyHostToDevice);  // Copy A to device
        cudaMemcpy(d_B_half, h_B_half, total_elements_B * sizeof(half), cudaMemcpyHostToDevice);  // Copy B to device
        cudaEventRecord(stop);  // Stop timing H2D transfer
        cudaEventSynchronize(stop);  // Wait for transfer to complete
        cudaEventElapsedTime(&tensor_transfer_time, start, stop);  // Calculate H2D transfer time

        // Configure kernel launch parameters
        dim3 tensorBlock(256, 1, 1);  // Define block dimensions for tensor core kernel
        dim3 tensorGrid(  // Define grid dimensions for tensor core kernel
            (m + 15) / 16,  // Calculate number of blocks for rows (ceiling division by 16)
            (l + 15) / 16,  // Calculate number of blocks for columns (ceiling division by 16)
            batch_size      // One block per batch
        );

        // Time kernel execution
        cudaEventRecord(start);  // Start timing kernel execution
        tensor_mul_tensorcore<<<tensorGrid, tensorBlock>>>(  // Launch tensor core kernel
            d_A_half, d_B_half, d_C, batch_size, m, n, k, l);  // Pass parameters to kernel
        cudaEventRecord(stop);  // Stop timing kernel execution
        cudaEventSynchronize(stop);  // Wait for kernel to complete
        cudaEventElapsedTime(&tensor_time, start, stop);  // Calculate kernel execution time

        // Time device to host transfer
        cudaEventRecord(start);  // Start timing D2H transfer
        cudaMemcpy(h_C, d_C, total_elements_C * sizeof(float), cudaMemcpyDeviceToHost);  // Copy result back to host
        cudaEventRecord(stop);  // Stop timing D2H transfer
        cudaEventSynchronize(stop);  // Wait for transfer to complete
        cudaEventElapsedTime(&tensor_result_time, start, stop);  // Calculate D2H transfer time

        // Clean up resources
        cudaFree(d_A_half);  // Free device memory for half-precision A
        cudaFree(d_B_half);  // Free device memory for half-precision B
        free(h_A_half);      // Free host memory for half-precision A
        free(h_B_half);      // Free host memory for half-precision B

        // Verify results
        bool tensor_matches = true;  // Initialize accuracy check flag
        float tensor_max_diff = 0.0f;  // Initialize maximum difference tracker
        for (size_t i = 0; i < total_elements_C; i++) {  // Check each element
            float diff = fabs(h_C[i] - h_C_original[i]);  // Calculate absolute difference
            tensor_max_diff = max(tensor_max_diff, diff);  // Update maximum difference
            if (diff > 2e-2) {  // 2% tolerance for large matrices
                tensor_matches = false;  // Mark as failed if tolerance exceeded
                break;  // Exit loop on first failure
            }
        }

        // Print performance metrics
        printf("   Memory Transfer (H2D): %.3f ms\n", tensor_transfer_time);  // Print H2D transfer time
        printf("   Computation Time: %.3f ms\n", tensor_time);  // Print computation time
        printf("   Memory Transfer (D2H): %.3f ms\n", tensor_result_time);  // Print D2H transfer time
        printf("   Total Time: %.3f ms\n", tensor_transfer_time + tensor_time + tensor_result_time);  // Print total time
        printf("   TFLOPS: %.2f\n", (2.0 * batch_size * m * n * l) / (tensor_time * 1000000000.0));  // Print TFLOPS

        printf("   Speedup vs Test 0 (Naive): %.2fx\n", original_time / tensor_time);
        printf("   Speedup vs Test 1 (CPU): %.2fx\n", cpu_time / tensor_time);  // Print speedup vs naive
        printf("   Speedup vs Test 2 (Shared Memory): %.2fx\n", shared_compute_time / tensor_time);  // Print speedup vs shared
        printf("   Speedup vs Test 3 (cuBLAS): %.2fx\n", tc_time / tensor_time);  // Print speedup vs cuBLAS
        printf("   Speedup vs Test 4 (Vectorized): %.2fx\n", vectorized_time / tensor_time);  // Print speedup vs vectorized
        printf("   Speedup vs Test 5 (Warp-Optimized): %.2fx\n", warp_time / tensor_time);  // Print speedup vs warp
        printf("   Speedup vs Test 6 (Double-Buffered): %.2fx\n", buffered_time / tensor_time);  // Print speedup vs buffered
        printf("   Accuracy Check (vs Baseline): %s (max diff: %e)\n",  // Print accuracy check results
               tensor_matches ? "PASSED" : "FAILED", tensor_max_diff);  // Show pass/fail and maximum difference
    }

    //------------------------------------------------------------------------------
    // Final Performance Summary.
    //------------------------------------------------------------------------------
    printf("\n=== Performance Summary ===\n");  // Print summary header.
    printf("0. Naive Implementation:\n");
    printf("   Computation: %.3f ms, Memory H2D: %.3f ms, D2H: %.3f ms, Total: %.3f ms\n", 
           original_time, transfer_time, result_time, transfer_time + original_time + result_time);
    
    printf("1. CPU Implementation (OpenMP):\n");
    printf("   Computation: %.3f ms\n", cpu_time);
    
    printf("2. Shared Memory Implementation:\n");  // Print label.
    printf("   Computation: %.3f ms, Memory H2D: %.3f ms, D2H: %.3f ms, Total: %.3f ms\n", shared_compute_time, shared_transfer_time, shared_result_time, shared_transfer_time + shared_compute_time + shared_result_time);  // Print metrics.
    
    printf("3. cuBLAS Implementation:\n");  // Print label.
    printf("   Computation: %.3f ms, Memory H2D: %.3f ms, D2H: %.3f ms, Total: %.3f ms\n", tc_time, cublas_transfer_time, cublas_result_time, cublas_transfer_time + tc_time + cublas_result_time);  // Print metrics.
    
    if (n % 4 != 0 || l % 4 != 0) {  // Check if vectorized test was skipped.
        printf("4. Vectorized Implementation: Skipped (dimensions must be multiples of 4)\n");  // Print message.
    } else {
        printf("4. Vectorized Implementation:\n");  // Print label.
        printf("   Computation: %.3f ms, Memory H2D: %.3f ms, D2H: %.3f ms, Total: %.3f ms\n", vectorized_time, vec_transfer_time, vec_result_time, vec_transfer_time + vectorized_time + vec_result_time);  // Print metrics.
    }
    
    printf("5. Warp-Optimized Implementation:\n");  // Print label.
    printf("   Computation: %.3f ms, Memory H2D: %.3f ms, D2H: %.3f ms, Total: %.3f ms\n", warp_time, warp_transfer_time, warp_result_time, warp_transfer_time + warp_time + warp_result_time);  // Print metrics.
    
    printf("6. Double-Buffered Implementation:\n");  // Print label.
    printf("   Computation: %.3f ms, Memory H2D: %.3f ms, D2H: %.3f ms, Total: %.3f ms\n", buffered_time, buf_transfer_time, buf_result_time, buf_transfer_time + buffered_time + buf_result_time);  // Print metrics.
    
    printf("7. Tensor Core Implementation:\n");  // Print label.
    if (skip_tensor) {
        printf("   Skipped: Insufficient GPU memory\n");
    } else if (m % 16 != 0 || n % 16 != 0 || l % 16 != 0) {
        printf("   Skipped: Matrix dimensions must be multiples of 16\n");
    } else {
        printf("   Computation: %.3f ms, Memory H2D: %.3f ms, D2H: %.3f ms, Total: %.3f ms\n", 
               tensor_time, tensor_transfer_time, tensor_result_time, 
               tensor_transfer_time + tensor_time + tensor_result_time);
    }
    
    //------------------------------------------------------------------------------
    // Resource Cleanup.
    //------------------------------------------------------------------------------
    for (int i = 0; i < NUM_STREAMS; i++) {  // Loop over each stream.
        cudaStreamDestroy(streams[i]);  // Destroy the CUDA stream.
    }
    
    cublasDestroy(handle);  // Destroy the cuBLAS handle to free resources.
    
    cudaEventDestroy(start);  // Destroy the CUDA event 'start'.
    cudaEventDestroy(stop);   // Destroy the CUDA event 'stop'.
    
    cudaFree(d_A);  // Free the device memory allocated for matrix A.
    cudaFree(d_B);  // Free the device memory allocated for matrix B.
    cudaFree(d_C);  // Free the device memory allocated for matrix C.
    
    free(h_A);  // Free the host memory allocated for matrix A.
    free(h_B);  // Free the host memory allocated for matrix B.
    free(h_C);  // Free the host memory allocated for matrix C.
    free(h_C_original);  // Free the host memory allocated for the baseline output.
    
    return 0;  // Return 0 to indicate successful execution of the program.
}  // End of main function.
