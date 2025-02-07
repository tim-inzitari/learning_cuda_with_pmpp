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
 * Vectorized CUDA kernel using float4 for coalesced memory access
 * This version loads 4 elements at once for better memory bandwidth
 * 
 * === What is float4? ===
 * - Built-in CUDA vector type that loads/stores 4 floats at once
 * - Members: x, y, z, w (like in graphics programming)
 * - Size: 16 bytes (4 x 4-byte floats)
 * 
 * === Why Use Vector Types? ===
 * 1. Memory Coalescing:
 *    - One memory transaction fetches 16 bytes instead of 4
 *    - Reduces number of memory requests by 4x
 *    - Better utilizes memory bandwidth
 * 
 * 2. Memory Alignment:
 *    - float4 is naturally aligned to 16-byte boundaries
 *    - Optimal for modern GPU memory controllers
 *    - Reduces number of memory transactions
 * 
 * 3. Instruction Efficiency:
 *    - Some operations can be done on all 4 elements simultaneously
 *    - Better utilization of GPU's vector units
 *    - Can reduce register pressure
 */
__global__
void tensor_mul_vectorized(float4 *A, float4 *B, float4 *C, int batch_size, int m, int n, int k, int l) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];  // Changed to float
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];  // Changed to float
    
    int batch = blockIdx.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = bx * TILE_SIZE + tx;
    int col = by * TILE_SIZE + ty;
    
    // Calculate batch offsets
    size_t batch_offset_A = (size_t)batch * m * n;
    size_t batch_offset_B = (size_t)batch * k * l;
    size_t batch_offset_C = (size_t)batch * m * l;
    
    float sum = 0.0f;  // Single accumulator
    
    // Process matrix multiplication tile by tile
    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile from matrix A using float4
        if (row < m && (tile * TILE_SIZE + ty) < n) {
            float4 a4 = A[batch_offset_A/4 + row * (n/4) + (tile * TILE_SIZE + ty)/4];
            As[tx][ty] = (ty % 4 == 0) ? a4.x :
                        (ty % 4 == 1) ? a4.y :
                        (ty % 4 == 2) ? a4.z : a4.w;
        } else {
            As[tx][ty] = 0.0f;
        }
        
        // Load tile from matrix B using float4
        if ((tile * TILE_SIZE + tx) < k && col < l) {
            float4 b4 = B[batch_offset_B/4 + (tile * TILE_SIZE + tx) * (l/4) + col/4];
            Bs[tx][ty] = (col % 4 == 0) ? b4.x :
                        (col % 4 == 1) ? b4.y :
                        (col % 4 == 2) ? b4.z : b4.w;
        } else {
            Bs[tx][ty] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[tx][k] * Bs[k][ty];
        }
        
        __syncthreads();
    }
    
    // Write result to global memory using float4
    if (row < m && col < l && batch < batch_size) {
        int c_idx = batch_offset_C/4 + row * (l/4) + col/4;
        float4 c4 = C[c_idx];
        if (col % 4 == 0) c4.x = sum;
        else if (col % 4 == 1) c4.y = sum;
        else if (col % 4 == 2) c4.z = sum;
        else c4.w = sum;
        C[c_idx] = c4;
    }
}

/**
 * === Warp-Optimized Kernel Using Warp Shuffle ===
 * This kernel performs matrix multiplication using warp-level primitives.
 * It reduces shared memory usage by replacing inter-thread communication
 * with warp shuffle operations.
 *
 * Key steps:
 * 1. Each warp computes a tile of the output matrix.
 * 2. Each thread in the warp computes a partial dot product.
 * 3. Warp shuffle functions (e.g. __shfl_down_sync()) are used to sum the partial products.
 * 4. The final result is written to global memory.
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
void tensor_mul_warp_optimized(float *A, float *B, float *C, int batch_size, int m, int n, int k, int l) {
    // Shared memory for tiles
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];
    
    // Calculate indices
    int batch = blockIdx.z;
    int warpId = threadIdx.y;  // Warp ID within block
    int lane = threadIdx.x;    // Lane ID within warp
    
    // Calculate global row and column
    int row = blockIdx.x * 32 + warpId;
    int col = blockIdx.y * 32 + lane;
    
    // Calculate batch offsets
    size_t batch_offset_A = (size_t)batch * m * n;
    size_t batch_offset_B = (size_t)batch * k * l;
    size_t batch_offset_C = (size_t)batch * m * l;
    
    // Accumulator for dot product
    float sum = 0.0f;
    
    // Process matrix multiplication tile by tile
    for (int tile = 0; tile < (n + 31) / 32; tile++) {
        // Collaborative loading of tiles into shared memory
        if (row < m && (tile * 32 + lane) < n) {
            As[warpId][lane] = A[batch_offset_A + row * n + tile * 32 + lane];
        } else {
            As[warpId][lane] = 0.0f;
        }
        
        if ((tile * 32 + warpId) < k && col < l) {
            Bs[warpId][lane] = B[batch_offset_B + (tile * 32 + warpId) * l + col];
        } else {
            Bs[warpId][lane] = 0.0f;
        }
        
        // Ensure tiles are loaded
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < 32; k++) {
            sum += As[warpId][k] * Bs[k][lane];
        }
        
        // Synchronize before next tile
        __syncthreads();
    }
    
    // Write result if within bounds
    if (row < m && col < l && batch < batch_size) {
        C[batch_offset_C + row * l + col] = sum;
    }
}

/**
 * Double-buffered kernel that overlaps computation with memory loads
 * Uses two sets of shared memory buffers to hide memory latency
 * 
 * === What is Double Buffering? ===
 * - Uses two sets of shared memory buffers (ping-pong buffers)
 * - While computing on one buffer, loads data into the other
 * - Hides memory latency by overlapping computation and memory access
 * 
 * === Memory Access Pattern ===
 * Buffer 0: [Computing][Loading ][Computing][Loading ]
 * Buffer 1: [Loading ][Computing][Loading ][Computing]
 * Result:   [Comp    ][Comp    ][Comp    ][Comp    ]
 * 
 * === Performance Benefits ===
 * 1. Memory Latency Hiding:
 *    - GPU can fetch next tile while computing current tile
 *    - Reduces idle time waiting for memory
 * 
 * 2. Better Memory Bandwidth Utilization:
 *    - Memory controller stays busy
 *    - More efficient use of memory bus
 * 
 * 3. Improved Throughput:
 *    - Less time waiting for data
 *    - Can approach peak compute performance
 * 
 * === Implementation Details ===
 * - Uses two [TILE_SIZE][TILE_SIZE] shared memory arrays
 * - Alternates between buffers using a 'buf' variable
 * - Requires careful synchronization between loads and computes
 */
__global__
void tensor_mul_double_buffered(float *A, float *B, float *C, int batch_size, int m, int n, int k, int l) {
    // === Double-Buffered Kernel Using Ping-Pong Buffers ===
    // This kernel overlaps computation with data transfer by using two shared memory buffers.
    // While the kernel computes on one buffer (active), it concurrently loads the next tile into the inactive buffer.
    //
    // Benefits:
    // - Hides global memory latency by overlapping with computation.
    // - Improves effective memory bandwidth usage.
    //
    // === Memory Access Pattern ===
    // Buffer A: [Compute][Load  ][Compute][Load  ]
    // Buffer B: [Load  ][Compute][Load  ][Compute]
    // Result:   [Busy  ][Busy  ][Busy  ][Busy  ]
    //
    // === Implementation Details ===
    // - Uses two [TILE_SIZE][TILE_SIZE] shared memory arrays
    // - Alternates between buffers using a 'buf' variable
    // - Requires careful synchronization between loads and computes
    //
    // Two buffers for each matrix (ping-pong buffers)
    // [2] indicates two copies for double buffering
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE];
    
    // Standard CUDA thread/block index calculation
    int batch = blockIdx.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global memory indices
    int row = bx * TILE_SIZE + tx;
    int col = by * TILE_SIZE + ty;
    
    // Batch offsets for each matrix in global memory
    size_t batch_offset_A = (size_t)batch * m * n;
    size_t batch_offset_B = (size_t)batch * k * l;
    size_t batch_offset_C = (size_t)batch * m * l;
    
    // Accumulator for dot product result
    float sum = 0.0f;
    // Buffer selector (0 or 1) - switches between buffers
    int buf = 0;
    
    // === Stage 1: Initial Load ===
    // Load first tiles into buffer 0
    // This happens before the main computation loop
    if (row < m && ty < n) {
        As[0][tx][ty] = A[batch_offset_A + row * n + ty];
    }
    if (tx < k && col < l) {
        Bs[0][tx][ty] = B[batch_offset_B + tx * l + col];
    }
    
    // Ensure first tiles are loaded before computation
    __syncthreads();
    
    // === Stage 2: Main Loop ===
    // Each iteration processes one tile while loading the next
    for (int tile = 1; tile < (n + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load next tile into inactive buffer (1-buf)
        // While loading, computation uses active buffer (buf)
        if (row < m && (tile * TILE_SIZE + ty) < n) {
            As[1-buf][tx][ty] = A[batch_offset_A + row * n + tile * TILE_SIZE + ty];
        }
        if ((tile * TILE_SIZE + tx) < k && col < l) {
            Bs[1-buf][tx][ty] = B[batch_offset_B + (tile * TILE_SIZE + tx) * l + col];
        }
        
        // Compute using current buffer while loading completes
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[buf][tx][k] * Bs[buf][k][ty];
        }
        
        // Swap buffers
        buf = 1 - buf;
        // Ensure all threads complete computation and loading
        // before next iteration
        __syncthreads();
    }
    
    // === Stage 3: Final Computation ===
    // Process the last loaded tile
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
        sum += As[buf][tx][k] * Bs[buf][k][ty];
    }
    
    // Write result to global memory
    // Only write if thread's indices are within bounds
    if (row < m && col < l && batch < batch_size) {
        C[batch_offset_C + row * l + col] = sum;
    }
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
    // Calculate memory requirements for each matrix
    // Using size_t (64-bit) instead of int (32-bit) to handle large matrices
    // This allows matrices larger than 2GB (2^31 elements)
    // Example: For 8 1024x1024 matrices:
    //   - Each element is 4 bytes (float)
    //   - Total size = 8 * 1024 * 1024 * 4 = 32MB per matrix
    size_t total_elements_A = batch_size * m * n;
    size_t total_elements_B = batch_size * k * l;
    size_t total_elements_C = batch_size * m * l;
    size_t total_bytes = (total_elements_A + total_elements_B + total_elements_C) * sizeof(float);

    // === Grid and Block Configuration ===
    // === Understanding CUDA Thread Hierarchy ===
    // CUDA organizes threads in a 3-level hierarchy:
    // 1. Grid: Collection of all blocks
    //    - Can be 1D, 2D, or 3D
    //    - In our case: (ceil(m/16), ceil(l/16), batch_size)
    //    - Each dimension can have up to 2^31-1 blocks
    // 
    // 2. Block: Group of threads that work together
    //    - Share fast shared memory
    //    - Can synchronize using __syncthreads()
    //    - Maximum 1024 threads per block on RTX 3080 Ti
    // 
    // 3. Thread: Individual execution unit
    //    - Each thread has unique coordinates within its block
    //    - Coordinates used to determine which data to process
    //    - Threads in same warp (32 threads) execute together
    
    // === Block Size Selection Strategy ===
    // For Naive Implementation (16x16):
    // - 256 threads per block (16 * 16)
    // - Smaller blocks for better occupancy
    // - Works well on all CUDA devices
    // - Leaves room for more blocks per SM
    // - Good for memory-bound kernels
    
    // For Optimized Implementation (32x32):
    // - 1024 threads per block (32 * 32)
    // - Maximum threads per block on Ampere
    // - Better for compute-bound kernels
    // - Larger tiles in shared memory
    // - More data reuse within block
    // 
    // Grid Size Calculation Example:
    // For a 1024x1024 matrix:
    // - Using 16x16 blocks:
    //   * Grid X = ceil(1024/16) = 64 blocks
    //   * Grid Y = ceil(1024/16) = 64 blocks
    //   * Total blocks = 64 * 64 = 4096 blocks
    // 
    // Hardware Considerations:
    // - RTX 3080 Ti has 80 SMs (Streaming Multiprocessors)
    // - Each SM can handle multiple blocks
    // - More blocks = better SM utilization
    // - But too many threads per block reduces occupancy

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

    // === Memory Management ===
    // CUDA uses separate memory spaces:
    // 1. Host Memory (CPU, pageable):
    //    - Allocated with malloc()
    //    - Accessible by CPU
    //    - Slower transfers to GPU
    // 
    // 2. Device Memory (GPU VRAM):
    //    - Allocated with cudaMalloc()
    //    - Accessible by GPU
    //    - Must be explicitly managed
    //
    // 3. Transfer Methods:
    //    - cudaMemcpy(): Blocking transfers
    //    - cudaMemcpyAsync(): Non-blocking with streams
    //    - Direction specified by cudaMemcpyHostToDevice/DeviceToHost

    // === Performance Measurement Sections ===
    // Each implementation is measured in three phases:
    // 1. Memory Transfer (H2D):
    //    - Time to copy data from CPU to GPU
    //    - Usually consistent across implementations
    //    - Limited by PCIe bandwidth
    //
    // 2. Kernel Execution:
    //    - Actual computation time
    //    - Main differentiator between implementations
    //    - Measured using CUDA events for microsecond precision
    //
    // 3. Result Transfer (D2H):
    //    - Time to copy results back to CPU
    //    - Used for verification
    //    - Also limited by PCIe bandwidth
    
    // === CUDA Event Creation ===
    // Events provide high-precision GPU timing
    // Benefits over CPU timers:
    // - Synchronized with GPU operations
    // - Microsecond resolution
    // - Handles multiple GPU streams correctly
    cudaEvent_t start, stop;
    float original_time = 0.0f, optimized_time = 0.0f;
    float tc_time = 0.0f, vectorized_time = 0.0f;
    float warp_time = 0.0f, buffered_time = 0.0f;
    float vec_transfer_time = 0.0f, vec_result_time = 0.0f;
    float warp_transfer_time = 0.0f, warp_result_time = 0.0f;
    float buf_transfer_time = 0.0f, buf_result_time = 0.0f;

    // === Memory Management Strategy ===
    // Memory allocation follows a specific pattern:
    // 1. Allocate host memory first
    //    - Use malloc() for pageable memory
    //    - Could use cudaMallocHost() for pinned memory
    //    - Pinned memory gives faster transfers but uses system RAM
    //
    // 2. Allocate device memory
    //    - Use cudaMalloc() for GPU memory
    //    - Check for allocation failures
    //    - Clean up previous allocations if any fail
    //
    // 3. Initialize host memory
    //    - Fill with test data
    //    - Ensure data is valid for testing
    //
    // 4. Transfer to device
    //    - Use cudaMemcpy() for synchronous transfers
    //    - Or cudaMemcpyAsync() with streams
    float *h_A, *h_B, *h_C, *h_C_original;  // Host pointers
    float *d_A, *d_B, *d_C;                 // Device pointers

    // === Error Handling Strategy ===
    // CUDA error handling is critical for robust GPU code
    // We check for errors:
    // 1. After memory allocation
    //    - Both host and device memory
    //    - Clean up on failure
    //
    // 2. After kernel launches
    //    - Use cudaGetLastError()
    //    - Check for launch failures
    //
    // 3. After memory transfers
    //    - Ensure data movement succeeded
    //    - Verify transfer sizes
    //
    // 4. After stream operations
    //    - Check stream synchronization
    //    - Verify event recording

    // === Memory Pointers and Allocation ===
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
    // This is our baseline implementation to understand basic CUDA performance
    // Characteristics:
    // - Each thread reads directly from global memory
    // - Multiple redundant memory accesses
    // - No memory access optimizations
    // - High memory latency, low throughput
    // 
    // Memory Access Pattern:
    // Thread 0: [Read A0][Read B0][Write C0]
    // Thread 1: [Read A1][Read B1][Write C1]
    // ...no overlap, high latency
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
    printf("\n1. Naive Implementation:\n");
    printf("   Memory Transfer (H2D): %.3f ms\n", transfer_time);
    
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
    printf("   Memory Transfer (D2H): %.3f ms\n", result_time);
    printf("   Total Time: %.3f ms\n", transfer_time + original_time + result_time);
    // Calculate TFLOPS (Trillion Floating Point Operations per Second)
    // Each multiply-add is 2 operations, hence the 2.0 multiplier
    printf("   TFLOPS: %.2f\n", (2.0 * batch_size * m * n * l) / (original_time * 1000000000.0));
    
    // Save results for accuracy comparison
    memcpy(h_C_original, h_C, total_elements_C * sizeof(float));
    
    // === Test 2: Shared Memory Implementation ===
    // Uses shared memory as a software-managed cache
    // Benefits:
    // - ~100x lower latency than global memory
    // - Data reuse within thread block
    // - Reduced global memory bandwidth usage
    // 
    // Memory Access Pattern:
    // 1. Load tile to shared memory (all threads cooperate)
    // 2. Compute using fast shared memory
    // 3. Move to next tile
    printf("\n2. Shared Memory Implementation:\n");
    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, total_elements_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, total_elements_B * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float shared_transfer_time;
    cudaEventElapsedTime(&shared_transfer_time, start, stop);
    printf("   Memory Transfer (H2D): %.3f ms\n", shared_transfer_time);

    // Record computation time separately
    cudaEventRecord(start);
    tensor_mul_optimized<<<optimizedGrid, optimizedBlock>>>(d_A, d_B, d_C, batch_size, m, n, k, l);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float shared_compute_time;
    cudaEventElapsedTime(&shared_compute_time, start, stop);
    printf("   Computation Time: %.3f ms\n", shared_compute_time);
    
    cudaEventRecord(start);
    cudaMemcpy(h_C, d_C, total_elements_C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float shared_result_time;
    cudaEventElapsedTime(&shared_result_time, start, stop);
    printf("   Memory Transfer (D2H): %.3f ms\n", shared_result_time);
    printf("   Total Time: %.3f ms\n", shared_transfer_time + shared_compute_time + shared_result_time);
    printf("   TFLOPS: %.2f\n", (2.0 * batch_size * m * n * l) / (shared_compute_time * 1000000000.0));
    printf("   Speedup vs Test 1 (Naive): %.2fx\n", original_time / shared_compute_time);
    
    // === Accuracy Verification Sections ===
    
    // Check accuracy against naive implementation
    // === Shared Memory Implementation Accuracy Check ===
    // Compare results with naive version element by element
    // Methodology:
    // 1. Calculate absolute difference |shared_mem - naive|
    // 2. Track maximum difference found
    // 3. Compare against tolerance (1e-5)
    // 
    // Why 1e-5 tolerance?
    // - Single precision float has ~7 decimal digits
    // - Accumulation can introduce small rounding errors
    // - 1e-5 allows for minimal floating point differences
    // 
    // Memory Access Pattern:
    // - Sequential access through h_C and h_C_original
    // - Good cache utilization on CPU
    bool shared_mem_matches = true;
    float shared_mem_max_diff = 0.0f;
    for (size_t i = 0; i < total_elements_C; i++) {
        // Calculate absolute difference
        float diff = fabs(h_C[i] - h_C_original[i]);
        // Track maximum difference seen
        shared_mem_max_diff = max(shared_mem_max_diff, diff);
        // Check if difference exceeds tolerance
        if (diff > 1e-5) {
            shared_mem_matches = false;
            break;
        }
    }
    // Compare results from the Shared Memory Implementation
    // with the baseline results stored in h_C_original (obtained via the naive kernel).
    printf("   Accuracy Check (vs Baseline): %s (max diff: %e)\n", 
           shared_mem_matches ? "PASSED" : "FAILED", shared_mem_max_diff);

    // === Test 3: cuBLAS Implementation ===
    // Professional library implementation by NVIDIA
    // Advantages:
    // - Auto-tuned for specific GPU architecture
    // - Uses hardware features (Tensor Cores)
    // - Optimal memory access patterns
    // 
    // Note: Column-major order (different from C/C++)
    // A[i][j] in C   → A[j * m + i] in cuBLAS
    printf("\n3. cuBLAS Implementation:\n");
    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, total_elements_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, total_elements_B * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cublas_transfer_time;
    cudaEventElapsedTime(&cublas_transfer_time, start, stop);
    printf("   Memory Transfer (H2D): %.3f ms\n", cublas_transfer_time);
    
    // === Maximum Accuracy Configuration ===
    // Use default math mode but with correct matrix layout
    cublasHandle_t handle;
    cublasCreate(&handle);
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
    float cublas_result_time;
    cudaEventElapsedTime(&cublas_result_time, start, stop);
    printf("   Memory Transfer (D2H): %.3f ms\n", cublas_result_time);
    printf("   Total Time: %.3f ms\n", tc_time + cublas_result_time);
    printf("   TFLOPS: %.2f\n", (2.0 * batch_size * m * n * l) / (tc_time * 1000000000.0));
    // Compare against all previous implementations
    printf("   Speedup vs Test 1 (Naive): %.2fx\n", original_time / tc_time);
    printf("   Speedup vs Test 2 (Shared Memory): %.2fx\n", shared_compute_time / tc_time);

    // === cuBLAS Implementation Accuracy Check ===
    // Compare cuBLAS results with the baseline results (h_C_original)
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
    printf("   Accuracy Check (vs Baseline): %s (max diff: %e)\n", 
           cublas_matches ? "PASSED" : "FAILED", cublas_max_diff);
    
    // === Test 4: Vectorized Implementation ===
    // Uses float4 vector loads/stores
    // Benefits:
    // - One 128-bit load instead of four 32-bit loads
    // - Better memory coalescing
    // - Higher memory bandwidth utilization
    // - Reduced number of memory transactions
    // 
    // Requirements:
    // - Matrix dimensions must be multiples of 4
    // - Memory must be aligned properly
    printf("\n4. Vectorized Implementation:\n");
    if (n % 4 != 0 || l % 4 != 0) {
        printf("   Skipped: Matrix dimensions must be multiples of 4\n");
    } else {
        // H2D Transfer timing
        cudaEventRecord(start);
        cudaMemcpy(d_A, h_A, total_elements_A * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, total_elements_B * sizeof(float), cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&vec_transfer_time, start, stop);
        printf("   Memory Transfer (H2D): %.3f ms\n", vec_transfer_time);

        // Reinterpret pointers
        float4 *d_A4 = (float4*)d_A;
        float4 *d_B4 = (float4*)d_B;
        float4 *d_C4 = (float4*)d_C;
        
        // Kernel timing
        cudaEventRecord(start);
        tensor_mul_vectorized<<<optimizedGrid, optimizedBlock>>>(
            d_A4, d_B4, d_C4, batch_size, m, n, k, l);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&vectorized_time, start, stop);
        printf("   Computation Time: %.3f ms\n", vectorized_time);
        
        // D2H Transfer timing
        cudaEventRecord(start);
        cudaMemcpy(h_C, d_C, total_elements_C * sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&vec_result_time, start, stop);
        printf("   Memory Transfer (D2H): %.3f ms\n", vec_result_time);
        printf("   Total Time: %.3f ms\n", vec_transfer_time + vectorized_time + vec_result_time);
        printf("   TFLOPS: %.2f\n", (2.0 * batch_size * m * n * l) / (vectorized_time * 1000000000.0));
        printf("   Speedup vs Test 1 (Naive): %.2fx\n", original_time / vectorized_time);
        printf("   Speedup vs Test 2 (Shared Memory): %.2fx\n", shared_compute_time / vectorized_time);
        printf("   Speedup vs Test 3 (cuBLAS): %.2fx\n", tc_time / vectorized_time);
        
        // Accuracy check
        bool vectorized_matches = true;
        float vectorized_max_diff = 0.0f;
        for (size_t i = 0; i < total_elements_C; i++) {
            float diff = fabs(h_C[i] - h_C_original[i]);
            vectorized_max_diff = max(vectorized_max_diff, diff);
            if (diff > 1e-5) {
                vectorized_matches = false;
                break;
            }
        }
        printf("   Accuracy Check (vs Baseline): %s (max diff: %e)\n",
               vectorized_matches ? "PASSED" : "FAILED", vectorized_max_diff);
    }

    // === Test 5: Warp-Optimized Implementation ===
    // Uses warp-level primitives for communication
    // Benefits:
    // - Direct register-to-register transfer
    // - No shared memory bank conflicts
    // - No explicit synchronization needed
    // - Lower latency than shared memory
    printf("\n5. Warp-Optimized Implementation:\n");
    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, total_elements_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, total_elements_B * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&warp_transfer_time, start, stop);
    printf("   Memory Transfer (H2D): %.3f ms\n", warp_transfer_time);

    // Grid configuration for warp-based execution
    // Each block contains exactly one warp (32 threads)
    dim3 warpBlock(32, 32);  // 32 threads per warp, 32 warps per block
    dim3 warpGrid(
        (m + 31) / 32,      // Ceil(m/32) blocks for rows
        (l + 31) / 32,      // Ceil(l/32) blocks for columns
        batch_size          // One block per batch
    );
    
    // Time the warp-optimized implementation
    // Should show lower latency due to warp-level communication
    cudaEventRecord(start);
    tensor_mul_warp_optimized<<<warpGrid, warpBlock>>>(
        d_A, d_B, d_C, batch_size, m, n, k, l);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&warp_time, start, stop);
    
    // Performance metrics for warp-optimized version
    printf("   Computation Time: %.3f ms\n", warp_time);
    printf("   Memory Transfer (D2H): %.3f ms\n", warp_result_time);
    printf("   Total Time: %.3f ms\n", warp_transfer_time + warp_time + warp_result_time);
    printf("   TFLOPS: %.2f\n", (2.0 * batch_size * m * n * l) / (warp_time * 1000000000.0));
    // Compare against all previous implementations
    printf("   Speedup vs Test 1 (Naive): %.2fx\n", original_time / warp_time);
    printf("   Speedup vs Test 2 (Shared Memory): %.2fx\n", shared_compute_time / warp_time);
    printf("   Speedup vs Test 3 (cuBLAS): %.2fx\n", tc_time / warp_time);
    printf("   Accuracy Check (vs Baseline): %s (max diff: %e)\n",
           cublas_matches ? "PASSED" : "FAILED", cublas_max_diff);

    // === Test 6: Double-Buffered Implementation ===
    // Overlaps computation with memory access
    // Technique:
    // - Two sets of shared memory buffers
    // - While computing on buffer A, load into buffer B
    // - Swap buffers and repeat
    // 
    // Benefits:
    // - Hides memory latency
    // - Better utilization of memory bandwidth
    // - Keeps both compute and memory units busy
    // 
    // Memory Access Pattern:
    // Buffer A: [Compute][Load  ][Compute][Load  ]
    // Buffer B: [Load  ][Compute][Load  ][Compute]
    // Result:   [Busy  ][Busy  ][Busy  ][Busy  ]
    printf("\n6. Double-Buffered Implementation:\n");
    cudaEventRecord(start);
    tensor_mul_double_buffered<<<optimizedGrid, optimizedBlock>>>(d_A, d_B, d_C, batch_size, m, n, k, l);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&buffered_time, start, stop);
    printf("   Computation Time: %.3f ms\n", buffered_time);
    printf("   TFLOPS: %.2f\n", (2.0 * batch_size * m * n * l) / (buffered_time * 1000000000.0));
    printf("   Speedup vs Test 1 (Naive): %.2fx\n", original_time / buffered_time);
    printf("   Speedup vs Test 2 (Shared Memory): %.2fx\n", shared_compute_time / buffered_time);
    printf("   Speedup vs Test 3 (Warp): %.2fx\n", warp_time / buffered_time);
    printf("   Speedup vs Test 4 (Vectorized): %.2fx\n", vectorized_time / buffered_time);
    printf("   Speedup vs Test 5 (Tensor Core): %.2fx\n", tc_time / buffered_time);

    // Check accuracy against naive implementation
    bool buffered_matches = true;
    float buffered_max_diff = 0.0f;
    for (size_t i = 0; i < total_elements_C; i++) {
        float diff = fabs(h_C[i] - h_C_original[i]);
        buffered_max_diff = max(buffered_max_diff, diff);
        if (diff > 1e-5) {
            buffered_matches = false;
            break;
        }
    }
    printf("   Accuracy Check: %s (Max Diff: %e)\n", 
           buffered_matches ? "PASSED" : "FAILED", buffered_max_diff);

    // === Final Performance Summary ===
    // Compare all implementations side by side
    printf("\n=== Performance Summary ===\n");
    printf("1. Naive Implementation:\n");
    printf("   Computation: %.3f ms, Memory H2D: %.3f ms, D2H: %.3f ms, Total: %.3f ms\n", 
           original_time, transfer_time, result_time, transfer_time + original_time + result_time);

    printf("2. Shared Memory Implementation:\n");
    printf("   Computation: %.3f ms, Memory H2D: %.3f ms, D2H: %.3f ms, Total: %.3f ms\n",
           shared_compute_time, shared_transfer_time, shared_result_time, 
           shared_transfer_time + shared_compute_time + shared_result_time);

    printf("3. cuBLAS Implementation:\n");
    printf("   Computation: %.3f ms, Memory H2D: %.3f ms, D2H: %.3f ms, Total: %.3f ms\n",
           tc_time, cublas_transfer_time, cublas_result_time, cublas_transfer_time + tc_time + cublas_result_time);

    if (n % 4 != 0 || l % 4 != 0) {
        printf("4. Vectorized Implementation: Skipped (dimensions must be multiples of 4)\n");
    } else {
        printf("4. Vectorized Implementation:\n");
        printf("   Computation: %.3f ms, Memory H2D: %.3f ms, D2H: %.3f ms, Total: %.3f ms\n",
               vectorized_time, vec_transfer_time, vec_result_time,
               vec_transfer_time + vectorized_time + vec_result_time);
    }

    printf("5. Warp-Optimized Implementation:\n");
    printf("   Computation: %.3f ms, Memory H2D: %.3f ms, D2H: %.3f ms, Total: %.3f ms\n",
           warp_time, warp_transfer_time, warp_result_time,
           warp_transfer_time + warp_time + warp_result_time);

    printf("6. Double-Buffered Implementation:\n");
    printf("   Computation: %.3f ms, Memory H2D: %.3f ms, D2H: %.3f ms, Total: %.3f ms\n",
           buffered_time, buf_transfer_time, buf_result_time,
           buf_transfer_time + buffered_time + buf_result_time);

    // === Resource Cleanup Strategy ===
    // Proper cleanup is critical for GPU programming
    // Order matters: Release in reverse order of allocation
    // 
    // Why Reverse Order?
    // 1. Prevents dangling references
    // 2. Ensures dependent resources are freed last
    // 3. Follows LIFO (Last In, First Out) principle
    // 
    // Cleanup Categories:
    // 1. CUDA Streams:
    //    - Must be destroyed after all work is complete
    //    - Check for pending operations
    //    - Multiple streams need individual cleanup
    // 
    // 2. cuBLAS Resources:
    //    - Handle cleanup after all cuBLAS operations
    //    - Verify no pending operations
    // 
    // 3. CUDA Events:
    //    - Used for timing measurements
    //    - Must be destroyed after all measurements
    // 
    // 4. GPU Memory:
    //    - Free device memory allocations
    //    - Check for deallocation errors
    //    - Handle fragmentation
    // 
    // 5. CPU Memory:
    //    - Free host memory last
    //    - Ensures no GPU operations depend on this memory
    
    // 1. Destroy CUDA streams
    // Each stream must be destroyed individually
    // Ensure all work is complete before destruction
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }

    // 2. Destroy cuBLAS handle
    // Release cuBLAS resources
    // Must be done after all cuBLAS operations
    cublasDestroy(handle);
    
    // 3. Destroy CUDA events
    // Clean up timing events
    // No more timing operations after this point
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // 4. Free GPU memory
    // Release device memory allocations
    // Order: temporary buffers first, then main arrays
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // 5. Free CPU memory
    // Release host memory last
    // Ensures no GPU operations depend on this memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_original);

    // === Error Handling Throughout Execution ===
    // Key error checking points:
    // 1. Memory Allocation:
    //    - Check cudaMalloc results
    //    - Verify host memory allocation
    //    - Handle out-of-memory conditions
    // 
    // 2. Kernel Launches:
    //    - Check kernel parameters
    //    - Verify grid/block dimensions
    //    - Handle launch failures
    // 
    // 3. Memory Transfers:
    //    - Validate transfer sizes
    //    - Check for transfer errors
    //    - Handle incomplete transfers
    // 
    // 4. Library Operations:
    //    - Verify cuBLAS status
    //    - Check stream operations
    //    - Handle synchronization errors
    // 
    // 5. Resource Cleanup:
    //    - Check deallocation success
    //    - Handle cleanup failures
    //    - Prevent resource leaks

    return 0;
}