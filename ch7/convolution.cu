__global__ void conv2d_basic_kernel(float *N, float *F, float *P, int r, int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    float Pvalue = 0;
    int inRow, inCol;

    for (int fRow = 0; fRow < 2*r+1; fRow++) {
        for (int fCol = 0; fCol < 2*r+1; fCol++) {
            inRow = outRow - r + fRow;
            inCol = outCol - r + fCol;

            if (inRow>=0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += F[fRow][fCol] * N[inRow*width + inCol];
            }
        }
    }

    P[outRow][outCol] = Pvalue;
}




    