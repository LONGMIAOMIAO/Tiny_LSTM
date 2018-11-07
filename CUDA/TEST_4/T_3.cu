#include <stdio.h>
#include <math.h>

// template <typename T>
// __global__
// void cal( T* left, T* out )
// {
//     const int i = blockIdx.x * blockDim.x + threadIdx.x;
//     out[i] = left[i] * 3;
// }

template <typename T>
__global__
void cal( T* left, T* out, T* w, int left_Row, int left_Col, int w_Row, int w_Col )
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    T sum = 0;
    for ( int i = 0; i < left_Col; ++i )
    {
        sum += left[ row * left_Col + i ] * w[ i * w_Col + col ];
    } 
    out[ row * w_Col + col ] = sum;
}

int main()
{
    float* left;
    float* w;
    float* out;

    cudaMallocManaged( &left, 1 * 6 * sizeof(float) );
    cudaMallocManaged( &w, 1 * 6 * sizeof(float) );
    cudaMallocManaged( &out, 1 * 4 * sizeof(float) );

    left[0] = 0.1;
    left[1] = 0.2;
    left[2] = 0.3;
    left[3] = 0.4;
    left[4] = 0.5;
    left[5] = 0.6;

    w[0] = 1;
    w[1] = 2;
    w[2] = 3;
    w[3] = 4;
    w[4] = 5;
    w[5] = 6;

    dim3 DimGrid(1, 1, 1);
    dim3 DimBlock(2, 2, 1);

    cal<<< DimGrid, DimBlock >>>( left, out, w, 2, 3, 3, 2 );
    cudaDeviceSynchronize();

    printf( "%f\n", out[0] );
    printf( "%f\n", out[1] );
    printf( "%f\n", out[2] );
    printf( "%f\n", out[3] );

    return 0;
}


// // Compute C = A * B , Matrix C = hA * wB = rowA * columnB
// __global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
//                                int numAColumns, int numBRows, int numBColumns,
//                                int numCRows, int numCColumns) {
//   //@@ Insert code to implement matrix multiplication here
//      float sum = 0.0f;

//     int row = blockIdx.y*blockDim.y + threadIdx.y;
//     int col = blockIdx.x*blockDim.x + threadIdx.x;


//     if(row < numCRows && col < numCColumns){
//         for (int i = 0; i < numAColumns; ++i)
//         {
//             sum += A[row*numAColumns + i] * B[i*numBColumns + col];
//         }
//         C[row*numBColumns + col] = sum;
//     }
//     printf("C = %f\n",C[row*numBColumns + col]);

// }