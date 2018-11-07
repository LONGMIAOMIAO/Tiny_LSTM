#include <stdio.h>
#include <math.h>
// __global__ 
// void cal_1( double* in, double* out )
// {
//     const int i = blockIdx.x * blockDim.x + threadIdx.x;
//     out[i] = sqrt(sqrt(sqrt( in[i] )));
// }

template <typename T>
__global__
void cal( T* left, T* out )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = left[i] * 3;
}

int main()
{
    float* left;
    float* out;

    cudaMallocManaged( &left, 2 * 512 * sizeof(float) );
    cudaMallocManaged( &out, 2 * 512 * sizeof(float) );

    float s = 0;
    for( int i = 0; i < 2 * 512; i++ )
    {
        left[i] = i / 2.0 / 512.0 ;
        s = s + left[i];
    }
    printf( "%f\n", s * 3 );

    cal<<< 2, 512 >>>( left, out );

    cudaDeviceSynchronize();

    float num = 0;
    for( int i = 0; i < 2* 512; i++ )
    {
        num = num + out[i];
    }

    printf( "%f\n", num );

    return 0;
}