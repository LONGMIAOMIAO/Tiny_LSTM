#include "kernel.h"
#include <stdio.h>
#include <vector>
#define TPB 32

__device__ float distance( float x1, float x2 )
{
    return sqrt( (x2 - x1) * ( x2 - x1) );
}

__global__ void distanceKernel( float* d_out, float* d_in, float ref )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    d_out[i] = distance(d_in[i], ref);
    printf( "%f\n", d_out[i] );
}

void distanceArray( float* out, float* in, float ref, int n )
{
    std::vector<int> vec;

    float* d_in = 0;
    float* d_out = 0;

    cudaMalloc( &d_in, n*sizeof(float) );
    cudaMalloc( &d_out, n*sizeof(float) );

    cudaMemcpy( d_in, in, n*sizeof( float ), cudaMemcpyHostToDevice );

    distanceKernel<<<n/TPB, TPB>>>( d_out, d_in, ref );

    cudaMemcpy( out, d_out, n*sizeof(float), cudaMemcpyDeviceToHost );

    cudaFree( d_in );
    cudaFree( d_out );
}