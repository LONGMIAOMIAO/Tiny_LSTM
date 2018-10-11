#include <stdio.h>
#define N 64
#define TPB 32

float scale( int i, int n )
{
    return ( (float)i / (n - 1) );
}

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

int main()
{
    const float ref = 0.5f;

    float* in = 0;
    float* out = 0;

    cudaMallocManaged( &in, N* sizeof(float) );
    cudaMallocManaged( &out, N* sizeof(float) );

    for(int i = 0; i < N; i++)
    {
        in[i] = scale(i, N);
    }

    distanceKernel<<<N/TPB, TPB>>>( out, in, ref );

    cudaFree(in);
    cudaFree(out);
    return 0;
}