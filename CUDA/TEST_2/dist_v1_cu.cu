#include <stdio.h>
#define N 64
#define TPB 64

//  Creating Data to be Calculated
__device__ float scale( int i, int n )
{
    return ((float)i) / (n - 1);
}
//  Calculating Distance Between X1 And X2
__device__ float distance( float x1, float x2 )
{
    return sqrt( (x1 - x2)* (x1 - x2) );
}
//  Caculating By CUDA WOWOWOWOW!
__global__ void distance_Kernel( float* d_out, float ref, int len )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float x = scale(i, len);
    d_out[i] = distance(x,ref);
    printf( "%f\n",d_out[i] );
    //std::cout << d_out[i] << std::endl;
}

int main()
{
    const float ref = 0.5f;
    float* d_out = 0;

    cudaMalloc(&d_out,N*sizeof(float));

    distance_Kernel<<<N/TPB,TPB>>>(d_out,ref,N);

    cudaFree(d_out);
    return 0;
}