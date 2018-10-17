#include <stdio.h>
#include <math.h>
__global__ 
void cal_1( double* in, double* out )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = sqrt(sqrt(sqrt( in[i] )));
    //out[i] = in[i];
}

int main()
{
    double* in;
    double* out;

    cudaMallocManaged( &in, 1024 * 256 * sizeof(double) );
    cudaMallocManaged( &out, 1024 * 256 * sizeof(double) );

    for( int i = 0; i < 1024 * 256; i++ )
    {
        in[i] = i;
    }
    
    cal_1<<< 1024, 256 >>>( in, out );

    cudaDeviceSynchronize();

    double sum = 0;
    for( int i = 0; i < 1024 * 256; i++ )
    {
        sum += out[i];
    }
    printf( "%f\n", sum );
    
    double sum_2 = 0;
    for( int i = 0; i < 1024 * 256; i++ )
    {
        sum_2 += sqrt(sqrt(sqrt(in[i])));
    }
    printf( "%f\n", sum_2 );

    cudaFree(in);
    cudaFree(out);
    return 0;
}