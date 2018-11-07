#include <stdio.h>
#include <math.h>

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

    cudaMallocManaged( &left, 2 * 2 * 30 * 30 * sizeof(float) );
    cudaMallocManaged( &w, 2 * 2 * 30 * 30 * sizeof(float) );
    cudaMallocManaged( &out, 2 * 2 * 30 * 30 * sizeof(float) );

    float s = 1;
    for( int i = 0; i < 2 * 2 * 30 * 30; i++ )
    {
        left[i] = s;
        w[i] = s;
        //s += 1;
    }

    dim3 DimGrid(2, 2, 1);
    dim3 DimBlock(30, 30, 1);

    cal<<< DimGrid, DimBlock >>>( left, out, w, 30 * 2, 30 * 2, 30 * 2, 30 * 2 );


    cudaDeviceSynchronize();

    float num = 0;
    for( int i = 0; i < 3600; i++ )
    {
        if( out[i] != 60 )
        {
            num++;
        }
    }

    printf( "%f\n", out[0] );
    printf( "%f\n", out[1] );
    printf( "%f\n", out[2] );
    printf( "%f\n", out[3599] );
    //printf( "%f\n", out[3600] );

    printf( "%f\n", 3333333333333.0 );
    printf( "%f\n", num );

    return 0;
}