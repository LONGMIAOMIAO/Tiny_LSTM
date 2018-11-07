#include <stdio.h>
#include <math.h>

template <typename T>
struct Mat
{
    int width;
    int height;
    T* elements;
};

template <typename T>
__device__ 
T getElement( Mat<T> *A, int row, int col )
{
    return A->elements[ row * A->width + col ];
}

template <typename T>
__device__
void setElement( Mat<T>* A, int row, int col, T value )
{
    A->elements[ row * A->width + col ] = value;
}

template <typename T>
__global__
void matMulKernel(Mat<T> *A, Mat<T> *B, Mat<T> *C)
{
    T cvalue = 0;

	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = 0; i < A->width; ++i)
	{
		cvalue += getElement(A, row, i) * getElement(B, i, col);
	}
	setElement(C, row, col, cvalue);
}

void T_1()
{
    Mat<float>* left;
    Mat<float>* w;
    Mat<float>* out;

    cudaMallocManaged( &left, sizeof(Mat<float>) );
    cudaMallocManaged( &w, sizeof(Mat<float>) );
    cudaMallocManaged( &out, sizeof(Mat<float>) );

    cudaMallocManaged( &left->elements, 784 * sizeof(float) );
    cudaMallocManaged( &w->elements, 784 * 10 * sizeof(float) );
    cudaMallocManaged( &out->elements, 10 * sizeof(float) );

    left->width = 784;
    left->height = 1;

    w->width = 10;
    w->height = 784;

    out->width = 10;
    out->height = 1;

    float s = 1;
    for( int i = 0; i < 784; i++ )
    {
        left->elements[i] = s;
    }

    float t = 1;
    for( int i = 0; i < 784*10; i++ )
    {
        w->elements[i] = t;
    }


    dim3 DimGrid(1, 1, 1);
    dim3 DimBlock(10, 1, 1);

    matMulKernel<float><<< DimGrid, DimBlock >>>( left, w, out );

    cudaDeviceSynchronize();

    float num = 0;
    for( int i = 0; i < 10; i++ )
    {
        if( out->elements[i] != 78.4 )
        {
            num++;
        }
    }

    printf( "%f\n", out->elements[0] );
    printf( "%f\n", out->elements[1] );
    printf( "%f\n", out->elements[2] );
    printf( "%f\n", out->elements[9] );

    printf( "%f\n", num );
}

void T_2()
{
    Mat<float>* left;
    Mat<float>* w;
    Mat<float>* out;

    cudaMallocManaged( &left, sizeof(Mat<float>) );
    cudaMallocManaged( &w, sizeof(Mat<float>) );
    cudaMallocManaged( &out, sizeof(Mat<float>) );

    cudaMallocManaged( &left->elements, 784 * sizeof(float) );
    cudaMallocManaged( &w->elements, 10 * sizeof(float) );
    cudaMallocManaged( &out->elements, 784 * 10 * sizeof(float) );

    left->width = 1;
    left->height = 784;

    w->width = 10;
    w->height = 1;

    out->width = 10;
    out->height = 784;

    float s = 1;
    for( int i = 0; i < 784; i++ )
    {
        left->elements[i] = s;
        s++;
    }

    float t = 1;
    for( int i = 0; i < 10; i++ )
    {
        w->elements[i] = t;
        t++;
    }

    dim3 DimGrid(1, 16*7, 1);
    dim3 DimBlock(10, 7, 1);

    matMulKernel<float><<< DimGrid, DimBlock >>>( left, w, out );

    cudaDeviceSynchronize();

    float num = 0;
    for( int i = 0; i < 784 * 10; i++ )
    {
        if( out->elements[i] == 0 )
        {
            num++;
        }
    }

    printf( "%f\n", out->elements[0] );
    printf( "%f\n", out->elements[1] );
    printf( "%f\n", out->elements[2] );
    printf( "%f\n", out->elements[784*10-555] );
    printf( "%f\n", out->elements[784*10-3] );
    printf( "%f\n", out->elements[784*10-1] );
    //printf( "%f\n", out->elements[784*10] );

    printf( "%f\n", num );
}

int main()
{
    T_1();
    //T_2();
    //T_1();
    T_2();
    return 0;
}