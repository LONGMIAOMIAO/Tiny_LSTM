#include <stdio.h>
#include <math.h>
#include <fstream>  
#include <sstream>  
#include <iostream>
#include <map>
#include <time.h>

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

//  =============================================================================================

template <typename T>
__device__ 
T cal( Mat<T>* A, Mat<T>* B, int col, int i)
{
    //return A->elements[ row * A->width + col ];
    T val = 0;

    for( int j = 0; j < 784; ++j )
    {
        val += abs( getElement(A, col, j) - getElement(B, i, j) ); 
    }
    return val;
}

template <typename T>
__global__
void calDistance( Mat<T>* A, Mat<T>* B, int i, Mat<T>* C )
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    C->elements[col] = cal<T>( A, B, col, i );
    //C->elements[col] = 3.0;
}



template <typename T>
void loadMnist( Mat<T>*& mat_L_Tr, Mat<T>*& mat_R_Tr, Mat<T>* mat_L_Te, Mat<T>* mat_R_Te )
{
    std::ifstream L_Tr_File( "../../../DATA/MNIST/train.csv", std::ios::in );
    std::string L_Tr_Str;
    int L_Tr_Num = 0;
    while (std::getline(L_Tr_File, L_Tr_Str))
    {
        //  printf( "121" );
        std::stringstream ss(L_Tr_Str);
        std::string str;
        int inNum = 0;
        while (std::getline(ss, str, ','))
        {
            //  ORIGDATA::mat_L_Tr_2D->data_f(L_Tr_Num,inNum) = atoi( str.c_str() );
            mat_L_Tr->elements[ L_Tr_Num * 784 + inNum ] = atoi( str.c_str() );
            //  printf( "%f", atof( str.c_str() ) );
            //  printf( "\n" );
            inNum++;
        }
        L_Tr_Num++;
    }

    std::ifstream R_Tr_File( "../../../DATA/MNIST/trainL.csv", std::ios::in );
    std::string R_Tr_Str;
    int R_Tr_Num = 0;
    while (std::getline(R_Tr_File, R_Tr_Str))
    {
        std::stringstream ss(R_Tr_Str);
        std::string str;
        int inNum = 0;
        while (std::getline(ss, str, ','))
        {
            //  ORIGDATA::mat_R_Tr_2D->data_b(R_Tr_Num,inNum) = atoi( str.c_str() );
            mat_R_Tr->elements[ R_Tr_Num * 10 + inNum ] = atoi( str.c_str() );
            //  printf( "%f", atof( str.c_str() ) );
            //  printf( "\n" );
            inNum++;
        }
        R_Tr_Num++;
    }

    std::ifstream L_Te_File( "../../../DATA/MNIST/test.csv", std::ios::in );
    std::string L_Te_Str;
    int L_Te_Num = 0;
    while (std::getline(L_Te_File, L_Te_Str))
    {
        std::stringstream ss(L_Te_Str);
        std::string str;
        int inNum = 0;
        while (std::getline(ss, str, ','))
        {
            //  ORIGDATA::mat_L_Te_2D->data_f(L_Te_Num,inNum) = atoi( str.c_str() );
            mat_L_Te->elements[ L_Te_Num * 784 + inNum ] = atoi( str.c_str() );
            inNum++;
        }
        L_Te_Num++;
    }

    std::ifstream R_Te_File( "../../../DATA/MNIST/testL.csv", std::ios::in );
    std::string R_Te_Str;
    int R_Te_Num = 0;
    while (std::getline(R_Te_File, R_Te_Str))
    {
        std::stringstream ss(R_Te_Str);
        std::string str;
        int inNum = 0;
        while (std::getline(ss, str, ','))
        {
            //  ORIGDATA::mat_R_Te_2D->data_b(R_Te_Num,inNum) = atoi( str.c_str() );
            mat_R_Te->elements[ R_Te_Num * 10 + inNum ] = atoi( str.c_str() );            
            inNum++;
        }
        R_Te_Num++;
    }
}


//  nvcc Mnist_KNN_CU.cu -O3
//  72s     96% correct
int main()
{
    auto start = clock();

    Mat<float>* mat_L_Tr;
    Mat<float>* mat_R_Tr;
    Mat<float>* mat_L_Te;
    Mat<float>* mat_R_Te;
    Mat<float>* mat_C;


    cudaMallocManaged( &mat_L_Tr, sizeof(Mat<float>) );
    cudaMallocManaged( &mat_R_Tr, sizeof(Mat<float>) );
    cudaMallocManaged( &mat_L_Te, sizeof(Mat<float>) );
    cudaMallocManaged( &mat_R_Te, sizeof(Mat<float>) );  
    cudaMallocManaged( &mat_C, sizeof(Mat<float>) );  

    cudaMallocManaged( &mat_L_Tr->elements, 55000 * 784 * sizeof(float) );
    cudaMallocManaged( &mat_R_Tr->elements, 55000 * 10  * sizeof(float) );
    cudaMallocManaged( &mat_L_Te->elements, 10000 * 784 * sizeof(float) );
    cudaMallocManaged( &mat_R_Te->elements, 10000 * 10  * sizeof(float) );
    cudaMallocManaged( &mat_C->elements, 55000 * sizeof(float) );

    mat_L_Tr->width     =    784;
    mat_L_Tr->height    =    55000;

    mat_R_Tr->width     =    10;
    mat_R_Tr->height    =    55000;

    mat_L_Te->width     =    784;
    mat_L_Te->height    =    10000;

    mat_R_Te->width     =    10;
    mat_R_Te->height    =    10000;

    mat_C->width    =   55000;
    mat_C->height   =   1;


    loadMnist<float>( mat_L_Tr, mat_R_Tr, mat_L_Te, mat_R_Te );

    dim3 DimGrid(55, 1, 1);
    dim3 DimBlock(1000, 1, 1);

    int totalNum = 0;
    for( int i = 0; i < 10000; i++ )
    {
        calDistance<float><<< DimGrid, DimBlock >>>( mat_L_Tr, mat_L_Te, i, mat_C );
        cudaDeviceSynchronize();

        std::pair<float,int> min_Pair;
        min_Pair.first = mat_C->elements[0];
        for( int j = 0; j < 55000; j++ )
        {
            if( mat_C->elements[j] <= min_Pair.first )
            {
                min_Pair.first = mat_C->elements[j];
                min_Pair.second = j;
            }    
        }
        // std::map<float, int> m_map;
        // for( int j = 0; j < 55000; j++ )
        // {
        //     m_map[mat_C->elements[j]] = j;
        // }
        // auto top_K = m_map.begin();
        // int seq = (*top_K).second;
        int seq = min_Pair.second;

        int r_val = -1;
        for( int k = 0; k < 10; k++ )
        {
            //if ( getElement(mat_R_Tr, seq, k) == 1 )
            if ( mat_R_Tr->elements[ seq * 10 + k ] == 1 )
            {
                r_val = k;
                break;
            }        
        }

        int l_val = -2;
        for( int k = 0; k < 10; k++ )
        {
            //if ( getElement(mat_R_Te, i, k) == 1 )
            if ( mat_R_Te->elements[ i * 10 + k ] == 1 )
            {
                l_val = k;
                break;
            }        
        }
        if( r_val == l_val )
        {
            totalNum++;
        }

        // std::map<float, int> m_map;
        // float s = mat_C->elements[0];
        // for( int j = 0; j < 55000; j++ )
        // {
        //     if( mat_C->elements[j] < s )
        //     {
        //         s = mat_C->elements[j];
        //     }
        //     m_map[mat_C->elements[j]] = j;
        // }

        // int v = 0;
        // for( auto u = m_map.begin(); u != m_map.end(); u++ )
        // {
        //     if(v==10) break;

        //     printf("%d", (*u).second );
        //     printf( "\n" );
        //     v++;
        // }
        
        // printf( "%f", s );
        // printf( "\n" );
    }



    // void calDistance( Mat<T>*& A, Mat<T>*& B, int i, Mat<T>*& C )
    // {
    // int col = threadIdx.x + blockIdx.x * blockDim.x;

    // C->elements[col] = cal<T>( A, B, col, i );
    // }

    //matMulKernel<float><<< DimGrid, DimBlock >>>( left, w, out );
    printf( "%d", totalNum );
    printf( "\n" );
    auto end = clock();
    printf( "%lf", (end - start) / 1000000.0 );
    return 0;
}