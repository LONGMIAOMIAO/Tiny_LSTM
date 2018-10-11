#include <stdlib.h>
#include <stdio.h>
#include "F_1.h"

#define N 20000000

//  nvcc dist_v2.cpp F_1.cpp
int main()
{
    float* in = (float*)calloc(N, sizeof(float));
    float* out = (float*)calloc(N, sizeof(float));
    const float ref = 0.5f;

    for( int i = 0; i < N; i++ )
    {
        in[i] = scale(i,N);
    }

    distanceArray( out, in, ref, N );

    // for( int i = 0; i < N; i++ )
    // {
    //     printf( "%f\n" ,out[i] );
    // }

    free(in);
    free(out);
    return 0;
}