#include "kernel.h"
#include <stdlib.h>
#include <vector>
#define N 64

float scale( int i, int n )
{
    return ( (float)i / (n - 1) );
}

int main()
{
    std::vector<int> vec;
    int s = 13;
    vec.push_back( std::move(s) );

    const float ref = 0.5f;

    float* in = (float*)calloc(N, sizeof(float));
    float* out = (float*)calloc(N, sizeof(float));

    for(int i = 0; i < N; i++)
    {
        in[i] = scale(i, N);
    }

    distanceArray( out, in, ref, N );

    free(in);
    free(out);
    return 0;
}