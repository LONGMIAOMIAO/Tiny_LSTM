#include <stdio.h>
#include <math.h>
int main()
{
    float* left = (float*)malloc( 60 * sizeof(float) );
    float* w = (float*)malloc( 60 * sizeof(float) );
    float* out = (float*)malloc( 60 * sizeof(float) );

    //int *p = (int *)malloc(n * sizeof(int));

    float s = 0 + 59*60;
    float p = 59;
    for( int i = 0; i < 2 * 30 ; i++ )
    {
        left[i] = s;
        w[i] = p;
        s += 1.0;
        p += 60.0;
    }

    float sum = 0;
    for( int i = 0; i < 2 * 30 ; i++ )
    {
        sum += left[i] * w[i];
    }
    printf( "%f\n", sum );
}