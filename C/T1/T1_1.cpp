#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_COLS 20
#define MAX_INPUT 100

void T1_1()
{
    float vec[3][3];
    vec[0][0] = 12.3;
    vec[0][1] = 13.9;

    printf( "%f\n", vec[0][0] );
    printf( "%f\n", vec[0][1] );
    //printf( "%f\n", vec[0][2] );
    printf( "%f\n", 66.99 );

    //double (*d)[n] = new double[m][n]；

    float (*d)[10] = new float[20][10];

    d[0][0] = 33.33;
    d[0][1] = 98.45;

    printf( "%f\n", d[0][0] );
    printf( "%f\n", d[0][1] );
}