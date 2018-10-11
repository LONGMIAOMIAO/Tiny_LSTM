#ifndef KERNEL_H
#define KERNEL_H

// //float scale( int i, int n );
// __device__ float distance( float x1, float x2 );

// __global__ void distanceKernel( float* d_out, float* d_in, float ref );

void distanceArray( float* out, float* in, float ref, int n );

#endif