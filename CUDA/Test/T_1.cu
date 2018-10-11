#include <stdio.h>

__global__ void sayHellow(void)
{
    printf("Hello Outer!\n");
}

//  nvcc -arch sm_30 T_1.cu
//  nvcc -arch sm_60 T_1.cu
int main()
{
    printf("Hellow World!\n");

    sayHellow<<<1, 10>>>();

    //cudaDeviceSynchronize();
    cudaDeviceReset();
    
    return 0;
}