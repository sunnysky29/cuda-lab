/*
数组的平方和
运行：
  nvcc -o square square.cu
*/

// NOTES:
//  * Device: is the term of the GPU
//  * Host: is the term of the CPU
//  * kernels are the only things run in parallel on the GPU
//  * everything in the main is run on CPU
//  * memory transfers between Host (CPU) and Device (GPU) should be minimal
//  * kernels all run at the same time
//  * threads can know their Id's with threadIdx.x, blocks are similar
//

#include <stdio.h>

// kernel 
__global__ void square(float *d_out, float *d_in){
  int idx = threadIdx.x; // this is how you get the thread index
  float f = d_in[idx];
  d_out[idx] = f*f;
}

// main is here. this is the CPU code. 
int main(){
  // 声明数组大小
  const int ARRAY_SIZE = 64; 
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

  // generate the input array on the host
  // 主机数据，h_xx； device GPU 上数据，d_xx
  float h_in[ARRAY_SIZE];
  for (int i = 0; i < ARRAY_SIZE; i++){
    h_in[i] = float(i);
  }
  float h_out[ARRAY_SIZE];
  
  // declare GPU memory pointers
  float *d_in;
  float *d_out;
  //allocate GPU  memory
  cudaMalloc((void **) &d_in, ARRAY_BYTES);
  cudaMalloc((void **) &d_out, ARRAY_BYTES);

  // transfer the array to the GPU
  cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

  // launch the kernel
  // <<< block,  >>>
  square<<<1,ARRAY_SIZE>>>(d_out,d_in);

  // copy back the result array to the CPU
  cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  // print the result
  for (int i = 0; i < ARRAY_SIZE; i++){
    printf("%f", h_out[i]);
    printf(((i % 4) != 3) ? "\t" : "\n");
  }

  // free GPU memory allocation
  cudaFree(d_in);
  cudaFree(d_out);

  return 0;
}