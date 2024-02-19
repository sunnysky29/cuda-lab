/*
实现数组+1
nvcc -o tmp/sum sum1.cu

参考： [通过例子理解CUDA driver api和CUDA PTX](https://zhuanlan.zhihu.com/p/595143420)
*/

// sum.cu
#include <cuda.h>
#include <stdio.h>

__global__ void sum(int *x, int *y, int *z) {
  int tid = threadIdx.x;
  x[tid] += 1;
}

int main() {
  int N = 32;
  int nbytes = N * sizeof(int);
  int *dx = NULL, *hx = NULL;
  // 申请显存
  cudaMalloc((void **)&dx, nbytes);

  // 申请成功
  if (dx == NULL) {
    printf("GPU alloc fail");
    return -1;
  }

  // 申请CPU内存
  hx = (int *)malloc(nbytes);
  if (hx == NULL) {
    printf("CPU alloc fail");
    return -1;
  }

  // init: hx: 0..31
  printf("hx original:\n");
  for (int i = 0; i < N; i++) {
    hx[i] = i;
    printf("%d ", hx[i]);
  }
  printf("\n");

  // copy to GPU
  cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);

  // call GPU
  sum<<<1, N>>>(dx, dx, dx);

  // let gpu finish
  cudaThreadSynchronize();

  // copy data to CPU
  cudaMemcpy(hx, dx, nbytes, cudaMemcpyDeviceToHost);

  printf("hx after:\n");
  for (int i = 0; i < N; i++) {
    printf("%d ", hx[i]);
  }
  printf("\n");
  cudaFree(dx);
  free(hx);
  return 0;
}