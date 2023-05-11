#include<time.h>
#include<iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)


__global__ void onehotdensesd_fu(float* in, float* y, float* w, float* wgrad, int M, int K, int N) {

    int m = blockIdx.x * 256 + threadIdx.x;
    int n = blockIdx.y;

    if (m >= M) {
        return;
    }

    float e = 2 * ((((int)y[m] == n) ? 1 : 0) - w[(int)(in[m]) * N + n]);
    
    int k = (int)in[m];
    atomicAdd(w + k * N + n, e);
}


DLLEXPORT void cuda_onehotdensesd_fu(float* din, float* dy, float* dw, float* dwgrad, int M, int K, int N) {
    cudaMemset(dwgrad, 0, sizeof(float) * K * N);
    dim3 gridDims((int)ceil((float)M / 256), N, 1);
    dim3 blockDims(256, 1, 1);
    onehotdensesd_fu << <gridDims, blockDims >> > (din, dy, dw, dwgrad, M, K, N);
}
