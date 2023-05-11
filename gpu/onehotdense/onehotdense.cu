#include<time.h>
#include<iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)


__global__ void onehotdense_f(float* in, float* w, float* out, int M, int K, int N) {

    int m = blockIdx.x * 256 + threadIdx.x;
    int n = blockIdx.y;

    if (m >= M) {
        return;
    }

    out[m * N + n] = w[(int)(in[m]) * N + n];
}

__global__ void onehotdense_u(float* in, float* w, float* out, int M, int K, int N) {

    int m = blockIdx.x * 256 + threadIdx.x;
    int n = blockIdx.y;

    if (m >= M) {
        return;
    }

    int k = (int)in[m];
    atomicAdd(w + k * N + n, out[m * N + n]);
}

DLLEXPORT void cuda_onehotdense_f(float* din, float* dw, float* dout, int M, int K, int N) {
    dim3 gridDims((int)ceil((float)M / 256), N, 1);
    dim3 blockDims(256, 1, 1);
    onehotdense_f << <gridDims, blockDims >> > (din, dw, dout, M, K, N);
}

DLLEXPORT void cuda_onehotdense_u(float* din, float* dw, float* dout, int M, int K, int N) {
    cudaMemset(dw, 0, sizeof(float) * K * N);
    dim3 gridDims((int)ceil((float)M / 256), N, 1);
    dim3 blockDims(256, 1, 1);
    onehotdense_u << <gridDims, blockDims >> > (din, dw, dout, M, K, N);
}
