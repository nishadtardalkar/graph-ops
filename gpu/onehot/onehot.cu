#include<time.h>
#include<iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)


__global__ void onehot_m(float* in, float* c, float* out, int maxc, int n, int batchsize) {

    int i = blockIdx.x * 256 + threadIdx.x;
    int b = blockIdx.y;

    if (i >= c[b]) { return; }

    i += b * maxc;

    out[b * n + (int)(in[i])] = 1;
}

DLLEXPORT void cuda_onehot_m(float* din, float* dc, float* dout, int maxc, int n, int batchsize) {
    cudaMemset(dout, 0, sizeof(float) * n * batchsize);
    dim3 gridDims((int)ceil((float)maxc / 256), batchsize, 1);
    dim3 blockDims(256, 1, 1);
    onehot_m << <gridDims, blockDims >> > (din, dc, dout, maxc, n, batchsize);
}
