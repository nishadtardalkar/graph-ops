#include<time.h>
#include<iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)


__global__ void softmax_s(float* in, float* out, int n, int batchsize) {

    int b = blockIdx.x * 256 + threadIdx.x;

    if (b >= batchsize) { return; }

    float s = 0;
    float* inb = in + b * n;
    for (int i = 0; i < n; i++) {
        s += expf(inb[i]);
    }
    out[b] = s;
}

__global__ void softmax_f(float* in, float* sum, float* out, int n, int batchsize) {

    int i = blockIdx.x * 256 + threadIdx.x;
    int b = blockIdx.y;

    if (i >= n) { return; }

    out[b * n + i] = in[b * n + i] / sum[b];
}

DLLEXPORT void cuda_softmax_f(float* din, float* dsum, float* dout, int n, int batchsize) {
    dim3 s_gd((int)ceil((float)batchsize / 256), 1, 1);
    dim3 s_bd(256, 1, 1);
    softmax_s << <s_gd, s_bd >> > (din, dsum, n, batchsize);

    dim3 f_gd((int)ceil((float)n / 256), batchsize, 1);
    dim3 f_bd(256, 1, 1);
    softmax_f << <f_gd, f_bd >> > (din, dsum, dout, n, batchsize);
}
