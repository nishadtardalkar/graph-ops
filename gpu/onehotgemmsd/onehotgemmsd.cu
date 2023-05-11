#include<time.h>
#include<iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)

struct float8 {
    float a, b, c, d, e, f, g, h;
};


__global__ void ohsgemm128sd_fub(float* A, float* B, float* out, float* wgrad, int M, int N, int K, float* e, float* bi) {

    __shared__ __align__(16) float sA[8][132];
    __shared__ __align__(16) float sB[8][128];
    __shared__ __align__(16) float sAb[8][132];
    __shared__ __align__(16) float sBb[8][128];

    int tx = threadIdx.x;
    int mb = blockIdx.z * 256 + (blockIdx.x / 2) * 128;
    int nb = blockIdx.y * 256 + (blockIdx.x % 2) * 128;

    if (mb >= M || nb >= N) {
        return;
    }

    float rC[8][8];
    float4 rA;
    float4 rB;
    float rAs[8];
    float rBs[8];
    float rAsb[8];
    float rBsb[8];

    float(*psA)[8][132] = &sA;
    float(*psB)[8][128] = &sB;

    int txm32 = tx % 32;
    int txb32 = tx / 32;
    int txm16 = tx % 16;
    int txb16 = tx / 16;
    int txm2 = tx % 2;
    int txb2 = tx / 2;
    int txm8 = tx % 8;
    int txb8 = tx / 8;
    int txm128 = tx % 128;
    int txb128 = tx / 128;
    int txm32b4 = txm32 / 4;

    int wnb = (txb32 / 4) * 64;
    int wmb = (txb32 % 4) * 32;

    int m, n, k, kb, kbb;

    float4 f4;
    float8 f8;

    int sAi1 = wmb + (txm32 % 2) * 8 + (txm32 / 16) * 16;
    int sBi1 = wnb + (txm32 / 2) * 4;
    int sBi2 = wnb + (32 + (txm32 / 2) * 4) % 64;

#pragma unroll
    for (m = 0; m < 8; m++) {
#pragma unroll
        for (n = 0; n < 8; n++) {
            rC[m][n] = 0;
        }
    }

    if (mb + txm128 < M) {
        int t = K - txb128 * 4;
        if (4 <= t) {
            rA.x = A[(mb + txm128) * K + txb128 * 4 + 0];
            rA.y = A[(mb + txm128) * K + txb128 * 4 + 1];
            rA.z = A[(mb + txm128) * K + txb128 * 4 + 2];
            rA.w = A[(mb + txm128) * K + txb128 * 4 + 3];
        }
        else if (3 <= t) {
            rA.x = A[(mb + txm128) * K + txb128 * 4 + 0];
            rA.y = A[(mb + txm128) * K + txb128 * 4 + 1];
            rA.z = A[(mb + txm128) * K + txb128 * 4 + 2];
            rA.w = 0;
        }
        else if (2 <= t) {
            rA.x = A[(mb + txm128) * K + txb128 * 4 + 0];
            rA.y = A[(mb + txm128) * K + txb128 * 4 + 1];
            rA.z = rA.w = 0;
        }
        else if (1 <= t) {
            rA.x = A[(mb + txm128) * K + txb128 * 4 + 0];
            rA.y = rA.z = rA.w = 0;
        }
        else {
            rA.x = rA.y = rA.z = rA.w = 0;
        }
    }
    else {
        rA.x = rA.y = rA.z = rA.w = 0;
    }
    if (txb32 < K) {
        int t = N - nb - txm32 * 4;
        if (4 <= t) {
            rB.x = B[txb32 * N + nb + txm32 * 4 + 0];
            rB.y = B[txb32 * N + nb + txm32 * 4 + 1];
            rB.z = B[txb32 * N + nb + txm32 * 4 + 2];
            rB.w = B[txb32 * N + nb + txm32 * 4 + 3];
        }
        else if (3 <= t) {
            rB.x = B[txb32 * N + nb + txm32 * 4 + 0];
            rB.y = B[txb32 * N + nb + txm32 * 4 + 1];
            rB.z = B[txb32 * N + nb + txm32 * 4 + 2];
            rB.w = 0;
        }
        else if (2 <= t) {
            rB.x = B[txb32 * N + nb + txm32 * 4 + 0];
            rB.y = B[txb32 * N + nb + txm32 * 4 + 1];
            rB.z = rB.w = 0;
        }
        else if (1 <= t) {
            rB.x = B[txb32 * N + nb + txm32 * 4 + 0];
            rB.y = rB.z = rB.w = 0;
        }
        else {
            rB.x = rB.y = rB.z = rB.w = 0;
        }
    }
    else {
        rB.x = rB.y = rB.z = rB.w = 0;
    }

    for (kb = 0; kb < K; kb += 8) {

        (*psA)[txb128 * 4 + 0][txm128] = rA.x;
        (*psA)[txb128 * 4 + 1][txm128] = rA.y;
        (*psA)[txb128 * 4 + 2][txm128] = rA.z;
        (*psA)[txb128 * 4 + 3][txm128] = rA.w;
        (*psB)[txb32][txm32 * 4 + 0] = rB.x;
        (*psB)[txb32][txm32 * 4 + 1] = rB.y;
        (*psB)[txb32][txm32 * 4 + 2] = rB.z;
        (*psB)[txb32][txm32 * 4 + 3] = rB.w;

        __syncthreads();

        if (kb + 8 < K) {
            if (mb + txm128 < M) {
                int t = K - kb - 8 - txb128 * 4;
                if (4 <= t) {
                    rA.x = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 0];
                    rA.y = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 1];
                    rA.z = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 2];
                    rA.w = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 3];
                }
                else if (3 <= t) {
                    rA.x = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 0];
                    rA.y = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 1];
                    rA.z = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 2];
                    rA.w = 0;
                }
                else if (2 <= t) {
                    rA.x = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 0];
                    rA.y = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 1];
                    rA.z = rA.w = 0;
                }
                else if (1 <= t) {
                    rA.x = A[(mb + txm128) * K + kb + 8 + txb128 * 4 + 0];
                    rA.y = rA.z = rA.w = 0;
                }
                else {
                    rA.x = rA.y = rA.z = rA.w = 0;
                }
            }
            else {
                rA.x = rA.y = rA.z = rA.w = 0;
            }
            if (kb + 8 + txb32 < K) {
                int t = N - nb - txm32 * 4;
                if (4 <= t) {
                    rB.x = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 0];
                    rB.y = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 1];
                    rB.z = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 2];
                    rB.w = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 3];
                }
                else if (3 <= t) {
                    rB.x = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 0];
                    rB.y = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 1];
                    rB.z = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 2];
                    rB.w = 0;
                }
                else if (2 <= t) {
                    rB.x = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 0];
                    rB.y = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 1];
                    rB.z = rB.w = 0;
                }
                else if (1 <= t) {
                    rB.x = B[(kb + 8 + txb32) * N + nb + txm32 * 4 + 0];
                    rB.y = rB.z = rB.w = 0;
                }
                else {
                    rB.x = rB.y = rB.z = rB.w = 0;
                }
            }
            else {
                rB.x = rB.y = rB.z = rB.w = 0;
            }
        }

        // COMPUTE -------------------

        f8 = *reinterpret_cast<float8*>((*psA)[0] + sAi1);
        rAs[0] = f8.a;
        rAs[1] = f8.b;
        rAs[2] = f8.c;
        rAs[3] = f8.d;
        rAs[4] = f8.e;
        rAs[5] = f8.f;
        rAs[6] = f8.g;
        rAs[7] = f8.h;
        f4 = *reinterpret_cast<float4*>((*psB)[0] + sBi1);
        rBs[0] = f4.x;
        rBs[1] = f4.y;
        rBs[2] = f4.z;
        rBs[3] = f4.w;
        f4 = *reinterpret_cast<float4*>((*psB)[0] + sBi2);
        rBs[4] = f4.x;
        rBs[5] = f4.y;
        rBs[6] = f4.z;
        rBs[7] = f4.w;

#pragma unroll
        for (k = 0; k < 8; k++) {

            if (k < 7) {
                f8 = *reinterpret_cast<float8*>((*psA)[k + 1] + sAi1);
                rAsb[0] = f8.a;
                rAsb[1] = f8.b;
                rAsb[2] = f8.c;
                rAsb[3] = f8.d;
                rAsb[4] = f8.e;
                rAsb[5] = f8.f;
                rAsb[6] = f8.g;
                rAsb[7] = f8.h;
                f4 = *reinterpret_cast<float4*>((*psB)[k + 1] + sBi1);
                rBsb[0] = f4.x;
                rBsb[1] = f4.y;
                rBsb[2] = f4.z;
                rBsb[3] = f4.w;
                f4 = *reinterpret_cast<float4*>((*psB)[k + 1] + sBi2);
                rBsb[4] = f4.x;
                rBsb[5] = f4.y;
                rBsb[6] = f4.z;
                rBsb[7] = f4.w;
            }

#pragma unroll
            for (m = 0; m < 8; m++) {
#pragma unroll
                for (n = 0; n < 8; n++) {
                    rC[m][n] += rAs[m] * rBs[n];
                }
            }

#pragma unroll
            for (m = 0; m < 8; m++) {
                rAs[m] = rAsb[m];
                rBs[m] = rBsb[m];
            }

        }

        // ---------------------------

        if (psA == &sA) {
            psA = &sAb;
            psB = &sBb;
        }
        else {
            psA = &sA;
            psB = &sB;
        }

    }


    // W grad

#pragma unroll
    for (m = 0; m < 8; m++) {
        float4 t;
        if (mb + sAi1 + m < M) {
            int t = N - nb - sBi1;
            if (4 <= t) {
                rC[m][0] = (out[mb + sAi1 + m] == nb + sBi1 + 0) ? rC[m][0] - 1 : rC[m][0];
                rC[m][1] = (out[mb + sAi1 + m] == nb + sBi1 + 1) ? rC[m][1] - 1 : rC[m][1];
                rC[m][2] = (out[mb + sAi1 + m] == nb + sBi1 + 2) ? rC[m][2] - 1 : rC[m][2];
                rC[m][3] = (out[mb + sAi1 + m] == nb + sBi1 + 3) ? rC[m][3] - 1 : rC[m][3];
                for (int i = 0; i < K; i++) {
                    atomicAdd(wgrad + i * N + (nb + sBi1 + 0), A[(mb + sAi1 + m) * K + i] * rC[m][0]);
                    atomicAdd(wgrad + i * N + (nb + sBi1 + 1), A[(mb + sAi1 + m) * K + i] * rC[m][1]);
                    atomicAdd(wgrad + i * N + (nb + sBi1 + 2), A[(mb + sAi1 + m) * K + i] * rC[m][2]);
                    atomicAdd(wgrad + i * N + (nb + sBi1 + 3), A[(mb + sAi1 + m) * K + i] * rC[m][3]);
                }
            }
            /*
            else if (3 <= t) {
                rC[m][0] = (out[mb + sAi1 + m] == nb + sBi1 + 0) ? rC[m][0] - 1 : rC[m][0];
                rC[m][1] = (out[mb + sAi1 + m] == nb + sBi1 + 1) ? rC[m][1] - 1 : rC[m][1];
                rC[m][2] = (out[mb + sAi1 + m] == nb + sBi1 + 2) ? rC[m][2] - 1 : rC[m][2];
                for (int i = 0; i < K; i++) {
                    atomicAdd(wgrad + i * N + (nb + sBi1 + 0), A[(mb + sAi1 + m) * K + i] * rC[m][0]);
                    atomicAdd(wgrad + i * N + (nb + sBi1 + 1), A[(mb + sAi1 + m) * K + i] * rC[m][1]);
                    atomicAdd(wgrad + i * N + (nb + sBi1 + 2), A[(mb + sAi1 + m) * K + i] * rC[m][2]);
                }
            }
            else if (2 <= t) {
                rC[m][0] = (out[mb + sAi1 + m] == nb + sBi1 + 0) ? rC[m][0] - 1 : rC[m][0];
                rC[m][1] = (out[mb + sAi1 + m] == nb + sBi1 + 1) ? rC[m][1] - 1 : rC[m][1];
                for (int i = 0; i < K; i++) {
                    atomicAdd(wgrad + i * N + (nb + sBi1 + 0), A[(mb + sAi1 + m) * K + i] * rC[m][0]);
                    atomicAdd(wgrad + i * N + (nb + sBi1 + 1), A[(mb + sAi1 + m) * K + i] * rC[m][1]);
                }
            }
            else if (1 <= t) {
                rC[m][0] = (out[mb + sAi1 + m] == nb + sBi1 + 0) ? rC[m][0] - 1 : rC[m][0];
                for (int i = 0; i < K; i++) {
                    atomicAdd(wgrad + i * N + (nb + sBi1 + 0), A[(mb + sAi1 + m) * K + i] * rC[m][0]);
                }
            }
            */
            t = N - nb - sBi2;
            if (4 <= t) {
                rC[m][4] = (out[mb + sAi1 + m] == nb + sBi2 + 0) ? rC[m][4] - 1 : rC[m][4];
                rC[m][5] = (out[mb + sAi1 + m] == nb + sBi2 + 1) ? rC[m][5] - 1 : rC[m][5];
                rC[m][6] = (out[mb + sAi1 + m] == nb + sBi2 + 2) ? rC[m][6] - 1 : rC[m][6];
                rC[m][7] = (out[mb + sAi1 + m] == nb + sBi2 + 3) ? rC[m][7] - 1 : rC[m][7];
                for (int i = 0; i < K; i++) {
                    atomicAdd(wgrad + i * N + (nb + sBi2 + 0), A[(mb + sAi1 + m) * K + i] * rC[m][4]);
                    atomicAdd(wgrad + i * N + (nb + sBi2 + 1), A[(mb + sAi1 + m) * K + i] * rC[m][5]);
                    atomicAdd(wgrad + i * N + (nb + sBi2 + 2), A[(mb + sAi1 + m) * K + i] * rC[m][6]);
                    atomicAdd(wgrad + i * N + (nb + sBi2 + 3), A[(mb + sAi1 + m) * K + i] * rC[m][7]);
                }
                /*
                C[(mb + sAi1 + m) * N + nb + sBi2 + 0] = rC[m][4];
                C[(mb + sAi1 + m) * N + nb + sBi2 + 1] = rC[m][5];
                C[(mb + sAi1 + m) * N + nb + sBi2 + 2] = rC[m][6];
                C[(mb + sAi1 + m) * N + nb + sBi2 + 3] = rC[m][7];
                */
            }
            /*
            else if (3 <= t) {
                rC[m][4] = (out[mb + sAi1 + m] == nb + sBi2 + 0) ? rC[m][4] - 1 : rC[m][4];
                rC[m][5] = (out[mb + sAi1 + m] == nb + sBi2 + 1) ? rC[m][5] - 1 : rC[m][5];
                rC[m][6] = (out[mb + sAi1 + m] == nb + sBi2 + 2) ? rC[m][6] - 1 : rC[m][6];
                for (int i = 0; i < K; i++) {
                    atomicAdd(wgrad + i * N + (nb + sBi2 + 0), A[(mb + sAi1 + m) * K + i] * rC[m][4]);
                    atomicAdd(wgrad + i * N + (nb + sBi2 + 1), A[(mb + sAi1 + m) * K + i] * rC[m][5]);
                    atomicAdd(wgrad + i * N + (nb + sBi2 + 2), A[(mb + sAi1 + m) * K + i] * rC[m][6]);
                }
            }
            else if (2 <= t) {
                rC[m][4] = (out[mb + sAi1 + m] == nb + sBi2 + 0) ? rC[m][4] - 1 : rC[m][4];
                rC[m][5] = (out[mb + sAi1 + m] == nb + sBi2 + 1) ? rC[m][5] - 1 : rC[m][5];
                for (int i = 0; i < K; i++) {
                    atomicAdd(wgrad + i * N + (nb + sBi2 + 0), A[(mb + sAi1 + m) * K + i] * rC[m][4]);
                    atomicAdd(wgrad + i * N + (nb + sBi2 + 1), A[(mb + sAi1 + m) * K + i] * rC[m][5]);
                }
            }
            else if (1 <= t) {
                rC[m][4] = (out[mb + sAi1 + m] == nb + sBi2 + 0) ? rC[m][4] - 1 : rC[m][4];
                for (int i = 0; i < K; i++) {
                    atomicAdd(wgrad + i * N + (nb + sBi2 + 0), A[(mb + sAi1 + m) * K + i] * rC[m][4]);
                }
            }
            */
        }

#pragma unroll
        for (n = 0; n < 8; n++) {
            atomicAdd(e + mb + sAi1 + m, rC[m][n]);
        }
    }


    // I grad

    /*
#pragma unroll
    for (m = 0; m < 8; m++) {
        float4 t;
        if (mb + sAi1 + m < M) {
            int t = N - nb - sBi1;
            if (4 <= t) {
                for (int i = 0; i < K; i++) {
                    atomicAdd(bi + (mb + sAi1 + m) * K + i, rC[m][0] * B[i * N + nb + sBi1 + 0]);
                    atomicAdd(bi + (mb + sAi1 + m) * K + i, rC[m][1] * B[i * N + nb + sBi1 + 1]);
                    atomicAdd(bi + (mb + sAi1 + m) * K + i, rC[m][2] * B[i * N + nb + sBi1 + 2]);
                    atomicAdd(bi + (mb + sAi1 + m) * K + i, rC[m][3] * B[i * N + nb + sBi1 + 3]);
                }
            }
            else if (3 <= t) {
                for (int i = 0; i < K; i++) {
                    atomicAdd(bi + (mb + sAi1 + m) * K + i, rC[m][0] * B[i * N + nb + sBi1 + 0]);
                    atomicAdd(bi + (mb + sAi1 + m) * K + i, rC[m][1] * B[i * N + nb + sBi1 + 1]);
                    atomicAdd(bi + (mb + sAi1 + m) * K + i, rC[m][2] * B[i * N + nb + sBi1 + 2]);
                }
            }
            else if (2 <= t) {
                for (int i = 0; i < K; i++) {
                    atomicAdd(bi + (mb + sAi1 + m) * K + i, rC[m][0] * B[i * N + nb + sBi1 + 0]);
                    atomicAdd(bi + (mb + sAi1 + m) * K + i, rC[m][1] * B[i * N + nb + sBi1 + 1]);
                }
            }
            else if (1 <= t) {
                for (int i = 0; i < K; i++) {
                    atomicAdd(bi + (mb + sAi1 + m) * K + i, rC[m][0] * B[i * N + nb + sBi1 + 0]);
                }
            }
            t = N - nb - sBi2;
            if (4 <= t) {
                for (int i = 0; i < K; i++) {
                    atomicAdd(bi + (mb + sAi1 + m) * K + i, rC[m][4] * B[i * N + nb + sBi2 + 0]);
                    atomicAdd(bi + (mb + sAi1 + m) * K + i, rC[m][5] * B[i * N + nb + sBi2 + 1]);
                    atomicAdd(bi + (mb + sAi1 + m) * K + i, rC[m][6] * B[i * N + nb + sBi2 + 2]);
                    atomicAdd(bi + (mb + sAi1 + m) * K + i, rC[m][7] * B[i * N + nb + sBi2 + 3]);
                }
            }
            else if (3 <= t) {
                for (int i = 0; i < K; i++) {
                    atomicAdd(bi + (mb + sAi1 + m) * K + i, rC[m][4] * B[i * N + nb + sBi2 + 0]);
                    atomicAdd(bi + (mb + sAi1 + m) * K + i, rC[m][5] * B[i * N + nb + sBi2 + 1]);
                    atomicAdd(bi + (mb + sAi1 + m) * K + i, rC[m][6] * B[i * N + nb + sBi2 + 2]);
                }
            }
            else if (2 <= t) {
                for (int i = 0; i < K; i++) {
                    atomicAdd(bi + (mb + sAi1 + m) * K + i, rC[m][4] * B[i * N + nb + sBi2 + 0]);
                    atomicAdd(bi + (mb + sAi1 + m) * K + i, rC[m][5] * B[i * N + nb + sBi2 + 1]);
                }
            }
            else if (1 <= t) {
                for (int i = 0; i < K; i++) {
                    atomicAdd(bi + (mb + sAi1 + m) * K + i, rC[m][4] * B[i * N + nb + sBi2 + 0]);
                }
            }
        }

    }
    */
}




DLLEXPORT void cuda_onehotgemmsd_fu(float* din, float* dy, float* dw, float* dwgrad, int M, int K, int N, float* de, float* dbi) {
    cudaMemset(dwgrad, 0, sizeof(float) * K * N);
    cudaMemset(de, 0, sizeof(float) * M);
    cudaMemset(dbi, 0, sizeof(float) * M * K);
    dim3 gridDims(4, (int)ceil((float)N / 256), (int)ceil((float)M / 256));
    dim3 blockDims(256, 1, 1);
    ohsgemm128sd_fub << < gridDims, blockDims >> > (din, dw, dy, dwgrad, M, N, K, de, dbi);
    cudaMemcpy(din, dbi, sizeof(float) * M * K, cudaMemcpyDeviceToDevice);
}
