#include<time.h>
#include<iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)


__global__ void out(float* X, float* W, float* B, float* R, float* OUT, int indim, int outdim, int bs) {

    __shared__ float sW[64][9];
    __shared__ float sB[64][9];
    __shared__ float sR[64][9];
    __shared__ float sX[8][257];
    __shared__ float sWb[64][9];
    __shared__ float sBb[64][9];
    __shared__ float sRb[64][9];
    __shared__ float sXb[8][257];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int txm2 = tx % 2;
    int txb2 = tx / 2;
    int txm16 = tx % 16;
    int txb16 = tx / 16;

    float rW[4];
    float rB[4];
    float rR[4];
    float rX[16];
    float4 rWb;
    float4 rBb;
    float4 rRb;
    float4 rXb[2];
    float rO[16][4];

    float(*psW)[64][9] = &sW;
    float(*psB)[64][9] = &sB;
    float(*psR)[64][9] = &sR;
    float(*psX)[8][257] = &sX;

    bool aW = (outdim % 4 == 0) ? true : false;
    bool aX = (indim % 4 == 0) ? true : false;

    rWb.x = rWb.y = rWb.z = rWb.w = rBb.x = rBb.y = rBb.z = rBb.w = rRb.x = rRb.y = rRb.z = rRb.w = 0;
    rXb[0].x = rXb[0].y = rXb[0].z = rXb[0].w = rXb[1].x = rXb[1].y = rXb[1].z = rXb[1].w = 0;

#pragma unroll
    for (int m = 0; m < 16; m++) {
        rX[0] = 0;
#pragma unroll
        for (int n = 0; n < 4; n++) {
            rO[m][n] = 0;
            rW[n] = 0;
        }
    }

    if (txb16 < 8 && txb16 < indim) {
        int q = outdim - (by * 64 + txm16 * 4);
        if (4 <= q) {
            if (aW) {
                rWb = *reinterpret_cast<float4*>(W + txb16 * outdim + (by * 64 + txm16 * 4));
                rBb = *reinterpret_cast<float4*>(B + txb16 * outdim + (by * 64 + txm16 * 4));
                rRb = *reinterpret_cast<float4*>(R + txb16 * outdim + (by * 64 + txm16 * 4));
            }
            else {
                rWb.x = W[txb16 * outdim + (by * 64 + txm16 * 4 + 0)];
                rWb.y = W[txb16 * outdim + (by * 64 + txm16 * 4 + 1)];
                rWb.z = W[txb16 * outdim + (by * 64 + txm16 * 4 + 2)];
                rWb.w = W[txb16 * outdim + (by * 64 + txm16 * 4 + 3)];
                rBb.x = B[txb16 * outdim + (by * 64 + txm16 * 4 + 0)];
                rBb.y = B[txb16 * outdim + (by * 64 + txm16 * 4 + 1)];
                rBb.z = B[txb16 * outdim + (by * 64 + txm16 * 4 + 2)];
                rBb.w = B[txb16 * outdim + (by * 64 + txm16 * 4 + 3)];
                rRb.x = R[txb16 * outdim + (by * 64 + txm16 * 4 + 0)];
                rRb.y = R[txb16 * outdim + (by * 64 + txm16 * 4 + 1)];
                rRb.z = R[txb16 * outdim + (by * 64 + txm16 * 4 + 2)];
                rRb.w = R[txb16 * outdim + (by * 64 + txm16 * 4 + 3)];
            }
        }
        else if (3 <= q) {
            rWb.x = W[txb16 * outdim + (by * 64 + txm16 * 4 + 0)];
            rWb.y = W[txb16 * outdim + (by * 64 + txm16 * 4 + 1)];
            rWb.z = W[txb16 * outdim + (by * 64 + txm16 * 4 + 2)];
            rBb.x = B[txb16 * outdim + (by * 64 + txm16 * 4 + 0)];
            rBb.y = B[txb16 * outdim + (by * 64 + txm16 * 4 + 1)];
            rBb.z = B[txb16 * outdim + (by * 64 + txm16 * 4 + 2)];
            rRb.x = R[txb16 * outdim + (by * 64 + txm16 * 4 + 0)];
            rRb.y = R[txb16 * outdim + (by * 64 + txm16 * 4 + 1)];
            rRb.z = R[txb16 * outdim + (by * 64 + txm16 * 4 + 2)];
            rWb.w = rBb.w = rRb.w = 0;
        }
        else if (2 <= q) {
            rWb.x = W[txb16 * outdim + (by * 64 + txm16 * 4 + 0)];
            rWb.y = W[txb16 * outdim + (by * 64 + txm16 * 4 + 1)];
            rBb.x = B[txb16 * outdim + (by * 64 + txm16 * 4 + 0)];
            rBb.y = B[txb16 * outdim + (by * 64 + txm16 * 4 + 1)];
            rRb.x = R[txb16 * outdim + (by * 64 + txm16 * 4 + 0)];
            rRb.y = R[txb16 * outdim + (by * 64 + txm16 * 4 + 1)];
            rWb.z = rWb.w = rBb.z = rBb.w = rRb.z = rRb.w = 0;
        }
        else if (1 <= q) {
            rWb.x = W[txb16 * outdim + (by * 64 + txm16 * 4 + 0)];
            rBb.x = B[txb16 * outdim + (by * 64 + txm16 * 4 + 0)];
            rRb.x = R[txb16 * outdim + (by * 64 + txm16 * 4 + 0)];
            rWb.y = rWb.z = rWb.w = rBb.y = rBb.z = rBb.w = rRb.y = rRb.z = rRb.w = 0;
        }
        else {
            rWb.x = rWb.y = rWb.z = rWb.w = rBb.x = rBb.y = rBb.z = rBb.w = rRb.x = rRb.y = rRb.z = rRb.w = 0;
        }
    }
    if (bx * 256 + txb2 + 0 < bs) {
        int q = indim - txm2 * 4;
        if (4 <= q) {
            if (aX) {
                rXb[0] = *reinterpret_cast<float4*>(X + (bx * 256 + txb2 + 0) * indim + txm2 * 4);
            }
            else {
                rXb[0].x = X[(bx * 256 + txb2 + 0) * indim + txm2 * 4 + 0];
                rXb[0].y = X[(bx * 256 + txb2 + 0) * indim + txm2 * 4 + 1];
                rXb[0].z = X[(bx * 256 + txb2 + 0) * indim + txm2 * 4 + 2];
                rXb[0].w = X[(bx * 256 + txb2 + 0) * indim + txm2 * 4 + 3];
            }
        }
        else if (3 <= q) {
            rXb[0].x = X[(bx * 256 + txb2 + 0) * indim + txm2 * 4 + 0];
            rXb[0].y = X[(bx * 256 + txb2 + 0) * indim + txm2 * 4 + 1];
            rXb[0].z = X[(bx * 256 + txb2 + 0) * indim + txm2 * 4 + 2];
            rXb[0].w = 0;
        }
        else if (2 <= q) {
            rXb[0].x = X[(bx * 256 + txb2 + 0) * indim + txm2 * 4 + 0];
            rXb[0].y = X[(bx * 256 + txb2 + 0) * indim + txm2 * 4 + 1];
            rXb[0].z = 0;
            rXb[0].w = 0;
        }
        else if (1 <= q){
            rXb[0].x = X[(bx * 256 + txb2 + 0) * indim + txm2 * 4 + 0];
            rXb[0].y = 0;
            rXb[0].z = 0;
            rXb[0].w = 0;
        }
        else {
            rXb[0].x = 0;
            rXb[0].y = 0;
            rXb[0].z = 0;
            rXb[0].w = 0;
        }
    }
    if (bx * 256 + txb2 + 128 < bs) {
        int q = indim - txm2 * 4;
        if (4 <= q) {
            if (aX) {
                rXb[1] = *reinterpret_cast<float4*>(X + (bx * 256 + txb2 + 128) * indim + txm2 * 4);
            }
            else {
                rXb[1].x = X[(bx * 256 + txb2 + 128) * indim + txm2 * 4 + 0];
                rXb[1].y = X[(bx * 256 + txb2 + 128) * indim + txm2 * 4 + 1];
                rXb[1].z = X[(bx * 256 + txb2 + 128) * indim + txm2 * 4 + 2];
                rXb[1].w = X[(bx * 256 + txb2 + 128) * indim + txm2 * 4 + 3];
            }
        }
        else if (3 <= q) {
            rXb[1].x = X[(bx * 256 + txb2 + 128) * indim + txm2 * 4 + 0];
            rXb[1].y = X[(bx * 256 + txb2 + 128) * indim + txm2 * 4 + 1];
            rXb[1].z = X[(bx * 256 + txb2 + 128) * indim + txm2 * 4 + 2];
            rXb[1].w = 0;
        }
        else if (2 <= q) {
            rXb[1].x = X[(bx * 256 + txb2 + 128) * indim + txm2 * 4 + 0];
            rXb[1].y = X[(bx * 256 + txb2 + 128) * indim + txm2 * 4 + 1];
            rXb[1].z = 0;
            rXb[1].w = 0;
        }
        else if (1 <= q) {
            rXb[1].x = X[(bx * 256 + txb2 + 128) * indim + txm2 * 4 + 0];
            rXb[1].y = 0;
            rXb[1].z = 0;
            rXb[1].w = 0;
        }
        else {
            rXb[1].x = 0;
            rXb[1].y = 0;
            rXb[1].z = 0;
            rXb[1].w = 0;
        }
    }

    for (int k = 0; k < indim; k += 8) {

        if (txb16 < 8) {
            (*psW)[txm16 * 4 + 0][txb16] = rWb.x;
            (*psW)[txm16 * 4 + 1][txb16] = rWb.y;
            (*psW)[txm16 * 4 + 2][txb16] = rWb.z;
            (*psW)[txm16 * 4 + 3][txb16] = rWb.w;
            (*psB)[txm16 * 4 + 0][txb16] = rBb.x;
            (*psB)[txm16 * 4 + 1][txb16] = rBb.y;
            (*psB)[txm16 * 4 + 2][txb16] = rBb.z;
            (*psB)[txm16 * 4 + 3][txb16] = rBb.w;
            (*psR)[txm16 * 4 + 0][txb16] = rRb.x;
            (*psR)[txm16 * 4 + 1][txb16] = rRb.y;
            (*psR)[txm16 * 4 + 2][txb16] = rRb.z;
            (*psR)[txm16 * 4 + 3][txb16] = rRb.w;
        }
        (*psX)[txm2 * 4 + 0][txb2] = rXb[0].x;
        (*psX)[txm2 * 4 + 1][txb2] = rXb[0].y;
        (*psX)[txm2 * 4 + 2][txb2] = rXb[0].z;
        (*psX)[txm2 * 4 + 3][txb2] = rXb[0].w;
        (*psX)[txm2 * 4 + 0][txb2 + 128] = rXb[1].x;
        (*psX)[txm2 * 4 + 1][txb2 + 128] = rXb[1].y;
        (*psX)[txm2 * 4 + 2][txb2 + 128] = rXb[1].z;
        (*psX)[txm2 * 4 + 3][txb2 + 128] = rXb[1].w;

        __syncthreads();

        if (k + 8 < indim) {
            if (txb16 < 8 && k + 8 + txb16 < indim) {
                int q = outdim - (by * 64 + txm16 * 4);
                if (4 <= q) {
                    if (aW) {
                        rWb = *reinterpret_cast<float4*>(W + (k + 8 + txb16) * outdim + (by * 64 + txm16 * 4));
                        rBb = *reinterpret_cast<float4*>(B + (k + 8 + txb16) * outdim + (by * 64 + txm16 * 4));
                        rRb = *reinterpret_cast<float4*>(R + (k + 8 + txb16) * outdim + (by * 64 + txm16 * 4));
                    }
                    else {
                        rWb.x = W[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 0)];
                        rWb.y = W[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 1)];
                        rWb.z = W[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 2)];
                        rWb.w = W[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 3)];
                        rBb.x = B[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 0)];
                        rBb.y = B[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 1)];
                        rBb.z = B[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 2)];
                        rBb.w = B[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 3)];
                        rRb.x = R[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 0)];
                        rRb.y = R[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 1)];
                        rRb.z = R[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 2)];
                        rRb.w = R[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 3)];
                    }
                }
                else if (3 <= q) {
                    rWb.x = W[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 0)];
                    rWb.y = W[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 1)];
                    rWb.z = W[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 2)];
                    rBb.x = B[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 0)];
                    rBb.y = B[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 1)];
                    rBb.z = B[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 2)];
                    rRb.x = R[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 0)];
                    rRb.y = R[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 1)];
                    rRb.z = R[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 2)];
                    rWb.w = rBb.w = rRb.w = 0;
                }
                else if (2 <= q) {
                    rWb.x = W[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 0)];
                    rWb.y = W[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 1)];
                    rBb.x = B[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 0)];
                    rBb.y = B[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 1)];
                    rRb.x = R[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 0)];
                    rRb.y = R[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 1)];
                    rWb.z = rWb.w = rBb.z = rBb.w = rRb.z = rRb.w = 0;
                }
                else if (1 <= q) {
                    rWb.x = W[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 0)];
                    rBb.x = B[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 0)];
                    rRb.x = R[(k + 8 + txb16) * outdim + (by * 64 + txm16 * 4 + 0)];
                    rWb.y = rWb.z = rWb.w = rBb.y = rBb.z = rBb.w = rRb.y = rRb.z = rRb.w = 0;
                }
                else {
                    rWb.x = rWb.y = rWb.z = rWb.w = rBb.x = rBb.y = rBb.z = rBb.w = rRb.x = rRb.y = rRb.z = rRb.w = 0;
                }
            }
            else {
                rWb.x = rWb.y = rWb.z = rWb.w = rBb.x = rBb.y = rBb.z = rBb.w = rRb.x = rRb.y = rRb.z = rRb.w = 0;
            }
            if (bx * 256 + txb2 + 0 < bs) {
                int q = indim - (k + 8 + txm2 * 4);
                if (4 <= q) {
                    if (aX) {
                        rXb[0] = *reinterpret_cast<float4*>(X + (bx * 256 + txb2 + 0) * indim + (k + 8 + txm2 * 4));
                    }
                    else {
                        rXb[0].x = X[(bx * 256 + txb2 + 0) * indim + (k + 8 + txm2 * 4) + 0];
                        rXb[0].y = X[(bx * 256 + txb2 + 0) * indim + (k + 8 + txm2 * 4) + 1];
                        rXb[0].z = X[(bx * 256 + txb2 + 0) * indim + (k + 8 + txm2 * 4) + 2];
                        rXb[0].w = X[(bx * 256 + txb2 + 0) * indim + (k + 8 + txm2 * 4) + 3];
                    }
                }
                else if (3 <= q) {
                    rXb[0].x = X[(bx * 256 + txb2 + 0) * indim + (k + 8 + txm2 * 4) + 0];
                    rXb[0].y = X[(bx * 256 + txb2 + 0) * indim + (k + 8 + txm2 * 4) + 1];
                    rXb[0].z = X[(bx * 256 + txb2 + 0) * indim + (k + 8 + txm2 * 4) + 2];
                    rXb[0].w = 0;
                }
                else if (2 <= q) {
                    rXb[0].x = X[(bx * 256 + txb2 + 0) * indim + (k + 8 + txm2 * 4) + 0];
                    rXb[0].y = X[(bx * 256 + txb2 + 0) * indim + (k + 8 + txm2 * 4) + 1];
                    rXb[0].z = 0;
                    rXb[0].w = 0;
                }
                else if (1 <= q) {
                    rXb[0].x = X[(bx * 256 + txb2 + 0) * indim + (k + 8 + txm2 * 4) + 0];
                    rXb[0].y = 0;
                    rXb[0].z = 0;
                    rXb[0].w = 0;
                }
                else {
                    rXb[0].x = 0;
                    rXb[0].y = 0;
                    rXb[0].z = 0;
                    rXb[0].w = 0;
                }
            }
            else {
                rXb[0].x = 0;
                rXb[0].y = 0;
                rXb[0].z = 0;
                rXb[0].w = 0;
            }
            if (bx * 256 + txb2 + 128 < bs) {
                int q = indim - (k + 8 + txm2 * 4);
                if (4 <= q) {
                    if (aX) {
                        rXb[1] = *reinterpret_cast<float4*>(X + (bx * 256 + txb2 + 128) * indim + (k + 8 + txm2 * 4));
                    }
                    else {
                        rXb[1].x = X[(bx * 256 + txb2 + 128) * indim + (k + 8 + txm2 * 4) + 0];
                        rXb[1].y = X[(bx * 256 + txb2 + 128) * indim + (k + 8 + txm2 * 4) + 1];
                        rXb[1].z = X[(bx * 256 + txb2 + 128) * indim + (k + 8 + txm2 * 4) + 2];
                        rXb[1].w = X[(bx * 256 + txb2 + 128) * indim + (k + 8 + txm2 * 4) + 3];
                    }
                }
                else if (3 <= q) {
                    rXb[1].x = X[(bx * 256 + txb2 + 128) * indim + (k + 8 + txm2 * 4) + 0];
                    rXb[1].y = X[(bx * 256 + txb2 + 128) * indim + (k + 8 + txm2 * 4) + 1];
                    rXb[1].z = X[(bx * 256 + txb2 + 128) * indim + (k + 8 + txm2 * 4) + 2];
                    rXb[1].w = 0;
                }
                else if (2 <= q) {
                    rXb[1].x = X[(bx * 256 + txb2 + 128) * indim + (k + 8 + txm2 * 4) + 0];
                    rXb[1].y = X[(bx * 256 + txb2 + 128) * indim + (k + 8 + txm2 * 4) + 1];
                    rXb[1].z = 0;
                    rXb[1].w = 0;
                }
                else if (1 <= q) {
                    rXb[1].x = X[(bx * 256 + txb2 + 128) * indim + (k + 8 + txm2 * 4) + 0];
                    rXb[1].y = 0;
                    rXb[1].z = 0;
                    rXb[1].w = 0;
                }
                else {
                    rXb[1].x = 0;
                    rXb[1].y = 0;
                    rXb[1].z = 0;
                    rXb[1].w = 0;
                }
            }
            else {
                rXb[1].x = 0;
                rXb[1].y = 0;
                rXb[1].z = 0;
                rXb[1].w = 0;
            }
        }

#pragma unroll
        for (int kb = 0; kb < 8; kb++) {
#pragma unroll
            for (int m = 0; m < 4; m++) {
                rW[m] = (*psW)[txm16 * 4 + m][kb];
                rB[m] = (*psB)[txm16 * 4 + m][kb];
                rR[m] = (*psR)[txm16 * 4 + m][kb];
            }
#pragma unroll
            for (int m = 0; m < 16; m++) {
                rX[m] = (*psX)[kb][txb16 * 16 + m];
            }

#pragma unroll
            for (int m = 0; m < 16; m++) {
#pragma unroll
                for (int n = 0; n < 4; n++) {
                    if (rR[n] != 0) {
                        float v = (1 - fabsf(rX[m] - rB[n]) / fabsf(rR[n]));
                        if (v > 0) {
                            rO[m][n] += rW[n] * v;
                        }
                        //rO[m][n] += rW[n] * (1 - fabsf(rX[m] - rB[n]) / fabsf(rR[n]));
                    }
                }
            }
        }

        if (psW == &sW) {
            psW = &sWb;
            psB = &sBb;
            psR = &sRb;
            psX = &sXb;
        }
        else {
            psW = &sW;
            psB = &sB;
            psR = &sR;
            psX = &sX;
        }

    }

    float4 t;
    int q = bs - (bx * 256 + txb16 * 16);
#pragma unroll
    for (int m = 0; m < 16; m++) {        
        if (m < q) {
            int qq = outdim - (by * 64 + txm16 * 4);
            if (4 <= qq) {
                if (aW) {
                    t.x = rO[m][0];
                    t.y = rO[m][1];
                    t.z = rO[m][2];
                    t.w = rO[m][3];
                    *reinterpret_cast<float4*>(OUT + (bx * 256 + txb16 * 16 + m) * outdim + (by * 64 + txm16 * 4)) = t;
                }
                else {
                    OUT[(bs - q + m) * outdim + outdim - qq + 0] = rO[m][0];
                    OUT[(bs - q + m) * outdim + outdim - qq + 1] = rO[m][1];
                    OUT[(bs - q + m) * outdim + outdim - qq + 2] = rO[m][2];
                    OUT[(bs - q + m) * outdim + outdim - qq + 3] = rO[m][3];
                }
            }
            else if (3 <= qq) {
                OUT[(bs - q + m) * outdim + outdim - qq + 0] = rO[m][0];
                OUT[(bs - q + m) * outdim + outdim - qq + 1] = rO[m][1];
                OUT[(bs - q + m) * outdim + outdim - qq + 2] = rO[m][2];
            }
            else if (2 <= qq) {
                OUT[(bs - q + m) * outdim + outdim - qq + 0] = rO[m][0];
                OUT[(bs - q + m) * outdim + outdim - qq + 1] = rO[m][1];
            }
            else if (1 <= qq) {
                OUT[(bs - q + m) * outdim + outdim - qq + 0] = rO[m][0];
            }
        }
    }

}


__global__ void wrbgrad(float* X, float* W, float* B, float* R, float* O, float* WG, float* BG, float* RG, int indim, int outdim, int bs) {

    __shared__ float sX[64][33];
    __shared__ float sXb[64][33];
    __shared__ float sO[64][33];
    __shared__ float sOb[64][33];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int txm16 = tx % 16;
    int txb16 = tx / 16;

    float rW[4][4];
    float rB[4][4];
    float rR[4][4];

    float rWg[4][4];
    float rBg[4][4];
    float rRg[4][4];

    float(*psO)[64][33] = &sO;
    float(*psX)[64][33] = &sX;

    float4 rO[2];
    float4 rX[2];

    bool aO = (outdim % 4 == 0) ? true : false;
    bool aX = (indim % 4 == 0) ? true : false;

#pragma unroll
    for (int m = 0; m < 4; m++) {
#pragma unroll
        for (int n = 0; n < 4; n++) {
            if ((bx * 64 + txb16 * 4 + m) < indim && (by * 64 + txm16 * 4 + n) < outdim) {
                rW[m][n] = W[(bx * 64 + txb16 * 4 + m) * outdim + (by * 64 + txm16 * 4 + n)];
                rB[m][n] = B[(bx * 64 + txb16 * 4 + m) * outdim + (by * 64 + txm16 * 4 + n)];
                rR[m][n] = R[(bx * 64 + txb16 * 4 + m) * outdim + (by * 64 + txm16 * 4 + n)];
            }
            else {
                rW[m][n] = 0;
                rB[m][n] = 0;
                rR[m][n] = 0;
            }
            rWg[m][n] = 0;
            rBg[m][n] = 0;
            rRg[m][n] = 0;
        }
    }

    int q = outdim - (by * 64 + txm16 * 4);
    if (4 <= q) {
        if (aO) {
            if (txb16 < bs) {
                rO[0] = *reinterpret_cast<float4*>(O + (txb16 + 0) * outdim + outdim - q);
            }
            else {
                rO[0].x = rO[0].y = rO[0].z = rO[0].w = 0;
            }
            if (txb16 + 16 < bs) {
                rO[1] = *reinterpret_cast<float4*>(O + (txb16 + 16) * outdim + outdim - q);
            }
            else {
                rO[1].x = rO[1].y = rO[1].z = rO[1].w = 0;
            }
        }
        else {
            if (txb16 < bs) {
                rO[0].x = O[(txb16 + 0) * outdim + outdim - q + 0];
                rO[0].y = O[(txb16 + 0) * outdim + outdim - q + 1];
                rO[0].z = O[(txb16 + 0) * outdim + outdim - q + 2];
                rO[0].w = O[(txb16 + 0) * outdim + outdim - q + 3];
            }
            else {
                rO[0].x = rO[0].y = rO[0].z = rO[0].w = 0;
            }
            if (txb16 + 16 < bs) {
                rO[1].x = O[(txb16 + 16) * outdim + outdim - q + 0];
                rO[1].y = O[(txb16 + 16) * outdim + outdim - q + 1];
                rO[1].z = O[(txb16 + 16) * outdim + outdim - q + 2];
                rO[1].w = O[(txb16 + 16) * outdim + outdim - q + 3];
            }
            else {
                rO[1].x = rO[1].y = rO[1].z = rO[1].w = 0;
            }
        }
    }
    else if (3 <= q) {
        if (txb16 < bs) {
            rO[0].x = O[(txb16 + 0) * outdim + outdim - q + 0];
            rO[0].y = O[(txb16 + 0) * outdim + outdim - q + 1];
            rO[0].z = O[(txb16 + 0) * outdim + outdim - q + 2];
            rO[0].w = 0;
        }
        else {
            rO[0].x = rO[0].y = rO[0].z = rO[0].w = 0;
        }
        if (txb16 + 16 < bs) {
            rO[1].x = O[(txb16 + 16) * outdim + outdim - q + 0];
            rO[1].y = O[(txb16 + 16) * outdim + outdim - q + 1];
            rO[1].z = O[(txb16 + 16) * outdim + outdim - q + 2];
            rO[1].w = 0;
        }
        else {
            rO[1].x = rO[1].y = rO[1].z = rO[1].w = 0;
        }
    }
    else if (2 <= q) {
        if (txb16 < bs) {
            rO[0].x = O[(txb16 + 0) * outdim + outdim - q + 0];
            rO[0].y = O[(txb16 + 0) * outdim + outdim - q + 1];
            rO[0].z = 0;
            rO[0].w = 0;
        }
        else {
            rO[0].x = rO[0].y = rO[0].z = rO[0].w = 0;
        }
        if (txb16 + 16 < bs) {
            rO[1].x = O[(txb16 + 16) * outdim + outdim - q + 0];
            rO[1].y = O[(txb16 + 16) * outdim + outdim - q + 1];
            rO[1].z = 0;
            rO[1].w = 0;
        }
        else {
            rO[1].x = rO[1].y = rO[1].z = rO[1].w = 0;
        }
    }
    else if (1 <= q) {
        if (txb16 < bs) {
            rO[0].x = O[(txb16 + 0) * outdim + outdim - q + 0];
            rO[0].y = 0;
            rO[0].z = 0;
            rO[0].w = 0;
        }
        else {
            rO[0].x = rO[0].y = rO[0].z = rO[0].w = 0;
        }
        if (txb16 + 16 < bs) {
            rO[1].x = O[(txb16 + 16) * outdim + outdim - q + 0];
            rO[1].y = 0;
            rO[1].z = 0;
            rO[1].w = 0;
        }
        else {
            rO[1].x = rO[1].y = rO[1].z = rO[1].w = 0;
        }
    }
    else {
        rO[0].x = rO[0].y = rO[0].z = rO[0].w = 0;
        rO[1].x = rO[1].y = rO[1].z = rO[1].w = 0;
    }

    q = indim - (bx * 64 + txm16 * 4);
    if (4 <= q) {
        if (aX) {
            if (txb16 < bs) {
                rX[0] = *reinterpret_cast<float4*>(X + (txb16 + 0) * indim + indim - q);
            }
            else {
                rX[0].x = rX[0].y = rX[0].z = rX[0].w = 0;
            }
            if (txb16 + 16 < bs) {
                rX[1] = *reinterpret_cast<float4*>(X + (txb16 + 16) * indim + indim - q);
            }
            else {
                rX[1].x = rX[1].y = rX[1].z = rX[1].w = 0;
            }
        }
        else {
            if (txb16 < bs) {
                rX[0].x = X[(txb16 + 0) * indim + indim - q + 0];
                rX[0].y = X[(txb16 + 0) * indim + indim - q + 1];
                rX[0].z = X[(txb16 + 0) * indim + indim - q + 2];
                rX[0].w = X[(txb16 + 0) * indim + indim - q + 3];
            }
            else {
                rX[0].x = rX[0].y = rX[0].z = rX[0].w = 0;
            }
            if (txb16 + 16 < bs) {
                rX[1].x = X[(txb16 + 16) * indim + indim - q + 0];
                rX[1].y = X[(txb16 + 16) * indim + indim - q + 1];
                rX[1].z = X[(txb16 + 16) * indim + indim - q + 2];
                rX[1].w = X[(txb16 + 16) * indim + indim - q + 3];
            }
            else {
                rX[1].x = rX[1].y = rX[1].z = rX[1].w = 0;
            }
        }
    }
    else if (3 <= q) {
        if (txb16 < bs) {
            rX[0].x = X[(txb16 + 0) * indim + indim - q + 0];
            rX[0].y = X[(txb16 + 0) * indim + indim - q + 1];
            rX[0].z = X[(txb16 + 0) * indim + indim - q + 2];
            rX[0].w = 0;
        }
        else {
            rX[0].x = rX[0].y = rX[0].z = rX[0].w = 0;
        }
        if (txb16 + 16 < bs) {
            rX[1].x = X[(txb16 + 16) * indim + indim - q + 0];
            rX[1].y = X[(txb16 + 16) * indim + indim - q + 1];
            rX[1].z = X[(txb16 + 16) * indim + indim - q + 2];
            rX[1].w = 0;
        }
        else {
            rX[1].x = rX[1].y = rX[1].z = rX[1].w = 0;
        }
    }
    else if (2 <= q) {
        if (txb16 < bs) {
            rX[0].x = X[(txb16 + 0) * indim + indim - q + 0];
            rX[0].y = X[(txb16 + 0) * indim + indim - q + 1];
            rX[0].z = 0;
            rX[0].w = 0;
        }
        else {
            rX[0].x = rX[0].y = rX[0].z = rX[0].w = 0;
        }
        if (txb16 + 16 < bs) {
            rX[1].x = X[(txb16 + 16) * indim + indim - q + 0];
            rX[1].y = X[(txb16 + 16) * indim + indim - q + 1];
            rX[1].z = 0;
            rX[1].w = 0;
        }
        else {
            rX[1].x = rX[1].y = rX[1].z = rX[1].w = 0;
        }
    }
    else if (1 <= q) {
        if (txb16 < bs) {
            rX[0].x = X[(txb16 + 0) * indim + indim - q + 0];
            rX[0].y = 0;
            rX[0].z = 0;
            rX[0].w = 0;
        }
        else {
            rX[0].x = rX[0].y = rX[0].z = rX[0].w = 0;
        }
        if (txb16 + 16 < bs) {
            rX[1].x = X[(txb16 + 16) * indim + indim - q + 0];
            rX[1].y = 0;
            rX[1].z = 0;
            rX[1].w = 0;
        }
        else {
            rX[1].x = rX[1].y = rX[1].z = rX[1].w = 0;
        }
    }
    else {
        rX[0].x = rX[0].y = rX[0].z = rX[0].w = 0;
        rX[1].x = rX[1].y = rX[1].z = rX[1].w = 0;
    }

    for (int k = 0; k < bs; k += 32) {

        (*psO)[txm16 * 4 + 0][txb16 + 0] = rO[0].x;
        (*psO)[txm16 * 4 + 1][txb16 + 0] = rO[0].y;
        (*psO)[txm16 * 4 + 2][txb16 + 0] = rO[0].z;
        (*psO)[txm16 * 4 + 3][txb16 + 0] = rO[0].w;
        (*psO)[txm16 * 4 + 0][txb16 + 16] = rO[1].x;
        (*psO)[txm16 * 4 + 1][txb16 + 16] = rO[1].y;
        (*psO)[txm16 * 4 + 2][txb16 + 16] = rO[1].z;
        (*psO)[txm16 * 4 + 3][txb16 + 16] = rO[1].w;

        (*psX)[txm16 * 4 + 0][txb16 + 0] = rX[0].x;
        (*psX)[txm16 * 4 + 1][txb16 + 0] = rX[0].y;
        (*psX)[txm16 * 4 + 2][txb16 + 0] = rX[0].z;
        (*psX)[txm16 * 4 + 3][txb16 + 0] = rX[0].w;
        (*psX)[txm16 * 4 + 0][txb16 + 16] = rX[1].x;
        (*psX)[txm16 * 4 + 1][txb16 + 16] = rX[1].y;
        (*psX)[txm16 * 4 + 2][txb16 + 16] = rX[1].z;
        (*psX)[txm16 * 4 + 3][txb16 + 16] = rX[1].w;

        __syncthreads();

        if (k + 32 < bs) {

            q = outdim - (by * 64 + txm16 * 4);
            if (4 <= q) {
                if (aO) {
                    if (k + 32 + txb16 < bs) {
                        rO[0] = *reinterpret_cast<float4*>(O + (k + 32 + txb16 + 0) * outdim + outdim - q);
                    }
                    else {
                        rO[0].x = rO[0].y = rO[0].z = rO[0].w = 0;
                    }
                    if (k + 32 + txb16 + 16 < bs) {
                        rO[1] = *reinterpret_cast<float4*>(O + (k + 32 + txb16 + 16) * outdim + outdim - q);
                    }
                    else {
                        rO[1].x = rO[1].y = rO[1].z = rO[1].w = 0;
                    }
                }
                else {
                    if (k + 32 + txb16 < bs) {
                        rO[0].x = O[(k + 32 + txb16 + 0) * outdim + outdim - q + 0];
                        rO[0].y = O[(k + 32 + txb16 + 0) * outdim + outdim - q + 1];
                        rO[0].z = O[(k + 32 + txb16 + 0) * outdim + outdim - q + 2];
                        rO[0].w = O[(k + 32 + txb16 + 0) * outdim + outdim - q + 3];
                    }
                    else {
                        rO[0].x = rO[0].y = rO[0].z = rO[0].w = 0;
                    }
                    if (k + 32 + txb16 + 16 < bs) {
                        rO[1].x = O[(k + 32 + txb16 + 16) * outdim + outdim - q + 0];
                        rO[1].y = O[(k + 32 + txb16 + 16) * outdim + outdim - q + 1];
                        rO[1].z = O[(k + 32 + txb16 + 16) * outdim + outdim - q + 2];
                        rO[1].w = O[(k + 32 + txb16 + 16) * outdim + outdim - q + 3];
                    }
                    else {
                        rO[1].x = rO[1].y = rO[1].z = rO[1].w = 0;
                    }
                }
            }
            else if (3 <= q) {
                if (k + 32 + txb16 < bs) {
                    rO[0].x = O[(k + 32 + txb16 + 0) * outdim + outdim - q + 0];
                    rO[0].y = O[(k + 32 + txb16 + 0) * outdim + outdim - q + 1];
                    rO[0].z = O[(k + 32 + txb16 + 0) * outdim + outdim - q + 2];
                    rO[0].w = 0;
                }
                else {
                    rO[0].x = rO[0].y = rO[0].z = rO[0].w = 0;
                }
                if (k + 32 + txb16 + 16 < bs) {
                    rO[1].x = O[(k + 32 + txb16 + 16) * outdim + outdim - q + 0];
                    rO[1].y = O[(k + 32 + txb16 + 16) * outdim + outdim - q + 1];
                    rO[1].z = O[(k + 32 + txb16 + 16) * outdim + outdim - q + 2];
                    rO[1].w = 0;
                }
                else {
                    rO[1].x = rO[1].y = rO[1].z = rO[1].w = 0;
                }
            }
            else if (2 <= q) {
                if (k + 32 + txb16 < bs) {
                    rO[0].x = O[(k + 32 + txb16 + 0) * outdim + outdim - q + 0];
                    rO[0].y = O[(k + 32 + txb16 + 0) * outdim + outdim - q + 1];
                    rO[0].z = 0;
                    rO[0].w = 0;
                }
                else {
                    rO[0].x = rO[0].y = rO[0].z = rO[0].w = 0;
                }
                if (k + 32 + txb16 + 16 < bs) {
                    rO[1].x = O[(k + 32 + txb16 + 16) * outdim + outdim - q + 0];
                    rO[1].y = O[(k + 32 + txb16 + 16) * outdim + outdim - q + 1];
                    rO[1].z = 0;
                    rO[1].w = 0;
                }
                else {
                    rO[1].x = rO[1].y = rO[1].z = rO[1].w = 0;
                }
            }
            else if (1 <= q) {
                if (k + 32 + txb16 < bs) {
                    rO[0].x = O[(k + 32 + txb16 + 0) * outdim + outdim - q + 0];
                    rO[0].y = 0;
                    rO[0].z = 0;
                    rO[0].w = 0;
                }
                else {
                    rO[0].x = rO[0].y = rO[0].z = rO[0].w = 0;
                }
                if (k + 32 + txb16 + 16 < bs) {
                    rO[1].x = O[(k + 32 + txb16 + 16) * outdim + outdim - q + 0];
                    rO[1].y = 0;
                    rO[1].z = 0;
                    rO[1].w = 0;
                }
                else {
                    rO[1].x = rO[1].y = rO[1].z = rO[1].w = 0;
                }
            }
            else {
                rO[0].x = rO[0].y = rO[0].z = rO[0].w = 0;
                rO[1].x = rO[1].y = rO[1].z = rO[1].w = 0;
            }

            q = indim - (bx * 64 + txm16 * 4);
            if (4 <= q) {
                if (aX) {
                    if (k + 32 + txb16 < bs) {
                        rX[0] = *reinterpret_cast<float4*>(X + (k + 32 + txb16 + 0) * indim + indim - q);
                    }
                    else {
                        rX[0].x = rX[0].y = rX[0].z = rX[0].w = 0;
                    }
                    if (k + 32 + txb16 + 16 < bs) {
                        rX[1] = *reinterpret_cast<float4*>(X + (k + 32 + txb16 + 16) * indim + indim - q);
                    }
                    else {
                        rX[1].x = rX[1].y = rX[1].z = rX[1].w = 0;
                    }
                }
                else {
                    if (k + 32 + txb16 < bs) {
                        rX[0].x = X[(k + 32 + txb16 + 0) * indim + indim - q + 0];
                        rX[0].y = X[(k + 32 + txb16 + 0) * indim + indim - q + 1];
                        rX[0].z = X[(k + 32 + txb16 + 0) * indim + indim - q + 2];
                        rX[0].w = X[(k + 32 + txb16 + 0) * indim + indim - q + 3];
                    }
                    else {
                        rX[0].x = rX[0].y = rX[0].z = rX[0].w = 0;
                    }
                    if (k + 32 + txb16 + 16 < bs) {
                        rX[1].x = X[(k + 32 + txb16 + 16) * indim + indim - q + 0];
                        rX[1].y = X[(k + 32 + txb16 + 16) * indim + indim - q + 1];
                        rX[1].z = X[(k + 32 + txb16 + 16) * indim + indim - q + 2];
                        rX[1].w = X[(k + 32 + txb16 + 16) * indim + indim - q + 3];
                    }
                    else {
                        rX[1].x = rX[1].y = rX[1].z = rX[1].w = 0;
                    }
                }
            }
            else if (3 <= q) {
                if (k + 32 + txb16 < bs) {
                    rX[0].x = X[(k + 32 + txb16 + 0) * indim + indim - q + 0];
                    rX[0].y = X[(k + 32 + txb16 + 0) * indim + indim - q + 1];
                    rX[0].z = X[(k + 32 + txb16 + 0) * indim + indim - q + 2];
                    rX[0].w = 0;
                }
                else {
                    rX[0].x = rX[0].y = rX[0].z = rX[0].w = 0;
                }
                if (k + 32 + txb16 + 16 < bs) {
                    rX[1].x = X[(k + 32 + txb16 + 16) * indim + indim - q + 0];
                    rX[1].y = X[(k + 32 + txb16 + 16) * indim + indim - q + 1];
                    rX[1].z = X[(k + 32 + txb16 + 16) * indim + indim - q + 2];
                    rX[1].w = 0;
                }
                else {
                    rX[1].x = rX[1].y = rX[1].z = rX[1].w = 0;
                }
            }
            else if (2 <= q) {
                if (k + 32 + txb16 < bs) {
                    rX[0].x = X[(k + 32 + txb16 + 0) * indim + indim - q + 0];
                    rX[0].y = X[(k + 32 + txb16 + 0) * indim + indim - q + 1];
                    rX[0].z = 0;
                    rX[0].w = 0;
                }
                else {
                    rX[0].x = rX[0].y = rX[0].z = rX[0].w = 0;
                }
                if (k + 32 + txb16 + 16 < bs) {
                    rX[1].x = X[(k + 32 + txb16 + 16) * indim + indim - q + 0];
                    rX[1].y = X[(k + 32 + txb16 + 16) * indim + indim - q + 1];
                    rX[1].z = 0;
                    rX[1].w = 0;
                }
                else {
                    rX[1].x = rX[1].y = rX[1].z = rX[1].w = 0;
                }
            }
            else if (1 <= q) {
                if (k + 32 + txb16 < bs) {
                    rX[0].x = X[(k + 32 + txb16 + 0) * indim + indim - q + 0];
                    rX[0].y = 0;
                    rX[0].z = 0;
                    rX[0].w = 0;
                }
                else {
                    rX[0].x = rX[0].y = rX[0].z = rX[0].w = 0;
                }
                if (k + 32 + txb16 + 16 < bs) {
                    rX[1].x = X[(k + 32 + txb16 + 16) * indim + indim - q + 0];
                    rX[1].y = 0;
                    rX[1].z = 0;
                    rX[1].w = 0;
                }
                else {
                    rX[1].x = rX[1].y = rX[1].z = rX[1].w = 0;
                }
            }
            else {
                rX[0].x = rX[0].y = rX[0].z = rX[0].w = 0;
                rX[1].x = rX[1].y = rX[1].z = rX[1].w = 0;
            }
        }

#pragma unroll
        for (int kk = 0; kk < 32; kk++) {
#pragma unroll
            for (int m = 0; m < 4; m++) {
#pragma unroll
                for (int n = 0; n < 4; n++) {
                    float o = (*psO)[txm16 * 4 + n][kk];
                    float xmb = (*psX)[txb16 * 4 + m][kk] - rB[m][n];
                    if (xmb != 0 && rR[m][n] != 0) {
                        float v = (1 - fabs(xmb) / fabs(rR[m][n]));
                        float sgn = xmb / fabs(xmb);
                        if (v >= 0) {
                            rWg[m][n] += o * v;
                            rBg[m][n] += o * rW[m][n] * sgn / fabs(rR[m][n]);
                            rRg[m][n] += o * rW[m][n] * fabs(xmb) / (rR[m][n] * rR[m][n]);
                        }
                        else {
                            rBg[m][n] += o * -sgn;
                            rRg[m][n] += o * -sgn;
                        }
                    }
                }
            }
        }

        if (psO == &sO) {
            psO = &sOb;
            psX = &sXb;
        }
        else {
            psO = &sO;
            psX = &sX;
        }

    }


#pragma unroll
    for (int m = 0; m < 4; m++) {
#pragma unroll
        for (int n = 0; n < 4; n++) {
            int ix = bx * 64 + txb16 * 4 + m;
            int ox = by * 64 + txm16 * 4 + n;
            if (ix < indim && ox < outdim) {
                WG[ix * outdim + ox] = rWg[m][n];
                BG[ix * outdim + ox] = rBg[m][n];
                RG[ix * outdim + ox] = rRg[m][n];
            }
        }
    }


}


__global__ void xgrad(float* X, float* W, float* B, float* R, float* O, float* XG, int indim, int outdim, int bs) {

    __shared__ float sW[16][65];
    __shared__ float sB[16][65];
    __shared__ float sR[16][65];
    __shared__ float sO[16][129];
    __shared__ float sWb[16][65];
    __shared__ float sBb[16][65];
    __shared__ float sRb[16][65];
    __shared__ float sOb[16][129];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;

    int txm4 = tx % 4;
    int txb4 = tx / 4;
    int txm16 = tx % 16;
    int txb16 = tx / 16;

    float(*psW)[16][65] = &sW;
    float(*psB)[16][65] = &sB;
    float(*psR)[16][65] = &sR;
    float(*psO)[16][129] = &sO;

    float4 rW;
    float4 rB;
    float4 rR;
    float4 rO[2];

    bool aW = (outdim % 4 == 0) ? true : false;

    float rX[8][4];
    float rXg[8][4];

#pragma unroll
    for (int m = 0; m < 8; m++) {
#pragma unroll
        for (int n = 0; n < 4; n++) {
            if (bx * 128 + txb16 * 8 + m < bs && by * 64 + txm16 * 4 + n < indim) {
                rX[m][n] = X[(bx * 128 + txb16 * 8 + m) * indim + by * 64 + txm16 * 4 + n];
            }
            else {
                rX[m][n] = 0;
            }
            rXg[m][n] = 0;
        }
    }

    int q = outdim - txm4 * 4;
    if (4 <= q) {
        if (by * 64 + txb4 < indim) {
            if (aW) {
                rW = *reinterpret_cast<float4*>(W + (by * 64 + txb4) * outdim + txm4 * 4);
                rB = *reinterpret_cast<float4*>(B + (by * 64 + txb4) * outdim + txm4 * 4);
                rR = *reinterpret_cast<float4*>(R + (by * 64 + txb4) * outdim + txm4 * 4);
            }
            else {
                rW.x = W[(by * 64 + txb4) * outdim + txm4 * 4 + 0];
                rW.y = W[(by * 64 + txb4) * outdim + txm4 * 4 + 1];
                rW.z = W[(by * 64 + txb4) * outdim + txm4 * 4 + 2];
                rW.w = W[(by * 64 + txb4) * outdim + txm4 * 4 + 3];
                rB.x = B[(by * 64 + txb4) * outdim + txm4 * 4 + 0];
                rB.y = B[(by * 64 + txb4) * outdim + txm4 * 4 + 1];
                rB.z = B[(by * 64 + txb4) * outdim + txm4 * 4 + 2];
                rB.w = B[(by * 64 + txb4) * outdim + txm4 * 4 + 3];
                rR.x = R[(by * 64 + txb4) * outdim + txm4 * 4 + 0];
                rR.y = R[(by * 64 + txb4) * outdim + txm4 * 4 + 1];
                rR.z = R[(by * 64 + txb4) * outdim + txm4 * 4 + 2];
                rR.w = R[(by * 64 + txb4) * outdim + txm4 * 4 + 3];
            }
        }
        else {
            rW.x = rW.y = rW.z = rW.w = rB.x = rB.y = rB.z = rB.w = rR.x = rR.y = rR.z = rR.w = 0;
        }
        if (bx * 128 + txb4 < bs) {
            if (aW) {
                rO[0] = *reinterpret_cast<float4*>(O + (bx * 128 + txb4 + 0) * outdim + txm4 * 4);
            }
            else {
                rO[0].x = O[(bx * 128 + txb4 + 0) * outdim + txm4 * 4 + 0];
                rO[0].y = O[(bx * 128 + txb4 + 0) * outdim + txm4 * 4 + 1];
                rO[0].z = O[(bx * 128 + txb4 + 0) * outdim + txm4 * 4 + 2];
                rO[0].w = O[(bx * 128 + txb4 + 0) * outdim + txm4 * 4 + 3];
            }
        }
        else {
            rO[0].x = rO[0].y = rO[0].z = rO[0].w = 0;
        }
        if (bx * 128 + txb4 + 64 < bs) {
            if (aW) {
                rO[1] = *reinterpret_cast<float4*>(O + (bx * 128 + txb4 + 64) * outdim + txm4 * 4);
            }
            else {
                rO[1].x = O[(bx * 128 + txb4 + 64) * outdim + txm4 * 4 + 0];
                rO[1].y = O[(bx * 128 + txb4 + 64) * outdim + txm4 * 4 + 1];
                rO[1].z = O[(bx * 128 + txb4 + 64) * outdim + txm4 * 4 + 2];
                rO[1].w = O[(bx * 128 + txb4 + 64) * outdim + txm4 * 4 + 3];
            }
        }
        else {
            rO[1].x = rO[1].y = rO[1].z = rO[1].w = 0;
        }
    }
    else if (3 <= q) {
        if (by * 64 + txb4 < indim) {
            rW.x = W[(by * 64 + txb4) * outdim + txm4 * 4 + 0];
            rW.y = W[(by * 64 + txb4) * outdim + txm4 * 4 + 1];
            rW.z = W[(by * 64 + txb4) * outdim + txm4 * 4 + 2];
            rB.x = B[(by * 64 + txb4) * outdim + txm4 * 4 + 0];
            rB.y = B[(by * 64 + txb4) * outdim + txm4 * 4 + 1];
            rB.z = B[(by * 64 + txb4) * outdim + txm4 * 4 + 2];
            rR.x = R[(by * 64 + txb4) * outdim + txm4 * 4 + 0];
            rR.y = R[(by * 64 + txb4) * outdim + txm4 * 4 + 1];
            rR.z = R[(by * 64 + txb4) * outdim + txm4 * 4 + 2];
            rW.w = rB.w = rR.w = 0;
        }
        else {
            rW.x = rW.y = rW.z = rW.w = rB.x = rB.y = rB.z = rB.w = rR.x = rR.y = rR.z = rR.w = 0;
        }
        if (bx * 128 + txb4 < bs) {
            rO[0].x = O[(bx * 128 + txb4 + 0) * outdim + txm4 * 4 + 0];
            rO[0].y = O[(bx * 128 + txb4 + 0) * outdim + txm4 * 4 + 1];
            rO[0].z = O[(bx * 128 + txb4 + 0) * outdim + txm4 * 4 + 2];
            rO[0].w = 0;
        }
        else {
            rO[0].x = rO[0].y = rO[0].z = rO[0].w = 0;
        }
        if (bx * 128 + txb4 + 64 < bs) {
            rO[1].x = O[(bx * 128 + txb4 + 64) * outdim + txm4 * 4 + 0];
            rO[1].y = O[(bx * 128 + txb4 + 64) * outdim + txm4 * 4 + 1];
            rO[1].z = O[(bx * 128 + txb4 + 64) * outdim + txm4 * 4 + 2];
            rO[1].w = 0;
        }
        else {
            rO[1].x = rO[1].y = rO[1].z = rO[1].w = 0;
        }
    }
    else if (2 <= q) {
        if (by * 64 + txb4 < indim) {
            rW.x = W[(by * 64 + txb4) * outdim + txm4 * 4 + 0];
            rW.y = W[(by * 64 + txb4) * outdim + txm4 * 4 + 1];
            rB.x = B[(by * 64 + txb4) * outdim + txm4 * 4 + 0];
            rB.y = B[(by * 64 + txb4) * outdim + txm4 * 4 + 1];
            rR.x = R[(by * 64 + txb4) * outdim + txm4 * 4 + 0];
            rR.y = R[(by * 64 + txb4) * outdim + txm4 * 4 + 1];
            rW.z = rB.z = rR.z = 0;
            rW.w = rB.w = rR.w = 0;
        }
        else {
            rW.x = rW.y = rW.z = rW.w = rB.x = rB.y = rB.z = rB.w = rR.x = rR.y = rR.z = rR.w = 0;
        }
        if (bx * 128 + txb4 < bs) {
            rO[0].x = O[(bx * 128 + txb4 + 0) * outdim + txm4 * 4 + 0];
            rO[0].y = O[(bx * 128 + txb4 + 0) * outdim + txm4 * 4 + 1];
            rO[0].z = 0;
            rO[0].w = 0;
        }
        else {
            rO[0].x = rO[0].y = rO[0].z = rO[0].w = 0;
        }
        if (bx * 128 + txb4 + 64 < bs) {
            rO[1].x = O[(bx * 128 + txb4 + 64) * outdim + txm4 * 4 + 0];
            rO[1].y = O[(bx * 128 + txb4 + 64) * outdim + txm4 * 4 + 1];
            rO[1].z = 0;
            rO[1].w = 0;
        }
        else {
            rO[1].x = rO[1].y = rO[1].z = rO[1].w = 0;
        }
    }
    else if (1 <= q) {
        if (by * 64 + txb4 < indim) {
            rW.x = W[(by * 64 + txb4) * outdim + txm4 * 4 + 0];
            rB.x = B[(by * 64 + txb4) * outdim + txm4 * 4 + 0];
            rR.x = R[(by * 64 + txb4) * outdim + txm4 * 4 + 0];
            rW.y = rB.y = rR.y = 0;
            rW.z = rB.z = rR.z = 0;
            rW.w = rB.w = rR.w = 0;
        }
        else {
            rW.x = rW.y = rW.z = rW.w = rB.x = rB.y = rB.z = rB.w = rR.x = rR.y = rR.z = rR.w = 0;
        }
        if (bx * 128 + txb4 < bs) {
            rO[0].x = O[(bx * 128 + txb4 + 0) * outdim + txm4 * 4 + 0];
            rO[0].y = 0;
            rO[0].z = 0;
            rO[0].w = 0;
        }
        else {
            rO[0].x = rO[0].y = rO[0].z = rO[0].w = 0;
        }
        if (bx * 128 + txb4 + 64 < bs) {
            rO[1].x = O[(bx * 128 + txb4 + 64) * outdim + txm4 * 4 + 0];
            rO[1].y = 0;
            rO[1].z = 0;
            rO[1].w = 0;
        }
        else {
            rO[1].x = rO[1].y = rO[1].z = rO[1].w = 0;
        }
    }
    else {
        rW.x = rW.y = rW.z = rW.w = rB.x = rB.y = rB.z = rB.w = rR.x = rR.y = rR.z = rR.w = 0;
        rO[0].x = rO[0].y = rO[0].z = rO[0].w = 0;
        rO[1].x = rO[1].y = rO[1].z = rO[1].w = 0;
    }


    for (int k = 0; k < outdim; k += 16) {

        (*psW)[txm4 * 4 + 0][txb4] = rW.x;
        (*psW)[txm4 * 4 + 1][txb4] = rW.y;
        (*psW)[txm4 * 4 + 2][txb4] = rW.z;
        (*psW)[txm4 * 4 + 3][txb4] = rW.w;
        (*psB)[txm4 * 4 + 0][txb4] = rB.x;
        (*psB)[txm4 * 4 + 1][txb4] = rB.y;
        (*psB)[txm4 * 4 + 2][txb4] = rB.z;
        (*psB)[txm4 * 4 + 3][txb4] = rB.w;
        (*psR)[txm4 * 4 + 0][txb4] = rR.x;
        (*psR)[txm4 * 4 + 1][txb4] = rR.y;
        (*psR)[txm4 * 4 + 2][txb4] = rR.z;
        (*psR)[txm4 * 4 + 3][txb4] = rR.w;
        (*psO)[txm4 * 4 + 0][txb4 + 0] = rO[0].x;
        (*psO)[txm4 * 4 + 1][txb4 + 0] = rO[0].y;
        (*psO)[txm4 * 4 + 2][txb4 + 0] = rO[0].z;
        (*psO)[txm4 * 4 + 3][txb4 + 0] = rO[0].w;
        (*psO)[txm4 * 4 + 0][txb4 + 64] = rO[1].x;
        (*psO)[txm4 * 4 + 1][txb4 + 64] = rO[1].y;
        (*psO)[txm4 * 4 + 2][txb4 + 64] = rO[1].z;
        (*psO)[txm4 * 4 + 3][txb4 + 64] = rO[1].w;

        __syncthreads();

        if (k + 16 < outdim) {
            int q = outdim - (k + 16 + txm4 * 4);
            if (4 <= q) {
                if (by * 64 + txb4 < indim) {
                    if (aW) {
                        rW = *reinterpret_cast<float4*>(W + (by * 64 + txb4) * outdim + k + 16 + txm4 * 4);
                        rB = *reinterpret_cast<float4*>(B + (by * 64 + txb4) * outdim + k + 16 + txm4 * 4);
                        rR = *reinterpret_cast<float4*>(R + (by * 64 + txb4) * outdim + k + 16 + txm4 * 4);
                    }
                    else {
                        rW.x = W[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 0];
                        rW.y = W[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 1];
                        rW.z = W[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 2];
                        rW.w = W[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 3];
                        rB.x = B[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 0];
                        rB.y = B[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 1];
                        rB.z = B[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 2];
                        rB.w = B[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 3];
                        rR.x = R[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 0];
                        rR.y = R[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 1];
                        rR.z = R[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 2];
                        rR.w = R[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 3];
                    }
                }
                else {
                    rW.x = rW.y = rW.z = rW.w = rB.x = rB.y = rB.z = rB.w = rR.x = rR.y = rR.z = rR.w = 0;
                }
                if (bx * 128 + txb4 < bs) {
                    if (aW) {
                        rO[0] = *reinterpret_cast<float4*>(O + (bx * 128 + txb4 + 0) * outdim + k + 16 + txm4 * 4);
                    }
                    else {
                        rO[0].x = O[(bx * 128 + txb4 + 0) * outdim + k + 16 + txm4 * 4 + 0];
                        rO[0].y = O[(bx * 128 + txb4 + 0) * outdim + k + 16 + txm4 * 4 + 1];
                        rO[0].z = O[(bx * 128 + txb4 + 0) * outdim + k + 16 + txm4 * 4 + 2];
                        rO[0].w = O[(bx * 128 + txb4 + 0) * outdim + k + 16 + txm4 * 4 + 3];
                    }
                }
                else {
                    rO[0].x = rO[0].y = rO[0].z = rO[0].w = 0;
                }
                if (bx * 128 + txb4 + 64 < bs) {
                    if (aW) {
                        rO[1] = *reinterpret_cast<float4*>(O + (bx * 128 + txb4 + 64) * outdim + k + 16 + txm4 * 4);
                    }
                    else {
                        rO[1].x = O[(bx * 128 + txb4 + 64) * outdim + k + 16 + txm4 * 4 + 0];
                        rO[1].y = O[(bx * 128 + txb4 + 64) * outdim + k + 16 + txm4 * 4 + 1];
                        rO[1].z = O[(bx * 128 + txb4 + 64) * outdim + k + 16 + txm4 * 4 + 2];
                        rO[1].w = O[(bx * 128 + txb4 + 64) * outdim + k + 16 + txm4 * 4 + 3];
                    }
                }
                else {
                    rO[1].x = rO[1].y = rO[1].z = rO[1].w = 0;
                }
            }
            else if (3 <= q) {
                if (by * 64 + txb4 < indim) {
                    rW.x = W[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 0];
                    rW.y = W[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 1];
                    rW.z = W[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 2];
                    rB.x = B[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 0];
                    rB.y = B[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 1];
                    rB.z = B[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 2];
                    rR.x = R[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 0];
                    rR.y = R[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 1];
                    rR.z = R[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 2];
                    rW.w = rB.w = rR.w = 0;
                }
                else {
                    rW.x = rW.y = rW.z = rW.w = rB.x = rB.y = rB.z = rB.w = rR.x = rR.y = rR.z = rR.w = 0;
                }
                if (bx * 128 + txb4 < bs) {
                    rO[0].x = O[(bx * 128 + txb4 + 0) * outdim + k + 16 + txm4 * 4 + 0];
                    rO[0].y = O[(bx * 128 + txb4 + 0) * outdim + k + 16 + txm4 * 4 + 1];
                    rO[0].z = O[(bx * 128 + txb4 + 0) * outdim + k + 16 + txm4 * 4 + 2];
                    rO[0].w = 0;
                }
                else {
                    rO[0].x = rO[0].y = rO[0].z = rO[0].w = 0;
                }
                if (bx * 128 + txb4 + 64 < bs) {
                    rO[1].x = O[(bx * 128 + txb4 + 64) * outdim + k + 16 + txm4 * 4 + 0];
                    rO[1].y = O[(bx * 128 + txb4 + 64) * outdim + k + 16 + txm4 * 4 + 1];
                    rO[1].z = O[(bx * 128 + txb4 + 64) * outdim + k + 16 + txm4 * 4 + 2];
                    rO[1].w = 0;
                }
                else {
                    rO[1].x = rO[1].y = rO[1].z = rO[1].w = 0;
                }
            }
            else if (2 <= q) {
                if (by * 64 + txb4 < indim) {
                    rW.x = W[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 0];
                    rW.y = W[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 1];
                    rB.x = B[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 0];
                    rB.y = B[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 1];
                    rR.x = R[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 0];
                    rR.y = R[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 1];
                    rW.z = rB.z = rR.z = 0;
                    rW.w = rB.w = rR.w = 0;
                }
                else {
                    rW.x = rW.y = rW.z = rW.w = rB.x = rB.y = rB.z = rB.w = rR.x = rR.y = rR.z = rR.w = 0;
                }
                if (bx * 128 + txb4 < bs) {
                    rO[0].x = O[(bx * 128 + txb4 + 0) * outdim + k + 16 + txm4 * 4 + 0];
                    rO[0].y = O[(bx * 128 + txb4 + 0) * outdim + k + 16 + txm4 * 4 + 1];
                    rO[0].z = 0;
                    rO[0].w = 0;
                }
                else {
                    rO[0].x = rO[0].y = rO[0].z = rO[0].w = 0;
                }
                if (bx * 128 + txb4 + 64 < bs) {
                    rO[1].x = O[(bx * 128 + txb4 + 64) * outdim + k + 16 + txm4 * 4 + 0];
                    rO[1].y = O[(bx * 128 + txb4 + 64) * outdim + k + 16 + txm4 * 4 + 1];
                    rO[1].z = 0;
                    rO[1].w = 0;
                }
                else {
                    rO[1].x = rO[1].y = rO[1].z = rO[1].w = 0;
                }
            }
            else if (1 <= q) {
                if (by * 64 + txb4 < indim) {
                    rW.x = W[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 0];
                    rB.x = B[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 0];
                    rR.x = R[(by * 64 + txb4) * outdim + k + 16 + txm4 * 4 + 0];
                    rW.y = rB.y = rR.y = 0;
                    rW.z = rB.z = rR.z = 0;
                    rW.w = rB.w = rR.w = 0;
                }
                else {
                    rW.x = rW.y = rW.z = rW.w = rB.x = rB.y = rB.z = rB.w = rR.x = rR.y = rR.z = rR.w = 0;
                }
                if (bx * 128 + txb4 < bs) {
                    rO[0].x = O[(bx * 128 + txb4 + 0) * outdim + k + 16 + txm4 * 4 + 0];
                    rO[0].y = 0;
                    rO[0].z = 0;
                    rO[0].w = 0;
                }
                else {
                    rO[0].x = rO[0].y = rO[0].z = rO[0].w = 0;
                }
                if (bx * 128 + txb4 + 64 < bs) {
                    rO[1].x = O[(bx * 128 + txb4 + 64) * outdim + k + 16 + txm4 * 4 + 0];
                    rO[1].y = 0;
                    rO[1].z = 0;
                    rO[1].w = 0;
                }
                else {
                    rO[1].x = rO[1].y = rO[1].z = rO[1].w = 0;
                }
            }
            else {
                rW.x = rW.y = rW.z = rW.w = rB.x = rB.y = rB.z = rB.w = rR.x = rR.y = rR.z = rR.w = 0;
                rO[0].x = rO[0].y = rO[0].z = rO[0].w = 0;
                rO[1].x = rO[1].y = rO[1].z = rO[1].w = 0;
            }
        }

#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
#pragma unroll
            for (int m = 0; m < 8; m++) {
#pragma unroll
                for (int n = 0; n < 4; n++) {
                    float xmb = rX[m][n] - (*psB)[kk][txm16 * 4 + n];
                    float sgn = (xmb == 0) ? 1 : (xmb / fabs(xmb));
                    float r = fabsf((*psR)[kk][txm16 * 4 + n]);
                    if (r != 0) {
                        if (xmb <= r) {
                            rXg[m][n] += (*psO)[kk][txb16 * 8 + m] * -sgn * (*psW)[kk][txm16 * 4 + n] / fabsf(r);
                        }
                        else {
                            rXg[m][n] += (*psO)[kk][txb16 * 8 + m] * -sgn;
                        }
                    }
                }
            }
        }

        if (psW == &sW) {
            psW = &sWb;
            psB = &sBb;
            psR = &sRb;
            psO = &sOb;
        }
        else {
            psW = &sW;
            psB = &sB;
            psR = &sR;
            psO = &sO;
        }

    }

    bool aX = (indim % 4 == 0) ? true : false;

    if (aX) {
#pragma unroll
        for (int m = 0; m < 8; m++) {
            if (bx * 128 + txb16 * 8 + m < bs && by * 64 + txm16 * 4 < indim) {
                float4 t;
                t.x = rXg[m][0];
                t.y = rXg[m][1];
                t.z = rXg[m][2];
                t.w = rXg[m][3];
                *reinterpret_cast<float4*>(XG + (bx * 128 + txb16 * 8 + m) * indim + (by * 64 + txm16 * 4)) = t;
            }
        }
    }
    else {
#pragma unroll
        for (int m = 0; m < 8; m++) {
            if (bx * 128 + txb16 * 8 + m < bs) {
#pragma unroll
                for (int n = 0; n < 4; n++) {
                    if (by * 64 + txm16 * 4 + n < indim) {
                        XG[(bx * 128 + txb16 * 8 + m) * indim + (by * 64 + txm16 * 4 + n)] = rXg[m][n];
                    }
                }
            }
        }
    }

}



DLLEXPORT void cuda_ndmap_forward(float* din, float* dw, float* db, float* dr, float* dout, int indim, int outdim, int bs) {

    dim3 gridDims((int)ceil((float)bs / 256), (int)ceil((float)outdim / 64), 1);
    dim3 blockDims(256, 1, 1);
    out << <gridDims, blockDims >> > (din, dw, db, dr, dout, indim, outdim, bs);

}

DLLEXPORT void cuda_ndmap_paramgrad(float* din, float* dw, float* db, float* dr, float* dout, float* dwg, float* dbg, float* drg, int indim, int outdim, int bs) {

    dim3 gridDims((int)ceil((float)indim / 64), (int)ceil((float)outdim / 64), 1);
    dim3 blockDims(256, 1, 1);
    wrbgrad << <gridDims, blockDims >> > (din, dw, db, dr, dout, dwg, dbg, drg, indim, outdim, bs);

}

DLLEXPORT void cuda_ndmap_inputgrad(float* din, float* dw, float* db, float* dr, float* dout, float* dxg, int indim, int outdim, int bs) {

    dim3 gridDims((int)ceil((float)bs / 128), (int)ceil((float)indim / 64), 1);
    dim3 blockDims(256, 1, 1);
    xgrad << <gridDims, blockDims >> > (din, dw, db, dr, dout, dxg, indim, outdim, bs);

}