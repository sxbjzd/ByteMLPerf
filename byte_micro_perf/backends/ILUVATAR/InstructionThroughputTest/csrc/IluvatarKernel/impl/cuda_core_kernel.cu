#include "../include/common.h"

namespace IluvatarKernel::kernel {
template<typename T>
__global__ void test_fma(const void* A, const void* B, void* C, int numElements, int out_cnt)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        float Sum1 = 0;
        float a = (reinterpret_cast<const float *>(A))[i];
        float b = (reinterpret_cast<const float *>(B))[i];

        for (unsigned ou = 0; ou < out_cnt; ou++)
        {
#pragma unroll
            for (unsigned u = 0; u < INNER_CNT; u++)
            {
                if constexpr (std::is_same_v<T, float>){
                    Sum1 = fma(a, b, Sum1);
                }else if constexpr (std::is_same_v<T, half>){
                    Sum1 = __hdot2(reinterpret_cast<half2&>(a), reinterpret_cast<half2&>(b), Sum1);
                }else if constexpr (std::is_same_v<T, __nv_bfloat16>){
                    Sum1 = __hdot2(reinterpret_cast<__nv_bfloat162&>(a), reinterpret_cast<__nv_bfloat162&>(b), Sum1);
                }
            }
        }
        reinterpret_cast<float *>(C)[i] = Sum1;
    }
}
template<typename T>
__global__ void test_sfu(const void* A, const void* B, void* C, int numElements, int out_cnt)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        float res = 0;
        float a = (reinterpret_cast<const float *>(A))[i];

        for (unsigned ou = 0; ou < out_cnt; ou++)
        {
#pragma unroll
            for (unsigned u = 0; u < INNER_CNT; u++)
            {
                a = exp2(a);
            }
        }

        reinterpret_cast<float *>(C)[i] = a;
    }
}

// Launch the Vector CUDA Kernel
template<typename T>
void LaunchKernel_FMA(const void* d_A, const void* d_B, void* d_C, int numElements)
{
    int threadsPerBlock = 1024;
    int elementsPerThread = sizeof(float) / sizeof(T);
    int blocksPerGrid   = (numElements + threadsPerBlock * elementsPerThread - 1) / (threadsPerBlock * elementsPerThread);
    test_fma<T><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements, OUT_CNT);
}
template<typename T>
void LaunchKernel_SFU(const void* d_A, const void* d_B, void* d_C, int numElements)
{
    int threadsPerBlock = 1024;
    int elementsPerThread = sizeof(float) / sizeof(T);
    int blocksPerGrid   = (numElements + threadsPerBlock * elementsPerThread - 1) / (threadsPerBlock * elementsPerThread);
    test_sfu<T><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements, OUT_CNT);
}
template void IluvatarKernel::kernel::LaunchKernel_FMA<float>(const void* , const void* , void* , int);
template void IluvatarKernel::kernel::LaunchKernel_FMA<half>(const void* , const void* , void* , int);
template void IluvatarKernel::kernel::LaunchKernel_FMA<__nv_bfloat16>(const void* , const void* , void* , int);
template void IluvatarKernel::kernel::LaunchKernel_SFU<float>(const void* , const void* , void* , int);
template void IluvatarKernel::kernel::LaunchKernel_SFU<half>(const void* , const void* , void* , int);
template void IluvatarKernel::kernel::LaunchKernel_SFU<__nv_bfloat16>(const void* , const void* , void* , int);
} // namespace IluvatarKernel::kernel
