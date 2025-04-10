#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <bits/stdc++.h>
#define OUT_CNT 1024
#define INNER_CNT 256
namespace IluvatarKernel {

#define DivUp(x, y) (((x) + (y) -1) / (y))
#define INLINE_FUNC __forceinline__


#undef CUDA_CHECK
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        const cudaError_t error_code = call;                                \
        if (error_code != cudaSuccess) {                                    \
            printf("CUDA Error:\n");                                        \
            printf("    File:       %s\n", __FILE__);                       \
            printf("    Line:       %d\n", __LINE__);                       \
            printf("    Error code: %d\n", error_code);                     \
            printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
            throw std::runtime_error("CUDA_CHECK ERROR");                   \
        }                                                                   \
    } while (0)

#undef CHECK_TRUE
#define CHECK_TRUE(value, commit)                                               \
    {                                                                           \
        if (not(value)) {                                                       \
            std::cerr << __FILE__ << " (" << __LINE__ << ")"                    \
                      << "-" << __FUNCTION__ << " : " << (commit) << std::endl; \
            throw std::runtime_error("CHECK_TRUE ERROR");                       \
        }                                                                       \
    }

#undef CHECK_CUDA_KERNEL
#define CHECK_CUDA_KERNEL() CUDA_CHECK(cudaGetLastError());

// from torchCheckMsgImpl
inline const char *IluvatarKernelCheckMsgImpl(const char *msg) { return msg; }
// If there is just 1 user-provided C-string argument, use it.
inline const char *IluvatarKernelCheckMsgImpl(const char *msg, const char *args) {
    return args;
}

#define KERNEL_CHECK_MSG(cond, type, ...)                                     \
    (IluvatarKernelCheckMsgImpl("Expected " #cond " to be true, but got false.  "     \
                          "(Could this error message be improved?  If so, "     \
                          "please report an enhancement request to IluvatarKernel.)", \
                          ##__VA_ARGS__))

#define KERNEL_CHECK(cond, ...)                                                  \
    {                                                                              \
        if (!(cond)) {                                                             \
            std::cerr << __FILE__ << " (" << __LINE__ << ")"                       \
                      << "-" << __FUNCTION__ << " : "                              \
                      << KERNEL_CHECK_MSG(cond, "", ##__VA_ARGS__) << std::endl; \
            throw std::runtime_error("KERNEL_CHECK ERROR");                      \
        }                                                                          \
    }

// 检查tentor: cuda, contiguous
#define CHECK_TENSOR_CUDA_CONTIGUOUS(x) \
    KERNEL_CHECK(x.device().is_cuda());              \
    KERNEL_CHECK(x.is_contiguous());

#define CHECK_TENSOR_MIN_NUMEL(x, min_size)  \
    {                                        \
        int numel_x = 1;                     \
        for (int i = 0; i < x.dim(); ++i) {  \
            numel_x *= x.size(i);            \
        }                                    \
        KERNEL_CHECK(numel_x >= min_size); \
    }

}// namespace IluvatarKernel
