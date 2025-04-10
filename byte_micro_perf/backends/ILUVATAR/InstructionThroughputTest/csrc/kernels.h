#pragma once

#include <cuda_runtime.h>
#include "IluvatarKernel/include/common.h"

namespace IluvatarKernel::kernel {
template<typename T>
void LaunchKernel_FMA(const void* d_A, const void* d_B, void* d_C, int numElements);
template<typename T>
void LaunchKernel_SFU(const void* d_A, const void* d_B, void* d_C, int numElements);
} // namespace IluvatarKernel::kernel