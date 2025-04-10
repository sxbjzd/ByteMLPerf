#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "IluvatarKernel/include/Instruction_test_f32.h"
#include "IluvatarKernel/include/common.h"

#include "c10/cuda/CUDAGuard.h"
#include "kernels.h"

namespace IluvatarKernel {
void Instruction_test_f32(const std::string &config, torch::Tensor &inputA, torch::Tensor &inputB, torch::Tensor &out) {
  CHECK_TENSOR_CUDA_CONTIGUOUS(inputA);
  CHECK_TENSOR_CUDA_CONTIGUOUS(inputB);
  CHECK_TENSOR_CUDA_CONTIGUOUS(out);
  unsigned numElements = inputA.size(0);
  KERNEL_CHECK(inputA.size(0) == inputB.size(0));
  KERNEL_CHECK(inputA.size(0) == out.size(0));
  KERNEL_CHECK((inputA.dtype() == inputB.dtype()) && (inputA.dtype() == out.dtype()));
  KERNEL_CHECK((inputA.dtype() == at::ScalarType::Float) || (inputA.dtype() == at::ScalarType::Half) || (inputA.dtype() == at::ScalarType::BFloat16));
  KERNEL_CHECK((inputB.dtype() == at::ScalarType::Float) || (inputB.dtype() == at::ScalarType::Half) || (inputB.dtype() == at::ScalarType::BFloat16));
  KERNEL_CHECK((out.dtype() == at::ScalarType::Float) || (out.dtype() == at::ScalarType::Half) || (out.dtype() == at::ScalarType::BFloat16));
  const c10::cuda::OptionalCUDAGuard device_guard(device_of(out));
  auto d_A = inputA.data_ptr();
  auto d_B = inputB.data_ptr();
  auto d_C = out.data_ptr();
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  int  loop = 10;
  auto test = [&](const char* name, auto kernel, int test_ratio) {
      printf("Running %s insruction throughput test\n", name);
      kernel(d_A, d_B, d_C, numElements);//warm up

      cudaEventRecord(start);

      for (int i = 0; i < loop; i++)
          kernel(d_A, d_B, d_C, numElements);

      cudaEventRecord(stop);
      cudaEventSynchronize(stop);

      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      double tflops = (double)numElements * OUT_CNT * INNER_CNT * loop * test_ratio / milliseconds / 1.0e9;
      printf("Elapsed time: %lf ms, TFLOPS: %lf\n", (double)milliseconds, tflops);
  };
    std::string config_lower = config;
    std::transform(config.begin(), config.end(), config_lower.begin(), ::tolower);
    if (config_lower == "fma"){
      if(inputA.dtype() == at::ScalarType::Float){
        test("test_fma", IluvatarKernel::kernel::LaunchKernel_FMA<float>, 2);
      }else if(inputA.dtype() == at::ScalarType::Half){
        test("test_fma", IluvatarKernel::kernel::LaunchKernel_FMA<half>, 2);
      }else if(inputA.dtype() == at::ScalarType::BFloat16){
        test("test_fma", IluvatarKernel::kernel::LaunchKernel_FMA<__nv_bfloat16>, 2);
      }
    }else if(config_lower == "sfu"){
      if(inputA.dtype() == at::ScalarType::Float){
        test("test_sfu", IluvatarKernel::kernel::LaunchKernel_SFU<float>, 1);
      }else if(inputA.dtype() == at::ScalarType::Half){
        test("test_sfu", IluvatarKernel::kernel::LaunchKernel_SFU<half>, 1);
      }else if(inputA.dtype() == at::ScalarType::BFloat16){
        test("test_sfu", IluvatarKernel::kernel::LaunchKernel_SFU<__nv_bfloat16>, 1);
      }
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("Instruction_test_f32", &Instruction_test_f32, "Test instruction throughput float32");
}
} // namespace IluvatarKernel