#include <torch/extension.h>
#include <vector>
#include <iostream>

namespace IluvatarKernel {

void Instruction_test_f32(const std::string &config, torch::Tensor &inputA, torch::Tensor &inputB, torch::Tensor &out);

} // namespace IluvatarKernel