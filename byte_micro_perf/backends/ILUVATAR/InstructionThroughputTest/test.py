import random
import unittest
import torch
import numpy as np
import itertools

import iluvatar_kernel as iluvatar_kernel

def manual_seed(seed=41):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


manual_seed(43)


@torch.inference_mode()
def run_cuda_core_test(size, dtype):
    inputA = torch.zeros(size).cuda().to(dtype)
    inputB = torch.zeros(size).cuda().to(dtype)
    output = torch.zeros(size).cuda().to(dtype)
        
    output = iluvatar_kernel.cuda_core_test("fma", inputA, inputB, output)
    output = iluvatar_kernel.cuda_core_test("sfu", inputA, inputB, output)

def run_test() -> None:
    size = 1024 * 256
    print(f"Running cuda_core_test_fp32 test size {size}")
    run_cuda_core_test(size, torch.float32)
    print(f"\n Running cuda_core_test_bf16 test size {size}")
    run_cuda_core_test(size, torch.bfloat16)
    print(f"\n Running cuda_core_test_fp16 test size {size}")
    run_cuda_core_test(size, torch.float16)


class Test(unittest.TestCase):
    def test_forward(self):
        run_test()