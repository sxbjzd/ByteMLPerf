import torch
from typing import Optional

from ILUVATARKERNEL import Instruction_test_f32

__all__ = ["cuda_core_test"]

def cuda_core_test(
    config: str,
    inputA: torch.Tensor, 
    inputB: torch.Tensor, 
    output: torch.Tensor,
):
    assert inputA.dim() == 1, "only support input dim = 1"
    assert inputB.dim() == 1, "only support input dim = 1"
    assert output.dim() == 1, "only support input dim = 1"

    Instruction_test_f32(config, inputA, inputB, output)

    return output