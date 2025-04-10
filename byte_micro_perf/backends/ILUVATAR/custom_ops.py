import sys
import pathlib
import torch
import random

from functools import partial
from typing import List, Dict, Union, Tuple

FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.utils import logger
from core.utils import OpTensorInfo, calc_tensor_size
from core.op import BasicOp
from core.ops.gemm_ops import GemmOp, GemmFP8Op, GroupGemmFP8Op
from core.ops.attn_ops import FlashAttentionOp

# import cutlass
# from backends.ILUVATAR.ixgemmblaslt import gemm88

# from backends.module_store import GemmOp, BatchGemmOp, GroupGemmOp, AddOp
import ixformer.functions as F
import ixformer.distributed as ixfd
from ixformer.contrib.vllm_flash_attn import flash_attn_varlen_func, flash_attn_with_cache_batch_idx_int8

class ILUVATARAddOp(BasicOp):
    # def forward(self, input_tensor_a, input_tensor_b, input_tensor_c):
    #     F.add(input_tensor_a, input_tensor_b, out=input_tensor_c)
    #     return input_tensor_c

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)
        
        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "a": OpTensorInfo(
                shape=[self.batch_size, self.dim_size],
                dtype=self.torch_dtype, 
                device=self.backend.get_device(), 
            ), 
            "b": OpTensorInfo(
                shape=[self.batch_size, self.dim_size],
                dtype=self.torch_dtype, 
                device=self.backend.get_device(), 
            )
        }
        self.output_tensor_info = {
            "c": OpTensorInfo(
                shape=[self.batch_size, self.dim_size],
                dtype=self.torch_dtype, 
                device=self.backend.get_device(), 
            )
        }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = 0
        self.bus_size = 0

        self.calc_flops = self.batch_size * self.dim_size 

        self._run_func = self.add_run

    def add_run(self, tensor_mapping):
        a = tensor_mapping["a"]
        b = tensor_mapping["b"]
        c = tensor_mapping["c"]
        c = F.add(a, b, out=c)
        # F.add(input_tensor_a, input_tensor_b, out=input_tensor_c)
        return c

# gemm(pytorch) float32/float16/bfloat16 --> float32/float16/bfloat16
# gemm(cutlass) int8 --> int32
class ILUVATARGemmOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.dtype = self.args_dict["dtype"]
        
        self.M = self.args_dict["M"]
        self.K = self.args_dict["K"]
        self.N = self.args_dict["N"]

        # bf16 * bf16 --> bf16
        # fp16 * fp16 --> fp16
        if self.dtype in ["bfloat16", "float16"]:
            self.torch_dtype = getattr(torch, self.dtype)
            self.out_dtype = self.torch_dtype
        # fp32 * fp32 --> fp32
        elif self.dtype == "float32":
            self.torch_dtype = torch.float32
            self.out_dtype = torch.float32
            # use float32 gemm
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        # fp32(tf32) * fp32(tf32) --> fp32
        elif self.dtype == "tfloat32":
            self.torch_dtype = torch.float32
            self.out_dtype = torch.float32
            # use tfloat32 gemm
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        # int8 (+scale) * int8 (+scale) --> bf16
        elif self.dtype == "int8":
            self.torch_dtype = torch.int8
            self.out_dtype = torch.bfloat16
        else:
            raise NotImplementedError

        if self.dtype in ["float32", "tfloat32", "float16", "bfloat16"]:
            self.input_tensor_info = {
                "a": OpTensorInfo(
                    shape=[self.M, self.K],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                ),
                "b": OpTensorInfo(
                    shape=[self.K, self.N],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                ),
            }
            self.output_tensor_info = {
                "c": OpTensorInfo(
                    shape=[self.M, self.N],
                    dtype=self.out_dtype,
                    device=self.backend.get_device(),
                )
            }
        elif self.dtype == "int8":
            self.input_tensor_info = {
                "a": OpTensorInfo(
                    shape=[self.M, self.K],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                ),
                "b": OpTensorInfo(
                    shape=[self.K, self.N],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                ),
                "a_scale": OpTensorInfo(
                    shape=[self.M],
                    dtype=torch.float32,
                    device=self.backend.get_device(),
                ),
                "b_scale": OpTensorInfo(
                    shape=[self.N],
                    dtype=torch.float32,
                    device=self.backend.get_device(),
                ),
            }
            self.output_tensor_info = {
                "c": OpTensorInfo(
                    shape=[self.M, self.N],
                    dtype=self.out_dtype,
                    device=self.backend.get_device(),
                )
            }
        
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes
        
        self.calc_flops = self.M * self.N * self.K * 2
        self._run_func = self.gemm_run

    def _create_in_out_tensors(
        self, 
        instance_num, 
        create_inputs=True, 
        create_outputs=True,
    ):
        all_tensor_list = []

        # create first instance
        first_tensor_mapping = {}
        if create_inputs:
            if 'a_scale' in self.input_tensor_info:
                a_shape, b_shape = self.input_tensor_info['a'].shape, self.input_tensor_info['b'].shape
                xpu_device = self.input_tensor_info['a'].device
                M, K = a_shape
                _, N = b_shape
                d_shape = [M, N]
                pad_m = 0
                pad_k = 0
                if K % 64 != 0:
                    pad_k = 64 - K % 64
                if M % 2 != 0:
                    pad_m = 2 - M % 2
                # 为了让值合理，我们进行量化的到int8
                a_tensor = torch.randn(a_shape, dtype=torch.float16, device=xpu_device)
                b_tensor = torch.randn(b_shape, dtype=torch.float16, device=xpu_device)
                # create output tensors
                d_tensor = torch.randn(d_shape, dtype=torch.float16, device=xpu_device)
                # pad 0
                a_tensor = torch.nn.functional.pad(a_tensor, (0,pad_k,0,pad_m))
                
                d_tensor = torch.nn.functional.pad(d_tensor, (0, 0, 0,pad_m))
                b_tensor = b_tensor.T.contiguous()
                b_tensor = torch.nn.functional.pad(b_tensor, (0,pad_k))
                a_tensor_i8, a_tensor_scales = F.dynamic_scaled_quant_smoothquant(a_tensor)
                b_tensor_i8, b_tensor_scales = F.dynamic_scaled_quant_smoothquant(b_tensor)
                first_tensor_mapping['a'] = a_tensor_i8
                first_tensor_mapping['b'] = b_tensor_i8
                first_tensor_mapping['a_scale'] = a_tensor_scales
                first_tensor_mapping['b_scale'] = b_tensor_scales
                first_tensor_mapping['c'] = d_tensor
            else:
                for key, value in self.input_tensor_info.items():
                    first_tensor_mapping[key] = torch.zeros(
                        size=value.shape,
                        dtype=value.dtype,
                        device=value.device
                    )
                    if value.device == "cpu":
                        first_tensor_mapping[key] = first_tensor_mapping[key].pin_memory()
        if create_outputs:
            if 'a_scale' in self.input_tensor_info:
                pass
            else:
                for key, value in self.output_tensor_info.items():
                    first_tensor_mapping[key] = torch.zeros(
                        size=value.shape,
                        dtype=value.dtype,
                        device=value.device
                    )
                    if value.device == "cpu":
                        first_tensor_mapping[key] = first_tensor_mapping[key].pin_memory()
        all_tensor_list.append(first_tensor_mapping)


        # clone following instances
        for _ in range(instance_num - 1):
            tensor_mapping = {}
            for key, value in first_tensor_mapping.items():
                tensor_mapping[key] = value.clone()
            all_tensor_list.append(tensor_mapping)

        return all_tensor_list
        
    def gemm_run(self, tensor_mapping):
        a = tensor_mapping["a"]
        b = tensor_mapping["b"]
        c = tensor_mapping["c"]
        if self.torch_dtype == torch.int8:
            input_tensor_a_scales = tensor_mapping['a_scale']
            input_tensor_b_scales = tensor_mapping['b_scale']
            F.w8a8(a, b, input_tensor_a_scales, 
                input_tensor_b_scales, output=c)
        else:
            if self.torch_dtype == torch.float32:
                torch.mm(a, b, out=c)
            else:
                F.mm(a, b, out=c)
        return c

"""
attn ops
"""
class ILUVATARFlashAttentionOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        # llm args
        self.arg_type = self.args_dict["arg_type"]
        if self.arg_type != "llm":
            raise NotImplementedError

        # llm phase: prefill or decode
        self.phase = self.args_dict["phase"]
        if self.phase not in ["prefill"]:
            raise NotImplementedError

        # dtype: bfloat16
        self.dtype = self.args_dict["dtype"]
        if self.dtype not in  ["bfloat16", "int8", "qkint8"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, "int8") if self.dtype == "qkint8" else getattr(torch, self.dtype)
        self.torch_dtype_size = torch.tensor([], dtype=self.torch_dtype).element_size()


        self.batch_size = self.args_dict["batch_size"]
        self.q_seq_len = self.args_dict["q_seq_len"]
        self.kv_seq_len = self.args_dict["kv_seq_len"]
        self.q_head_num = self.args_dict["q_head_num"]
        self.kv_head_num = self.args_dict["kv_head_num"]
        self.head_dim = self.args_dict["head_dim"]
        
        self.is_causal = self.args_dict["is_causal"]
        if not self.is_causal:
            raise NotImplementedError

        self.softmax_scale = self.head_dim ** (-0.5)

        if self.dtype == "bfloat16":
            self.input_tensor_info = {
                "q": OpTensorInfo(
                    shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.head_dim],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                ),
                "k": OpTensorInfo(
                    shape=[self.batch_size, self.kv_seq_len, self.kv_head_num, self.head_dim],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                ),
                "v": OpTensorInfo(
                    shape=[self.batch_size, self.kv_seq_len, self.kv_head_num, self.head_dim],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                )
            }
            self.output_tensor_info = {
                "out": OpTensorInfo(
                    shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.head_dim],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                )    
            }
        # elif self.dtype == "int8":
        elif "int8" in self.dtype:
            self.input_tensor_info = {
                "q": OpTensorInfo(
                    shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.head_dim],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                ),
                "k_cache": OpTensorInfo(
                    shape=[self.batch_size, self.kv_head_num, self.kv_seq_len, self.head_dim],
                    dtype=self.torch_dtype,
                    device=self.backend.get_device(),
                ),
                "v_cache": OpTensorInfo(
                    shape=[self.batch_size, self.kv_head_num, self.kv_seq_len, self.head_dim],
                    dtype=torch.bfloat16 if self.dtype=="qkint8" else torch.int8,
                    device=self.backend.get_device(),
                ),
                "q_scales": OpTensorInfo(
                    shape=[self.batch_size, self.q_head_num, (self.q_seq_len+15)//16],
                    dtype=torch.float32,
                    device=self.backend.get_device(),
                ),
                "k_scales": OpTensorInfo(
                    shape=[self.batch_size, self.kv_head_num, (self.kv_seq_len+15)//16],
                    dtype=torch.float32,
                    device=self.backend.get_device(),
                ),
                "v_scales": OpTensorInfo(
                    shape=[self.batch_size, self.kv_head_num, self.head_dim],
                    dtype=torch.float32,
                    device=self.backend.get_device(),
                ),
            }
            self.output_tensor_info = {
                "out": OpTensorInfo(
                    shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.head_dim],
                    dtype=torch.bfloat16,
                    device=self.backend.get_device(),
                )    
            }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes


        p_gemm_b = self.batch_size * self.q_head_num
        p_gemm_m = self.q_seq_len
        p_gemm_k = self.head_dim
        p_gemm_n = self.kv_seq_len
        p_gemm_calc_flops = p_gemm_b * p_gemm_m * p_gemm_k * p_gemm_n * 2

        o_gemm_b = self.batch_size * self.q_head_num
        o_gemm_m = self.q_seq_len
        o_gemm_k = self.kv_seq_len
        o_gemm_n = self.head_dim
        o_gemm_calc_flops = o_gemm_b * o_gemm_m * o_gemm_k * o_gemm_n * 2

        flops_ratio = (1 + self.kv_seq_len) * self.q_seq_len / 2 / (self.q_seq_len * self.kv_seq_len) if self.is_causal else 1
        self.calc_flops = (p_gemm_calc_flops + o_gemm_calc_flops) * flops_ratio

        self._run_func = self.fa_run

    def _create_in_out_tensors(
        self, 
        instance_num, 
        create_inputs=True, 
        create_outputs=True,
    ):
        all_tensor_list = []

        # create first instance
        first_tensor_mapping = {}
        if create_inputs:
            if "q_scales" in self.input_tensor_info:
                device = self.input_tensor_info['q'].device
                batch, q_len, num_attention_heads, head_dim = self.input_tensor_info['q'].shape
                q = torch.empty([batch, q_len, num_attention_heads, head_dim], dtype=self.input_tensor_info['q'].dtype, device=device)
                batch, num_key_value_heads, kv_len, head_dim = self.input_tensor_info['k_cache'].shape
                k_cache = torch.empty([batch, num_key_value_heads, kv_len, head_dim], device=device, dtype=self.input_tensor_info['k_cache'].dtype)
                v_cache = torch.empty([batch, num_key_value_heads, kv_len, head_dim], device=device, dtype=self.input_tensor_info['v_cache'].dtype)
                q_scales = torch.randn(batch, num_attention_heads, (q_len+15)//16, device=device, dtype=self.input_tensor_info['q_scales'].dtype)
                k_scales = torch.randn(batch, num_key_value_heads, (q_len+15)//16, device=device, dtype=self.input_tensor_info['k_scales'].dtype)
                v_scales = torch.randn(batch, num_key_value_heads, head_dim, device=device, dtype=self.input_tensor_info['v_scales'].dtype)
                # cache_seqlens = torch.randint(batch, dtype=torch.int32, device=device)
                # cache_seqlens = (torch.randint(1, q_len, [batch,],).int().to(device))
                cache_seqlens = (torch.ones(batch).int()*q_len).to(device)
                # cache_batch_idx = torch.randint(batch, dtype=torch.int32, device=device)
                cache_batch_idx = torch.tensor(list(range(batch))[:batch]).int().cuda()
                max_context_len = cache_seqlens.max().item()
                softmax_scale = head_dim**-0.5
                output = torch.zeros(batch, q_len, num_attention_heads, head_dim, device=device, dtype=torch.bfloat16)
                first_tensor_mapping['q'] = q
                first_tensor_mapping['k_cache'] = k_cache
                first_tensor_mapping['v_cache'] = v_cache
                first_tensor_mapping['q_scales'] = q_scales
                first_tensor_mapping['k_scales'] = k_scales
                first_tensor_mapping['v_scales'] = v_scales
                first_tensor_mapping['cache_seqlens'] = cache_seqlens
                first_tensor_mapping['cache_batch_idx'] = cache_batch_idx
                first_tensor_mapping['max_context_len'] = max_context_len
                first_tensor_mapping['softmax_scale'] = softmax_scale
                first_tensor_mapping['output'] = output
            else:
                device = self.input_tensor_info['q'].device
                batch, q_len, num_attention_heads, head_dim = self.input_tensor_info['q'].shape
                q = torch.randn(batch * q_len, num_attention_heads, head_dim, dtype=self.input_tensor_info['q'].dtype, device=device)
                batch, kv_len, num_key_value_heads, head_dim = self.input_tensor_info['k'].shape
                k = torch.randn(batch * kv_len, num_key_value_heads, head_dim, dtype=self.input_tensor_info['k'].dtype, device=device)
                k /= head_dim**0.5
                v = torch.randn_like(k)
                v /= head_dim**0.5
                cu_query_lens = torch.tensor([0, q_len], dtype=torch.int32, device=device).cumsum(dim=0, dtype=torch.int32)
                cu_kv_lens = torch.tensor([0, kv_len], dtype=torch.int32, device=device).cumsum(dim=0, dtype=torch.int32)

                first_tensor_mapping['q'] = q
                first_tensor_mapping['k'] = k
                first_tensor_mapping['v'] = v
                first_tensor_mapping['cu_query_lens'] = cu_query_lens
                first_tensor_mapping['cu_kv_lens'] = cu_kv_lens
                first_tensor_mapping['scale'] = head_dim**-0.5
                first_tensor_mapping['max_query_len'] = q_len
                first_tensor_mapping['max_kv_len'] = kv_len
                first_tensor_mapping['out'] = torch.randn_like(first_tensor_mapping['q'])
        if create_outputs:
            pass
        all_tensor_list.append(first_tensor_mapping)


        # clone following instances
        for _ in range(instance_num - 1):
            tensor_mapping = {}
            for key, value in first_tensor_mapping.items():
                if isinstance(value, torch.Tensor):
                    tensor_mapping[key] = value.clone()
                else:
                    tensor_mapping[key] = value
            all_tensor_list.append(tensor_mapping)

        return all_tensor_list

    def fa_run(self, tensor_mapping):
        # if self.torch_dtype == torch.int8:
        if "int8" in self.dtype:
            q = tensor_mapping['q']
            k_cache = tensor_mapping['k_cache']
            v_cache = tensor_mapping['v_cache']
            q_scales = tensor_mapping['q_scales']
            k_scales = tensor_mapping['k_scales']
            v_scales = tensor_mapping['v_scales'] if self.dtype=="int8" else None
            cache_seqlens = tensor_mapping['cache_seqlens']
            cache_batch_idx = tensor_mapping['cache_batch_idx']
            max_context_len = tensor_mapping['max_context_len']
            softmax_scale = tensor_mapping['softmax_scale']
            output = tensor_mapping['output']
            output = flash_attn_with_cache_batch_idx_int8(
                q,
                k_cache,
                v_cache,
                q_scales,
                k_scales,
                v_scales,
                cache_seqlens,
                cache_batch_idx,
                max_context_len,
                softmax_scale,
                causal=True,
                output_dtype=torch.float16,
                output=output
            )
        else:
            query = tensor_mapping['q']
            key = tensor_mapping['k']
            value = tensor_mapping['v']
            cu_query_lens = tensor_mapping['cu_query_lens']
            cu_kv_lens = tensor_mapping['cu_kv_lens']
            max_query_len = tensor_mapping['max_query_len']
            max_kv_len = tensor_mapping['max_kv_len']
            scale = tensor_mapping['scale']
            output = flash_attn_varlen_func(
                query,
                key,
                value,
                cu_query_lens,
                cu_kv_lens,
                max_query_len,
                max_kv_len,
                softmax_scale=scale,
                causal=True,
                )
        return output
        



class BatchGemmOp(torch.nn.Module):
    def forward(self, input_tensor_a, input_tensor_b, input_tensor_d):
        compute_dtype = input_tensor_a.dtype
        if compute_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            torch.bmm(input_tensor_a, input_tensor_b, out=input_tensor_d)
        else:
            raise Exception(f"BatchGemmOp with dtype {compute_dtype} is not implemented")
# batch_gemm(pytorch)   float32/float16/bfloat16 --> float32/float16/bfloat16
# batch_gemm(cutlass)   int8 --> int32
class ILUVATARBatchGemmOp(BatchGemmOp):
    def forward(
        self, 
        *args
    ):
        compute_dtype = args[0].dtype
        if compute_dtype == torch.int8:
            input_tensor_a, input_tensor_b, alpha, a_scale, b_scale, d_tensor = args
            output_tensor = F.bmm(input_tensor_a, input_tensor_b, alpha, 
                                  input_scales=a_scale, 
                                  mat2_scales=b_scale,
                                  out_dtype=torch.float16,
                                  format="NT",
                                  out=d_tensor)
        elif compute_dtype == torch.float32:
            input_tensor_a, input_tensor_b, d_tensor = args
            output_tensor = torch.bmm(input_tensor_a, input_tensor_b, out=d_tensor)
        else:
            input_tensor_a, input_tensor_b, d_tensor = args
            output_tensor = F.bmm(input_tensor_a, input_tensor_b, 1.0, out=d_tensor)
        return output_tensor


class GroupGemmOp(torch.nn.Module):
    def forward(self, input_tensor_a, input_tensor_b, input_tensor_d):
        compute_dtype = input_tensor_a[0].dtype
        for a, b, d in zip(input_tensor_a, input_tensor_b, input_tensor_d):
            if compute_dtype in [torch.float32, torch.float16, torch.bfloat16]:
                torch.mm(a, b, out=d)
            else:
                raise Exception(f"GroupGemmOp with dtype {compute_dtype} is not implemented")
# group_gemm(pytorch)   float32/float16/bfloat16 --> float32/float16/bfloat16
# group_gemm(cutlass)   int8 --> int32
class ILUVATARGroupGemmOp(GroupGemmOp):
    def forward(self, 
        *args
    ):
        if isinstance(args[0], list):
            compute_dtype = args[0][0].dtype
        else:
            compute_dtype = args[0].dtype
        if compute_dtype == torch.int8:
            # output_tensors = gemm88.gemm_run(self.blasLtIns, a_list, b_list)
            # a_list, a_list_scale, b_list, b_list_scale = args
            # output_tensors = [F.w8a8(a, b, a_s,
            #     b_s, out_dtype=torch.bfloat16)
            #     for a, a_s, b, b_s in zip(a_list, a_list_scale, b_list, b_list_scale) ]
            i8_left_tensor, i8_right_tensors, left_scales, right_scales, tokens_per_group = args
            output_tensors = F.moe_w8a8_group_gemm(
                                i8_left_tensor,
                                i8_right_tensors,
                                left_scales,
                                right_scales,
                                torch.float16,
                                tokens_per_group,
                                None,
                                "NT",
                                None,
                            )
        elif compute_dtype == torch.float32:
            a_list, b_list, d_list = args
            output_tensors = [torch.mm(a, b, out=c) for a, b, c in zip(a_list, b_list, d_list)]
        else:
            a_list, b_list, d_list = args
            output_tensors = [F.mm(a, b, out=c) for a, b, c in zip(a_list, b_list, d_list)]
        return output_tensors

class ILUVATARAllReduceOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)

        self.world_size = self.args_dict["world_size"]
        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_device(),
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_device(),
            )
        }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = 0
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.input_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.input_tensor_size
        self.bus_size = 2 * (self.world_size - 1) * self.algo_size / self.world_size

        self.calc_flops = 0

        self._run_func = self.all_reduce_run
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False
        )


    def all_reduce_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        # dist_module = self.backend.get_dist_module()
        # dist_module.all_reduce(src, op=dist_module.ReduceOp.SUM, group=self.op_group)
        #import torch.distributed as dist
        #print(f"rank: {dist.get_rank(group=self.op_group)}, device: {torch.cuda.current_device()}, tensor device: {src.device}, numel: {src.numel()}")
        ixfd.all_reduce(src, op=ixfd.ReduceOp.SUM, async_op=True, group=self.op_group)
        return src

    def is_concurrent():
        return True


############### example ops #################
"""
gemm ops
"""
class GPUGemmOp(GemmOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

        if self.dtype == "float32":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        elif self.dtype == "tfloat32":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True




try:
    import deep_gemm
    from deep_gemm import bench_kineto, calc_diff, ceil_div, get_col_major_tma_aligned_tensor
except ImportError:
    deep_gemm = None
    # logger.warning("deep_gemm is not available, please install it first.")

FP8_E4M3_MAX = 448.0  # Maximum representable value in FP8 E4M3 format

def per_token_cast_to_fp8(x: torch.Tensor, group_size=128) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % group_size == 0
    m, n = x.shape
    x_view = x.view(m, -1, group_size)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (
        (x_view * (FP8_E4M3_MAX / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n),
        (x_amax / FP8_E4M3_MAX).view(m, -1)
    )

def per_block_cast_to_fp8(x: torch.Tensor, group_size=128) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, group_size) * group_size, ceil_div(n, group_size) * group_size), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, group_size, x_padded.size(1) // group_size, group_size)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (FP8_E4M3_MAX / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / FP8_E4M3_MAX).view(x_view.size(0), x_view.size(2))

def construct(m: int, k: int, n: int, group_size, device) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    y = torch.randn((n, k), device=device, dtype=torch.bfloat16)
    out = torch.empty((m, n), device=device, dtype=torch.bfloat16)

    x_fp8, y_fp8 = per_token_cast_to_fp8(x, group_size), per_block_cast_to_fp8(y, group_size)
    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out


def construct_grouped(num_groups: int, m: int, k: int, n: int, is_masked: bool, group_size, device) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((num_groups, m, k), device=device, dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device=device, dtype=torch.bfloat16)
    out = torch.empty((num_groups, m, n), device=device, dtype=torch.bfloat16)

    assert m % 4 == 0, f'TMA alignment error: {m}'
    x_fp8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.empty((num_groups, m, k // group_size), device=device, dtype=torch.float))
    y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((num_groups, (n + group_size - 1) // group_size, k // group_size), device=device, dtype=torch.float))
    for i in range(num_groups):
        x_fp8[0][i], x_fp8[1][i] = per_token_cast_to_fp8(x[i], group_size)
        y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i], group_size)

    # For non-masked input, we must merge the group and M dims
    if not is_masked:
        x_fp8 = (x_fp8[0].view(-1, k), per_token_cast_to_fp8(x.view(-1, k), group_size)[1])
        out = out.view(-1, n)

    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out



class GPUGemmFP8Op(GemmFP8Op):
    def __init__(self, args_dict, backend, *args, **kwargs):
        if deep_gemm is None:
            raise ImportError("deep_gemm is not available, please install it first.")
            
        super().__init__(args_dict, backend, *args, **kwargs)

        self._custom_run = True
        self._run_func = self.gemm_fp8_run


    def gemm_fp8_run(self):
        def test_func():
            x_fp8, y_fp8, out = construct(
                self.M, self.K, self.N, 
                self.quant_group_size, 
                self.backend.get_torch_device_name()
            )
            deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        return t * 1e6



class GPUGroupGemmFP8Op(GroupGemmFP8Op):
    def __init__(self, args_dict, backend, *args, **kwargs):
        if deep_gemm is None:
            raise ImportError("deep_gemm is not available, please install it first.")

        super().__init__(args_dict, backend, *args, **kwargs)

        self._custom_run = True
        self._run_func = self.group_gemm_fp8_run

    def group_gemm_fp8_run(self):

        def test_func_contiguous():
            x_fp8, y_fp8, out = construct_grouped(
                self.num_groups, self.M, self.K, self.N, False, 
                self.quant_group_size, 
                self.backend.get_torch_device_name()
            )
            m_indices = torch.arange(0, self.num_groups, device='cuda', dtype=torch.int)
            m_indices = m_indices.unsqueeze(-1).expand(self.num_groups, self.M).contiguous().view(-1)
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)

        def test_func_masked():
            x_fp8, y_fp8, out = construct_grouped(
                self.num_groups, self.M, self.K, self.N, True, 
                self.quant_group_size, 
                self.backend.get_torch_device_name()
            )
            masked_m = torch.ones((self.num_groups, ), device='cuda', dtype=torch.int) * self.M
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x_fp8, y_fp8, out, masked_m, self.M)

        if self.mode == "contiguous":
            t = bench_kineto(test_func_contiguous, 'fp8_gemm', suppress_kineto_output=True)
        elif self.mode == "masked":
            t = bench_kineto(test_func_masked, 'fp8_gemm', suppress_kineto_output=True)
        return t * 1e6




# try:
#     import flash_attn
#     from flash_attn import flash_attn_func
# except ImportError:
#     flash_attn = None

# try:
#     import flash_attn_interface
#     from flash_attn_interface import flash_attn_func
# except ImportError:
#     flash_attn_interface = None

# class GPUFlashAttentionOp(FlashAttentionOp):
#     def __init__(self, args_dict, backend, *args, **kwargs):
#         if flash_attn is None and flash_attn_interface is None:
#             raise ImportError("flash_attention is not available, please install it first.")

#         super().__init__(args_dict, backend, *args, **kwargs)
#         self._run_func = self.flash_attention_run

#         # create output tensor during testing
#         self.output_tensor_info = {}

#     def flash_attention_run(self, tensor_mapping):
#         q = tensor_mapping["q"]
#         k = tensor_mapping["k"]
#         v = tensor_mapping["v"]        
#         return flash_attn_func(q, k, v, causal=self.is_causal)





try:
    import flash_mla
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata
except ImportError:
    flash_mla = None
    # logger.warning("flash_mla is not available, please install it first.")

class GPUFlashMLAOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        if flash_mla is None or flash_attn_interface is None:
            raise ImportError("flash_mla or flash_attn is not available, please install it first.")

        super().__init__(args_dict, backend, *args, **kwargs)
        self._run_func = self.flash_mla_run

    def prepare(self):
        # llm args
        self.arg_type = self.args_dict["arg_type"]
        if self.arg_type != "llm":
            raise NotImplementedError

        # llm phase: prefill or decode
        self.phase = self.args_dict["phase"]
        if self.phase not in ["prefill", "decode"]:
            raise NotImplementedError

        # dtype: bfloat16
        self.dtype = self.args_dict["dtype"]
        if self.dtype != "bfloat16":
            raise NotImplementedError
        self.torch_dtype = torch.bfloat16
        self.torch_dtype_size = torch.tensor([], dtype=self.torch_dtype).element_size()


        self.batch_size = self.args_dict["batch_size"]
        self.q_seq_len = self.args_dict["q_seq_len"]
        self.kv_seq_len = self.args_dict["kv_seq_len"]
        self.q_head_num = self.args_dict["q_head_num"]
        self.kv_head_num = self.args_dict["kv_head_num"]
        self.qk_dim_size = self.args_dict["qk_dim_size"]
        self.v_dim_size = self.args_dict["v_dim_size"]

        self.is_causal = self.args_dict["is_causal"]
        if not self.is_causal:
            raise NotImplementedError

        self.varlen = self.args_dict["varlen"]
        if self.varlen:
            raise NotImplementedError



        # q: [batch_size, q_seq_len, q_head_num, qk_dim_size]
        self.q = torch.randn(
            self.batch_size, self.q_seq_len, self.q_head_num, self.qk_dim_size,
            dtype=self.torch_dtype,
            device=self.backend.get_torch_device_name()
        )

        # prefill, not absorb weight, use flash_attention
        if self.phase == "prefill":
            self.k = torch.randn(
                self.batch_size, self.kv_seq_len, self.kv_head_num, self.qk_dim_size,
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name()
            )
            self.v = torch.randn(
                self.batch_size, self.kv_seq_len, self.kv_head_num, self.qk_dim_size,
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name()
            )

            self.input_tensor_size = {
                "q": OpTensorInfo(
                    shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.qk_dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name()
                ), 
                "k": OpTensorInfo(
                    shape=[self.batch_size, self.kv_seq_len, self.kv_head_num, self.qk_dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name()
                ),
                "v": OpTensorInfo(
                    shape=[self.batch_size, self.kv_seq_len, self.kv_head_num, self.qk_dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name()
                )
            }
            self.output_tensor_size = {
                "out": OpTensorInfo(
                    shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.qk_dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name()
                )
            }

            self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_size.values()])
            self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_size.values()])
            self.tensor_size = self.input_tensor_size + self.output_tensor_size

            self.read_bytes = self.input_tensor_size
            self.write_bytes = self.output_tensor_size
            self.io_bytes = self.read_bytes + self.write_bytes

            self.algo_size = 0
            self.bus_size = 0

            self.attn_ratio = (1 + self.kv_seq_len) / 2 / self.kv_seq_len
            self.calc_flops = self.batch_size * self.q_head_num * self.q_seq_len * self.kv_seq_len * (self.qk_dim_size + self.v_dim_size) * 2 * self.attn_ratio


        # decode, absorb weight, use flash_mla
        elif self.phase == "decode":
            self.cache_seqlens = torch.full(
                (self.batch_size,), 
                self.kv_seq_len, 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name()
            )
            self.total_seqlens = self.cache_seqlens.sum().item()
            self.mean_seqlens = self.cache_seqlens.float().mean().int().item()
            self.max_seqlen = self.cache_seqlens.max().item()
            self.max_seqlen_pad = (self.max_seqlen + 255) // 256 * 256

            self.block_size = 64
            self.block_table = torch.arange(
                self.batch_size * self.max_seqlen_pad // self.block_size,
                dtype=torch.int32,
                device=self.backend.get_torch_device_name()
            ).view(self.batch_size, self.max_seqlen_pad // self.block_size)

            self.blocked_k = torch.randn(
                self.block_table.numel(), self.block_size, self.kv_head_num, self.qk_dim_size,
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name()
            )
            for i in range(self.batch_size):
                self.blocked_k.view(self.batch_size, self.max_seqlen_pad, self.kv_head_num, self.qk_dim_size)[i, self.cache_seqlens[i].item():] = (
                    float("nan")
                )
            self.tile_scheduler_metadata, self.num_splits = get_mla_metadata(
                self.cache_seqlens, self.q_seq_len * self.q_head_num // self.kv_head_num, self.kv_head_num
            )

            # q:            [batch_size, q_seq_len, q_head_num, qk_dim_size]
            # blocked_k:    [batch_size * max_seqlen_pad // block_size, block_size, kv_head_num, qk_dim_size]
            # block_table:  [batch_size, max_seqlen_pad // block_size]
            # cache_seqlens:[batch_size]
            self.input_tensor_size = {
                "q": OpTensorInfo(
                    shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.qk_dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name()
                ), 
                "blocked_k": OpTensorInfo(
                    shape=[self.block_table.numel(), self.block_size, self.kv_head_num, self.qk_dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name()
                ),
                "block_table": OpTensorInfo(
                    shape=[self.batch_size, self.max_seqlen_pad // self.block_size],
                    dtype=torch.int32,
                    device=self.backend.get_torch_device_name()
                ),
                "cache_seqlens": OpTensorInfo(
                    shape=[self.batch_size],
                    dtype=torch.int32,
                    device=self.backend.get_torch_device_name()
                )
            }

            # out:          [batch_size, q_seq_len, q_head_num, v_dim_size]
            # softmax_lse   [batch_size, q_head_num, q_seq_len]
            self.output_tensor_size = {
                "out": OpTensorInfo(
                    shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.v_dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name()
                ),
                "softmax_lse": OpTensorInfo(
                    shape=[self.batch_size, self.q_head_num, self.q_seq_len],
                    dtype=torch.float32,
                    device=self.backend.get_torch_device_name()
                )
            }
            self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_size.values()])
            self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_size.values()])
            self.tensor_size = self.input_tensor_size + self.output_tensor_size

            # q + kv_compress, ignore block_table and cache_seqlens
            self.read_bytes = \
                (self.batch_size * self.q_seq_len * self.q_head_num * self.qk_dim_size + \
                self.total_seqlens * self.kv_head_num * self.qk_dim_size) * self.torch_dtype_size
            # out + softmax_lse
            self.write_bytes = self.output_tensor_size
            self.io_bytes = self.read_bytes + self.write_bytes

            self.algo_size = 0
            self.bus_size = 0

            # q * k, p * v
            self.calc_flops = self.total_seqlens * self.q_head_num * self.q_seq_len * (self.qk_dim_size + self.v_dim_size) * 2



    def create_tensors(self, instance_num : int):
        all_tensor_list = []
        for i in range(instance_num):
            tensor_mapping = {}
            if self.phase == "prefill":
                tensor_mapping["q"] = self.q.clone()
                tensor_mapping["k"] = self.k.clone()
                tensor_mapping["v"] = self.v.clone()
            elif self.phase == "decode":
                tensor_mapping["q"] = self.q.clone()
                tensor_mapping["blocked_k"] = self.blocked_k.clone()
                tensor_mapping["block_table"] = self.block_table.clone()
                tensor_mapping["cache_seqlens"] = self.cache_seqlens.clone()
            all_tensor_list.append(tensor_mapping)
        return all_tensor_list



    @torch.inference_mode()
    def flash_mla_run(self, tensor_mapping):
        if self.phase == "prefill":
            q = tensor_mapping["q"]
            k = tensor_mapping["k"]
            v = tensor_mapping["v"]
            return flash_attn_func(q, k, v, causal=self.is_causal)
        elif self.phase == "decode":
            q = tensor_mapping["q"]
            blocked_k = tensor_mapping["blocked_k"]
            block_table = tensor_mapping["block_table"]
            cache_seqlens = tensor_mapping["cache_seqlens"]
            return_vals = flash_mla_with_kvcache(
                q,
                blocked_k,
                block_table,
                cache_seqlens,
                self.v_dim_size,
                self.tile_scheduler_metadata,
                self.num_splits,
                causal=self.is_causal,
            )
            return return_vals
