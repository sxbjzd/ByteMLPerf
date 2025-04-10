# micro perf 操作说明

## 环境准备：
- sdk版本： 由天数智芯工程师提供

## 测试方法：

### op test:
```
cd ByteMLPerf/byte_micro_perf
```
* gemm
```
python3 launch.py --hardware_type ILUVATAR --device all --task gemm
```
* flash_attention
```
cp ByteMLPerf/byte_micro_perf/backends/ILUVATAR/workloads/flash_attention.json ByteMLPerf/byte_micro_perf/workloads/llm/
python3 launch.py --hardware_type ILUVATAR --device all --task flash_attention --task_dir ./workloads/llm
```
* all_reduce
```
python3 launch.py --hardware_type ILUVATAR --device 0,1 --task all_reduce
```

### fma test:
```
cd ByteMLPerf/byte_micro_perf/backends/ILUVATAR/InstructionThroughputTest
```
* build && install:
```
cd bytemlperf/byte_micro_perf/backends/ILUVATAR/InstructionThroughputTest/scripts
bash build_package.sh
bash install_package.sh  
```
* run test:
```
cd bytemlperf/byte_micro_perf/backends/ILUVATAR/InstructionThroughputTest
python3 -m unittest test.py
```
### llm test:
#### dpsk w8a8
* vllm openai server
```
export VLLM_PP_LAYER_PARTITION="9,7,7,7,7,8,8,8"
export VLLM_W8A8_MOE_USE_M4A8=1
export VLLM_MLA_DISABLE=0
python3 -m vllm.entrypoints.openai.api_server --model DeepSeek-R1-int4-pack8  --pipeline-parallel-size 8  --tensor-parallel-size 2 --trust-remote-code --max-model-len 8192 --gpu_memory_utilization 0.95 --port 12345
```
* benchmark
```
cd bytemlperf/byte_micro_perf/backends/ILUVATAR/llmBench
python3 benchmark_serving.py --model DeepSeek-R1-int4-pack8 --host 127.0.0.1 --port 12345 --num-prompts 1 --input-tokens 2048 --output-tokens 8
```

#### llama2-7b w8a8
* vllm openai server
```
python3 -m vllm.entrypoints.openai.api_server --model Llama-2-7b-chat-quantized.w8a8  --tensor-parallel-size 2 --trust-remote-code --max-model-len 4096 --gpu_memory_utilization 0.95 --port 12345
```
* benchmark
```
cd bytemlperf/byte_micro_perf/backends/ILUVATAR/llmBench
python3 benchmark_serving.py --model Llama-2-7b-chat-quantized.w8a8 --host 127.0.0.1 --port 12345 --num-prompts 1 --input-tokens 2048 --output-tokens 8
```