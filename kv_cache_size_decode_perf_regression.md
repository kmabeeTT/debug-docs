# KV Cache Size Causes Decode Performance Regression

## Summary

Increasing `gpu_memory_utilization` (which allocates more KV cache blocks) causes a proportional slowdown in decode tok/s, even when actual sequence length is short. This is a bug — unused allocated blocks should not affect decode speed.

## Evidence

All runs: Llama-3.1-8B-Instruct, single P150 chip, `max_seq_len=4096`, same prompt (~40 tokens), same `additional_config`. Only `gpu_memory_utilization` changed.

| gpu_memory_utilization | KV cache tokens | Decode tok/s | Relative |
|---|---|---|---|
| 0.1 (default) | 9,824 | 11.0 | 1.0x |
| 0.3 | ~29K | 5.6 | 0.5x |
| 0.9 | 88,448 | 2.5 | 0.23x |

Decode speed is inversely proportional to total KV cache pool size, regardless of actual usage.

## Reproduction

Observed via tt-inference-server on QB2, but should reproduce with any model using the vLLM TT plugin standalone.

### Via tt-inference-server (original observation)

```bash
# Fast (default)
DEVICE=p150 MODEL=Llama-3.1-8B-Instruct PORT=8002 ./launch_server.sh

# Slow
GPU_MEMORY_UTILIZATION=0.9 DEVICE=p150 MODEL=Llama-3.1-8B-Instruct PORT=8002 ./launch_server.sh
```

Send a short prompt via `client_demo.sh` and compare tok/s.

### Standalone vLLM repro (recommended for debugging)

Should reproduce with a smaller model (faster iteration). Suggested approach:

```python
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
import time, asyncio

model = "meta-llama/Llama-3.2-1B"  # or "facebook/opt-125m"

async def bench(gpu_mem_util):
    engine_args = AsyncEngineArgs(
        model=model,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        max_num_seqs=1,
        gpu_memory_utilization=gpu_mem_util,
        enable_chunked_prefill=False,
        additional_config={
            "enable_const_eval": True,
            "min_context_len": 32,
            "experimental_weight_dtype": "bfp_bf8",
            "cpu_sampling": True,
            "optimization_level": 1,
        },
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Warmup
    params = SamplingParams(temperature=0.0, max_tokens=10)
    async for _ in engine.generate("warmup", params, "warmup"):
        pass

    # Benchmark: short prompt, measure decode tok/s
    params = SamplingParams(temperature=0.0, max_tokens=128)
    start = time.perf_counter()
    token_count = 0
    async for output in engine.generate("Tell me a short poem", params, "bench"):
        token_count = len(output.outputs[0].token_ids)
    elapsed = time.perf_counter() - start
    print(f"gpu_mem={gpu_mem_util}: {token_count} tokens in {elapsed:.2f}s = {token_count/elapsed:.1f} tok/s")

# Run at different utilization levels
for util in [0.1, 0.3, 0.5, 0.9]:
    asyncio.run(bench(util))
```

**Note:** Script is untested — may need adjustments for TT device setup (TT_MESH_GRAPH_DESC_PATH, etc.). The key is varying `gpu_memory_utilization` while keeping everything else constant. Restart the engine between runs (KV cache size is set at init).

## Expected Behavior

Decode speed should be independent of total KV cache pool size. Paged attention should only access blocks that are actively in use for the current request, not iterate over the entire allocated pool.

## Impact

- Cannot increase `gpu_memory_utilization` for longer context lengths (8K, 16K, 64K, 128K) without crippling decode performance
- Multi-user batch deployments need more KV cache but performance degrades when it's allocated
- Stuck at `gpu_memory_utilization=0.1` (~9.8K tokens) for acceptable performance

## Likely Cause

The TT paged attention implementation or KV cache kernel likely scales with the total number of allocated blocks rather than just the active blocks for the current request. Possible locations:

- `paged_fill_cache` in `tt_torch/custom_ops.py` — cache write operation
- `paged_attention` kernel in tt-metal/TTNN — cache read during attention
- Page table tensor dimensions compiled into the XLA graph
- KV cache tensor size itself — if the full cache tensor (not just active pages) is passed through the compiled graph, larger allocations mean more data movement per step

## Environment

- Machine: QB2 (P300X2, 4x BH chips)
- Container: Ubuntu 24.04, tt-xla-2
- tt-inference-server-2 rebased on latest dev
- Model: Llama-3.1-8B-Instruct (bfp_bf8 weights)
- Qwen3-8B at default `gpu_memory_utilization=0.1` runs at 10.5 tok/s (not tested at higher utilization)
