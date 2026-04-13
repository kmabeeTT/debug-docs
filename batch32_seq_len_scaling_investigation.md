# Batch 32 + High Seq Len Scaling Investigation (WIP)

## Goal

1. Get batch 32 working with the highest possible seq len
2. Understand what limits scaling on BH single chip (P150)

## Summary of Findings

Compilation OOM is the primary bottleneck. Two dimensions contribute:

1. **Token dimension (max_num_batched_tokens):** Capping this to `max_model_len` via chunked prefill avoids the largest bucket OOM. Implemented and tested.
2. **Batch dimension (max_num_reqs):** Even with token dimension capped at 4096, batch=32 still OOMs because precompilation creates tensors of shape `(32, 4096)` and the attention intermediate buffers scale with the batch dimension.

The limit is somewhere between batch=16 (works) and batch=32 (OOM) at len=4096 on a single P150 chip.

## Benchmark Results

All runs: Llama-3.1-8B-Instruct, single P150, device sampling, greedy (temp=0), bfp_bf8, consteval, opt_level=1. KV cache split fix (#4209) in place.

### Batch scaling at different configs

| Batch | gpu=0.05, len=128 | gpu=0.1, len=128 | gpu=0.1, len=4096 |
|---|---|---|---|
| 1 | 22.5 | 22.5 | 22.4 |
| 2 | 21.8 | 21.7 | 21.6 |
| 4 | 20.9 | 20.8 | 20.4 |
| 8 | 20.0 | 20.0 | 19.4 |
| 16 | 18.3 | 18.2 | 16.8 |
| 32 | 14.8 | 14.8 | OOM |

### Observations

- **gpu_memory_utilization 0.05 vs 0.1** at len=128: No perf difference. KV cache pool size doesn't affect decode speed (confirmed by fix #4209).
- **len=128 vs len=4096** at gpu=0.1: Slight degradation at higher batch sizes (batch 16: 18.3→16.8). Batch 1 essentially unchanged (22.5→22.4).
- **Batch 32 at len=4096**: OOM during compilation regardless of `max_num_batched_tokens` cap.
- **Batch 32 at len=128**: Works fine (14.8 tok/s) — smaller batch*len and smaller page tables.

## Compilation OOM Analysis

### Attempt 1: Uncapped max_num_batched_tokens (default)

`max_num_batched_tokens = 4096 * 32 = 131,072`

```
TT_FATAL: Out of Memory: Not enough space to allocate 3758096384 B DRAM buffer
across 8 banks, where each bank needs to store 469762048 B,
but bank size is 4273390016 B (allocated: 3078993536 B, free: 1194396480 B)
```

OOM compiling the 131K-token bucket.

### Attempt 2: Capped max_num_batched_tokens=4096 (chunked prefill)

`max_num_batched_tokens = 4096`, `compile_ranges_split_points: [4096]`

Required relaxing an assertion in `model_runner.py` (originally required `max_num_batched_tokens >= max_model_len * max_num_seqs`).

```
TT_FATAL: Out of Memory: Not enough space to allocate 3758096384 B DRAM buffer
across 8 banks, where each bank needs to store 469762048 B,
but bank size is 4273390016 B (allocated: 3078993536 B, free: 1194396480 B)
```

**Same OOM, same buffer size.** Token bucket is only 4096, but `max_num_reqs=32` means input tensors are `(32, 4096)` and page_table is `(32, 128)`. The attention intermediate buffers scale with the batch dimension independently of the token dimension.

### Key Insight

The compilation DRAM usage is a function of **both** dimensions:
- Token dimension: controlled by `max_num_batched_tokens` → capped via chunked prefill
- Batch dimension: controlled by `max_num_reqs` → **no workaround currently**

| Batch | Token dim | Tensor shape | Compiles? |
|---|---|---|---|
| 1 | 4096 | (1, 4096) | Yes |
| 16 | 4096 | (16, 4096) | Yes |
| 32 | 4096 | (32, 4096) | OOM |
| 32 | 128 | (32, 128) | Yes |
| 1 | 65536 | (1, 65536) | OOM |

The max compilable product `batch * max_model_len` is somewhere between 65K (batch=16*4096) and 131K (batch=32*4096) on a P150 chip with Llama-3.1-8B.

## Changes Made

### tt-inference-server-2: `vllm_settings.py`
- Changed `max_num_batched_tokens` from `max_model_length * max_num_seqs` to `max_model_length`
- This enables chunked prefill, reducing the largest compiled bucket

### tt-xla-2: `model_runner.py`
- Relaxed assertion from `max_num_batched_tokens >= max_model_len * max_num_seqs` to `max_num_batched_tokens >= max_model_len`
- Added comment explaining the chunked prefill motivation

### tt-xla-2: `vllm_benchmark.py`
- Added `max_num_batched_tokens` config field to `VLLMBenchmarkConfig`
- Defaults to `None` (old behavior: `batch_size * max_model_len`)

## KV Cache Decode Perf Regression (FIXED)

Previously, increasing `gpu_memory_utilization` caused proportional decode slowdown (11→2.5 tok/s). Fixed in [#4209](https://github.com/tenstorrent/tt-xla/pull/4209) by splitting K/V into separate tensors.

**TODO:** Re-run benchmarks at higher gpu_memory_utilization to confirm fix.

## Open Questions

- What is the exact max compilable `batch * len` product on P150? (between 65K and 131K)
- Can the compiler reduce intermediate DRAM usage for large batch dims? (tiling, streaming)
- Would batch=24 or batch=20 work at len=4096? (find the exact cutoff)
- Does TP (tensor parallel across 2+ chips) help by splitting the attention computation?

## Next Steps

- [ ] Find exact max batch at len=4096: try batch=20, 24, 28
- [ ] Re-run gpu=0.1 vs gpu=0.7 at len=4096, batch=16 with KV cache fix
- [ ] Test higher seq lens (8K, 16K) at batch=1 to find compilation limit
- [ ] Open issue to track "high batch + high seq len" compilation OOM

## Environment

- Machine: QB2 (P300X2, 4x BH chips, 32GB DRAM per chip)
- Container: Ubuntu 24.04, tt-xla-2
- SDPA decode grid fix in place (commit 88d894e46)
- KV cache split fix in place (#4209, commit 2f0d7e7e6)
