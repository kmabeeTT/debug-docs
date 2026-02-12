# VLLM TP Sampling Graph Comparison: Greedy vs Non-Greedy on TT Device

**Date**: 2026-02-12
**Model**: meta-llama/Llama-3.2-3B (vocab_size=128256, 2-chip TP)
**Test**: `test_tensor_parallel_generation.py::test_tensor_parallel_generation_n300[meta-llama/Llama-3.2-3B]`
**Context**: Verifying that non-greedy sampling compiles and runs on TT device even in tensor parallel mode, where `sample_from_logits` is NOT wrapped in `torch.compile(backend="tt")`

## Background: The TP Sampling Concern

In `model_runner.py`, the sampling function is only compiled for non-TP mode:

```python
if not self.enable_tensor_parallel:
    self.sample_from_logits_func = torch.compile(
        self.sample_from_logits,
        backend="tt",
        fullgraph=True,
        dynamic=False,
    )
else:
    self.sample_from_logits_func = self.sample_from_logits  # raw function!
```

The concern: without `torch.compile`, would sampling operations fall back to CPU instead of running on TT device?

**Answer: No, sampling still runs on TT device.** XLA lazy tracing captures all operations on XLA tensors regardless of `torch.compile`. The graphs get compiled to TTIR either way.

## Test Procedure

1. **Greedy run**: With `if sampling_metadata.all_greedy or True:` (forces greedy)
2. **Non-greedy run**: With `or True` removed, test uses `temperature=0.8, top_p=0.95`

```bash
# Greedy
TTXLA_LOGGER_LEVEL=DEBUG pytest -svv tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py::test_tensor_parallel_generation_n300[meta-llama/Llama-3.2-3B]

# Extract graphs
python3 ../scripts/extract_mlir_graphs.py llama3p2_3b_greedy.log --subdir test_llama_tp_greedy
# 13 graphs, 10,861 ops

python3 ../scripts/extract_mlir_graphs.py llama3p2_3b_non_greedy.log --subdir test_llama_tp_non_greedy
# 13 graphs, 10,986 ops
```

## Side-by-Side Comparison

| Component | Greedy TP (13 graphs) | Non-greedy TP (13 graphs) |
|---|---|---|
| Graphs 1-12 | **Identical** | **Identical** |
| **Sampling graph** | Graph 13: `SyncTensorsGraph.37` — 26 lines, 9 ops | Graph 13: `SyncTensorsGraph.300` — 263 lines, 134 ops |

### Model Forward Pass Graphs (identical in both runs)

| Graph | Module Name | Lines | Ops | Role |
|-------|-------------|-------|-----|------|
| 1 | (large) | 6653 | 3413 | Model forward (prefill/decode variant) |
| 2 | (large) | 6909 | 3524 | Model forward (decode variant) |
| 3 | (large) | 7527 | 3864 | Model forward (largest variant) |
| 4-5 | (small) | 33 | 12 | Small utility graphs |
| 6 | (small) | 27 | 9 | Small utility |
| 7-9 | (tiny) | 15 | 2 | Tiny ops |
| 10 | (tiny) | 13 | 1 | Single op |
| 11-12 | (small) | 22-25 | 5-6 | Small utility |

## Sampling Graph Details

### Greedy TP (`SyncTensorsGraph.37`, 9 ops)

```
Input:  tensor<1x128256xbf16> (logits, presharded across 2 devices)
Output: tensor<1x1xi64>       (token ID)

Pipeline:
  mesh_shard (full_to_shard) → reshape → all_gather → reshape →
  arange → argmax → typecast → reshape → mesh_shard (shard_to_full)
```

Same as the single-device greedy graph but with `ttir.mesh_shard` and `ttir.all_gather` bookending for TP.

### Non-greedy TP (`SyncTensorsGraph.300`, 134 ops)

```
Inputs:
  - tensor<i64>            (RNG state, presharded)
  - tensor<1xf32>          (top_p, presharded)
  - tensor<1xf32>          (temperature, presharded)
  - tensor<1x128256xbf16>  (logits, presharded across 2 devices)
  - tensor<1xf32>          (min_p, presharded)
  - tensor<1xi32>          (top_k, presharded)

Output: tensor<1x1xi64>    (token ID)
```

**Pipeline stages (all on TT device):**

1. **TP mesh ops**: `mesh_shard` (full_to_shard) on all inputs, `all_gather` on sharded logits
2. **Temperature scaling**: `logits / temperature`
3. **Softmax**: numerically stable (max, subtract, exp, sum, div)
4. **Min-p filtering**: threshold = `min_p * max(probs)`, mask with `-inf`
5. **Re-softmax** after filtering
6. **Sort + cumsum** for top-p nucleus sampling (`ttir.sort` ascending, `ttir.cumsum`)
7. **CPU fallback**: `dynamic_update_slice` for top-p mask index update (hoisted to CPU)
8. **Top-p mask application**: sum sorted probs, mask beyond threshold
9. **Top-k filtering**: gather-based k-th value lookup, mask below threshold
10. **Combined filtering**: merge masks, apply to logits
11. **Final softmax** on filtered logits
12. **Gumbel noise**: `ttir.rand` → bit manipulation → log transform
13. **Token selection**: `probs / gumbel_noise` → `ttir.argmax`
14. **Greedy fallback**: `ttir.where(temperature < epsilon, greedy_argmax, sampled_token)`
15. **TP output**: `mesh_shard` (shard_to_full) on result

### CPU Fallback Note

One operation is hoisted to CPU: `stablehlo.dynamic_update_slice` (line 105 in TTIR). This is the top-p mask update where the last index of the cumsum mask is set. The compiler generates a `ttcore.cpu_module` with LLVM IR for this operation. This is a small scalar operation and does not significantly impact performance.

## Key Findings

1. **Sampling runs on TT device in TP mode** despite `sample_from_logits` not being `torch.compile`'d. XLA lazy tracing captures the operations automatically.

2. **TP adds mesh_shard/all_gather overhead**: The non-greedy TP graph has 134 ops vs 122 for single-device (OPT). The extra ops are `mesh_shard` and `all_gather` for tensor parallel communication.

3. **One CPU fallback**: `dynamic_update_slice` is hoisted to CPU. This doesn't appear in the single-device OPT graphs (which used a different lowering path for the same logic).

4. **The `torch.compile` guard in model_runner.py is about eager compilation, not about whether ops run on device.** Without `torch.compile`, XLA still traces lazily. The difference is:
   - With `torch.compile`: graphs are eagerly traced and compiled during warmup
   - Without `torch.compile`: graphs are lazily traced on first execution (may cause a one-time delay)

## Comparison with Single-Device (OPT) Results

| Aspect | Single-device (OPT) | TP (Llama-3.2-3B) |
|--------|---------------------|---------------------|
| Greedy sampling ops | 6 | 9 (+ mesh_shard, all_gather) |
| Non-greedy sampling ops | 122 | 134 (+ mesh_shard, all_gather, cpu_hoisted) |
| Vocab size | 50272 | 128256 |
| torch.compile wrapping | Yes | No (lazy XLA tracing) |
| Sampling on TT device | Yes | Yes |

## Related Documents

- [VLLM Sampling Graph Comparison (Single-Device OPT)](vllm_sampling_graph_comparison.md)
