# VLLM Sampling Graph Comparison: Greedy vs Non-Greedy on TT Device

**Date**: 2026-02-12
**Model**: OPT (facebook/opt-350m, vocab_size=50272)
**Test**: `test_opt_generation.py` with SamplingParams variations
**Context**: Confirming that both greedy and non-greedy sampling paths compile and run on TT device via TTIR

## Summary

Both greedy and non-greedy sampling are compiled to TTIR and executed on the TT device. The XLA compiler generates different sampling graphs depending on the sampling parameters:

- **Greedy**: Simple 6-op `argmax` graph
- **Non-greedy**: Full 122-op sampling pipeline with temperature, top-k, top-p, min-p, and Gumbel noise

## Graph Extraction

Graphs were extracted from debug logs using `extract_mlir_graphs.py`:

```bash
# Non-greedy run
python3 ../scripts/extract_mlir_graphs.py test_opt_generation_feb12_sampling_params_debug.log --subdir test_opt_non_greedy
# Result: 14 TTIR graphs, 2,802 ops, 6,139 lines

# Greedy run
python3 ../scripts/extract_mlir_graphs.py test_opt_generation_feb12_greedy_debug.log --subdir test_opt_greedy
# Result: 13 TTIR graphs, 2,680 ops, 5,900 lines
```

## Side-by-Side Comparison

| Component | Non-greedy (14 graphs) | Greedy (13 graphs) |
|---|---|---|
| Graphs 1-12 | **Identical** module names, line/op counts | **Identical** |
| **Sampling graph** | Graph 13: `SyncTensorsGraph.318` - 239 lines, 122 ops | Graph 13: `SyncTensorsGraph.37` - 23 lines, 6 ops |
| Extra graph | Graph 14: `SyncTensorsGraph.37` - 23 lines, 6 ops | *(not present)* |

### Model Forward Pass Graphs (identical in both runs)

| Graph | Module Name | Lines | Ops | Role |
|-------|-------------|-------|-----|------|
| 1 | `SyncTensorsGraph.3477` | 1792 | 841 | Model forward (prefill/decode variant) |
| 2 | `SyncTensorsGraph.3665` | 1767 | 797 | Model forward (decode variant) |
| 3 | `SyncTensorsGraph.3757` | 2121 | 987 | Model forward (largest variant) |
| 4-5 | `SyncTensorsGraph.31` | 33 | 12 | Small utility graphs |
| 6 | `SyncTensorsGraph.13` | 26 | 7 | Small utility |
| 7-9 | `SyncTensorsGraph.6` | 15 | 2 | Tiny ops |
| 10 | `SyncTensorsGraph.5` | 13 | 1 | Single op |
| 11-12 | `SyncTensorsGraph.14` | 22-25 | 5-6 | Small utility |

## Sampling Graph Details

### Greedy Sampling Graph (`SyncTensorsGraph.37`, 6 ops)

A bare `argmax` on the raw logits:

```
Input:  tensor<1x50272xbf16>  (logits)
Output: tensor<1x1xi64>       (token ID)

Pipeline: reshape -> reshape -> arange -> argmax -> typecast -> reshape
```

No temperature scaling, no top-k/top-p filtering, no randomness. Just picks the highest-probability token directly.

### Non-Greedy Sampling Graph (`SyncTensorsGraph.318`, 122 ops)

Full `forward_tpu` sampling pipeline:

```
Inputs:
  - tensor<i64>            (RNG state)
  - tensor<1xf32>          (temperature)
  - tensor<1x50272xbf16>   (logits)
  - tensor<1xf32>          (top_p)
  - tensor<1xi32>          (top_k)
  - tensor<1xf32>          (min_p)

Output: tensor<1x1xi64>    (token ID)
```

**Pipeline stages:**

1. **Temperature scaling**: `logits / temperature` (div by broadcast temperature)
2. **Softmax**: max -> subtract -> exp -> sum -> div (numerically stable softmax)
3. **Top-p (min_p) filtering**: compute probability threshold from `min_p * max(probs)`, mask tokens below threshold with `-inf`
4. **Re-softmax** after top-p masking
5. **Sort + cumsum** for top-p nucleus sampling (`ttir.sort` ascending, `ttir.cumsum`)
6. **Top-k filtering**: gather-based lookup to find k-th value threshold, mask tokens below
7. **Combined filtering**: merge top-k and top-p masks, apply to logits
8. **Final softmax** on filtered logits
9. **Gumbel noise sampling**: `ttir.rand` -> bit manipulation -> log transform to generate Gumbel noise
10. **Token selection**: `probs / gumbel_noise` -> `ttir.argmax`
11. **Greedy fallback**: `ttir.where(temperature < epsilon, greedy_argmax, sampled_token)` - falls back to greedy if temperature is near zero

### Key Observation

The non-greedy run also compiles the simple argmax graph (graph 14, `SyncTensorsGraph.37`) - the same one used by greedy. This means XLA compiled **both** sampling paths. The `forward_tpu` code contains a `where` branch that selects between greedy argmax and sampled output based on temperature, so the simple argmax graph may be used during warmup or for sequences that happen to use greedy within a non-greedy batch.

## How to Identify the Sampling Graph

When looking at extracted TTIR graphs, identify the sampling graph by:

1. **Input signature**: Takes `tensor<1xNxbf16>` where N is vocab_size (e.g., 50272 for OPT), plus scalar sampling parameter tensors
2. **Output**: Returns `tensor<1x1xi64>` (single token ID)
3. **Size**: Medium-sized (100-200 ops for non-greedy, ~6 ops for greedy) - much smaller than model forward passes (800+ ops) but larger than utility graphs
4. **Key operations**: Look for `ttir.argmax` on a vocab-sized tensor. For non-greedy, also `ttir.sort`, `ttir.cumsum`, `ttir.rand`

## Related Issues

- **XLA sort+scatter corruption**: `ttir.sort` in the non-greedy path can return incorrect indices for large tensors (vocab_size=50272), causing the `forward_native` path to produce garbage. The `forward_tpu` path (used here) avoids the scatter-back and works correctly. See `sampler_debug_feb9/` artifacts.
- **ttnn.sort type mismatch**: Sort op requires BFLOAT16 or UINT16 input but may receive FLOAT32. Workaround pass needed in TTNNWorkaroundsPass.
