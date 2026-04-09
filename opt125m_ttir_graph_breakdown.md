# OPT-125M vLLM TTIR Graph Breakdown

Analysis of the 15 TTIR graphs produced by `extract_mlir_graphs.py --type ttir` from an OPT-125M generation run through vLLM's torch.compile pipeline on TT hardware.

## How to Reproduce

```bash
# Run the test with debug logging to capture MLIR graphs
TTXLA_LOGGER_LEVEL=DEBUG pytest -svv tests/integrations/vllm_plugin/generative/test_opt_generation.py::test_opt_generation |& tee test_opt_generation_debug.log

# Extract TTIR graphs from the log
python ../scripts/extract_mlir_graphs.py --type ttir --subdir cool_test test_opt_generation_debug.log
```

## Model Info

- **Model**: facebook/opt-125m (12 transformer layers, hidden_dim=768, vocab=50272)
- **Framework**: vLLM V1 with torch.compile("tt") backend
- **Total**: 15 graphs, 4,849 operations, 10,509 lines of TTIR

## Graph Summary

| Graph | Lines | Ops | Purpose |
|-------|-------|-----|---------|
| 1 | 11 | 1 | Scalar constant init |
| 2 | 1792 | 841 | Initial prefill (no KV cache) |
| 3 | 1778 | 797 | First decode step (with KV cache) |
| 4-6 | 2121 | 987 | Subsequent decode steps (identical) |
| 7-10 | 33 | 12 | select_hidden_states gather (identical) |
| 11 | 26 | 7 | LM head (hidden states -> logits) |
| 12 | 53 | 22 | Logits processor / vocab mask |
| 13 | 248 | 128 | Full sampler (top-k/top-p) |
| 14 | 23 | 6 | Greedy argmax |
| 15 | 83 | 38 | Prompt logprobs |

## Detailed Breakdown

### Graph 1 — Scalar constant initialization

Returns 48 scalar `tensor<f32>` values, all `1.0`. This is a `SyncTensorsGraph` that materializes scalar constants (sampling parameters, scaling factors) before the model runs. No inputs. This is a torch.compile artifact — when vLLM first synchronizes tensors to device, these get traced as their own trivial graph.

### Graph 2 — Initial prefill (no KV cache)

The first prefill pass. Takes all model weights (embed_tokens, embed_positions, all 12 layers of QKV/out_proj/FC1/FC2/layernorm) plus input token IDs and position IDs. **No KV cache inputs.** This is a full forward pass: embedding, positional embedding, all 12 transformer layers computing Q/K/V from scratch, FFN, final layer norm. 37 `custom-call` ops (TTNN attention/matmul kernels).

### Graph 3 — First decode step (with KV cache)

Same model weights but now includes `kv_cache` tensors (one per layer) plus page_table and attention mask. Slightly fewer ops than graph 2 because it reads cached K/V via `paged_scaled_dot_product_attention_decode`. 61 `custom-call` ops (separate paged_fill + paged_decode per layer).

### Graphs 4, 5, 6 — Subsequent decode steps (identical)

Decode steps after the first. Larger than graph 3 because torch.compile captures additional `lifted_tensor` constants (per-layer attention scaling/causal mask metadata) that weren't present in graph 3's trace. All three are identical — torch.compile re-traced due to guard failures but converged on the same graph. After stabilization, vLLM reuses this cached compilation for all remaining decode steps.

### Graphs 7, 8, 9, 10 — select_hidden_states (identical)

The `logits_indices` gather — selecting the last token's hidden states from the padded output tensor:
- Input: `tensor<1xi32>` (logits index) + `tensor<1x1x768xbf16>` (hidden states)
- Output: `tensor<1x768xbf16>`
- Ops: reshape, typecast, compare (negative index handling), gather

Four identical copies from torch.compile re-tracing before caching.

### Graph 11 — LM head

Projects hidden states to vocabulary logits:
- Input: `lm_head_weight` (50304x768) + hidden states (1x768)
- Output: `tensor<1x50272xbf16>` (logits)
- Ops: reshape, transpose, dot_general (matmul), slice (trim padding 50304 -> 50272)

### Graph 12 — Logits processor / vocab mask

Logits post-processing with vocabulary masking:
- Input: logits (1x50272), bitmask (32xi32), allowed_token_ids (1x1571xi32), bool flag
- Decodes the bitmask via bitwise_and, masks forbidden tokens to `-inf` (0xFF80)
- Implements vLLM's `LogitsProcessor` for allowed_token_ids / bad_words constraints

### Graph 13 — Full sampler (top-k/top-p)

The main sampling graph (`topk_topp_sampler.forward_tpu`):
- Inputs: RNG seed (i64), temperature (1xf32), logits (1x50272xbf16), top_p (1xf32), top_k (1xi32), min_p (1xf32)
- Operations: temperature scaling, softmax, PRNG (linear congruential: `seed * 214013 + 2531011`), top-k sort, top-p cumulative probability filtering, categorical sampling
- Output: sampled token ID (1x1xi64)

### Graph 14 — Greedy argmax

Simple greedy decoding fallback — argmax over logits:
- Input: logits (1x50272xbf16)
- Output: token ID (1x1xi64)
- Ops: reshape, arange, argmax, typecast

### Graph 15 — Prompt logprobs

Log-probability computation for prompt tokens:
- Input: logits (1x50272xbf16) + selected token ID (1x1xi64)
- Output: top-21 log-probs + indices (1x21xf32, 1x21xi32) + count of tokens >= selected token's prob
- Ops: log_softmax, gather (selected token's logprob), sort (descending), slice (top 20), comparison counting

## Why Duplicate Graphs?

torch.compile traces a new graph whenever a guard fails (tensor shapes change, new code paths taken). During OPT generation:

- **Graph 2 vs 3**: Different code path — prefill (no KV cache) vs decode (with KV cache)
- **Graph 3 vs 4**: First decode didn't have `lifted_tensor` constants captured yet
- **Graphs 4=5=6**: Identical re-traces — torch stabilizes after a few steps
- **Graphs 7=8=9=10**: Same `select_hidden_states` re-trace pattern

After warmup, vLLM settles on reusing the cached compilations for the rest of generation.
