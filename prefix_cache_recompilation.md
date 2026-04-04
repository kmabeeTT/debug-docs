# Prefix Cache Recompilation Bug

## Problem

When prefix caching is enabled in vLLM on TT hardware, the first request that hits a prefix cache match triggers a full XLA graph recompilation (~24s TTFT penalty). Subsequent requests with the same offset are fast, but each new unique `prefill_block_offset` value causes another recompilation.

## Root Cause

File: `tt-xla/integrations/vllm_plugin/vllm_tt/attention.py` (~line 482)

```python
fill_page_table = attn_metadata.page_table
if attn_metadata.prefill_block_offset > 0:
    fill_page_table = attn_metadata.page_table[
        :, attn_metadata.prefill_block_offset :
    ]
```

Two problems:

1. **Branch mismatch**: During warmup, `prefill_block_offset` is always 0, so dynamo traces only the `False` branch. At runtime, a prefix cache hit produces `offset > 0`, tracing the `True` branch for the first time -- new graph, full recompilation.

2. **Shape change from slice**: `page_table[:, offset:]` produces a different tensor shape for each unique offset value (e.g., `(1, 128)` vs `(1, 127)` vs `(1, 126)`). Each new shape triggers another recompilation.

## How We Detected It

Added debug logging to `_verify_num_xla_graphs()` in `model_runner.py` (replaced assert with warning):

```
RECOMPILATION DETECTED during execute_model: 1 new graph(s) compiled (was 23, now 24).
Runtime shapes: {'input_ids': (1, 32), 'position_ids': (1, 32), 'num_reqs': 1,
'page_table': (1, 128), 'cache_position': (1,), 'num_computed_tokens_per_req': [32],
'num_tokens_per_req': [42]}
```

Key clue: `num_computed_tokens_per_req: [32]` -- prefix cache hit (32 tokens already computed). Warmup always uses `num_computed_tokens = 0`.

## Fix Options

### Option A: Always slice (remove the if branch)

```python
fill_page_table = attn_metadata.page_table[
    :, attn_metadata.prefill_block_offset :
]
```

**Pros:**
- Simple one-line change
- Eliminates the if/else branch recompilation (the worst one)

**Cons:**
- Still recompiles for each unique offset value (different slice shapes: 128, 127, 126...)
- In practice ~3-5 recompilations for typical usage, but not zero

### Option B: torch.roll outside compiled graph

```python
# Pre-roll page table so prefix blocks move to the end (outside compiled graph)
fill_page_table = torch.roll(
    attn_metadata.page_table, shifts=-attn_metadata.prefill_block_offset, dims=1
)
```

**Pros:**
- Zero recompilations -- tensor shape is always `(1, 128)` regardless of offset
- Prefix blocks rotate to the end where paged_fill_cache won't reach them
- Clean separation: shape-preserving transform outside the graph, compiled code sees constant shapes

**Cons:**
- Requires that `paged_fill_cache` only writes to blocks based on `fill_value` sequence length, not the full page_table. Must verify this assumption.
- `torch.roll` has a small CPU cost per call (negligible vs 24s recompilation)
- Slightly less obvious what the code is doing without comments

### Option C: Pass offset as tensor parameter to paged_fill_cache

Add an `offset` parameter to the `paged_fill_cache` custom op so it can skip blocks internally without changing the page_table shape.

**Pros:**
- Cleanest long-term solution
- No shape changes, no recompilation, semantically clear

**Cons:**
- Requires changing the custom op signature in `tt_torch/custom_ops.py`
- Requires updating the StableHLO lowering in C++
- Biggest engineering effort

### Option D: Precompile both branches during warmup

Run one extra dummy with `prefill_block_offset > 0` during `_precompile_backbone()`.

**Pros:**
- No changes to attention code

**Cons:**
- Only fixes the branch mismatch (problem 1), not the per-offset shape issue (problem 2)
- Still recompiles for each unique offset value
- Worse than Option A with more code

## Recommendation

**Option B (torch.roll)** is the best balance of correctness and simplicity. Zero recompilations, constant tensor shapes, minimal code change. Verify that `paged_fill_cache` respects `fill_value` size before deploying.

**Option C** is the right long-term fix if the op gets refactored.

## Reproduction

1. Start server with prefix caching enabled (default): `DEVICE=p150 MODEL=Llama-3.1-8B-Instruct ./launch_server.sh`
2. Send a request via `/v1/completions` (no prefix cache, warms the shape)
3. Send a request via `/v1/chat/completions` (chat template creates shared prefix, triggers prefix cache hit)
4. Second request triggers recompilation (~24s TTFT)

Enable detection: `VLLM_XLA_CHECK_RECOMPILATION=1`

## Related Files

- `tt-xla/integrations/vllm_plugin/vllm_tt/attention.py` -- the `if prefill_block_offset > 0` branch
- `tt-xla/integrations/vllm_plugin/vllm_tt/model_runner.py` -- `_precompile_backbone()`, `_verify_num_xla_graphs()`, `_prepare_inputs()`
- `tt-xla/python_package/tt_torch/custom_ops.py` -- `paged_fill_cache` op definition
