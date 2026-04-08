# vLLM TT Shape Precompilation and Runtime Bucketing

## Overview

On TT hardware, the XLA/PJRT compiler needs to compile a separate graph for each unique tensor shape. To avoid expensive runtime compilation (~15-25s per shape), vLLM TT precompiles graphs for a set of "bucketed" shapes during server warmup. At runtime, each request's actual token count is padded up to the nearest precompiled bucket size.

## Bucket Generation

**File:** `tt-xla/integrations/vllm_plugin/vllm_tt/model_runner.py`

### How buckets are chosen

Buckets are generated as powers of 2, starting from `min_context_len` up to `max_model_len`:

```python
def _get_token_paddings(min_token_size, max_token_size):
    num = _adjust_min_token(min_token_size)  # round up to power of 2, minimum 32
    paddings = [1]  # always include 1 for single-token decode
    while True:
        paddings.append(num)
        if num >= max_token_size:
            break
        num *= 2
    return paddings
```

**Example** with `min_context_len=32`, `max_model_len=4096`:

```
Buckets: [1, 32, 64, 128, 256, 512, 1024, 2048, 4096]
```

- `1` — for decode steps (generating one token at a time)
- `32` through `4096` — for prefill (processing the input prompt)

### Configuration

- `min_context_len`: Starting bucket size (default 128, often set to 32 via `additional_config`)
- `max_model_len`: Maximum sequence length the model supports (e.g., 4096)
- Both set in the runner's `additional_config` dict passed to `AsyncEngineArgs`

## Precompilation (Warmup)

During `capture_model()`, the server compiles graphs for every bucket size:

```python
def _precompile_backbone(self):
    for num_tokens in self.num_tokens_paddings:  # [1, 32, 64, 128, ...]
        self._dummy_run(num_tokens, num_reqs, num_blocks)
```

`_dummy_run` creates zero-filled tensors at the target shape and runs the model forward pass, forcing XLA to compile and cache the graph:

```python
def _dummy_run(self, num_tokens, num_reqs, num_blocks):
    input_ids = torch.zeros((max_num_reqs, num_tokens), dtype=torch.int32)
    position_ids = torch.zeros((max_num_reqs, num_tokens), dtype=torch.int32)
    page_table = torch.zeros((num_reqs, num_blocks), dtype=torch.int32)
    # ... run model forward pass, XLA compiles and caches the graph
```

### What gets precompiled

`capture_model()` precompiles several subgraphs:

1. **Backbone** — main model forward pass (attention + FFN), all bucket sizes
2. **Select hidden states** — indexed extraction from hidden states
3. **Compute logits** — final projection to vocabulary
4. **Structured decoding** — grammar/constraint processing
5. **Sample from logits** — device-side sampling (greedy + non-greedy variants)
6. **Gather logprobs** — log probability extraction

Total warmup time is typically 3-7 minutes depending on model size and number of buckets.

## Runtime Bucketing

When a request arrives, the actual token count is padded up to the nearest bucket:

```python
def _get_padded_token_len(paddings, x):
    """Return the first element in paddings >= x."""
    index = bisect.bisect_left(paddings, x)
    return paddings[index]
```

**Example:** Request with 42 tokens → padded to bucket 64.

This happens in `_prepare_inputs()`:

```python
max_num_scheduled_tokens = max(num_scheduled_tokens_per_req)
padded_total = _get_padded_token_len(self.num_tokens_paddings, max_num_scheduled_tokens)

# All tensors use the padded size
input_ids = torch.zeros((max_num_reqs, padded_total), ...)
position_ids = torch.zeros((max_num_reqs, padded_total), ...)
```

Since `(max_num_reqs, 64)` was already precompiled during warmup, XLA finds the cached graph and executes it immediately — no compilation needed.

## When Recompilation Happens

Recompilation occurs when the runtime produces a tensor shape that wasn't precompiled. Known causes:

### 1. Prefix cache offset changing page_table shape

When prefix caching hits, `prefill_block_offset > 0` causes a `page_table[:, offset:]` slice that produces a shape not seen during warmup (e.g., `(1, 127)` instead of `(1, 128)`).

**Fix:** `torch.roll` the page_table outside the compiled graph to keep shape constant. See `prefix_cache_recompilation.md`.

### 2. Batch size mismatch

Warmup compiles with `max_num_reqs` (e.g., 4 for batch=4). If runtime has `num_reqs=1`, special handling (`cache_position[1:] = -1`) could trace a different graph.

### 3. Token count exceeding max bucket

If `max_num_batched_tokens` is set higher than `max_model_len` (e.g., `4096 * 4 = 16384` for batch=4), the largest bucket might not be precompiled.

## Detecting Recompilation

Set `VLLM_XLA_CHECK_RECOMPILATION=1` when launching the server. The `_verify_num_xla_graphs()` method checks if new graphs were compiled after warmup. With our debug logging patch, it prints the runtime shapes that triggered the recompile.

The "Failed to deserialize executable" warning in logs means XLA tried to load a cached graph from disk (not supported on TT PJRT), then compiled from scratch. This appears during both warmup (expected) and runtime recompilation (unexpected).

## Key Tradeoffs

| More buckets (smaller `min_context_len`) | Fewer buckets (larger `min_context_len`) |
|---|---|
| Less wasted padding per request | More wasted padding |
| Longer warmup time | Shorter warmup |
| More memory for cached graphs | Less memory |
| Better latency for small prompts | Small prompts padded to large bucket |

## Relevant Files

| File | What it does |
|---|---|
| `model_runner.py:_get_token_paddings()` | Generates bucket list |
| `model_runner.py:_get_padded_token_len()` | Maps token count to nearest bucket |
| `model_runner.py:_precompile_backbone()` | Compiles graphs for all buckets |
| `model_runner.py:_dummy_run()` | Runs model with dummy tensors to trigger compilation |
| `model_runner.py:_prepare_inputs()` | Pads runtime inputs to bucket size |
| `model_runner.py:capture_model()` | Entry point for all precompilation |
| `model_runner.py:_verify_num_xla_graphs()` | Checks for unexpected recompilation |
| `platform.py:TTConfig` | `min_context_len` and other config |
