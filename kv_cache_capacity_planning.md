# KV Cache Capacity Planning

## Overview

The KV cache stores computed key/value tensors for all active requests. Its size determines how many concurrent users can be served and how much conversation context each can maintain. On TT hardware, this is controlled by `gpu_memory_utilization` in vLLM settings.

## Key Formula

```
KV cache tokens = (device_memory * gpu_memory_utilization - model_weights) / per_token_kv_bytes

per_token_kv_bytes = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
                     ^K+V

max_concurrent_users = KV_cache_tokens / max_seq_len
```

## Observed Values at gpu_memory_utilization=0.1

From server logs (`GPU KV cache size` and `Maximum concurrency` lines):

| Model | KV cache tokens | Max concurrency @ 4096 seq | Safetensor shards |
|---|---|---|---|
| Llama-3.2-1B | 39,296 | 9.59x | 1 |
| Llama-3.2-3B | 11,232 | 2.74x | 2 |
| Llama-3.1-8B | 9,824 | 2.40x | 4 |
| Qwen3-8B | 8,736 | 2.13x | 5 |

Larger models leave less memory for KV cache at the same utilization setting.

## Scaling with gpu_memory_utilization

The relationship is roughly linear (more utilization = more cache, less headroom for other allocations):

**Example: Llama-3.1-8B, target 16 concurrent users**

| gpu_memory_utilization | ~KV cache tokens | Max users @ 4096 context | Max users @ 512 avg context |
|---|---|---|---|
| 0.1 | 9,824 | 2.4 | 19 |
| 0.3 | ~29,472 | 7.2 | 57 |
| 0.5 | ~49,120 | 12.0 | 96 |
| 0.7 | ~68,768 | 16.8 | 134 |
| 0.9 | ~88,416 | 21.6 | 172 |

## What Happens When KV Cache Is Full

- New requests must wait for existing requests to complete and free their blocks
- With prefix caching enabled, least-recently-used cached prefixes get evicted
- Evicted prefixes must be re-prefilled on the next request (slower TTFT)
- Requests are NOT rejected — they queue until space is available

## Prefix Cache Sharing Reduces Effective Usage

With prefix caching enabled, users sharing the same token prefix (e.g., same chat template, same system prompt) share KV cache blocks:

- First user: pays full prefill, blocks cached
- Subsequent users with same prefix: reuse cached blocks, only prefill unique suffix
- This means 16 users don't necessarily need 16x the cache if they share prefixes

## Where to Change gpu_memory_utilization

File: `tt-inference-server/tt-media-server/config/vllm_settings.py`

```python
class VLLMSettings(BaseModel):
    gpu_memory_utilization: float = 0.1  # default is very conservative
```

Can also be overridden per model in `ModelConfigs` in `config/constants.py` via the `"vllm"` dict.

## Relevant Log Lines

When the server starts, each worker logs:

```
(EngineCore_DP0 pid=XXXXX) INFO ... GPU KV cache size: N tokens
(EngineCore_DP0 pid=XXXXX) INFO ... Maximum concurrency for M tokens per request: X.XXx
```

These are the key lines to check when adjusting capacity.

## Recommendations

- `gpu_memory_utilization=0.1` is only suitable for development/testing with 1-2 concurrent users
- For production with batch_size=4: use at least `0.3` (gives ~7 users at full context)
- For production with batch_size=16: use at least `0.7`
- Monitor actual context lengths in production — most chat requests use far less than max_seq_len, so real capacity is higher than the worst-case formula suggests
