# CPU Sampling Optimization for vLLM on TT Hardware

## Context

Non-greedy sampling on TT device is slow (5 tok/s vs 22 greedy). CPU sampling (`cpu_sampling=True`) is the fallback, but has its own overhead at higher batch sizes. This investigation optimized the CPU sampling path to close the gap.

## Results Summary

Llama-3.1-8B-Instruct, single P150, cpu_sampling=True, enable_trace=True, bfp_bf8, consteval, opt_level=1.

### Batch 2 (demo config)

| Config | tok/s |
|---|---|
| Greedy device | ~21.7 |
| Non-greedy CPU (before optimization) | ~17* |
| Non-greedy CPU (after Gumbel-max) | 19.0 |
| Non-greedy device | ~5 |

*estimated from pre-optimization measurements

### Batch 8

| Config | tok/s |
|---|---|
| Greedy device | 20.0 |
| Non-greedy CPU (after Gumbel-max) | 15.1 |
| **Skip sampling entirely (dummy token)** | **19.7** |
| Non-greedy device | ~5 |

**Key insight**: At batch 8, the gap between skip-sampling (19.7) and CPU sampling (15.1) is 4.6 tok/s — CPU sampling overhead is ~23% of total time. At batch 2, the overhead is negligible (19.0 vs ~19.7 ceiling).

### Console vs client_demo.sh

Console sends `temperature=1.0`, client_demo.sh sends `None` (falls back to runner defaults `temp=0.6, rep_penalty=1.1`). Performance is similar — console slightly faster due to no penalty computation.

## Optimizations Applied

### Fix 1: Gumbel-max trick (SHIPPED)

Replaced `torch.softmax(logits) + torch.multinomial(probs)` with `torch.argmax(logits + gumbel_noise)`.

- Mathematically equivalent for categorical sampling
- `torch.argmax` is ~4x faster than `torch.multinomial` at vocab=128K
- Works as a single batched op (doesn't scale linearly with batch like multinomial)
- Commit: `44ae22b1f`

```python
# Before:
probs = torch.softmax(logits, dim=-1)
random = torch.multinomial(probs, num_samples=1).squeeze(-1)

# After:
gumbel = -torch.log(-torch.log(torch.rand_like(logits.float()) + 1e-20) + 1e-20)
random = torch.argmax(logits + gumbel, dim=-1)
```

### Fix 2: Batched top-k fast path (SHIPPED)

When all requests share the same `top_k > 0`, single batched `torch.topk` call reduces vocab from 128K to k elements before top-p and Gumbel-max. Eliminates per-request for loops.

Only activates when `top_k > 0` is set in the request. With current demo settings (`top_k=0`), this path isn't used.

### Fix 3: CPU top-128 after transfer (SHIPPED)

After transferring full logits to CPU, immediately reduce to top-128 elements via `torch.topk`. All subsequent operations (penalties, temperature, top-p, Gumbel-max) run on 128 elements instead of 128K.

No measurable improvement at batch 2 (transfer is the bottleneck, not CPU ops).

## Optimizations Attempted and Reverted

### Micro-optimization: exponential_() for Gumbel noise

Replaced `−log(−log(rand + ε) + ε)` with `torch.empty().exponential_().log()`. Was actually **slower** — caused 19→18.1 regression. Reverted.

### Micro-optimization: non_blocking=True transfer

`logits.to(device="cpu", non_blocking=True)` — no measurable improvement, possibly hurt. The next line accesses the tensor immediately, forcing a sync.

### Micro-optimization: skip temp=1.0 division

`torch.allclose` check to skip the division was slower than just dividing. Reverted.

### Float16 transfer

`logits.half().cpu().float()` — halves transfer bandwidth. No measurable improvement at batch 2 (transfer is small). Not tested at batch 8 independently.

## Attempted but Failed

### Device-side topk (would eliminate transfer)

Run `torch.topk(logits, 128)` on device, transfer only 128 values + indices to CPU. Would cut transfer from ~2MB to ~2KB at batch 8.

**Failed**: TT runtime crashes with `Tensor data type must be the alias of the unsupported data type: Int32` — the topk indices are int32 and the TT PJRT memcpy doesn't support that dtype. Converting to int64 before transfer didn't help (the int32 is generated inside the XLA op).

This is the most promising lead — if the int32 memcpy issue is fixed in TT runtime, device-side topk would recover most of the 4.6 tok/s gap at batch 8.

### Skip sampling hack (diagnostic only)

Return dummy token without any transfer or computation. Measured 19.7 tok/s at batch 8 — confirms the forward pass ceiling and quantifies CPU sampling overhead at 4.6 tok/s.

## Bottleneck Analysis

| Source | Cost at batch 8 |
|---|---|
| Model forward pass | ~80% (19.7 tok/s ceiling) |
| Device→CPU logits transfer | ~15% (128K × batch × 4 bytes per step) |
| CPU sampling ops (Gumbel, topk, etc.) | ~5% (negligible after optimizations) |

The transfer (`logits.cpu()`) is the remaining bottleneck for CPU sampling. The actual sampling math is fast after the Gumbel-max optimization.

## Open Leads

1. **Fix TT runtime int32 memcpy** → enables device-side topk → eliminates transfer bottleneck
2. **Device-side Gumbel sampling** — compile `argmax(logits + gumbel)` as a single device op, transfer only 1 token per request. Needs gumbel noise generation on device.
3. **Improve device non-greedy sampling** — currently 5 tok/s, ongoing work by tt-xla team. If this reaches 15+ tok/s, CPU sampling becomes unnecessary.

## Files Changed

- `integrations/vllm_plugin/vllm_tt/model_runner.py` — `sample_from_logits_cpu` method (~line 2190)
- `tt-inference-server-2/tt-media-server/tt_model_runners/vllm_runner.py` — `SAMPLING_DEFAULTS`, `enable_trace`
- `tt-inference-server-2/tt-media-server/open_ai_api/chat.py` — request param logging

## Environment

- Machine: QB2 (P300X2), single P150 chip
- Model: Llama-3.1-8B-Instruct, bfp_bf8, consteval, opt_level=1, enable_trace=True
- Branch: `kmabee/vllm_perf_apr12` on tt-xla-2
- Container: Ubuntu 24.04
