# CPU Sampling Optimization for vLLM on TT Hardware

## Context

Non-greedy sampling on TT device is slow (5 tok/s vs 22 greedy). CPU sampling (`cpu_sampling=True`) is the fallback, but has its own overhead at higher batch sizes. This investigation optimized the CPU sampling path to close the gap.

## Results Summary

Llama-3.1-8B-Instruct, single P150, cpu_sampling=True, enable_trace=True, bfp_bf8, consteval, opt_level=1.

### Batch 1

| Config | tok/s | Notes |
|---|---|---|
| Skip sampling (ceiling) | 22.5 | Known garbage output at batch 1 — pre-existing bug |

Note: batch 1 has a pre-existing coherence issue (garbage output). Only skip-sampling ceiling was measured.

### Batch 2 (demo config)

| Config | tok/s | Notes |
|---|---|---|
| Skip sampling (ceiling) | 21.7 | |
| Greedy device | ~21.7 | |
| Non-greedy CPU (before optimization) | ~17* | Estimated |
| **Non-greedy CPU (after Gumbel-max)** | **19.0** | **Current best — shipped** |
| Non-greedy CPU + CPU top-128 before Gumbel | 19.0 | No improvement — transfer dominates |
| Non-greedy CPU + float16 transfer | 18.4 | Regression — device-side .half() is slow |
| Device-side topk | ~5 | No TTNN lowering |
| Non-greedy device | ~5 | |

*estimated from pre-optimization measurements

**Gap**: 2.7 tok/s (19.0 vs 21.7 ceiling, ~12%). Bottleneck is device→CPU transfer.

### Batch 8

| Config | tok/s | Notes |
|---|---|---|
| Skip sampling (ceiling) | 19.7 | |
| Greedy device | 20.0 | |
| Non-greedy CPU (after Gumbel-max) | 15.1 | |
| Non-greedy CPU + CPU top-128 before Gumbel | 15.5 | +0.4 tok/s from reducing Gumbel set |
| Non-greedy device | ~5 | |

**Gap**: 4.6 tok/s (15.1 vs 19.7 ceiling, ~23%). Bottleneck is transfer (~15%) + CPU ops (~5%).

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

### Fix 3: CPU top-128 before Gumbel-max (GATED — `VLLM_TT_CPU_TOPK=1`)

After penalties, temperature, and top-k/top-p have been applied on the full 128K vocab, reduce to top-128 elements before Gumbel-max sampling. Gumbel noise generation + argmax runs on 128 elements instead of 128K.

- No measurable improvement at batch 2 (transfer is the bottleneck, not CPU ops)
- +0.4 tok/s at batch 8 (15.1→15.5)
- Gated behind `VLLM_TT_CPU_TOPK=1` env var since improvement is marginal

## Optimizations Attempted and Reverted

### Micro-optimization: exponential_() for Gumbel noise

Replaced `−log(−log(rand + ε) + ε)` with `torch.empty().exponential_().log()`. Was actually **slower** — caused 19→18.1 regression. Reverted.

### Micro-optimization: non_blocking=True transfer

`logits.to(device="cpu", non_blocking=True)` — no measurable improvement, possibly hurt. The next line accesses the tensor immediately, forcing a sync.

### Micro-optimization: skip temp=1.0 division

`torch.allclose` check to skip the division was slower than just dividing. Reverted.

### Float16 transfer

`logits.half().cpu().float()` — halves transfer bandwidth but the `.half()` dtype conversion on TT device is slow, causing 19.0→18.4 regression at batch 2. The on-device dtype cast costs more than the bandwidth saved.

## Attempted but Failed

### Device-side topk with float32 index cast

Run `torch.topk(logits, 128)` on device, cast int32 indices to float32 for transfer (workaround for TT runtime int32 memcpy crash), transfer only 128 values + 128 indices to CPU.

**Result**: `torch.topk` has no efficient TTNN lowering — dropped to ~5 tok/s (same as device non-greedy sampling). The op runs but is extremely slow on TT hardware.

Infrastructure is implemented and gated behind `VLLM_TT_DEVICE_TOPK=1`:
- `_precompile_topk()` method precompiles the XLA graph during warmup
- `capture_model()` conditionally calls it
- Index cast uses `torch.float32` (not `logits.dtype` which is bfloat16 — bfloat16 only represents integers up to 256, corrupting vocab indices up to 128K)

### Device-side topk with int32 indices (original attempt)

**Failed**: TT runtime crashes with `Tensor data type must be the alias of the unsupported data type: Int32` — the topk indices are int32 and the TT PJRT memcpy doesn't support that dtype.

### Skip sampling hack (diagnostic only)

Return dummy token without any transfer or computation. Used to measure forward-pass ceiling:
- Batch 1: 22.5 tok/s
- Batch 2: 21.7 tok/s
- Batch 8: 19.7 tok/s

Gated behind `SKIP_SAMPLING=1` env var in `sample_from_logits_cpu`.

## Bottleneck Analysis

| Source | Cost at batch 2 | Cost at batch 8 |
|---|---|---|
| Model forward pass | ~88% (21.7 ceiling) | ~80% (19.7 ceiling) |
| Device→CPU logits transfer | ~11% | ~15% (128K × batch × 4 bytes) |
| CPU sampling ops | ~1% (negligible) | ~5% (negligible after Gumbel-max) |

The transfer (`logits.cpu()`) is the remaining bottleneck for CPU sampling. The actual sampling math is fast after the Gumbel-max optimization. At batch 2, the transfer alone accounts for nearly all the gap.

## Open Leads

1. **Fix TTNN topk lowering** (most impactful) — if `torch.topk` runs efficiently on TT hardware, device-side topk infrastructure is ready (`VLLM_TT_DEVICE_TOPK=1`). Would recover most of the 4.6 tok/s gap at batch 8.
2. **Fix TT runtime int32 memcpy** → would allow transferring topk indices directly without float32 cast workaround.
3. **Device-side Gumbel sampling** — compile `argmax(logits + gumbel)` as a single device op, transfer only 1 token per request. Needs gumbel noise generation on device.
4. **Improve device non-greedy sampling** — currently 5 tok/s, ongoing work by tt-xla team. If this reaches 15+ tok/s, CPU sampling becomes unnecessary.

## Env Vars Reference

| Env var | Default | Effect |
|---|---|---|
| `SKIP_SAMPLING` | unset | Returns dummy token — measures forward-pass ceiling |
| `VLLM_TT_DEVICE_TOPK` | `0` | Runs topk on device (slow — no TTNN lowering, ~5 tok/s) |
| `VLLM_TT_CPU_TOPK` | `0` | CPU top-128 before Gumbel-max (+0.4 tok/s at batch 8) |

## Files Changed

- `integrations/vllm_plugin/vllm_tt/model_runner.py` — `sample_from_logits_cpu`, `_precompile_topk`, `capture_model`
- `tt-inference-server-2/tt-media-server/tt_model_runners/vllm_runner.py` — `SAMPLING_DEFAULTS`, `enable_trace`
- `tt-inference-server-2/tt-media-server/open_ai_api/chat.py` — request param logging

## Environment

- Machine: QB2 (P300X2), single P150 chip
- Model: Llama-3.1-8B-Instruct, bfp_bf8, consteval, opt_level=1, enable_trace=True
- Branch: `kmabee/vllm_perf_apr12` on tt-xla-2
- Container: Ubuntu 24.04
