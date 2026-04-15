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

1. **Hybrid device-topk + CPU sampling** (most promising — see detailed design below)
2. **Fix TT runtime int32 memcpy** → enables full device-side topk with indices
3. **Device-side Gumbel sampling** — compile `argmax(logits + gumbel)` as a single device op, transfer only 1 token per request. Needs gumbel noise generation on device.
4. **Improve device non-greedy sampling** — currently 5 tok/s, ongoing work by tt-xla team. If this reaches 15+ tok/s, CPU sampling becomes unnecessary.

## Detailed Design: Hybrid Device-Topk + CPU Sampling

### Goal

Run `torch.topk` on device to reduce the logits from 128K to 128 elements, transfer only the small result to CPU, then do all sampling (penalties, temperature, top-p, Gumbel-max) on CPU. This cuts the device→CPU transfer from ~2MB to ~2KB at batch=8.

### Why This Should Work

- `torch.topk` is a standard PyTorch op supported by XLA
- The output shape is static: `(max_num_reqs, 128)` for values and indices — no dynamic shapes
- It can be precompiled during warmup alongside the backbone graphs
- All sampling math stays on CPU — no need to compile penalties/temperature/Gumbel on device
- The greedy path already does `argmax` on device and transfers 1 token — this extends that pattern

### Expected Performance

| Batch | Current CPU sampling | After hybrid | Transfer reduction |
|---|---|---|---|
| 1 | 19.0 tok/s | ~19.5+ | 128K → 128 floats + 128 indices |
| 2 | 19.0 tok/s | ~19.5+ | 256K → 256 floats + 256 indices |
| 8 | 15.1 tok/s | ~18-19 | 1M → 1K floats + 1K indices |
| 32 | ~7 tok/s* | ~15+ | 4M → 4K floats + 4K indices |

Skip-sampling diagnostic showed 19.7 tok/s ceiling at batch 8. Hybrid approach should get close to that since transfer becomes negligible.

### Implementation Plan

#### Step 1: Precompile device-side topk during warmup

In `capture_model()` or a new `_precompile_topk()`, run a dummy topk to compile the XLA graph:

```python
def _precompile_topk(self):
    """Precompile the device-side top-k graph for CPU sampling."""
    dummy_logits = torch.zeros(
        (self.max_num_reqs, self.vocab_size), dtype=self.dtype, device=self.device
    )
    _ = torch.topk(dummy_logits, self.CPU_SAMPLING_TOPK, dim=-1)
    xm.wait_device_ops()
```

This ensures the topk graph is cached before any real requests arrive.

#### Step 2: Handle the int32 index problem

The previous attempt failed because `torch.topk` returns int32/int64 indices and TT runtime's `memcpy` crashes on int32. Three approaches to try:

**Approach A: Transfer values only, reconstruct indices on CPU**
```python
# On device: only get the top-k values
topk_vals, _ = torch.topk(logits, 128, dim=-1)  # discard indices on device
topk_vals_cpu = topk_vals.cpu()  # transfer only float values (works)

# On CPU: find which vocab tokens these values correspond to
logits_cpu = logits.cpu()  # still need full transfer for index lookup
```
This doesn't help — still transfers full logits for index reconstruction.

**Approach B: Use argsort instead of topk**
```python
# If argsort returns int64 (not int32), it might transfer fine
sorted_indices = torch.argsort(logits, dim=-1, descending=True)[:, :128]
sorted_indices_cpu = sorted_indices.to(torch.int64).cpu()
# Then gather values on CPU
logits_cpu_full = logits.cpu()
topk_vals = logits_cpu_full.gather(1, sorted_indices_cpu)
```
Still transfers full logits. Not helpful.

**Approach C: Encode indices as floats (workaround)**
```python
# On device: get topk values and indices
topk_vals, topk_idx = torch.topk(logits, 128, dim=-1)
# Cast indices to float to avoid int32 memcpy issue
topk_idx_as_float = topk_idx.to(torch.float32)  # or .to(logits.dtype)
# Transfer both as floats
topk_vals_cpu = topk_vals.cpu()
topk_idx_cpu = topk_idx_as_float.cpu().to(torch.int64)  # cast back on CPU
```
This encodes int32 indices as float32 for the device→CPU transfer, then converts back. Float32 can exactly represent all integers up to 2^24 (16M) — well above any vocab size (128K). **This is the most promising workaround.**

#### Step 3: Modify sample_from_logits_cpu

```python
def sample_from_logits_cpu(self, logits, sampling_metadata, num_reqs=None):
    # Greedy: argmax on device (existing fast path)
    if sampling_metadata.all_greedy:
        return torch.argmax(logits, dim=-1, keepdim=True).cpu()

    # Device-side top-k: reduce 128K to 128 elements
    k = min(self.CPU_SAMPLING_TOPK, logits.size(-1))
    topk_vals, topk_idx = torch.topk(logits, k, dim=-1)
    
    # Workaround: cast indices to float for transfer (int32 memcpy unsupported)
    topk_vals_cpu = topk_vals.cpu()
    topk_idx_cpu = topk_idx.to(logits.dtype).cpu().to(torch.int64)
    
    # ... rest of sampling on CPU using topk_vals_cpu / topk_idx_cpu ...
    # (penalties, temperature, top-p, Gumbel-max on 128 elements)
```

#### Step 4: Handle penalties approximation

Penalties (repetition, frequency, presence) modify logits based on which tokens appeared in the output. After device-side topk, we only have 128 tokens — tokens outside the top-128 that penalties would have promoted are missed.

This is acceptable because:
- Repetition penalty only boosts tokens that already appeared (likely in top-128)
- The penalty magnitude is small (1.1x) — rarely enough to promote a token from outside top-128 into the sampling set
- For exact correctness, penalties could be applied on device before topk (would need a compiled graph)

### Key Risks

1. **`topk_idx.to(logits.dtype)` cast might fail on XLA** — the int→float cast needs to work on the XLA device. Test this first.
2. **Topk XLA graph recompilation** — if the topk is called outside `torch.compile`, it might compile a new graph every step. Need to verify it caches or precompile it.
3. **`torch.topk` not supported on TT** — the op might not have a TTNN lowering. If so, it would fall back to CPU anyway (defeating the purpose). Check TTNN op support.
4. **Graph breaks** — calling topk between the compiled model forward and the CPU sampling might cause graph breaks or sync points.

### Verification Plan

1. First test: just `topk_idx.to(torch.float32).cpu()` in isolation — does it crash?
2. If yes, try `topk_idx.to(torch.bfloat16).cpu()` (bfloat16 can represent integers up to 256 exactly — enough for k=128 indices but NOT for vocab indices up to 128K)
3. If float32 cast works, integrate into `sample_from_logits_cpu` and benchmark
4. Check for recompilation: `VLLM_XLA_CHECK_RECOMPILATION=1` — topk should not trigger backbone recompilation

### File to Modify

`integrations/vllm_plugin/vllm_tt/model_runner.py`:
- `sample_from_logits_cpu` method (~line 2195)
- `capture_model` method — add `_precompile_topk()` call
- `CPU_SAMPLING_TOPK` class constant (currently 128)

## Files Changed

- `integrations/vllm_plugin/vllm_tt/model_runner.py` — `sample_from_logits_cpu` method (~line 2190)
- `tt-inference-server-2/tt-media-server/tt_model_runners/vllm_runner.py` — `SAMPLING_DEFAULTS`, `enable_trace`
- `tt-inference-server-2/tt-media-server/open_ai_api/chat.py` — request param logging

## Environment

- Machine: QB2 (P300X2), single P150 chip
- Model: Llama-3.1-8B-Instruct, bfp_bf8, consteval, opt_level=1, enable_trace=True
- Branch: `kmabee/vllm_perf_apr12` on tt-xla-2
- Container: Ubuntu 24.04
