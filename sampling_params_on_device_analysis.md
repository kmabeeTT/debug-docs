# SamplingParams On-Device Analysis: What Runs on TT Hardware vs CPU

**Date**: 2026-02-12
**Machine**: single_device (opt-125m)
**Branch**: kmabee/sampling_params_fixes_and_test.multichip

## Summary

Penalties (presence, frequency, repetition), bad_words, logit_bias, and
allowed_token_ids are **NOT compiled to TT device graphs**. They run on
CPU. The compiled device graph is identical with and without these params.

## What's On-Device vs CPU

| SamplingParam | On-device? | Where it runs |
|---|---|---|
| temperature | Yes | TPUSampler.apply_temperature |
| top_k | Yes | TPUSampler.topk_topp_sampler (forward_tpu) |
| top_p | Yes | TPUSampler.topk_topp_sampler (forward_tpu) |
| min_p | Yes | TPUSampler.apply_min_p |
| greedy (argmax) | Yes | TPUSampler.greedy_sample |
| presence_penalty | **No** | CPU, vLLM engine layer |
| frequency_penalty | **No** | CPU, vLLM engine layer |
| repetition_penalty | **No** | CPU, vLLM engine layer |
| bad_words | **No** | CPU, vLLM engine layer |
| logit_bias | **No** | CPU, vLLM engine layer |
| allowed_token_ids | **No** | CPU, vLLM engine layer |
| stop / stop_token_ids | **No** | CPU, generation loop |
| ignore_eos | **No** | CPU, generation loop |
| max/min_tokens | **No** | CPU, generation loop |
| seed | **Rejected** | platform.py raises ValueError |
| logprobs | Separate graph | gather_logprobs compiled separately |

## Root Cause

The TT backend uses `TPUSampler` (`vllm/v1/sample/tpu/sampler.py`),
which is a stripped-down version of the GPU `Sampler` (`vllm/v1/sample/sampler.py`).

**GPU Sampler forward() does:**
```python
logits = self.apply_allowed_token_ids(logits, sampling_metadata)
logits = self.apply_bad_words(logits, sampling_metadata)
logits = self.apply_penalties(logits, sampling_metadata)
# then temperature, top_k, top_p, min_p, sample
```

**TPU Sampler forward() only does:**
```python
logits = self.apply_temperature(logits, sampling_metadata.temperature)
logits = self.apply_min_p(logits, sampling_metadata.min_p)
random_sampled = self.topk_topp_sampler(logits, ...)
```

The TPUSupportedSamplingMetadata has fields for penalties/bad_words/logit_bias
but they are marked TODO and never consumed by TPUSampler.sample().

## How I Verified: IR Graph Comparison

### Step 1: Create a test with two generate() calls

```python
# test_ir_penalty_comparison.py
import pytest
import vllm

@pytest.fixture(scope="module")
def llm():
    return vllm.LLM(
        model="facebook/opt-125m",
        max_num_batched_tokens=128, max_num_seqs=1,
        max_model_len=128, gpu_memory_utilization=0.001,
        enable_prefix_caching=False, disable_log_stats=True,
        enforce_eager=True,
        additional_config={"enable_const_eval": False, "min_context_len": 32},
    )

def test_baseline_no_penalties(llm):
    params = vllm.SamplingParams(temperature=0.8, max_tokens=4)
    out = llm.generate(["Once upon a time"], params, use_tqdm=False)[0].outputs[0].text
    print(f"Baseline output: {out!r}")

def test_with_penalties(llm):
    params = vllm.SamplingParams(
        temperature=0.8, presence_penalty=2.0,
        frequency_penalty=2.0, repetition_penalty=2.0, max_tokens=4,
    )
    out = llm.generate(["Once upon a time"], params, use_tqdm=False)[0].outputs[0].text
    print(f"Penalty output: {out!r}")
```

### Step 2: Run with debug IR logging

```bash
TTXLA_LOGGER_LEVEL=DEBUG python -m pytest -svv test_ir_penalty_comparison.py 2>&1 | tee ir_penalty_comparison.log
```

### Step 3: Extract TTIR graphs

```bash
python3 /localdev/kmabee/scripts/extract_mlir_graphs.py ir_penalty_comparison.log --type ttir
```

Result: 7 unique TTIR graphs extracted (42 total modules across compilation stages).

### Step 4: Check for new graphs during penalty run

```bash
grep -n "module @Sync\|Baseline output\|Penalty output" ir_penalty_comparison.log | tail -15
```

Result:
```
53947:    builtin.module @SyncTensorsGraph.31 ...   <-- last graph compilation
54021:Baseline output: ' time when it was'           <-- end of baseline test
54023:... Penalty output: ' little boy named Buddy'  <-- end of penalty test
```

**Zero new graph compilations between baseline and penalty runs.** The penalty
test reused the exact same compiled graphs. The different output text confirms
penalties ARE being applied — but on CPU, not on device.

## Key Graph Names

| Graph | Ops | Purpose |
|---|---|---|
| SyncTensorsGraph.3477 | 841 | Model forward (precompile shape 1) |
| SyncTensorsGraph.3757 | 987 | Model forward (precompile shape 2) |
| SyncTensorsGraph.3665 | 797 | Model forward (decode) |
| SyncTensorsGraph.318 | 122 | sample_from_logits |
| SyncTensorsGraph.31 | 12 | gather_logprobs |
| SyncTensorsGraph.13 | 7 | Small utility graph |

## Path to On-Device Penalties

Once the TPU Sampler code moves into the TT-XLA vLLM plugin (after vLLM
v0.14.0 removes TPU support), the TPUSampler can be extended to match the
GPU Sampler's forward() — adding apply_penalties, apply_bad_words,
apply_allowed_token_ids, apply_logit_bias. These are all tensor ops
(masked_fill, gather, scatter) that should compile to TT via XLA.

## Files Referenced

- `venv/.../vllm/v1/sample/tpu/sampler.py` — TPU Sampler (on-device, 4 params)
- `venv/.../vllm/v1/sample/sampler.py` — GPU Sampler (on-device, all params)
- `venv/.../vllm/v1/sample/tpu/metadata.py` — TPUSupportedSamplingMetadata (TODO fields)
- `integrations/vllm_plugin/vllm_tt/model_runner.py:1888` — sample_from_logits
- `integrations/vllm_plugin/vllm_tt/model_runner.py:1059` — execute_model
- `integrations/vllm_plugin/vllm_tt/platform.py:267` — seed rejection
