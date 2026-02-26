# vLLM Bad Words Architecture on TT Hardware

How `bad_words` works in the TT vLLM sampler — covering single-token,
multi-token prefix matching, and the CPU/device execution boundary.

PR: https://github.com/tenstorrent/tt-xla/pull/3482
Issue: https://github.com/tenstorrent/tt-xla/issues/3363

## The big picture

When vLLM generates text, it works token-by-token in a loop called "decode steps." Each step:
1. The model produces **logits** — a score for every token in the vocabulary (e.g. 50,272 scores for OPT-125m)
2. The **sampler** picks the next token based on those scores
3. The chosen token is appended to the output, and the loop repeats

`bad_words` says: "never generate this word/phrase." The job is to suppress the banned tokens' logits to `-inf` so the sampler can never pick them.

## Single-token vs multi-token

**Single-token** (e.g. banning "the" → token `5`): Simple — set `logits[5] = -inf` every step.

**Multi-token** (e.g. banning "New York" → tokens `[188, 469]`): You can't always ban token `469` ("York") — it's a valid word on its own. You only ban `469` when the **previous** token was `188` ("New"). This requires checking the output history each step.

## The flow

```
  ┌─────────────────────────────────────────────────────────────┐
  │                    DECODE LOOP (per step)                   │
  │                                                             │
  │  ┌──────────────┐                                           │
  │  │  Model runs   │   logits: [batch, vocab_size]             │
  │  │  on TT device │──────────────────────┐                   │
  │  └──────────────┘                       │                   │
  │                                         ▼                   │
  │                              ┌─────────────────────┐        │
  │                              │  logits on device    │        │
  │                              └─────────┬───────────┘        │
  │                                        │                    │
  │  ┌─ CPU ──────────────────────────┐    │                    │
  │  │                                │    │                    │
  │  │  _compute_bad_words_mask()     │    │                    │
  │  │                                │    │                    │
  │  │  For each request:             │    │                    │
  │  │    bad_words = [[188, 469]]    │    │                    │
  │  │    output_so_far = [... 188]   │    │                    │
  │  │                                │    │                    │
  │  │    "Last token is 188,         │    │                    │
  │  │     prefix matches [188]!      │    │                    │
  │  │     Ban token 469."            │    │                    │
  │  │                                │    │                    │
  │  │  Result: mask[req, 469] = -inf │    │                    │
  │  │          mask[req, *]   = 0    │    │                    │
  │  │                                │    │                    │
  │  │  mask shape: [batch, vocab]    │    │                    │
  │  └──────────────┬─────────────────┘    │                    │
  │                 │ .to(device)           │                    │
  │                 ▼                       ▼                    │
  │         ┌──────────────────────────────────────┐            │
  │         │  Sampler (on TT device)              │            │
  │         │                                      │            │
  │         │  1. logits += bad_words_mask  ◄─ -inf │            │
  │         │  2. logits += logit_bias             │            │
  │         │  3. apply penalties                  │            │
  │         │  4. apply temperature                │            │
  │         │  5. top-k / top-p filtering          │            │
  │         │  6. sample token                     │            │
  │         └──────────────┬───────────────────────┘            │
  │                        │                                    │
  │                        ▼                                    │
  │               selected_token = 302  (not 469!)              │
  │                        │                                    │
  │                        ▼                                    │
  │              output_so_far.append(302)                      │
  │              (history grows for next step's prefix matching) │
  │                                                             │
  └──────────────────── next decode step ───────────────────────┘
```

## Why the CPU/device split?

The prefix matching logic is inherently dynamic:
- Different requests have different bad_words lists
- Token histories grow each step and vary per request
- Prefix comparison is a variable-length list equality check

None of this can be expressed as fixed-shape tensor ops for XLA compilation. But the **result** — "which tokens are banned right now" — is a fixed-shape `[batch, vocab]` tensor. So we do the dynamic work on CPU, produce a static tensor, and send it to the device.

This is the same pattern used throughout the TT sampler:
- `_compute_token_counts` → penalty mask (CPU build, device apply)
- `_compute_prompt_mask` → repetition penalty mask (CPU build, device apply)
- `_compute_bad_words_mask` → bad words mask (CPU build, device apply)

## How we know `_compute_bad_words_mask` runs on CPU

It executes on CPU because of **where it's called** — inside `from_input_batch()`,
which is a regular Python classmethod that runs before the compiled graph:

```python
# metadata.py, from_input_batch() — this is plain Python, not inside torch.compile
bad_words_cpu = cls._compute_bad_words_mask(
    input_batch.bad_words_token_ids,   # Python dict
    input_batch.req_output_token_ids,  # Python list[list[int]]
    padded_num_reqs,                   # Python int
    vocab_size,                        # Python int
)
bad_words_mask = bad_words_cpu.to(xla_device)  # <-- transfer happens here
```

The inputs are all Python data structures (dicts, lists of lists, ints) — not device
tensors. `torch.zeros()` inside the method creates a CPU tensor by default. The
`.to(xla_device)` call afterward is the explicit boundary where data moves to device.

The full call chain shows where the compiled boundary is:

```
model_runner.sample_tokens()                            # regular Python method
  → XLASupportedSamplingMetadata.from_input_batch()     # regular Python classmethod
      → cls._compute_bad_words_mask()                   # CPU — builds tensor
      → .to(xla_device)                                 # transfer to device
  → self.sample_from_logits_func(logits, metadata)      # THIS is torch.compile'd
      → Sampler.sample()                                # on device
          → logits += bad_words_mask                    # on device
```

The compiled boundary is `sample_from_logits_func`. Everything before it (including
`from_input_batch`) is regular Python on CPU. Everything inside it runs on device.
The mask crosses that boundary as a device tensor input to the compiled function.

This is identical to how `_compute_token_counts` and `_compute_prompt_mask` work —
they're called from the same `from_input_batch()` method, build CPU tensors, and
`.to(xla_device)` them.

## Concrete example

Prompt: `"The capital of"`, bad_words: `["New York"]` → tokens `[188, 469]`

| Step | Output so far | Prefix check | Mask | Sampled |
|------|--------------|--------------|------|---------|
| 1 | `[]` | Need 1 token for prefix, have 0 → skip | all zeros | `188` ("New") |
| 2 | `[188]` | Last 1 token `[188]` == prefix `[188]` → **ban 469** | `mask[469]=-inf` | `312` (not "York"!) |
| 3 | `[188, 312]` | Last 1 token `[312]` != prefix `[188]` → skip | all zeros | `...` |

The model wanted to say "New York" but step 2 banned "York" because "New" had just been generated. The model picks a different continuation.

## Key files

- `integrations/vllm_plugin/vllm_tt/metadata.py` — `_compute_bad_words_mask()` and `from_input_batch()` call site
- `integrations/vllm_plugin/vllm_tt/sampler.py` — `apply_bad_words()` (on-device `logits += mask`)
- `integrations/vllm_plugin/vllm_tt/model_runner.py` — `sample_from_logits()` greedy guard checks `no_bad_words`
- `tests/integrations/vllm_plugin/sampling/test_logit_bias_bad_words_correctness.py` — CPU correctness tests
- `tests/integrations/vllm_plugin/sampling/test_sampling_params_synthetic.py` — on-device compilation test
- `tests/integrations/vllm_plugin/sampling/test_sampling_params.py` — E2E model test
