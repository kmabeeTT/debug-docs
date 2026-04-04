# Debug Docs

A collection of debug notes, findings, and learnings from working with TT-XLA, VLLM, and Tenstorrent hardware.

## Purpose

This repo captures important debugging insights that are worth preserving across sessions - root cause analyses, graph comparisons, workaround documentation, and patterns discovered during bringup and testing.

## Contents

- [VLLM Sampling Graph Comparison](vllm_sampling_graph_comparison.md) - Greedy vs non-greedy sampling TTIR graph analysis showing both paths compile and run on TT device
- [VLLM TP Sampling Graph Comparison](vllm_tp_sampling_graph_comparison.md) - Tensor parallel (Llama-3.2-3B, N300) greedy vs non-greedy analysis confirming sampling runs on device even without torch.compile
- [SamplingParams On-Device Analysis](sampling_params_on_device_analysis.md) - Which SamplingParams run on TT device vs CPU. IR graph comparison proving penalties/bad_words/logit_bias don't generate device graphs
- [VLLM Bad Words Architecture](vllm_bad_words_architecture.md) - How bad_words works end-to-end: single-token bans, multi-token prefix matching, CPU/device execution boundary, and the full call chain from model_runner through the compiled sampler
- [Stress Test & KV Cache Bleed Analysis](stress_test_kv_cache_bleed_analysis.md) - Confirmed KV cache bleed bug on single-chip P100a with dynamic batcher (batch=4, Llama-3.2-1B-Instruct). 20% repro rate (6/30 runs). Last batch slot (index 3) involved in every failure. Includes initial stress test coherence analysis and detailed cross-session contamination evidence
- [Prefix Cache Recompilation](prefix_cache_recompilation.md) - Root cause analysis of ~24s TTFT caused by torch.compile recompilation when prefix caching triggers a page_table slice with a new shape. Four fix options evaluated; torch.roll approach recommended and implemented. Includes debug logging additions to model_runner.py for detecting recompilation at runtime
- [KV Cache Capacity Planning](kv_cache_capacity_planning.md) - Formula and observed values for KV cache sizing per model at different gpu_memory_utilization settings. Scaling guidance for multi-user deployments on TT hardware
