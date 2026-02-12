# Debug Docs

A collection of debug notes, findings, and learnings from working with TT-XLA, VLLM, and Tenstorrent hardware.

## Purpose

This repo captures important debugging insights that are worth preserving across sessions - root cause analyses, graph comparisons, workaround documentation, and patterns discovered during bringup and testing.

## Contents

- [VLLM Sampling Graph Comparison](vllm_sampling_graph_comparison.md) - Greedy vs non-greedy sampling TTIR graph analysis showing both paths compile and run on TT device
