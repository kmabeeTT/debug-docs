# CLAUDE.md

## Purpose

This is a debug documentation repo. It stores important learnings, root cause analyses, and debugging insights from working with TT-XLA, TT-MLIR, VLLM, and Tenstorrent hardware.

## Writing Guidelines

- Each document should be a standalone markdown file focused on a single topic or investigation
- Include the date, model/test context, and a clear summary at the top of each doc
- Use tables for comparisons and code blocks for IR snippets or commands
- Link to related issues, artifacts, or other docs when relevant
- Keep findings factual and reproducible - include commands used and how to identify key patterns
- Update the README.md table of contents when adding new docs

## File Naming

Use descriptive snake_case names that indicate the topic:
- `vllm_sampling_graph_comparison.md` - comparison of two approaches/behaviors
- `ttnn_sort_type_mismatch_rca.md` - root cause analysis of a specific bug
- `opt_350m_prefill_decode_graphs.md` - analysis of a specific model's compiled graphs

## Organization

Files live flat in the repo root. If the repo grows large, consider grouping by area:
- `vllm/` - VLLM integration findings
- `mlir/` - MLIR/TTIR/TTNN compiler findings
- `hardware/` - Device-level observations
- `models/` - Model-specific debug notes
