# New Models Bringup Target List (66 models, 14B-288B)

Target hardware: **Blackhole QuietBox 2** (4x 32GB = 128GB DRAM) or **Wormhole Galaxy** (32x 12GB = 384GB DRAM) with BFP8 weights.

## Memory Budget

All model weights (including all MoE experts) must fit in DRAM. BFP8 is ~1 byte/param.

| Hardware | Total DRAM | Usable for Weights (~75%) | Fits Total Params Up To |
|----------|-----------|--------------------------|------------------------|
| BH QuietBox 2 | 128 GB | ~96 GB | ~96B |
| WH Galaxy | 384 GB | ~288 GB | ~288B |

## Selection Criteria
- All model weights (total params, not active params) must fit on at least one target platform
- Not already in tt-xla test configs (tensor parallel, single device, or data parallel)
- Prioritized by HuggingFace downloads and industry relevance

---

## 1. General LLMs (28 models)

| # | HuggingFace Model ID | Total Params | Active Params | BFP8 Size | Downloads/mo | Target HW | In forge-models? | Notes |
|---|----------------------|-------------|--------------|-----------|-------------|-----------|-----------------|-------|
| 1 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | 32B | 32B (dense) | ~32 GB | 1,026K | BH QB2 | Partial (32B only) | Most-downloaded R1 distill |
| 2 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | 14B | 14B (dense) | ~14 GB | 568K | BH QB2 | No | Lightweight R1 distill |
| 3 | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | 70B | 70B (dense) | ~70 GB | 153K | BH QB2 | No | Largest dense R1 distill |
| 4 | `deepseek-ai/DeepSeek-V2-Lite-Chat` | 16B | 2.4B (MoE) | ~16 GB | 826K | BH QB2 | No | Small MoE, efficiency benchmark |
| 5 | `mistralai/Mixtral-8x7B-Instruct-v0.1` | 46.7B | 12.9B (MoE) | ~47 GB | 530K | BH QB2 | Partial (dir exists) | Pioneering open MoE |
| 6 | `mistralai/Mixtral-8x22B-Instruct-v0.1` | 176B | 39B (MoE) | ~176 GB | 27K | Galaxy | Partial (dir exists) | Largest open Mixtral |
| 7 | `CohereLabs/c4ai-command-r-v01` | 35B | 35B (dense) | ~35 GB | 14K | BH QB2 | No | Unique Cohere arch, RAG-optimized |
| 8 | `CohereLabs/c4ai-command-r-plus-08-2024` | 104B | 104B (dense) | ~104 GB | 3.8K | Galaxy | No | Enterprise-grade, 128K context |
| 9 | `01-ai/Yi-1.5-34B-Chat` | 34B | 34B (dense) | ~34 GB | 13K | BH QB2 | No | Top Chinese-English bilingual |
| 10 | `LGAI-EXAONE/EXAONE-3.5-32B-Instruct` | 32B | 32B (dense) | ~32 GB | 77K | BH QB2 | No | LG AI Research |
| 11 | `LGAI-EXAONE/EXAONE-4.0-32B` | 32B | 32B (dense) | ~32 GB | 24K | BH QB2 | No | Latest generation EXAONE |
| 12 | `baichuan-inc/Baichuan-M2-32B` | 32B | 32B (dense) | ~32 GB | 84K | BH QB2 | No | Medical/bilingual focus |
| 13 | `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5` | 49B | 49B (dense) | ~49 GB | 203K | BH QB2 | No | NVIDIA flagship |
| 14 | `nvidia/OpenReasoning-Nemotron-32B` | 32B | 32B (dense) | ~32 GB | 132K | BH QB2 | No | NVIDIA reasoning model |
| 15 | `CohereLabs/aya-expanse-32b` | 32B | 32B (dense) | ~32 GB | 8.5K | BH QB2 | No | 23-language multilingual |
| 16 | `internlm/internlm2-chat-20b` | 20B | 20B (dense) | ~20 GB | 19K | BH QB2 | No | Shanghai AI Lab |
| 17 | `ai21labs/AI21-Jamba-Large-1.6` | 94B | ~12B (MoE+Mamba) | ~94 GB | 2.5K | BH QB2 (tight) | No | Mamba+Transformer hybrid, novel arch |
| 18 | `databricks/dbrx-instruct` | 132B | 36B (MoE) | ~132 GB | gated | Galaxy | No | Enterprise MoE, 16 experts choose 4 |
| 19 | `tiiuae/falcon-40b-instruct` | 40B | 40B (dense) | ~40 GB | 30K | BH QB2 | No | Falcon 1 arch (different from Falcon 3) |
| 20 | `deepseek-ai/deepseek-llm-67b-chat` | 67B | 67B (dense) | ~67 GB | 3.3K | BH QB2 | No | Original DeepSeek dense LLM |
| 21 | `meta-llama/Llama-2-70b-chat-hf` | 70B | 70B (dense) | ~70 GB | 63K | BH QB2 | No | Classic Llama 2, still widely used |
| 22 | `deepseek-ai/DeepSeek-V2-Chat` | 236B | ~21B (MoE) | ~236 GB | 10K | Galaxy | No | Full DeepSeek V2 MoE |
| 23 | `zai-org/GLM-4.7-Flash` | 31B | ~3.5B (MoE) | ~31 GB | 770K | BH QB2 | No (has full 358B GLM-4.7, not Flash) | Zhipu AI, very high downloads |
| 24 | `bigscience/bloom` | 176B | 176B (dense) | ~176 GB | 5.9K | Galaxy | Partial (1.1B only) | Landmark multilingual (46 languages) |
| 25 | `lmsys/vicuna-33b-v1.3` | 33B | 33B (dense) | ~33 GB | 2.2K | BH QB2 | No | LMSYS chatbot benchmark model |
| 26 | `OrionStarAI/Orion-14B-Chat` | 14B | 14B (dense) | ~14 GB | 13K | BH QB2 | No | Chinese-English bilingual |
| 27 | `tiiuae/falcon-180B-chat` | 180B | 180B (dense) | ~180 GB | 836 | Galaxy | No | Largest Falcon 1, biggest open dense LLM at release |
| 28 | `mosaicml/mpt-30b-chat` | 30B | 30B (dense) | ~30 GB | gated | BH QB2 | No | Unique MPT architecture (Databricks/MosaicML) |

## 2. Code / Math / Reasoning Models (27 models)

| # | HuggingFace Model ID | Total Params | Active Params | BFP8 Size | Downloads/mo | Target HW | In forge-models? | Notes |
|---|----------------------|-------------|--------------|-----------|-------------|-----------|-----------------|-------|
| 29 | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | 16B | 2.4B (MoE) | ~16 GB | 431K | BH QB2 | No | Code-specialized MoE |
| 30 | `mistralai/Codestral-22B-v0.1` | 22B | 22B (dense) | ~22 GB | 7.5K | BH QB2 | No | Mistral code model, fill-in-middle |
| 31 | `bigcode/starcoder2-15b` | 15B | 15B (dense) | ~15 GB | 7.5K | BH QB2 | No | BigCode project, top open code model |
| 32 | `microsoft/Phi-3.5-MoE-instruct` | 42B | 6.6B (MoE) | ~42 GB | 103K | BH QB2 | Yes (`phi3/phi_3_5/`) | Microsoft MoE, 16 experts choose 2 |
| 33 | `microsoft/Phi-3-medium-4k-instruct` | 14B | 14B (dense) | ~14 GB | 29K | BH QB2 | Partial (`phi3/`) | Strong benchmarks for size |
| 34 | `naver-hyperclovax/HyperCLOVAX-SEED-Think-32B` | 32B | 32B (dense) | ~32 GB | 61K | BH QB2 | No | Naver reasoning model |
| 35 | `naver-hyperclovax/HyperCLOVAX-SEED-Think-14B` | 14B | 14B (dense) | ~14 GB | 32K | BH QB2 | No | Lighter reasoning variant |
| 36 | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | 120B | 12B (MoE) | ~120 GB | 500K | Galaxy | No | NVIDIA large MoE |
| 37 | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | 30B | 3B (MoE) | ~30 GB | 1,568K | BH QB2 | No | Efficient MoE, highest-dl Nemotron |
| 38 | `nvidia/Nemotron-Cascade-2-30B-A3B` | 30B | 3B (MoE) | ~30 GB | 314K | BH QB2 | No | Cascade reasoning |
| 39 | `WizardLMTeam/WizardLM-70B-V1.0` | 70B | 70B (dense) | ~70 GB | 15K | BH QB2 | No | Instruction-tuned reasoning |
| 40 | `alpindale/WizardLM-2-8x22B` | 176B | 39B (MoE) | ~176 GB | 9.2K | Galaxy | No | MoE reasoning variant |
| 41 | `NousResearch/Hermes-4-70B-FP8` | 70B | 70B (dense) | ~70 GB | 11K | BH QB2 | No | Tool use + reasoning |
| 42 | `NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO` | 46.7B | 12.9B (MoE) | ~47 GB | 8.7K | BH QB2 | No | DPO-tuned MoE |
| 43 | `NousResearch/Nous-Hermes-2-Yi-34B` | 34B | 34B (dense) | ~34 GB | 8.3K | BH QB2 | No | Yi-based reasoning |
| 44 | `abacusai/Smaug-72B-v0.1` | 72B | 72B (dense) | ~72 GB | 8K | BH QB2 | No | Top arena reasoning model |
| 45 | `CohereLabs/aya-23-35B` | 35B | 35B (dense) | ~35 GB | 5.4K | BH QB2 | No | Multilingual reasoning, 23 languages |
| 46 | `LGAI-EXAONE/K-EXAONE-236B-A23B` | 236B | 23B (MoE) | ~236 GB | 37K | Galaxy | No | LG AI's largest MoE |
| 47 | `deepseek-ai/deepseek-coder-33b-instruct` | 33B | 33B (dense) | ~33 GB | 7.4K | BH QB2 | No | Popular DeepSeek code model |
| 48 | `Phind/Phind-CodeLlama-34B-v2` | 34B | 34B (dense) | ~34 GB | 1.8K | BH QB2 | No | CodeLlama fine-tune for code gen |
| 49 | `codellama/CodeLlama-34b-Instruct-hf` | 34B | 34B (dense) | ~34 GB | 20K | BH QB2 | No | Meta's code-specialized Llama 2 |
| 50 | `Qwen/Qwen2.5-Math-72B-Instruct` | 73B | 73B (dense) | ~73 GB | 2K | BH QB2 | No | Math-specialized (diff from base Qwen2.5) |
| 51 | `deepseek-ai/deepseek-moe-16b-chat` | 16B | ~2.8B (MoE) | ~16 GB | 20K | BH QB2 | No | Early DeepSeek MoE architecture |
| 52 | `WizardLMTeam/WizardMath-70B-V1.0` | 70B | 70B (dense) | ~70 GB | 756 | BH QB2 | No | Math-specialized 70B |
| 53 | `meta-math/MetaMath-70B-V1.0` | 70B | 70B (dense) | ~70 GB | 1.1K | BH QB2 | No | MetaMath bootstrapped reasoning |
| 54 | `microsoft/Phi-3-small-8k-instruct` | 14B | 14B (dense) | ~14 GB | 15K | BH QB2 | Partial (`phi3/`) | Strong embedding/retrieval use |
| 55 | `upstage/SOLAR-Pro-Preview-Instruct` | 22B | 22B (dense) | ~22 GB | 21K | BH QB2 | No | Upstage, Korean-English bilingual reasoning |

## 3. Multi-Modal Models (6 models)

| # | HuggingFace Model ID | Total Params | Active Params | BFP8 Size | Downloads/mo | Target HW | In forge-models? | Notes |
|---|----------------------|-------------|--------------|-----------|-------------|-----------|-----------------|-------|
| 56 | `OpenGVLab/InternVL3-38B` | 38B | 38B (dense) | ~38 GB | 91K | BH QB2 | No | Top open VLM |
| 57 | `OpenGVLab/InternVL3-78B` | 78B | 78B (dense) | ~78 GB | 40K | BH QB2 | No | Largest InternVL, SoTA benchmarks |
| 58 | `nvidia/NVLM-D-72B` | 72B | 72B (dense) | ~72 GB | 63K | BH QB2 | No | NVIDIA vision-language |
| 59 | `liuhaotian/llava-v1.6-34b` | 34B | 34B (dense) | ~34 GB | 33K | BH QB2 | Partial (1.5-7B only) | Popular VLM, Yi-34B backbone |
| 60 | `allenai/Molmo-72B-0924` | 72B | 72B (dense) | ~72 GB | 5.4K | BH QB2 | No | Allen AI, pointing + grounding |
| 61 | `OpenGVLab/InternVL2-26B` | 26B | 26B (dense) | ~26 GB | 2.5K | BH QB2 | No | InternViT + InternLM2 backbone |

## 4. Specialized Models (5 models)

| # | HuggingFace Model ID | Total Params | Active Params | BFP8 Size | Downloads/mo | Target HW | In forge-models? | Notes |
|---|----------------------|-------------|--------------|-----------|-------------|-----------|-----------------|-------|
| 62 | `moonshotai/Kimi-Linear-48B-A3B-Instruct` | 48B | 3B (MoE) | ~48 GB | 61K | BH QB2 | No | Linear attention MoE, novel arch |
| 63 | `LGAI-EXAONE/EXAONE-4.5-33B-FP8` | 33B | 33B (dense) | ~33 GB | 21K | BH QB2 | No | Multimodal EXAONE, FP8-native |
| 64 | `zai-org/cogvlm2-llama3-chat-19B` | 19B | 19B (dense) | ~19 GB | 4.9K | BH QB2 | No | Vision-language, Llama3 backbone |
| 65 | `Qwen/Qwen2-VL-72B-Instruct` | 72B | 72B (dense) | ~72 GB | TBD | BH QB2 | Partial (Qwen2.5-VL) | Qwen2-VL, not in current configs |
| 66 | `microsoft/Phi-4-reasoning-plus` | 15B | 15B (dense) | ~15 GB | 20K | BH QB2 | No | Reasoning-enhanced Phi-4 variant |

---

## Summary by Target Hardware

### BH QuietBox 2 (128 GB) - 55 models

| Size Bucket | BFP8 Weight Size | Count |
|------------|-----------------|-------|
| 14B-20B | ~14-20 GB | 15 |
| 30B-35B | ~30-35 GB | 20 |
| 36B-50B | ~36-50 GB | 8 |
| 67B-94B | ~67-94 GB | 12 |

### WH Galaxy (384 GB) - 11 models (Galaxy-only due to size)

| Size Bucket | BFP8 Weight Size | Models |
|------------|-----------------|--------|
| 104B-132B | ~104-132 GB | #8, 18, 36 |
| 176B | ~176 GB | #6, 24, 27, 40 |
| 236B | ~236 GB | #22, 46 |

Note: All BH QB2 models also run on Galaxy.

## tt-forge-models Coverage

| Status | Count | Details |
|--------|-------|---------|
| Yes (exact match) | 1 | Phi-3.5-MoE (#32) |
| Partial (dir exists, wrong variant/size) | 7 | DeepSeek-R1-Distill-32B (#1), Mixtral (#5-6), Phi-3-medium (#33), Phi-3-small (#54), LLaVA (#59), BLOOM (#24), Qwen2-VL (#65) |
| No | 58 | Everything else (GLM-4.7-Flash #23: forge-models has full 358B GLM-4.7, not the 31B Flash) |

## Prioritization Tiers

### Tier 1 - High Priority (do first)
Highest downloads, broadest user base, fits BH QB2:
- #37 Nemotron-3-Nano-30B (1,568K dl/mo, 30B MoE)
- #1 DeepSeek-R1-Distill-Qwen-32B (1,026K dl/mo, dense 32B)
- #4 DeepSeek-V2-Lite-Chat (826K dl/mo, 16B MoE)
- #23 GLM-4.7-Flash (770K dl/mo, 31B MoE)
- #2 DeepSeek-R1-Distill-Qwen-14B (568K dl/mo, dense 14B)
- #5 Mixtral-8x7B-Instruct (530K dl/mo, 47B MoE)
- #29 DeepSeek-Coder-V2-Lite (431K dl/mo, 16B MoE)
- #38 Nemotron-Cascade-2-30B (314K dl/mo, 30B MoE)
- #13 Nemotron-Super-49B (203K dl/mo, NVIDIA partnership value)

### Tier 2 - Medium Priority
Important for ecosystem coverage and competitive positioning:
- #36 Nemotron-3-Super-120B (500K dl/mo, Galaxy)
- #3 DeepSeek-R1-Distill-Llama-70B (153K dl/mo, dense 70B)
- #14 OpenReasoning-Nemotron-32B (132K dl/mo)
- #32 Phi-3.5-MoE (103K dl/mo, MoE validation -- already in forge-models)
- #56 InternVL3-38B (91K dl/mo, top VLM)
- #12 Baichuan-M2-32B (84K dl/mo)
- #10 EXAONE-3.5-32B (77K dl/mo)
- #62 Kimi-Linear-48B (61K dl/mo, novel linear attention)
- #34 HyperCLOVA-X-32B (61K dl/mo)
- #21 Llama-2-70b-chat (63K dl/mo, classic model)
- #58 NVLM-D-72B (63K dl/mo, NVIDIA VLM)
- #33 Phi-3-medium-14B (29K dl/mo, Microsoft ecosystem)
- #55 SOLAR-Pro (21K dl/mo, Korean-English reasoning)
- #17 Jamba-Large (2.5K dl/mo, novel Mamba+Transformer hybrid)

### Tier 3 - Ecosystem Completeness
Coverage for competitive positioning, lower urgency:
- Galaxy-only large models (#6, 8, 18, 22, 24, 27, 40, 46)
- Code models (#30-31, 47-49)
- Older/niche reasoning (#39, 41-45, 50-53)
- Multi-modal (#57, 59-61, 64-65)
- Specialized (#28, 63, 66)
