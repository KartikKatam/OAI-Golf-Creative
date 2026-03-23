# Session Handoff — 2026-03-23

## Project

OpenAI Parameter Golf challenge — creative architecture track submission.
Repo: https://github.com/KartikKatam/OAI-Golf-Creative
Forked from: https://github.com/openai/parameter-golf

## Challenge Constraints

- **16MB artifact** (code + zstd-compressed weights)
- **10 min training** on 8×H100 SXM
- **10 min eval** budget (separate)
- **Metric**: val_bpb (bits per byte) on FineWeb validation set — lower is better
- No network calls or external data at eval. Self-contained artifact.
- Custom tokenizers allowed but scrutinized carefully for correct BPB calculation.
- Baseline: 1.2244 BPB. Current SOTA: 1.1428 BPB (as of 2026-03-20).

## What Exists

- `ideas/00-dense-compression-track.md` — Full architecture track document with:
  - 4 compression variants (C → A → B → D, ordered by implementation priority)
  - Tokenizer coupling analysis
  - 3070 local dev protocol (smoke test config)
  - Parameter budget analysis
  - Open questions list
- Baseline code: `train_gpt.py` (CUDA/torchrun), `train_gpt_mlx.py` (Apple Silicon)
- Data pipeline: `data/cached_challenge_fineweb.py` downloads tokenized FineWeb shards
- Records: 17 leaderboard submissions in `records/track_10min_16mb/` for reference

## Architecture Direction

**Dense Compression Transformers** — compress tokens in middle layers (encoder→bottleneck→decoder), inspired by CV encoder-decoder / U-Net patterns. The baseline already uses U-Net skip connections; our extension compresses the actual sequence length between encoder and decoder halves.

### Key Variants (priority order)

1. **Variant C (Learned Pooling)**: reshape (B,N,D)→(B,N/R,R*D), linear project R*D→D'. Simplest, cheapest to implement. Start here.
2. **Variant A (Strided Conv)**: 1D conv kernel_size=R, stride=R. Slightly more expressive.
3. **Variant B (Cross-Attention)**: Perceiver-style learned queries. Most expressive but parameter-heavy.
4. **Variant D (Adaptive)**: Content-aware variable compression. Research risk, last priority.

### Critical Insight Discovered

Attention **weight** count doesn't depend on seq_len — only activations and compute do. So compression doesn't directly free bytes for more layers within the 16MB budget. The value proposition is:
- **Compute savings** → more training steps in 10 min → lower loss
- **Information density** → each compressed position carries R tokens of context
- **Byte savings come indirectly** via smaller vocab (fewer embedding params) paired with compression to handle the resulting longer token sequences

This makes the **tokenizer + compression co-design** the real creative angle:
- Smaller vocab (256-512) saves embedding bytes, produces more tokens
- Compression architecture handles the longer sequences efficiently
- Net effect: more of the 16MB budget goes to transformer layers, less to embeddings

## Current SOTA Techniques (from leaderboard)

- Int5/int6 QAT (quantization-aware training)
- BigramHash embeddings (hash consecutive token pairs into embedding table)
- SmearGate (learned gating)
- SWA (stochastic weight averaging of late checkpoints)
- Sliding window eval at stride=64
- 3× MLP expansion with relu²
- Muon optimizer (spectral gradient normalization)
- U-Net skip connections (already in baseline)
- Test-Time Training with LoRA (creative entry, not SOTA but novel)

## Open Questions (unresolved)

1. **Causal masking after compression**: compressing 4 tokens into 1 — does the compressed token leak future information? Need to verify causal structure is preserved.
2. **RoPE in compressed space**: position IDs become 4× sparser. Re-index or use original positions?
3. **Muon optimizer on compression layers**: these are small dense matrices, not typical large transformer weights. May need different LR group.
4. **Minimum full-res decoder layers**: how many layers after decompression before the lm_head can produce good per-token predictions?

## Next Steps

1. **Download data**: `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1` (small subset for local dev)
2. **Run baseline on 3070**: verify it trains, get local BPB reference number
3. **Implement Variant C** (learned pooling): modify `GPT.forward()` to add compression/decompression around the middle blocks
4. **Local A/B test**: Variant C vs baseline on 3070, same hyperparameters, compare val_loss curves
5. **If signal is positive**: iterate on compression ratio, layer split, then add SOTA tricks (int6 QAT, SWA, etc.)

## Dev Environment

- Local: RTX 3070 (8GB VRAM), Linux
- Remote: RunPod 8×H100 SXM (for final validation runs)
- Python venv at `.venv/` (not created yet locally)
- Data not yet downloaded locally

## User Context

Senior engineer with deep CV + Python/Rust background. Approaching this from a computer vision encoder-decoder perspective. Interested in novel architecture research, not just systems optimization. Targeting the creative architecture track specifically.
