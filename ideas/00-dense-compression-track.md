# Track: Dense Compression Transformers

## Core Thesis

The baseline processes N tokens through all layers at full resolution. We compress
tokens in the middle layers (encoder→bottleneck→decoder, like a vision U-Net), allowing:

1. **More layers in 16MB** — attention params scale with seq_len². Halving seq_len in
   middle layers cuts their attention weight count ~4×, freeing bytes for more depth.
2. **Richer per-position representations** — each compressed position encodes information
   from multiple tokens, increasing information density per forward-pass step.
3. **Better training efficiency** — fewer positions in expensive attention layers = more
   steps in 10 minutes, or deeper models at the same step count.

The baseline already uses U-Net skip connections (encoder half stores activations,
decoder half adds them back with learned weights). Our extension: **actually compress
the sequence length** between encoder and decoder halves.

## Baseline Architecture (for reference)

```
Embedding (vocab=1024, dim=512)  →  RMSNorm
│
├── Block 0-3  (encoder half)  ←  store skip activations
│     └── CausalSelfAttention(8 heads, 4 KV, GQA) + MLP(relu², 2× expand)
│
├── Block 4-8  (decoder half)  ←  add skip activations back
│     └── same structure
│
└── RMSNorm → tied lm_head → softcap → CE loss

Total: 9 layers, 512 dim, ~26M params pre-quantization, ~15MB after int8+zlib
Current SOTA: 10 layers, int5 MLP / int6 attn, ~1.1428 BPB
```

## Our Architecture Family

All variants share this skeleton:

```
Embedding → RMSNorm
│
├── Full-res encoder layers (2-3 layers, seq_len=N)
│     └── store skip activations at full resolution
│
├── COMPRESSION (N → N/R tokens, R = compression ratio)
│     └── [VARIANT: how we compress]
│
├── Compressed layers (4-8 layers, seq_len=N/R)
│     └── attention is R² cheaper per layer
│     └── can widen dim or add layers within same byte budget
│
├── DECOMPRESSION (N/R → N tokens)
│     └── [VARIANT: how we decompress]
│     └── add skip connections from encoder
│
├── Full-res decoder layers (1-2 layers, seq_len=N)
│
└── RMSNorm → lm_head → loss
```

### Variant A: Strided Conv Compression (CV-native)

**Compression**: 1D conv with kernel_size=R, stride=R (like strided conv in ResNet downsampling)
**Decompression**: transposed 1D conv (learnable upsampling)
**Skip connection**: full-res encoder output added after decompression

Pros: Simple, fast, very familiar from CV, minimal parameters.
Cons: Fixed receptive field, no content-awareness in what gets compressed.

### Variant B: Cross-Attention Compression (Perceiver-style)

**Compression**: N/R learned query tokens cross-attend to N full-res tokens
**Decompression**: N learned query tokens cross-attend to N/R compressed tokens
**Skip connection**: concatenate encoder features before decompression cross-attention

Pros: Content-aware, theoretically most expressive compression.
Cons: Cross-attention adds parameters and compute. Query tokens need position encoding.

### Variant C: Learned Pooling + Linear Upsample

**Compression**: reshape (B, N, D) → (B, N/R, R*D), then linear project R*D → D'
**Decompression**: linear project D' → R*D, then reshape back to (B, N, D)
**Skip connection**: additive at full resolution after reshape

Pros: Very parameter-efficient, no attention overhead. The linear layer learns
what combination of R adjacent tokens to keep.
Cons: Strictly local (only sees R adjacent tokens), positional info baked into reshape order.

### Variant D: Adaptive Compression (Deformable-Conv inspired)

**Compression**: learned scoring network predicts which token groups to merge vs keep.
Variable compression ratio across the sequence.
**Decompression**: scatter back to original positions + interpolate gaps.

Pros: Most expressive — preserves semantic boundaries, compresses redundant spans harder.
Cons: Most complex, variable-length tensors are hard to batch, may not be worth
the engineering for this challenge.

## Recommended Starting Order

1. **Variant C first** (learned pooling) — simplest, fastest to implement, easiest to
   ablate against baseline. If this shows signal, the thesis is validated.
2. **Variant A second** (strided conv) — slightly more expressive, still fast.
3. **Variant B if C/A show promise** — cross-attention compression for maximum quality.
4. **Variant D only if we have time** — engineering-heavy, research risk.

## Tokenizer Considerations

The compression architecture and tokenizer are **coupled**:

| Tokenizer choice | Tokens per doc | Compression benefit |
|-----------------|----------------|-------------------|
| sp1024 (baseline) | ~3.5 bytes/token → many tokens | Moderate — room to compress |
| sp256 or byte-level | ~1 byte/token → 3.5× more tokens | Maximum — compression is essential |
| sp4096+ | ~4.5 bytes/token → fewer tokens | Minimal — less redundancy to exploit |

**Key insight**: A smaller vocabulary produces more tokens with more local redundancy,
which our compression layers can exploit better. The byte/character level is where
this architecture wins hardest — but the baseline uses sp1024 and training byte-level
models is slower.

**Custom tokenizer strategy**: Design a tokenizer that maximizes local redundancy
patterns that the compression layers can learn to exploit:
- Smaller vocab (256-512) with token sequences that have predictable local structure
- Pair with higher compression ratio (8:1) since more tokens to compress
- Trade: more embedding bytes saved (smaller vocab) vs more tokens to process

**Risk**: Custom tokenizer submissions "will be examined more carefully" per the rules.
BPB calculation must be provably correct. Start with sp1024, add custom tokenizer
as a later optimization once the architecture is validated.

## 3070 Local Development Protocol

**Goal**: fast iteration loop that preserves relative architecture ranking.

The 3070 has 8GB VRAM. The baseline at full config uses ~5-6GB on a single GPU.
We can run a scaled-down version:

```bash
# Local smoke test config (~2-3 min on 3070)
RUN_ID=local_test \
ITERATIONS=500 \
TRAIN_BATCH_TOKENS=32768 \
VAL_BATCH_SIZE=32768 \
VAL_LOSS_EVERY=0 \
TRAIN_SEQ_LEN=512 \
MAX_WALLCLOCK_SECONDS=180 \
python3 train_gpt.py
```

**What local runs CAN tell you**:
- Does the architecture train stably? (loss decreasing, no NaN)
- Relative ranking: does variant A beat variant B at small scale?
- Parameter count verification: does the quantized artifact fit in 16MB?
- Compression/decompression correctness: are skip connections working?

**What local runs CANNOT tell you**:
- Absolute BPB numbers (won't match H100 runs)
- Whether the 10-minute budget is sufficient
- Multi-GPU scaling behavior

**Validation approach**: Train 2+ architectures with identical hyperparameters
on 3070, compare val_loss curves. The winner locally should directionally win
on H100. Promote to H100 only after local signal is positive.

## Parameter Budget Analysis

16MB = 16,000,000 bytes for code + compressed weights.

Code is ~50KB. So ~15,950,000 bytes for weights.

With int6 + zstd (SOTA compression): effective ratio ~1.5× from zstd on int6 data.
That gives us ~15.95M × 8/6 × 1.5 ≈ 31.9M parameter-equivalents at int6.

With int5 MLP + int6 attn (current #1): slightly more room.

**Compression architecture overhead**:
- Variant C (learned pooling): 2 linear layers × D × (R*D) ≈ 2 × 512 × 2048 = 2M params
  At int6+zstd: ~1MB overhead → ~6% of budget
- Variant A (strided conv): much smaller, <0.5MB
- The savings in attention layers should more than compensate

**Break-even calculation** for R=4 compression, 6 compressed layers:
- Saved per compressed layer: KV projections see N/4 positions → ~4× less attention compute
  But weight count is the same (weights don't depend on seq_len — only activations do)
- Wait — **attention weights don't change with seq_len**. Only activation memory and
  compute change. So the byte savings come from being able to reduce dim or remove
  layers, not from the compression itself.

**Correction**: The real value proposition is:
1. Compressed layers can process 4× more context in the same compute budget
2. OR: with compute savings, train more steps → lower loss
3. The compression/decompression layers ARE the overhead — but they replace
   embeddings worth of parameters if we go to smaller vocab

This needs empirical validation. Variant C is the cheapest test.

## Open Questions

- [ ] Does causal masking work correctly after compression? If we compress 4 tokens
      into 1, the compressed token represents future tokens — potential information leakage.
      Need to verify causal structure is preserved.
- [ ] How does RoPE interact with compressed positions? Position IDs in compressed
      space are 4× sparser — do we re-index, or use the original positions?
- [ ] Does the Muon optimizer handle compression/decompression layers well?
      These are small dense matrices, not the typical large transformer weights.
- [ ] What's the minimum number of full-res layers needed at the end for the
      lm_head to produce good per-token predictions?
