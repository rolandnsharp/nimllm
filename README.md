# nimllm

A grockable language model. No black boxes except the matrix multiplier.

One binary. Trains on your data. Chats with you. Reads books. Grows.
Runs entirely on your GPU. You own every line of code and every weight.

## Quick Start

```bash
# Build
nimllm build

# First training run (cosine LR schedule with warmup)
nimllm train

# Chat
nimllm chat

# Single prompt (non-interactive)
nimllm prompt "what is the meaning of life?"
```

## The Growth Loop

nimllm is not a one-shot training run. It's a living process.

```bash
# 1. Train from scratch on your data
nimllm train

# 2. Test it
nimllm chat

# 3. Add better data (books, distilled Q&A, curriculum)
nimllm data

# 4. Continue training from checkpoint (constant low LR, no disruption)
nimllm continue

# 5. Test again — it's better now
nimllm chat

# 6. Repeat forever
```

Each pass the model gets better. Each pass the data gets better. They co-evolve.

## Commands

```
nimllm train              First run: full cosine LR schedule with warmup
nimllm continue           Every run after: constant low LR, refines from checkpoint
nimllm chat               Interactive conversation
nimllm prompt "..."       Single question, prints answer, exits
nimllm read <file>        Absorb a text file (elastic pull prevents forgetting)
nimllm grow <checkpoint>  Expand model from smaller checkpoint
nimllm data               Rebuild training_data.txt from all sources
nimllm feed               Feed all books sequentially
nimllm build              Compile CUDA kernels + Nim binaries
nimllm status             Show model info and training status
nimllm distill            Generate synthetic Q&A from teacher model via Ollama
```

## Architecture

91M parameters. LLaMA-compatible architecture:

- **12 layers, 768 dim, 12 query heads, 4 KV heads (GQA)**
- **RoPE** positional encoding (no learned position embeddings)
- **RMSNorm** (not LayerNorm)
- **SwiGLU** activation (not GELU)
- **cuBLAS attention** — fast matmuls for QK^T and probs@V
- **BPE tokenizer** (4259 vocab, trained on your data, atomic special tokens)

Same ideas as LLaMA, Mistral, Qwen — but from scratch in ~1000 lines of Nim.

```
src/microgpt.nim  — training: forward, backward, Adam, checkpoint save/load
src/chat.nim      — interactive generation with sampling
src/gpu.nim       — CUDA bindings: cuBLAS, memory, kernel declarations
src/kernels.cu    — CUDA kernels: softmax, SwiGLU, RMSNorm, RoPE, GQA, etc.
src/bpe.nim       — byte pair encoding tokenizer
src/autograd.nim  — scratch arena allocator
```

## How It Works

**Forward pass:** Token embedding → 12 transformer layers (RMSNorm → QKV
projection → RoPE → GQA cuBLAS attention → output projection → RMSNorm →
SwiGLU FFN) → final norm → logits → log-softmax → loss.

**Backward pass:** Explicit reverse. No autograd graph, no closures, no
topological sort. Each operation saves what backward needs. Backward runs
in reverse order. You can read it top to bottom.

**Attention:** cuBLAS matmuls for QK^T and probs@V. Custom kernels for
softmax and causal masking. Grouped Query Attention — 4 KV heads serve
12 query heads.

**Memory management:** Scratch arena — one big GPU allocation at startup,
bump-allocated per step, zeroed on reset. Zero cudaMalloc calls in the
hot loop.

## Growing the Model

nimllm is designed to grow incrementally:

1. Train at current size until loss plateaus
2. Bump dimensions/layers in the config constants
3. `nimllm grow nimllm.bin` — loads old weights into the bigger model
4. `nimllm continue` — old knowledge preserved, new capacity learns

## Absorbing Books

```bash
nimllm read ~/data/books/meditations.txt
```

Elastic weight consolidation saves current weights as an anchor before
reading. After each optimizer step, weights are pulled 0.1% back toward
the anchor. The model absorbs new knowledge without forgetting what it
already knows.

## Distillation

Generate high-quality synthetic training data from a teacher model:

```bash
nimllm distill --teacher qwen3.5:9b --mode qa --max-docs 5000
nimllm data       # rebuild training set with distilled data included
nimllm continue   # train on enriched data
```

## Training Data

nimllm trains on everything mixed together:

- **Chat conversations** — teaches dialogue
- **207 books from Project Gutenberg** — philosophy, literature, science
- **Curriculum** — Karpathy's blogs, Nim manual, transformer science,
  Thinking in Forth, history of backpropagation, the Book of nimllm
- **Distilled Q&A** — synthetic data from Qwen3.5 teacher
- **Claude Code logs** — real programming conversations

All shuffled. The model learns all domains simultaneously. No catastrophic
forgetting because it never stops seeing any domain.

## Self-Knowledge

nimllm trains on its own source code and a document called the Book of
nimllm that describes what it is, how it works, who built the ideas it
uses, and why it exists. The model knows itself.

## Hardware

Designed for NVIDIA GPUs with 8-12GB VRAM. Tested on RTX 3060 12GB.

| Model Size | Dim  | Layers | Q Heads | KV Heads | Speed      |
|-----------|------|--------|---------|----------|------------|
| 91M       | 768  | 12     | 12      | 4        | 13.5 opt/s |

## Inspired By

- Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)
- The LLaMA architecture (RoPE, RMSNorm, SwiGLU, GQA)
- The Phi papers (quality data beats quantity)
- Alan Turing: "produce a programme which simulates the child's mind,
  then subject it to an appropriate course of education"

## License

MIT
