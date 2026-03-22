# nimllm

A grockable language model. No black boxes except the matrix multiplier.

One binary. Trains on your data. Chats with you. Reads books. Grows.
Runs entirely on your GPU. You own every line of code and every weight.

## Quick Start

```bash
# Build (needs CUDA toolkit + Nim)
cd src && nvcc -c -O2 -Xcompiler -fPIC -o kernels.o kernels.cu
cd .. && nim c -d:release src/microgpt.nim
nim c -d:release src/chat.nim

# Train
./src/microgpt

# Chat
./src/chat

# Feed it a book
./src/microgpt --read ~/data/books/meditations.txt

# Grow from a smaller model
./src/microgpt --grow nimllm_old.bin
```

## Architecture

95M parameters. 12 layers, 768 dim, 12 heads. Learned positional embeddings.
BPE tokenizer (4259 vocab, trained on your data). Everything in ~600 lines of Nim.

```
microgpt.nim    — training: forward, backward, Adam, checkpoint save/load
chat.nim        — interactive generation with sampling
gpu.nim         — CUDA bindings: cuBLAS, memory, kernel declarations
kernels.cu      — CUDA kernels: softmax, GELU, RMSNorm, embedding, etc.
bpe.nim         — byte pair encoding tokenizer
autograd.nim    — scratch arena allocator (no autograd graph used)
```

## How It Works

**Forward pass:** Token embedding → position embedding → 12 transformer layers
(RMSNorm → QKV projection → cuBLAS attention → output projection → RMSNorm →
FFN) → final norm → logits → log-softmax → loss.

**Backward pass:** Explicit reverse. No autograd graph, no closures, no
topological sort. Each operation saves what backward needs. Backward runs in
reverse order. You can read it top to bottom.

**Attention:** cuBLAS matmuls for QK^T and probs@V. Custom kernels only for
softmax and causal masking. 7x faster than per-thread flash attention.

**Memory management:** Scratch arena — one big GPU allocation at startup,
bump-allocated per step, zeroed on reset. Zero cudaMalloc calls in the hot loop.

## Growing the Model

nimllm is designed to grow incrementally:

1. Train at current size until loss plateaus
2. Bump dimensions/layers in the config constants
3. `./src/microgpt --grow nimllm.bin` — loads old weights into the bigger model
4. Continue training — old knowledge preserved, new capacity learns

This is how you build a personal model over weeks and months on a single GPU.
No cloud. No API. No one else's weights.

## Hardware

Designed for NVIDIA GPUs with 8-12GB VRAM. Tested on RTX 3060 12GB.

| Model Size | Dim  | Layers | Heads | VRAM  | Speed     |
|-----------|------|--------|-------|-------|-----------|
| 28M       | 512  | 8      | 8     | ~2GB  | 7 opt/s   |
| 95M       | 768  | 12     | 12    | ~4GB  | 18 opt/s  |
| 200M      | 1024 | 16     | 16    | ~8GB  | ~10 opt/s |

## Inspired By

Andrej Karpathy's [microGPT](https://github.com/karpathy/nanoGPT) — the idea
that you can build a real language model in one file, understand every line,
and train it on your own hardware.

## License

MIT
