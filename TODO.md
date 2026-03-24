# NimLLM TODO

## Vision

LLaMA in Nim. Like llama.cpp but readable, trainable, and growable.
Inference and training share one forward pass. The model learns from
every interaction. Weights are memory.

## Current Priority: Clean Architecture

1. Move backward pass from microgpt.nim into model.nim (forward + backward together)
2. microgpt.nim becomes just: optimizer + training loop
3. chat.nim becomes just: sampling + generation loop
4. model.nim becomes the complete model: types, init, forward, backward, save/load

```
model.nim      — the model (forward + backward + save/load)
chat.nim       — inference (imports model)
train.nim      — training loop (imports model)
kernels.cu     — GPU operations
gpu.nim        — CUDA bindings
bpe.nim        — tokenizer
autograd.nim   — scratch arena
```

## Next: Performance

- [ ] Batched GEMM for attention heads (one call instead of 9)
- [ ] Fused QKV projection (one matmul instead of 3)
- [ ] KV cache for fast inference (don't recompute full sequence every token)
- [ ] Currently ~24% GPU utilization, target 50%+

## Next: Memory Through Training

- [ ] Conversation → weight updates (the core differentiator vs llama.cpp)
- [ ] Active learning: model identifies what it doesn't know, requests data
- [ ] Sleep cycle: consolidate day's interactions into base weights
- [ ] Elastic pull tuning for book absorption

## Data

- [x] Pre-tokenized binary loading (49M tokens, 1.4s startup)
- [x] 207 books from Project Gutenberg
- [x] 32 curriculum files (Q&A, philosophy, science, history, Forth, Nim)
- [x] Book of nimllm (self-knowledge)
- [x] Distilled Q&A from Qwen3.5
- [x] SmolLM2-135M-Instruct weights loaded and verified
- [ ] More distillation (5000+ docs)
- [ ] Nim compiler source code
- [ ] Formal language spec (Forth + classical grammar)

## Architecture — Done

- [x] RoPE, RMSNorm, SwiGLU, GQA, cuBLAS attention
- [x] Scratch arena allocator
- [x] GPU CE loss + backward kernels
- [x] Elastic weight consolidation for --read
- [x] Progressive model growing (--grow)
- [x] HuggingFace weight loader (load_hf.py)
- [x] Greedy longest-match tokenizer for pre-trained vocabs
- [x] ChatML formatting with system prompt
- [x] Pre-tokenized binary data
- [x] Shared forward pass (model.nim)
- [x] Sequence packing (batch 8)

## Future

- [ ] Formal philosophical language (Forth + classical grammar)
- [ ] PicoCalc integration via serial port
- [ ] nimllm as coding agent (bash/CLI tool use)
- [ ] Self-improvement loop
- [ ] Scale to 400M+ when 135M is maxed
- [ ] GGUF format support (load Ollama models directly)
