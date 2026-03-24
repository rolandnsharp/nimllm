# NimLLM TODO

## Current Priority: Bootstrap from SmolLM2-135M

1. Download SmolLM2-135M weights from HuggingFace
2. Update constants to match (30L, 576d, 9h, 3kv, ffn=1536, vocab=49152, rope_theta=100000)
3. Update chat.nim to match
4. Convert weights with load_hf.py
5. Convert their tokenizer to our BPE format
6. Test chat — should be instantly conversational
7. Fine-tune on our data (books + curriculum + distilled Q&A) with `nimllm continue`
8. Grow to clean dimensions (24L, 768d, 12h, 4kv) with `nimllm grow`
9. Continue training at new dimensions

## Data Pipeline

- [x] 207 books from Project Gutenberg (120MB)
- [x] 37K chat conversations (25MB)
- [x] 32 curriculum files: Q&A, philosophy, science, history, coding, wisdom (1.7MB)
- [x] Book of nimllm (self-knowledge)
- [x] nimllm source code as training data
- [x] Thinking in Forth by Leo Brodie
- [x] Nim language manual + tutorials
- [x] Karpathy blogs, Turing paper, transformer science
- [x] 500 docs distilled from Qwen3.5:9b (1.6MB)
- [x] Claude Code conversation logs
- [ ] More distillation (5000+ docs overnight when GPU free)
- [ ] Nim compiler source code (~200K lines of idiomatic Nim)
- [ ] Wikipedia Simple English dump
- [ ] The formal language spec (Forth + classical grammar)

## Architecture — Done

- [x] RoPE positional encoding
- [x] RMSNorm
- [x] SwiGLU activation
- [x] GQA (grouped query attention)
- [x] cuBLAS attention (fast matmuls)
- [x] Scratch arena allocator
- [x] GPU CE loss + backward kernels
- [x] Softmax power-of-2 fix
- [x] Self-describing checkpoint format (v2 header)
- [x] Elastic weight consolidation for --read mode
- [x] Progressive model growing (--grow)
- [x] Constant LR continue mode (--continue)
- [x] Non-interactive chat (--prompt)
- [x] BPE tokenizer with atomic special tokens
- [x] HuggingFace weight loader (load_hf.py)
- [x] Distillation pipeline (distill.py)
- [x] Training data builder (build_data.py)
- [x] nimllm CLI wrapper

## Performance

- [ ] Batch size > 1 (biggest single speedup)
- [ ] Kernel fusion (RMSNorm + projection)
- [ ] Skip unnecessary arena zeroing
- [ ] Currently ~12% GPU utilization, PyTorch gets ~50-60%

## Future

- [ ] Formal philosophical language (Forth + classical grammar)
- [ ] PicoCalc integration via serial port
- [ ] nimllm as coding agent (bash/CLI tool use)
- [ ] Active learning (model chooses what data to study)
- [ ] Self-improvement loop (read own code, suggest improvements)
- [ ] Scale to 400M+ params when 135M is maxed
- [ ] Tenstorrent Blackhole port
