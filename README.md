# nimllm

LLaMA in Nim. Readable. Trainable. Growable.

Like llama.cpp but you can read it like Python, train it on your books,
and grow it over time. Inference and training in one codebase — the model
learns from every interaction.

## What It Is

A complete LLM implementation in ~2000 lines of Nim + CUDA. No PyTorch,
no frameworks, no dependencies beyond the CUDA toolkit. Loads pre-trained
weights from HuggingFace (SmolLM, LLaMA, Mistral, Qwen — anything
LLaMA-compatible). Trains on your data. Grows when you need more capacity.

```bash
nimllm chat                        # talk to it
nimllm prompt "what is Forth?"     # one-shot question
nimllm continue                    # keep training on new data
nimllm read meditations.txt        # absorb a book without forgetting
nimllm grow old_model.bin          # expand to bigger dimensions
nimllm build                       # compile from source
```

## Why Not llama.cpp?

llama.cpp is an inference engine — it loads frozen weights and generates
text. It cannot learn. It cannot grow. It cannot absorb a book you give it.

nimllm's weights are alive. Every conversation, every book, every training
pass changes them. The model accumulates knowledge over time like a mind
accumulates experience. Inference and training are not separate programs —
they share the same forward pass because learning IS inference plus feedback.

## Architecture

LLaMA-compatible transformer. Same ideas as LLaMA, Mistral, Qwen:

- **RoPE** positional encoding
- **RMSNorm** (not LayerNorm)
- **SwiGLU** activation (not GELU)
- **GQA** grouped query attention
- **cuBLAS** matmuls for attention

Currently running SmolLM2-135M (30 layers, 576 dim, 9 heads, 3 KV heads).
Can load any model with this architecture by changing 6 constants.

## Code Structure

```
src/model.nim      shared: types, init, forward pass, save/load
src/chat.nim       inference: sampling, generation, chat loop
src/microgpt.nim   training: backward pass, optimizer, training loop
src/gpu.nim        CUDA bindings: cuBLAS, memory, kernel declarations
src/kernels.cu     GPU kernels: softmax, SwiGLU, RMSNorm, RoPE, GQA
src/bpe.nim        tokenizer: BPE + greedy longest-match for pre-trained vocabs
src/autograd.nim   scratch arena allocator
```

The forward pass lives in `model.nim` and is shared by both inference and
training. No code duplication. Change the architecture once, both paths
update.

## The Growth Loop

```
1. Load pre-trained weights (SmolLM, LLaMA, etc.)
2. Chat — test what it knows
3. Feed it data (books, Q&A, your conversations)
4. nimllm continue — train on the new data
5. Chat — it's better now
6. Repeat
7. When capacity maxes out: nimllm grow → bigger model, preserved knowledge
```

## Memory Through Training

The core idea: instead of a context window that forgets, nimllm moves
knowledge into weights. Read a book → weights change permanently.
Have a conversation → the important parts get trained into the model.
Context is temporary. Weights are memory.

`nimllm read` uses elastic weight consolidation — current weights are
saved as an anchor, and during training the model is gently pulled back
toward the anchor. New knowledge is absorbed without destroying old knowledge.

## Loading Pre-Trained Models

nimllm can load any LLaMA-compatible model from HuggingFace:

```bash
# Download model
python3 -c "from huggingface_hub import snapshot_download; \
  snapshot_download('HuggingFaceTB/SmolLM2-135M-Instruct', \
  local_dir='models/smollm2')"

# Convert to nimllm format
python3 load_hf.py models/smollm2 nimllm.bin

# Convert tokenizer
python3 -c "..."  # (see TODO for tokenizer conversion script)

# Chat
nimllm chat
```

## Training Data

Pre-tokenize once (fast), train many times:

```bash
python3 pretokenize.py              # 49M tokens → binary in 91 seconds
nimllm train                        # loads binary in 1.4 seconds
```

The training set includes:
- Chat conversations (dialogue patterns)
- 207 books from Project Gutenberg (language depth)
- Curriculum: Q&A, philosophy, science, history, Forth, Nim (knowledge)
- Distilled data from Qwen3.5 (reasoning quality)
- The Book of nimllm (self-knowledge)

## Hardware

NVIDIA GPU with 8-12GB VRAM. Tested on RTX 3060 12GB.
Inference: any model that fits in VRAM.
Training: models up to ~200M params at full precision.

## Inspired By

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — inference engine we're extending with training
- [nanoGPT](https://github.com/karpathy/nanoGPT) — the seed: one-file GPT you can understand
- [llm.c](https://github.com/karpathy/llm.c) — pre-tokenization, speed, C-level simplicity
- The LLaMA architecture — RoPE, RMSNorm, SwiGLU, GQA
- Alan Turing: "produce a programme which simulates the child's mind,
  then subject it to an appropriate course of education"

## License

MIT
