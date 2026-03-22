# NimLLM TODO

## Done
- [x] Flash attention (forward + backward) — numerically stable, zero NaN
- [x] cuBLAS attention — 7x faster than per-thread flash attention
- [x] Scratch arena allocator — eliminates ~300 cudaMalloc/free per step
- [x] Pre-allocated token/position ID buffers
- [x] BPE encode optimization: hash table lookup (60s → 3.9s)
- [x] `nimllm chat` — interactive conversation mode
- [x] `nimllm read` — absorb documents into weights (--read flag)
- [x] Checkpoint save/load every 10K steps
- [x] Softmax kernel power-of-2 fix (parallel reduction bug)
- [x] TF32 tensor cores enabled for cuBLAS matmuls

## Performance
- [ ] Batch multiple short sequences to fill GPU (currently 1 doc per step)
- [ ] Fuse embedding lookup + position add into one kernel
- [ ] Move dLogits computation to GPU (currently CPU-side exp + upload)
- [ ] Consider weight tying (lm_head = wte^T) — saves 1.2M params, reduces memory

## Architecture (for 100M scale-up)
- [ ] SwiGLU activation (kernel ready, not wired in)
- [ ] GQA — grouped query attention (kernel ready, not wired in)
- [ ] Dropout (kernel ready, not wired in)
- [ ] RoPE instead of learned positional embeddings
- [ ] Larger vocab (8K+ merges)
- [ ] Progressive model growing: expand dims/layers preserving weights

## Memory Mechanism
- [ ] Sparse gradient masking: top 1% of gradients per interaction
- [ ] Elastic weight consolidation: pull toward base weights during --read
- [ ] Interactive RL: retrain on chosen/rejected chat responses
- [ ] Sleep cycle: consolidate day's learning into base weights

## Books to Feed
- [x] The Unknown God
- [x] KJV New Testament
- [ ] Beyond Good and Evil (Nietzsche) — downloaded, ready
- [ ] Meditations (Marcus Aurelius) — downloaded, ready
- [ ] Thus Spoke Zarathustra (Nietzsche) — downloaded, ready
- [ ] The Republic (Plato) — downloaded, ready
- [ ] The Prince (Machiavelli) — downloaded, ready
- [ ] The Art of War (Sun Tzu) — downloaded, ready
- [ ] Walden (Thoreau) — downloaded, ready
- [ ] Frankenstein (Shelley) — downloaded, ready
- [ ] Alice in Wonderland (Carroll) — downloaded, ready
- [ ] On the Origin of Species (Darwin) — downloaded, ready
- [ ] Leviathan (Hobbes) — downloaded, ready

## Deployment
- [ ] Single binary packaging (embed tokenizer)
- [ ] Safetensors export
- [ ] Tenstorrent Blackhole port (Nim → C++ → TT-Metalium)
