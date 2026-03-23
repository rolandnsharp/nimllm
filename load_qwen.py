#!/usr/bin/env python3
"""Convert Qwen3 safetensors weights to nimllm checkpoint format.

Reads Qwen3-0.6B (or any Qwen3) safetensors weights and writes them
in nimllm's v2 checkpoint format (NLLM header + weight buffers).

Usage:
  python3 load_qwen.py ~/data/models/qwen3-0.6b nimllm_qwen.bin

The output checkpoint can be loaded by nimllm's chat binary if the
architecture constants match (nLayer, nEmbd, nHead, nKvHead, etc).
"""

import struct, json, os, sys, glob
import numpy as np

def read_safetensors(path):
    """Read a safetensors file, return dict of name → numpy array."""
    with open(path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_size))
        data_start = 8 + header_size
        tensors = {}
        for name, info in header.items():
            if name == '__metadata__':
                continue
            dtype_str = info['dtype']
            shape = info['data_offsets']
            start, end = shape[0], shape[1]
            f.seek(data_start + start)
            raw = f.read(end - start)
            if dtype_str == 'BF16':
                # BF16 → float32: pad each 2-byte bf16 with 2 zero bytes
                arr = np.frombuffer(raw, dtype=np.uint16)
                f32 = np.zeros(len(arr), dtype=np.float32)
                f32.view(np.uint32)[:] = arr.astype(np.uint32) << 16
                tensors[name] = f32.reshape(info.get('shape', [-1]))
            elif dtype_str == 'F16':
                tensors[name] = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(info.get('shape', [-1]))
            elif dtype_str == 'F32':
                tensors[name] = np.frombuffer(raw, dtype=np.float32).reshape(info.get('shape', [-1]))
            else:
                print(f"  skip {name}: unsupported dtype {dtype_str}")
        return tensors

def main():
    if len(sys.argv) < 3:
        print("usage: load_qwen.py <model_dir> <output.bin>")
        print("  model_dir: directory containing safetensors + config.json")
        print("  output.bin: nimllm checkpoint to write")
        sys.exit(1)

    model_dir = sys.argv[1]
    output = sys.argv[2]

    # Read config
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    n_layer = config['num_hidden_layers']
    n_embd = config['hidden_size']
    n_head = config['num_attention_heads']
    n_kv_head = config['num_key_value_heads']
    head_dim = n_embd // n_head
    n_kv_dim = n_kv_head * head_dim
    ffn_dim = config['intermediate_size']
    vocab_size = config['vocab_size']
    block_size = min(config.get('max_position_embeddings', 32768), 32768)

    print(f"Qwen3 config:")
    print(f"  layers={n_layer} embd={n_embd} heads={n_head} kv_heads={n_kv_head}")
    print(f"  head_dim={head_dim} ffn_dim={ffn_dim} vocab={vocab_size}")

    # Read all safetensors files
    st_files = sorted(glob.glob(os.path.join(model_dir, '*.safetensors')))
    if not st_files:
        print(f"No safetensors files found in {model_dir}")
        sys.exit(1)

    print(f"Reading {len(st_files)} safetensors files...")
    all_tensors = {}
    for sf in st_files:
        print(f"  {os.path.basename(sf)}")
        tensors = read_safetensors(sf)
        all_tensors.update(tensors)

    print(f"Loaded {len(all_tensors)} tensors")

    # Map Qwen tensor names to nimllm weight order
    # Qwen3 naming: model.embed_tokens.weight, model.layers.N.self_attn.{q,k,v,o}_proj.weight, etc.

    def get(name):
        if name in all_tensors:
            return all_tensors[name].flatten().astype(np.float32)
        print(f"  WARNING: {name} not found!")
        return None

    def write_buf(f, arr):
        """Write int32(count) + float32[count]"""
        f.write(struct.pack('<i', len(arr)))
        f.write(arr.tobytes())

    with open(output, 'wb') as f:
        # v2 header
        f.write(b'NLLM')
        f.write(struct.pack('<i', 2))          # version
        f.write(struct.pack('<i', n_layer))
        f.write(struct.pack('<i', n_embd))
        f.write(struct.pack('<i', n_head))
        f.write(struct.pack('<i', vocab_size))
        f.write(struct.pack('<i', block_size))

        # wte (token embeddings)
        wte = get('model.embed_tokens.weight')
        write_buf(f, wte)
        print(f"  wte: {wte.shape}")

        # lmHead (output projection) — Qwen ties weights, use embed if no lm_head
        lm_head = get('lm_head.weight')
        if lm_head is None:
            print("  lm_head not found, using tied embed_tokens")
            lm_head = wte
        write_buf(f, lm_head)

        # lnFg (final RMSNorm gamma)
        ln_fg = get('model.norm.weight')
        write_buf(f, ln_fg)
        print(f"  lnFg: {ln_fg.shape}")

        # Layers
        for i in range(n_layer):
            prefix = f'model.layers.{i}'
            wq = get(f'{prefix}.self_attn.q_proj.weight')
            wk = get(f'{prefix}.self_attn.k_proj.weight')
            wv = get(f'{prefix}.self_attn.v_proj.weight')
            wo = get(f'{prefix}.self_attn.o_proj.weight')
            fc_gate = get(f'{prefix}.mlp.gate_proj.weight')
            fc_up = get(f'{prefix}.mlp.up_proj.weight')
            fc_down = get(f'{prefix}.mlp.down_proj.weight')
            ln1g = get(f'{prefix}.input_layernorm.weight')
            ln2g = get(f'{prefix}.post_attention_layernorm.weight')

            write_buf(f, wq)
            write_buf(f, wk)
            write_buf(f, wv)
            write_buf(f, wo)
            write_buf(f, fc_gate)
            write_buf(f, fc_up)
            write_buf(f, fc_down)
            write_buf(f, ln1g)
            write_buf(f, ln2g)

            if (i + 1) % 10 == 0 or i == n_layer - 1:
                print(f"  layer {i+1}/{n_layer}")

    size_mb = os.path.getsize(output) // (1024 * 1024)
    print(f"\nWrote {output} ({size_mb}MB)")
    print(f"\nTo use with nimllm, set these constants in microgpt.nim:")
    print(f"  nLayer   = {n_layer}")
    print(f"  nEmbd    = {n_embd}")
    print(f"  nHead    = {n_head}")
    print(f"  nKvHead  = {n_kv_head}")
    print(f"  blockSize = {block_size}")
    print(f"  ffnMul   = {ffn_dim // n_embd}  # (intermediate_size / hidden_size)")

if __name__ == "__main__":
    main()
