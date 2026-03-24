import model, gpu, bpe, autograd, gguf
import std/[strformat, os]

when isMainModule:
  gpuInit()
  let baseDir = getAppDir().parentDir()
  let vidyaRoot = baseDir.parentDir()
  let tok = loadTokenizer(vidyaRoot / "tokenizer_nim.bin")
  var m = initModel(tok.vocab.len, withGradients=false, withWeights=false)
  loadModelGguf(m, paramStr(1))

  trackingEnabled = false  # no arena, use individual allocs

  echo ""
  echo "=== LLaMA 1B Forward Pass Verification ==="

  # Test 1: Embedding for token 128000
  var tokId = [int32(128000)]
  discard cudaMemcpy(m.tokIdBuf, addr tokId[0], csize_t(4), CudaMemcpyHostToDevice)
  let x = gpuCreate(1 * nEmbd)
  gpu_embed_fwd(m.wte.data, m.tokIdBuf, x.data, cint(1), cint(nEmbd))
  let emb = gpuDownload(x)
  echo "1. Embedding[:5]:  ", emb[0], " ", emb[1], " ", emb[2], " ", emb[3], " ", emb[4]
  echo "   Ref:            0.00269  0.00308 -0.00681  0.04199 -0.00266"

  # Test 2: RMSNorm
  let xNorm = gpuCreate(1 * nEmbd)
  let rms = gpuCreate(1)
  gpu_rmsnorm_affine_fwd(x.data, m.layers[0].ln1g.data, xNorm.data, rms.data, cint(1), cint(nEmbd))
  let normed = gpuDownload(xNorm)
  echo "2. RMSNorm[:5]:    ", normed[0], " ", normed[1], " ", normed[2], " ", normed[3], " ", normed[4]

  # Test 3: Q8_0 matvec for Q projection
  let q = gpuCreate(1 * nEmbd)
  gpu_matvec_q8_0(m.layers[0].wq_q4, xNorm.data, q.data, cint(nEmbd), cint(nEmbd))
  let qv = gpuDownload(q)
  echo "3. Q proj[:5]:     ", qv[0], " ", qv[1], " ", qv[2], " ", qv[3], " ", qv[4]

  # Test 4: Float32 matmul for Q projection (reference, using dequanted weights)
  # We don't have float32 Q weights for GGUF, so skip this
  echo ""
  echo "Compare with Python reference to check if values match."
