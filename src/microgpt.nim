## microgpt.nim — Karpathy's microGPT ported to Nim with GPU
##
## Explicit forward + backward. No autograd graph. No closures.
## Each operation saves what backward needs. Backward runs in reverse.
## Same structure as microgpt.py, but with GPU tensors.
##
## Start small (verify correctness), then scale up.

import gpu, bpe, autograd
import std/[math, random, strformat, os, streams, times]

# ── Configuration ─────────────────────────────────────────────────

const
  nLayer   = 12
  nEmbd    = 768
  nHead    = 12          # query heads
  nKvHead  = 4           # key/value heads (GQA: 3:1 ratio)
  headDim  = nEmbd div nHead  # 64
  kvRepeat = nHead div nKvHead  # 3
  nKvDim   = nKvHead * headDim  # 256
  blockSize = 512
  ffnMul   = 4
  ropeTheta = 500000.0f

# ── Model ─────────────────────────────────────────────────────────

type
  Layer = object
    wq: GpuBuf                    # Q projection [nEmbd, nEmbd]
    wk: GpuBuf                    # K projection [nKvDim, nEmbd] (GQA: fewer KV heads)
    wv: GpuBuf                    # V projection [nKvDim, nEmbd]
    wo: GpuBuf                    # output projection [nEmbd, nEmbd]
    fcGate: GpuBuf                 # SwiGLU gate [ffnMul*nEmbd, nEmbd]
    fcUp: GpuBuf                   # SwiGLU up [ffnMul*nEmbd, nEmbd]
    fcDown: GpuBuf                 # SwiGLU down [nEmbd, ffnMul*nEmbd]
    ln1g, ln2g: GpuBuf            # RMSNorm gamma [nEmbd]
    # Gradients
    dwq, dwk, dwv, dwo: GpuBuf
    dfcGate, dfcUp, dfcDown: GpuBuf
    dln1g, dln2g: GpuBuf

  Model = object
    wte: GpuBuf                    # token embeddings [vocab, nEmbd]
    lmHead: GpuBuf                 # output projection [vocab, nEmbd]
    lnFg: GpuBuf                   # final RMSNorm gamma [nEmbd]
    layers: seq[Layer]
    vocabSize: int
    # Gradients
    dwte, dlmHead: GpuBuf
    dlnFg: GpuBuf
    # Pre-allocated buffers
    tokIdBuf: pointer              # [blockSize] int32 on GPU — input tokens
    targetIdBuf: pointer           # [blockSize] int32 on GPU — target tokens (shifted by 1)
    # RoPE tables (precomputed, shared across all layers)
    ropeCos: GpuBuf                # [blockSize, headDim/2]
    ropeSin: GpuBuf                # [blockSize, headDim/2]

proc initModel(vocabSize: int): Model =
  randomize(42)
  let std = 0.02f

  proc randBuf(n: int, s: float32 = std): GpuBuf =
    var host = newSeq[float32](n)
    for i in 0 ..< n:
      let u1 = rand(1.0).float32
      let u2 = rand(1.0).float32
      host[i] = s * sqrt(-2f * ln(max(u1, 1e-10f))) * cos(2f * PI.float32 * u2)
    toGpu(host)

  result.vocabSize = vocabSize
  result.wte = randBuf(vocabSize * nEmbd)
  result.lmHead = randBuf(vocabSize * nEmbd)
  result.dwte = gpuCreate(vocabSize * nEmbd)
  result.dlmHead = gpuCreate(vocabSize * nEmbd)

  proc onesBuf(n: int): GpuBuf =
    var h = newSeq[float32](n)
    for i in 0 ..< n: h[i] = 1.0f
    toGpu(h)

  result.lnFg = onesBuf(nEmbd)
  result.dlnFg = gpuCreate(nEmbd)

  for i in 0 ..< nLayer:
    let resStd = std / sqrt(float32(2 * nLayer))
    result.layers.add(Layer(
      wq: randBuf(nEmbd * nEmbd),
      wk: randBuf(nKvDim * nEmbd),      # GQA: K is [nKvDim, nEmbd]
      wv: randBuf(nKvDim * nEmbd),      # GQA: V is [nKvDim, nEmbd]
      wo: randBuf(nEmbd * nEmbd, resStd),
      fcGate: randBuf(ffnMul * nEmbd * nEmbd),
      fcUp: randBuf(ffnMul * nEmbd * nEmbd),
      fcDown: randBuf(nEmbd * ffnMul * nEmbd, resStd),
      ln1g: onesBuf(nEmbd),
      ln2g: onesBuf(nEmbd),
      dwq: gpuCreate(nEmbd * nEmbd),
      dwk: gpuCreate(nKvDim * nEmbd),
      dwv: gpuCreate(nKvDim * nEmbd),
      dwo: gpuCreate(nEmbd * nEmbd),
      dfcGate: gpuCreate(ffnMul * nEmbd * nEmbd),
      dfcUp: gpuCreate(ffnMul * nEmbd * nEmbd),
      dfcDown: gpuCreate(nEmbd * ffnMul * nEmbd),
      dln1g: gpuCreate(nEmbd),
      dln2g: gpuCreate(nEmbd),
    ))

  # Pre-allocate token ID buffers on GPU
  discard cudaMalloc(addr result.tokIdBuf, csize_t(blockSize * sizeof(int32)))
  discard cudaMalloc(addr result.targetIdBuf, csize_t(blockSize * sizeof(int32)))

  # Precompute RoPE cos/sin tables: theta_f = pos / 10000^(2f/headDim)
  let halfDim = headDim div 2
  var cosTab = newSeq[float32](blockSize * halfDim)
  var sinTab = newSeq[float32](blockSize * halfDim)
  for pos in 0 ..< blockSize:
    for f in 0 ..< halfDim:
      let theta = float32(pos) / pow(ropeTheta, 2.0f * float32(f) / float32(headDim))
      cosTab[pos * halfDim + f] = cos(theta)
      sinTab[pos * halfDim + f] = sin(theta)
  result.ropeCos = toGpu(cosTab)
  result.ropeSin = toGpu(sinTab)

  # GQA: wq [n,n] + wk [kvd,n] + wv [kvd,n] + wo [n,n] + 3 SwiGLU FFN
  let total = vocabSize * nEmbd * 2 +
    nLayer * (2 * nEmbd * nEmbd + 2 * nKvDim * nEmbd + 3 * ffnMul * nEmbd * nEmbd)
  echo &"  params: {total}"

# ── Saved intermediates for backward ──────────────────────────────

type
  LayerCache = object
    x1: GpuBuf           # input to layer (for residual + rmsnorm backward)
    xNorm1: GpuBuf       # after rmsnorm1
    rms1: GpuBuf         # rms value for rmsnorm1 backward
    q, k, v: GpuBuf      # QKV projections
    attnOut: GpuBuf       # flash attention output
    x2: GpuBuf           # after first residual (for residual + rmsnorm backward)
    xNorm2: GpuBuf       # after rmsnorm2
    rms2: GpuBuf         # rms value for rmsnorm2 backward
    fc1Out: GpuBuf       # SwiGLU gate output (pre-activation)
    upOut: GpuBuf        # SwiGLU up output
    geluOut: GpuBuf      # SwiGLU output: swish(gate) * up

  ForwardCache = object
    embedded: GpuBuf      # token + position embeddings
    layerCaches: seq[LayerCache]
    xPreFinalNorm: GpuBuf # ORIGINAL input to final rmsnorm
    finalNormed: GpuBuf
    rmsF: GpuBuf
    logits: GpuBuf
    logProbs: GpuBuf      # log-softmax output

# ── Forward pass ──────────────────────────────────────────────────

proc forward(m: Model, tokens: seq[int32], seqLen: int): (ForwardCache, float32) =
  var cache: ForwardCache
  let n = nEmbd
  let S = seqLen

  # Token embedding (no positional embedding — RoPE handles position)
  var x = trackedCreate(S * n)
  discard cudaMemcpy(m.tokIdBuf, unsafeAddr tokens[0],
                     csize_t(S * sizeof(int32)), CudaMemcpyHostToDevice)
  gpu_embed_fwd(m.wte.data, m.tokIdBuf, x.data, cint(S), cint(n))
  cache.embedded = x

  # Transformer layers
  for li in 0 ..< nLayer:
    var lc: LayerCache
    let layer = m.layers[li]

    # Save input for residual + rmsnorm backward
    lc.x1 = trackedCreate(S * n)
    gpuCopy(x, lc.x1, S * n)
    lc.xNorm1 = trackedCreate(S * n)
    lc.rms1 = trackedCreate(S)
    gpu_rmsnorm_affine_fwd(x.data, layer.ln1g.data, lc.xNorm1.data, lc.rms1.data, cint(S), cint(n))

    # Q, K, V projections (GQA: K and V have fewer heads)
    lc.q = trackedCreate(S * n)
    lc.k = trackedCreate(S * nKvDim)
    lc.v = trackedCreate(S * nKvDim)
    gpuSgemm(2, S, n, n, lc.xNorm1, layer.wq, lc.q)
    gpuSgemm(2, S, nKvDim, n, lc.xNorm1, layer.wk, lc.k)
    gpuSgemm(2, S, nKvDim, n, lc.xNorm1, layer.wv, lc.v)

    # RoPE: rotate Q and K by position
    ropeFwd(lc.q, m.ropeCos, m.ropeSin, S, n, nHead, headDim)
    ropeFwd(lc.k, m.ropeCos, m.ropeSin, S, nKvDim, nKvHead, headDim)

    # GQA attention: each KV head serves kvRepeat Q heads
    let scale = 1.0f / sqrt(float32(headDim))
    lc.attnOut = trackedCreate(S * n)

    let qH = trackedCreate(S * headDim)
    let kH = trackedCreate(S * headDim)
    let vH = trackedCreate(S * headDim)
    let attnH = trackedCreate(S * headDim)

    let scores = trackedCreate(S * S)
    let probs = trackedCreate(S * S)

    for h in 0 ..< nHead:
      extractHead(lc.q, qH, h, S, n, headDim)
      # GQA: map Q head h to KV head h div kvRepeat
      let kvH = h div kvRepeat
      gpu_extract_kv_head(lc.k.data, kH.data, cint(h), cint(kvRepeat),
                          cint(S), cint(nKvDim), cint(headDim))
      gpu_extract_kv_head(lc.v.data, vH.data, cint(h), cint(kvRepeat),
                          cint(S), cint(nKvDim), cint(headDim))

      # scores = Q_h @ K_h^T  [S,hd] × [hd,S] → [S,S]
      gpuSgemm(2, S, S, headDim, qH, kH, scores)
      # Scale + causal mask (lower triangle × scale, upper → -1e10)
      causalMask(scores, scale, S)
      # Row-wise softmax → attention weights
      softmaxFwd(scores, probs, S, S)
      # output = probs @ V_h  [S,S] × [S,hd] → [S,hd]
      gpuSgemm(0, S, headDim, S, probs, vH, attnH)

      insertHead(attnH, lc.attnOut, h, S, n, headDim)

    # Output projection
    var projected = trackedCreate(S * n)
    gpuSgemm(2, S, n, n, lc.attnOut, layer.wo, projected)

    # Residual 1: x2 = x1 + projected
    lc.x2 = trackedCreate(S * n)
    gpu_add(lc.x1.data, projected.data, lc.x2.data, cint(S * n))
    x = lc.x2

    lc.xNorm2 = trackedCreate(S * n)
    lc.rms2 = trackedCreate(S)
    gpu_rmsnorm_affine_fwd(x.data, layer.ln2g.data, lc.xNorm2.data, lc.rms2.data, cint(S), cint(n))

    # MLP: SwiGLU — gate and up projections, then swish(gate) * up, then down
    lc.fc1Out = trackedCreate(S * ffnMul * n)  # gate projection
    gpuSgemm(2, S, ffnMul * n, n, lc.xNorm2, layer.fcGate, lc.fc1Out)

    lc.upOut = trackedCreate(S * ffnMul * n)    # up projection
    gpuSgemm(2, S, ffnMul * n, n, lc.xNorm2, layer.fcUp, lc.upOut)

    lc.geluOut = trackedCreate(S * ffnMul * n)  # swish(gate) * up
    gpu_swiglu_fwd(lc.fc1Out.data, lc.upOut.data, lc.geluOut.data, cint(S * ffnMul * n))

    var mlpOut = trackedCreate(S * n)
    gpuSgemm(2, S, n, ffnMul * n, lc.geluOut, layer.fcDown, mlpOut)

    # Residual 2: x = x2 + mlpOut
    let xNew = trackedCreate(S * n)
    gpu_add(lc.x2.data, mlpOut.data, xNew.data, cint(S * n))
    x = xNew

    cache.layerCaches.add(lc)

  # Final norm with learnable gamma
  cache.xPreFinalNorm = trackedCreate(S * n)
  gpuCopy(x, cache.xPreFinalNorm, S * n)
  cache.finalNormed = trackedCreate(S * n)
  cache.rmsF = trackedCreate(S)
  gpu_rmsnorm_affine_fwd(x.data, m.lnFg.data, cache.finalNormed.data, cache.rmsF.data, cint(S), cint(n))

  # Logits
  cache.logits = trackedCreate(S * m.vocabSize)
  gpuSgemm(2, S, m.vocabSize, n, cache.finalNormed, m.lmHead, cache.logits)

  # Log-softmax + cross-entropy loss
  cache.logProbs = trackedCreate(S * m.vocabSize)
  gpu_log_softmax(cache.logits.data, cache.logProbs.data, cint(S), cint(m.vocabSize))

  # Upload target IDs (tokens shifted by 1) and compute mean loss on GPU.
  # Downloads only S floats (~2KB) instead of S×V (~4.6MB).
  var targets = newSeq[int32](S)
  for i in 0 ..< S: targets[i] = tokens[i + 1]
  discard cudaMemcpy(m.targetIdBuf, unsafeAddr targets[0],
                     csize_t(S * sizeof(int32)), CudaMemcpyHostToDevice)
  let lossScratch = trackedCreate(S)
  let loss = gpu_ce_loss(cache.logProbs.data, m.targetIdBuf, lossScratch.data,
                         cint(S), cint(m.vocabSize))

  (cache, loss)

# ── Backward pass ─────────────────────────────────────────────────

proc backward(m: var Model, tokens: seq[int32], seqLen: int,
              cache: ForwardCache) =
  let S = seqLen
  let n = nEmbd
  let V = m.vocabSize

  # dLogits = (exp(logProbs) - one_hot(target)) / S  — entirely on GPU.
  # targetIdBuf already has targets from forward's loss computation.
  let dLogits = trackedCreate(S * V)
  gpu_ce_backward(cache.logProbs.data, m.targetIdBuf, dLogits.data, cint(S), cint(V))

  # dLmHead += dLogits^T @ finalNormed
  gpuSgemm(5, V, n, S, dLogits, cache.finalNormed, m.dlmHead)
  # dFinalNormed = dLogits @ lmHead
  var dx = trackedCreate(S * n)
  gpuSgemm(1, S, n, V, dLogits, m.lmHead, dx)

  # Final RMSNorm backward
  let dxPreNorm = trackedCreate(S * n)
  gpu_rmsnorm_affine_bwd(cache.xPreFinalNorm.data, m.lnFg.data, dx.data,
                         cache.rmsF.data, dxPreNorm.data, m.dlnFg.data,
                         cint(S), cint(n))
  dx = dxPreNorm

  # Backward through layers in reverse
  for li in countdown(nLayer - 1, 0):
    let lc = cache.layerCaches[li]
    let layer = m.layers[li]

    # Residual 2 backward: dx flows to both MLP path and skip
    let dMlpOut = trackedCreate(S * n)
    gpuCopy(dx, dMlpOut, S * n)
    # dx also flows through residual (already in dx)

    # MLP backward: SwiGLU
    # down projection backward
    let dSwigluOut = trackedCreate(S * ffnMul * n)
    gpuSgemm(1, S, ffnMul * n, n, dMlpOut, layer.fcDown, dSwigluOut)
    gpuSgemm(5, n, ffnMul * n, S, dMlpOut, lc.geluOut, m.layers[li].dfcDown)

    # SwiGLU backward: dGate and dUp from dSwigluOut
    let dGate = trackedCreate(S * ffnMul * n)
    let dUp = trackedCreate(S * ffnMul * n)
    gpu_swiglu_bwd(lc.fc1Out.data, lc.upOut.data, dSwigluOut.data,
                   dGate.data, dUp.data, cint(S * ffnMul * n))

    # Gate and up projection backward
    let dNorm2 = trackedCreate(S * n)
    gpuSgemm(1, S, n, ffnMul * n, dGate, layer.fcGate, dNorm2)
    gpuSgemm(5, ffnMul * n, n, S, dGate, lc.xNorm2, m.layers[li].dfcGate)

    let dNorm2up = trackedCreate(S * n)
    gpuSgemm(1, S, n, ffnMul * n, dUp, layer.fcUp, dNorm2up)
    gpuSgemm(5, ffnMul * n, n, S, dUp, lc.xNorm2, m.layers[li].dfcUp)
    gpu_add_inplace(dNorm2.data, dNorm2up.data, cint(S * n))

    # RMSNorm 2 backward
    let dResid2 = trackedCreate(S * n)
    gpu_rmsnorm_affine_bwd(lc.x2.data, layer.ln2g.data, dNorm2.data,
                           lc.rms2.data, dResid2.data, m.layers[li].dln2g.data,
                           cint(S), cint(n))
    # Add residual gradient
    gpu_add_inplace(dx.data, dResid2.data, cint(S * n))

    # Attention output projection backward
    let dAttnOut = trackedCreate(S * n)
    # dProjected = dx (from residual 1 backward)
    gpuSgemm(1, S, n, n, dx, layer.wo, dAttnOut)
    gpuSgemm(5, n, n, S, dx, lc.attnOut, m.layers[li].dwo)

    # GQA attention backward via cuBLAS
    let dq = trackedCreate(S * n)
    let dk = trackedCreate(S * nKvDim)
    let dv = trackedCreate(S * nKvDim)
    let scale = 1.0f / sqrt(float32(headDim))

    let doutH = trackedCreate(S * headDim)
    let qH = trackedCreate(S * headDim)
    let kH = trackedCreate(S * headDim)
    let vH = trackedCreate(S * headDim)
    let dqH = trackedCreate(S * headDim)
    let dkH = trackedCreate(S * headDim)
    let dvH = trackedCreate(S * headDim)

    let bwdScores = trackedCreate(S * S)
    let bwdProbs = trackedCreate(S * S)
    let dWeights = trackedCreate(S * S)
    let dScores = trackedCreate(S * S)

    for h in 0 ..< nHead:
      extractHead(dAttnOut, doutH, h, S, n, headDim)
      extractHead(lc.q, qH, h, S, n, headDim)
      gpu_extract_kv_head(lc.k.data, kH.data, cint(h), cint(kvRepeat),
                          cint(S), cint(nKvDim), cint(headDim))
      gpu_extract_kv_head(lc.v.data, vH.data, cint(h), cint(kvRepeat),
                          cint(S), cint(nKvDim), cint(headDim))

      # Recompute attention weights
      gpuSgemm(2, S, S, headDim, qH, kH, bwdScores)
      causalMask(bwdScores, scale, S)
      softmaxFwd(bwdScores, bwdProbs, S, S)

      gpuSgemm(2, S, S, headDim, doutH, vH, dWeights)
      gpuSgemm(4, S, headDim, S, bwdProbs, doutH, dvH)

      discard cudaMemset(dScores.data, 0, csize_t(S * S * sizeof(float32)))
      softmaxBwd(bwdProbs, dWeights, dScores, S, S)
      gpu_scale(dScores.data, scale, dScores.data, cint(S * S))

      gpuSgemm(0, S, headDim, S, dScores, kH, dqH)
      gpuSgemm(4, S, headDim, S, dScores, qH, dkH)

      insertHeadAcc(dqH, dq, h, S, n, headDim)
      # GQA: accumulate dK/dV back to the shared KV head
      gpu_insert_kv_head_acc(dkH.data, dk.data, cint(h), cint(kvRepeat),
                             cint(S), cint(nKvDim), cint(headDim))
      gpu_insert_kv_head_acc(dvH.data, dv.data, cint(h), cint(kvRepeat),
                             cint(S), cint(nKvDim), cint(headDim))

    # Inverse RoPE before weight backward
    ropeBwd(dq, m.ropeCos, m.ropeSin, S, n, nHead, headDim)
    ropeBwd(dk, m.ropeCos, m.ropeSin, S, nKvDim, nKvHead, headDim)

    # QKV projection backward (K/V use smaller nKvDim)
    let dNorm1 = trackedCreate(S * n)
    gpuSgemm(1, S, n, n, dq, layer.wq, dNorm1)
    gpuSgemm(5, n, n, S, dq, lc.xNorm1, m.layers[li].dwq)

    let dNorm1k = trackedCreate(S * n)
    gpuSgemm(1, S, n, nKvDim, dk, layer.wk, dNorm1k)
    gpuSgemm(5, nKvDim, n, S, dk, lc.xNorm1, m.layers[li].dwk)
    gpu_add_inplace(dNorm1.data, dNorm1k.data, cint(S * n))

    let dNorm1v = trackedCreate(S * n)
    gpuSgemm(1, S, n, nKvDim, dv, layer.wv, dNorm1v)
    gpuSgemm(5, nKvDim, n, S, dv, lc.xNorm1, m.layers[li].dwv)
    gpu_add_inplace(dNorm1.data, dNorm1v.data, cint(S * n))

    # RMSNorm 1 backward
    let dResid1 = trackedCreate(S * n)
    gpu_rmsnorm_affine_bwd(lc.x1.data, layer.ln1g.data, dNorm1.data,
                           lc.rms1.data, dResid1.data, m.layers[li].dln1g.data,
                           cint(S), cint(n))

    # Residual 1: dx for next layer = dResid1 + dResid2_skip
    gpu_add(dResid1.data, dx.data, dx.data, cint(S * n))

  # Embedding backward (no position embedding — RoPE has no learnable params)
  gpu_embed_bwd(m.dwte.data, m.tokIdBuf, dx.data, cint(S), cint(n))

# ── Zero gradients ────────────────────────────────────────────────

# ── Checkpoint ────────────────────────────────────────────────────

proc saveModel(m: Model, filename: string) =
  ## Save model with config header. Format:
  ##   magic (4 bytes: "NLLM"), version (int32: 2),
  ##   nLayer, nEmbd, nHead, vocabSize, blockSize (5 × int32),
  ##   then weight buffers: each is int32(count) + float32[count]
  echo "saving to ", filename, "..."
  let s = newFileStream(filename, fmWrite)
  defer: s.close()
  # Header
  s.writeData("NLLM".cstring, 4)
  s.write(int32(2))          # version
  s.write(int32(nLayer))
  s.write(int32(nEmbd))
  s.write(int32(nHead))
  s.write(int32(m.vocabSize))
  s.write(int32(blockSize))
  # Weights
  proc w(buf: GpuBuf) =
    let d = gpuDownload(buf)
    s.write(int32(d.len))
    for v in d: s.write(v)
  w(m.wte); w(m.lmHead); w(m.lnFg)
  for layer in m.layers:
    w(layer.wq); w(layer.wk); w(layer.wv); w(layer.wo)
    w(layer.fcGate); w(layer.fcUp); w(layer.fcDown); w(layer.ln1g); w(layer.ln2g)
  echo "  done"

proc loadModel(m: var Model, filename: string) =
  ## Load model checkpoint. Reads header to verify config matches.
  echo "loading from ", filename, "..."
  let s = newFileStream(filename, fmRead)
  defer: s.close()
  # Check for header (v2 format starts with "NLLM")
  var magic: array[4, char]
  discard s.readData(addr magic[0], 4)
  if magic == ['N', 'L', 'L', 'M']:
    let ver = s.readInt32()
    let fLayer = s.readInt32().int
    let fEmbd = s.readInt32().int
    let fHead = s.readInt32().int
    let fVocab = s.readInt32().int
    let fBlock = s.readInt32().int
    echo &"  checkpoint: {fLayer}L {fEmbd}d {fHead}h vocab={fVocab} block={fBlock} (v{ver})"
    if fLayer != nLayer or fEmbd != nEmbd or fHead != nHead:
      echo "  WARNING: config mismatch! Use --grow instead."
      echo &"    checkpoint: {fLayer}L {fEmbd}d {fHead}h"
      echo &"    current:    {nLayer}L {nEmbd}d {nHead}h"
      quit(1)
  else:
    # Legacy format (no header) — rewind and read directly
    echo "  (legacy format, no header)"
    s.setPosition(0)
  proc r(buf: GpuBuf) =
    let n = s.readInt32().int
    var d = newSeq[float32](n)
    for i in 0 ..< n: d[i] = s.readFloat32()
    gpuUpload(buf, d)
  r(m.wte); r(m.lmHead); r(m.lnFg)
  for i in 0 ..< m.layers.len:
    r(m.layers[i].wq); r(m.layers[i].wk)
    r(m.layers[i].wv); r(m.layers[i].wo)
    r(m.layers[i].fcGate); r(m.layers[i].fcUp); r(m.layers[i].fcDown)
    r(m.layers[i].ln1g); r(m.layers[i].ln2g)
  echo "  done"

proc growModel(m: var Model, oldFile: string) =
  ## Load a smaller checkpoint into a larger model.
  ## Copies old weights into the top-left corner of new weights.
  ## New capacity stays at init values (small random / ones for gamma).
  ## Works across dim changes, layer count changes, and vocab changes.
  echo "growing from ", oldFile, "..."
  let s = newFileStream(oldFile, fmRead)
  defer: s.close()

  proc growBuf(dst: GpuBuf, oldRows, oldCols, newCols: int) =
    ## Copy a [oldRows, oldCols] weight into the top-left of [?, newCols].
    let n = s.readInt32().int
    var old = newSeq[float32](n)
    for i in 0 ..< n: old[i] = s.readFloat32()
    if oldCols == newCols and n == dst.numel:
      # Same shape — direct copy
      gpuUpload(dst, old)
    else:
      # Different shape — copy row by row into top-left corner
      var grown = gpuDownload(dst)  # get current init values
      for r in 0 ..< oldRows:
        for c in 0 ..< oldCols:
          if r * newCols + c < grown.len:
            grown[r * newCols + c] = old[r * oldCols + c]
      gpuUpload(dst, grown)

  proc growVec(dst: GpuBuf, oldLen: int) =
    ## Copy a [oldLen] vector into the first oldLen elements of dst.
    let n = s.readInt32().int
    var old = newSeq[float32](n)
    for i in 0 ..< n: old[i] = s.readFloat32()
    var grown = gpuDownload(dst)
    for i in 0 ..< min(oldLen, grown.len):
      if i < old.len: grown[i] = old[i]
    gpuUpload(dst, grown)

  # Read old model header to detect dimensions
  # The first buffer is wte [oldVocab, oldEmbd]. Its size tells us the old dims.
  let wteSize = s.peekInt32().int  # peek without advancing
  # We need to know old dims. Checkpoint stores: wte, wpe, lmHead, lnFg, then layers.
  # wte has vocabSize * nEmbd elements. We can figure out oldEmbd from wte and vocab.
  # But we don't know oldVocab directly. We need a header.
  # For now: assume we can read oldEmbd from the checkpoint.
  # Strategy: read wte size, then wpe size. wpe = blockSize * oldEmbd → oldEmbd = wpeSize / blockSize.

  # Read wte
  let wteN = s.readInt32().int
  var wteOld = newSeq[float32](wteN)
  for i in 0 ..< wteN: wteOld[i] = s.readFloat32()

  # Read wpe to determine oldEmbd
  let wpeN = s.readInt32().int
  let oldEmbd = wpeN div blockSize
  let oldVocab = wteN div oldEmbd
  var wpeOld = newSeq[float32](wpeN)
  for i in 0 ..< wpeN: wpeOld[i] = s.readFloat32()

  echo &"  old model: vocab={oldVocab} embd={oldEmbd}"
  echo &"  new model: vocab={m.vocabSize} embd={nEmbd}"

  # Grow wte [vocab, embd]
  var wte = gpuDownload(m.wte)
  for r in 0 ..< min(oldVocab, m.vocabSize):
    for c in 0 ..< min(oldEmbd, nEmbd):
      wte[r * nEmbd + c] = wteOld[r * oldEmbd + c]
  gpuUpload(m.wte, wte)

  # Skip wpe from old checkpoint (RoPE replaces it, no wpe in new model)

  # Read and grow lmHead [vocab, embd]
  let lmN = s.readInt32().int
  var lmOld = newSeq[float32](lmN)
  for i in 0 ..< lmN: lmOld[i] = s.readFloat32()
  var lmHead = gpuDownload(m.lmHead)
  for r in 0 ..< min(oldVocab, m.vocabSize):
    for c in 0 ..< min(oldEmbd, nEmbd):
      lmHead[r * nEmbd + c] = lmOld[r * oldEmbd + c]
  gpuUpload(m.lmHead, lmHead)

  # Read and grow lnFg [embd]
  let lnN = s.readInt32().int
  var lnOld = newSeq[float32](lnN)
  for i in 0 ..< lnN: lnOld[i] = s.readFloat32()
  var lnFg = gpuDownload(m.lnFg)
  for i in 0 ..< min(oldEmbd, nEmbd):
    lnFg[i] = lnOld[i]
  gpuUpload(m.lnFg, lnFg)

  # Read and grow layers (copy as many as exist in old model)
  proc readGrowMatrix(dst: GpuBuf, oldR, oldC, newC: int) =
    let n = s.readInt32().int
    var old = newSeq[float32](n)
    for i in 0 ..< n: old[i] = s.readFloat32()
    var grown = gpuDownload(dst)
    for r in 0 ..< oldR:
      for c in 0 ..< oldC:
        if r * newC + c < grown.len:
          grown[r * newC + c] = old[r * oldC + c]
    gpuUpload(dst, grown)

  proc readGrowVec(dst: GpuBuf, oldLen: int) =
    let n = s.readInt32().int
    var old = newSeq[float32](n)
    for i in 0 ..< n: old[i] = s.readFloat32()
    var grown = gpuDownload(dst)
    for i in 0 ..< min(oldLen, grown.len):
      grown[i] = old[i]
    gpuUpload(dst, grown)

  var li = 0
  while not s.atEnd() and li < m.layers.len:
    # Each layer: wq[e,e], wk[e,e], wv[e,e], wo[e,e], fc1[4e,e], fc2[e,4e], ln1g[e], ln2g[e]
    readGrowMatrix(m.layers[li].wq, oldEmbd, oldEmbd, nEmbd)
    readGrowMatrix(m.layers[li].wk, oldEmbd, oldEmbd, nEmbd)
    readGrowMatrix(m.layers[li].wv, oldEmbd, oldEmbd, nEmbd)
    readGrowMatrix(m.layers[li].wo, oldEmbd, oldEmbd, nEmbd)
    readGrowMatrix(m.layers[li].fcGate, ffnMul * oldEmbd, oldEmbd, nEmbd)
    readGrowMatrix(m.layers[li].fcUp, ffnMul * oldEmbd, oldEmbd, nEmbd)
    readGrowMatrix(m.layers[li].fcDown, oldEmbd, ffnMul * oldEmbd, ffnMul * nEmbd)
    readGrowVec(m.layers[li].ln1g, oldEmbd)
    readGrowVec(m.layers[li].ln2g, oldEmbd)
    li += 1

  echo &"  copied {li} layers (new model has {m.layers.len})"
  echo "  growth complete"

proc zeroGrads(m: var Model) =
  gpuZero(m.dwte)
  gpuZero(m.dlmHead)
  gpuZero(m.dlnFg)
  for i in 0 ..< m.layers.len:
    gpuZero(m.layers[i].dwq)
    gpuZero(m.layers[i].dwk)
    gpuZero(m.layers[i].dwv)
    gpuZero(m.layers[i].dwo)
    gpuZero(m.layers[i].dfcGate)
    gpuZero(m.layers[i].dfcUp)
    gpuZero(m.layers[i].dfcDown)
    gpuZero(m.layers[i].dln1g)
    gpuZero(m.layers[i].dln2g)

# ── Adam optimizer ────────────────────────────────────────────────

type AdamState = object
  m, v: seq[GpuBuf]

proc initAdam(m: Model): AdamState =
  proc addPair(s: var AdamState, buf: GpuBuf) =
    s.m.add(gpuCreate(buf.numel))
    s.v.add(gpuCreate(buf.numel))
  addPair(result, m.wte)
  addPair(result, m.lmHead)
  addPair(result, m.lnFg)
  for layer in m.layers:
    addPair(result, layer.wq)
    addPair(result, layer.wk)
    addPair(result, layer.wv)
    addPair(result, layer.wo)
    addPair(result, layer.ln1g)
    addPair(result, layer.ln2g)
    addPair(result, layer.fcGate)
    addPair(result, layer.fcUp)
    addPair(result, layer.fcDown)

proc adamUpdate(m: var Model, adam: var AdamState, lr: float32,
                step: int, beta1 = 0.9f, beta2 = 0.999f, wd = 0.1f) =
  let bc1 = 1.0f / (1.0f - pow(beta1, float32(step + 1)))
  let bc2 = 1.0f / (1.0f - pow(beta2, float32(step + 1)))

  var pairs: seq[(GpuBuf, GpuBuf)] # (param, grad)
  pairs.add((m.wte, m.dwte))
  pairs.add((m.lmHead, m.dlmHead))
  pairs.add((m.lnFg, m.dlnFg))
  for i in 0 ..< m.layers.len:
    pairs.add((m.layers[i].wq, m.layers[i].dwq))
    pairs.add((m.layers[i].wk, m.layers[i].dwk))
    pairs.add((m.layers[i].wv, m.layers[i].dwv))
    pairs.add((m.layers[i].wo, m.layers[i].dwo))
    pairs.add((m.layers[i].ln1g, m.layers[i].dln1g))
    pairs.add((m.layers[i].ln2g, m.layers[i].dln2g))
    pairs.add((m.layers[i].fcGate, m.layers[i].dfcGate))
    pairs.add((m.layers[i].fcUp, m.layers[i].dfcUp))
    pairs.add((m.layers[i].fcDown, m.layers[i].dfcDown))

  for i in 0 ..< pairs.len:
    gpu_adamw(pairs[i][0].data, pairs[i][1].data,
              adam.m[i].data, adam.v[i].data,
              lr, beta1, beta2, bc1, bc2, wd,
              cint(pairs[i][0].numel))

# ── Training ──────────────────────────────────────────────────────

proc formatDuration(secs: float): string =
  let h = int(secs) div 3600
  let m = (int(secs) mod 3600) div 60
  let s = int(secs) mod 60
  if h > 0: &"{h}h{m:02d}m{s:02d}s"
  elif m > 0: &"{m}m{s:02d}s"
  else: &"{s}s"

when isMainModule:
  let baseDir = getAppDir().parentDir()
  let vidyaRoot = baseDir.parentDir()
  let dataFile = vidyaRoot / "training_data.txt"  # chat + books + claude logs
  let tokenizerFile = vidyaRoot / "tokenizer_nim.bin"

  gpuInit()
  echo "microgpt (nim) starting..."

  # Tokenizer
  var tok: Tokenizer
  if fileExists(tokenizerFile):
    tok = loadTokenizer(tokenizerFile)
  else:
    let docs = loadDocs(dataFile)
    tok = trainBpe(docs)
    saveTokenizer(tok, tokenizerFile)
  echo &"vocab: {tok.vocab.len}"

  # Model
  let checkpointFile = vidyaRoot / "nimllm.bin"

  # Check modes: --read <file>, --grow <old_checkpoint>
  let readMode = paramCount() >= 1 and paramStr(1) == "--read"
  let readFile = if readMode and paramCount() >= 2: paramStr(2) else: ""
  let growMode = paramCount() >= 1 and paramStr(1) == "--grow"
  let growFile = if growMode and paramCount() >= 2: paramStr(2) else: ""
  let continueMode = paramCount() >= 1 and paramStr(1) == "--continue"

  echo "init model..."
  var m = initModel(tok.vocab.len)
  if growMode:
    if growFile == "" or not fileExists(growFile):
      echo "usage: microgpt --grow <old_checkpoint>"
      echo "  Loads a smaller model's weights into the current (larger) model."
      echo "  Old knowledge preserved. New capacity learns from training."
      quit(1)
    growModel(m, growFile)
  elif fileExists(checkpointFile):
    loadModel(m, checkpointFile)

  # Data
  var tokenizedDocs: seq[seq[int32]]
  if readMode:
    if readFile == "" or not fileExists(readFile):
      echo "usage: microgpt --read <textfile>"
      quit(1)
    echo "reading ", readFile, "..."
    let text = readFile(readFile)
    echo &"  {text.len} chars"
    # Split into chunks at sentence boundaries
    var chunks: seq[string]
    var pos = 0
    while pos < text.len:
      var endPos = min(pos + 500, text.len)
      if endPos < text.len:
        for i in countdown(endPos, max(pos + 200, 0)):
          if text[i] == '.' or text[i] == '\n':
            endPos = i + 1; break
      chunks.add(text[pos ..< endPos])
      pos = endPos
    let t0 = cpuTime()
    for chunk in chunks:
      let ids = tok.encode(chunk)
      var ids32 = newSeq[int32](ids.len)
      for i in 0 ..< ids.len: ids32[i] = int32(ids[i])
      if ids32.len >= 3:
        tokenizedDocs.add(ids32)
    echo &"  {tokenizedDocs.len} chunks in {cpuTime() - t0:.1f}s"
  else:
    echo "loading data..."
    let docs = loadDocs(dataFile)
    let t0 = cpuTime()
    for doc in docs:
      let ids = tok.encode(doc)
      var ids32 = newSeq[int32](ids.len)
      for i in 0 ..< ids.len: ids32[i] = int32(ids[i])
      if ids32.len >= 3:
        tokenizedDocs.add(ids32)
    echo &"  {tokenizedDocs.len} docs in {cpuTime() - t0:.1f}s"

  # Shuffle
  randomize()
  var order = newSeq[int](tokenizedDocs.len)
  for i in 0 ..< order.len: order[i] = i
  shuffle(order)

  # Adam
  var adam = initAdam(m)

  # Pre-allocate scratch arena — one big GPU alloc instead of ~300 malloc/free per step.
  # Sizes: forward (embed + 8 layers + final norm + logits) + backward (same structure).
  let S = blockSize  # max sequence length
  let n = nEmbd
  let V = tok.vocab.len
  # SwiGLU: 3 FFN buffers in forward (gate, up, swiglu out), 4 in backward (+dGate, +dUp, +dNorm2up)
  let fwdPerLayer = 10 * S * n + 4 * S * headDim + 2 * S + 3 * S * ffnMul * n + 2 * S * S
  let fwdGlobal = S * n + S + 2 * S * V
  let bwdPerLayer = 11 * S * n + 7 * S * headDim + 3 * S * ffnMul * n + 4 * S * S
  let bwdGlobal = S * V + 2 * S * n
  let arenaSize = fwdGlobal + fwdPerLayer * nLayer + bwdGlobal + bwdPerLayer * nLayer
  let arenaMB = arenaSize * sizeof(float32) div (1024 * 1024)
  initScratchArena(arenaSize + arenaSize div 4)  # +25% headroom
  echo &"  scratch arena: {arenaMB} MB ({arenaSize} floats)"

  # Train
  # Read mode: ~3 passes over the material. More for short texts, fewer for long.
  let readSteps = if readMode: max(1000, min(20000, tokenizedDocs.len * 3)) else: 0
  let numSteps = if readMode: readSteps elif continueMode: 200000 else: 200000
  let warmupSteps = if readMode: numSteps div 50 elif continueMode: 0 else: 2000
  let peakLr = if readMode: 0.00005f elif continueMode: 0.00003f else: 0.0001f
  let minLr = if readMode: 0.000005f elif continueMode: 0.00003f else: 0.00001f
  let gradAccum = 1      # batch 1 — research says it works with right settings

  # Elastic weight consolidation for --read mode.
  # Save current weights as anchor. After each optimizer step, pull weights
  # back toward anchor. Prevents catastrophic forgetting of base knowledge.
  type Anchor = object
    bufs: seq[GpuBuf]
  var anchor: Anchor
  if readMode:
    echo "  saving weight anchor for elastic pull..."
    proc saveAnchor(buf: GpuBuf): GpuBuf =
      result = gpuCreate(buf.numel)
      gpuCopy(buf, result, buf.numel)
    anchor.bufs.add(saveAnchor(m.wte))
    anchor.bufs.add(saveAnchor(m.lmHead))
    anchor.bufs.add(saveAnchor(m.lnFg))
    for i in 0 ..< m.layers.len:
      anchor.bufs.add(saveAnchor(m.layers[i].wq))
      anchor.bufs.add(saveAnchor(m.layers[i].wk))
      anchor.bufs.add(saveAnchor(m.layers[i].wv))
      anchor.bufs.add(saveAnchor(m.layers[i].wo))
      anchor.bufs.add(saveAnchor(m.layers[i].fcGate))
      anchor.bufs.add(saveAnchor(m.layers[i].fcUp))
      anchor.bufs.add(saveAnchor(m.layers[i].fcDown))
      anchor.bufs.add(saveAnchor(m.layers[i].ln1g))
      anchor.bufs.add(saveAnchor(m.layers[i].ln2g))

  echo &"training {numSteps} steps (grad_accum={gradAccum})..."

  let tStart = cpuTime()
  var lossSum = 0.0f
  var microStep = 0
  var optStep = 0
  let logInterval = 50  # in optimizer steps

  for step in 0 ..< numSteps:
    let tokens = tokenizedDocs[order[step mod order.len]]
    let seqLen = min(blockSize, tokens.len - 1)
    if seqLen < 2: continue

    var (cache, loss) = forward(m, tokens[0 ..< seqLen + 1], seqLen)
    backward(m, tokens[0 ..< seqLen + 1], seqLen, cache)
    freeStepAllocations()

    lossSum += loss
    microStep += 1

    if microStep < gradAccum:
      continue

    # Scale gradients by 1/gradAccum
    proc scaleAllGrads(m: var Model, s: float32) =
      gpu_scale(m.dwte.data, s, m.dwte.data, cint(m.dwte.numel))
      gpu_scale(m.dlmHead.data, s, m.dlmHead.data, cint(m.dlmHead.numel))
      for i in 0 ..< m.layers.len:
        gpu_scale(m.layers[i].dwq.data, s, m.layers[i].dwq.data, cint(m.layers[i].dwq.numel))
        gpu_scale(m.layers[i].dwk.data, s, m.layers[i].dwk.data, cint(m.layers[i].dwk.numel))
        gpu_scale(m.layers[i].dwv.data, s, m.layers[i].dwv.data, cint(m.layers[i].dwv.numel))
        gpu_scale(m.layers[i].dwo.data, s, m.layers[i].dwo.data, cint(m.layers[i].dwo.numel))
        gpu_scale(m.layers[i].dfcGate.data, s, m.layers[i].dfcGate.data, cint(m.layers[i].dfcGate.numel))
        gpu_scale(m.layers[i].dfcUp.data, s, m.layers[i].dfcUp.data, cint(m.layers[i].dfcUp.numel))
        gpu_scale(m.layers[i].dfcDown.data, s, m.layers[i].dfcDown.data, cint(m.layers[i].dfcDown.numel))
    scaleAllGrads(m, 1.0f / float32(gradAccum))

    # LR schedule
    let totalOptSteps = numSteps div gradAccum
    let lr = if optStep < warmupSteps:
        peakLr * float32(optStep) / float32(warmupSteps)
      else:
        let progress = float32(optStep - warmupSteps) / float32(totalOptSteps - warmupSteps)
        minLr + (peakLr - minLr) * 0.5f * (1f + cos(PI.float32 * progress))

    # Gradient clipping — clip global norm to 1.0
    var gradPtrs: seq[pointer]
    var gradSizes: seq[cint]
    gradPtrs.add(m.dwte.data); gradSizes.add(cint(m.dwte.numel))
    gradPtrs.add(m.dlmHead.data); gradSizes.add(cint(m.dlmHead.numel))
    gradPtrs.add(m.dlnFg.data); gradSizes.add(cint(m.dlnFg.numel))
    for i in 0 ..< m.layers.len:
      gradPtrs.add(m.layers[i].dwq.data); gradSizes.add(cint(m.layers[i].dwq.numel))
      gradPtrs.add(m.layers[i].dwk.data); gradSizes.add(cint(m.layers[i].dwk.numel))
      gradPtrs.add(m.layers[i].dwv.data); gradSizes.add(cint(m.layers[i].dwv.numel))
      gradPtrs.add(m.layers[i].dwo.data); gradSizes.add(cint(m.layers[i].dwo.numel))
      gradPtrs.add(m.layers[i].dfcGate.data); gradSizes.add(cint(m.layers[i].dfcGate.numel))
      gradPtrs.add(m.layers[i].dfcUp.data); gradSizes.add(cint(m.layers[i].dfcUp.numel))
      gradPtrs.add(m.layers[i].dfcDown.data); gradSizes.add(cint(m.layers[i].dfcDown.numel))
      gradPtrs.add(m.layers[i].dln1g.data); gradSizes.add(cint(m.layers[i].dln1g.numel))
      gradPtrs.add(m.layers[i].dln2g.data); gradSizes.add(cint(m.layers[i].dln2g.numel))
    clipGradNorm(gradPtrs, gradSizes, 1.0f)

    adamUpdate(m, adam, lr, optStep, beta2 = 0.9999f, wd = 0.0f)

    # Elastic pull: in read mode, gently pull weights back toward anchor.
    # alpha=0.001 means each step moves 0.1% back toward base weights.
    # This prevents the model from drifting too far from what it already knows.
    if readMode and anchor.bufs.len > 0:
      let alpha = 0.001f
      var idx = 0
      proc pull(buf: GpuBuf) =
        gpu_elastic(buf.data, anchor.bufs[idx].data, alpha, cint(buf.numel))
        idx += 1
      pull(m.wte); pull(m.lmHead); pull(m.lnFg)
      for i in 0 ..< m.layers.len:
        pull(m.layers[i].wq); pull(m.layers[i].wk)
        pull(m.layers[i].wv); pull(m.layers[i].wo)
        pull(m.layers[i].fcGate); pull(m.layers[i].fcUp); pull(m.layers[i].fcDown)
        pull(m.layers[i].ln1g); pull(m.layers[i].ln2g)

    zeroGrads(m)
    microStep = 0
    optStep += 1

    if optStep mod logInterval == 0:
      let elapsed = cpuTime() - tStart
      let opsps = float(optStep) / elapsed
      let totalOpt = numSteps div gradAccum
      let eta = float(totalOpt - optStep) / max(opsps, 0.01)
      echo &"opt {optStep:>5} / {totalOpt} | loss {lossSum / float32(logInterval * gradAccum):.4f} | lr {lr:.6f} | {opsps:.1f} opt/s | {formatDuration(elapsed)} elapsed | {formatDuration(eta)} remaining"
      lossSum = 0.0f

    # Checkpoint every ~1 hour (40K steps at 11 steps/sec)
    if optStep > 0 and optStep mod 10000 == 0:
      saveModel(m, checkpointFile)

  # Final save
  saveModel(m, checkpointFile)
