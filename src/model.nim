## model.nim — nimllm model: forward pass + backward pass
##
## The complete model in one file. Forward and backward are two halves
## of the same operation — separating them would require duplicating
## the forward pass. Shared by training and inference.

import gpu, autograd
import std/[math, random, streams, strformat, tables]

# ── Configuration ─────────────────────────────────────────────────

const
  nLayer*   = 30
  nEmbd*    = 576
  nHead*    = 9
  nKvHead*  = 3
  headDim*  = nEmbd div nHead  # 64
  kvRepeat* = nHead div nKvHead  # 3
  nKvDim*   = nKvHead * headDim  # 192
  blockSize* = 512
  ffnDim*   = 1536
  ropeTheta* = 100000.0f

# ── Types ─────────────────────────────────────────────────────────

type
  Layer* = object
    wq*, wk*, wv*, wo*: GpuBuf
    fcGate*, fcUp*, fcDown*: GpuBuf
    ln1g*, ln2g*: GpuBuf
    # Gradients (only used during training)
    dwq*, dwk*, dwv*, dwo*: GpuBuf
    dfcGate*, dfcUp*, dfcDown*: GpuBuf
    dln1g*, dln2g*: GpuBuf

  Model* = object
    wte*, lmHead*, lnFg*: GpuBuf
    layers*: seq[Layer]
    vocabSize*: int
    dwte*, dlmHead*, dlnFg*: GpuBuf
    tokIdBuf*: pointer
    targetIdBuf*: pointer
    ropeCos*, ropeSin*: GpuBuf

  LayerCache* = object
    x1*: GpuBuf
    xNorm1*: GpuBuf
    rms1*: GpuBuf
    q*, k*, v*: GpuBuf
    attnOut*: GpuBuf
    x2*: GpuBuf
    xNorm2*: GpuBuf
    rms2*: GpuBuf
    fc1Out*: GpuBuf
    upOut*: GpuBuf
    geluOut*: GpuBuf

  # KV cache for fast inference (don't recompute full sequence every token)
  KvCache* = object
    k*: seq[GpuBuf]       # [nLayer] each [blockSize, nKvDim] — accumulated K values
    v*: seq[GpuBuf]       # [nLayer] each [blockSize, nKvDim] — accumulated V values
    pos*: int             # current position (number of tokens processed)

  ForwardCache* = object
    embedded*: GpuBuf
    layerCaches*: seq[LayerCache]
    xPreFinalNorm*: GpuBuf
    finalNormed*: GpuBuf
    rmsF*: GpuBuf
    logits*: GpuBuf
    logProbs*: GpuBuf

# ── Init ──────────────────────────────────────────────────────────

proc initModel*(vocabSize: int, withGradients: bool = true): Model =
  randomize(42)
  let std = 0.02f

  proc randBuf(n: int, s: float32 = std): GpuBuf =
    var host = newSeq[float32](n)
    for i in 0 ..< n:
      let u1 = rand(1.0).float32
      let u2 = rand(1.0).float32
      host[i] = s * sqrt(-2f * ln(max(u1, 1e-10f))) * cos(2f * PI.float32 * u2)
    toGpu(host)

  proc onesBuf(n: int): GpuBuf =
    var h = newSeq[float32](n)
    for i in 0 ..< n: h[i] = 1.0f
    toGpu(h)

  result.vocabSize = vocabSize
  result.wte = randBuf(vocabSize * nEmbd)
  result.lmHead = randBuf(vocabSize * nEmbd)
  result.lnFg = onesBuf(nEmbd)

  if withGradients:
    result.dwte = gpuCreate(vocabSize * nEmbd)
    result.dlmHead = gpuCreate(vocabSize * nEmbd)
    result.dlnFg = gpuCreate(nEmbd)

  for i in 0 ..< nLayer:
    let resStd = std / sqrt(float32(2 * nLayer))
    var layer = Layer(
      wq: randBuf(nEmbd * nEmbd),
      wk: randBuf(nKvDim * nEmbd),
      wv: randBuf(nKvDim * nEmbd),
      wo: randBuf(nEmbd * nEmbd, resStd),
      fcGate: randBuf(ffnDim * nEmbd),
      fcUp: randBuf(ffnDim * nEmbd),
      fcDown: randBuf(nEmbd * ffnDim, resStd),
      ln1g: onesBuf(nEmbd),
      ln2g: onesBuf(nEmbd),
    )
    if withGradients:
      layer.dwq = gpuCreate(nEmbd * nEmbd)
      layer.dwk = gpuCreate(nKvDim * nEmbd)
      layer.dwv = gpuCreate(nKvDim * nEmbd)
      layer.dwo = gpuCreate(nEmbd * nEmbd)
      layer.dfcGate = gpuCreate(ffnDim * nEmbd)
      layer.dfcUp = gpuCreate(ffnDim * nEmbd)
      layer.dfcDown = gpuCreate(nEmbd * ffnDim)
      layer.dln1g = gpuCreate(nEmbd)
      layer.dln2g = gpuCreate(nEmbd)
    result.layers.add(layer)

  discard cudaMalloc(addr result.tokIdBuf, csize_t(blockSize * sizeof(int32)))
  discard cudaMalloc(addr result.targetIdBuf, csize_t(blockSize * sizeof(int32)))

  # RoPE tables
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

  let total = vocabSize * nEmbd * 2 +
    nLayer * (2 * nEmbd * nEmbd + 2 * nKvDim * nEmbd + 3 * ffnDim * nEmbd)
  echo &"  params: {total}"

# ── Save / Load ───────────────────────────────────────────────────

proc saveModel*(m: Model, filename: string) =
  echo "saving to ", filename, "..."
  let s = newFileStream(filename, fmWrite)
  defer: s.close()
  s.writeData("NLLM".cstring, 4)
  s.write(int32(2))
  s.write(int32(nLayer))
  s.write(int32(nEmbd))
  s.write(int32(nHead))
  s.write(int32(m.vocabSize))
  s.write(int32(blockSize))
  proc w(buf: GpuBuf) =
    let d = gpuDownload(buf)
    s.write(int32(d.len))
    for v in d: s.write(v)
  w(m.wte); w(m.lmHead); w(m.lnFg)
  for layer in m.layers:
    w(layer.wq); w(layer.wk); w(layer.wv); w(layer.wo)
    w(layer.fcGate); w(layer.fcUp); w(layer.fcDown)
    w(layer.ln1g); w(layer.ln2g)
  echo "  done"

proc loadModel*(m: var Model, filename: string) =
  echo "loading from ", filename, "..."
  let s = newFileStream(filename, fmRead)
  defer: s.close()
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
      quit(1)
  else:
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

# ── Forward pass ──────────────────────────────────────────────────

proc forward*(m: Model, tokens: seq[int32], seqLen: int,
              saveCache: bool = true): (ForwardCache, seq[float32]) =
  ## Forward pass. Returns cache (for backward) and logits for last position.
  ## Set saveCache=false for inference (saves memory, slightly faster).
  var cache: ForwardCache
  let n = nEmbd
  let S = seqLen

  # Token embedding
  var x = trackedCreate(S * n)
  discard cudaMemcpy(m.tokIdBuf, unsafeAddr tokens[0],
                     csize_t(S * sizeof(int32)), CudaMemcpyHostToDevice)
  gpu_embed_fwd(m.wte.data, m.tokIdBuf, x.data, cint(S), cint(n))
  if saveCache: cache.embedded = x

  let scale = 1.0f / sqrt(float32(headDim))
  let qH = trackedCreate(S * headDim)
  let kH = trackedCreate(S * headDim)
  let vH = trackedCreate(S * headDim)
  let attnH = trackedCreate(S * headDim)
  let scores = trackedCreate(S * S)
  let probs = trackedCreate(S * S)

  # Transformer layers
  for li in 0 ..< nLayer:
    var lc: LayerCache
    let layer = m.layers[li]

    if saveCache: lc.x1 = x
    lc.xNorm1 = trackedCreate(S * n)
    lc.rms1 = trackedCreate(S)
    gpu_rmsnorm_affine_fwd(x.data, layer.ln1g.data, lc.xNorm1.data, lc.rms1.data, cint(S), cint(n))

    lc.q = trackedCreate(S * n)
    lc.k = trackedCreate(S * nKvDim)
    lc.v = trackedCreate(S * nKvDim)
    gpuSgemm(2, S, n, n, lc.xNorm1, layer.wq, lc.q)
    gpuSgemm(2, S, nKvDim, n, lc.xNorm1, layer.wk, lc.k)
    gpuSgemm(2, S, nKvDim, n, lc.xNorm1, layer.wv, lc.v)

    ropeFwd(lc.q, m.ropeCos, m.ropeSin, S, n, nHead, headDim)
    ropeFwd(lc.k, m.ropeCos, m.ropeSin, S, nKvDim, nKvHead, headDim)

    lc.attnOut = trackedCreate(S * n)
    for h in 0 ..< nHead:
      extractHead(lc.q, qH, h, S, n, headDim)
      gpu_extract_kv_head(lc.k.data, kH.data, cint(h), cint(kvRepeat),
                          cint(S), cint(nKvDim), cint(headDim))
      gpu_extract_kv_head(lc.v.data, vH.data, cint(h), cint(kvRepeat),
                          cint(S), cint(nKvDim), cint(headDim))
      gpuSgemm(2, S, S, headDim, qH, kH, scores)
      causalMask(scores, scale, S)
      softmaxFwd(scores, probs, S, S)
      gpuSgemm(0, S, headDim, S, probs, vH, attnH)
      insertHead(attnH, lc.attnOut, h, S, n, headDim)

    var projected = trackedCreate(S * n)
    gpuSgemm(2, S, n, n, lc.attnOut, layer.wo, projected)

    lc.x2 = trackedCreate(S * n)
    gpu_add(x.data, projected.data, lc.x2.data, cint(S * n))
    x = lc.x2

    lc.xNorm2 = trackedCreate(S * n)
    lc.rms2 = trackedCreate(S)
    gpu_rmsnorm_affine_fwd(x.data, layer.ln2g.data, lc.xNorm2.data, lc.rms2.data, cint(S), cint(n))

    lc.fc1Out = trackedCreate(S * ffnDim)
    gpuSgemm(2, S, ffnDim, n, lc.xNorm2, layer.fcGate, lc.fc1Out)
    lc.upOut = trackedCreate(S * ffnDim)
    gpuSgemm(2, S, ffnDim, n, lc.xNorm2, layer.fcUp, lc.upOut)
    lc.geluOut = trackedCreate(S * ffnDim)
    gpu_swiglu_fwd(lc.fc1Out.data, lc.upOut.data, lc.geluOut.data, cint(S * ffnDim))
    var mlpOut = trackedCreate(S * n)
    gpuSgemm(2, S, n, ffnDim, lc.geluOut, layer.fcDown, mlpOut)

    let xNew = trackedCreate(S * n)
    gpu_add(lc.x2.data, mlpOut.data, xNew.data, cint(S * n))
    x = xNew

    if saveCache: cache.layerCaches.add(lc)

  if saveCache: cache.xPreFinalNorm = x
  cache.finalNormed = trackedCreate(S * n)
  cache.rmsF = trackedCreate(S)
  gpu_rmsnorm_affine_fwd(x.data, m.lnFg.data, cache.finalNormed.data, cache.rmsF.data, cint(S), cint(n))

  cache.logits = trackedCreate(S * m.vocabSize)
  gpuSgemm(2, S, m.vocabSize, n, cache.finalNormed, m.lmHead, cache.logits)

  # Download last row of logits for sampling/loss
  let allLogits = gpuDownload(cache.logits)
  let lastRow = (S - 1) * m.vocabSize
  var lastLogits = allLogits[lastRow ..< lastRow + m.vocabSize]

  (cache, lastLogits)

# ── Backward pass ──────────────────────────────────────────────────

proc backward*(m: var Model, tokens: seq[int32], seqLen: int,
               cache: ForwardCache) =
  ## Backward pass. Computes gradients for all weights.
  ## Must be called after forward() with saveCache=true.
  let S = seqLen
  let n = nEmbd
  let V = m.vocabSize

  # dLogits = (exp(logProbs) - one_hot(target)) / S — entirely on GPU.
  let dLogits = trackedCreate(S * V)
  gpu_ce_backward(cache.logProbs.data, m.targetIdBuf, dLogits.data, cint(S), cint(V))

  # dLmHead += dLogits^T @ finalNormed
  gpuSgemm(5, V, n, S, dLogits, cache.finalNormed, m.dlmHead)
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

    let dMlpOut = trackedCreate(S * n)
    gpuCopy(dx, dMlpOut, S * n)

    # SwiGLU backward
    let dSwigluOut = trackedCreate(S * ffnDim)
    gpuSgemm(1, S, ffnDim, n, dMlpOut, layer.fcDown, dSwigluOut)
    gpuSgemm(5, n, ffnDim, S, dMlpOut, lc.geluOut, m.layers[li].dfcDown)

    let dGate = trackedCreate(S * ffnDim)
    let dUp = trackedCreate(S * ffnDim)
    gpu_swiglu_bwd(lc.fc1Out.data, lc.upOut.data, dSwigluOut.data,
                   dGate.data, dUp.data, cint(S * ffnDim))

    let dNorm2 = trackedCreate(S * n)
    gpuSgemm(1, S, n, ffnDim, dGate, layer.fcGate, dNorm2)
    gpuSgemm(5, ffnDim, n, S, dGate, lc.xNorm2, m.layers[li].dfcGate)

    let dNorm2up = trackedCreate(S * n)
    gpuSgemm(1, S, n, ffnDim, dUp, layer.fcUp, dNorm2up)
    gpuSgemm(5, ffnDim, n, S, dUp, lc.xNorm2, m.layers[li].dfcUp)
    gpu_add_inplace(dNorm2.data, dNorm2up.data, cint(S * n))

    let dResid2 = trackedCreate(S * n)
    gpu_rmsnorm_affine_bwd(lc.x2.data, layer.ln2g.data, dNorm2.data,
                           lc.rms2.data, dResid2.data, m.layers[li].dln2g.data,
                           cint(S), cint(n))
    gpu_add_inplace(dx.data, dResid2.data, cint(S * n))

    # Attention backward
    let dAttnOut = trackedCreate(S * n)
    gpuSgemm(1, S, n, n, dx, layer.wo, dAttnOut)
    gpuSgemm(5, n, n, S, dx, lc.attnOut, m.layers[li].dwo)

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
      gpu_insert_kv_head_acc(dkH.data, dk.data, cint(h), cint(kvRepeat),
                             cint(S), cint(nKvDim), cint(headDim))
      gpu_insert_kv_head_acc(dvH.data, dv.data, cint(h), cint(kvRepeat),
                             cint(S), cint(nKvDim), cint(headDim))

    ropeBwd(dq, m.ropeCos, m.ropeSin, S, n, nHead, headDim)
    ropeBwd(dk, m.ropeCos, m.ropeSin, S, nKvDim, nKvHead, headDim)

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

    let dResid1 = trackedCreate(S * n)
    gpu_rmsnorm_affine_bwd(lc.x1.data, layer.ln1g.data, dNorm1.data,
                           lc.rms1.data, dResid1.data, m.layers[li].dln1g.data,
                           cint(S), cint(n))
    gpu_add(dResid1.data, dx.data, dx.data, cint(S * n))

  # Embedding backward
  gpu_embed_bwd(m.dwte.data, m.tokIdBuf, dx.data, cint(S), cint(n))

# ── Zero gradients ────────────────────────────────────────────────

proc zeroGrads*(m: var Model) =
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

# ── KV Cache for fast inference ───────────────────────────────────

proc initKvCache*(): KvCache =
  ## Pre-allocate KV cache for all layers.
  for i in 0 ..< nLayer:
    result.k.add(gpuCreate(blockSize * nKvDim))
    result.v.add(gpuCreate(blockSize * nKvDim))
  result.pos = 0

proc resetKvCache*(kv: var KvCache) =
  for i in 0 ..< nLayer:
    gpuZero(kv.k[i])
    gpuZero(kv.v[i])
  kv.pos = 0

proc forwardCached*(m: Model, kv: var KvCache, tokens: seq[int32]): seq[float32] =
  ## Forward pass with KV cache. Processes new tokens and appends K/V to cache.
  ## Returns logits for the LAST token only.
  ## Much faster than full forward for autoregressive generation.
  let n = nEmbd
  let newTokens = tokens.len
  let S = kv.pos + newTokens  # total sequence length including cache

  # Embed new tokens only
  var x = trackedCreate(newTokens * n)
  var tokBuf: pointer
  discard cudaMalloc(addr tokBuf, csize_t(newTokens * sizeof(int32)))
  discard cudaMemcpy(tokBuf, unsafeAddr tokens[0],
                     csize_t(newTokens * sizeof(int32)), CudaMemcpyHostToDevice)
  gpu_embed_fwd(m.wte.data, tokBuf, x.data, cint(newTokens), cint(n))
  discard cudaFree(tokBuf)

  let scale = 1.0f / sqrt(float32(headDim))

  for li in 0 ..< nLayer:
    let layer = m.layers[li]

    # RMSNorm
    let xNorm = trackedCreate(newTokens * n)
    let rms = trackedCreate(newTokens)
    gpu_rmsnorm_affine_fwd(x.data, layer.ln1g.data, xNorm.data, rms.data,
                           cint(newTokens), cint(n))

    # QKV for new tokens only
    let q = trackedCreate(newTokens * n)
    let kNew = trackedCreate(newTokens * nKvDim)
    let vNew = trackedCreate(newTokens * nKvDim)
    gpuSgemm(2, newTokens, n, n, xNorm, layer.wq, q)
    gpuSgemm(2, newTokens, nKvDim, n, xNorm, layer.wk, kNew)
    gpuSgemm(2, newTokens, nKvDim, n, xNorm, layer.wv, vNew)

    # RoPE — position starts at kv.pos, not 0
    # Need to offset the RoPE cos/sin tables
    let ropeOffset = kv.pos
    # For simplicity, create offset RoPE tables for new positions
    let halfDim = headDim div 2
    var cosOff = newSeq[float32](newTokens * halfDim)
    var sinOff = newSeq[float32](newTokens * halfDim)
    for pos in 0 ..< newTokens:
      for f in 0 ..< halfDim:
        let theta = float32(pos + ropeOffset) / pow(ropeTheta, 2.0f * float32(f) / float32(headDim))
        cosOff[pos * halfDim + f] = cos(theta)
        sinOff[pos * halfDim + f] = sin(theta)
    let ropeCosOff = toGpu(cosOff)
    let ropeSinOff = toGpu(sinOff)
    ropeFwd(q, ropeCosOff, ropeSinOff, newTokens, n, nHead, headDim)
    ropeFwd(kNew, ropeCosOff, ropeSinOff, newTokens, nKvDim, nKvHead, headDim)

    # Append new K/V to cache
    # kv.k[li] is [blockSize, nKvDim], we write at row kv.pos
    let kDst = cast[pointer](cast[int](kv.k[li].data) + kv.pos * nKvDim * sizeof(float32))
    let vDst = cast[pointer](cast[int](kv.v[li].data) + kv.pos * nKvDim * sizeof(float32))
    discard cudaMemcpy(kDst, kNew.data,
                       csize_t(newTokens * nKvDim * sizeof(float32)),
                       CudaMemcpyDeviceToDevice)
    discard cudaMemcpy(vDst, vNew.data,
                       csize_t(newTokens * nKvDim * sizeof(float32)),
                       CudaMemcpyDeviceToDevice)

    # Attention: Q_new × K_all^T → scores [newTokens, S]
    # K_all is kv.k[li][0:S, nKvDim]
    let qH = trackedCreate(newTokens * headDim)
    let kH = trackedCreate(S * headDim)
    let vH = trackedCreate(S * headDim)
    let attnH = trackedCreate(newTokens * headDim)
    let scores = trackedCreate(newTokens * S)
    let probs = trackedCreate(newTokens * S)

    let attnOut = trackedCreate(newTokens * n)
    for h in 0 ..< nHead:
      # Extract Q head from new tokens [newTokens, headDim]
      extractHead(q, qH, h, newTokens, n, headDim)
      # Extract K/V heads from FULL cache [S, headDim]
      gpu_extract_kv_head(kv.k[li].data, kH.data, cint(h), cint(kvRepeat),
                          cint(S), cint(nKvDim), cint(headDim))
      gpu_extract_kv_head(kv.v[li].data, vH.data, cint(h), cint(kvRepeat),
                          cint(S), cint(nKvDim), cint(headDim))

      # scores = Q_new @ K_all^T  [newTokens, headDim] × [headDim, S] → [newTokens, S]
      gpuSgemm(2, newTokens, S, headDim, qH, kH, scores)
      # Causal mask: position i (absolute: kv.pos+i) can attend to positions 0..kv.pos+i
      # For the score matrix [newTokens, S], entry [i, j] should be masked if j > kv.pos+i
      # Simple approach: use full causal mask on the [newTokens, S] matrix with offset
      # For now, no mask needed if newTokens=1 (single token generation)
      # For prompt processing (newTokens > 1), we need proper masking
      if newTokens == 1:
        # Single token: attends to all S positions, no masking needed
        gpu_scale(scores.data, scale, scores.data, cint(S))
      else:
        # Multi-token: need offset causal mask (TODO: proper implementation)
        causalMask(scores, scale, S)

      softmaxFwd(scores, probs, newTokens, S)
      # output = probs @ V_all  [newTokens, S] × [S, headDim] → [newTokens, headDim]
      gpuSgemm(0, newTokens, headDim, S, probs, vH, attnH)
      insertHead(attnH, attnOut, h, newTokens, n, headDim)

    # Output projection + residual
    let projected = trackedCreate(newTokens * n)
    gpuSgemm(2, newTokens, n, n, attnOut, layer.wo, projected)
    let x2 = trackedCreate(newTokens * n)
    gpu_add(x.data, projected.data, x2.data, cint(newTokens * n))

    # FFN
    let xNorm2 = trackedCreate(newTokens * n)
    let rms2 = trackedCreate(newTokens)
    gpu_rmsnorm_affine_fwd(x2.data, layer.ln2g.data, xNorm2.data, rms2.data,
                           cint(newTokens), cint(n))

    let gateOut = trackedCreate(newTokens * ffnDim)
    gpuSgemm(2, newTokens, ffnDim, n, xNorm2, layer.fcGate, gateOut)
    let upOut = trackedCreate(newTokens * ffnDim)
    gpuSgemm(2, newTokens, ffnDim, n, xNorm2, layer.fcUp, upOut)
    let swigluOut = trackedCreate(newTokens * ffnDim)
    gpu_swiglu_fwd(gateOut.data, upOut.data, swigluOut.data, cint(newTokens * ffnDim))
    let mlpOut = trackedCreate(newTokens * n)
    gpuSgemm(2, newTokens, n, ffnDim, swigluOut, layer.fcDown, mlpOut)

    let xNew = trackedCreate(newTokens * n)
    gpu_add(x2.data, mlpOut.data, xNew.data, cint(newTokens * n))
    x = xNew

  # Final norm + logits
  let finalNormed = trackedCreate(newTokens * n)
  let rmsF = trackedCreate(newTokens)
  gpu_rmsnorm_affine_fwd(x.data, m.lnFg.data, finalNormed.data, rmsF.data,
                         cint(newTokens), cint(n))

  let logits = trackedCreate(newTokens * m.vocabSize)
  gpuSgemm(2, newTokens, m.vocabSize, n, finalNormed, m.lmHead, logits)

  # Update cache position
  kv.pos += newTokens

  # Return logits for last token
  let allLogits = gpuDownload(logits)
  let lastRow = (newTokens - 1) * m.vocabSize
  result = allLogits[lastRow ..< lastRow + m.vocabSize]
