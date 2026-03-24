## model.nim — nimllm model definition and forward pass
##
## Shared between training (microgpt.nim) and inference (chat.nim).
## One forward pass. Two consumers. Zero duplication.

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
