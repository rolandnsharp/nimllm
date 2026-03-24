## model.nim — nimllm model: forward pass + backward pass
##
## The complete model in one file. Forward and backward are two halves
## of the same operation — separating them would require duplicating
## the forward pass. Shared by training and inference.

import gpu, autograd, gguf
import std/[math, random, streams, strformat, tables]

# ── Configuration ─────────────────────────────────────────────────

const
  nLayer*   = 16
  nEmbd*    = 2048
  nHead*    = 32
  nKvHead*  = 8
  headDim*  = nEmbd div nHead  # 64
  kvRepeat* = nHead div nKvHead  # 4
  nKvDim*   = nKvHead * headDim  # 512
  blockSize* = 512
  ffnDim*   = 8192
  ropeTheta* = 500000.0f
  frozenLayers* = 12

var gQuantType*: int = 0  # 0=Q4_0, 8=Q8_0, set during model load     # freeze bottom layers for fine-tuning (0 = train all)

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
    # Q4_0 quantized weights (inference only — 7x smaller)
    wq_q4*, wk_q4*, wv_q4*, wo_q4*: pointer
    fcGate_q4*, fcUp_q4*, fcDown_q4*: pointer

  Model* = object
    wte*, lmHead*, lnFg*: GpuBuf
    layers*: seq[Layer]
    vocabSize*: int
    quantized*: bool        # true if quantized weights are available
    quantType*: int         # 0=Q4_0, 8=Q8_0
    dwte*, dlmHead*, dlnFg*: GpuBuf
    tokIdBuf*: pointer
    targetIdBuf*: pointer
    ropeCos*, ropeSin*: GpuBuf
    # Q4_0 quantized (inference only)
    lmHead_q4*: pointer

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
  # Stored per-head for zero-copy head access: [nLayer][nKvHead] each [blockSize, headDim]
  KvCache* = object
    k*: seq[seq[GpuBuf]]  # [nLayer][nKvHead] each [blockSize, headDim]
    v*: seq[seq[GpuBuf]]  # [nLayer][nKvHead] each [blockSize, headDim]
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

proc initModel*(vocabSize: int, withGradients: bool = true, withWeights: bool = true): Model =
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
  if withWeights:
    result.wte = randBuf(vocabSize * nEmbd)
    result.lmHead = randBuf(vocabSize * nEmbd)
  else:
    # GGUF mode: allocate float32 only for embeddings (needed for lookup)
    # and norms. Weight matrices loaded as Q8_0 directly.
    result.wte = gpuCreate(vocabSize * nEmbd)
    result.lmHead = gpuCreate(vocabSize * nEmbd)
  result.lnFg = onesBuf(nEmbd)

  if withGradients:
    result.dwte = gpuCreate(vocabSize * nEmbd)
    result.dlmHead = gpuCreate(vocabSize * nEmbd)
    result.dlnFg = gpuCreate(nEmbd)

  for i in 0 ..< nLayer:
    let resStd = std / sqrt(float32(2 * nLayer))
    var layer: Layer
    if withWeights:
      layer = Layer(
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
    else:
      # GGUF mode: only allocate norms (weight matrices come from GGUF Q8_0)
      layer.ln1g = onesBuf(nEmbd)
      layer.ln2g = onesBuf(nEmbd)
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

# ── Q4_0 Quantization for fast inference ─────────────────────────

proc loadModelGguf*(m: var Model, ggufPath: string) =
  ## Load model directly from GGUF file (Ollama format).
  ## Q8_0 weights load directly — no float32 intermediate, no OOM.
  echo "loading GGUF from ", ggufPath, "..."
  let gf = openGguf(ggufPath)

  proc uploadQ8(name: string): pointer =
    ## Load Q8_0 tensor raw bytes to GPU
    let raw = gf.loadTensorRaw(name)
    if raw.len == 0: return nil
    var p: pointer
    let err = cudaMalloc(addr p, csize_t(raw.len))
    assert err == CudaSuccess, "cudaMalloc failed for " & name
    discard cudaMemcpy(p, unsafeAddr raw[0], csize_t(raw.len), CudaMemcpyHostToDevice)
    p

  proc uploadF32(name: string, buf: GpuBuf) =
    ## Load F32 tensor and upload to existing GpuBuf
    let raw = gf.loadTensorRaw(name)
    if raw.len == 0: return
    # Raw is already float32 bytes
    assert raw.len == buf.numel * sizeof(float32), name & " size mismatch"
    discard cudaMemcpy(buf.data, unsafeAddr raw[0], csize_t(raw.len), CudaMemcpyHostToDevice)

  proc f16toF32(b0, b1: uint8): float32 =
    ## Convert IEEE float16 (2 bytes, little-endian) to float32
    let h = (b1.uint16 shl 8) or b0.uint16
    let sign = (h shr 15) and 1
    let expo = (h shr 10) and 0x1F
    let mant = h and 0x3FF
    if expo == 0:
      # Subnormal or zero
      if mant == 0: return if sign == 1: -0.0f else: 0.0f
      var m = mant.float32 / 1024.0f
      var e = -14
      result = (if sign == 1: -1.0f else: 1.0f) * m * pow(2.0f, e.float32)
    elif expo == 31:
      return if mant != 0: NaN else: (if sign == 1: -Inf else: Inf)
    else:
      let e = expo.int - 15
      let m = 1.0f + mant.float32 / 1024.0f
      result = (if sign == 1: -1.0f else: 1.0f) * m * pow(2.0f, e.float32)

  proc dequantQ8toF32(name: string, buf: GpuBuf) =
    ## Load Q8_0 tensor, dequantize on CPU, upload as float32
    let raw = gf.loadTensorRaw(name)
    if raw.len == 0: return
    let nBlocks = raw.len div 34  # 34 bytes per Q8_0 block
    var f32 = newSeq[float32](nBlocks * 32)
    var pos = 0
    for b in 0 ..< nBlocks:
      let base = b * 34
      # Scale is IEEE float16 (NOT BF16!)
      let scale = f16toF32(raw[base], raw[base + 1])
      # Dequant 32 int8 values
      for i in 0 ..< 32:
        let v = cast[int8](raw[base + 2 + i])
        f32[pos] = scale * float32(v)
        pos += 1
    gpuUpload(buf, f32)

  # Embeddings: dequant Q8_0 → float32 (lookup table, needs float32)
  echo "  loading embeddings..."
  dequantQ8toF32("token_embd.weight", m.wte)
  if "output.weight" in gf.tensors:
    dequantQ8toF32("output.weight", m.lmHead)
  else:
    # Tied weights — copy from wte
    gpuCopy(m.wte, m.lmHead, m.wte.numel)

  # Final norm (F32)
  uploadF32("output_norm.weight", m.lnFg)

  # Layers: load Q8_0 weight matrices directly (no dequant)
  for li in 0 ..< nLayer:
    echo &"  layer {li+1}/{nLayer}"
    let p = &"blk.{li}"
    m.layers[li].wq_q4 = uploadQ8(&"{p}.attn_q.weight")
    m.layers[li].wk_q4 = uploadQ8(&"{p}.attn_k.weight")
    m.layers[li].wv_q4 = uploadQ8(&"{p}.attn_v.weight")
    m.layers[li].wo_q4 = uploadQ8(&"{p}.attn_output.weight")
    m.layers[li].fcGate_q4 = uploadQ8(&"{p}.ffn_gate.weight")
    m.layers[li].fcUp_q4 = uploadQ8(&"{p}.ffn_up.weight")
    m.layers[li].fcDown_q4 = uploadQ8(&"{p}.ffn_down.weight")
    # Norms (F32)
    uploadF32(&"{p}.attn_norm.weight", m.layers[li].ln1g)
    uploadF32(&"{p}.ffn_norm.weight", m.layers[li].ln2g)

  m.quantized = true
  m.quantType = 8  # Q8_0
  gQuantType = 8
  echo "  done (GGUF Q8_0 loaded directly)"

proc quantizeQ4(buf: GpuBuf): pointer =
  ## Quantize a float32 GPU buffer to Q4_0. Returns pointer to Q4_0 data.
  ## Q4_0: 18 bytes per 32 floats (block_q4_0 = {half d, uint8 qs[16]}).
  let numBlocks = buf.numel div 32
  let q4Size = numBlocks * 18  # 18 bytes per block
  var q4ptr: pointer
  let err = cudaMalloc(addr q4ptr, csize_t(q4Size))
  assert err == CudaSuccess, "Q4_0 cudaMalloc failed"
  gpu_quantize_q4_0(buf.data, q4ptr, cint(buf.numel))
  q4ptr

proc quantizeModel*(m: var Model) =
  ## Quantize all weight matrices to Q4_0 for fast inference.
  ## Keeps float32 weights for training. Q4_0 used only in forwardCached.
  echo "quantizing to Q4_0..."
  m.lmHead_q4 = quantizeQ4(m.lmHead)
  for i in 0 ..< m.layers.len:
    m.layers[i].wq_q4 = quantizeQ4(m.layers[i].wq)
    m.layers[i].wk_q4 = quantizeQ4(m.layers[i].wk)
    m.layers[i].wv_q4 = quantizeQ4(m.layers[i].wv)
    m.layers[i].wo_q4 = quantizeQ4(m.layers[i].wo)
    m.layers[i].fcGate_q4 = quantizeQ4(m.layers[i].fcGate)
    m.layers[i].fcUp_q4 = quantizeQ4(m.layers[i].fcUp)
    m.layers[i].fcDown_q4 = quantizeQ4(m.layers[i].fcDown)
  m.quantized = true
  let origMB = (m.vocabSize * nEmbd * 2 + nLayer * (2*nEmbd*nEmbd + 2*nKvDim*nEmbd + 3*ffnDim*nEmbd)) * 4 div (1024*1024)
  let q4MB = (m.vocabSize * nEmbd + nLayer * (nEmbd*nEmbd + nKvDim*nEmbd + nKvDim*nEmbd + nEmbd*nEmbd + ffnDim*nEmbd + ffnDim*nEmbd + nEmbd*ffnDim)) * 18 div 32 div (1024*1024)
  echo &"  done ({origMB}MB float32 → ~{q4MB}MB Q4_0)"

# ── Quantized matvec dispatch ─────────────────────────────────────

proc quantMatvec(A: pointer, x: pointer, y: pointer, rows, cols: cint) =
  ## Dispatch to the right dequant kernel based on quantization type
  if gQuantType == 8:
    gpu_matvec_q8_0(A, x, y, rows, cols)
  else:
    gpu_matvec_q4_0(A, x, y, rows, cols)

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
    # Skip frozen layers — no gradients needed
    if li < frozenLayers:
      break
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
  ## Pre-allocate KV cache: [nLayer][nKvHead] each [blockSize, headDim]
  for li in 0 ..< nLayer:
    var kHeads: seq[GpuBuf]
    var vHeads: seq[GpuBuf]
    for h in 0 ..< nKvHead:
      kHeads.add(gpuCreate(blockSize * headDim))
      vHeads.add(gpuCreate(blockSize * headDim))
    result.k.add(kHeads)
    result.v.add(vHeads)
  result.pos = 0

proc resetKvCache*(kv: var KvCache) =
  for li in 0 ..< nLayer:
    for h in 0 ..< nKvHead:
      gpuZero(kv.k[li][h])
      gpuZero(kv.v[li][h])
  kv.pos = 0

proc forwardCached*(m: Model, kv: var KvCache, tokens: seq[int32]): seq[float32] =
  ## Forward pass with KV cache. Processes new tokens and appends K/V to cache.
  ## Returns logits for the LAST token only.
  ## Much faster than full forward for autoregressive generation.
  let n = nEmbd
  let newTokens = tokens.len
  let S = kv.pos + newTokens  # total sequence length including cache

  # Embed new tokens only (reuse model's pre-allocated tokIdBuf)
  var x = trackedCreate(newTokens * n)
  discard cudaMemcpy(m.tokIdBuf, unsafeAddr tokens[0],
                     csize_t(newTokens * sizeof(int32)), CudaMemcpyHostToDevice)
  gpu_embed_fwd(m.wte.data, m.tokIdBuf, x.data, cint(newTokens), cint(n))

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
    if m.quantized and newTokens == 1:
      # Q4_0 matvec: 7x less memory bandwidth
      quantMatvec(layer.wq_q4, xNorm.data, q.data, cint(n), cint(n))
      quantMatvec(layer.wk_q4, xNorm.data, kNew.data, cint(nKvDim), cint(n))
      quantMatvec(layer.wv_q4, xNorm.data, vNew.data, cint(nKvDim), cint(n))
    else:
      gpuSgemm(2, newTokens, n, n, xNorm, layer.wq, q)
      gpuSgemm(2, newTokens, nKvDim, n, xNorm, layer.wk, kNew)
      gpuSgemm(2, newTokens, nKvDim, n, xNorm, layer.wv, vNew)

    # RoPE — use model's pre-computed tables with position offset
    # The tables are [blockSize, halfDim]. We need rows starting at kv.pos.
    let halfDim = headDim div 2
    let ropeOffset = kv.pos * halfDim * sizeof(float32)
    let cosPtr = cast[pointer](cast[int](m.ropeCos.data) + ropeOffset)
    let sinPtr = cast[pointer](cast[int](m.ropeSin.data) + ropeOffset)
    var ropeCosOff = GpuBuf(data: cosPtr, numel: newTokens * halfDim)
    var ropeSinOff = GpuBuf(data: sinPtr, numel: newTokens * halfDim)
    ropeFwd(q, ropeCosOff, ropeSinOff, newTokens, n, nHead, headDim)
    ropeFwd(kNew, ropeCosOff, ropeSinOff, newTokens, nKvDim, nKvHead, headDim)

    # Append new K/V to per-head cache buffers
    # kNew is [newTokens, nKvDim] — split into nKvHead heads of [newTokens, headDim]
    for kvh in 0 ..< nKvHead:
      let srcK = cast[pointer](cast[int](kNew.data) + kvh * headDim * sizeof(float32))
      let srcV = cast[pointer](cast[int](vNew.data) + kvh * headDim * sizeof(float32))
      let dstK = cast[pointer](cast[int](kv.k[li][kvh].data) + kv.pos * headDim * sizeof(float32))
      let dstV = cast[pointer](cast[int](kv.v[li][kvh].data) + kv.pos * headDim * sizeof(float32))
      # For newTokens=1 and contiguous headDim, direct copy works
      if newTokens == 1:
        discard cudaMemcpy(dstK, srcK, csize_t(headDim * sizeof(float32)), CudaMemcpyDeviceToDevice)
        discard cudaMemcpy(dstV, srcV, csize_t(headDim * sizeof(float32)), CudaMemcpyDeviceToDevice)
      else:
        # Multi-token: need to extract per-head from interleaved layout
        gpu_extract_kv_head(kNew.data, dstK, cint(kvh * kvRepeat), cint(kvRepeat),
                            cint(newTokens), cint(nKvDim), cint(headDim))
        gpu_extract_kv_head(vNew.data, dstV, cint(kvh * kvRepeat), cint(kvRepeat),
                            cint(newTokens), cint(nKvDim), cint(headDim))

    # Attention with per-head KV cache — no extract/insert kernels needed
    let scores = trackedCreate(newTokens * S)
    let probs = trackedCreate(newTokens * S)
    let attnOut = trackedCreate(newTokens * n)

    for h in 0 ..< nHead:
      let kvh = h div kvRepeat
      # Q head: pointer offset into q[1, nEmbd] at h*headDim
      let qPtr = cast[pointer](cast[int](q.data) + h * headDim * sizeof(float32))
      var qH = GpuBuf(data: qPtr, numel: newTokens * headDim)
      # K/V: direct from per-head cache [S, headDim] — zero copy!
      var kH = GpuBuf(data: kv.k[li][kvh].data, numel: S * headDim)
      var vH = GpuBuf(data: kv.v[li][kvh].data, numel: S * headDim)
      # Attention output: pointer offset into attnOut
      let outPtr = cast[pointer](cast[int](attnOut.data) + h * headDim * sizeof(float32))
      var attnH = GpuBuf(data: outPtr, numel: newTokens * headDim)

      gpuSgemm(2, newTokens, S, headDim, qH, kH, scores)
      if newTokens == 1:
        gpu_scale(scores.data, scale, scores.data, cint(S))
      else:
        causalMask(scores, scale, S)
      softmaxFwd(scores, probs, newTokens, S)
      gpuSgemm(0, newTokens, headDim, S, probs, vH, attnH)

    # Output projection + residual
    let projected = trackedCreate(newTokens * n)
    if m.quantized and newTokens == 1:
      quantMatvec(layer.wo_q4, attnOut.data, projected.data, cint(n), cint(n))
    else:
      gpuSgemm(2, newTokens, n, n, attnOut, layer.wo, projected)
    let x2 = trackedCreate(newTokens * n)
    gpu_add(x.data, projected.data, x2.data, cint(newTokens * n))

    # FFN
    let xNorm2 = trackedCreate(newTokens * n)
    let rms2 = trackedCreate(newTokens)
    gpu_rmsnorm_affine_fwd(x2.data, layer.ln2g.data, xNorm2.data, rms2.data,
                           cint(newTokens), cint(n))

    let gateOut = trackedCreate(newTokens * ffnDim)
    let upOut = trackedCreate(newTokens * ffnDim)
    if m.quantized and newTokens == 1:
      quantMatvec(layer.fcGate_q4, xNorm2.data, gateOut.data, cint(ffnDim), cint(n))
      quantMatvec(layer.fcUp_q4, xNorm2.data, upOut.data, cint(ffnDim), cint(n))
    else:
      gpuSgemm(2, newTokens, ffnDim, n, xNorm2, layer.fcGate, gateOut)
      gpuSgemm(2, newTokens, ffnDim, n, xNorm2, layer.fcUp, upOut)
    let swigluOut = trackedCreate(newTokens * ffnDim)
    gpu_swiglu_fwd(gateOut.data, upOut.data, swigluOut.data, cint(newTokens * ffnDim))
    let mlpOut = trackedCreate(newTokens * n)
    if m.quantized and newTokens == 1:
      quantMatvec(layer.fcDown_q4, swigluOut.data, mlpOut.data, cint(n), cint(ffnDim))
    else:
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
  if m.quantized and newTokens == 1:
    quantMatvec(m.lmHead_q4, finalNormed.data, logits.data, cint(m.vocabSize), cint(n))
  else:
    gpuSgemm(2, newTokens, m.vocabSize, n, finalNormed, m.lmHead, logits)

  # Update cache position
  kv.pos += newTokens

  # Return logits for last token
  let allLogits = gpuDownload(logits)
  let lastRow = (newTokens - 1) * m.vocabSize
  result = allLogits[lastRow ..< lastRow + m.vocabSize]
