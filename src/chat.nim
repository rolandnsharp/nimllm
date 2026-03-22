## chat.nim — Talk to your NimLLM
##
## Loads the trained model, generates text token by token.
## Inspired by Girvent's clean Nim TUI.

import gpu, bpe, autograd
import std/[math, random, strformat, os, streams, terminal, strutils]

# Must match microgpt.nim constants exactly
const
  nLayer   = 12
  nEmbd    = 768
  nHead    = 12
  headDim  = nEmbd div nHead
  blockSize = 512
  ffnMul   = 4

# Import the model type and forward from microgpt
# For now, duplicate the minimal types needed

type
  Layer = object
    wq, wk, wv, wo: GpuBuf
    fc1, fc2: GpuBuf
    ln1g, ln2g: GpuBuf
    dwq, dwk, dwv, dwo: GpuBuf
    dfc1, dfc2: GpuBuf
    dln1g, dln2g: GpuBuf

  Model = object
    wte, wpe, lmHead, lnFg: GpuBuf
    layers: seq[Layer]
    vocabSize: int
    dwte, dwpe, dlmHead, dlnFg: GpuBuf

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
  proc onesBuf(n: int): GpuBuf =
    var h = newSeq[float32](n)
    for i in 0 ..< n: h[i] = 1.0f
    toGpu(h)

  result.vocabSize = vocabSize
  result.wte = randBuf(vocabSize * nEmbd)
  result.wpe = randBuf(blockSize * nEmbd)
  result.lmHead = randBuf(vocabSize * nEmbd)
  result.lnFg = onesBuf(nEmbd)
  result.dwte = gpuCreate(vocabSize * nEmbd)
  result.dwpe = gpuCreate(blockSize * nEmbd)
  result.dlmHead = gpuCreate(vocabSize * nEmbd)
  result.dlnFg = gpuCreate(nEmbd)
  let resStd = std / sqrt(float32(2 * nLayer))
  for i in 0 ..< nLayer:
    result.layers.add(Layer(
      wq: randBuf(nEmbd * nEmbd), wk: randBuf(nEmbd * nEmbd),
      wv: randBuf(nEmbd * nEmbd), wo: randBuf(nEmbd * nEmbd, resStd),
      fc1: randBuf(ffnMul * nEmbd * nEmbd),
      fc2: randBuf(nEmbd * ffnMul * nEmbd, resStd),
      ln1g: onesBuf(nEmbd), ln2g: onesBuf(nEmbd),
      dwq: gpuCreate(nEmbd * nEmbd), dwk: gpuCreate(nEmbd * nEmbd),
      dwv: gpuCreate(nEmbd * nEmbd), dwo: gpuCreate(nEmbd * nEmbd),
      dfc1: gpuCreate(ffnMul * nEmbd * nEmbd),
      dfc2: gpuCreate(nEmbd * ffnMul * nEmbd),
      dln1g: gpuCreate(nEmbd), dln2g: gpuCreate(nEmbd),
    ))

proc loadModel(m: var Model, filename: string) =
  echo "loading brain from ", filename, "..."
  let s = newFileStream(filename, fmRead)
  defer: s.close()
  proc r(buf: GpuBuf) =
    let n = s.readInt32().int
    var d = newSeq[float32](n)
    for i in 0 ..< n: d[i] = s.readFloat32()
    gpuUpload(buf, d)
  r(m.wte); r(m.wpe); r(m.lmHead); r(m.lnFg)
  for i in 0 ..< m.layers.len:
    r(m.layers[i].wq); r(m.layers[i].wk)
    r(m.layers[i].wv); r(m.layers[i].wo)
    r(m.layers[i].fc1); r(m.layers[i].fc2)
    r(m.layers[i].ln1g); r(m.layers[i].ln2g)
  echo "  done"

# ── Single-token forward (for generation) ─────────────────────────

proc forwardOneToken(m: Model, tokenIds: seq[int32], seqLen: int): seq[float32] =
  ## Forward pass, returns logits for the LAST position as CPU array.
  let n = nEmbd
  let S = seqLen

  # Embedding
  let tokEmb = trackedCreate(S * n)
  var tokBuf: pointer
  discard cudaMalloc(addr tokBuf, csize_t(S * sizeof(int32)))
  discard cudaMemcpy(tokBuf, unsafeAddr tokenIds[0],
                     csize_t(S * sizeof(int32)), CudaMemcpyHostToDevice)
  gpu_embed_fwd(m.wte.data, tokBuf, tokEmb.data, cint(S), cint(n))

  let posEmb = trackedCreate(S * n)
  var posIds = newSeq[int32](S)
  for i in 0 ..< S: posIds[i] = int32(i)
  var posBuf: pointer
  discard cudaMalloc(addr posBuf, csize_t(S * sizeof(int32)))
  discard cudaMemcpy(posBuf, unsafeAddr posIds[0],
                     csize_t(S * sizeof(int32)), CudaMemcpyHostToDevice)
  gpu_embed_fwd(m.wpe.data, posBuf, posEmb.data, cint(S), cint(n))

  var x = trackedCreate(S * n)
  gpu_add(tokEmb.data, posEmb.data, x.data, cint(S * n))
  discard cudaFree(tokBuf)
  discard cudaFree(posBuf)

  let scale = 1.0f / sqrt(float32(headDim))
  let qH = trackedCreate(S * headDim)
  let kH = trackedCreate(S * headDim)
  let vH = trackedCreate(S * headDim)
  let attnH = trackedCreate(S * headDim)
  let scores = trackedCreate(S * S)
  let probs = trackedCreate(S * S)

  for li in 0 ..< nLayer:
    let layer = m.layers[li]
    let xNorm = trackedCreate(S * n)
    let rms = trackedCreate(S)
    gpu_rmsnorm_affine_fwd(x.data, layer.ln1g.data, xNorm.data, rms.data, cint(S), cint(n))

    let q = trackedCreate(S * n)
    let k = trackedCreate(S * n)
    let v = trackedCreate(S * n)
    gpuSgemm(2, S, n, n, xNorm, layer.wq, q)
    gpuSgemm(2, S, n, n, xNorm, layer.wk, k)
    gpuSgemm(2, S, n, n, xNorm, layer.wv, v)

    # cuBLAS attention: fast matmuls + custom softmax/mask
    let attnOut = trackedCreate(S * n)
    for h in 0 ..< nHead:
      extractHead(q, qH, h, S, n, headDim)
      extractHead(k, kH, h, S, n, headDim)
      extractHead(v, vH, h, S, n, headDim)
      gpuSgemm(2, S, S, headDim, qH, kH, scores)
      causalMask(scores, scale, S)
      softmaxFwd(scores, probs, S, S)
      gpuSgemm(0, S, headDim, S, probs, vH, attnH)
      insertHead(attnH, attnOut, h, S, n, headDim)

    let projected = trackedCreate(S * n)
    gpuSgemm(2, S, n, n, attnOut, layer.wo, projected)
    let x2 = trackedCreate(S * n)
    gpu_add(x.data, projected.data, x2.data, cint(S * n))

    let xNorm2 = trackedCreate(S * n)
    let rms2 = trackedCreate(S)
    gpu_rmsnorm_affine_fwd(x2.data, layer.ln2g.data, xNorm2.data, rms2.data, cint(S), cint(n))

    let fc1Out = trackedCreate(S * ffnMul * n)
    gpuSgemm(2, S, ffnMul * n, n, xNorm2, layer.fc1, fc1Out)
    let geluOut = trackedCreate(S * ffnMul * n)
    gpu_gelu_fwd(fc1Out.data, geluOut.data, cint(S * ffnMul * n))
    let mlpOut = trackedCreate(S * n)
    gpuSgemm(2, S, n, ffnMul * n, geluOut, layer.fc2, mlpOut)

    let xNew = trackedCreate(S * n)
    gpu_add(x2.data, mlpOut.data, xNew.data, cint(S * n))
    x = xNew

  let finalNormed = trackedCreate(S * n)
  let rmsF = trackedCreate(S)
  gpu_rmsnorm_affine_fwd(x.data, m.lnFg.data, finalNormed.data, rmsF.data, cint(S), cint(n))

  let logits = trackedCreate(S * m.vocabSize)
  gpuSgemm(2, S, m.vocabSize, n, finalNormed, m.lmHead, logits)

  # Download last row of logits
  let allLogits = gpuDownload(logits)
  let lastRow = (S - 1) * m.vocabSize
  result = allLogits[lastRow ..< lastRow + m.vocabSize]

# ── Sampling ──────────────────────────────────────────────────────

proc sample(logits: seq[float32], temperature: float32 = 0.8f,
            topK: int = 40): int =
  ## Sample a token from logits with temperature and top-k.
  var scaled = newSeq[float32](logits.len)
  for i in 0 ..< logits.len:
    scaled[i] = logits[i] / temperature

  # Top-k: keep only the k highest logits
  var indices = newSeq[int](logits.len)
  for i in 0 ..< indices.len: indices[i] = i
  # Simple partial sort: find top-k by selecting max k times
  var topIndices = newSeq[int](topK)
  var used = newSeq[bool](logits.len)
  for k in 0 ..< topK:
    var bestI = -1
    var bestV = -1e30f
    for i in 0 ..< scaled.len:
      if not used[i] and scaled[i] > bestV:
        bestV = scaled[i]
        bestI = i
    topIndices[k] = bestI
    used[bestI] = true

  # Softmax over top-k
  var maxVal = -1e30f
  for i in topIndices:
    if scaled[i] > maxVal: maxVal = scaled[i]
  var probs = newSeq[float32](topK)
  var total = 0.0f
  for k in 0 ..< topK:
    probs[k] = exp(scaled[topIndices[k]] - maxVal)
    total += probs[k]
  for k in 0 ..< topK:
    probs[k] /= total

  # Weighted random choice
  var r = rand(1.0).float32
  for k in 0 ..< topK:
    r -= probs[k]
    if r <= 0:
      return topIndices[k]
  topIndices[topK - 1]

# ── Main ──────────────────────────────────────────────────────────

when isMainModule:
  let baseDir = getAppDir().parentDir()
  let vidyaRoot = baseDir.parentDir()
  let tokenizerFile = vidyaRoot / "tokenizer_nim.bin"
  let modelFile = vidyaRoot / "nimllm.bin"

  gpuInit()

  # Load tokenizer
  if not fileExists(tokenizerFile):
    echo "no tokenizer found at ", tokenizerFile
    quit(1)
  let tok = loadTokenizer(tokenizerFile)

  # Load model
  var m = initModel(tok.vocab.len)
  if fileExists(modelFile):
    loadModel(m, modelFile)
  else:
    echo "no model found at ", modelFile
    echo "train first: ./src/microgpt"
    quit(1)

  randomize()
  trackingEnabled = true

  echo ""
  styledEcho(styleBright, "  nimllm", resetStyle,
             fgBlack, "  ·  ", resetStyle,
             "28M params  ·  loss 2.0")
  styledEcho(fgBlack, "  type to chat  ·  ctrl-c to exit")
  echo ""

  var history: seq[int32]  # token history for context

  while true:
    stdout.styledWrite(fgGreen, "> ")
    stdout.flushFile()
    let input = stdin.readLine().strip()
    if input.len == 0: continue

    # Encode input
    let inputTokens = tok.encode("<|user|> " & input & " <|assistant|> ")
    for id in inputTokens:
      history.add(int32(id))

    # Generate response
    var response = ""
    var genTokens: seq[int32]

    for _ in 0 ..< 200:  # max generation length
      # Build context: keep last blockSize tokens
      var context = history
      if context.len > blockSize:
        context = context[context.len - blockSize ..< context.len]

      let logits = forwardOneToken(m, context, context.len)
      freeStepAllocations()  # free GPU buffers from this forward pass
      let tokenId = sample(logits)

      # Stop on special tokens
      if tokenId == tok.bosId or tokenId == tok.userId or
         tokenId == tok.assistantId:
        break

      history.add(int32(tokenId))
      genTokens.add(int32(tokenId))

      let tokenStr = tok.vocab[tokenId]
      stdout.write(tokenStr)
      stdout.flushFile()

    echo ""
    echo ""
