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
  nKvHead  = 4
  headDim  = nEmbd div nHead  # 64
  kvRepeat = nHead div nKvHead  # 3
  nKvDim   = nKvHead * headDim  # 256
  blockSize = 512
  ffnMul   = 4
  ropeTheta = 500000.0f

# Import the model type and forward from microgpt
# For now, duplicate the minimal types needed

type
  Layer = object
    wq, wk, wv, wo: GpuBuf
    fcGate, fcUp, fcDown: GpuBuf
    ln1g, ln2g: GpuBuf
    dwq, dwk, dwv, dwo: GpuBuf
    dfcGate, dfcUp, dfcDown: GpuBuf
    dln1g, dln2g: GpuBuf

  Model = object
    wte, lmHead, lnFg: GpuBuf
    layers: seq[Layer]
    vocabSize: int
    dwte, dlmHead, dlnFg: GpuBuf
    ropeCos, ropeSin: GpuBuf

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
  result.lmHead = randBuf(vocabSize * nEmbd)
  result.lnFg = onesBuf(nEmbd)
  result.dwte = gpuCreate(vocabSize * nEmbd)
  result.dlmHead = gpuCreate(vocabSize * nEmbd)
  result.dlnFg = gpuCreate(nEmbd)
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
  let resStd = std / sqrt(float32(2 * nLayer))
  for i in 0 ..< nLayer:
    result.layers.add(Layer(
      wq: randBuf(nEmbd * nEmbd),
      wk: randBuf(nKvDim * nEmbd),
      wv: randBuf(nKvDim * nEmbd),
      wo: randBuf(nEmbd * nEmbd, resStd),
      fcGate: randBuf(ffnMul * nEmbd * nEmbd),
      fcUp: randBuf(ffnMul * nEmbd * nEmbd),
      fcDown: randBuf(nEmbd * ffnMul * nEmbd, resStd),
      ln1g: onesBuf(nEmbd), ln2g: onesBuf(nEmbd),
      dwq: gpuCreate(nEmbd * nEmbd),
      dwk: gpuCreate(nKvDim * nEmbd),
      dwv: gpuCreate(nKvDim * nEmbd),
      dwo: gpuCreate(nEmbd * nEmbd),
      dfcGate: gpuCreate(ffnMul * nEmbd * nEmbd),
      dfcUp: gpuCreate(ffnMul * nEmbd * nEmbd),
      dfcDown: gpuCreate(nEmbd * ffnMul * nEmbd),
      dln1g: gpuCreate(nEmbd), dln2g: gpuCreate(nEmbd),
    ))

proc loadModel(m: var Model, filename: string) =
  echo "loading brain from ", filename, "..."
  let s = newFileStream(filename, fmRead)
  defer: s.close()
  # Skip v2 header if present
  var magic: array[4, char]
  discard s.readData(addr magic[0], 4)
  if magic == ['N', 'L', 'L', 'M']:
    discard s.readInt32()  # version
    discard s.readInt32()  # nLayer
    discard s.readInt32()  # nEmbd
    discard s.readInt32()  # nHead
    discard s.readInt32()  # vocabSize
    discard s.readInt32()  # blockSize
  else:
    s.setPosition(0)  # legacy format, rewind
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

# ── Single-token forward (for generation) ─────────────────────────

proc forwardOneToken(m: Model, tokenIds: seq[int32], seqLen: int): seq[float32] =
  ## Forward pass, returns logits for the LAST position as CPU array.
  let n = nEmbd
  let S = seqLen

  # Token embedding (RoPE handles position — no positional embedding)
  var x = trackedCreate(S * n)
  var tokBuf: pointer
  discard cudaMalloc(addr tokBuf, csize_t(S * sizeof(int32)))
  discard cudaMemcpy(tokBuf, unsafeAddr tokenIds[0],
                     csize_t(S * sizeof(int32)), CudaMemcpyHostToDevice)
  gpu_embed_fwd(m.wte.data, tokBuf, x.data, cint(S), cint(n))
  discard cudaFree(tokBuf)

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
    let k = trackedCreate(S * nKvDim)
    let v = trackedCreate(S * nKvDim)
    gpuSgemm(2, S, n, n, xNorm, layer.wq, q)
    gpuSgemm(2, S, nKvDim, n, xNorm, layer.wk, k)
    gpuSgemm(2, S, nKvDim, n, xNorm, layer.wv, v)

    # RoPE: rotate Q and K by position
    ropeFwd(q, m.ropeCos, m.ropeSin, S, n, nHead, headDim)
    ropeFwd(k, m.ropeCos, m.ropeSin, S, nKvDim, nKvHead, headDim)

    # GQA attention via cuBLAS
    let attnOut = trackedCreate(S * n)
    for h in 0 ..< nHead:
      extractHead(q, qH, h, S, n, headDim)
      gpu_extract_kv_head(k.data, kH.data, cint(h), cint(kvRepeat),
                          cint(S), cint(nKvDim), cint(headDim))
      gpu_extract_kv_head(v.data, vH.data, cint(h), cint(kvRepeat),
                          cint(S), cint(nKvDim), cint(headDim))
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

    # SwiGLU FFN
    let gateOut = trackedCreate(S * ffnMul * n)
    gpuSgemm(2, S, ffnMul * n, n, xNorm2, layer.fcGate, gateOut)
    let upOut = trackedCreate(S * ffnMul * n)
    gpuSgemm(2, S, ffnMul * n, n, xNorm2, layer.fcUp, upOut)
    let swigluOut = trackedCreate(S * ffnMul * n)
    gpu_swiglu_fwd(gateOut.data, upOut.data, swigluOut.data, cint(S * ffnMul * n))
    let mlpOut = trackedCreate(S * n)
    gpuSgemm(2, S, n, ffnMul * n, swigluOut, layer.fcDown, mlpOut)

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

proc generate(m: Model, tok: Tokenizer, prompt: string,
              history: var seq[int32], maxTokens: int = 200): string =
  ## Generate a response to a prompt. Returns the generated text.
  let inputTokens = tok.encode("<|user|> " & prompt & " <|assistant|> ")
  for id in inputTokens:
    history.add(int32(id))

  var response = ""
  for _ in 0 ..< maxTokens:
    var context = history
    if context.len > blockSize:
      context = context[context.len - blockSize ..< context.len]

    let logits = forwardOneToken(m, context, context.len)
    freeStepAllocations()
    let tokenId = sample(logits)

    # Stop on special tokens (with fixed tokenizer, these are always atomic)
    if tokenId == tok.bosId or tokenId == tok.userId or
       tokenId == tok.assistantId:
      break

    history.add(int32(tokenId))
    let tokenStr = tok.vocab[tokenId]
    response.add(tokenStr)

  response

when isMainModule:
  let baseDir = getAppDir().parentDir()
  let vidyaRoot = baseDir.parentDir()
  let tokenizerFile = vidyaRoot / "tokenizer_nim.bin"
  let modelFile = vidyaRoot / "nimllm.bin"

  gpuInit()

  if not fileExists(tokenizerFile):
    echo "no tokenizer found at ", tokenizerFile
    quit(1)
  let tok = loadTokenizer(tokenizerFile)

  var m = initModel(tok.vocab.len)
  if fileExists(modelFile):
    loadModel(m, modelFile)
  else:
    echo "no model found at ", modelFile
    echo "train first: ./src/microgpt"
    quit(1)

  randomize()
  trackingEnabled = true

  # Non-interactive mode: --prompt "question" prints answer and exits
  if paramCount() >= 2 and paramStr(1) == "--prompt":
    var prompt = paramStr(2)
    for i in 3 .. paramCount(): prompt &= " " & paramStr(i)
    var history: seq[int32]
    echo generate(m, tok, prompt, history)
    quit(0)

  # Interactive mode
  echo ""
  let pc = tok.vocab.len * nEmbd * 2 + blockSize * nEmbd +
    nLayer * (4 * nEmbd * nEmbd + 2 * ffnMul * nEmbd * nEmbd)
  styledEcho(styleBright, "  nimllm", resetStyle,
             fgBlack, "  ·  ", resetStyle,
             $(pc div 1000000), "M params")
  styledEcho(fgBlack, "  type to chat  ·  ctrl-c to exit")
  echo ""

  var history: seq[int32]

  while true:
    stdout.styledWrite(fgGreen, "> ")
    stdout.flushFile()
    let input = stdin.readLine().strip()
    if input.len == 0: continue

    let response = generate(m, tok, input, history)
    echo response
    echo ""
