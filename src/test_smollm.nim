## test_smollm.nim — Feed pre-tokenized input, check if SmolLM generates coherent text.
## Bypasses our BPE encoder to isolate forward pass from tokenizer.

import gpu, bpe, autograd
import std/[math, strformat, os, streams, tables, random]

const
  nLayer = 30
  nEmbd = 576
  nHead = 9
  nKvHead = 3
  headDim = nEmbd div nHead
  kvRepeat = nHead div nKvHead
  nKvDim = nKvHead * headDim
  blockSize = 512
  ffnDim = 1536
  ropeTheta = 100000.0f

type
  Layer = object
    wq, wk, wv, wo, fcGate, fcUp, fcDown, ln1g, ln2g: GpuBuf
    dwq, dwk, dwv, dwo, dfcGate, dfcUp, dfcDown, dln1g, dln2g: GpuBuf
  Model = object
    wte, lmHead, lnFg: GpuBuf
    layers: seq[Layer]
    vocabSize: int
    dwte, dlmHead, dlnFg: GpuBuf
    tokIdBuf, targetIdBuf: pointer
    ropeCos, ropeSin: GpuBuf

proc initModel(vocabSize: int): Model =
  result.vocabSize = vocabSize
  result.wte = gpuCreate(vocabSize * nEmbd)
  result.lmHead = gpuCreate(vocabSize * nEmbd)
  result.dwte = gpuCreate(vocabSize * nEmbd)
  result.dlmHead = gpuCreate(vocabSize * nEmbd)
  var ones = newSeq[float32](nEmbd)
  for i in 0 ..< nEmbd: ones[i] = 1.0f
  result.lnFg = toGpu(ones)
  result.dlnFg = gpuCreate(nEmbd)
  discard cudaMalloc(addr result.tokIdBuf, csize_t(blockSize * sizeof(int32)))
  discard cudaMalloc(addr result.targetIdBuf, csize_t(blockSize * sizeof(int32)))
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
  for i in 0 ..< nLayer:
    var ln1 = newSeq[float32](nEmbd)
    var ln2 = newSeq[float32](nEmbd)
    for j in 0 ..< nEmbd: ln1[j] = 1.0f; ln2[j] = 1.0f
    result.layers.add(Layer(
      wq: gpuCreate(nEmbd * nEmbd), wk: gpuCreate(nKvDim * nEmbd),
      wv: gpuCreate(nKvDim * nEmbd), wo: gpuCreate(nEmbd * nEmbd),
      fcGate: gpuCreate(ffnDim * nEmbd), fcUp: gpuCreate(ffnDim * nEmbd),
      fcDown: gpuCreate(nEmbd * ffnDim),
      ln1g: toGpu(ln1), ln2g: toGpu(ln2),
      dwq: gpuCreate(nEmbd * nEmbd), dwk: gpuCreate(nKvDim * nEmbd),
      dwv: gpuCreate(nKvDim * nEmbd), dwo: gpuCreate(nEmbd * nEmbd),
      dfcGate: gpuCreate(ffnDim * nEmbd), dfcUp: gpuCreate(ffnDim * nEmbd),
      dfcDown: gpuCreate(nEmbd * ffnDim),
      dln1g: gpuCreate(nEmbd), dln2g: gpuCreate(nEmbd)))

proc loadModel(m: var Model, filename: string) =
  let s = newFileStream(filename, fmRead)
  defer: s.close()
  var magic: array[4, char]
  discard s.readData(addr magic[0], 4)
  if magic == ['N', 'L', 'L', 'M']:
    for i in 0..5: discard s.readInt32()
  else: s.setPosition(0)
  proc r(buf: GpuBuf) =
    let n = s.readInt32().int
    var d = newSeq[float32](n)
    for i in 0 ..< n: d[i] = s.readFloat32()
    gpuUpload(buf, d)
  r(m.wte); r(m.lmHead); r(m.lnFg)
  for i in 0 ..< m.layers.len:
    r(m.layers[i].wq); r(m.layers[i].wk); r(m.layers[i].wv); r(m.layers[i].wo)
    r(m.layers[i].fcGate); r(m.layers[i].fcUp); r(m.layers[i].fcDown)
    r(m.layers[i].ln1g); r(m.layers[i].ln2g)

when isMainModule:
  gpuInit()
  let baseDir = getAppDir().parentDir()
  let vidyaRoot = baseDir.parentDir()
  let tok = loadTokenizer(vidyaRoot / "tokenizer_nim.bin")

  var m = initModel(tok.vocab.len)
  loadModel(m, vidyaRoot / "nimllm.bin")
  trackingEnabled = true

  # Pre-tokenized: <|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\n
  var tokens = @[int32(1), 4093, 198, 19556, 28, 638, 359, 346, 47, 2, 198, 1, 520, 9531, 198]
  let n = nEmbd

  echo "Generating from pre-tokenized input (", tokens.len, " tokens)..."
  randomize()

  for step in 0 ..< 50:
    let S = tokens.len
    var x = trackedCreate(S * n)
    discard cudaMemcpy(m.tokIdBuf, unsafeAddr tokens[0],
                       csize_t(S * sizeof(int32)), CudaMemcpyHostToDevice)
    gpu_embed_fwd(m.wte.data, m.tokIdBuf, x.data, cint(S), cint(n))

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
      ropeFwd(q, m.ropeCos, m.ropeSin, S, n, nHead, headDim)
      ropeFwd(k, m.ropeCos, m.ropeSin, S, nKvDim, nKvHead, headDim)

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

      let gateOut = trackedCreate(S * ffnDim)
      gpuSgemm(2, S, ffnDim, n, xNorm2, layer.fcGate, gateOut)
      let upOut = trackedCreate(S * ffnDim)
      gpuSgemm(2, S, ffnDim, n, xNorm2, layer.fcUp, upOut)
      let swigluOut = trackedCreate(S * ffnDim)
      gpu_swiglu_fwd(gateOut.data, upOut.data, swigluOut.data, cint(S * ffnDim))
      let mlpOut = trackedCreate(S * n)
      gpuSgemm(2, S, n, ffnDim, swigluOut, layer.fcDown, mlpOut)

      let xNew = trackedCreate(S * n)
      gpu_add(x2.data, mlpOut.data, xNew.data, cint(S * n))
      x = xNew

    let finalNormed = trackedCreate(S * n)
    let rmsF = trackedCreate(S)
    gpu_rmsnorm_affine_fwd(x.data, m.lnFg.data, finalNormed.data, rmsF.data, cint(S), cint(n))

    let logits = trackedCreate(S * m.vocabSize)
    gpuSgemm(2, S, m.vocabSize, n, finalNormed, m.lmHead, logits)

    let allLogits = gpuDownload(logits)
    let lastRow = (S - 1) * m.vocabSize

    # Greedy: pick highest logit
    var bestId = 0
    var bestVal = -1e30f
    for i in 0 ..< m.vocabSize:
      if allLogits[lastRow + i] > bestVal:
        bestVal = allLogits[lastRow + i]
        bestId = i

    freeStepAllocations()

    # Stop on EOS or im_end
    if bestId == 0 or bestId == 2:
      break

    tokens.add(int32(bestId))
    stdout.write(tok.vocab[bestId])
    stdout.flushFile()

  echo ""
