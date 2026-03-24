## test_forward.nim — Verify forward pass against reference values
## Run after loading SmolLM weights to find computation bugs.

import gpu, bpe, autograd
import std/[math, strformat, os, streams, tables]

const
  nLayer   = 30
  nEmbd    = 576
  nHead    = 9
  nKvHead  = 3
  headDim  = nEmbd div nHead
  kvRepeat = nHead div nKvHead
  nKvDim   = nKvHead * headDim
  blockSize = 512
  ffnDim   = 1536
  ropeTheta = 100000.0f

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
    dwte, dlmHead: GpuBuf
    dlnFg: GpuBuf
    tokIdBuf: pointer
    targetIdBuf: pointer
    ropeCos, ropeSin: GpuBuf

proc initModel(vocabSize: int): Model =
  let std = 0.02f
  proc randBuf(n: int, s: float32 = std): GpuBuf = gpuCreate(n)
  proc onesBuf(n: int): GpuBuf =
    var h = newSeq[float32](n)
    for i in 0 ..< n: h[i] = 1.0f
    toGpu(h)

  result.vocabSize = vocabSize
  result.wte = gpuCreate(vocabSize * nEmbd)
  result.lmHead = gpuCreate(vocabSize * nEmbd)
  result.dwte = gpuCreate(vocabSize * nEmbd)
  result.dlmHead = gpuCreate(vocabSize * nEmbd)
  result.lnFg = onesBuf(nEmbd)
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
    result.layers.add(Layer(
      wq: gpuCreate(nEmbd * nEmbd), wk: gpuCreate(nKvDim * nEmbd),
      wv: gpuCreate(nKvDim * nEmbd), wo: gpuCreate(nEmbd * nEmbd),
      fcGate: gpuCreate(ffnDim * nEmbd), fcUp: gpuCreate(ffnDim * nEmbd),
      fcDown: gpuCreate(nEmbd * ffnDim),
      ln1g: onesBuf(nEmbd), ln2g: onesBuf(nEmbd),
      dwq: gpuCreate(nEmbd * nEmbd), dwk: gpuCreate(nKvDim * nEmbd),
      dwv: gpuCreate(nKvDim * nEmbd), dwo: gpuCreate(nEmbd * nEmbd),
      dfcGate: gpuCreate(ffnDim * nEmbd), dfcUp: gpuCreate(ffnDim * nEmbd),
      dfcDown: gpuCreate(nEmbd * ffnDim),
      dln1g: gpuCreate(nEmbd), dln2g: gpuCreate(nEmbd),
    ))

proc loadModel(m: var Model, filename: string) =
  let s = newFileStream(filename, fmRead)
  defer: s.close()
  var magic: array[4, char]
  discard s.readData(addr magic[0], 4)
  if magic == ['N', 'L', 'L', 'M']:
    for i in 0..5: discard s.readInt32()
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

when isMainModule:
  let baseDir = getAppDir().parentDir()
  let vidyaRoot = baseDir.parentDir()
  gpuInit()

  var m = initModel(49152)
  loadModel(m, vidyaRoot / "nimllm.bin")

  let n = nEmbd
  let S = 1  # single token

  # Token 19556 = "Hello"
  var tok = [int32(19556)]
  discard cudaMemcpy(m.tokIdBuf, addr tok[0], csize_t(sizeof(int32)), CudaMemcpyHostToDevice)

  # 1. Embedding
  let x = trackedCreate(S * n)
  gpu_embed_fwd(m.wte.data, m.tokIdBuf, x.data, cint(S), cint(n))
  let emb = gpuDownload(x)
  echo "1. Embedding[:5]: ", emb[0], " ", emb[1], " ", emb[2], " ", emb[3], " ", emb[4]
  echo "   Ref:           -0.0737  0.0354  0.0200  0.0889  0.0608"

  # 2. RMSNorm
  let xNorm = trackedCreate(S * n)
  let rms = trackedCreate(S)
  gpu_rmsnorm_affine_fwd(x.data, m.layers[0].ln1g.data, xNorm.data, rms.data, cint(S), cint(n))
  let normed = gpuDownload(xNorm)
  echo "2. RMSNorm[:5]:   ", normed[0], " ", normed[1], " ", normed[2], " ", normed[3], " ", normed[4]
  echo "   Ref:           -0.0127  0.0060 -0.0067 -0.0272 -0.0155"

  # 3. Q projection
  let q = trackedCreate(S * n)
  gpuSgemm(2, S, n, n, xNorm, m.layers[0].wq, q)
  let qv = gpuDownload(q)
  echo "3. Q proj[:5]:    ", qv[0], " ", qv[1], " ", qv[2], " ", qv[3], " ", qv[4]
  echo "   Ref:           -0.4361 -0.1873 -0.1716 -0.7897  0.5963"

  # 4. K projection
  let k = trackedCreate(S * nKvDim)
  gpuSgemm(2, S, nKvDim, n, xNorm, m.layers[0].wk, k)
  let kv = gpuDownload(k)
  echo "4. K proj[:5]:    ", kv[0], " ", kv[1], " ", kv[2], " ", kv[3], " ", kv[4]
  echo "   Ref:           -2.3189 -0.4074 -0.6630 -0.8802  1.3037"

  # 5. Full attention for S=1 (attention is identity — one token attends to itself)
  # V projection
  let v = trackedCreate(S * nKvDim)
  gpuSgemm(2, S, nKvDim, n, xNorm, m.layers[0].wv, v)
  let vv = gpuDownload(v)
  echo "5. V proj[:5]:    ", vv[0], " ", vv[1], " ", vv[2], " ", vv[3], " ", vv[4]

  # For S=1, attention output = V (single token, softmax=1.0 on self)
  # But we need to project through wo: attnOut = V @ wo^T
  # Actually: for each head, extract V head, that IS the attention output for that head
  # Then project: projected = attnOut @ wo^T
  
  # Since S=1, let's just do the O projection on V directly
  # But V is [S, nKvDim=192], we need to expand to [S, nEmbd=576] via GQA repeat
  # For S=1: each Q head gets the same V from its KV head
  # So attnOut = [v_kv0, v_kv0, v_kv0, v_kv1, v_kv1, v_kv1, v_kv2, v_kv2, v_kv2]
  # Which is just V repeated by kvRepeat within each head group
  
  # Build the expanded attention output manually
  var attnExpanded = newSeq[float32](nEmbd)
  for h in 0 ..< nHead:
    let kvh = h div kvRepeat
    for d in 0 ..< headDim:
      attnExpanded[h * headDim + d] = vv[kvh * headDim + d]
  let attnOut = trackedCreate(S * n)
  gpuUpload(attnOut, attnExpanded)
  
  # O projection
  let projected = trackedCreate(S * n)
  gpuSgemm(2, S, n, n, attnOut, m.layers[0].wo, projected)
  let proj = gpuDownload(projected)
  echo "6. O proj[:5]:    ", proj[0], " ", proj[1], " ", proj[2], " ", proj[3], " ", proj[4]
  
  # Residual 1
  var x2 = newSeq[float32](n)
  for i in 0 ..< n: x2[i] = emb[i] + proj[i]
  echo "7. Resid1[:5]:    ", x2[0], " ", x2[1], " ", x2[2], " ", x2[3], " ", x2[4]
  echo "   Ref L1 out:    4.2636  0.0586  1.8364  7.3656  3.0198"
