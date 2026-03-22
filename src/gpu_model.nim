## gpu_model.nim — 103M parameter transformer on GPU
##
## All weights stored as GpuBuf (float32 in VRAM).
## Initialised with random weights on CPU, then uploaded.

import gpu
import std/[math, random, streams, strformat]

const
  nLayer*   = 8
  nEmbd*    = 1024
  blockSize* = 512
  nHead*    = 16
  headDim*  = nEmbd div nHead   # 64
  halfDim*  = headDim div 2     # 32
  ffnDim*   = 4 * nEmbd         # 4096

type
  GpuParam* = object
    ## A trainable parameter: data + gradient, both on GPU.
    data*: GpuBuf
    grad*: GpuBuf
    numel*: int

  GpuLayer* = object
    attnWq*, attnWk*, attnWv*, attnWo*: GpuParam
    qNorm*, kNorm*: GpuParam     # QK-Norm
    mlpFc1*, mlpFc2*: GpuParam
    ln1*, ln2*: GpuParam

  GpuModel* = object
    wte*: GpuParam             # [vocab, nEmbd] — also used as lm_head
    layers*: seq[GpuLayer]
    embedNorm*: GpuParam       # [nEmbd]
    finalNorm*: GpuParam       # [nEmbd]
    vocabSize*: int
    ropeCos*: GpuBuf           # [blockSize, halfDim] on GPU
    ropeSin*: GpuBuf           # [blockSize, halfDim] on GPU

# ── Initialization helpers ────────────────────────────────────────

proc randomNormal(n: int, std: float32): seq[float32] =
  ## Generate n float32 values from N(0, std) using Box-Muller.
  result = newSeq[float32](n)
  var i = 0
  while i < n - 1:
    let u1 = rand(1.0).float32
    let u2 = rand(1.0).float32
    let mag = std * sqrt(-2.0f * ln(max(u1, 1e-10f)))
    result[i]     = mag * cos(2.0f * PI.float32 * u2)
    result[i + 1] = mag * sin(2.0f * PI.float32 * u2)
    i += 2
  if i < n:
    let u1 = rand(1.0).float32
    let u2 = rand(1.0).float32
    result[i] = std * sqrt(-2.0f * ln(max(u1, 1e-10f))) * cos(2.0f * PI.float32 * u2)

proc makeParam(n: int, std: float32 = 0.02f): GpuParam =
  ## Create a trainable parameter with random init on GPU.
  let hostData = randomNormal(n, std)
  result.numel = n
  result.data = toGpu(hostData)
  result.grad = gpuCreate(n)  # zeroed

proc makeOnesParam(n: int): GpuParam =
  ## Create a parameter initialised to all 1.0 (for RMSNorm scale).
  var hostData = newSeq[float32](n)
  for i in 0 ..< n: hostData[i] = 1.0f
  result.numel = n
  result.data = toGpu(hostData)
  result.grad = gpuCreate(n)

# ── Model construction ────────────────────────────────────────────

proc initGpuModel*(vocabSize: int): GpuModel =
  ## Create 103M parameter model with all weights on GPU.
  randomize(42)
  let residualStd = 0.02f / sqrt(float32(2 * nLayer))

  echo "  uploading wte..."
  result.vocabSize = vocabSize
  result.wte = makeParam(vocabSize * nEmbd)
  result.embedNorm = makeOnesParam(nEmbd)
  result.finalNorm = makeOnesParam(nEmbd)

  result.layers = newSeq[GpuLayer](nLayer)
  for i in 0 ..< nLayer:
    echo "  uploading layer ", i, "..."
    result.layers[i] = GpuLayer(
      attnWq: makeParam(nEmbd * nEmbd),
      attnWk: makeParam(nEmbd * nEmbd),
      attnWv: makeParam(nEmbd * nEmbd),
      attnWo: makeParam(nEmbd * nEmbd, residualStd),
      qNorm: makeOnesParam(nEmbd),
      kNorm: makeOnesParam(nEmbd),
      mlpFc1: makeParam(ffnDim * nEmbd),
      mlpFc2: makeParam(nEmbd * ffnDim, residualStd),
      ln1: makeOnesParam(nEmbd),
      ln2: makeOnesParam(nEmbd),
    )

  # Upload RoPE tables
  echo "  uploading RoPE tables..."
  var cosData = newSeq[float32](blockSize * halfDim)
  var sinData = newSeq[float32](blockSize * halfDim)
  for pos in 0 ..< blockSize:
    for i in 0 ..< halfDim:
      let freq = 1.0f / pow(10000.0f, float32(2 * i) / float32(headDim))
      cosData[pos * halfDim + i] = cos(float32(pos) * freq)
      sinData[pos * halfDim + i] = sin(float32(pos) * freq)
  result.ropeCos = toGpu(cosData)
  result.ropeSin = toGpu(sinData)

proc paramCount*(m: GpuModel): int =
  result = m.wte.numel + m.embedNorm.numel + m.finalNorm.numel
  for layer in m.layers:
    result += layer.attnWq.numel + layer.attnWk.numel
    result += layer.attnWv.numel + layer.attnWo.numel
    result += layer.mlpFc1.numel + layer.mlpFc2.numel
    result += layer.ln1.numel + layer.ln2.numel

# ── Checkpoint save/load ──────────────────────────────────────────

proc writeParam(s: FileStream, p: GpuParam) =
  let data = gpuDownload(p.data)
  s.write(int32(data.len))
  for v in data: s.write(v)

proc readParam(s: FileStream, p: var GpuParam) =
  let n = s.readInt32().int
  var data = newSeq[float32](n)
  for i in 0 ..< n: data[i] = s.readFloat32()
  gpuUpload(p.data, data)

proc saveCheckpoint*(m: GpuModel, filename: string) =
  ## Download all weights from GPU and save as flat float32 binary.
  echo &"saving checkpoint to {filename}..."
  let s = newFileStream(filename, fmWrite)
  defer: s.close()
  s.write(int32(m.vocabSize))
  s.write(int32(paramCount(m)))
  s.writeParam(m.wte)
  s.writeParam(m.embedNorm)
  for i in 0 ..< m.layers.len:
    s.writeParam(m.layers[i].attnWq)
    s.writeParam(m.layers[i].attnWk)
    s.writeParam(m.layers[i].attnWv)
    s.writeParam(m.layers[i].attnWo)
    s.writeParam(m.layers[i].qNorm)
    s.writeParam(m.layers[i].kNorm)
    s.writeParam(m.layers[i].mlpFc1)
    s.writeParam(m.layers[i].mlpFc2)
    s.writeParam(m.layers[i].ln1)
    s.writeParam(m.layers[i].ln2)
  s.writeParam(m.finalNorm)
  echo "  done"

proc loadCheckpoint*(m: var GpuModel, filename: string) =
  ## Load weights from flat float32 binary file into GPU.
  echo &"loading checkpoint from {filename}..."
  let s = newFileStream(filename, fmRead)
  defer: s.close()
  let vocabSize = s.readInt32().int
  let nParams = s.readInt32().int
  echo &"  vocab={vocabSize} params={nParams}"
  s.readParam(m.wte)
  s.readParam(m.embedNorm)
  for i in 0 ..< m.layers.len:
    s.readParam(m.layers[i].attnWq)
    s.readParam(m.layers[i].attnWk)
    s.readParam(m.layers[i].attnWv)
    s.readParam(m.layers[i].attnWo)
    s.readParam(m.layers[i].qNorm)
    s.readParam(m.layers[i].kNorm)
    s.readParam(m.layers[i].mlpFc1)
    s.readParam(m.layers[i].mlpFc2)
    s.readParam(m.layers[i].ln1)
    s.readParam(m.layers[i].ln2)
  s.readParam(m.finalNorm)
  echo "  done"

proc collectParams*(m: var GpuModel): seq[ptr GpuParam] =
  ## Gather pointers to all trainable parameters for the optimizer.
  result.add(addr m.wte)
  result.add(addr m.embedNorm)
  for i in 0 ..< m.layers.len:
    result.add(addr m.layers[i].attnWq)
    result.add(addr m.layers[i].attnWk)
    result.add(addr m.layers[i].attnWv)
    result.add(addr m.layers[i].attnWo)
    result.add(addr m.layers[i].qNorm)
    result.add(addr m.layers[i].kNorm)
    result.add(addr m.layers[i].mlpFc1)
    result.add(addr m.layers[i].mlpFc2)
    result.add(addr m.layers[i].ln1)
    result.add(addr m.layers[i].ln2)
  result.add(addr m.finalNorm)
