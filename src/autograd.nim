## autograd.nim — Reverse-mode automatic differentiation on GPU
##
## Each operation creates a Node with a backward closure.
## backward() walks the graph in reverse topological order.
## All data on GPU. Graph structure on CPU.

import gpu, gpu_model
import std/[tables, math]

# ── Scratch arena ─────────────────────────────────────────────────
# One big GPU allocation, bump-allocated per step, reset at end.
# Eliminates ~300 cudaMalloc/Free calls per training step.
# Falls back to individual allocs when arena is not initialised
# (e.g. chat inference).

var arenaBase: pointer = nil
var arenaCap: int = 0      # total floats
var arenaPos: int = 0      # current offset in floats
var arenaHWM: int = 0      # high-water mark (max arenaPos ever seen)

proc initScratchArena*(nFloats: int) =
  ## Pre-allocate GPU scratch memory. Call once before training.
  if arenaBase != nil:
    discard cudaFree(arenaBase)
  let err = cudaMalloc(addr arenaBase, csize_t(nFloats * sizeof(float32)))
  assert err == CudaSuccess, "scratch arena cudaMalloc failed"
  discard cudaMemset(arenaBase, 0, csize_t(nFloats * sizeof(float32)))
  arenaCap = nFloats
  arenaPos = 0

# Fallback tracking for non-arena mode (chat, etc.)
var stepAllocations*: seq[GpuBuf]
var trackingEnabled* = false

const arenaAlign = 32  # align to 32 floats = 128 bytes (GPU cache line)

proc trackedCreate*(n: int): GpuBuf =
  ## Allocate n float32s on GPU. Arena mode: bump pointer (fast).
  ## Fallback mode: cudaMalloc (slow but always works).
  if arenaBase != nil:
    # Align offset to 128-byte boundary for cuBLAS/kernel efficiency
    let aligned = (arenaPos + arenaAlign - 1) div arenaAlign * arenaAlign
    assert aligned + n <= arenaCap,
      "scratch arena overflow: need " & $(aligned + n) & " have " & $arenaCap
    result.data = cast[pointer](cast[int](arenaBase) + aligned * sizeof(float32))
    result.numel = n
    arenaPos = aligned + n
  else:
    result = gpuCreate(n)
    if trackingEnabled:
      stepAllocations.add(result)

proc freeStepAllocations*() =
  ## Reset scratch memory for next step.
  ## Arena mode: zero used region, reset pointer. One cudaMemset.
  ## Fallback mode: free each buffer individually.
  if arenaBase != nil:
    if arenaPos > arenaHWM: arenaHWM = arenaPos
    if arenaHWM > 0:
      discard cudaMemset(arenaBase, 0, csize_t(arenaHWM * sizeof(float32)))
      arenaPos = 0
  else:
    for i in 0 ..< stepAllocations.len:
      gpuFree(stepAllocations[i])
    stepAllocations.setLen(0)

type
  BackwardFn* = proc() {.closure.}

  Node* = ref object
    id*: int
    data*: GpuBuf
    grad*: GpuBuf
    numel*: int
    children*: seq[Node]
    backwardFn*: BackwardFn

var nextId = 0
proc freshId(): int =
  result = nextId; inc nextId

proc newNode*(data: GpuBuf, numel: int, children: seq[Node] = @[],
              bwd: BackwardFn = nil): Node =
  Node(id: freshId(), data: data, grad: trackedCreate(numel),
       numel: numel, children: children, backwardFn: bwd)

proc paramNode*(p: GpuParam): Node =
  ## Leaf node wrapping a model parameter. Shares data/grad buffers.
  Node(id: freshId(), data: p.data, grad: p.grad,
       numel: p.numel, children: @[], backwardFn: nil)

proc paramNode*(data, grad: GpuBuf, numel: int): Node =
  ## Generic version: wrap any data/grad pair as a leaf node.
  Node(id: freshId(), data: data, grad: grad,
       numel: numel, children: @[], backwardFn: nil)

# ── Operations ────────────────────────────────────────────────────
# Pattern: create node, capture its grad by let binding, set backwardFn.

proc agMatmul*(w, x: Node, s, m, n: int): Node =
  ## y[s,m] = x[s,n] @ w[m,n]^T
  let yBuf = trackedCreate(s * m)
  gpuSgemm(2, s, m, n, x.data, w.data, yBuf)
  result = newNode(yBuf, s * m, @[w, x])
  let yGrad = result.grad
  result.backwardFn = proc() =
    gpuSgemm(5, m, n, s, yGrad, x.data, w.grad)
    gpuSgemm(1, s, n, m, yGrad, w.data, x.grad)

proc agAdd*(a, b: Node): Node =
  let n = a.numel
  let yBuf = trackedCreate(n)
  gpu_add(a.data.data, b.data.data, yBuf.data, cint(n))
  result = newNode(yBuf, n, @[a, b])
  let yGrad = result.grad
  result.backwardFn = proc() =
    gpu_add_inplace(a.grad.data, yGrad.data, cint(n))
    gpu_add_inplace(b.grad.data, yGrad.data, cint(n))

proc agGelu*(x: Node): Node =
  let n = x.numel
  let yBuf = trackedCreate(n)
  geluFwd(x.data, yBuf)
  result = newNode(yBuf, n, @[x])
  let yGrad = result.grad
  result.backwardFn = proc() =
    geluBwd(x.data, yGrad, x.grad)

proc agRmsNormAffine*(x, gamma: Node, rows, dim: int): Node =
  let n = rows * dim
  let yBuf = trackedCreate(n)
  let rmsBuf = trackedCreate(rows)
  rmsnormAffineFwd(x.data, gamma.data, yBuf, rmsBuf, rows, dim)
  result = newNode(yBuf, n, @[x, gamma])
  let yGrad = result.grad
  result.backwardFn = proc() =
    rmsnormAffineBwd(x.data, gamma.data, yGrad, rmsBuf,
                     x.grad, gamma.grad, rows, dim)

proc agSoftmax*(x: Node, rows, cols: int): Node =
  let n = rows * cols
  let yBuf = trackedCreate(n)
  softmaxFwd(x.data, yBuf, rows, cols)
  result = newNode(yBuf, n, @[x])
  let yData = result.data
  let yGrad = result.grad
  result.backwardFn = proc() =
    softmaxBwd(yData, yGrad, x.grad, rows, cols)

proc agEmbed*(wte: Node, tokenIds: seq[int32], seqLen, dim: int): Node =
  let n = seqLen * dim
  let yBuf = trackedCreate(n)
  var tokBuf: pointer
  discard cudaMalloc(addr tokBuf, csize_t(seqLen * sizeof(int32)))
  discard cudaMemcpy(tokBuf, unsafeAddr tokenIds[0],
                     csize_t(seqLen * sizeof(int32)), CudaMemcpyHostToDevice)
  gpu_embed_fwd(wte.data.data, tokBuf, yBuf.data, cint(seqLen), cint(dim))
  result = newNode(yBuf, n, @[wte])
  let yGrad = result.grad
  result.backwardFn = proc() =
    gpu_embed_bwd(wte.grad.data, tokBuf, yGrad.data, cint(seqLen), cint(dim))
    # Note: tokBuf leaks — need to free after backward. TODO: cleanup list.

proc agRow*(mat: Node, rowIdx, cols: int): Node =
  let yBuf = trackedCreate(cols)
  discard cudaMemcpy(yBuf.data,
    cast[pointer](cast[int](mat.data.data) + rowIdx * cols * sizeof(cfloat)),
    csize_t(cols * sizeof(cfloat)), CudaMemcpyDeviceToDevice)
  result = newNode(yBuf, cols, @[mat])
  let yGrad = result.grad
  result.backwardFn = proc() =
    # Accumulate row gradient back
    gpu_add_inplace(
      cast[pointer](cast[int](mat.grad.data) + rowIdx * cols * sizeof(cfloat)),
      yGrad.data, cint(cols))

proc agSwiGLU*(gate, up: Node): Node =
  ## SwiGLU: swish(gate) * up. Used in FFN instead of GELU.
  let n = gate.numel
  let yBuf = trackedCreate(n)
  gpu_swiglu_fwd(gate.data.data, up.data.data, yBuf.data, cint(n))
  result = newNode(yBuf, n, @[gate, up])
  let yGrad = result.grad
  result.backwardFn = proc() =
    gpu_swiglu_bwd(gate.data.data, up.data.data, yGrad.data,
                   gate.grad.data, up.grad.data, cint(n))

proc agCrossEntropy*(logits: Node, target: int, vocabSize: int): Node =
  ## Numerically stable cross-entropy using log-softmax.
  ## Forward: loss = -log_softmax(logits)[target]
  ## Backward: d_logits = softmax(logits) - one_hot(target)
  ## No exp() overflow. No 1/p explosion. Bounded [-1, 1].
  let logProbsBuf = trackedCreate(vocabSize)
  gpu_log_softmax(logits.data.data, logProbsBuf.data, 1, cint(vocabSize))
  let lossVal = gpu_cross_entropy_loss(logProbsBuf.data, cint(target), cint(vocabSize))
  let lossBuf = trackedCreate(1)
  gpuUpload(lossBuf, @[lossVal])
  result = newNode(lossBuf, 1, @[logits])
  let yGrad = result.grad
  result.backwardFn = proc() =
    let dy = gpuDownload(yGrad)
    let upstream = dy[0]
    # probs = exp(log_probs), then grad = upstream * (probs - one_hot)
    var logProbsCpu = gpuDownload(logProbsBuf)
    var gradCpu = newSeq[float32](vocabSize)
    for i in 0 ..< vocabSize:
      gradCpu[i] = upstream * exp(logProbsCpu[i])
    gradCpu[target] -= upstream
    let gradBuf = trackedCreate(vocabSize)
    gpuUpload(gradBuf, gradCpu)
    gpu_add_inplace(logits.grad.data, gradBuf.data, cint(vocabSize))

proc agNll*(probs: Node, target: int, vocabSize: int): Node =
  let lossVal = gpu_nll_fwd(probs.data.data, cint(target))
  let yBuf = trackedCreate(1)
  gpuUpload(yBuf, @[lossVal])
  result = newNode(yBuf, 1, @[probs])
  let yGrad = result.grad
  result.backwardFn = proc() =
    let dy = gpuDownload(yGrad)
    let probsCpu = gpuDownload(probs.data)
    let gradVal = -dy[0] / max(probsCpu[target], 1e-10f)
    var probGradCpu = gpuDownload(probs.grad)
    probGradCpu[target] += gradVal
    gpuUpload(probs.grad, probGradCpu)

# ── Attention ─────────────────────────────────────────────────────

proc agAttention*(q, k, v: Node, ropeCos, ropeSin: GpuBuf,
                  seqLen: int): Node =
  let n = nEmbd
  let hd = headDim
  let scale = 1.0f / sqrt(float32(hd))

  let qRot = trackedCreate(seqLen * n)
  gpuCopy(q.data, qRot, seqLen * n)
  let kRot = trackedCreate(seqLen * n)
  gpuCopy(k.data, kRot, seqLen * n)
  ropeFwd(qRot, ropeCos, ropeSin, seqLen, n, nHead, hd)
  ropeFwd(kRot, ropeCos, ropeSin, seqLen, n, nHead, hd)

  let outBuf = trackedCreate(seqLen * n)
  let qH = trackedCreate(seqLen * hd)
  let kH = trackedCreate(seqLen * hd)
  let vH = trackedCreate(seqLen * hd)
  let scores = trackedCreate(seqLen * seqLen)
  let weights = trackedCreate(seqLen * seqLen)
  let attnH = trackedCreate(seqLen * hd)

  var allWeights = newSeq[GpuBuf](nHead)
  for h in 0 ..< nHead:
    extractHead(qRot, qH, h, seqLen, n, hd)
    extractHead(kRot, kH, h, seqLen, n, hd)
    extractHead(v.data, vH, h, seqLen, n, hd)
    gpuSgemm(2, seqLen, seqLen, hd, qH, kH, scores)
    causalMask(scores, scale, seqLen)
    # Clamp scores to prevent exp() overflow in softmax
    gpu_clamp(scores.data, -65000.0f, 65000.0f, cint(seqLen * seqLen))
    softmaxFwd(scores, weights, seqLen, seqLen)
    allWeights[h] = trackedCreate(seqLen * seqLen)
    gpuCopy(weights, allWeights[h], seqLen * seqLen)
    gpuSgemm(0, seqLen, hd, seqLen, weights, vH, attnH)
    insertHead(attnH, outBuf, h, seqLen, n, hd)

  result = newNode(outBuf, seqLen * n, @[q, k, v])
  let outGrad = result.grad
  result.backwardFn = proc() =
    let dqRot = trackedCreate(seqLen * n)
    let dkRot = trackedCreate(seqLen * n)
    let doutH = trackedCreate(seqLen * hd)
    let dvH = trackedCreate(seqLen * hd)
    let dw = trackedCreate(seqLen * seqLen)      # dWeights (input to softmax bwd)
    let dScores = trackedCreate(seqLen * seqLen)  # output of softmax bwd
    let dqH = trackedCreate(seqLen * hd)
    let dkH = trackedCreate(seqLen * hd)

    for h in 0 ..< nHead:
      extractHead(outGrad, doutH, h, seqLen, n, hd)
      extractHead(v.data, vH, h, seqLen, n, hd)
      # dWeights = dAttn @ V^T
      gpuSgemm(2, seqLen, seqLen, hd, doutH, vH, dw)
      # dV += Weights^T @ dAttn
      gpuSgemm(4, seqLen, hd, seqLen, allWeights[h], doutH, dvH)
      insertHeadAcc(dvH, v.grad, h, seqLen, n, hd)
      # Causal softmax backward — matches OCaml implementation exactly.
      # CPU-side for correctness. Only processes lower triangle.
      let dwCpu = gpuDownload(dw)
      let wtCpu = gpuDownload(allWeights[h])
      var dScoresCpu = newSeq[float32](seqLen * seqLen)
      for i in 0 ..< seqLen:
        var dot = 0.0f
        for j in 0 .. i:
          dot += dwCpu[i * seqLen + j] * wtCpu[i * seqLen + j]
        for j in 0 .. i:
          dScoresCpu[i * seqLen + j] =
            wtCpu[i * seqLen + j] * (dwCpu[i * seqLen + j] - dot) * scale
        for j in i + 1 ..< seqLen:
          dScoresCpu[i * seqLen + j] = 0.0f
      gpuUpload(dScores, dScoresCpu)
      # dQ_rot += dScores @ K_rot
      extractHead(kRot, kH, h, seqLen, n, hd)
      gpuSgemm(0, seqLen, hd, seqLen, dScores, kH, dqH)
      insertHeadAcc(dqH, dqRot, h, seqLen, n, hd)
      # dK_rot += dScores^T @ Q_rot
      extractHead(qRot, qH, h, seqLen, n, hd)
      gpuSgemm(4, seqLen, hd, seqLen, dScores, qH, dkH)
      insertHeadAcc(dkH, dkRot, h, seqLen, n, hd)

    ropeBwd(dqRot, ropeCos, ropeSin, seqLen, n, nHead, hd)
    ropeBwd(dkRot, ropeCos, ropeSin, seqLen, n, nHead, hd)
    gpu_add_inplace(q.grad.data, dqRot.data, cint(seqLen * n))
    gpu_add_inplace(k.grad.data, dkRot.data, cint(seqLen * n))

# ── Backward ──────────────────────────────────────────────────────

proc backward*(loss: Node) =
  var visited = initTable[int, bool]()
  var topo: seq[Node]
  proc buildTopo(node: Node) =
    if node.id in visited: return
    visited[node.id] = true
    for child in node.children: buildTopo(child)
    topo.add(node)
  buildTopo(loss)

  # Seed
  gpuUpload(loss.grad, @[1.0f])

  # Reverse order
  for i in countdown(topo.high, 0):
    if topo[i].backwardFn != nil:
      topo[i].backwardFn()

proc zeroGrad*(params: seq[ptr GpuParam]) =
  for p in params: gpuZero(p[].grad)

# ── Cleanup ───────────────────────────────────────────────────────

proc freeGraph*(loss: Node) =
  ## Free all intermediate GPU buffers in the autograd graph.
  ## Does NOT free paramNode buffers (they're owned by the model).
  ## Call this after each training step to reclaim VRAM.
  var visited = initTable[int, bool]()
  proc walk(node: Node) =
    if node.id in visited: return
    visited[node.id] = true
    for child in node.children: walk(child)
    # Only free if this node owns its buffers (not a param node).
    # Param nodes have children.len == 0 and backwardFn == nil.
    if node.backwardFn != nil:
      gpuFree(node.data)
      gpuFree(node.grad)
  walk(loss)
