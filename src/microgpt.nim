## microgpt.nim — nimllm training loop
##
## Imports the model from model.nim (forward, backward, save/load).
## This file is just: optimizer + training loop + CLI.

import model, gpu, bpe, autograd, gguf
import std/[math, random, strformat, os, streams, times, tables]

# ── Configuration ─────────────────────────────────────────────────

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
  let growMode = paramCount() >= 1 and paramStr(1) == "--grow"
  let growFile = if growMode and paramCount() >= 2: paramStr(2) else: ""
  # Check for GGUF model
  var ggufFile = ""
  for i in 1 .. paramCount():
    if paramStr(i) == "--gguf" and i < paramCount(): ggufFile = paramStr(i + 1)
  # Any non-flag argument is a file to train on
  let trainFile = block:
    var f = ""
    for i in 1 .. paramCount():
      if paramStr(i)[0] != '-' and (i < 2 or paramStr(i-1) != "--gguf" and paramStr(i-1) != "--grow"):
        f = paramStr(i); break
    f
  let readMode = trainFile.len > 0
  let readFile = trainFile
  let continueMode = not readMode

  echo "init model..."
  var m: Model
  if ggufFile.len > 0:
    m = initModel(tok.vocab.len, withGradients = false, withWeights = false)
    loadModelGguf(m, ggufFile)
  else:
    m = initModel(tok.vocab.len)
    if fileExists(checkpointFile):
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
    # Load pre-tokenized binary if available (much faster than re-encoding)
    let tokenBinFile = vidyaRoot / "training_tokens.bin"
    if fileExists(tokenBinFile):
      echo "loading pre-tokenized data..."
      let t0 = cpuTime()
      let s = newFileStream(tokenBinFile, fmRead)
      let totalTokens = s.readInt32().int
      # Split into chunks of blockSize for training
      var chunk: seq[int32]
      for i in 0 ..< totalTokens:
        chunk.add(s.readInt32())
        if chunk.len >= blockSize + 1:
          tokenizedDocs.add(chunk)
          chunk = @[]
      if chunk.len >= 3:
        tokenizedDocs.add(chunk)
      s.close()
      echo &"  {totalTokens} tokens -> {tokenizedDocs.len} chunks in {cpuTime() - t0:.1f}s"
    else:
      echo "loading text data (slow — run pre-tokenizer for fast startup)..."
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
  let fwdPerLayer = 10 * S * n + 4 * S * headDim + 2 * S + 3 * S * ffnDim + 2 * S * S
  let fwdGlobal = S * n + S + 2 * S * V
  let bwdPerLayer = 11 * S * n + 7 * S * headDim + 3 * S * ffnDim + 4 * S * S
  let bwdGlobal = S * V + 2 * S * n
  let arenaSize = fwdGlobal + fwdPerLayer * nLayer + bwdGlobal + bwdPerLayer * nLayer
  let arenaMB = arenaSize * sizeof(float32) div (1024 * 1024)
  initScratchArena(arenaSize + arenaSize div 4)  # +25% headroom
  echo &"  scratch arena: {arenaMB} MB ({arenaSize} floats)"

  # Train
  # Read mode: ~3 passes over the material. More for short texts, fewer for long.
  # If a checkpoint exists, always continue from it (never restart from scratch).
  # Old weights fade naturally as new data overwrites them — like memory.
  # Only use cosine warmup schedule on the very first training run.
  let hasCheckpoint = fileExists(checkpointFile) and not growMode
  let readSteps = if readMode: max(1000, min(20000, tokenizedDocs.len * 3)) else: 0
  let numSteps = if readMode: readSteps else: 200000
  let warmupSteps = if readMode: numSteps div 50 elif hasCheckpoint: 0 else: 2000
  let peakLr = if readMode: 0.00005f elif hasCheckpoint: 0.00003f else: 0.0001f
  let minLr = if readMode: 0.000005f elif hasCheckpoint: 0.00003f else: 0.00001f
  if hasCheckpoint and not readMode:
    echo "  continuing from checkpoint (constant LR, no warmup)"
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

  let batchSize = 8  # pack this many docs into one forward pass

  for step in 0 ..< numSteps:
    # Pack multiple docs into one long sequence (up to blockSize tokens)
    var packed: seq[int32]
    var docIdx = step * batchSize
    for b in 0 ..< batchSize:
      let doc = tokenizedDocs[order[(docIdx + b) mod order.len]]
      let remaining = blockSize - packed.len
      if remaining < 4: break
      let take = min(doc.len, remaining)
      for i in 0 ..< take:
        packed.add(doc[i])

    let seqLen = min(blockSize, packed.len - 1)
    if seqLen < 2: continue

    var (cache, loss) = forwardTrain(m, packed[0 ..< seqLen + 1], seqLen)
    backward(m, packed[0 ..< seqLen + 1], seqLen, cache)
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
