## chat.nim — Talk to nimllm
##
## Loads the trained model, generates text token by token.
## Uses the shared forward pass from model.nim.

import model, gpu, bpe, autograd
import std/[math, random, strformat, os, terminal, strutils, tables]

# ── Sampling ──────────────────────────────────────────────────────

proc sample(logits: seq[float32], recentTokens: seq[int32],
            temperature: float32 = 0.6f, topK: int = 40,
            repetitionPenalty: float32 = 1.2f): int =
  var scaled = newSeq[float32](logits.len)
  for i in 0 ..< logits.len:
    scaled[i] = logits[i] / temperature
  # Penalize recently generated tokens
  for tok in recentTokens:
    if tok >= 0 and tok < scaled.len:
      if scaled[tok] > 0:
        scaled[tok] /= repetitionPenalty
      else:
        scaled[tok] *= repetitionPenalty

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

  var r = rand(1.0).float32
  for k in 0 ..< topK:
    r -= probs[k]
    if r <= 0:
      return topIndices[k]
  topIndices[topK - 1]

# ── Generation ────────────────────────────────────────────────────

var gKv: KvCache  # global KV cache, persists across turns

proc generate(m: Model, tok: Tokenizer, prompt: string,
              history: var seq[int32], maxTokens: int = 200): string =
  # Format prompt — detect chat format from tokenizer
  let hasLlama = tok.tokenToId.getOrDefault("<|start_header_id|>", -1) >= 0
  let hasIm = tok.tokenToId.getOrDefault("<|im_start|>", -1) >= 0
  let chatPrompt = if hasLlama:
      # LLaMA 3 format
      (if history.len == 0: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>" else: "") &
      "<|start_header_id|>user<|end_header_id|>\n\n" & prompt & "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif hasIm:
      # ChatML format (SmolLM, Qwen)
      (if history.len == 0: "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" else: "") &
      "<|im_start|>user\n" & prompt & "<|im_end|>\n<|im_start|>assistant\n"
    else:
      # nimllm native format
      "<|user|> " & prompt & " <|assistant|> "
  let inputTokens = tok.encode(chatPrompt)
  for id in inputTokens:
    history.add(int32(id))

  # Stop tokens — support all formats
  let eosId = tok.tokenToId.getOrDefault("<|endoftext|>", -1)
  let imStartId = tok.tokenToId.getOrDefault("<|im_start|>", -1)
  let imEndId = tok.tokenToId.getOrDefault("<|im_end|>", -1)
  let eotId = tok.tokenToId.getOrDefault("<|eot_id|>", -1)
  let endTextId = tok.tokenToId.getOrDefault("<|end_of_text|>", -1)

  # Init or reset KV cache if near full
  if gKv.k.len == 0:
    gKv = initKvCache()
  elif gKv.pos + inputTokens.len + maxTokens >= blockSize:
    resetKvCache(gKv)
    history.setLen(0)

  # Process prompt tokens through KV cache (one at a time for correct causal masking)
  var logits: seq[float32]
  for id in inputTokens:
    if gKv.pos >= blockSize - 1: break  # safety check
    logits = forwardCached(m, gKv, @[int32(id)])
    freeStepAllocations()

  # Generate tokens one at a time
  var response = ""
  for _ in 0 ..< maxTokens:
    if gKv.pos >= blockSize - 1: break  # context full
    # Pass last 64 tokens for repetition penalty
    let recent = if history.len > 64: history[history.len - 64 ..< history.len] else: history
    let tokenId = sample(logits, recent)

    if tokenId == tok.bosId or tokenId == tok.userId or
       tokenId == tok.assistantId or
       tokenId == eosId or tokenId == imStartId or tokenId == imEndId or
       tokenId == eotId or tokenId == endTextId:
      break

    history.add(int32(tokenId))
    let tokenStr = tok.vocab[tokenId]
    if "<|im_start|>" notin tokenStr and "<|im_end|>" notin tokenStr:
      response.add(tokenStr)
      stdout.write(tokenStr)
      stdout.flushFile()

    # Next token: just the one we generated
    logits = forwardCached(m, gKv, @[int32(tokenId)])
    freeStepAllocations()

  response.strip()

# ── Main ──────────────────────────────────────────────────────────

when isMainModule:
  let baseDir = getAppDir().parentDir()
  let vidyaRoot = baseDir.parentDir()
  let tokenizerFile = vidyaRoot / "tokenizer_nim.bin"
  let modelFile = vidyaRoot / "nimllm.bin"
  # Check for GGUF file argument or Ollama model
  let ggufFile = if paramCount() >= 2 and paramStr(1) == "--gguf": paramStr(2)
                 else: ""

  gpuInit()

  if not fileExists(tokenizerFile):
    echo "no tokenizer found at ", tokenizerFile
    quit(1)
  let tok = loadTokenizer(tokenizerFile)

  var m: Model
  if ggufFile.len > 0:
    # Minimal init — only alloc embeddings + norms, not float32 weight matrices
    m = initModel(tok.vocab.len, withGradients = false, withWeights = false)
    loadModelGguf(m, ggufFile)
    # Already quantized — skip Q4_0 quantization
  elif fileExists(modelFile):
    loadModel(m, modelFile)
    if nEmbd >= 2048: quantizeModel(m)  # quantize float32 to Q4_0
  else:
    echo "no model found at ", modelFile
    echo "usage: chat [--gguf <path_to_gguf>]"
    quit(1)

  # Quantize to Q4_0 only if not already quantized (GGUF loads Q8_0 directly)
  if not m.quantized and nEmbd >= 2048:
    quantizeModel(m)

  randomize()
  # For inference, use small arena (1 token at a time with KV cache)
  # Don't allocate the huge training arena
  let infS = 32  # max tokens per forward (prompt processing)
  let infArena = (infS * nEmbd * 20 + infS * infS * 4 + infS * m.vocabSize * 2) * nLayer div 4
  initScratchArena(infArena)
  echo &"  inference arena: {infArena * 4 div 1024 div 1024}MB"

  # Non-interactive mode — find --prompt anywhere in args
  var promptIdx = -1
  for i in 1 .. paramCount():
    if paramStr(i) == "--prompt": promptIdx = i
  if promptIdx > 0 and promptIdx < paramCount():
    var prompt = paramStr(promptIdx + 1)
    for i in promptIdx + 2 .. paramCount():
      if paramStr(i) != "--gguf" and (i < 2 or paramStr(i-1) != "--gguf"):
        prompt &= " " & paramStr(i)
    var history: seq[int32]
    echo generate(m, tok, prompt, history)
    quit(0)

  # Interactive mode
  echo ""
  let pc = tok.vocab.len * nEmbd * 2 +
    nLayer * (2 * nEmbd * nEmbd + 2 * nKvDim * nEmbd + 3 * ffnDim * nEmbd)
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

    discard generate(m, tok, input, history)
    echo ""  # newline after streamed output
    echo ""
