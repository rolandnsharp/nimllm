## chat.nim — Talk to nimllm
##
## Loads the trained model, generates text token by token.
## Uses the shared forward pass from model.nim.

import model, gpu, bpe, autograd
import std/[math, random, strformat, os, terminal, strutils, tables]

# ── Sampling ──────────────────────────────────────────────────────

proc sample(logits: seq[float32], temperature: float32 = 0.6f,
            topK: int = 40): int =
  var scaled = newSeq[float32](logits.len)
  for i in 0 ..< logits.len:
    scaled[i] = logits[i] / temperature

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
  # Format prompt
  let hasImTokens = tok.tokenToId.getOrDefault("<|im_start|>", -1) >= 0
  let imStart = if hasImTokens: "<|im_start|>" else: ""
  let imEnd = if hasImTokens: "<|im_end|>" else: ""
  let chatPrompt = if imStart.len > 0:
      (if history.len == 0: imStart & "system\nYou are a helpful assistant." & imEnd & "\n" else: "") &
      imStart & "user\n" & prompt & imEnd & "\n" & imStart & "assistant\n"
    else:
      "<|user|> " & prompt & " <|assistant|> "
  let inputTokens = tok.encode(chatPrompt)
  for id in inputTokens:
    history.add(int32(id))

  # Stop tokens
  let eosId = tok.tokenToId.getOrDefault("<|endoftext|>", -1)
  let imStartId = tok.tokenToId.getOrDefault("<|im_start|>", -1)
  let imEndId = tok.tokenToId.getOrDefault("<|im_end|>", -1)

  # Init KV cache on first use
  if gKv.k.len == 0:
    gKv = initKvCache()

  # Process prompt tokens through KV cache (all at once)
  var promptIds = newSeq[int32](inputTokens.len)
  for i, id in inputTokens: promptIds[i] = int32(id)
  var logits = forwardCached(m, gKv, promptIds)
  freeStepAllocations()

  # Generate tokens one at a time
  var response = ""
  for _ in 0 ..< maxTokens:
    let tokenId = sample(logits)

    if tokenId == tok.bosId or tokenId == tok.userId or
       tokenId == tok.assistantId or
       tokenId == eosId or tokenId == imStartId or tokenId == imEndId:
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

  gpuInit()

  if not fileExists(tokenizerFile):
    echo "no tokenizer found at ", tokenizerFile
    quit(1)
  let tok = loadTokenizer(tokenizerFile)

  var m = initModel(tok.vocab.len, withGradients = false)
  if fileExists(modelFile):
    loadModel(m, modelFile)
  else:
    echo "no model found at ", modelFile
    quit(1)

  randomize()
  trackingEnabled = true

  # Non-interactive mode
  if paramCount() >= 2 and paramStr(1) == "--prompt":
    var prompt = paramStr(2)
    for i in 3 .. paramCount(): prompt &= " " & paramStr(i)
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
