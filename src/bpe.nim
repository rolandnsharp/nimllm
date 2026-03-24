## bpe.nim — Byte Pair Encoding tokenizer
##
## Trains on a corpus of text, learns merge rules (default 2000), produces
## a vocabulary of 256 byte-level + nMerges + 3 special tokens.
##
## Can save/load to a portable binary format (not OCaml Marshal).

import std/[tables, strutils, streams, strformat]

const
  nMerges* = 4000
  bosToken* = "<|bos|>"
  userToken* = "<|user|>"
  assistantToken* = "<|assistant|>"

type
  MergeRule = tuple[a, b: int, merged: int]

  Tokenizer* = object
    vocab*: seq[string]          # id → string
    tokenToId*: Table[string, int]
    merges: seq[MergeRule]
    bosId*: int
    userId*: int
    assistantId*: int

# ── Build base vocabulary ─────────────────────────────────────────

proc buildBaseVocab(): (seq[string], Table[string, int]) =
  ## Create byte-level base vocabulary (all single chars found + specials).
  var vocab: seq[string]
  var tokenToId: Table[string, int]

  # Add all printable ASCII + common bytes
  for c in 0 .. 255:
    let s = $chr(c)
    if s.len > 0:
      tokenToId[s] = vocab.len
      vocab.add(s)

  # Special tokens
  for special in [bosToken, userToken, assistantToken]:
    tokenToId[special] = vocab.len
    vocab.add(special)

  (vocab, tokenToId)

# ── Training ──────────────────────────────────────────────────────

proc countPairs(corpus: seq[seq[int]], specialIds: set[uint16]): CountTable[(int, int)] =
  ## Count adjacent token pairs across all documents.
  ## Never count pairs involving special tokens — they must stay atomic.
  result = initCountTable[(int, int)]()
  for doc in corpus:
    for i in 0 ..< doc.len - 1:
      if doc[i].uint16 notin specialIds and doc[i+1].uint16 notin specialIds:
        result.inc((doc[i], doc[i + 1]))

proc mergePair(corpus: var seq[seq[int]], pair: (int, int), newId: int) =
  ## Replace all occurrences of pair with newId in-place.
  for doc in corpus.mitems:
    var i = 0
    var merged: seq[int]
    while i < doc.len:
      if i < doc.len - 1 and doc[i] == pair[0] and doc[i + 1] == pair[1]:
        merged.add(newId)
        i += 2
      else:
        merged.add(doc[i])
        i += 1
    doc = merged

proc trainBpe*(docs: seq[string], numMerges: int = nMerges,
               maxDocs: int = 50000): Tokenizer =
  ## Train BPE tokenizer on a corpus of documents.
  ## Subsamples to maxDocs for speed — 50K docs gives the same vocab.
  var (vocab, tokenToId) = buildBaseVocab()
  var merges: seq[MergeRule]

  # Subsample if corpus is too large — BPE vocab converges fast
  var trainDocs = docs
  if docs.len > maxDocs:
    echo &"  subsampling {maxDocs} of {docs.len} docs for BPE training"
    trainDocs = docs[0 ..< maxDocs]

  # Initial tokenization: each char becomes its byte-level token ID.
  echo &"  tokenizing {trainDocs.len} docs for BPE..."
  var corpus: seq[seq[int]]
  for idx, doc in trainDocs:
    var tokens: seq[int]
    var i = 0
    while i < doc.len:
      var matched = false
      for special in [bosToken, userToken, assistantToken]:
        if doc[i ..< min(i + special.len, doc.len)] == special:
          tokens.add(tokenToId[special])
          i += special.len
          matched = true
          break
      if not matched:
        let c = $doc[i]
        if c in tokenToId:
          tokens.add(tokenToId[c])
        i += 1
    corpus.add(tokens)
    if (idx + 1) mod 10000 == 0:
      echo &"    {idx + 1}/{trainDocs.len} docs tokenized"

  echo "training BPE: ", trainDocs.len, " docs (of ", docs.len, " total), base vocab: ", vocab.len

  # Special tokens that must never be merged with adjacent tokens
  let specialIds = {tokenToId[bosToken].uint16,
                    tokenToId[userToken].uint16,
                    tokenToId[assistantToken].uint16}

  # Iteratively merge the most frequent pair
  for m in 0 ..< numMerges:
    let pairs = countPairs(corpus, specialIds)
    if pairs.len == 0:
      break
    let (bestPair, _) = pairs.largest()
    let newId = vocab.len
    let newToken = vocab[bestPair[0]] & vocab[bestPair[1]]
    tokenToId[newToken] = newId
    vocab.add(newToken)
    merges.add((bestPair[0], bestPair[1], newId))
    mergePair(corpus, bestPair, newId)

    if (m + 1) mod 100 == 0:
      echo "  merge ", m + 1, "/", numMerges, " vocab: ", vocab.len

  result = Tokenizer(
    vocab: vocab,
    tokenToId: tokenToId,
    merges: merges,
    bosId: tokenToId[bosToken],
    userId: tokenToId[userToken],
    assistantId: tokenToId[assistantToken],
  )

# ── Encoding ──────────────────────────────────────────────────────

proc encode*(tok: Tokenizer, text: string): seq[int] =
  ## Encode text to token IDs.
  ## If merges exist: BPE (character-level + merge rules).
  ## If no merges: greedy longest-match against vocabulary.
  if tok.merges.len == 0:
    # Greedy longest-match encoding for pre-trained tokenizers (e.g. SmolLM).
    # At each position, find the longest vocab entry that matches.
    var tokens: seq[int]
    var i = 0
    while i < text.len:
      var bestLen = 0
      var bestId = -1
      # Try lengths from longest reasonable down to 1
      for tryLen in countdown(min(text.len - i, 32), 1):
        let sub = text[i ..< i + tryLen]
        if sub in tok.tokenToId:
          bestLen = tryLen
          bestId = tok.tokenToId[sub]
          break
      if bestId >= 0:
        tokens.add(bestId)
        i += bestLen
      else:
        # Unknown byte — skip
        i += 1
    return tokens

  # BPE encoding: start with byte-level tokenization, then apply merges
  var tokens: seq[int]
  var i = 0
  while i < text.len:
    var matched = false
    for special in [bosToken, userToken, assistantToken]:
      if text[i ..< min(i + special.len, text.len)] == special:
        tokens.add(tok.tokenToId[special])
        i += special.len
        matched = true
        break
    if not matched:
      let c = $text[i]
      if c in tok.tokenToId:
        tokens.add(tok.tokenToId[c])
      i += 1

  # Fast merge: hash table lookup, merge in-place.
  var pairLookup {.global.}: Table[(int, int), int]
  if pairLookup.len == 0:
    for merge in tok.merges:
      pairLookup[(merge.a, merge.b)] = merge.merged

  var changed = true
  while changed:
    changed = false
    var j = 0
    var newTokens = newSeqOfCap[int](tokens.len)
    while j < tokens.len:
      if j < tokens.len - 1:
        let pair = (tokens[j], tokens[j + 1])
        if pair in pairLookup:
          newTokens.add(pairLookup[pair])
          j += 2
          changed = true
          continue
      newTokens.add(tokens[j])
      j += 1
    tokens = newTokens

  tokens

proc decode*(tok: Tokenizer, ids: seq[int]): string =
  ## Decode token IDs back to text.
  for id in ids:
    if id >= 0 and id < tok.vocab.len:
      result.add(tok.vocab[id])

# ── Save / Load (portable binary format) ─────────────────────────

proc saveTokenizer*(tok: Tokenizer, filename: string) =
  ## Save tokenizer to a portable binary format.
  ## Format: vocab_size (int32), then for each token: len (int16) + bytes,
  ## then num_merges (int32), then for each merge: a (int32) + b (int32) + merged (int32).
  let s = newFileStream(filename, fmWrite)
  defer: s.close()

  # Vocab
  s.write(int32(tok.vocab.len))
  for token in tok.vocab:
    s.write(int16(token.len))
    s.writeData(token.cstring, token.len)

  # Merges
  s.write(int32(tok.merges.len))
  for merge in tok.merges:
    s.write(int32(merge.a))
    s.write(int32(merge.b))
    s.write(int32(merge.merged))

proc loadTokenizer*(filename: string): Tokenizer =
  ## Load tokenizer from portable binary format.
  let s = newFileStream(filename, fmRead)
  defer: s.close()

  # Vocab
  let vocabSize = s.readInt32().int
  var vocab = newSeq[string](vocabSize)
  var tokenToId = initTable[string, int]()
  for i in 0 ..< vocabSize:
    let len = s.readInt16().int
    var buf = newString(len)
    if len > 0:
      discard s.readData(addr buf[0], len)
    vocab[i] = buf
    tokenToId[buf] = i

  # Merges
  let numMerges = s.readInt32().int
  var merges = newSeq[MergeRule](numMerges)
  for i in 0 ..< numMerges:
    let a = s.readInt32().int
    let b = s.readInt32().int
    let merged = s.readInt32().int
    merges[i] = (a, b, merged)

  result = Tokenizer(
    vocab: vocab,
    tokenToId: tokenToId,
    merges: merges,
    bosId: tokenToId.getOrDefault(bosToken, 0),
    userId: tokenToId.getOrDefault(userToken, 0),
    assistantId: tokenToId.getOrDefault(assistantToken, 0),
  )

# ── Data loading ──────────────────────────────────────────────────

proc loadDocs*(filename: string): seq[string] =
  ## Load documents from a text file (one per line).
  let content = readFile(filename)
  for line in content.splitLines():
    let trimmed = line.strip()
    if trimmed.len > 0:
      result.add(trimmed)
