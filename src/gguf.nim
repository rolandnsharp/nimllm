## gguf.nim — Load GGUF model files (from Ollama / llama.cpp)
##
## Reads pre-quantized weights directly. No float32 intermediate.
## Supports Q8_0 and F32 tensor types.

import std/[streams, strutils, tables, strformat]

type
  GgufTensor* = object
    name*: string
    dims*: seq[int64]
    dtype*: int32       # GGML type: 0=F32, 1=F16, 8=Q8_0, etc.
    offset*: int64      # offset from data section start

  GgufFile* = object
    version*: int
    tensors*: Table[string, GgufTensor]
    dataStart*: int64
    path*: string

const
  GGML_TYPE_F32* = 0'i32
  GGML_TYPE_F16* = 1'i32
  GGML_TYPE_Q4_0* = 2'i32
  GGML_TYPE_Q8_0* = 8'i32

proc readGgufString(s: Stream): string =
  let n = s.readInt64().int
  if n > 10_000_000:
    # Skip huge strings (tokenizer data)
    s.setPosition(s.getPosition() + n)
    return ""
  result = newString(n)
  if n > 0:
    discard s.readData(addr result[0], n)

proc skipBytes(s: Stream, n: int) =
  s.setPosition(s.getPosition() + n)

proc skipGgufValue(s: Stream, typ: int32) =
  case typ
  of 0: skipBytes(s, 1)           # uint8
  of 1: skipBytes(s, 1)           # int8
  of 2: skipBytes(s, 2)           # uint16
  of 3: skipBytes(s, 2)           # int16
  of 4: skipBytes(s, 4)           # uint32
  of 5: skipBytes(s, 4)           # int32
  of 6: skipBytes(s, 4)           # float32
  of 7: skipBytes(s, 1)           # bool
  of 8: discard readGgufString(s) # string
  of 9: # array
    let atype = s.readInt32()
    let alen = s.readInt64()
    for i in 0 ..< alen:
      skipGgufValue(s, atype)
  of 10: skipBytes(s, 8)          # uint64
  of 11: skipBytes(s, 8)          # int64
  of 12: skipBytes(s, 8)          # float64
  else:
    echo "  WARNING: unknown GGUF type ", typ

proc openGguf*(path: string): GgufFile =
  ## Parse GGUF header and tensor metadata. Does not load tensor data.
  result.path = path
  let s = newFileStream(path, fmRead)

  # Header
  var magic: array[4, char]
  discard s.readData(addr magic[0], 4)
  assert magic == ['G', 'G', 'U', 'F'], "Not a GGUF file"

  result.version = s.readUint32().int
  let tensorCount = s.readInt64()
  let kvCount = s.readInt64()

  echo &"  GGUF v{result.version}: {tensorCount} tensors, {kvCount} metadata"

  # Skip metadata KVs
  for i in 0 ..< kvCount:
    discard readGgufString(s)  # key
    let vtype = s.readInt32()
    skipGgufValue(s, vtype)

  # Read tensor infos
  result.tensors = initTable[string, GgufTensor]()
  for i in 0 ..< tensorCount:
    var t: GgufTensor
    t.name = readGgufString(s)
    let ndims = s.readUint32().int
    t.dims = newSeq[int64](ndims)
    for d in 0 ..< ndims:
      t.dims[d] = s.readInt64()
    t.dtype = s.readInt32()
    t.offset = s.readInt64()
    result.tensors[t.name] = t

  # Data starts at next alignment boundary
  let align = 32
  result.dataStart = ((s.getPosition() + align - 1) div align) * align

  s.close()
  echo &"  data at offset {result.dataStart} ({result.dataStart div 1024 div 1024}MB)"

proc tensorSize*(t: GgufTensor): int =
  ## Size in bytes of tensor data
  var numel: int64 = 1
  for d in t.dims: numel *= d
  case t.dtype
  of GGML_TYPE_F32: return numel.int * 4
  of GGML_TYPE_F16: return numel.int * 2
  of GGML_TYPE_Q8_0: return (numel.int div 32) * 34   # 34 bytes per 32 elements
  of GGML_TYPE_Q4_0: return (numel.int div 32) * 18   # 18 bytes per 32 elements
  else: return numel.int * 4  # assume F32

proc loadTensorRaw*(gguf: GgufFile, name: string): seq[uint8] =
  ## Load raw tensor bytes from GGUF file
  if name notin gguf.tensors:
    echo &"  WARNING: tensor {name} not found"
    return @[]
  let t = gguf.tensors[name]
  let size = tensorSize(t)
  let s = newFileStream(gguf.path, fmRead)
  s.setPosition(gguf.dataStart + t.offset)
  result = newSeq[uint8](size)
  discard s.readData(addr result[0], size)
  s.close()
