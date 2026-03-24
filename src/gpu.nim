## gpu.nim — Direct CUDA bindings for Vidya
##
## Nim compiles to C, so calling CUDA is just C FFI.
## No bridge files, no custom blocks, no bytecode wrappers.
## This is why we chose Nim.

{.passL: "src/kernels.o -lcudart -lcublas -lcurand -lstdc++".}
import std/math

# ── CUDA types ────────────────────────────────────────────────────

type
  CudaError* {.importc: "cudaError_t", header: "<cuda_runtime.h>".} = cint
  CublasHandle* {.importc: "cublasHandle_t", header: "<cublas_v2.h>".} = pointer
  CublasStatus* {.importc: "cublasStatus_t", header: "<cublas_v2.h>".} = cint
  CublasOperation* {.importc: "cublasOperation_t", header: "<cublas_v2.h>".} = cint
  CurandGenerator* {.importc: "curandGenerator_t", header: "<curand.h>".} = pointer

const
  CudaSuccess* = 0.CudaError
  CublasOpN* = 0.CublasOperation
  CublasOpT* = 1.CublasOperation

# ── CUDA runtime ──────────────────────────────────────────────────

proc cudaMalloc*(devPtr: ptr pointer, size: csize_t): CudaError
  {.importc, header: "<cuda_runtime.h>".}

proc cudaFree*(devPtr: pointer): CudaError
  {.importc, header: "<cuda_runtime.h>".}

proc cudaMemcpy*(dst, src: pointer, count: csize_t, kind: cint): CudaError
  {.importc, header: "<cuda_runtime.h>".}

proc cudaMemset*(devPtr: pointer, value: cint, count: csize_t): CudaError
  {.importc, header: "<cuda_runtime.h>".}

proc cudaDeviceSynchronize*(): CudaError
  {.importc, header: "<cuda_runtime.h>".}

const
  CudaMemcpyHostToDevice* = 1.cint
  CudaMemcpyDeviceToHost* = 2.cint
  CudaMemcpyDeviceToDevice* = 3.cint

# ── cuBLAS ────────────────────────────────────────────────────────

proc cublasCreate*(handle: ptr CublasHandle): CublasStatus
  {.importc: "cublasCreate_v2", header: "<cublas_v2.h>".}

proc cublasSgemm*(handle: CublasHandle,
                  transa, transb: CublasOperation,
                  m, n, k: cint,
                  alpha: ptr cfloat,
                  a: pointer, lda: cint,
                  b: pointer, ldb: cint,
                  beta: ptr cfloat,
                  c: pointer, ldc: cint): CublasStatus
  {.importc: "cublasSgemm_v2", header: "<cublas_v2.h>".}

proc cublasSaxpy*(handle: CublasHandle, n: cint,
                  alpha: ptr cfloat,
                  x: pointer, incx: cint,
                  y: pointer, incy: cint): CublasStatus
  {.importc: "cublasSaxpy_v2", header: "<cublas_v2.h>".}

proc cublasSgemmStridedBatched*(handle: CublasHandle,
    transa, transb: CublasOperation,
    m, n, k: cint,
    alpha: ptr cfloat,
    a: pointer, lda: cint, strideA: clonglong,
    b: pointer, ldb: cint, strideB: clonglong,
    beta: ptr cfloat,
    c: pointer, ldc: cint, strideC: clonglong,
    batchCount: cint): CublasStatus
  {.importc: "cublasSgemmStridedBatched", header: "<cublas_v2.h>".}

# ── cuRAND ────────────────────────────────────────────────────────

proc curandCreateGenerator*(gen: ptr CurandGenerator, rngType: cint): cint
  {.importc, header: "<curand.h>".}

proc curandGenerateUniform*(gen: CurandGenerator, outputPtr: pointer, num: csize_t): cint
  {.importc, header: "<curand.h>".}

# ── GPU Buffer type ───────────────────────────────────────────────

type
  GpuBuf* = object
    data*: pointer     ## cudaMalloc'd float32 device pointer
    numel*: int        ## number of float32 elements

## Note on memory: GpuBuf does NOT auto-free. The forward pass creates
## many temporaries per step. With auto-free, we'd need careful lifetime
## management. Instead, we manually free with gpuFree when needed, and
## let the process reclaim all GPU memory on exit.
##
## For training, we pre-allocate scratch buffers and reuse them.

proc gpuCreate*(n: int): GpuBuf =
  ## Allocate n float32 elements on GPU, zero-initialised.
  result.numel = n
  let err = cudaMalloc(addr result.data, csize_t(n * sizeof(cfloat)))
  assert err == CudaSuccess, "cudaMalloc failed"
  discard cudaMemset(result.data, 0, csize_t(n * sizeof(cfloat)))

proc gpuFree*(buf: var GpuBuf) =
  ## Free GPU memory.
  if buf.data != nil:
    discard cudaFree(buf.data)
    buf.data = nil
    buf.numel = 0

proc gpuUpload*(buf: GpuBuf, hostData: openArray[float32]) =
  ## Upload float32 array from CPU to GPU.
  assert hostData.len == buf.numel
  discard cudaMemcpy(buf.data, unsafeAddr hostData[0],
                     csize_t(buf.numel * sizeof(cfloat)),
                     CudaMemcpyHostToDevice)

proc gpuDownload*(buf: GpuBuf): seq[float32] =
  ## Download float32 array from GPU to CPU.
  result = newSeq[float32](buf.numel)
  discard cudaMemcpy(addr result[0], buf.data,
                     csize_t(buf.numel * sizeof(cfloat)),
                     CudaMemcpyDeviceToHost)

proc gpuCopy*(src, dst: GpuBuf, n: int) =
  ## Device-to-device copy.
  discard cudaMemcpy(dst.data, src.data,
                     csize_t(n * sizeof(cfloat)),
                     CudaMemcpyDeviceToDevice)

proc gpuZero*(buf: GpuBuf) =
  ## Zero all elements.
  discard cudaMemset(buf.data, 0, csize_t(buf.numel * sizeof(cfloat)))

# ── Global cuBLAS handle ─────────────────────────────────────────

var gCublas: CublasHandle

proc cublasSetMathMode*(handle: CublasHandle, mode: cint): CublasStatus
  {.importc: "cublasSetMathMode", header: "<cublas_v2.h>".}

const CUBLAS_PEDANTIC_MATH* = 2.cint  # full float32, no TF32

proc gpuInit*() =
  ## Initialise cuBLAS. Call once at startup.
  let err = cublasCreate(addr gCublas)
  assert err == 0, "cublasCreate failed"
  # Use default math mode (allows TF32 tensor cores for matmuls).
  # TF32 is safe — it only affects the multiply precision (10-bit mantissa),
  # accumulation is still full float32. The NaN issue was in attention scores,
  # not matmuls. Flash attention handles that now.
  # discard cublasSetMathMode(gCublas, CUBLAS_PEDANTIC_MATH)

# ── GEMM wrapper ──────────────────────────────────────────────────

proc gpuSgemm*(op: int, m, n, k: int, a, b, c: GpuBuf) =
  ## Row-major sgemm with 3-bit op encoding (same as OCaml version).
  ##   bit 2 (4): transpose A
  ##   bit 1 (2): transpose B
  ##   bit 0 (1): accumulate (beta=1) vs overwrite (beta=0)
  var alpha: cfloat = 1.0
  var beta: cfloat = if (op and 1) != 0: 1.0 else: 0.0

  let ta = (op and 4) != 0
  let tb = (op and 2) != 0

  # Row-major → col-major trick: swap A↔B and flip transposes
  let opB = if tb: CublasOpT else: CublasOpN
  let opA = if ta: CublasOpT else: CublasOpN
  let lda = if ta: cint(m) else: cint(k)
  let ldb = if tb: cint(k) else: cint(n)
  let ldc = cint(n)

  discard cublasSgemm(gCublas, opB, opA,
                      cint(n), cint(m), cint(k),
                      addr alpha, b.data, ldb,
                      a.data, lda,
                      addr beta, c.data, ldc)

# ── Convenience ───────────────────────────────────────────────────

proc toGpu*(data: openArray[float32]): GpuBuf =
  ## Upload a float32 array to a new GPU buffer.
  result = gpuCreate(data.len)
  gpuUpload(result, data)

# ── CUDA kernel bindings (from kernels.cu) ────────────────────────
# kernels.o is linked via the passL at the top of this file

proc gpu_gelu_fwd*(x, y: pointer, n: cint)
  {.importc, cdecl.}
proc gpu_gelu_bwd*(x, dy, dx: pointer, n: cint)
  {.importc, cdecl.}
proc gpu_add*(a, b, y: pointer, n: cint)
  {.importc, cdecl.}
proc gpu_add_inplace*(a, b: pointer, n: cint)
  {.importc, cdecl.}
proc gpu_scale*(x: pointer, s: cfloat, y: pointer, n: cint)
  {.importc, cdecl.}
proc gpu_rmsnorm_affine_fwd*(x, gamma, y, rms_out: pointer, rows, dim: cint)
  {.importc, cdecl.}
proc gpu_rmsnorm_affine_bwd*(x, gamma, dy, rms_out, dx, dgamma: pointer,
                             rows, dim: cint)
  {.importc, cdecl.}
proc gpu_causal_mask*(scores: pointer, scale: cfloat, seq_len: cint)
  {.importc, cdecl.}
proc gpu_softmax_fwd*(x, y: pointer, rows, cols: cint)
  {.importc, cdecl.}
proc gpu_softmax_bwd*(y, dy, dx: pointer, rows, cols: cint)
  {.importc, cdecl.}
proc gpu_extract_head*(src, dst: pointer, h, seq_len, n_embd, head_dim: cint)
  {.importc, cdecl.}
proc gpu_insert_head*(src, dst: pointer, h, seq_len, n_embd, head_dim: cint)
  {.importc, cdecl.}
proc gpu_insert_head_acc*(src, dst: pointer, h, seq_len, n_embd, head_dim: cint)
  {.importc, cdecl.}
proc gpu_rope_fwd*(data, cos_tab, sin_tab: pointer,
                   seq_len, n_embd, n_head, head_dim: cint)
  {.importc, cdecl.}
proc gpu_rope_bwd*(grad, cos_tab, sin_tab: pointer,
                   seq_len, n_embd, n_head, head_dim: cint)
  {.importc, cdecl.}
proc gpu_embed_fwd*(wte: pointer, tokens: pointer, output: pointer,
                    seq_len, dim: cint)
  {.importc, cdecl.}
proc gpu_embed_bwd*(wte_grad: pointer, tokens: pointer, dout: pointer,
                    seq_len, dim: cint)
  {.importc, cdecl.}
proc gpu_adam*(param, grad, m, v: pointer,
              lr, b1, b2, bc1, bc2: cfloat, n: cint)
  {.importc, cdecl.}
proc gpu_adamw*(param, grad, m, v: pointer,
               lr, b1, b2, bc1, bc2, wd: cfloat, n: cint)
  {.importc, cdecl.}
proc gpu_elastic*(param, anchor: pointer, alpha: cfloat, n: cint)
  {.importc, cdecl.}
proc gpu_swiglu_fwd*(gate, up, output: pointer, n: cint)
  {.importc, cdecl.}
proc gpu_swiglu_bwd*(gate, up, dout, dgate, dup: pointer, n: cint)
  {.importc, cdecl.}
proc gpu_dropout_fwd*(x, y, mask: pointer, rate: cfloat, n: cint)
  {.importc, cdecl.}
proc gpu_dropout_bwd*(dy, mask, dx: pointer, n: cint)
  {.importc, cdecl.}
proc gpu_extract_kv_head*(src, dst: pointer,
                          qHead, kvRepeat, seqLen, nKvDim, headDim: cint)
  {.importc, cdecl.}
proc gpu_insert_kv_head_acc*(src, dst: pointer,
                             qHead, kvRepeat, seqLen, nKvDim, headDim: cint)
  {.importc, cdecl.}
proc gpu_zero_upper*(data: pointer, seq_len: cint)
  {.importc, cdecl.}
proc gpu_clamp*(data: pointer, minv, maxv: cfloat, n: cint)
  {.importc, cdecl.}

# Non-affine RMSNorm: just normalize, no gamma. Uses affine kernel with ones.
var rmsnormOnes: GpuBuf
var rmsnormOnesSize = 0

proc gpu_rmsnorm_forward*(x, y, rms: pointer, rows, dim: cint) =
  ## RMSNorm without learnable gamma. Creates a shared ones buffer.
  if rmsnormOnesSize < dim.int:
    var ones = newSeq[float32](dim.int)
    for i in 0 ..< dim.int: ones[i] = 1.0f
    rmsnormOnes = toGpu(ones)
    rmsnormOnesSize = dim.int
  gpu_rmsnorm_affine_fwd(x, rmsnormOnes.data, y, rms, rows, dim)

proc gpu_rmsnorm_backward*(x_normed, dy, rms, dx: pointer, rows, dim: cint) =
  ## RMSNorm backward without gamma.
  if rmsnormOnesSize < dim.int:
    var ones = newSeq[float32](dim.int)
    for i in 0 ..< dim.int: ones[i] = 1.0f
    rmsnormOnes = toGpu(ones)
    rmsnormOnesSize = dim.int
  var dummy = gpuCreate(dim.int)  # dgamma — discarded
  gpu_rmsnorm_affine_bwd(x_normed, rmsnormOnes.data, dy, rms, dx, dummy.data, rows, dim)
proc gpu_log_softmax*(x, y: pointer, rows, cols: cint)
  {.importc, cdecl.}
proc gpu_flash_attn_fwd*(Q, K, V, O: pointer, S, hd: cint,
                          scale: cfloat, causal: cint)
  {.importc, cdecl.}
proc gpu_flash_attn_bwd*(Q, K, V, gradO, gradQ, gradK, gradV: pointer,
                          S, hd: cint, scale: cfloat, causal: cint)
  {.importc, cdecl.}
proc gpu_cross_entropy_loss*(log_probs: pointer, target, vocab: cint): cfloat
  {.importc, cdecl.}
proc gpu_ce_backward*(log_probs, targets, d_logits: pointer, S, V: cint)
  {.importc, cdecl.}
proc gpu_ce_loss*(log_probs, targets, scratch: pointer, S, V: cint): cfloat
  {.importc, cdecl.}

# ── High-level wrappers using GpuBuf ─────────────────────────────

proc geluFwd*(x, y: GpuBuf) =
  gpu_gelu_fwd(x.data, y.data, cint(x.numel))

proc geluBwd*(x, dy, dx: GpuBuf) =
  gpu_gelu_bwd(x.data, dy.data, dx.data, cint(x.numel))

proc addBufs*(a, b, y: GpuBuf) =
  gpu_add(a.data, b.data, y.data, cint(a.numel))

proc addInplace*(a, b: GpuBuf) =
  gpu_add_inplace(a.data, b.data, cint(a.numel))

proc scaleBuf*(x: GpuBuf, s: float32, y: GpuBuf) =
  gpu_scale(x.data, s, y.data, cint(x.numel))

proc rmsnormAffineFwd*(x, gamma, y, rms: GpuBuf, rows, dim: int) =
  gpu_rmsnorm_affine_fwd(x.data, gamma.data, y.data, rms.data,
                         cint(rows), cint(dim))

proc rmsnormAffineBwd*(x, gamma, dy, rms, dx, dgamma: GpuBuf, rows, dim: int) =
  gpu_rmsnorm_affine_bwd(x.data, gamma.data, dy.data, rms.data,
                         dx.data, dgamma.data, cint(rows), cint(dim))

proc causalMask*(scores: GpuBuf, scale: float32, seqLen: int) =
  gpu_causal_mask(scores.data, scale, cint(seqLen))

proc softmaxFwd*(x, y: GpuBuf, rows, cols: int) =
  gpu_softmax_fwd(x.data, y.data, cint(rows), cint(cols))

proc softmaxBwd*(y, dy, dx: GpuBuf, rows, cols: int) =
  gpu_softmax_bwd(y.data, dy.data, dx.data, cint(rows), cint(cols))

proc extractHead*(src, dst: GpuBuf, h, seqLen, nEmbd, headDim: int) =
  gpu_extract_head(src.data, dst.data, cint(h), cint(seqLen),
                   cint(nEmbd), cint(headDim))

proc insertHead*(src, dst: GpuBuf, h, seqLen, nEmbd, headDim: int) =
  gpu_insert_head(src.data, dst.data, cint(h), cint(seqLen),
                  cint(nEmbd), cint(headDim))

proc insertHeadAcc*(src, dst: GpuBuf, h, seqLen, nEmbd, headDim: int) =
  gpu_insert_head_acc(src.data, dst.data, cint(h), cint(seqLen),
                      cint(nEmbd), cint(headDim))

proc ropeFwd*(data, cosTab, sinTab: GpuBuf, seqLen, nEmbd, nHead, headDim: int) =
  gpu_rope_fwd(data.data, cosTab.data, sinTab.data,
               cint(seqLen), cint(nEmbd), cint(nHead), cint(headDim))

proc ropeBwd*(grad, cosTab, sinTab: GpuBuf, seqLen, nEmbd, nHead, headDim: int) =
  gpu_rope_bwd(grad.data, cosTab.data, sinTab.data,
               cint(seqLen), cint(nEmbd), cint(nHead), cint(headDim))

proc adamStep*(param, grad, m, v: GpuBuf,
               lr, b1, b2, bc1, bc2: float32) =
  gpu_adam(param.data, grad.data, m.data, v.data,
           lr, b1, b2, bc1, bc2, cint(param.numel))

proc adamwStep*(param, grad, m, v: GpuBuf,
                lr, b1, b2, bc1, bc2, wd: float32) =
  gpu_adamw(param.data, grad.data, m.data, v.data,
            lr, b1, b2, bc1, bc2, wd, cint(param.numel))

proc elasticPull*(param, anchor: GpuBuf, alpha: float32) =
  gpu_elastic(param.data, anchor.data, alpha, cint(param.numel))

# ── Gradient clipping (GPU-side) ──────────────────────────────────

proc gpu_grad_norm*(grads: ptr pointer, sizes: ptr cint, n: cint): cfloat
  {.importc, cdecl.}
proc gpu_clip_grads*(grads: ptr pointer, sizes: ptr cint, n: cint, max_norm: cfloat)
  {.importc, cdecl.}

proc clipGradNorm*(gradPtrs: var seq[pointer], sizes: var seq[cint],
                   maxNorm: float32) =
  ## Clip global gradient norm. Caller prepares pointer/size arrays.
  gpu_clip_grads(addr gradPtrs[0], addr sizes[0], cint(gradPtrs.len), maxNorm)

# ── NLL (scalar return) ───────────────────────────────────────────

proc gpu_nll_fwd*(probs: pointer, target: cint): cfloat =
  ## Download probs[target], return -log(p). Tiny transfer.
  var p: cfloat
  discard cudaMemcpy(addr p,
    cast[pointer](cast[int](probs) + target.int * sizeof(cfloat)),
    csize_t(sizeof(cfloat)), CudaMemcpyDeviceToHost)
  result = -ln(max(p, 1e-10f))
