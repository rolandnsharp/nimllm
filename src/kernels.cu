/* kernels.cu — CUDA kernels for Vidya (Nim version)
 *
 * Clean reimplementation. No OCaml bridge overhead.
 * Called directly from Nim via {.importc.} pragmas. */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

#define BLOCK 256

extern "C" {

/* ── GELU ────────────────────────────────────────────────────────── */

__global__ void k_gelu_fwd(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi = x[i];
    float t = tanhf(0.7978845608f * (xi + 0.044715f * xi * xi * xi));
    y[i] = 0.5f * xi * (1.0f + t);
}

__global__ void k_gelu_bwd(const float *x, const float *dy, float *dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi = x[i];
    float inner = 0.7978845608f * (xi + 0.044715f * xi * xi * xi);
    float t = tanhf(inner);
    float dt = 1.0f - t * t;
    float dg = 0.5f * (1.0f + t)
        + 0.5f * xi * dt * 0.7978845608f * (1.0f + 3.0f * 0.044715f * xi * xi);
    dx[i] += dy[i] * dg;
}

void gpu_gelu_fwd(const float *x, float *y, int n) {
    k_gelu_fwd<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(x, y, n);
}

void gpu_gelu_bwd(const float *x, const float *dy, float *dx, int n) {
    k_gelu_bwd<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(x, dy, dx, n);
}

/* ── Dropout ──────────────────────────────────────────────────────
 *
 * Generate uniform randoms, threshold to binary mask, scale survivors.
 * Requires cuRAND — init generator at startup. */

static curandGenerator_t g_curand = NULL;

__attribute__((constructor))
static void init_curand(void) {
    curandCreateGenerator(&g_curand, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(g_curand, 42ULL);
}

__global__ void k_dropout_fwd(const float *x, float *y, float *mask,
                               float rate, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (mask[i] >= rate) {
        y[i] = x[i] * scale;
        mask[i] = scale;
    } else {
        y[i] = 0.0f;
        mask[i] = 0.0f;
    }
}

__global__ void k_dropout_bwd(const float *dy, const float *mask,
                               float *dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dx[i] += dy[i] * mask[i];
}

void gpu_dropout_fwd(const float *x, float *y, float *mask,
                     float rate, int n) {
    float scale = 1.0f / (1.0f - rate);
    curandGenerateUniform(g_curand, mask, n);
    k_dropout_fwd<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(x, y, mask, rate, scale, n);
}

void gpu_dropout_bwd(const float *dy, const float *mask, float *dx, int n) {
    k_dropout_bwd<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(dy, mask, dx, n);
}

/* ── SwiGLU ──────────────────────────────────────────────────────
 *
 * SwiGLU(gate, up) = swish(gate) * up
 * where swish(x) = x * sigmoid(x)
 *
 * Replaces GELU in the FFN. Two input projections (gate and up)
 * instead of one. Better empirically — used by Llama, Mistral, etc. */

__global__ void k_swiglu_fwd(const float *gate, const float *up,
                              float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float swish = g / (1.0f + expf(-g));
    out[i] = swish * up[i];
}

__global__ void k_swiglu_bwd(const float *gate, const float *up,
                              const float *dout,
                              float *dgate, float *dup, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float sig = 1.0f / (1.0f + expf(-g));
    float swish = g * sig;
    float dswish = sig * (1.0f + g * (1.0f - sig));
    dup[i] += dout[i] * swish;
    dgate[i] += dout[i] * up[i] * dswish;
}

void gpu_swiglu_fwd(const float *gate, const float *up, float *out, int n) {
    k_swiglu_fwd<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(gate, up, out, n);
}

void gpu_swiglu_bwd(const float *gate, const float *up, const float *dout,
                    float *dgate, float *dup, int n) {
    k_swiglu_bwd<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        gate, up, dout, dgate, dup, n);
}

/* ── Element-wise ────────────────────────────────────────────────── */

__global__ void k_add(const float *a, const float *b, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = a[i] + b[i];
}

__global__ void k_add_inplace(float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] += b[i];
}

__global__ void k_scale(const float *x, float s, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = x[i] * s;
}

void gpu_add(const float *a, const float *b, float *y, int n) {
    k_add<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(a, b, y, n);
}

void gpu_add_inplace(float *a, const float *b, int n) {
    k_add_inplace<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(a, b, n);
}

void gpu_scale(const float *x, float s, float *y, int n) {
    k_scale<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(x, s, y, n);
}

/* ── RMSNorm ─────────────────────────────────────────────────────
 *
 * Each block handles one row. Shared memory for sum-of-squares. */

__global__ void k_rmsnorm_affine(const float *x, const float *gamma,
                                 float *y, float *rms_out,
                                 int rows, int dim) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *xi = x + row * dim;
    float *yi = y + row * dim;

    extern __shared__ float sdata[];

    float ss = 0.0f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x)
        ss += xi[j] * xi[j];
    sdata[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = sqrtf(sdata[0] / (float)dim + 1e-5f);
    if (threadIdx.x == 0) rms_out[row] = rms;

    float inv = 1.0f / rms;
    for (int j = threadIdx.x; j < dim; j += blockDim.x)
        yi[j] = xi[j] * inv * gamma[j];
}

__global__ void k_rmsnorm_affine_bwd(const float *x, const float *gamma,
                                     const float *dy, const float *rms_out,
                                     float *dx, float *dgamma,
                                     int rows, int dim) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *xi = x + row * dim;
    const float *dyi = dy + row * dim;
    float *dxi = dx + row * dim;
    float inv = 1.0f / rms_out[row];
    float dimf = (float)dim;

    extern __shared__ float sdata[];

    float dot = 0.0f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float dn = dyi[j] * gamma[j];
        dot += dn * xi[j] * inv;
    }
    sdata[threadIdx.x] = dot;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean_dot = sdata[0] / dimf;

    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float ni = xi[j] * inv;
        atomicAdd(&dgamma[j], dyi[j] * ni);
        float dn = dyi[j] * gamma[j];
        dxi[j] += (dn - ni * mean_dot) * inv;
    }
}

void gpu_rmsnorm_affine_fwd(const float *x, const float *gamma,
                            float *y, float *rms_out, int rows, int dim) {
    int threads = dim < 256 ? dim : 256;
    k_rmsnorm_affine<<<rows, threads, threads * sizeof(float)>>>(
        x, gamma, y, rms_out, rows, dim);
}

void gpu_rmsnorm_affine_bwd(const float *x, const float *gamma,
                            const float *dy, const float *rms_out,
                            float *dx, float *dgamma, int rows, int dim) {
    int threads = dim < 256 ? dim : 256;
    k_rmsnorm_affine_bwd<<<rows, threads, threads * sizeof(float)>>>(
        x, gamma, dy, rms_out, dx, dgamma, rows, dim);
}

/* ── Causal mask + scale ─────────────────────────────────────────── */

__global__ void k_causal_mask(float *scores, float scale, int seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * seq_len;
    if (idx >= total) return;
    int row = idx / seq_len;
    int col = idx % seq_len;
    if (col > row)
        scores[idx] = -1e9f;
    else
        scores[idx] *= scale;
}

void gpu_causal_mask(float *scores, float scale, int seq_len) {
    int n = seq_len * seq_len;
    k_causal_mask<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(scores, scale, seq_len);
}

/* ── Softmax (row-wise) ──────────────────────────────────────────── */

__global__ void k_softmax(const float *x, float *y, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *xi = x + row * cols;
    float *yi = y + row * cols;

    extern __shared__ float sdata[];

    float mx = -FLT_MAX;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        mx = fmaxf(mx, xi[j]);
    sdata[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = sdata[0];

    float sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float e = expf(xi[j] - row_max);
        yi[j] = e;
        sum += e;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv = 1.0f / sdata[0];
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        yi[j] *= inv;
}

__global__ void k_softmax_bwd(const float *y, const float *dy,
                               float *dx, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *yi = y + row * cols;
    const float *dyi = dy + row * cols;
    float *dxi = dx + row * cols;

    extern __shared__ float sdata[];

    float dot = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        dot += dyi[j] * yi[j];
    sdata[threadIdx.x] = dot;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float d = sdata[0];
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        dxi[j] += yi[j] * (dyi[j] - d);
}

static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p < 256 ? p : 256;
}

void gpu_softmax_fwd(const float *x, float *y, int rows, int cols) {
    int threads = next_pow2(cols);
    k_softmax<<<rows, threads, threads * sizeof(float)>>>(x, y, rows, cols);
}

void gpu_softmax_bwd(const float *y, const float *dy, float *dx,
                     int rows, int cols) {
    int threads = next_pow2(cols);
    k_softmax_bwd<<<rows, threads, threads * sizeof(float)>>>(y, dy, dx, rows, cols);
}

/* ── Head extraction/insertion ───────────────────────────────────── */

__global__ void k_extract_head(const float *src, float *dst, int h,
                               int seq_len, int n_embd, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * head_dim) return;
    int pos = idx / head_dim;
    int j = idx % head_dim;
    dst[idx] = src[pos * n_embd + h * head_dim + j];
}

__global__ void k_insert_head(const float *src, float *dst, int h,
                              int seq_len, int n_embd, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * head_dim) return;
    int pos = idx / head_dim;
    int j = idx % head_dim;
    dst[pos * n_embd + h * head_dim + j] = src[idx];
}

__global__ void k_insert_head_acc(const float *src, float *dst, int h,
                                  int seq_len, int n_embd, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * head_dim) return;
    int pos = idx / head_dim;
    int j = idx % head_dim;
    dst[pos * n_embd + h * head_dim + j] += src[idx];
}

void gpu_extract_head(const float *src, float *dst, int h,
                      int seq_len, int n_embd, int head_dim) {
    int n = seq_len * head_dim;
    k_extract_head<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        src, dst, h, seq_len, n_embd, head_dim);
}

void gpu_insert_head(const float *src, float *dst, int h,
                     int seq_len, int n_embd, int head_dim) {
    int n = seq_len * head_dim;
    k_insert_head<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        src, dst, h, seq_len, n_embd, head_dim);
}

void gpu_insert_head_acc(const float *src, float *dst, int h,
                         int seq_len, int n_embd, int head_dim) {
    int n = seq_len * head_dim;
    k_insert_head_acc<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        src, dst, h, seq_len, n_embd, head_dim);
}

/* ── RoPE ────────────────────────────────────────────────────────── */

__global__ void k_rope(float *data, const float *cos_tab,
                       const float *sin_tab, int seq_len, int n_embd,
                       int n_head, int head_dim, int half_dim, int sign) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * n_head * half_dim) return;
    int pos = idx / (n_head * half_dim);
    int rem = idx % (n_head * half_dim);
    int h = rem / half_dim;
    int f = rem % half_dim;
    int base = pos * n_embd + h * head_dim;
    float c = cos_tab[pos * half_dim + f];
    float s = sin_tab[pos * half_dim + f];
    float x0 = data[base + f];
    float x1 = data[base + f + half_dim];
    data[base + f]            = x0 * c - (float)sign * x1 * s;
    data[base + f + half_dim] = (float)sign * x0 * s + x1 * c;
}

void gpu_rope_fwd(float *data, const float *cos_tab, const float *sin_tab,
                  int seq_len, int n_embd, int n_head, int head_dim) {
    int hd = head_dim / 2;
    int n = seq_len * n_head * hd;
    k_rope<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        data, cos_tab, sin_tab, seq_len, n_embd, n_head, head_dim, hd, 1);
}

void gpu_rope_bwd(float *grad, const float *cos_tab, const float *sin_tab,
                  int seq_len, int n_embd, int n_head, int head_dim) {
    int hd = head_dim / 2;
    int n = seq_len * n_head * hd;
    k_rope<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        grad, cos_tab, sin_tab, seq_len, n_embd, n_head, head_dim, hd, -1);
}

/* ── GQA head extraction ─────────────────────────────────────────
 *
 * Q heads: extract normally (each Q head has its own slice)
 * KV heads: fewer heads, each shared by kvRepeat Q heads.
 * extract_kv_head extracts KV head (h / kvRepeat) for Q head h. */

__global__ void k_extract_kv_head(const float *src, float *dst,
                                   int qHead, int kvRepeat,
                                   int seq_len, int n_kv_dim,
                                   int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * head_dim) return;
    int pos = idx / head_dim;
    int j = idx % head_dim;
    int kvHead = qHead / kvRepeat;
    dst[idx] = src[pos * n_kv_dim + kvHead * head_dim + j];
}

__global__ void k_insert_kv_head_acc(const float *src, float *dst,
                                      int qHead, int kvRepeat,
                                      int seq_len, int n_kv_dim,
                                      int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * head_dim) return;
    int pos = idx / head_dim;
    int j = idx % head_dim;
    int kvHead = qHead / kvRepeat;
    atomicAdd(&dst[pos * n_kv_dim + kvHead * head_dim + j], src[idx]);
}

void gpu_extract_kv_head(const float *src, float *dst,
                         int qHead, int kvRepeat,
                         int seq_len, int n_kv_dim, int head_dim) {
    int n = seq_len * head_dim;
    k_extract_kv_head<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        src, dst, qHead, kvRepeat, seq_len, n_kv_dim, head_dim);
}

void gpu_insert_kv_head_acc(const float *src, float *dst,
                            int qHead, int kvRepeat,
                            int seq_len, int n_kv_dim, int head_dim) {
    int n = seq_len * head_dim;
    k_insert_kv_head_acc<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        src, dst, qHead, kvRepeat, seq_len, n_kv_dim, head_dim);
}

/* ── Embedding ───────────────────────────────────────────────────── */

__global__ void k_embed_fwd(const float *wte, const int *tokens,
                            float *out, int seq_len, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * dim) return;
    int pos = idx / dim;
    int j = idx % dim;
    out[idx] = wte[tokens[pos] * dim + j];
}

__global__ void k_embed_bwd(float *wte_grad, const int *tokens,
                            const float *dout, int seq_len, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * dim) return;
    int pos = idx / dim;
    int j = idx % dim;
    atomicAdd(&wte_grad[tokens[pos] * dim + j], dout[idx]);
}

void gpu_embed_fwd(const float *wte, const int *tokens, float *out,
                   int seq_len, int dim) {
    int n = seq_len * dim;
    k_embed_fwd<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(wte, tokens, out, seq_len, dim);
}

void gpu_embed_bwd(float *wte_grad, const int *tokens, const float *dout,
                   int seq_len, int dim) {
    int n = seq_len * dim;
    k_embed_bwd<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(wte_grad, tokens, dout, seq_len, dim);
}

/* ── Adam ────────────────────────────────────────────────────────── */

/* AdamW: Adam with decoupled weight decay.
 * Weight decay is applied directly to params, not through gradient.
 * This prevents weights from growing unbounded — critical for stability. */
__global__ void k_adamw(float *param, float *grad, float *m, float *v,
                        float lr, float b1, float b2,
                        float bc1, float bc2, float wd, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    m[i] = b1 * m[i] + (1.0f - b1) * g;
    v[i] = b2 * v[i] + (1.0f - b2) * g * g;
    /* Decoupled weight decay: applied to param directly */
    param[i] *= (1.0f - lr * wd);
    /* Adam update */
    param[i] -= lr * (m[i] * bc1) / (sqrtf(v[i] * bc2) + 1e-8f);
    grad[i] = 0.0f;
}

void gpu_adamw(float *param, float *grad, float *m, float *v,
               float lr, float b1, float b2, float bc1, float bc2,
               float wd, int n) {
    k_adamw<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        param, grad, m, v, lr, b1, b2, bc1, bc2, wd, n);
}

/* Keep old adam for backward compat */
void gpu_adam(float *param, float *grad, float *m, float *v,
             float lr, float b1, float b2, float bc1, float bc2, int n) {
    gpu_adamw(param, grad, m, v, lr, b1, b2, bc1, bc2, 0.0f, n);
}

/* ── Elastic pull ────────────────────────────────────────────────── */

__global__ void k_elastic(float *param, const float *anchor, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    param[i] = (1.0f - alpha) * param[i] + alpha * anchor[i];
}

void gpu_elastic(float *param, const float *anchor, float alpha, int n) {
    k_elastic<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(param, anchor, alpha, n);
}

/* ── Gradient norm + clipping ─────────────────────────────────────── */

/* Compute sum of squares of a float array. Returns scalar on host. */
__global__ void k_sum_sq(const float *data, float *out, int n) {
    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? data[idx] * data[idx] : 0.0f;
    sdata[threadIdx.x] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(out, sdata[0]);
}

float gpu_grad_norm(const float **grads, const int *sizes, int n_tensors) {
    /* Allocate a single float on device for accumulation. */
    float *d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    for (int t = 0; t < n_tensors; t++) {
        int n = sizes[t];
        int blocks = (n + BLOCK - 1) / BLOCK;
        k_sum_sq<<<blocks, BLOCK, BLOCK * sizeof(float)>>>(
            grads[t], d_sum, n);
    }

    float h_sum;
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_sum);
    return sqrtf(h_sum);
}

void gpu_clip_grads(float **grads, const int *sizes, int n_tensors,
                    float max_norm) {
    float norm = gpu_grad_norm((const float **)grads, sizes, n_tensors);
    if (norm > max_norm) {
        float scale = max_norm / norm;
        for (int t = 0; t < n_tensors; t++) {
            int n = sizes[t];
            k_scale<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
                grads[t], scale, grads[t], n);
        }
    }
}

/* ── Zero upper triangle ─────────────────────────────────────────── */

__global__ void k_zero_upper(float *data, int seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * seq_len;
    if (idx >= total) return;
    int row = idx / seq_len;
    int col = idx % seq_len;
    if (col > row) data[idx] = 0.0f;
}

void gpu_zero_upper(float *data, int seq_len) {
    int n = seq_len * seq_len;
    k_zero_upper<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(data, seq_len);
}

/* ── Clamp ────────────────────────────────────────────────────────── */

__global__ void k_clamp(float *data, float minv, float maxv, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i] = fminf(fmaxf(data[i], minv), maxv);
}

void gpu_clamp(float *data, float minv, float maxv, int n) {
    k_clamp<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(data, minv, maxv, n);
}

/* ── Log-softmax (numerically stable) ─────────────────────────────
 *
 * log_softmax(x) = x - log(sum(exp(x - max(x))))
 * Never computes exp() then log() separately. No overflow. */

__global__ void k_log_softmax(const float *x, float *y, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *xi = x + row * cols;
    float *yi = y + row * cols;

    extern __shared__ float sdata[];

    /* Find max */
    float mx = -FLT_MAX;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        mx = fmaxf(mx, xi[j]);
    sdata[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = sdata[0];

    /* Sum of exp(x - max) */
    float sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        sum += expf(xi[j] - row_max);
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float log_sum = logf(sdata[0]);

    /* log_softmax = x - max - log_sum */
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        yi[j] = xi[j] - row_max - log_sum;
}

/* Cross-entropy from log-softmax: loss = -log_probs[target]
 * Backward: d_logits[i] = exp(log_probs[i]) - (i == target ? 1 : 0)
 * This is the STABLE version — no separate softmax + NLL. */

void gpu_log_softmax(const float *x, float *y, int rows, int cols) {
    int threads = cols < 256 ? cols : 256;
    k_log_softmax<<<rows, threads, threads * sizeof(float)>>>(x, y, rows, cols);
}

float gpu_cross_entropy_loss(const float *log_probs, int target, int vocab) {
    float lp;
    cudaMemcpy(&lp, log_probs + target, sizeof(float), cudaMemcpyDeviceToHost);
    return -lp;
}

/* ── Flash Attention (simplified) ─────────────────────────────────
 *
 * Fused attention: Q @ K^T → scale → causal mask → softmax → @ V
 * All in one kernel. Uses online softmax — never stores the full
 * S×S attention matrix. Numerically stable by construction.
 *
 * For simplicity, this processes one head at a time with one block
 * per query position. Not as optimized as Dao's Flash Attention
 * but correct, stable, and much better than separate kernels. */

__global__ void k_flash_attn_fwd(
    const float *Q,     /* [S, hd] */
    const float *K,     /* [S, hd] */
    const float *V,     /* [S, hd] */
    float *O,           /* [S, hd] output */
    int S, int hd, float scale,
    int causal          /* 1 = causal mask, 0 = no mask */
) {
    int row = blockIdx.x;   /* query position */
    if (row >= S) return;
    int j = threadIdx.x;    /* head dimension index */
    if (j >= hd) return;

    const float *q = Q + row * hd;

    /* Online softmax state */
    float m = -FLT_MAX;  /* running max */
    float l = 0.0f;      /* running sum of exp */
    float acc = 0.0f;    /* running output accumulator (unnormalized) */

    int max_col = causal ? (row + 1) : S;

    for (int col = 0; col < max_col; col++) {
        /* Compute attention score for this (row, col) pair */
        const float *k = K + col * hd;
        float score = 0.0f;
        for (int d = 0; d < hd; d++)
            score += q[d] * k[d];
        score *= scale;

        /* Online softmax update */
        float m_new = fmaxf(m, score);
        float exp_old = expf(m - m_new);  /* rescale old accumulator */
        float exp_new = expf(score - m_new);
        float l_new = l * exp_old + exp_new;

        /* Rescale accumulator and add new value */
        const float *v = V + col * hd;
        acc = acc * (l * exp_old / l_new) + v[j] * (exp_new / l_new);

        m = m_new;
        l = l_new;
    }

    O[row * hd + j] = acc;
}

void gpu_flash_attn_fwd(const float *Q, const float *K, const float *V,
                        float *O, int S, int hd, float scale, int causal) {
    /* One block per query position, hd threads per block */
    int threads = hd < 256 ? hd : 256;
    k_flash_attn_fwd<<<S, threads>>>(Q, K, V, O, S, hd, scale, causal);
}

/* ── Flash Attention Backward ─────────────────────────────────────
 *
 * Recomputes attention weights using online softmax (same as forward),
 * then computes gradients for Q, K, V. Never stores the full S×S
 * matrix. Numerically stable by construction. */

__global__ void k_flash_attn_bwd(
    const float *Q, const float *K, const float *V,
    const float *dO, float *dQ, float *dK, float *dV,
    int S, int hd, float scale, int causal
) {
    int row = blockIdx.x;
    int j = threadIdx.x;
    if (row >= S || j >= hd) return;

    const float *q = Q + row * hd;
    const float *do_row = dO + row * hd;
    int max_col = causal ? (row + 1) : S;

    /* Pass 1: online softmax stats + D (two values, one loop) */
    float m = -FLT_MAX;
    float l = 0.0f;
    float D = 0.0f;
    for (int col = 0; col < max_col; col++) {
        float score = 0.0f;
        for (int d = 0; d < hd; d++)
            score += q[d] * K[col * hd + d];
        score *= scale;

        /* Online softmax update */
        float m_new = fmaxf(m, score);
        float exp_old = expf(m - m_new);
        float exp_new = expf(score - m_new);
        float l_new = l * exp_old + exp_new;

        /* Online D update: D = sum(p * dO·V) */
        float dov = 0.0f;
        for (int d = 0; d < hd; d++)
            dov += do_row[d] * V[col * hd + d];
        D = D * (l * exp_old / fmaxf(l_new, 1e-10f))
          + dov * (exp_new / fmaxf(l_new, 1e-10f));

        m = m_new;
        l = l_new;
    }

    /* Pass 2: compute gradients (one loop, recompute scores) */
    float dq_acc = 0.0f;
    for (int col = 0; col < max_col; col++) {
        float score = 0.0f;
        for (int d = 0; d < hd; d++)
            score += q[d] * K[col * hd + d];
        score *= scale;
        float p = expf(score - m) / fmaxf(l, 1e-10f);

        /* dV += p * dO */
        atomicAdd(&dV[col * hd + j], p * do_row[j]);

        /* dScore = p * (dO·V - D) * scale */
        float dov = 0.0f;
        for (int d = 0; d < hd; d++)
            dov += do_row[d] * V[col * hd + d];
        float ds = p * (dov - D) * scale;

        /* dQ += ds * K, dK += ds * Q */
        dq_acc += ds * K[col * hd + j];
        atomicAdd(&dK[col * hd + j], ds * q[j]);
    }
    atomicAdd(&dQ[row * hd + j], dq_acc);
}

void gpu_flash_attn_bwd(const float *Q, const float *K, const float *V,
                        const float *dO, float *dQ, float *dK, float *dV,
                        int S, int hd, float scale, int causal) {
    int threads = hd < 256 ? hd : 256;
    k_flash_attn_bwd<<<S, threads>>>(Q, K, V, dO, dQ, dK, dV,
                                      S, hd, scale, causal);
}

/* ── Cross-entropy backward (GPU-side) ────────────────────────────
 *
 * dLogits = (exp(logProbs) - one_hot(target)) / S
 * Replaces CPU-side download + exp loop + upload.
 * Targets are int32 token IDs on GPU. */

__global__ void k_ce_backward(const float *log_probs, const int *targets,
                               float *d_logits, int S, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * V;
    if (idx >= total) return;
    int row = idx / V;
    int col = idx % V;
    float inv_s = 1.0f / (float)S;
    float grad = expf(log_probs[idx]) * inv_s;
    if (col == targets[row])
        grad -= inv_s;
    d_logits[idx] = grad;
}

void gpu_ce_backward(const float *log_probs, const int *targets,
                     float *d_logits, int S, int V) {
    int n = S * V;
    k_ce_backward<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        log_probs, targets, d_logits, S, V);
}

/* ── Mean cross-entropy loss (GPU-side) ──────────────────────────
 *
 * loss = -mean(log_probs[i, targets[i]]) for i in 0..S-1
 * Returns a single float. Avoids downloading S×V floats to CPU. */

__global__ void k_ce_loss(const float *log_probs, const int *targets,
                          float *losses, int S, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S) return;
    losses[i] = -log_probs[i * V + targets[i]];
}

float gpu_ce_loss(const float *log_probs, const int *targets,
                  float *scratch, int S, int V) {
    /* scratch must hold at least S floats */
    k_ce_loss<<<(S + BLOCK - 1) / BLOCK, BLOCK>>>(
        log_probs, targets, scratch, S, V);
    /* Download S losses and average on CPU (S is small, ~512 floats = 2KB) */
    float *h = (float *)malloc(S * sizeof(float));
    cudaMemcpy(h, scratch, S * sizeof(float), cudaMemcpyDeviceToHost);
    float sum = 0.0f;
    for (int i = 0; i < S; i++) sum += h[i];
    free(h);
    return sum / (float)S;
}

/* ── Q4_0 Quantization ────────────────────────────────────────────
 *
 * 4-bit quantization: 32 floats → 16 bytes + 1 float16 scale = 18 bytes.
 * 7x compression. Same format as GGML/llama.cpp Q4_0.
 *
 * Block layout: { float16 scale, uint8 qs[16] }
 * Each uint8 holds two 4-bit values (low nibble + high nibble).
 * Values are symmetric: -8..7, dequantized as: float = scale * (int4 - 8) */

#define QK4_0 32
#define QK4_0_BYTES (QK4_0 / 2)  /* 16 bytes of quantized data per block */

typedef struct {
    __half d;               /* scale (delta) */
    uint8_t qs[QK4_0_BYTES]; /* quantized 4-bit values, 2 per byte */
} block_q4_0;

/* Quantize float32 array to Q4_0 blocks */
__global__ void k_quantize_q4_0(const float *x, block_q4_0 *y, int n) {
    int block_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = n / QK4_0;
    if (block_id >= num_blocks) return;

    const float *src = x + block_id * QK4_0;
    block_q4_0 *dst = y + block_id;

    /* Find absolute max */
    float amax = 0.0f;
    for (int i = 0; i < QK4_0; i++) {
        float a = fabsf(src[i]);
        if (a > amax) amax = a;
    }

    float d = amax / 7.0f;  /* scale factor */
    float id = d ? 1.0f / d : 0.0f;  /* inverse scale */
    dst->d = __float2half(d);

    /* Quantize: pack two 4-bit values per byte */
    for (int i = 0; i < QK4_0_BYTES; i++) {
        int v0 = min(15, (int)(src[i*2 + 0] * id + 8.5f));
        int v1 = min(15, (int)(src[i*2 + 1] * id + 8.5f));
        dst->qs[i] = (uint8_t)(v0 | (v1 << 4));
    }
}

/* Dequantized matrix-vector multiply: y = Q4_0_matrix @ x_float
 * A is [rows, cols] in Q4_0 format (cols must be multiple of 32)
 * x is [cols] float32
 * y is [rows] float32
 *
 * Optimized: cache x in shared memory, warp-level reduction,
 * process multiple Q4_0 blocks per thread for better ILP. */
__global__ void k_matvec_q4_0(const block_q4_0 *A, const float *x, float *y,
                               int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int blocks_per_row = cols / QK4_0;
    const block_q4_0 *row_blocks = A + row * blocks_per_row;

    /* Cache input vector in shared memory — read once, use by all threads */
    extern __shared__ float sx[];
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        sx[i] = x[i];
    __syncthreads();

    /* Each thread accumulates over multiple Q4_0 blocks */
    float sum = 0.0f;
    for (int b = threadIdx.x; b < blocks_per_row; b += blockDim.x) {
        float d = __half2float(row_blocks[b].d);
        const uint8_t *qs = row_blocks[b].qs;
        int base = b * QK4_0;

        /* Unroll: process all 16 bytes (32 values) per block */
        float local_sum = 0.0f;
        for (int i = 0; i < QK4_0_BYTES; i++) {
            uint8_t q = qs[i];
            local_sum += (float)((q & 0x0F) - 8) * sx[base + i*2 + 0];
            local_sum += (float)((q >> 4)   - 8) * sx[base + i*2 + 1];
        }
        sum += d * local_sum;
    }

    /* Warp-level reduction (no shared memory needed, no __syncthreads) */
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    /* First thread of each warp writes partial sum */
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    /* Use shared memory only for cross-warp reduction */
    if (lane_id == 0) sx[warp_id] = sum;
    __syncthreads();

    /* First warp reduces across warps */
    if (warp_id == 0) {
        int num_warps = blockDim.x / 32;
        sum = (lane_id < num_warps) ? sx[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane_id == 0) y[row] = sum;
    }
}

void gpu_quantize_q4_0(const float *x, void *y, int n) {
    int num_blocks = n / QK4_0;
    k_quantize_q4_0<<<(num_blocks + BLOCK - 1) / BLOCK, BLOCK>>>(
        x, (block_q4_0 *)y, n);
}

void gpu_matvec_q4_0(const void *A, const float *x, float *y,
                      int rows, int cols) {
    /* 256 threads = 8 warps. Shared memory holds input vector (cols floats). */
    int threads = 256;
    int smem = cols * sizeof(float);  /* cache input vector */
    k_matvec_q4_0<<<rows, threads, smem>>>(
        (const block_q4_0 *)A, x, y, rows, cols);
}

} /* extern "C" */
