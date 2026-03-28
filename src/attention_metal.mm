#include "lightning_core/core/detail/attention_backend.hpp"

#if defined(CJ_HAS_METAL) && CJ_HAS_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace lightning_core::detail {

namespace {

constexpr uint32_t kMaxAsyncInflight = 64;
constexpr double kTrainVec4AdoptRatio = 0.95;
constexpr uint32_t kTuneCacheFlushEvery = 8;

constexpr int kTrainPlanFusedScalar = 0;
constexpr int kTrainPlanFusedVec4T32 = 1;
constexpr int kTrainPlanFusedVec4T16 = 2;
constexpr int kTrainPlanPreScalar = 3;
constexpr int kTrainPlanPreVec4T32 = 4;
constexpr int kTrainPlanPreVec4T16 = 5;

struct ForwardTuneKey {
  std::size_t seq = 0;
  std::size_t dim = 0;
  uint8_t causal = 0;
  uint8_t resident_hint = 0;
  uint8_t sync_hint = 0;

  bool operator==(const ForwardTuneKey& other) const {
    return seq == other.seq && dim == other.dim && causal == other.causal &&
           resident_hint == other.resident_hint && sync_hint == other.sync_hint;
  }
};

struct TrainTuneKey {
  std::size_t seq = 0;
  std::size_t dim = 0;
  uint8_t causal = 0;
  uint8_t resident_hint = 0;
  uint8_t sync_hint = 0;
  uint8_t compute_loss = 0;

  bool operator==(const TrainTuneKey& other) const {
    return seq == other.seq && dim == other.dim && causal == other.causal &&
           resident_hint == other.resident_hint && sync_hint == other.sync_hint &&
           compute_loss == other.compute_loss;
  }
};

struct ForwardTuneKeyHash {
  std::size_t operator()(const ForwardTuneKey& k) const {
    std::size_t h = k.seq;
    h ^= (k.dim + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U));
    h ^= (static_cast<std::size_t>(k.causal) << 1U);
    h ^= (static_cast<std::size_t>(k.resident_hint) << 2U);
    h ^= (static_cast<std::size_t>(k.sync_hint) << 3U);
    return h;
  }
};

struct TrainTuneKeyHash {
  std::size_t operator()(const TrainTuneKey& k) const {
    std::size_t h = k.seq;
    h ^= (k.dim + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U));
    h ^= (static_cast<std::size_t>(k.causal) << 1U);
    h ^= (static_cast<std::size_t>(k.resident_hint) << 2U);
    h ^= (static_cast<std::size_t>(k.sync_hint) << 3U);
    h ^= (static_cast<std::size_t>(k.compute_loss) << 4U);
    return h;
  }
};

struct MetalAttentionContext {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> pipe_forward_online = nil;
  id<MTLComputePipelineState> pipe_scores = nil;
  id<MTLComputePipelineState> pipe_softmax = nil;
  id<MTLComputePipelineState> pipe_softmax_out_fused = nil;
  id<MTLComputePipelineState> pipe_out = nil;
  id<MTLComputePipelineState> pipe_out_postprocess = nil;
  id<MTLComputePipelineState> pipe_grad_out = nil;
  id<MTLComputePipelineState> pipe_grad_out_only = nil;
  id<MTLComputePipelineState> pipe_loss_reduce_partials = nil;
  id<MTLComputePipelineState> pipe_loss_reduce_final = nil;
  id<MTLComputePipelineState> pipe_grad_v = nil;
  id<MTLComputePipelineState> pipe_grad_v_sgd = nil;
  id<MTLComputePipelineState> pipe_grad_v_sgd_seq512 = nil;
  id<MTLComputePipelineState> pipe_grad_v_sgd_vec4_tiled = nil;
  id<MTLComputePipelineState> pipe_grad_v_sgd_vec4_tiled16 = nil;
  id<MTLComputePipelineState> pipe_grad_v_sgd_from_out_target = nil;
  id<MTLComputePipelineState> pipe_grad_v_sgd_from_out_target_vec4_tiled = nil;
  id<MTLComputePipelineState> pipe_grad_v_sgd_from_out_target_vec4_tiled16 = nil;
  id<MTLComputePipelineState> pipe_sgd = nil;

  id<MTLBuffer> buf_q = nil;
  id<MTLBuffer> buf_k = nil;
  id<MTLBuffer> buf_v = nil;
  id<MTLBuffer> buf_target = nil;
  id<MTLBuffer> buf_out = nil;
  id<MTLBuffer> buf_scores = nil;
  id<MTLBuffer> buf_grad_out = nil;
  id<MTLBuffer> buf_grad_v = nil;
  id<MTLBuffer> buf_loss = nil;
  id<MTLBuffer> buf_loss_partials = nil;
  id<MTLBuffer> buf_loss_scalar = nil;

  std::size_t sd_capacity = 0;
  std::size_t ss_capacity = 0;
  std::size_t loss_partials_capacity = 0;
  bool probs_cache_valid = false;
  std::size_t probs_cache_seq = 0;
  std::size_t probs_cache_dim = 0;
  bool probs_cache_causal = false;
  std::unordered_map<ForwardTuneKey, int, ForwardTuneKeyHash> forward_mode_cache;
  std::unordered_map<TrainTuneKey, int, TrainTuneKeyHash> train_mode_cache;
  std::unordered_map<uint64_t, uint32_t> one_d_tg_cache;
  std::unordered_map<uint32_t, uint32_t> softmax_tg_cache;
  std::unordered_map<uint32_t, uint32_t> row_fused_tg_cache;
  std::unordered_map<uint64_t, uint32_t> grad_vec4_tg_cache;
  bool tune_cache_loaded = false;
  bool tune_cache_dirty = false;
  uint32_t tune_cache_pending_updates = 0;
  std::string tune_cache_path;
  uint32_t async_inflight = 0;
};

MetalAttentionContext& getContext();
std::mutex& getContextMutex();

std::size_t growCapacity(std::size_t current, std::size_t required) {
  if (required == 0) {
    return current;
  }
  std::size_t next = (current == 0) ? required : current;
  while (next < required) {
    std::size_t grown = next + (next / 2);
    if (grown <= next) {
      next = required;
      break;
    }
    next = grown;
  }
  return next;
}

runtime::Status waitForAttentionQueueIdle(MetalAttentionContext& ctx) {
  if (ctx.async_inflight == 0) {
    return runtime::Status::kSuccess;
  }

  id<MTLCommandBuffer> fence = [ctx.queue commandBuffer];
  if (!fence) {
    return runtime::Status::kDriverError;
  }
  [fence commit];
  [fence waitUntilCompleted];
  if (fence.status != MTLCommandBufferStatusCompleted) {
    return runtime::Status::kUnknown;
  }
  ctx.async_inflight = 0;
  return runtime::Status::kSuccess;
}

std::string resolveTuneCachePath() {
  const char* env = std::getenv("CJ_ATTN_TUNE_CACHE_FILE");
  if (env != nullptr && env[0] != '\0') {
    return std::string(env);
  }
  return std::string(".cudajun_attn_tune_cache.csv");
}

void saveTuneCacheIfDirty(MetalAttentionContext& ctx) {
  if (!ctx.tune_cache_loaded || !ctx.tune_cache_dirty) {
    return;
  }
  if (ctx.tune_cache_path.empty()) {
    return;
  }

  std::ofstream ofs(ctx.tune_cache_path, std::ios::out | std::ios::trunc);
  if (!ofs.is_open()) {
    return;
  }

  for (const auto& kv : ctx.forward_mode_cache) {
    const ForwardTuneKey& k = kv.first;
    ofs << "F," << k.seq << "," << k.dim << "," << static_cast<unsigned>(k.causal) << ","
        << static_cast<unsigned>(k.resident_hint) << "," << static_cast<unsigned>(k.sync_hint) << ","
        << kv.second << "\n";
  }
  for (const auto& kv : ctx.train_mode_cache) {
    const TrainTuneKey& k = kv.first;
    ofs << "T," << k.seq << "," << k.dim << "," << static_cast<unsigned>(k.causal) << ","
        << static_cast<unsigned>(k.resident_hint) << "," << static_cast<unsigned>(k.sync_hint) << ","
        << static_cast<unsigned>(k.compute_loss) << "," << kv.second << "\n";
  }

  if (ofs.good()) {
    ctx.tune_cache_dirty = false;
    ctx.tune_cache_pending_updates = 0;
  }
}

void markTuneCacheUpdated(MetalAttentionContext& ctx) {
  ctx.tune_cache_dirty = true;
  ctx.tune_cache_pending_updates += 1;
  if (ctx.tune_cache_pending_updates >= kTuneCacheFlushEvery) {
    saveTuneCacheIfDirty(ctx);
  }
}

void flushAttentionTuneCacheAtExit() {
  std::lock_guard<std::mutex> lock(getContextMutex());
  MetalAttentionContext& ctx = getContext();
  saveTuneCacheIfDirty(ctx);
}

void loadTuneCacheIfNeeded(MetalAttentionContext& ctx) {
  static bool exit_hook_registered = false;
  if (!exit_hook_registered) {
    std::atexit(flushAttentionTuneCacheAtExit);
    exit_hook_registered = true;
  }

  if (ctx.tune_cache_loaded) {
    return;
  }

  ctx.tune_cache_path = resolveTuneCachePath();
  ctx.tune_cache_loaded = true;

  std::ifstream ifs(ctx.tune_cache_path);
  if (!ifs.is_open()) {
    return;
  }

  std::string line;
  while (std::getline(ifs, line)) {
    if (line.empty()) {
      continue;
    }

    std::stringstream ss(line);
    std::string tok;
    std::vector<std::string> cols;
    while (std::getline(ss, tok, ',')) {
      cols.push_back(tok);
    }

    try {
      if (cols.size() == 7 && cols[0] == "F") {
        ForwardTuneKey k;
        k.seq = static_cast<std::size_t>(std::stoull(cols[1]));
        k.dim = static_cast<std::size_t>(std::stoull(cols[2]));
        k.causal = static_cast<uint8_t>(std::stoi(cols[3]));
        k.resident_hint = static_cast<uint8_t>(std::stoi(cols[4]));
        k.sync_hint = static_cast<uint8_t>(std::stoi(cols[5]));
        int mode = std::stoi(cols[6]);
        ctx.forward_mode_cache[k] = mode;
      } else if (cols.size() == 8 && cols[0] == "T") {
        TrainTuneKey k;
        k.seq = static_cast<std::size_t>(std::stoull(cols[1]));
        k.dim = static_cast<std::size_t>(std::stoull(cols[2]));
        k.causal = static_cast<uint8_t>(std::stoi(cols[3]));
        k.resident_hint = static_cast<uint8_t>(std::stoi(cols[4]));
        k.sync_hint = static_cast<uint8_t>(std::stoi(cols[5]));
        k.compute_loss = static_cast<uint8_t>(std::stoi(cols[6]));
        int mode = std::stoi(cols[7]);
        ctx.train_mode_cache[k] = mode;
      }
    } catch (...) {
      // Ignore malformed cache lines.
    }
  }
}

const char* kAttentionMetalShader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void attn_forward_online(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& seq [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant uint& causal [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
  uint c = gid.x;
  uint i = gid.y;
  if (i >= seq || c >= dim) {
    return;
  }

  float m = -1.0e30f;
  float l = 0.0f;
  float acc = 0.0f;

  uint qi = i * dim;
  uint j_end = (causal != 0) ? min(seq, i + 1) : seq;
  for (uint j = 0; j < j_end; ++j) {

    uint kj = j * dim;
    float score = 0.0f;
    uint d4 = dim & ~3u;
    for (uint x = 0; x < d4; x += 4) {
      float4 qv = float4(q[qi + x + 0], q[qi + x + 1], q[qi + x + 2], q[qi + x + 3]);
      float4 kv = float4(k[kj + x + 0], k[kj + x + 1], k[kj + x + 2], k[kj + x + 3]);
      score += dot(qv, kv);
    }
    for (uint x = d4; x < dim; ++x) {
      score += q[qi + x] * k[kj + x];
    }
    score *= scale;

    float m_new = max(m, score);
    float alpha = exp(m - m_new);
    float beta = exp(score - m_new);
    float l_new = l * alpha + beta;

    acc = acc * alpha + beta * v[j * dim + c];

    m = m_new;
    l = l_new;
  }

  out[i * dim + c] = acc / l;
}

kernel void attn_scores(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& seq [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    constant uint& causal [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
  uint j = gid.x;
  uint i = gid.y;
  if (i >= seq || j >= seq) {
    return;
  }

  if (causal != 0 && j > i) {
    scores[i * seq + j] = -1.0e30f;
    return;
  }

  float acc = 0.0f;
  uint qi = i * dim;
  uint kj = j * dim;
  uint c = 0;
  for (; c + 3 < dim; c += 4) {
    acc += q[qi + c + 0] * k[kj + c + 0];
    acc += q[qi + c + 1] * k[kj + c + 1];
    acc += q[qi + c + 2] * k[kj + c + 2];
    acc += q[qi + c + 3] * k[kj + c + 3];
  }
  for (; c < dim; ++c) {
    acc += q[qi + c] * k[kj + c];
  }
  scores[i * seq + j] = acc * scale;
}

kernel void attn_softmax_rows(
    device float* scores [[buffer(0)]],
    constant uint& seq [[buffer(1)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
  if (row >= seq) {
    return;
  }

  threadgroup float tg_max[128];
  threadgroup float tg_sum[128];

  uint base = row * seq;
  float local_max = -1.0e30f;
  for (uint j = tid; j < seq; j += tg_size) {
    local_max = max(local_max, scores[base + j]);
  }

  tg_max[tid] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      tg_max[tid] = max(tg_max[tid], tg_max[tid + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float m = tg_max[0];

  float local_sum = 0.0f;
  for (uint j = tid; j < seq; j += tg_size) {
    float e = exp(scores[base + j] - m);
    scores[base + j] = e;
    local_sum += e;
  }

  tg_sum[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      tg_sum[tid] += tg_sum[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float sum = tg_sum[0];

  float inv = 1.0f / sum;
  for (uint j = tid; j < seq; j += tg_size) {
    scores[base + j] *= inv;
  }
}

kernel void attn_softmax_out_fused(
    device const float* scores [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& seq [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
  if (row >= seq) {
    return;
  }

  threadgroup float tg_max[128];
  threadgroup float tg_sum[128];

  uint base = row * seq;
  float local_max = -1.0e30f;
  for (uint j = tid; j < seq; j += tg_size) {
    local_max = max(local_max, scores[base + j]);
  }
  tg_max[tid] = local_max;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      tg_max[tid] = max(tg_max[tid], tg_max[tid + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float row_max = tg_max[0];

  float local_sum = 0.0f;
  for (uint j = tid; j < seq; j += tg_size) {
    local_sum += exp(scores[base + j] - row_max);
  }
  tg_sum[tid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      tg_sum[tid] += tg_sum[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  float inv = 1.0f / tg_sum[0];
  for (uint c = tid; c < dim; c += tg_size) {
    float acc = 0.0f;
    for (uint j = 0; j < seq; ++j) {
      float p = exp(scores[base + j] - row_max) * inv;
      acc += p * v[j * dim + c];
    }
    out[row * dim + c] = acc;
  }
}

kernel void attn_out(
    device const float* probs [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& seq [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  uint c = gid.x;
  uint i = gid.y;
  if (i >= seq || c >= dim) {
    return;
  }

  float acc = 0.0f;
  uint pbase = i * seq;
  for (uint j = 0; j < seq; ++j) {
    acc += probs[pbase + j] * v[j * dim + c];
  }
  out[i * dim + c] = acc;
}

kernel void attn_out_postprocess(
    device float* out [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant float& bias [[buffer(3)]],
    constant uint& relu [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= n) {
    return;
  }
  float x = out[gid] * scale + bias;
  if (relu != 0 && x < 0.0f) {
    x = 0.0f;
  }
  out[gid] = x;
}

kernel void attn_grad_out_loss(
    device const float* out [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* grad_out [[buffer(2)]],
    device float* loss_vec [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant float& norm [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= n) {
    return;
  }
  float diff = out[gid] - target[gid];
  grad_out[gid] = 2.0f * diff * norm;
  loss_vec[gid] = diff * diff;
}

kernel void attn_grad_out_only(
    device const float* out [[buffer(0)]],
    device const float* target [[buffer(1)]],
    device float* grad_out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant float& norm [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= n) {
    return;
  }
  float diff = out[gid] - target[gid];
  grad_out[gid] = 2.0f * diff * norm;
}

kernel void attn_grad_v(
    device const float* probs [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* grad_v [[buffer(2)]],
    constant uint& seq [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    constant uint& causal [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  uint c = gid.x;
  uint j = gid.y;
  if (j >= seq || c >= dim) {
    return;
  }

  float acc = 0.0f;
  uint i_begin = causal != 0 ? j : 0;
  for (uint i = i_begin; i < seq; ++i) {
    acc += probs[i * seq + j] * grad_out[i * dim + c];
  }
  grad_v[j * dim + c] = acc;
}

kernel void attn_grad_v_sgd(
    device const float* probs [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* v [[buffer(2)]],
    constant uint& seq [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    constant float& lr [[buffer(5)]],
    constant uint& causal [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
  uint c = gid.x;
  uint j = gid.y;
  if (j >= seq || c >= dim) {
    return;
  }

  float acc = 0.0f;
  uint i_begin = causal != 0 ? j : 0;
  for (uint i = i_begin; i < seq; ++i) {
    acc += probs[i * seq + j] * grad_out[i * dim + c];
  }

  uint idx = j * dim + c;
  v[idx] -= lr * acc;
}

kernel void attn_grad_v_sgd_seq512_unroll8(
    device const float* probs [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* v [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant float& lr [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  uint c = gid.x;
  uint j = gid.y;
  if (j >= 512 || c >= dim) {
    return;
  }

  float acc = 0.0f;
  uint i = j;
  for (; i + 7 < 512; i += 8) {
    acc += probs[(i + 0) * 512 + j] * grad_out[(i + 0) * dim + c];
    acc += probs[(i + 1) * 512 + j] * grad_out[(i + 1) * dim + c];
    acc += probs[(i + 2) * 512 + j] * grad_out[(i + 2) * dim + c];
    acc += probs[(i + 3) * 512 + j] * grad_out[(i + 3) * dim + c];
    acc += probs[(i + 4) * 512 + j] * grad_out[(i + 4) * dim + c];
    acc += probs[(i + 5) * 512 + j] * grad_out[(i + 5) * dim + c];
    acc += probs[(i + 6) * 512 + j] * grad_out[(i + 6) * dim + c];
    acc += probs[(i + 7) * 512 + j] * grad_out[(i + 7) * dim + c];
  }
  for (; i < 512; ++i) {
    acc += probs[i * 512 + j] * grad_out[i * dim + c];
  }

  uint idx = j * dim + c;
  v[idx] -= lr * acc;
}

kernel void attn_grad_v_sgd_vec4_tiled(
    device const float* probs [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* v [[buffer(2)]],
    constant uint& seq [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    constant float& lr [[buffer(5)]],
    constant uint& causal [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint TILE_I = 32;
  threadgroup float probs_tile[TILE_I];

  uint c4 = gid.x;
  uint j = gid.y;
  uint c4n = dim >> 2;
  if (j >= seq || c4 >= c4n) {
    return;
  }

  float4 acc = float4(0.0f);
  uint i_begin = causal != 0 ? j : 0;
  for (uint ib = i_begin; ib < seq; ib += TILE_I) {
    uint i = ib + tid;
    probs_tile[tid] = (i < seq) ? probs[i * seq + j] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint valid = min(TILE_I, seq - ib);
    for (uint t = 0; t < valid; ++t) {
      uint ii = ib + t;
      uint base = ii * dim + c4 * 4;
      float4 go = float4(
          grad_out[base + 0],
          grad_out[base + 1],
          grad_out[base + 2],
          grad_out[base + 3]);
      acc += probs_tile[t] * go;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  uint v_base = j * dim + c4 * 4;
  v[v_base + 0] -= lr * acc.x;
  v[v_base + 1] -= lr * acc.y;
  v[v_base + 2] -= lr * acc.z;
  v[v_base + 3] -= lr * acc.w;
}

kernel void attn_grad_v_sgd_vec4_tiled16(
    device const float* probs [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* v [[buffer(2)]],
    constant uint& seq [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    constant float& lr [[buffer(5)]],
    constant uint& causal [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint TILE_I = 16;
  threadgroup float probs_tile[TILE_I];

  uint c4 = gid.x;
  uint j = gid.y;
  uint c4n = dim >> 2;
  if (j >= seq || c4 >= c4n) {
    return;
  }

  float4 acc = float4(0.0f);
  uint i_begin = causal != 0 ? j : 0;
  for (uint ib = i_begin; ib < seq; ib += TILE_I) {
    uint i = ib + tid;
    probs_tile[tid] = (i < seq) ? probs[i * seq + j] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint valid = min(TILE_I, seq - ib);
    for (uint t = 0; t < valid; ++t) {
      uint ii = ib + t;
      uint base = ii * dim + c4 * 4;
      float4 go = float4(
          grad_out[base + 0],
          grad_out[base + 1],
          grad_out[base + 2],
          grad_out[base + 3]);
      acc += probs_tile[t] * go;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  uint v_base = j * dim + c4 * 4;
  v[v_base + 0] -= lr * acc.x;
  v[v_base + 1] -= lr * acc.y;
  v[v_base + 2] -= lr * acc.z;
  v[v_base + 3] -= lr * acc.w;
}

kernel void attn_grad_v_sgd_from_out_target(
    device const float* probs [[buffer(0)]],
    device const float* out [[buffer(1)]],
    device const float* target [[buffer(2)]],
    device float* v [[buffer(3)]],
    constant uint& seq [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    constant float& lr [[buffer(6)]],
    constant uint& causal [[buffer(7)]],
    constant float& norm [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]) {
  uint c = gid.x;
  uint j = gid.y;
  if (j >= seq || c >= dim) {
    return;
  }

  float acc = 0.0f;
  uint i_begin = causal != 0 ? j : 0;
  for (uint i = i_begin; i < seq; ++i) {
    uint idx = i * dim + c;
    float grad_out = 2.0f * (out[idx] - target[idx]) * norm;
    acc += probs[i * seq + j] * grad_out;
  }

  uint v_idx = j * dim + c;
  v[v_idx] -= lr * acc;
}

kernel void attn_grad_v_sgd_from_out_target_vec4_tiled(
    device const float* probs [[buffer(0)]],
    device const float* out [[buffer(1)]],
    device const float* target [[buffer(2)]],
    device float* v [[buffer(3)]],
    constant uint& seq [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    constant float& lr [[buffer(6)]],
    constant uint& causal [[buffer(7)]],
    constant float& norm [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint TILE_I = 32;
  threadgroup float probs_tile[TILE_I];

  uint c4 = gid.x;
  uint j = gid.y;
  uint c4n = dim >> 2;
  if (j >= seq || c4 >= c4n) {
    return;
  }

  float4 acc = float4(0.0f);
  uint i_begin = causal != 0 ? j : 0;
  for (uint ib = i_begin; ib < seq; ib += TILE_I) {
    uint i = ib + tid;
    probs_tile[tid] = (i < seq) ? probs[i * seq + j] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint valid = min(TILE_I, seq - ib);
    for (uint t = 0; t < valid; ++t) {
      uint ii = ib + t;
      uint base = ii * dim + c4 * 4;
      float4 diff = float4(
          out[base + 0] - target[base + 0],
          out[base + 1] - target[base + 1],
          out[base + 2] - target[base + 2],
          out[base + 3] - target[base + 3]);
      float4 go = 2.0f * norm * diff;
      acc += probs_tile[t] * go;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  uint v_base = j * dim + c4 * 4;
  v[v_base + 0] -= lr * acc.x;
  v[v_base + 1] -= lr * acc.y;
  v[v_base + 2] -= lr * acc.z;
  v[v_base + 3] -= lr * acc.w;
}

kernel void attn_grad_v_sgd_from_out_target_vec4_tiled16(
    device const float* probs [[buffer(0)]],
    device const float* out [[buffer(1)]],
    device const float* target [[buffer(2)]],
    device float* v [[buffer(3)]],
    constant uint& seq [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    constant float& lr [[buffer(6)]],
    constant uint& causal [[buffer(7)]],
    constant float& norm [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint TILE_I = 16;
  threadgroup float probs_tile[TILE_I];

  uint c4 = gid.x;
  uint j = gid.y;
  uint c4n = dim >> 2;
  if (j >= seq || c4 >= c4n) {
    return;
  }

  float4 acc = float4(0.0f);
  uint i_begin = causal != 0 ? j : 0;
  for (uint ib = i_begin; ib < seq; ib += TILE_I) {
    uint i = ib + tid;
    probs_tile[tid] = (i < seq) ? probs[i * seq + j] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint valid = min(TILE_I, seq - ib);
    for (uint t = 0; t < valid; ++t) {
      uint ii = ib + t;
      uint base = ii * dim + c4 * 4;
      float4 diff = float4(
          out[base + 0] - target[base + 0],
          out[base + 1] - target[base + 1],
          out[base + 2] - target[base + 2],
          out[base + 3] - target[base + 3]);
      float4 go = 2.0f * norm * diff;
      acc += probs_tile[t] * go;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  uint v_base = j * dim + c4 * 4;
  v[v_base + 0] -= lr * acc.x;
  v[v_base + 1] -= lr * acc.y;
  v[v_base + 2] -= lr * acc.z;
  v[v_base + 3] -= lr * acc.w;
}

kernel void attn_sgd(
    device float* v [[buffer(0)]],
    device const float* grad_v [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant float& lr [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= n) {
    return;
  }
  v[gid] -= lr * grad_v[gid];
}

kernel void attn_reduce_loss_partials(
    device const float* loss_vec [[buffer(0)]],
    device float* partials [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_id [[threadgroup_position_in_grid]]) {
  threadgroup float tg_sum[256];

  uint idx = tg_id * 256 + tid;
  float local = idx < n ? loss_vec[idx] : 0.0f;
  tg_sum[tid] = local;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      tg_sum[tid] += tg_sum[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0) {
    partials[tg_id] = tg_sum[0];
  }
}

kernel void attn_reduce_loss_final(
    device const float* partials [[buffer(0)]],
    device float* loss_scalar [[buffer(1)]],
    constant uint& partial_count [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float tg_sum[256];

  float local = 0.0f;
  for (uint i = tid; i < partial_count; i += 256) {
    local += partials[i];
  }
  tg_sum[tid] = local;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      tg_sum[tid] += tg_sum[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0) {
    loss_scalar[0] = tg_sum[0];
  }
}
)";

MetalAttentionContext& getContext() {
  static MetalAttentionContext ctx;
  return ctx;
}

std::mutex& getContextMutex() {
  static std::mutex mu;
  return mu;
}

id<MTLDevice> getDevice() {
  static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  return device;
}

id<MTLComputePipelineState> makePipeline(id<MTLDevice> device, id<MTLLibrary> lib, NSString* fn_name) {
  id<MTLFunction> fn = [lib newFunctionWithName:fn_name];
  if (!fn) {
    return nil;
  }
  NSError* err = nil;
  return [device newComputePipelineStateWithFunction:fn error:&err];
}

runtime::Status ensurePipelines() {
  MetalAttentionContext& ctx = getContext();
  if (ctx.device == nil) {
    ctx.device = getDevice();
    if (ctx.device == nil) {
      return runtime::Status::kNotSupported;
    }
  }

  if (ctx.queue == nil) {
    ctx.queue = [ctx.device newCommandQueue];
    if (ctx.queue == nil) {
      return runtime::Status::kDriverError;
    }
  }

  if (ctx.pipe_forward_online != nil && ctx.pipe_scores != nil && ctx.pipe_softmax != nil &&
      ctx.pipe_softmax_out_fused != nil && ctx.pipe_out != nil && ctx.pipe_out_postprocess != nil && ctx.pipe_grad_out != nil &&
      ctx.pipe_grad_out_only != nil &&
      ctx.pipe_loss_reduce_partials != nil && ctx.pipe_loss_reduce_final != nil &&
      ctx.pipe_grad_v != nil && ctx.pipe_grad_v_sgd != nil &&
      ctx.pipe_grad_v_sgd_vec4_tiled != nil &&
      ctx.pipe_grad_v_sgd_vec4_tiled16 != nil &&
      ctx.pipe_grad_v_sgd_from_out_target != nil &&
      ctx.pipe_grad_v_sgd_from_out_target_vec4_tiled != nil &&
      ctx.pipe_grad_v_sgd_from_out_target_vec4_tiled16 != nil &&
      ctx.pipe_sgd != nil) {
    loadTuneCacheIfNeeded(ctx);
    return runtime::Status::kSuccess;
  }

  NSString* source = [NSString stringWithUTF8String:kAttentionMetalShader];
  NSError* err = nil;
  id<MTLLibrary> lib = [ctx.device newLibraryWithSource:source options:nil error:&err];
  if (!lib) {
    return runtime::Status::kDriverError;
  }

  ctx.pipe_forward_online = makePipeline(ctx.device, lib, @"attn_forward_online");
  ctx.pipe_scores = makePipeline(ctx.device, lib, @"attn_scores");
  ctx.pipe_softmax = makePipeline(ctx.device, lib, @"attn_softmax_rows");
  ctx.pipe_softmax_out_fused = makePipeline(ctx.device, lib, @"attn_softmax_out_fused");
  ctx.pipe_out = makePipeline(ctx.device, lib, @"attn_out");
  ctx.pipe_out_postprocess = makePipeline(ctx.device, lib, @"attn_out_postprocess");
  ctx.pipe_grad_out = makePipeline(ctx.device, lib, @"attn_grad_out_loss");
  ctx.pipe_grad_out_only = makePipeline(ctx.device, lib, @"attn_grad_out_only");
  ctx.pipe_loss_reduce_partials = makePipeline(ctx.device, lib, @"attn_reduce_loss_partials");
  ctx.pipe_loss_reduce_final = makePipeline(ctx.device, lib, @"attn_reduce_loss_final");
  ctx.pipe_grad_v = makePipeline(ctx.device, lib, @"attn_grad_v");
  ctx.pipe_grad_v_sgd = makePipeline(ctx.device, lib, @"attn_grad_v_sgd");
  ctx.pipe_grad_v_sgd_seq512 = makePipeline(ctx.device, lib, @"attn_grad_v_sgd_seq512_unroll8");
  ctx.pipe_grad_v_sgd_vec4_tiled = makePipeline(ctx.device, lib, @"attn_grad_v_sgd_vec4_tiled");
    ctx.pipe_grad_v_sgd_vec4_tiled16 = makePipeline(ctx.device, lib, @"attn_grad_v_sgd_vec4_tiled16");
  ctx.pipe_grad_v_sgd_from_out_target = makePipeline(ctx.device, lib, @"attn_grad_v_sgd_from_out_target");
  ctx.pipe_grad_v_sgd_from_out_target_vec4_tiled =
      makePipeline(ctx.device, lib, @"attn_grad_v_sgd_from_out_target_vec4_tiled");
    ctx.pipe_grad_v_sgd_from_out_target_vec4_tiled16 =
      makePipeline(ctx.device, lib, @"attn_grad_v_sgd_from_out_target_vec4_tiled16");
  ctx.pipe_sgd = makePipeline(ctx.device, lib, @"attn_sgd");

  if (ctx.pipe_forward_online == nil || ctx.pipe_scores == nil || ctx.pipe_softmax == nil ||
      ctx.pipe_softmax_out_fused == nil || ctx.pipe_out == nil || ctx.pipe_out_postprocess == nil || ctx.pipe_grad_out == nil ||
      ctx.pipe_grad_out_only == nil ||
      ctx.pipe_loss_reduce_partials == nil || ctx.pipe_loss_reduce_final == nil ||
      ctx.pipe_grad_v == nil || ctx.pipe_grad_v_sgd == nil ||
      ctx.pipe_grad_v_sgd_vec4_tiled == nil ||
      ctx.pipe_grad_v_sgd_vec4_tiled16 == nil ||
      ctx.pipe_grad_v_sgd_from_out_target == nil ||
      ctx.pipe_grad_v_sgd_from_out_target_vec4_tiled == nil ||
      ctx.pipe_grad_v_sgd_from_out_target_vec4_tiled16 == nil ||
      ctx.pipe_sgd == nil) {
    return runtime::Status::kDriverError;
  }

  loadTuneCacheIfNeeded(ctx);

  return runtime::Status::kSuccess;
}

runtime::Status ensureBuffers(const AttentionConfig& cfg) {
  MetalAttentionContext& ctx = getContext();
  const std::size_t sd = cfg.seq_len * cfg.head_dim;
  const std::size_t ss = cfg.seq_len * cfg.seq_len;

  if (ctx.sd_capacity < sd) {
    const std::size_t sd_cap = growCapacity(ctx.sd_capacity, sd);
    const std::size_t sd_bytes = sd_cap * sizeof(float);
    ctx.buf_q = [ctx.device newBufferWithLength:sd_bytes options:MTLResourceStorageModeShared];
    ctx.buf_k = [ctx.device newBufferWithLength:sd_bytes options:MTLResourceStorageModeShared];
    ctx.buf_v = [ctx.device newBufferWithLength:sd_bytes options:MTLResourceStorageModeShared];
    ctx.buf_target = [ctx.device newBufferWithLength:sd_bytes options:MTLResourceStorageModeShared];
    ctx.buf_out = [ctx.device newBufferWithLength:sd_bytes options:MTLResourceStorageModeShared];
    ctx.buf_grad_out = [ctx.device newBufferWithLength:sd_bytes options:MTLResourceStorageModeShared];
    ctx.buf_grad_v = [ctx.device newBufferWithLength:sd_bytes options:MTLResourceStorageModeShared];
    ctx.buf_loss = [ctx.device newBufferWithLength:sd_bytes options:MTLResourceStorageModeShared];
    if (!ctx.buf_q || !ctx.buf_k || !ctx.buf_v || !ctx.buf_target || !ctx.buf_out ||
        !ctx.buf_grad_out || !ctx.buf_grad_v || !ctx.buf_loss) {
      return runtime::Status::kOutOfMemory;
    }
    ctx.sd_capacity = sd_cap;
  }

  if (ctx.ss_capacity < ss) {
    const std::size_t ss_cap = growCapacity(ctx.ss_capacity, ss);
    const std::size_t ss_bytes = ss_cap * sizeof(float);
    ctx.buf_scores = [ctx.device newBufferWithLength:ss_bytes options:MTLResourceStorageModeShared];
    if (!ctx.buf_scores) {
      return runtime::Status::kOutOfMemory;
    }
    ctx.ss_capacity = ss_cap;
    ctx.probs_cache_valid = false;
  }

  const std::size_t partial_count = (sd + 255) / 256;
  if (ctx.loss_partials_capacity < partial_count) {
    const std::size_t partial_cap = growCapacity(ctx.loss_partials_capacity, partial_count);
    ctx.buf_loss_partials = [ctx.device newBufferWithLength:partial_cap * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    if (!ctx.buf_loss_partials) {
      return runtime::Status::kOutOfMemory;
    }
    ctx.loss_partials_capacity = partial_cap;
  }

  if (ctx.buf_loss_scalar == nil) {
    ctx.buf_loss_scalar = [ctx.device newBufferWithLength:sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    if (!ctx.buf_loss_scalar) {
      return runtime::Status::kOutOfMemory;
    }
  }

  return runtime::Status::kSuccess;
}

NSUInteger autoTune1DThreadgroup(
    MetalAttentionContext& ctx,
    id<MTLComputePipelineState> pipe,
    std::size_t n) {
  const uint32_t n_bucket = (n <= 32) ? 32u :
                            (n <= 64) ? 64u :
                            (n <= 128) ? 128u :
                            (n <= 256) ? 256u : 512u;
  const uint64_t key = (static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pipe)) << 8U) |
                       static_cast<uint64_t>(n_bucket);
  auto it = ctx.one_d_tg_cache.find(key);
  if (it != ctx.one_d_tg_cache.end()) {
    return static_cast<NSUInteger>(it->second);
  }

  const NSUInteger max_threads = pipe.maxTotalThreadsPerThreadgroup;
  const NSUInteger candidates[] = {32, 64, 128, 256, 512};
  double best_score = std::numeric_limits<double>::max();
  NSUInteger best = 64;
  for (NSUInteger c : candidates) {
    if (c > max_threads || c > static_cast<NSUInteger>(n_bucket)) {
      continue;
    }
    const double groups = std::ceil(static_cast<double>(n) / static_cast<double>(c));
    const double idle = groups * static_cast<double>(c) - static_cast<double>(n);
    const double score = groups + 0.01 * idle;
    if (score < best_score) {
      best_score = score;
      best = c;
    }
  }

  ctx.one_d_tg_cache[key] = static_cast<uint32_t>(best);
  return best;
}

void dispatch1D(id<MTLComputeCommandEncoder> enc,
                MetalAttentionContext& ctx,
                id<MTLComputePipelineState> pipe,
                std::size_t n) {
  MTLSize grid = MTLSizeMake(n, 1, 1);
  NSUInteger w = autoTune1DThreadgroup(ctx, pipe, n);
  MTLSize tg = MTLSizeMake(w, 1, 1);
  [enc setComputePipelineState:pipe];
  [enc dispatchThreads:grid threadsPerThreadgroup:tg];
}

void dispatch2D(id<MTLComputeCommandEncoder> enc,
                id<MTLComputePipelineState> pipe,
                std::size_t x,
                std::size_t y) {
  MTLSize grid = MTLSizeMake(x, y, 1);
  NSUInteger w = pipe.threadExecutionWidth;
  NSUInteger max_threads = pipe.maxTotalThreadsPerThreadgroup;
  NSUInteger h = max_threads / (w == 0 ? 1 : w);
  if (h == 0) {
    h = 1;
  }
  if (w > 16) {
    w = 16;
  }
  if (h > 16) {
    h = 16;
  }
  MTLSize tg = MTLSizeMake(w, h, 1);
  [enc setComputePipelineState:pipe];
  [enc dispatchThreads:grid threadsPerThreadgroup:tg];
}

void dispatchGradVSgdVec4Tiled(id<MTLComputeCommandEncoder> enc,
                               id<MTLComputePipelineState> pipe,
                               std::size_t dim,
                               std::size_t seq,
                               NSUInteger tg_x) {
  const std::size_t c4 = dim / 4;
  MTLSize tg = MTLSizeMake(tg_x, 1, 1);
  MTLSize tgs = MTLSizeMake((c4 + tg_x - 1) / tg_x, seq, 1);
  [enc setComputePipelineState:pipe];
  [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tg];
}

NSUInteger autoTuneRowThreadgroup(
    std::unordered_map<uint32_t, uint32_t>& cache,
    id<MTLComputePipelineState> pipe,
    std::size_t seq) {
  uint32_t key = static_cast<uint32_t>(seq);
  auto it = cache.find(key);
  if (it != cache.end()) {
    return static_cast<NSUInteger>(it->second);
  }

  const NSUInteger max_threads = pipe.maxTotalThreadsPerThreadgroup;
  if (max_threads == 0) {
    cache[key] = 64;
    return 64;
  }

  const NSUInteger candidates[] = {32, 64, 128};
  double best_score = std::numeric_limits<double>::max();
  NSUInteger best = 64;
  for (NSUInteger c : candidates) {
    if (c > max_threads) {
      continue;
    }
    double groups = std::ceil(static_cast<double>(seq) / static_cast<double>(c));
    double idle = groups * static_cast<double>(c) - static_cast<double>(seq);
    double score = groups + 0.01 * idle;
    if (score < best_score) {
      best_score = score;
      best = c;
    }
  }

  cache[key] = static_cast<uint32_t>(best);
  return best;
}

NSUInteger autoTuneGradVec4Threadgroup(
    MetalAttentionContext& ctx,
    id<MTLComputePipelineState> pipe,
    std::size_t dim,
    NSUInteger preferred) {
  uint32_t c4 = static_cast<uint32_t>(dim / 4);
  uint64_t key = (static_cast<uint64_t>(c4) << 32U) | static_cast<uint64_t>(preferred);
  auto it = ctx.grad_vec4_tg_cache.find(key);
  if (it != ctx.grad_vec4_tg_cache.end()) {
    return static_cast<NSUInteger>(it->second);
  }

  const NSUInteger max_threads = pipe.maxTotalThreadsPerThreadgroup;
  const NSUInteger candidates[] = {16, 32, 64};
  double best_score = std::numeric_limits<double>::max();
  NSUInteger best = preferred;
  for (NSUInteger c : candidates) {
    if (c > max_threads) {
      continue;
    }
    double groups = std::ceil(static_cast<double>(c4) / static_cast<double>(c));
    double idle = groups * static_cast<double>(c) - static_cast<double>(c4);
    double pref_penalty = (c == preferred) ? 0.0 : 0.05;
    double score = groups + 0.01 * idle + pref_penalty;
    if (score < best_score) {
      best_score = score;
      best = c;
    }
  }

  ctx.grad_vec4_tg_cache[key] = static_cast<uint32_t>(best);
  return best;
}

void dispatchSoftmaxRows(id<MTLComputeCommandEncoder> enc,
                         MetalAttentionContext& ctx,
                         id<MTLComputePipelineState> pipe,
                         std::size_t seq) {
  const NSUInteger tg_w = autoTuneRowThreadgroup(ctx.softmax_tg_cache, pipe, seq);
  MTLSize tg = MTLSizeMake(tg_w, 1, 1);
  MTLSize tgs = MTLSizeMake(seq, 1, 1);
  [enc setComputePipelineState:pipe];
  [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tg];
}

void dispatchRowFused(id<MTLComputeCommandEncoder> enc,
                      MetalAttentionContext& ctx,
                      id<MTLComputePipelineState> pipe,
                      std::size_t seq) {
  const NSUInteger tg_w = autoTuneRowThreadgroup(ctx.row_fused_tg_cache, pipe, seq);
  MTLSize tg = MTLSizeMake(tg_w, 1, 1);
  MTLSize tgs = MTLSizeMake(seq, 1, 1);
  [enc setComputePipelineState:pipe];
  [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tg];
}

bool canUseCachedProbs(MetalAttentionContext& ctx, const AttentionConfig& cfg) {
  return ctx.probs_cache_valid && ctx.probs_cache_seq == cfg.seq_len && ctx.probs_cache_dim == cfg.head_dim &&
         ctx.probs_cache_causal == cfg.causal;
}

void encodeForwardPass(
    id<MTLComputeCommandEncoder> enc,
    MetalAttentionContext& ctx,
    const AttentionConfig& cfg,
    int mode,
    uint32_t seq,
    uint32_t dim,
    float scale,
    uint32_t causal) {
  if (mode == 1) {
    [enc setBuffer:ctx.buf_q offset:0 atIndex:0];
    [enc setBuffer:ctx.buf_k offset:0 atIndex:1];
    [enc setBuffer:ctx.buf_v offset:0 atIndex:2];
    [enc setBuffer:ctx.buf_out offset:0 atIndex:3];
    [enc setBytes:&seq length:sizeof(seq) atIndex:4];
    [enc setBytes:&dim length:sizeof(dim) atIndex:5];
    [enc setBytes:&scale length:sizeof(scale) atIndex:6];
    [enc setBytes:&causal length:sizeof(causal) atIndex:7];
    dispatch2D(enc, ctx.pipe_forward_online, cfg.head_dim, cfg.seq_len);
  } else if (mode == 2) {
    [enc setBuffer:ctx.buf_q offset:0 atIndex:0];
    [enc setBuffer:ctx.buf_k offset:0 atIndex:1];
    [enc setBuffer:ctx.buf_scores offset:0 atIndex:2];
    [enc setBytes:&seq length:sizeof(seq) atIndex:3];
    [enc setBytes:&dim length:sizeof(dim) atIndex:4];
    [enc setBytes:&scale length:sizeof(scale) atIndex:5];
    [enc setBytes:&causal length:sizeof(causal) atIndex:6];
    dispatch2D(enc, ctx.pipe_scores, cfg.seq_len, cfg.seq_len);

    [enc setBuffer:ctx.buf_scores offset:0 atIndex:0];
    [enc setBuffer:ctx.buf_v offset:0 atIndex:1];
    [enc setBuffer:ctx.buf_out offset:0 atIndex:2];
    [enc setBytes:&seq length:sizeof(seq) atIndex:3];
    [enc setBytes:&dim length:sizeof(dim) atIndex:4];
    dispatchRowFused(enc, ctx, ctx.pipe_softmax_out_fused, cfg.seq_len);
  } else {
    [enc setBuffer:ctx.buf_q offset:0 atIndex:0];
    [enc setBuffer:ctx.buf_k offset:0 atIndex:1];
    [enc setBuffer:ctx.buf_scores offset:0 atIndex:2];
    [enc setBytes:&seq length:sizeof(seq) atIndex:3];
    [enc setBytes:&dim length:sizeof(dim) atIndex:4];
    [enc setBytes:&scale length:sizeof(scale) atIndex:5];
    [enc setBytes:&causal length:sizeof(causal) atIndex:6];
    dispatch2D(enc, ctx.pipe_scores, cfg.seq_len, cfg.seq_len);

    [enc setBuffer:ctx.buf_scores offset:0 atIndex:0];
    [enc setBytes:&seq length:sizeof(seq) atIndex:1];
    dispatchSoftmaxRows(enc, ctx, ctx.pipe_softmax, cfg.seq_len);

    [enc setBuffer:ctx.buf_scores offset:0 atIndex:0];
    [enc setBuffer:ctx.buf_v offset:0 atIndex:1];
    [enc setBuffer:ctx.buf_out offset:0 atIndex:2];
    [enc setBytes:&seq length:sizeof(seq) atIndex:3];
    [enc setBytes:&dim length:sizeof(dim) atIndex:4];
    dispatch2D(enc, ctx.pipe_out, cfg.head_dim, cfg.seq_len);
  }
}

void encodeForwardPostprocess(
    id<MTLComputeCommandEncoder> enc,
    MetalAttentionContext& ctx,
    const AttentionConfig& cfg,
    const AttentionIoPolicy& policy) {
  const bool identity = (policy.output_scale == 1.0f && policy.output_bias == 0.0f && !policy.output_relu);
  if (identity) {
    return;
  }

  const uint32_t n = static_cast<uint32_t>(cfg.seq_len * cfg.head_dim);
  const uint32_t relu = policy.output_relu ? 1u : 0u;
  [enc setBuffer:ctx.buf_out offset:0 atIndex:0];
  [enc setBytes:&n length:sizeof(n) atIndex:1];
  [enc setBytes:&policy.output_scale length:sizeof(policy.output_scale) atIndex:2];
  [enc setBytes:&policy.output_bias length:sizeof(policy.output_bias) atIndex:3];
  [enc setBytes:&relu length:sizeof(relu) atIndex:4];
  dispatch1D(enc, ctx, ctx.pipe_out_postprocess, n);
}

runtime::Status runForwardOnMetal(const float* q,
                                  const float* k,
                                  const float* v,
                                  float* out,
                                  const AttentionConfig& cfg,
                                  const AttentionIoPolicy& policy,
                                  bool upload_q,
                                  bool upload_k,
                                  bool upload_v,
                                  bool copy_out_back,
                                  int mode,
                                  bool wait_for_completion,
                                  bool synchronize) {
  MetalAttentionContext& ctx = getContext();
  const std::size_t sd = cfg.seq_len * cfg.head_dim;
  const std::size_t sd_bytes = sd * sizeof(float);

  if ((upload_q || upload_k || upload_v) && ctx.async_inflight > 0) {
    runtime::Status st = waitForAttentionQueueIdle(ctx);
    if (st != runtime::Status::kSuccess) {
      return st;
    }
  }

  if (upload_q) {
    std::memcpy(ctx.buf_q.contents, q, sd_bytes);
    ctx.probs_cache_valid = false;
  }
  if (upload_k) {
    std::memcpy(ctx.buf_k.contents, k, sd_bytes);
    ctx.probs_cache_valid = false;
  }
  if (upload_v) {
    std::memcpy(ctx.buf_v.contents, v, sd_bytes);
  }

  uint32_t seq = static_cast<uint32_t>(cfg.seq_len);
  uint32_t dim = static_cast<uint32_t>(cfg.head_dim);
  float scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
  uint32_t causal = cfg.causal ? 1u : 0u;

  id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  if (!cmd || !enc) {
    return runtime::Status::kDriverError;
  }

  encodeForwardPass(enc, ctx, cfg, mode, seq, dim, scale, causal);
  encodeForwardPostprocess(enc, ctx, cfg, policy);

  [enc endEncoding];
  [cmd commit];

  if (wait_for_completion || synchronize) {
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) {
      return runtime::Status::kUnknown;
    }
    ctx.async_inflight = 0;
  } else {
    ctx.async_inflight += 1;
    if (ctx.async_inflight >= kMaxAsyncInflight) {
      [cmd waitUntilCompleted];
      if (cmd.status != MTLCommandBufferStatusCompleted) {
        return runtime::Status::kUnknown;
      }
      ctx.async_inflight = 0;
    }
  }

  if (copy_out_back) {
    if (!wait_for_completion) {
      [cmd waitUntilCompleted];
      if (cmd.status != MTLCommandBufferStatusCompleted) {
        return runtime::Status::kUnknown;
      }
    }
    std::memcpy(out, ctx.buf_out.contents, sd_bytes);
  }

  if (mode == 3 && cmd.status == MTLCommandBufferStatusCompleted) {
    ctx.probs_cache_valid = true;
    ctx.probs_cache_seq = cfg.seq_len;
    ctx.probs_cache_dim = cfg.head_dim;
    ctx.probs_cache_causal = cfg.causal;
  }
  return runtime::Status::kSuccess;
}

runtime::Status autoTuneForwardMode(const AttentionConfig& cfg, const AttentionIoPolicy& policy, int* out_mode) {
  MetalAttentionContext& ctx = getContext();
  if (out_mode == nullptr) {
    return runtime::Status::kInvalidValue;
  }

  ForwardTuneKey key;
  key.seq = cfg.seq_len;
  key.dim = cfg.head_dim;
  key.causal = cfg.causal ? 1u : 0u;
  key.resident_hint =
      (!policy.upload_q && !policy.upload_k && !policy.upload_v && !policy.download_out) ? 1u : 0u;
  key.sync_hint = policy.synchronize ? 1u : 0u;

  auto it = ctx.forward_mode_cache.find(key);
  if (it != ctx.forward_mode_cache.end()) {
    *out_mode = it->second;
    return runtime::Status::kSuccess;
  }

  const std::size_t sd = cfg.seq_len * cfg.head_dim;
  std::vector<float> q(sd, 0.1f);
  std::vector<float> k(sd, 0.2f);
  std::vector<float> v(sd, 0.3f);
  std::vector<float> out(sd, 0.0f);

  // Reflect resident/non-resident behavior in autotune while keeping deterministic timing.
  const bool tune_upload_q = policy.upload_q;
  const bool tune_upload_k = policy.upload_k;
  const bool tune_upload_v = policy.upload_v;
  const bool tune_download_out = policy.download_out;

  int best_mode = 1;
  double best_ms = 1.0e30;
  constexpr int kForwardTuneSamples = 3;

  for (int mode : {1, 2, 3}) {
    runtime::Status warm =
        runForwardOnMetal(
            q.data(),
            k.data(),
            v.data(),
            out.data(),
            cfg,
            policy,
            true,
            true,
            true,
            false,
            mode,
            true,
            true);
    if (warm != runtime::Status::kSuccess) {
      continue;
    }

    std::vector<double> samples;
    samples.reserve(kForwardTuneSamples);
    bool failed = false;
    for (int s = 0; s < kForwardTuneSamples; ++s) {
      auto start = std::chrono::high_resolution_clock::now();
      runtime::Status st = runForwardOnMetal(
          q.data(),
          k.data(),
          v.data(),
          out.data(),
          cfg,
          policy,
          tune_upload_q,
          tune_upload_k,
          tune_upload_v,
          tune_download_out,
          mode,
          true,
          true);
      auto end = std::chrono::high_resolution_clock::now();
      if (st != runtime::Status::kSuccess) {
        failed = true;
        break;
      }
      std::chrono::duration<double, std::milli> elapsed = end - start;
      samples.push_back(elapsed.count());
    }
    if (failed || samples.empty()) {
      continue;
    }

    const std::size_t mid = samples.size() / 2;
    std::nth_element(samples.begin(), samples.begin() + static_cast<std::ptrdiff_t>(mid), samples.end());
    const double mode_ms = samples[mid];
    if (mode_ms < best_ms) {
      best_ms = mode_ms;
      best_mode = mode;
    }
  }

  ctx.forward_mode_cache[key] = best_mode;
  markTuneCacheUpdated(ctx);
  *out_mode = best_mode;
  return runtime::Status::kSuccess;
}

runtime::Status runTrainUpdateKernelMode(
    MetalAttentionContext& ctx,
    const AttentionConfig& cfg,
    float lr,
    float norm,
    bool compute_loss,
  int plan) {
  uint32_t seq = static_cast<uint32_t>(cfg.seq_len);
  uint32_t dim = static_cast<uint32_t>(cfg.head_dim);
  uint32_t n = static_cast<uint32_t>(cfg.seq_len * cfg.head_dim);
  uint32_t causal = cfg.causal ? 1u : 0u;

  id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  if (!cmd || !enc) {
    return runtime::Status::kDriverError;
  }

  const bool use_precompute = compute_loss || (plan >= kTrainPlanPreScalar);
  int mode = plan;
  if (use_precompute && !compute_loss) {
    mode -= kTrainPlanPreScalar;
  }

  if (use_precompute) {
    [enc setBuffer:ctx.buf_out offset:0 atIndex:0];
    [enc setBuffer:ctx.buf_target offset:0 atIndex:1];
    [enc setBuffer:ctx.buf_grad_out offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    [enc setBytes:&norm length:sizeof(norm) atIndex:4];
    dispatch1D(enc, ctx, ctx.pipe_grad_out_only, n);

    [enc setBuffer:ctx.buf_scores offset:0 atIndex:0];
    [enc setBuffer:ctx.buf_grad_out offset:0 atIndex:1];
    [enc setBuffer:ctx.buf_v offset:0 atIndex:2];
    [enc setBytes:&seq length:sizeof(seq) atIndex:3];
    [enc setBytes:&dim length:sizeof(dim) atIndex:4];
    [enc setBytes:&lr length:sizeof(lr) atIndex:5];
    [enc setBytes:&causal length:sizeof(causal) atIndex:6];
    if (cfg.seq_len == 512 && mode == 0 && ctx.pipe_grad_v_sgd_seq512 != nil) {
      [enc setBuffer:ctx.buf_scores offset:0 atIndex:0];
      [enc setBuffer:ctx.buf_grad_out offset:0 atIndex:1];
      [enc setBuffer:ctx.buf_v offset:0 atIndex:2];
      [enc setBytes:&dim length:sizeof(dim) atIndex:3];
      [enc setBytes:&lr length:sizeof(lr) atIndex:4];
      dispatch2D(enc, ctx.pipe_grad_v_sgd_seq512, cfg.head_dim, cfg.seq_len);
    } else if (mode == 1) {
      dispatchGradVSgdVec4Tiled(
          enc,
          ctx.pipe_grad_v_sgd_vec4_tiled,
          cfg.head_dim,
          cfg.seq_len,
          autoTuneGradVec4Threadgroup(ctx, ctx.pipe_grad_v_sgd_vec4_tiled, cfg.head_dim, 32));
    } else if (mode == 2) {
      dispatchGradVSgdVec4Tiled(
          enc,
          ctx.pipe_grad_v_sgd_vec4_tiled16,
          cfg.head_dim,
          cfg.seq_len,
          autoTuneGradVec4Threadgroup(ctx, ctx.pipe_grad_v_sgd_vec4_tiled16, cfg.head_dim, 16));
    } else {
      dispatch2D(enc, ctx.pipe_grad_v_sgd, cfg.head_dim, cfg.seq_len);
    }
  } else {
    [enc setBuffer:ctx.buf_scores offset:0 atIndex:0];
    [enc setBuffer:ctx.buf_out offset:0 atIndex:1];
    [enc setBuffer:ctx.buf_target offset:0 atIndex:2];
    [enc setBuffer:ctx.buf_v offset:0 atIndex:3];
    [enc setBytes:&seq length:sizeof(seq) atIndex:4];
    [enc setBytes:&dim length:sizeof(dim) atIndex:5];
    [enc setBytes:&lr length:sizeof(lr) atIndex:6];
    [enc setBytes:&causal length:sizeof(causal) atIndex:7];
    [enc setBytes:&norm length:sizeof(norm) atIndex:8];
    if (mode == 1) {
      dispatchGradVSgdVec4Tiled(
        enc,
        ctx.pipe_grad_v_sgd_from_out_target_vec4_tiled,
        cfg.head_dim,
        cfg.seq_len,
        autoTuneGradVec4Threadgroup(
          ctx,
          ctx.pipe_grad_v_sgd_from_out_target_vec4_tiled,
          cfg.head_dim,
          32));
    } else if (mode == 2) {
      dispatchGradVSgdVec4Tiled(
        enc,
        ctx.pipe_grad_v_sgd_from_out_target_vec4_tiled16,
        cfg.head_dim,
        cfg.seq_len,
        autoTuneGradVec4Threadgroup(
          ctx,
          ctx.pipe_grad_v_sgd_from_out_target_vec4_tiled16,
          cfg.head_dim,
          16));
    } else {
      dispatch2D(enc, ctx.pipe_grad_v_sgd_from_out_target, cfg.head_dim, cfg.seq_len);
    }
  }

  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  return cmd.status == MTLCommandBufferStatusCompleted ? runtime::Status::kSuccess : runtime::Status::kUnknown;
}

runtime::Status autoTuneTrainKernelMode(
    const AttentionConfig& cfg,
    const AttentionIoPolicy& policy,
    bool compute_loss,
    int* out_mode) {
  if (out_mode == nullptr) {
    return runtime::Status::kInvalidValue;
  }

  MetalAttentionContext& ctx = getContext();
  TrainTuneKey key;
  key.seq = cfg.seq_len;
  key.dim = cfg.head_dim;
  key.causal = cfg.causal ? 1u : 0u;
  key.resident_hint = (!policy.upload_q && !policy.upload_k && !policy.upload_v && !policy.download_v) ? 1u : 0u;
  key.sync_hint = policy.synchronize ? 1u : 0u;
  key.compute_loss = compute_loss ? 1u : 0u;

  auto it = ctx.train_mode_cache.find(key);
  if (it != ctx.train_mode_cache.end()) {
    *out_mode = it->second;
    return runtime::Status::kSuccess;
  }

  if ((cfg.head_dim % 4) != 0) {
    if (compute_loss) {
      ctx.train_mode_cache[key] = kTrainPlanFusedScalar;
      *out_mode = kTrainPlanFusedScalar;
    } else {
      ctx.train_mode_cache[key] = kTrainPlanPreScalar;
      *out_mode = kTrainPlanPreScalar;
    }
    return runtime::Status::kSuccess;
  }

  const std::size_t sd = cfg.seq_len * cfg.head_dim;
  const std::size_t sd_bytes = sd * sizeof(float);
  std::vector<float> v_backup(sd, 0.0f);
  std::memcpy(v_backup.data(), ctx.buf_v.contents, sd_bytes);

  const float lr = 0.001f;
  const float norm = 1.0f / static_cast<float>(sd);
  std::vector<int> candidates;
  if (compute_loss) {
    candidates = {kTrainPlanFusedScalar, kTrainPlanFusedVec4T32, kTrainPlanFusedVec4T16};
  } else {
    candidates = {
        kTrainPlanFusedScalar,
        kTrainPlanFusedVec4T32,
        kTrainPlanFusedVec4T16,
        kTrainPlanPreScalar,
        kTrainPlanPreVec4T32,
        kTrainPlanPreVec4T16};
  }

  double plan_ms[6] = {1.0e30, 1.0e30, 1.0e30, 1.0e30, 1.0e30, 1.0e30};

  for (int plan : candidates) {
    double sum_ms = 0.0;
    bool ok = true;
    for (int rep = 0; rep < 6; ++rep) {
      std::memcpy(ctx.buf_v.contents, v_backup.data(), sd_bytes);
      auto start = std::chrono::high_resolution_clock::now();
      runtime::Status st = runTrainUpdateKernelMode(ctx, cfg, lr, norm, compute_loss, plan);
      auto end = std::chrono::high_resolution_clock::now();
      if (st != runtime::Status::kSuccess) {
        ok = false;
        break;
      }
      std::chrono::duration<double, std::milli> elapsed = end - start;
      if (rep > 0) {
        sum_ms += elapsed.count();
      }
    }
    if (!ok) {
      continue;
    }
    plan_ms[plan] = sum_ms / 5.0;
  }

  std::memcpy(ctx.buf_v.contents, v_backup.data(), sd_bytes);
  int best_mode = candidates.front();
  double best_ms = 1.0e30;
  for (int plan : candidates) {
    if (plan_ms[plan] < best_ms) {
      best_ms = plan_ms[plan];
      best_mode = plan;
    }
  }

  if (compute_loss) {
    const double scalar_ms = plan_ms[kTrainPlanFusedScalar];
    double best_vec_ms = 1.0e30;
    int best_vec_mode = kTrainPlanFusedScalar;
    if (plan_ms[kTrainPlanFusedVec4T32] < best_vec_ms) {
      best_vec_ms = plan_ms[kTrainPlanFusedVec4T32];
      best_vec_mode = kTrainPlanFusedVec4T32;
    }
    if (plan_ms[kTrainPlanFusedVec4T16] < best_vec_ms) {
      best_vec_ms = plan_ms[kTrainPlanFusedVec4T16];
      best_vec_mode = kTrainPlanFusedVec4T16;
    }
    if (best_vec_mode != kTrainPlanFusedScalar && best_vec_ms < 1.0e29 && scalar_ms < 1.0e29) {
      best_mode = (best_vec_ms <= scalar_ms * kTrainVec4AdoptRatio) ? best_vec_mode : kTrainPlanFusedScalar;
    }
  }

  const char* mode_name = "fused_scalar";
  if (best_mode == kTrainPlanFusedVec4T32) {
    mode_name = "fused_vec4_t32";
  } else if (best_mode == kTrainPlanFusedVec4T16) {
    mode_name = "fused_vec4_t16";
  } else if (best_mode == kTrainPlanPreScalar) {
    mode_name = "pre_scalar";
  } else if (best_mode == kTrainPlanPreVec4T32) {
    mode_name = "pre_vec4_t32";
  } else if (best_mode == kTrainPlanPreVec4T16) {
    mode_name = "pre_vec4_t16";
  }
  std::printf(
      "[autotune][train] seq=%zu dim=%zu causal=%u loss=%u resident=%u sync=%u mode=%s (f_s=%.6f, f_t32=%.6f, f_t16=%.6f, p_s=%.6f, p_t32=%.6f, p_t16=%.6f)\n",
      key.seq,
      key.dim,
      static_cast<unsigned>(key.causal),
      static_cast<unsigned>(key.compute_loss),
      static_cast<unsigned>(key.resident_hint),
      static_cast<unsigned>(key.sync_hint),
      mode_name,
      plan_ms[kTrainPlanFusedScalar],
      plan_ms[kTrainPlanFusedVec4T32],
      plan_ms[kTrainPlanFusedVec4T16],
      plan_ms[kTrainPlanPreScalar],
      plan_ms[kTrainPlanPreVec4T32],
      plan_ms[kTrainPlanPreVec4T16]);

  ctx.train_mode_cache[key] = best_mode;
  markTuneCacheUpdated(ctx);
  *out_mode = best_mode;
  return runtime::Status::kSuccess;
}

}  // namespace

runtime::Status attentionForwardMetal(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const AttentionConfig& cfg) {
  return attentionForwardMetalWithPolicy(q, k, v, out, cfg, AttentionIoPolicy{});
}

runtime::Status attentionForwardMetalWithPolicy(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const AttentionConfig& cfg,
    const AttentionIoPolicy& policy) {
  if (q == nullptr || k == nullptr || v == nullptr || out == nullptr || cfg.seq_len == 0 || cfg.head_dim == 0) {
    return runtime::Status::kInvalidValue;
  }

  @autoreleasepool {
    std::lock_guard<std::mutex> lock(getContextMutex());
    runtime::Status st = ensurePipelines();
    if (st != runtime::Status::kSuccess) {
      return st;
    }
    st = ensureBuffers(cfg);
    if (st != runtime::Status::kSuccess) {
      return st;
    }
    int selected_mode = 1;
    st = autoTuneForwardMode(cfg, policy, &selected_mode);
    if (st != runtime::Status::kSuccess) {
      return st;
    }
    return runForwardOnMetal(
        q,
        k,
        v,
        out,
        cfg,
      policy,
        policy.upload_q,
        policy.upload_k,
        policy.upload_v,
        policy.download_out,
        selected_mode,
        policy.download_out,
        policy.synchronize);
  }
}

runtime::Status attentionTrainStepMetal(
    const float* q,
    const float* k,
    float* v,
    const float* target,
    float* out,
    float learning_rate,
    const AttentionConfig& cfg,
    float* loss_out) {
  return attentionTrainStepMetalWithPolicy(
      q, k, v, target, out, learning_rate, cfg, loss_out, AttentionIoPolicy{});
}

runtime::Status attentionTrainStepMetalWithPolicy(
    const float* q,
    const float* k,
    float* v,
    const float* target,
    float* out,
    float learning_rate,
    const AttentionConfig& cfg,
    float* loss_out,
    const AttentionIoPolicy& policy) {
  if (q == nullptr || k == nullptr || v == nullptr || target == nullptr || out == nullptr ||
      cfg.seq_len == 0 || cfg.head_dim == 0 || learning_rate <= 0.0f) {
    return runtime::Status::kInvalidValue;
  }

  @autoreleasepool {
    std::lock_guard<std::mutex> lock(getContextMutex());
    runtime::Status st = ensurePipelines();
    if (st != runtime::Status::kSuccess) {
      return st;
    }
    st = ensureBuffers(cfg);
    if (st != runtime::Status::kSuccess) {
      return st;
    }

    MetalAttentionContext& ctx = getContext();
    const std::size_t sd = cfg.seq_len * cfg.head_dim;
    const std::size_t sd_bytes = sd * sizeof(float);

    if ((policy.upload_q || policy.upload_k || policy.upload_v || policy.upload_target) &&
        ctx.async_inflight > 0) {
      st = waitForAttentionQueueIdle(ctx);
      if (st != runtime::Status::kSuccess) {
        return st;
      }
    }

    if (policy.upload_q) {
      std::memcpy(ctx.buf_q.contents, q, sd_bytes);
    }
    if (policy.upload_k) {
      std::memcpy(ctx.buf_k.contents, k, sd_bytes);
    }
    if (policy.upload_v) {
      std::memcpy(ctx.buf_v.contents, v, sd_bytes);
    }
    if (policy.upload_target) {
      std::memcpy(ctx.buf_target.contents, target, sd_bytes);
    }

    uint32_t seq = static_cast<uint32_t>(cfg.seq_len);
    uint32_t dim = static_cast<uint32_t>(cfg.head_dim);
    uint32_t n = static_cast<uint32_t>(sd);
    uint32_t causal = cfg.causal ? 1u : 0u;
    float norm = 1.0f / static_cast<float>(sd);
    float lr = learning_rate;
    const bool compute_loss = (loss_out != nullptr);

    AttentionIoPolicy fwd_policy = policy;
    fwd_policy.download_out = false;
    fwd_policy.synchronize = false;

    int forward_mode = 3;
    const bool reuse_probs = (!policy.upload_q && !policy.upload_k && canUseCachedProbs(ctx, cfg));
    if (!reuse_probs) {
      st = autoTuneForwardMode(cfg, fwd_policy, &forward_mode);
      if (st != runtime::Status::kSuccess) {
        return st;
      }
    }

    int train_mode = 0;
    st = autoTuneTrainKernelMode(cfg, policy, compute_loss, &train_mode);
    if (st != runtime::Status::kSuccess) {
      return st;
    }

    id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) {
      return runtime::Status::kDriverError;
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
    if (reuse_probs) {
      [enc setBuffer:ctx.buf_scores offset:0 atIndex:0];
      [enc setBuffer:ctx.buf_v offset:0 atIndex:1];
      [enc setBuffer:ctx.buf_out offset:0 atIndex:2];
      [enc setBytes:&seq length:sizeof(seq) atIndex:3];
      [enc setBytes:&dim length:sizeof(dim) atIndex:4];
      dispatch2D(enc, ctx.pipe_out, cfg.head_dim, cfg.seq_len);
    } else {
      encodeForwardPass(enc, ctx, cfg, forward_mode, seq, dim, scale, causal);
      if (forward_mode == 3) {
        ctx.probs_cache_valid = true;
        ctx.probs_cache_seq = cfg.seq_len;
        ctx.probs_cache_dim = cfg.head_dim;
        ctx.probs_cache_causal = cfg.causal;
      } else {
        ctx.probs_cache_valid = false;
      }
    }

    if (compute_loss) {
      [enc setBuffer:ctx.buf_out offset:0 atIndex:0];
      [enc setBuffer:ctx.buf_target offset:0 atIndex:1];
      [enc setBuffer:ctx.buf_grad_out offset:0 atIndex:2];
      [enc setBuffer:ctx.buf_loss offset:0 atIndex:3];
      [enc setBytes:&n length:sizeof(n) atIndex:4];
      [enc setBytes:&norm length:sizeof(norm) atIndex:5];
      dispatch1D(enc, ctx, ctx.pipe_grad_out, sd);

      const uint32_t partial_count = static_cast<uint32_t>((sd + 255) / 256);
      MTLSize tg = MTLSizeMake(256, 1, 1);
      MTLSize tgs = MTLSizeMake(partial_count, 1, 1);
      [enc setComputePipelineState:ctx.pipe_loss_reduce_partials];
      [enc setBuffer:ctx.buf_loss offset:0 atIndex:0];
      [enc setBuffer:ctx.buf_loss_partials offset:0 atIndex:1];
      [enc setBytes:&n length:sizeof(n) atIndex:2];
      [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tg];

      [enc setComputePipelineState:ctx.pipe_loss_reduce_final];
      [enc setBuffer:ctx.buf_loss_partials offset:0 atIndex:0];
      [enc setBuffer:ctx.buf_loss_scalar offset:0 atIndex:1];
      [enc setBytes:&partial_count length:sizeof(partial_count) atIndex:2];
      [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:tg];
    }

    if (compute_loss) {
      [enc setBuffer:ctx.buf_scores offset:0 atIndex:0];
      [enc setBuffer:ctx.buf_grad_out offset:0 atIndex:1];
      [enc setBuffer:ctx.buf_v offset:0 atIndex:2];
      [enc setBytes:&seq length:sizeof(seq) atIndex:3];
      [enc setBytes:&dim length:sizeof(dim) atIndex:4];
      [enc setBytes:&lr length:sizeof(lr) atIndex:5];
      [enc setBytes:&causal length:sizeof(causal) atIndex:6];
      if (cfg.seq_len == 512 && train_mode == kTrainPlanFusedScalar && ctx.pipe_grad_v_sgd_seq512 != nil) {
        [enc setBuffer:ctx.buf_scores offset:0 atIndex:0];
        [enc setBuffer:ctx.buf_grad_out offset:0 atIndex:1];
        [enc setBuffer:ctx.buf_v offset:0 atIndex:2];
        [enc setBytes:&dim length:sizeof(dim) atIndex:3];
        [enc setBytes:&lr length:sizeof(lr) atIndex:4];
        dispatch2D(enc, ctx.pipe_grad_v_sgd_seq512, cfg.head_dim, cfg.seq_len);
      } else if (train_mode == kTrainPlanFusedVec4T32) {
        dispatchGradVSgdVec4Tiled(
            enc,
            ctx.pipe_grad_v_sgd_vec4_tiled,
            cfg.head_dim,
            cfg.seq_len,
            autoTuneGradVec4Threadgroup(ctx, ctx.pipe_grad_v_sgd_vec4_tiled, cfg.head_dim, 32));
      } else if (train_mode == kTrainPlanFusedVec4T16) {
        dispatchGradVSgdVec4Tiled(
            enc,
            ctx.pipe_grad_v_sgd_vec4_tiled16,
            cfg.head_dim,
            cfg.seq_len,
            autoTuneGradVec4Threadgroup(ctx, ctx.pipe_grad_v_sgd_vec4_tiled16, cfg.head_dim, 16));
      } else {
        dispatch2D(enc, ctx.pipe_grad_v_sgd, cfg.head_dim, cfg.seq_len);
      }
    } else {
      if (train_mode >= kTrainPlanPreScalar) {
        int pre_mode = train_mode - kTrainPlanPreScalar;
        [enc setBuffer:ctx.buf_out offset:0 atIndex:0];
        [enc setBuffer:ctx.buf_target offset:0 atIndex:1];
        [enc setBuffer:ctx.buf_grad_out offset:0 atIndex:2];
        [enc setBytes:&n length:sizeof(n) atIndex:3];
        [enc setBytes:&norm length:sizeof(norm) atIndex:4];
        dispatch1D(enc, ctx, ctx.pipe_grad_out_only, n);

        [enc setBuffer:ctx.buf_scores offset:0 atIndex:0];
        [enc setBuffer:ctx.buf_grad_out offset:0 atIndex:1];
        [enc setBuffer:ctx.buf_v offset:0 atIndex:2];
        [enc setBytes:&seq length:sizeof(seq) atIndex:3];
        [enc setBytes:&dim length:sizeof(dim) atIndex:4];
        [enc setBytes:&lr length:sizeof(lr) atIndex:5];
        [enc setBytes:&causal length:sizeof(causal) atIndex:6];

        if (cfg.seq_len == 512 && pre_mode == 0 && ctx.pipe_grad_v_sgd_seq512 != nil) {
          [enc setBuffer:ctx.buf_scores offset:0 atIndex:0];
          [enc setBuffer:ctx.buf_grad_out offset:0 atIndex:1];
          [enc setBuffer:ctx.buf_v offset:0 atIndex:2];
          [enc setBytes:&dim length:sizeof(dim) atIndex:3];
          [enc setBytes:&lr length:sizeof(lr) atIndex:4];
          dispatch2D(enc, ctx.pipe_grad_v_sgd_seq512, cfg.head_dim, cfg.seq_len);
        } else if (pre_mode == 1) {
          dispatchGradVSgdVec4Tiled(
              enc,
              ctx.pipe_grad_v_sgd_vec4_tiled,
              cfg.head_dim,
              cfg.seq_len,
              autoTuneGradVec4Threadgroup(ctx, ctx.pipe_grad_v_sgd_vec4_tiled, cfg.head_dim, 32));
        } else if (pre_mode == 2) {
          dispatchGradVSgdVec4Tiled(
              enc,
              ctx.pipe_grad_v_sgd_vec4_tiled16,
              cfg.head_dim,
              cfg.seq_len,
              autoTuneGradVec4Threadgroup(ctx, ctx.pipe_grad_v_sgd_vec4_tiled16, cfg.head_dim, 16));
        } else {
          dispatch2D(enc, ctx.pipe_grad_v_sgd, cfg.head_dim, cfg.seq_len);
        }
      } else {
        [enc setBuffer:ctx.buf_scores offset:0 atIndex:0];
        [enc setBuffer:ctx.buf_out offset:0 atIndex:1];
        [enc setBuffer:ctx.buf_target offset:0 atIndex:2];
        [enc setBuffer:ctx.buf_v offset:0 atIndex:3];
        [enc setBytes:&seq length:sizeof(seq) atIndex:4];
        [enc setBytes:&dim length:sizeof(dim) atIndex:5];
        [enc setBytes:&lr length:sizeof(lr) atIndex:6];
        [enc setBytes:&causal length:sizeof(causal) atIndex:7];
        [enc setBytes:&norm length:sizeof(norm) atIndex:8];
        if (train_mode == kTrainPlanFusedVec4T32) {
          dispatchGradVSgdVec4Tiled(
              enc,
              ctx.pipe_grad_v_sgd_from_out_target_vec4_tiled,
              cfg.head_dim,
              cfg.seq_len,
            autoTuneGradVec4Threadgroup(
              ctx,
              ctx.pipe_grad_v_sgd_from_out_target_vec4_tiled,
              cfg.head_dim,
              32));
        } else if (train_mode == kTrainPlanFusedVec4T16) {
          dispatchGradVSgdVec4Tiled(
              enc,
              ctx.pipe_grad_v_sgd_from_out_target_vec4_tiled16,
              cfg.head_dim,
              cfg.seq_len,
            autoTuneGradVec4Threadgroup(
              ctx,
              ctx.pipe_grad_v_sgd_from_out_target_vec4_tiled16,
              cfg.head_dim,
              16));
        } else {
          dispatch2D(enc, ctx.pipe_grad_v_sgd_from_out_target, cfg.head_dim, cfg.seq_len);
        }
      }
    }

    [enc endEncoding];
    [cmd commit];

    const bool need_sync = policy.synchronize || policy.download_v || policy.download_out || (loss_out != nullptr);
    if (need_sync) {
      [cmd waitUntilCompleted];
      if (cmd.status != MTLCommandBufferStatusCompleted) {
        return runtime::Status::kUnknown;
      }
      ctx.async_inflight = 0;
    } else {
      ctx.async_inflight += 1;
      if (ctx.async_inflight >= kMaxAsyncInflight) {
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) {
          return runtime::Status::kUnknown;
        }
        ctx.async_inflight = 0;
      }
      return runtime::Status::kSuccess;
    }

    if (policy.download_v) {
      std::memcpy(v, ctx.buf_v.contents, sd_bytes);
    }

    if (policy.download_out) {
      std::memcpy(out, ctx.buf_out.contents, sd_bytes);
    }

    if (loss_out != nullptr) {
      const float* loss_scalar = static_cast<const float*>(ctx.buf_loss_scalar.contents);
      *loss_out = loss_scalar[0] * norm;
    }

    return runtime::Status::kSuccess;
  }
}

}  // namespace lightning_core::detail

#else

namespace lightning_core::detail {

runtime::Status attentionForwardMetal(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const AttentionConfig& cfg) {
  (void)q;
  (void)k;
  (void)v;
  (void)out;
  (void)cfg;
  return runtime::Status::kNotSupported;
}

runtime::Status attentionForwardMetalWithPolicy(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const AttentionConfig& cfg,
    const AttentionIoPolicy& policy) {
  (void)q;
  (void)k;
  (void)v;
  (void)out;
  (void)cfg;
  (void)policy;
  return runtime::Status::kNotSupported;
}

runtime::Status attentionTrainStepMetal(
    const float* q,
    const float* k,
    float* v,
    const float* target,
    float* out,
    float learning_rate,
    const AttentionConfig& cfg,
    float* loss_out) {
  (void)q;
  (void)k;
  (void)v;
  (void)target;
  (void)out;
  (void)learning_rate;
  (void)cfg;
  (void)loss_out;
  return runtime::Status::kNotSupported;
}

runtime::Status attentionTrainStepMetalWithPolicy(
    const float* q,
    const float* k,
    float* v,
    const float* target,
    float* out,
    float learning_rate,
    const AttentionConfig& cfg,
    float* loss_out,
    const AttentionIoPolicy& policy) {
  (void)q;
  (void)k;
  (void)v;
  (void)target;
  (void)out;
  (void)learning_rate;
  (void)cfg;
  (void)loss_out;
  (void)policy;
  return runtime::Status::kNotSupported;
}

}  // namespace lightning_core::detail

#endif
