#include "cudajun/detail/ops_backend.hpp"

#if defined(CJ_HAS_METAL) && CJ_HAS_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace cudajun::detail {

namespace {

constexpr uint32_t kMatMulMaxAsyncInflight = 256;

struct MatMulTuneKey {
  uint32_t m = 0;
  uint32_t k = 0;
  uint32_t n = 0;

  bool operator==(const MatMulTuneKey& other) const {
    return m == other.m && k == other.k && n == other.n;
  }
};

struct MatMulTuneKeyHash {
  std::size_t operator()(const MatMulTuneKey& key) const {
    std::size_t h = key.m;
    h ^= (static_cast<std::size_t>(key.k) + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U));
    h ^= (static_cast<std::size_t>(key.n) + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U));
    return h;
  }
};

struct MatMulTuneValue {
  uint32_t best_tile = 16;
  bool use_mps = false;
};

struct MatMulSmallTuneValue {
  bool use_small_kernel = false;
};

struct MetalMatMulContext {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> pipeline_m1 = nil;
  id<MTLComputePipelineState> pipeline_m2 = nil;
  id<MTLComputePipelineState> pipeline_t8 = nil;
  id<MTLComputePipelineState> pipeline_t12 = nil;
  id<MTLComputePipelineState> pipeline_t16 = nil;
  id<MTLComputePipelineState> pipeline_t32 = nil;
  id<MTLBuffer> buf_a = nil;
  id<MTLBuffer> buf_b = nil;
  id<MTLBuffer> buf_out = nil;
  std::size_t cap_a = 0;
  std::size_t cap_b = 0;
  std::size_t cap_out = 0;
  std::size_t tuned_m = 0;
  std::size_t tuned_k = 0;
  std::size_t tuned_n = 0;
  uint32_t best_tile = 0;
  bool use_mps = false;
  uint32_t mps_m = 0;
  uint32_t mps_k = 0;
  uint32_t mps_n = 0;
  MPSMatrixMultiplication* mps_op = nil;
  std::unordered_map<MatMulTuneKey, MatMulTuneValue, MatMulTuneKeyHash> tune_cache;
  std::unordered_map<MatMulTuneKey, MatMulSmallTuneValue, MatMulTuneKeyHash> small_tune_cache;
  bool tune_cache_loaded = false;
  bool tune_cache_dirty = false;
  std::string tune_cache_path;
  bool small_tune_cache_loaded = false;
  bool small_tune_cache_dirty = false;
  std::string small_tune_cache_path;
  uint32_t async_inflight = 0;
};

std::string resolveMatMulTuneCachePath() {
  const char* env = std::getenv("CJ_MATMUL_TUNE_CACHE_FILE");
  if (env != nullptr && env[0] != '\0') {
    return std::string(env);
  }
  return std::string(".cudajun_matmul_tune_cache.csv");
}

enum class MatMulSmallBatchMode {
  kAuto,
  kForceKernel,
  kForceMps,
};

MatMulSmallBatchMode matMulSmallBatchMode() {
  const char* env = std::getenv("CJ_MATMUL_MLE2_KERNEL");
  if (env == nullptr || env[0] == '\0') {
    return MatMulSmallBatchMode::kAuto;
  }
  if (std::strcmp(env, "1") == 0 || std::strcmp(env, "kernel") == 0) {
    return MatMulSmallBatchMode::kForceKernel;
  }
  if (std::strcmp(env, "mps") == 0 || std::strcmp(env, "2") == 0) {
    return MatMulSmallBatchMode::kForceMps;
  }
  return MatMulSmallBatchMode::kAuto;
}

double matMulSmallHysteresisPct() {
  const char* env = std::getenv("CJ_MATMUL_MLE2_HYST_PCT");
  if (env == nullptr || env[0] == '\0') {
    return 5.0;
  }
  char* end = nullptr;
  double v = std::strtod(env, &end);
  if (end == env || *end != '\0' || v < 0.0) {
    return 5.0;
  }
  return v;
}

std::string resolveMatMulSmallTuneCachePath() {
  const char* env = std::getenv("CJ_MATMUL_SMALL_TUNE_CACHE_FILE");
  if (env != nullptr && env[0] != '\0') {
    return std::string(env);
  }

  std::string suffix;
  if (const char* profile = std::getenv("CJ_MATMUL_SMALL_TUNE_PROFILE")) {
    if (profile[0] != '\0') {
      suffix = std::string("_") + profile;
    }
  }
  return std::string(".cudajun_matmul_mle2_tune_cache") + suffix + ".csv";
}

void saveMatMulTuneCacheIfDirty(MetalMatMulContext& ctx) {
  if (!ctx.tune_cache_loaded || !ctx.tune_cache_dirty || ctx.tune_cache_path.empty()) {
    return;
  }

  std::ofstream ofs(ctx.tune_cache_path, std::ios::out | std::ios::trunc);
  if (!ofs.is_open()) {
    return;
  }
  for (const auto& kv : ctx.tune_cache) {
    ofs << kv.first.m << "," << kv.first.k << "," << kv.first.n << "," << kv.second.best_tile << ","
        << static_cast<unsigned>(kv.second.use_mps ? 1 : 0) << "\n";
  }
  if (ofs.good()) {
    ctx.tune_cache_dirty = false;
  }
}

void saveMatMulSmallTuneCacheIfDirty(MetalMatMulContext& ctx) {
  if (!ctx.small_tune_cache_loaded || !ctx.small_tune_cache_dirty || ctx.small_tune_cache_path.empty()) {
    return;
  }

  std::ofstream ofs(ctx.small_tune_cache_path, std::ios::out | std::ios::trunc);
  if (!ofs.is_open()) {
    return;
  }
  for (const auto& kv : ctx.small_tune_cache) {
    ofs << kv.first.m << "," << kv.first.k << "," << kv.first.n << ","
        << static_cast<unsigned>(kv.second.use_small_kernel ? 1 : 0) << "\n";
  }
  if (ofs.good()) {
    ctx.small_tune_cache_dirty = false;
  }
}

void loadMatMulTuneCacheIfNeeded(MetalMatMulContext& ctx) {
  if (ctx.tune_cache_loaded) {
    return;
  }
  ctx.tune_cache_loaded = true;
  ctx.tune_cache_path = resolveMatMulTuneCachePath();

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
    if (cols.size() != 5) {
      continue;
    }
    try {
      MatMulTuneKey key;
      key.m = static_cast<uint32_t>(std::stoul(cols[0]));
      key.k = static_cast<uint32_t>(std::stoul(cols[1]));
      key.n = static_cast<uint32_t>(std::stoul(cols[2]));
      MatMulTuneValue val;
      val.best_tile = static_cast<uint32_t>(std::stoul(cols[3]));
      val.use_mps = std::stoi(cols[4]) != 0;
      ctx.tune_cache[key] = val;
    } catch (...) {
      // Ignore malformed lines.
    }
  }
}

void loadMatMulSmallTuneCacheIfNeeded(MetalMatMulContext& ctx) {
  if (ctx.small_tune_cache_loaded) {
    return;
  }
  ctx.small_tune_cache_loaded = true;
  ctx.small_tune_cache_path = resolveMatMulSmallTuneCachePath();

  std::ifstream ifs(ctx.small_tune_cache_path);
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
    if (cols.size() != 4) {
      continue;
    }
    try {
      MatMulTuneKey key;
      key.m = static_cast<uint32_t>(std::stoul(cols[0]));
      key.k = static_cast<uint32_t>(std::stoul(cols[1]));
      key.n = static_cast<uint32_t>(std::stoul(cols[2]));
      MatMulSmallTuneValue val;
      val.use_small_kernel = std::stoi(cols[3]) != 0;
      ctx.small_tune_cache[key] = val;
    } catch (...) {
      // Ignore malformed lines.
    }
  }
}

runtime::Status waitForQueueIdle(MetalMatMulContext& ctx) {
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

const char* kMatMulMetalShader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void matmul_f32_t8(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]) {
  constexpr uint TILE = 8;
  uint row = tgid.y * TILE + tid.y;
  uint col = tgid.x * TILE + tid.x;

  threadgroup float a_tile[TILE][TILE];
  threadgroup float b_tile[TILE][TILE];

  float acc = 0.0f;
  uint num_tiles = (k + TILE - 1) / TILE;
  for (uint t = 0; t < num_tiles; ++t) {
    uint a_col = t * TILE + tid.x;
    uint b_row = t * TILE + tid.y;

    a_tile[tid.y][tid.x] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    b_tile[tid.y][tid.x] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = 0; p < TILE; ++p) {
      acc += a_tile[tid.y][p] * b_tile[p][tid.x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (row < m && col < n) {
    out[row * n + col] = acc;
  }
}

kernel void matmul_f32_t12(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]) {
  constexpr uint TILE = 12;
  uint row = tgid.y * TILE + tid.y;
  uint col = tgid.x * TILE + tid.x;

  threadgroup float a_tile[TILE][TILE];
  threadgroup float b_tile[TILE][TILE];

  float acc = 0.0f;
  uint num_tiles = (k + TILE - 1) / TILE;
  for (uint t = 0; t < num_tiles; ++t) {
    uint a_col = t * TILE + tid.x;
    uint b_row = t * TILE + tid.y;

    a_tile[tid.y][tid.x] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    b_tile[tid.y][tid.x] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = 0; p < TILE; ++p) {
      acc += a_tile[tid.y][p] * b_tile[p][tid.x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (row < m && col < n) {
    out[row * n + col] = acc;
  }
}

kernel void matmul_f32_t16(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]) {
  constexpr uint TILE = 16;
  uint row = tgid.y * TILE + tid.y;
  uint col = tgid.x * TILE + tid.x;

  threadgroup float a_tile[TILE][TILE];
  threadgroup float b_tile[TILE][TILE];

  float acc = 0.0f;
  uint num_tiles = (k + TILE - 1) / TILE;
  for (uint t = 0; t < num_tiles; ++t) {
    uint a_col = t * TILE + tid.x;
    uint b_row = t * TILE + tid.y;

    a_tile[tid.y][tid.x] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    b_tile[tid.y][tid.x] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = 0; p < TILE; ++p) {
      acc += a_tile[tid.y][p] * b_tile[p][tid.x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (row < m && col < n) {
    out[row * n + col] = acc;
  }
}

kernel void matmul_f32_t32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]) {
  constexpr uint TILE = 32;
  uint row = tgid.y * TILE + tid.y;
  uint col = tgid.x * TILE + tid.x;

  threadgroup float a_tile[TILE][TILE];
  threadgroup float b_tile[TILE][TILE];

  float acc = 0.0f;
  uint num_tiles = (k + TILE - 1) / TILE;
  for (uint t = 0; t < num_tiles; ++t) {
    uint a_col = t * TILE + tid.x;
    uint b_row = t * TILE + tid.y;

    a_tile[tid.y][tid.x] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    b_tile[tid.y][tid.x] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = 0; p < TILE; ++p) {
      acc += a_tile[tid.y][p] * b_tile[p][tid.x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (row < m && col < n) {
    out[row * n + col] = acc;
  }
}

kernel void matmul_f32_m1(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& k [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
  uint col = tgid.x;
  if (col >= n || simd_width == 0) {
    return;
  }
  float partial = 0.0f;
  uint p = lane * 4;
  uint stride = simd_width * 4;
  for (; p + 3 < k; p += stride) {
    partial += a[p] * b[p * n + col];
    partial += a[p + 1] * b[(p + 1) * n + col];
    partial += a[p + 2] * b[(p + 2) * n + col];
    partial += a[p + 3] * b[(p + 3) * n + col];
  }
  for (p = (k & ~3u) + lane; p < k; p += simd_width) {
    partial += a[p] * b[p * n + col];
  }
  float acc = simd_sum(partial);
  if (lane == 0) {
    out[col] = acc;
  }
}

kernel void matmul_f32_m2(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& k [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
  uint col = tgid.x;
  if (col >= n || simd_width == 0) {
    return;
  }
  float partial0 = 0.0f;
  float partial1 = 0.0f;
  uint p = lane * 4;
  uint stride = simd_width * 4;
  for (; p + 3 < k; p += stride) {
    float bv0 = b[p * n + col];
    float bv1 = b[(p + 1) * n + col];
    float bv2 = b[(p + 2) * n + col];
    float bv3 = b[(p + 3) * n + col];
    partial0 += a[p] * bv0 + a[p + 1] * bv1 + a[p + 2] * bv2 + a[p + 3] * bv3;
    partial1 += a[k + p] * bv0 + a[k + p + 1] * bv1 + a[k + p + 2] * bv2 + a[k + p + 3] * bv3;
  }
  for (p = (k & ~3u) + lane; p < k; p += simd_width) {
    float bv = b[p * n + col];
    partial0 += a[p] * bv;
    partial1 += a[k + p] * bv;
  }
  float acc0 = simd_sum(partial0);
  float acc1 = simd_sum(partial1);
  if (lane == 0) {
    out[col] = acc0;
    out[n + col] = acc1;
  }
}
)";

MetalMatMulContext& getContext() {
  static MetalMatMulContext ctx;
  return ctx;
}

id<MTLComputePipelineState> pipelineForTile(MetalMatMulContext& ctx, uint32_t tile) {
  if (tile == 8) {
    return ctx.pipeline_t8;
  }
  if (tile == 12) {
    return ctx.pipeline_t12;
  }
  if (tile == 16) {
    return ctx.pipeline_t16;
  }
  return ctx.pipeline_t32;
}

runtime::Status ensureContext() {
  MetalMatMulContext& ctx = getContext();

  if (ctx.device == nil) {
    ctx.device = MTLCreateSystemDefaultDevice();
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

  if (ctx.pipeline_m1 != nil && ctx.pipeline_m2 != nil &&
      ctx.pipeline_t8 != nil && ctx.pipeline_t12 != nil && ctx.pipeline_t16 != nil && ctx.pipeline_t32 != nil) {
    loadMatMulTuneCacheIfNeeded(ctx);
    loadMatMulSmallTuneCacheIfNeeded(ctx);
    return runtime::Status::kSuccess;
  }

  NSString* source = [NSString stringWithUTF8String:kMatMulMetalShader];
  NSError* err = nil;
  id<MTLLibrary> lib = [ctx.device newLibraryWithSource:source options:nil error:&err];
  if (!lib) {
    return runtime::Status::kDriverError;
  }

  id<MTLFunction> fn_m1 = [lib newFunctionWithName:@"matmul_f32_m1"];
  id<MTLFunction> fn_m2 = [lib newFunctionWithName:@"matmul_f32_m2"];
  id<MTLFunction> fn8 = [lib newFunctionWithName:@"matmul_f32_t8"];
  id<MTLFunction> fn12 = [lib newFunctionWithName:@"matmul_f32_t12"];
  id<MTLFunction> fn16 = [lib newFunctionWithName:@"matmul_f32_t16"];
  id<MTLFunction> fn32 = [lib newFunctionWithName:@"matmul_f32_t32"];
  if (!fn_m1 || !fn_m2 || !fn8 || !fn12 || !fn16 || !fn32) {
    return runtime::Status::kDriverError;
  }

  ctx.pipeline_m1 = [ctx.device newComputePipelineStateWithFunction:fn_m1 error:&err];
  ctx.pipeline_m2 = [ctx.device newComputePipelineStateWithFunction:fn_m2 error:&err];
  ctx.pipeline_t8 = [ctx.device newComputePipelineStateWithFunction:fn8 error:&err];
  ctx.pipeline_t12 = [ctx.device newComputePipelineStateWithFunction:fn12 error:&err];
  ctx.pipeline_t16 = [ctx.device newComputePipelineStateWithFunction:fn16 error:&err];
  ctx.pipeline_t32 = [ctx.device newComputePipelineStateWithFunction:fn32 error:&err];
  if (!ctx.pipeline_m1 || !ctx.pipeline_m2 ||
      !ctx.pipeline_t8 || !ctx.pipeline_t12 || !ctx.pipeline_t16 || !ctx.pipeline_t32) {
    return runtime::Status::kDriverError;
  }

  loadMatMulTuneCacheIfNeeded(ctx);
  loadMatMulSmallTuneCacheIfNeeded(ctx);

  return runtime::Status::kSuccess;
}

runtime::Status ensureBuffers(std::size_t bytes_a, std::size_t bytes_b, std::size_t bytes_out) {
  MetalMatMulContext& ctx = getContext();

  if (ctx.cap_a < bytes_a) {
    ctx.buf_a = [ctx.device newBufferWithLength:bytes_a options:MTLResourceStorageModeShared];
    if (!ctx.buf_a) {
      return runtime::Status::kOutOfMemory;
    }
    ctx.cap_a = bytes_a;
  }

  if (ctx.cap_b < bytes_b) {
    ctx.buf_b = [ctx.device newBufferWithLength:bytes_b options:MTLResourceStorageModeShared];
    if (!ctx.buf_b) {
      return runtime::Status::kOutOfMemory;
    }
    ctx.cap_b = bytes_b;
  }

  if (ctx.cap_out < bytes_out) {
    ctx.buf_out = [ctx.device newBufferWithLength:bytes_out options:MTLResourceStorageModeShared];
    if (!ctx.buf_out) {
      return runtime::Status::kOutOfMemory;
    }
    ctx.cap_out = bytes_out;
  }

  return runtime::Status::kSuccess;
}

runtime::Status runMatMulKernel(
    MetalMatMulContext& ctx,
    uint32_t tile,
    uint32_t mm,
    uint32_t kk,
  uint32_t nn,
  bool wait_for_completion) {
  id<MTLComputePipelineState> pipe = pipelineForTile(ctx, tile);
  if (!pipe) {
    return runtime::Status::kDriverError;
  }
  if (pipe.maxTotalThreadsPerThreadgroup < (tile * tile)) {
    return runtime::Status::kNotSupported;
  }

  id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  if (!cmd || !enc) {
    return runtime::Status::kDriverError;
  }

  [enc setComputePipelineState:pipe];
  [enc setBuffer:ctx.buf_a offset:0 atIndex:0];
  [enc setBuffer:ctx.buf_b offset:0 atIndex:1];
  [enc setBuffer:ctx.buf_out offset:0 atIndex:2];
  [enc setBytes:&mm length:sizeof(mm) atIndex:3];
  [enc setBytes:&kk length:sizeof(kk) atIndex:4];
  [enc setBytes:&nn length:sizeof(nn) atIndex:5];

  MTLSize tg = MTLSizeMake(tile, tile, 1);
  MTLSize tgs = MTLSizeMake((nn + tile - 1) / tile, (mm + tile - 1) / tile, 1);
  [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tg];
  [enc endEncoding];

  [cmd commit];
  if (wait_for_completion) {
    [cmd waitUntilCompleted];
    return cmd.status == MTLCommandBufferStatusCompleted ? runtime::Status::kSuccess : runtime::Status::kUnknown;
  }
  return runtime::Status::kSuccess;
}

runtime::Status runMatMulMps(
    MetalMatMulContext& ctx,
    uint32_t mm,
    uint32_t kk,
  uint32_t nn,
  bool wait_for_completion) {
  if (ctx.mps_op == nil || ctx.mps_m != mm || ctx.mps_k != kk || ctx.mps_n != nn) {
    ctx.mps_op = [[MPSMatrixMultiplication alloc] initWithDevice:ctx.device
                                                    transposeLeft:NO
                                                   transposeRight:NO
                                                       resultRows:mm
                                                    resultColumns:nn
                                                  interiorColumns:kk
                                                            alpha:1.0f
                                                             beta:0.0f];
    if (ctx.mps_op == nil) {
      return runtime::Status::kNotSupported;
    }
    ctx.mps_m = mm;
    ctx.mps_k = kk;
    ctx.mps_n = nn;
  }

  MPSMatrixDescriptor* aDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:mm
                                                                      columns:kk
                                                                     rowBytes:kk * sizeof(float)
                                                                     dataType:MPSDataTypeFloat32];
  MPSMatrixDescriptor* bDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:kk
                                                                      columns:nn
                                                                     rowBytes:nn * sizeof(float)
                                                                     dataType:MPSDataTypeFloat32];
  MPSMatrixDescriptor* cDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:mm
                                                                      columns:nn
                                                                     rowBytes:nn * sizeof(float)
                                                                     dataType:MPSDataTypeFloat32];

  MPSMatrix* aMat = [[MPSMatrix alloc] initWithBuffer:ctx.buf_a descriptor:aDesc];
  MPSMatrix* bMat = [[MPSMatrix alloc] initWithBuffer:ctx.buf_b descriptor:bDesc];
  MPSMatrix* cMat = [[MPSMatrix alloc] initWithBuffer:ctx.buf_out descriptor:cDesc];
  if (!aMat || !bMat || !cMat) {
    return runtime::Status::kDriverError;
  }

  id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
  if (!cmd) {
    return runtime::Status::kDriverError;
  }
  [ctx.mps_op encodeToCommandBuffer:cmd leftMatrix:aMat rightMatrix:bMat resultMatrix:cMat];
  [cmd commit];
  if (wait_for_completion) {
    [cmd waitUntilCompleted];
    return cmd.status == MTLCommandBufferStatusCompleted ? runtime::Status::kSuccess : runtime::Status::kUnknown;
  }
  return runtime::Status::kSuccess;
}

runtime::Status runMatMulSmallBatch(
    MetalMatMulContext& ctx,
    uint32_t mm,
    uint32_t kk,
    uint32_t nn,
    bool wait_for_completion) {
  id<MTLComputePipelineState> pipe = (mm == 1) ? ctx.pipeline_m1 : ctx.pipeline_m2;
  if (!pipe) {
    return runtime::Status::kDriverError;
  }

  id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  if (!cmd || !enc) {
    return runtime::Status::kDriverError;
  }

  [enc setComputePipelineState:pipe];
  [enc setBuffer:ctx.buf_a offset:0 atIndex:0];
  [enc setBuffer:ctx.buf_b offset:0 atIndex:1];
  [enc setBuffer:ctx.buf_out offset:0 atIndex:2];
  [enc setBytes:&kk length:sizeof(kk) atIndex:3];
  [enc setBytes:&nn length:sizeof(nn) atIndex:4];

  NSUInteger width = pipe.threadExecutionWidth;
  if (width == 0) {
    width = 32;
  }
  MTLSize tg = MTLSizeMake(width, 1, 1);
  MTLSize tgs = MTLSizeMake(nn, 1, 1);
  [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tg];
  [enc endEncoding];

  [cmd commit];
  if (wait_for_completion) {
    [cmd waitUntilCompleted];
    return cmd.status == MTLCommandBufferStatusCompleted ? runtime::Status::kSuccess : runtime::Status::kUnknown;
  }
  return runtime::Status::kSuccess;
}

runtime::Status autoTuneSmallBatchImpl(
    MetalMatMulContext& ctx,
    uint32_t mm,
    uint32_t kk,
    uint32_t nn,
    bool* use_small_kernel_out) {
  if (use_small_kernel_out == nullptr) {
    return runtime::Status::kInvalidValue;
  }
  loadMatMulSmallTuneCacheIfNeeded(ctx);

  MatMulTuneKey key{mm, kk, nn};
  auto cached = ctx.small_tune_cache.find(key);
  if (cached != ctx.small_tune_cache.end()) {
    *use_small_kernel_out = cached->second.use_small_kernel;
    return runtime::Status::kSuccess;
  }

  const auto run_avg_ms = [&](bool use_small, double* out_ms) -> runtime::Status {
    runtime::Status warm = use_small ? runMatMulSmallBatch(ctx, mm, kk, nn, true) : runMatMulMps(ctx, mm, kk, nn, true);
    if (warm != runtime::Status::kSuccess) {
      return warm;
    }
    double sum_ms = 0.0;
    for (int rep = 0; rep < 4; ++rep) {
      auto start = std::chrono::high_resolution_clock::now();
      runtime::Status st = use_small ? runMatMulSmallBatch(ctx, mm, kk, nn, true) : runMatMulMps(ctx, mm, kk, nn, true);
      auto end = std::chrono::high_resolution_clock::now();
      if (st != runtime::Status::kSuccess) {
        return st;
      }
      std::chrono::duration<double, std::milli> elapsed = end - start;
      sum_ms += elapsed.count();
    }
    *out_ms = sum_ms / 4.0;
    return runtime::Status::kSuccess;
  };

  bool best_use_small = false;
  double best_ms = std::numeric_limits<double>::max();

  double small_ms = 0.0;
  runtime::Status small_st = run_avg_ms(true, &small_ms);
  if (small_st == runtime::Status::kSuccess && small_ms < best_ms) {
    best_ms = small_ms;
    best_use_small = true;
  }

  double mps_ms = 0.0;
  runtime::Status mps_st = run_avg_ms(false, &mps_ms);
  if (mps_st == runtime::Status::kSuccess && mps_ms < best_ms) {
    best_ms = mps_ms;
    best_use_small = false;
  }

  if (small_st != runtime::Status::kSuccess && mps_st != runtime::Status::kSuccess) {
    return runtime::Status::kNotSupported;
  }

  if (small_st == runtime::Status::kSuccess && mps_st == runtime::Status::kSuccess) {
    const double hysteresis_pct = matMulSmallHysteresisPct();
    // Prefer MPS when the specialized kernel doesn't beat it by a clear margin.
    const double mps_threshold = mps_ms * (1.0 - hysteresis_pct / 100.0);
    if (small_ms >= mps_threshold) {
      best_use_small = false;
      best_ms = mps_ms;
    }
  }

  ctx.small_tune_cache[key] = MatMulSmallTuneValue{best_use_small};
  ctx.small_tune_cache_dirty = true;
  saveMatMulSmallTuneCacheIfDirty(ctx);
  *use_small_kernel_out = best_use_small;
  return runtime::Status::kSuccess;
}

runtime::Status autoTuneTile(MetalMatMulContext& ctx, uint32_t mm, uint32_t kk, uint32_t nn) {
  MatMulTuneKey key{mm, kk, nn};
  auto cached = ctx.tune_cache.find(key);
  if (cached != ctx.tune_cache.end()) {
    ctx.best_tile = cached->second.best_tile;
    ctx.use_mps = cached->second.use_mps;
    ctx.tuned_m = static_cast<std::size_t>(mm);
    ctx.tuned_k = static_cast<std::size_t>(kk);
    ctx.tuned_n = static_cast<std::size_t>(nn);
    return runtime::Status::kSuccess;
  }

  if (ctx.best_tile != 0 &&
      ctx.tuned_m == static_cast<std::size_t>(mm) &&
      ctx.tuned_k == static_cast<std::size_t>(kk) &&
      ctx.tuned_n == static_cast<std::size_t>(nn)) {
    return runtime::Status::kSuccess;
  }

  const uint32_t candidates[] = {16, 32, 12, 8};
  double best_ms = std::numeric_limits<double>::max();
  uint32_t best_tile = 16;
  bool best_use_mps = false;

  for (uint32_t tile : candidates) {
    id<MTLComputePipelineState> pipe = pipelineForTile(ctx, tile);
    if (!pipe || pipe.maxTotalThreadsPerThreadgroup < (tile * tile)) {
      continue;
    }

    runtime::Status warm = runMatMulKernel(ctx, tile, mm, kk, nn, true);
    if (warm != runtime::Status::kSuccess) {
      continue;
    }

    double sum_ms = 0.0;
    bool ok = true;
    for (int rep = 0; rep < 3; ++rep) {
      auto start = std::chrono::high_resolution_clock::now();
      runtime::Status st = runMatMulKernel(ctx, tile, mm, kk, nn, true);
      auto end = std::chrono::high_resolution_clock::now();
      if (st != runtime::Status::kSuccess) {
        ok = false;
        break;
      }
      std::chrono::duration<double, std::milli> elapsed = end - start;
      sum_ms += elapsed.count();
    }
    if (!ok) {
      continue;
    }

    const double avg_ms = sum_ms / 3.0;
    if (avg_ms < best_ms) {
      best_ms = avg_ms;
      best_tile = tile;
      best_use_mps = false;
    }
  }

  // 하드웨어 최적 GEMM(MPS) 경로도 같은 조건에서 비교해 더 빠르면 채택한다.
  {
    runtime::Status warm = runMatMulMps(ctx, mm, kk, nn, true);
    if (warm == runtime::Status::kSuccess) {
      double sum_ms = 0.0;
      bool ok = true;
      for (int rep = 0; rep < 3; ++rep) {
        auto start = std::chrono::high_resolution_clock::now();
        runtime::Status st = runMatMulMps(ctx, mm, kk, nn, true);
        auto end = std::chrono::high_resolution_clock::now();
        if (st != runtime::Status::kSuccess) {
          ok = false;
          break;
        }
        std::chrono::duration<double, std::milli> elapsed = end - start;
        sum_ms += elapsed.count();
      }
      if (ok) {
        const double avg_ms = sum_ms / 3.0;
        if (avg_ms < best_ms) {
          best_ms = avg_ms;
          best_use_mps = true;
        }
      }
    }
  }

  ctx.best_tile = best_tile;
  ctx.use_mps = best_use_mps;
  ctx.tuned_m = static_cast<std::size_t>(mm);
  ctx.tuned_k = static_cast<std::size_t>(kk);
  ctx.tuned_n = static_cast<std::size_t>(nn);
  ctx.tune_cache[key] = MatMulTuneValue{best_tile, best_use_mps};
  ctx.tune_cache_dirty = true;
  saveMatMulTuneCacheIfDirty(ctx);
  return runtime::Status::kSuccess;
}

}  // namespace

runtime::Status matMulMetal(
    const float* a,
    const float* b,
    float* out,
    std::size_t m,
    std::size_t k,
    std::size_t n) {
  return matMulMetalWithPolicy(a, b, out, m, k, n, true, true, true, true);
}

runtime::Status matMulMetalWithPolicy(
    const float* a,
    const float* b,
    float* out,
    std::size_t m,
    std::size_t k,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  if ((upload_a && a == nullptr) || (upload_b && b == nullptr) || (download_out && out == nullptr) || m == 0 ||
      k == 0 || n == 0) {
    return runtime::Status::kInvalidValue;
  }

  @autoreleasepool {
    runtime::Status st = ensureContext();
    if (st != runtime::Status::kSuccess) {
      return st;
    }

    const std::size_t bytes_a = m * k * sizeof(float);
    const std::size_t bytes_b = k * n * sizeof(float);
    const std::size_t bytes_out = m * n * sizeof(float);
    st = ensureBuffers(bytes_a, bytes_b, bytes_out);
    if (st != runtime::Status::kSuccess) {
      return st;
    }

    MetalMatMulContext& ctx = getContext();
    const bool effective_sync = synchronize || download_out;
    const bool force_wait_for_capacity = (!effective_sync && (ctx.async_inflight + 1 >= kMatMulMaxAsyncInflight));
    const bool wait_for_completion = effective_sync || force_wait_for_capacity;

    if (upload_a || upload_b) {
      st = waitForQueueIdle(ctx);
      if (st != runtime::Status::kSuccess) {
        return st;
      }
    }

    if (upload_a) {
      std::memcpy(ctx.buf_a.contents, a, bytes_a);
    }
    if (upload_b) {
      std::memcpy(ctx.buf_b.contents, b, bytes_b);
    }

    uint32_t mm = static_cast<uint32_t>(m);
    uint32_t kk = static_cast<uint32_t>(k);
    uint32_t nn = static_cast<uint32_t>(n);

    if (mm <= 2) {
      MatMulSmallBatchMode mode = matMulSmallBatchMode();
      bool use_small = false;
      if (mode == MatMulSmallBatchMode::kForceKernel) {
        use_small = true;
      } else if (mode == MatMulSmallBatchMode::kForceMps) {
        use_small = false;
      } else {
        st = autoTuneSmallBatchImpl(ctx, mm, kk, nn, &use_small);
        if (st != runtime::Status::kSuccess) {
          return st;
        }
      }
      st = use_small ? runMatMulSmallBatch(ctx, mm, kk, nn, wait_for_completion)
                     : runMatMulMps(ctx, mm, kk, nn, wait_for_completion);
    } else {
      st = autoTuneTile(ctx, mm, kk, nn);
      if (st != runtime::Status::kSuccess) {
        return st;
      }
      st = ctx.use_mps ? runMatMulMps(ctx, mm, kk, nn, wait_for_completion)
                       : runMatMulKernel(ctx, ctx.best_tile, mm, kk, nn, wait_for_completion);
    }
    if (st != runtime::Status::kSuccess) {
      return st;
    }

    if (wait_for_completion) {
      ctx.async_inflight = 0;
    } else {
      ctx.async_inflight += 1;
    }

    if (download_out) {
      std::memcpy(out, ctx.buf_out.contents, bytes_out);
    }

    return runtime::Status::kSuccess;
  }
}

runtime::Status matMulMetal(
    const double* a,
    const double* b,
    double* out,
    std::size_t m,
    std::size_t k,
    std::size_t n) {
  (void)a;
  (void)b;
  (void)out;
  (void)m;
  (void)k;
  (void)n;
  return runtime::Status::kNotSupported;
}

runtime::Status matMulMetalWithPolicy(
    const double* a,
    const double* b,
    double* out,
    std::size_t m,
    std::size_t k,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)a;
  (void)b;
  (void)out;
  (void)m;
  (void)k;
  (void)n;
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return runtime::Status::kNotSupported;
}

runtime::Status matMulMetal(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t m,
    std::size_t k,
    std::size_t n) {
  (void)a;
  (void)b;
  (void)out;
  (void)m;
  (void)k;
  (void)n;
  return runtime::Status::kNotSupported;
}

runtime::Status matMulMetalWithPolicy(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t m,
    std::size_t k,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)a;
  (void)b;
  (void)out;
  (void)m;
  (void)k;
  (void)n;
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return runtime::Status::kNotSupported;
}

}  // namespace cudajun::detail

#else

namespace cudajun::detail {

runtime::Status matMulMetal(
    const float* a,
    const float* b,
    float* out,
    std::size_t m,
    std::size_t k,
    std::size_t n) {
  (void)a;
  (void)b;
  (void)out;
  (void)m;
  (void)k;
  (void)n;
  return runtime::Status::kNotSupported;
}

runtime::Status matMulMetalWithPolicy(
    const float* a,
    const float* b,
    float* out,
    std::size_t m,
    std::size_t k,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)a;
  (void)b;
  (void)out;
  (void)m;
  (void)k;
  (void)n;
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return runtime::Status::kNotSupported;
}

runtime::Status matMulMetal(
    const double* a,
    const double* b,
    double* out,
    std::size_t m,
    std::size_t k,
    std::size_t n) {
  (void)a;
  (void)b;
  (void)out;
  (void)m;
  (void)k;
  (void)n;
  return runtime::Status::kNotSupported;
}

runtime::Status matMulMetalWithPolicy(
    const double* a,
    const double* b,
    double* out,
    std::size_t m,
    std::size_t k,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)a;
  (void)b;
  (void)out;
  (void)m;
  (void)k;
  (void)n;
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return runtime::Status::kNotSupported;
}

runtime::Status matMulMetal(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t m,
    std::size_t k,
    std::size_t n) {
  (void)a;
  (void)b;
  (void)out;
  (void)m;
  (void)k;
  (void)n;
  return runtime::Status::kNotSupported;
}

runtime::Status matMulMetalWithPolicy(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t m,
    std::size_t k,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)a;
  (void)b;
  (void)out;
  (void)m;
  (void)k;
  (void)n;
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return runtime::Status::kNotSupported;
}

}  // namespace cudajun::detail

#endif
