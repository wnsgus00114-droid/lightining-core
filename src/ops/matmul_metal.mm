#include "lightning_core/core/detail/ops_backend.hpp"

#if defined(CJ_HAS_METAL) && CJ_HAS_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace lightning_core::detail {

namespace {

constexpr uint32_t kMatMulMaxAsyncInflight = 256;
constexpr const char* kTuneCacheHeaderTag = "#lc_tune_cache";
constexpr uint32_t kTuneCacheFormatVersion = 2;
constexpr uint32_t kTuneCacheMinSupportedVersion = 1;
constexpr uint32_t kTuneCacheMaxSupportedVersion = 2;

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
  bool use_vec2_kernel = false;
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
  id<MTLComputePipelineState> pipeline_t16x2 = nil;
  id<MTLComputePipelineState> pipeline_t32 = nil;
  id<MTLBuffer> buf_a = nil;
  id<MTLBuffer> buf_b = nil;
  id<MTLBuffer> buf_out = nil;
  std::size_t cap_a = 0;
  std::size_t cap_b = 0;
  std::size_t cap_out = 0;
  // Private-storage mirrors for high-throughput MPS batch path.
  id<MTLBuffer> mps_buf_a = nil;
  id<MTLBuffer> mps_buf_b = nil;
  id<MTLBuffer> mps_buf_out = nil;
  std::size_t mps_cap_a = 0;
  std::size_t mps_cap_b = 0;
  std::size_t mps_cap_out = 0;
  bool mps_a_dirty = true;
  bool mps_b_dirty = true;
  // Reduced-precision resident batch path (fp16 I/O buffers).
  id<MTLBuffer> mps_buf_a_f16 = nil;
  id<MTLBuffer> mps_buf_b_f16 = nil;
  id<MTLBuffer> mps_buf_out_f16 = nil;
  std::size_t mps_cap_a_f16 = 0;
  std::size_t mps_cap_b_f16 = 0;
  std::size_t mps_cap_out_f16 = 0;
  bool mps_f16_a_dirty = true;
  bool mps_f16_b_dirty = true;
  std::size_t tuned_m = 0;
  std::size_t tuned_k = 0;
  std::size_t tuned_n = 0;
  uint32_t best_tile = 0;
  bool use_mps = false;
  bool use_vec2 = false;
  uint32_t mps_m = 0;
  uint32_t mps_k = 0;
  uint32_t mps_n = 0;
  MPSMatrixMultiplication* mps_op = nil;
  MPSMatrix* mps_a_mat = nil;
  MPSMatrix* mps_b_mat = nil;
  MPSMatrix* mps_c_mat = nil;
  uint32_t mps_batch_m = 0;
  uint32_t mps_batch_k = 0;
  uint32_t mps_batch_n = 0;
  MPSMatrix* mps_a_batch_mat = nil;
  MPSMatrix* mps_b_batch_mat = nil;
  MPSMatrix* mps_c_batch_mat = nil;
  MPSMatrixMultiplication* mps_op_f16 = nil;
  uint32_t mps_f16_m = 0;
  uint32_t mps_f16_k = 0;
  uint32_t mps_f16_n = 0;
  MPSMatrix* mps_a_f16_batch_mat = nil;
  MPSMatrix* mps_b_f16_batch_mat = nil;
  MPSMatrix* mps_c_f16_batch_mat = nil;
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
  return std::string(".lightning_core_matmul_tune_cache.csv");
}

std::vector<std::string> splitCsvColumns(const std::string& line) {
  std::stringstream ss(line);
  std::string tok;
  std::vector<std::string> cols;
  while (std::getline(ss, tok, ',')) {
    cols.push_back(tok);
  }
  return cols;
}

bool parseTuneCacheHeaderVersion(
    const std::vector<std::string>& cols,
    const char* expected_kind,
    uint32_t* version_out) {
  if (version_out == nullptr || expected_kind == nullptr) {
    return false;
  }
  if (cols.size() != 3 || cols[0] != kTuneCacheHeaderTag || cols[1] != expected_kind) {
    return false;
  }
  try {
    const uint32_t version = static_cast<uint32_t>(std::stoul(cols[2]));
    *version_out = version;
    return true;
  } catch (...) {
    return false;
  }
}

enum class MatMulSmallBatchMode {
  kAuto,
  kForceKernel,
  kForceMps,
};

bool matMulDisablePromotedBuckets() {
  const char* env = std::getenv("CJ_MATMUL_DISABLE_PROMOTED_BUCKETS");
  if (env == nullptr || env[0] == '\0') {
    return false;
  }
  return (std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0 && std::strcmp(env, "off") != 0);
}

bool promotedShapeBucketPolicy(
    uint32_t mm,
    uint32_t kk,
    uint32_t nn,
    uint32_t* best_tile_out,
    bool* use_mps_out,
    bool* use_vec2_out) {
  if (best_tile_out == nullptr || use_mps_out == nullptr || use_vec2_out == nullptr) {
    return false;
  }
  if (matMulDisablePromotedBuckets()) {
    return false;
  }

  const uint32_t max_dim = std::max(mm, std::max(kk, nn));
  const bool square_like =
      (mm > 0 && kk > 0 && nn > 0 &&
       (std::max(mm, std::max(kk, nn)) - std::min(mm, std::min(kk, nn)) <= 512));

  // Promoted defaults from the latest large-gemm sweep (2026-03-30).
  // Large square-like shapes remain most stable on MPS in fresh-cache runs.
  if (square_like && max_dim >= 3072) {
    *best_tile_out = 16;
    *use_mps_out = true;
    *use_vec2_out = false;
    return true;
  }
  if (square_like && max_dim >= 2048) {
    *best_tile_out = 16;
    *use_mps_out = true;
    *use_vec2_out = false;
    return true;
  }
  if (square_like && max_dim >= 1536) {
    *best_tile_out = 16;
    *use_mps_out = true;
    *use_vec2_out = false;
    return true;
  }
  if (square_like && max_dim >= 1024) {
    *best_tile_out = 16;
    *use_mps_out = true;
    *use_vec2_out = false;
    return true;
  }

  // Wide-output case (e.g., 4096x1024x4096) favored vec2 kernel path in sweep.
  if (mm >= 3072 && nn >= 3072 && kk <= 1536) {
    *best_tile_out = 16;
    *use_mps_out = false;
    *use_vec2_out = true;
    return true;
  }

  return false;
}

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

double matMulMpsHysteresisPct() {
  const char* env = std::getenv("CJ_MATMUL_MPS_HYST_PCT");
  if (env == nullptr || env[0] == '\0') {
    return 0.0;
  }
  char* end = nullptr;
  double v = std::strtod(env, &end);
  if (end == env || *end != '\0' || v < 0.0) {
    return 0.0;
  }
  return v;
}

bool matMulPreferMpsOnLarge() {
  const char* env = std::getenv("CJ_MATMUL_PREFER_MPS_ON_LARGE");
  if (env == nullptr || env[0] == '\0') {
    return true;
  }
  return (std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0 && std::strcmp(env, "off") != 0);
}

bool matMulTryKernelOnLarge() {
  const char* env = std::getenv("CJ_MATMUL_TRY_KERNEL_ON_LARGE");
  if (env == nullptr || env[0] == '\0') {
    return true;
  }
  return (std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0 && std::strcmp(env, "off") != 0);
}

bool matMulSkipApiValidation() {
  const char* env = std::getenv("CJ_MATMUL_SKIP_API_VALIDATION");
  if (env == nullptr || env[0] == '\0') {
    return true;
  }
  return (std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0 && std::strcmp(env, "off") != 0);
}

bool matMulAllowReducedPrecision() {
  const char* env = std::getenv("CJ_MATMUL_ALLOW_REDUCED_PRECISION");
  if (env == nullptr || env[0] == '\0') {
    return true;
  }
  return (std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0 && std::strcmp(env, "off") != 0);
}

bool matMulBatchFp16Enabled() {
  const char* env = std::getenv("CJ_MATMUL_BATCH_FP16");
  if (env == nullptr || env[0] == '\0') {
    return true;
  }
  return (std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0 && std::strcmp(env, "off") != 0);
}

MPSKernelOptions matMulMpsOptions() {
  MPSKernelOptions opts = MPSKernelOptionsNone;
  if (matMulSkipApiValidation()) {
    opts = static_cast<MPSKernelOptions>(opts | MPSKernelOptionsSkipAPIValidation);
  }
  if (matMulAllowReducedPrecision()) {
    opts = static_cast<MPSKernelOptions>(opts | MPSKernelOptionsAllowReducedPrecision);
  }
  return opts;
}

uint16_t floatToHalfBits(float v) {
  uint32_t x = 0;
  std::memcpy(&x, &v, sizeof(x));

  const uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000U);
  const uint32_t exp = (x >> 23) & 0xFFU;
  const uint32_t mant = x & 0x7FFFFFU;

  if (exp == 0xFFU) {
    if (mant != 0) {
      return static_cast<uint16_t>(sign | 0x7E00U);
    }
    return static_cast<uint16_t>(sign | 0x7C00U);
  }

  const int32_t half_exp_signed = static_cast<int32_t>(exp) - 127 + 15;
  if (half_exp_signed >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00U);
  }
  if (half_exp_signed <= 0) {
    if (half_exp_signed < -10) {
      return sign;
    }
    uint32_t mantissa = mant | 0x800000U;
    const uint32_t shift = static_cast<uint32_t>(14 - half_exp_signed);
    uint32_t half_mant = mantissa >> shift;
    const uint32_t round_bit = (mantissa >> (shift - 1U)) & 1U;
    const uint32_t sticky = mantissa & ((1U << (shift - 1U)) - 1U);
    if (round_bit != 0U && (sticky != 0U || (half_mant & 1U) != 0U)) {
      half_mant += 1U;
    }
    return static_cast<uint16_t>(sign | static_cast<uint16_t>(half_mant));
  }

  uint32_t half_exp = static_cast<uint32_t>(half_exp_signed) << 10U;
  uint32_t half_mant = mant >> 13U;
  const uint32_t round_bits = mant & 0x1FFFU;
  if (round_bits > 0x1000U || (round_bits == 0x1000U && (half_mant & 1U) != 0U)) {
    half_mant += 1U;
    if (half_mant == 0x400U) {
      half_mant = 0;
      half_exp += 0x400U;
      if (half_exp >= 0x7C00U) {
        return static_cast<uint16_t>(sign | 0x7C00U);
      }
    }
  }
  return static_cast<uint16_t>(sign | static_cast<uint16_t>(half_exp | half_mant));
}

void convertFloatToHalf(const float* src, uint16_t* dst, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    dst[i] = floatToHalfBits(src[i]);
  }
}

std::uint64_t matMulLargeOpsThreshold() {
  const char* env = std::getenv("CJ_MATMUL_LARGE_OPS_THRESHOLD");
  if (env == nullptr || env[0] == '\0') {
    return 4000000000ULL;
  }
  char* end = nullptr;
  unsigned long long v = std::strtoull(env, &end, 10);
  if (end == env || *end != '\0' || v == 0ULL) {
    return 4000000000ULL;
  }
  return static_cast<std::uint64_t>(v);
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
  return std::string(".lightning_core_matmul_mle2_tune_cache") + suffix + ".csv";
}

void saveMatMulTuneCacheIfDirty(MetalMatMulContext& ctx) {
  if (!ctx.tune_cache_loaded || !ctx.tune_cache_dirty || ctx.tune_cache_path.empty()) {
    return;
  }

  std::ofstream ofs(ctx.tune_cache_path, std::ios::out | std::ios::trunc);
  if (!ofs.is_open()) {
    return;
  }
  ofs << kTuneCacheHeaderTag << ",matmul," << kTuneCacheFormatVersion << "\n";
  ofs << "#columns,m,k,n,best_tile,use_mps,use_vec2_kernel\n";
  for (const auto& kv : ctx.tune_cache) {
    ofs << kv.first.m << "," << kv.first.k << "," << kv.first.n << "," << kv.second.best_tile << ","
      << static_cast<unsigned>(kv.second.use_mps ? 1 : 0) << ","
      << static_cast<unsigned>(kv.second.use_vec2_kernel ? 1 : 0) << "\n";
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
  ofs << kTuneCacheHeaderTag << ",matmul_small," << kTuneCacheFormatVersion << "\n";
  ofs << "#columns,m,k,n,use_small_kernel\n";
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

  bool strict_layout_v2 = false;
  bool unsupported_header = false;
  std::string line;
  while (std::getline(ifs, line)) {
    if (line.empty()) {
      continue;
    }
    std::vector<std::string> cols = splitCsvColumns(line);
    if (cols.empty()) {
      continue;
    }
    if (!cols[0].empty() && cols[0][0] == '#') {
      if (cols[0] == kTuneCacheHeaderTag) {
        uint32_t version = 0;
        if (!parseTuneCacheHeaderVersion(cols, "matmul", &version) ||
            version < kTuneCacheMinSupportedVersion ||
            version > kTuneCacheMaxSupportedVersion) {
          unsupported_header = true;
          break;
        }
        strict_layout_v2 = (version >= 2);
      }
      continue;
    }
    if (strict_layout_v2) {
      if (cols.size() != 6) {
        continue;
      }
    } else if (cols.size() != 5 && cols.size() != 6) {
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
      val.use_vec2_kernel = (cols.size() >= 6) ? (std::stoi(cols[5]) != 0) : false;
      ctx.tune_cache[key] = val;
    } catch (...) {
      // Ignore malformed lines.
    }
  }
  if (unsupported_header) {
    ctx.tune_cache.clear();
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

  bool strict_layout_v2 = false;
  bool unsupported_header = false;
  std::string line;
  while (std::getline(ifs, line)) {
    if (line.empty()) {
      continue;
    }
    std::vector<std::string> cols = splitCsvColumns(line);
    if (cols.empty()) {
      continue;
    }
    if (!cols[0].empty() && cols[0][0] == '#') {
      if (cols[0] == kTuneCacheHeaderTag) {
        uint32_t version = 0;
        if (!parseTuneCacheHeaderVersion(cols, "matmul_small", &version) ||
            version < kTuneCacheMinSupportedVersion ||
            version > kTuneCacheMaxSupportedVersion) {
          unsupported_header = true;
          break;
        }
        strict_layout_v2 = (version >= 2);
      }
      continue;
    }
    if (strict_layout_v2) {
      if (cols.size() != 4) {
        continue;
      }
    } else if (cols.size() != 4) {
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
  if (unsupported_header) {
    ctx.small_tune_cache.clear();
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

kernel void matmul_f32_t16x2(
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
  uint col0 = tgid.x * (TILE * 2) + tid.x;
  uint col1 = col0 + TILE;

  threadgroup float a_tile[TILE][TILE];
  threadgroup float b_tile0[TILE][TILE];
  threadgroup float b_tile1[TILE][TILE];

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  uint num_tiles = (k + TILE - 1) / TILE;
  for (uint t = 0; t < num_tiles; ++t) {
    uint a_col = t * TILE + tid.x;
    uint b_row = t * TILE + tid.y;

    a_tile[tid.y][tid.x] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    b_tile0[tid.y][tid.x] = (b_row < k && col0 < n) ? b[b_row * n + col0] : 0.0f;
    b_tile1[tid.y][tid.x] = (b_row < k && col1 < n) ? b[b_row * n + col1] : 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = 0; p < TILE; ++p) {
      float av = a_tile[tid.y][p];
      acc0 += av * b_tile0[p][tid.x];
      acc1 += av * b_tile1[p][tid.x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (row < m && col0 < n) {
    out[row * n + col0] = acc0;
  }
  if (row < m && col1 < n) {
    out[row * n + col1] = acc1;
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
      ctx.pipeline_t8 != nil && ctx.pipeline_t12 != nil && ctx.pipeline_t16 != nil &&
      ctx.pipeline_t16x2 != nil && ctx.pipeline_t32 != nil) {
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
  id<MTLFunction> fn16x2 = [lib newFunctionWithName:@"matmul_f32_t16x2"];
  id<MTLFunction> fn32 = [lib newFunctionWithName:@"matmul_f32_t32"];
  if (!fn_m1 || !fn_m2 || !fn8 || !fn12 || !fn16 || !fn16x2 || !fn32) {
    return runtime::Status::kDriverError;
  }

  ctx.pipeline_m1 = [ctx.device newComputePipelineStateWithFunction:fn_m1 error:&err];
  ctx.pipeline_m2 = [ctx.device newComputePipelineStateWithFunction:fn_m2 error:&err];
  ctx.pipeline_t8 = [ctx.device newComputePipelineStateWithFunction:fn8 error:&err];
  ctx.pipeline_t12 = [ctx.device newComputePipelineStateWithFunction:fn12 error:&err];
  ctx.pipeline_t16 = [ctx.device newComputePipelineStateWithFunction:fn16 error:&err];
  ctx.pipeline_t16x2 = [ctx.device newComputePipelineStateWithFunction:fn16x2 error:&err];
  ctx.pipeline_t32 = [ctx.device newComputePipelineStateWithFunction:fn32 error:&err];
  if (!ctx.pipeline_m1 || !ctx.pipeline_m2 ||
      !ctx.pipeline_t8 || !ctx.pipeline_t12 || !ctx.pipeline_t16 || !ctx.pipeline_t16x2 || !ctx.pipeline_t32) {
    return runtime::Status::kDriverError;
  }

  loadMatMulTuneCacheIfNeeded(ctx);
  loadMatMulSmallTuneCacheIfNeeded(ctx);

  return runtime::Status::kSuccess;
}

runtime::Status ensureBuffers(std::size_t bytes_a, std::size_t bytes_b, std::size_t bytes_out) {
  MetalMatMulContext& ctx = getContext();
  bool shared_resized = false;
  bool mps_private_resized = false;

  if (ctx.cap_a < bytes_a) {
    ctx.buf_a = [ctx.device newBufferWithLength:bytes_a options:MTLResourceStorageModeShared];
    if (!ctx.buf_a) {
      return runtime::Status::kOutOfMemory;
    }
    ctx.cap_a = bytes_a;
    shared_resized = true;
  }

  if (ctx.cap_b < bytes_b) {
    ctx.buf_b = [ctx.device newBufferWithLength:bytes_b options:MTLResourceStorageModeShared];
    if (!ctx.buf_b) {
      return runtime::Status::kOutOfMemory;
    }
    ctx.cap_b = bytes_b;
    shared_resized = true;
  }

  if (ctx.cap_out < bytes_out) {
    ctx.buf_out = [ctx.device newBufferWithLength:bytes_out options:MTLResourceStorageModeShared];
    if (!ctx.buf_out) {
      return runtime::Status::kOutOfMemory;
    }
    ctx.cap_out = bytes_out;
    shared_resized = true;
  }

  if (ctx.mps_cap_a < bytes_a) {
    ctx.mps_buf_a = [ctx.device newBufferWithLength:bytes_a options:MTLResourceStorageModePrivate];
    if (!ctx.mps_buf_a) {
      return runtime::Status::kOutOfMemory;
    }
    ctx.mps_cap_a = bytes_a;
    mps_private_resized = true;
    ctx.mps_a_dirty = true;
  }

  if (ctx.mps_cap_b < bytes_b) {
    ctx.mps_buf_b = [ctx.device newBufferWithLength:bytes_b options:MTLResourceStorageModePrivate];
    if (!ctx.mps_buf_b) {
      return runtime::Status::kOutOfMemory;
    }
    ctx.mps_cap_b = bytes_b;
    mps_private_resized = true;
    ctx.mps_b_dirty = true;
  }

  if (ctx.mps_cap_out < bytes_out) {
    ctx.mps_buf_out = [ctx.device newBufferWithLength:bytes_out options:MTLResourceStorageModePrivate];
    if (!ctx.mps_buf_out) {
      return runtime::Status::kOutOfMemory;
    }
    ctx.mps_cap_out = bytes_out;
    mps_private_resized = true;
  }

  if (shared_resized) {
    // MPSMatrix objects are tied to buffer identity/shape and must be rebuilt when backing buffers grow.
    ctx.mps_a_mat = nil;
    ctx.mps_b_mat = nil;
    ctx.mps_c_mat = nil;
    ctx.mps_m = 0;
    ctx.mps_k = 0;
    ctx.mps_n = 0;
    ctx.mps_f16_a_dirty = true;
    ctx.mps_f16_b_dirty = true;
  }

  if (mps_private_resized) {
    ctx.mps_a_batch_mat = nil;
    ctx.mps_b_batch_mat = nil;
    ctx.mps_c_batch_mat = nil;
    ctx.mps_batch_m = 0;
    ctx.mps_batch_k = 0;
    ctx.mps_batch_n = 0;
  }

  return runtime::Status::kSuccess;
}

runtime::Status ensureFp16Buffers(std::size_t elems_a, std::size_t elems_b, std::size_t elems_out) {
  MetalMatMulContext& ctx = getContext();
  bool resized = false;

  const std::size_t bytes_a = elems_a * sizeof(uint16_t);
  const std::size_t bytes_b = elems_b * sizeof(uint16_t);
  const std::size_t bytes_out = elems_out * sizeof(uint16_t);

  if (ctx.mps_cap_a_f16 < bytes_a) {
    ctx.mps_buf_a_f16 = [ctx.device newBufferWithLength:bytes_a options:MTLResourceStorageModeShared];
    if (!ctx.mps_buf_a_f16) {
      return runtime::Status::kOutOfMemory;
    }
    ctx.mps_cap_a_f16 = bytes_a;
    resized = true;
    ctx.mps_f16_a_dirty = true;
  }

  if (ctx.mps_cap_b_f16 < bytes_b) {
    ctx.mps_buf_b_f16 = [ctx.device newBufferWithLength:bytes_b options:MTLResourceStorageModeShared];
    if (!ctx.mps_buf_b_f16) {
      return runtime::Status::kOutOfMemory;
    }
    ctx.mps_cap_b_f16 = bytes_b;
    resized = true;
    ctx.mps_f16_b_dirty = true;
  }

  if (ctx.mps_cap_out_f16 < bytes_out) {
    ctx.mps_buf_out_f16 = [ctx.device newBufferWithLength:bytes_out options:MTLResourceStorageModeShared];
    if (!ctx.mps_buf_out_f16) {
      return runtime::Status::kOutOfMemory;
    }
    ctx.mps_cap_out_f16 = bytes_out;
    resized = true;
  }

  if (resized) {
    ctx.mps_a_f16_batch_mat = nil;
    ctx.mps_b_f16_batch_mat = nil;
    ctx.mps_c_f16_batch_mat = nil;
    ctx.mps_f16_m = 0;
    ctx.mps_f16_k = 0;
    ctx.mps_f16_n = 0;
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

runtime::Status runMatMulKernelVec2(
    MetalMatMulContext& ctx,
    uint32_t mm,
    uint32_t kk,
    uint32_t nn,
    bool wait_for_completion) {
  id<MTLComputePipelineState> pipe = ctx.pipeline_t16x2;
  if (!pipe) {
    return runtime::Status::kDriverError;
  }
  constexpr uint32_t tile = 16;
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
  MTLSize tgs = MTLSizeMake((nn + (tile * 2) - 1) / (tile * 2), (mm + tile - 1) / tile, 1);
  [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tg];
  [enc endEncoding];

  [cmd commit];
  if (wait_for_completion) {
    [cmd waitUntilCompleted];
    return cmd.status == MTLCommandBufferStatusCompleted ? runtime::Status::kSuccess : runtime::Status::kUnknown;
  }
  return runtime::Status::kSuccess;
}

runtime::Status runMatMulKernelBatch(
    MetalMatMulContext& ctx,
    uint32_t tile,
    uint32_t mm,
    uint32_t kk,
    uint32_t nn,
    uint32_t repeat_count,
    bool wait_for_completion) {
  id<MTLComputePipelineState> pipe = pipelineForTile(ctx, tile);
  if (!pipe || repeat_count == 0) {
    return runtime::Status::kInvalidValue;
  }
  if (pipe.maxTotalThreadsPerThreadgroup < (tile * tile)) {
    return runtime::Status::kNotSupported;
  }

  id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
  if (!cmd) {
    return runtime::Status::kDriverError;
  }

  MTLSize tg = MTLSizeMake(tile, tile, 1);
  MTLSize tgs = MTLSizeMake((nn + tile - 1) / tile, (mm + tile - 1) / tile, 1);

  for (uint32_t i = 0; i < repeat_count; ++i) {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) {
      return runtime::Status::kDriverError;
    }
    [enc setComputePipelineState:pipe];
    [enc setBuffer:ctx.buf_a offset:0 atIndex:0];
    [enc setBuffer:ctx.buf_b offset:0 atIndex:1];
    [enc setBuffer:ctx.buf_out offset:0 atIndex:2];
    [enc setBytes:&mm length:sizeof(mm) atIndex:3];
    [enc setBytes:&kk length:sizeof(kk) atIndex:4];
    [enc setBytes:&nn length:sizeof(nn) atIndex:5];
    [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tg];
    [enc endEncoding];
  }

  [cmd commit];
  if (wait_for_completion) {
    [cmd waitUntilCompleted];
    return cmd.status == MTLCommandBufferStatusCompleted ? runtime::Status::kSuccess : runtime::Status::kUnknown;
  }
  return runtime::Status::kSuccess;
}

runtime::Status runMatMulKernelVec2Batch(
    MetalMatMulContext& ctx,
    uint32_t mm,
    uint32_t kk,
    uint32_t nn,
    uint32_t repeat_count,
    bool wait_for_completion) {
  id<MTLComputePipelineState> pipe = ctx.pipeline_t16x2;
  if (!pipe || repeat_count == 0) {
    return runtime::Status::kInvalidValue;
  }
  constexpr uint32_t tile = 16;
  if (pipe.maxTotalThreadsPerThreadgroup < (tile * tile)) {
    return runtime::Status::kNotSupported;
  }

  id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
  if (!cmd) {
    return runtime::Status::kDriverError;
  }

  MTLSize tg = MTLSizeMake(tile, tile, 1);
  MTLSize tgs = MTLSizeMake((nn + (tile * 2) - 1) / (tile * 2), (mm + tile - 1) / tile, 1);

  for (uint32_t i = 0; i < repeat_count; ++i) {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) {
      return runtime::Status::kDriverError;
    }
    [enc setComputePipelineState:pipe];
    [enc setBuffer:ctx.buf_a offset:0 atIndex:0];
    [enc setBuffer:ctx.buf_b offset:0 atIndex:1];
    [enc setBuffer:ctx.buf_out offset:0 atIndex:2];
    [enc setBytes:&mm length:sizeof(mm) atIndex:3];
    [enc setBytes:&kk length:sizeof(kk) atIndex:4];
    [enc setBytes:&nn length:sizeof(nn) atIndex:5];
    [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tg];
    [enc endEncoding];
  }

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
  if (ctx.mps_op == nil || ctx.mps_m != mm || ctx.mps_k != kk || ctx.mps_n != nn ||
      ctx.mps_a_mat == nil || ctx.mps_b_mat == nil || ctx.mps_c_mat == nil) {
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
    ctx.mps_op.options = matMulMpsOptions();
    ctx.mps_m = mm;
    ctx.mps_k = kk;
    ctx.mps_n = nn;

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

    ctx.mps_a_mat = [[MPSMatrix alloc] initWithBuffer:ctx.buf_a descriptor:aDesc];
    ctx.mps_b_mat = [[MPSMatrix alloc] initWithBuffer:ctx.buf_b descriptor:bDesc];
    ctx.mps_c_mat = [[MPSMatrix alloc] initWithBuffer:ctx.buf_out descriptor:cDesc];
  }

  if (!ctx.mps_a_mat || !ctx.mps_b_mat || !ctx.mps_c_mat) {
    return runtime::Status::kDriverError;
  }

  id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
  if (!cmd) {
    return runtime::Status::kDriverError;
  }
  [ctx.mps_op encodeToCommandBuffer:cmd leftMatrix:ctx.mps_a_mat rightMatrix:ctx.mps_b_mat resultMatrix:ctx.mps_c_mat];
  [cmd commit];
  if (wait_for_completion) {
    [cmd waitUntilCompleted];
    return cmd.status == MTLCommandBufferStatusCompleted ? runtime::Status::kSuccess : runtime::Status::kUnknown;
  }
  return runtime::Status::kSuccess;
}

runtime::Status runMatMulMpsBatch(
    MetalMatMulContext& ctx,
    uint32_t mm,
    uint32_t kk,
    uint32_t nn,
    uint32_t repeat_count,
    bool wait_for_completion) {
  if (repeat_count == 0) {
    return runtime::Status::kInvalidValue;
  }
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
    ctx.mps_op.options = matMulMpsOptions();
    ctx.mps_m = mm;
    ctx.mps_k = kk;
    ctx.mps_n = nn;
  }

  if (ctx.mps_buf_a == nil || ctx.mps_buf_b == nil || ctx.mps_buf_out == nil) {
    return runtime::Status::kOutOfMemory;
  }

  if (ctx.mps_a_batch_mat == nil || ctx.mps_b_batch_mat == nil || ctx.mps_c_batch_mat == nil ||
      ctx.mps_batch_m != mm || ctx.mps_batch_k != kk || ctx.mps_batch_n != nn) {
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

    ctx.mps_a_batch_mat = [[MPSMatrix alloc] initWithBuffer:ctx.mps_buf_a descriptor:aDesc];
    ctx.mps_b_batch_mat = [[MPSMatrix alloc] initWithBuffer:ctx.mps_buf_b descriptor:bDesc];
    ctx.mps_c_batch_mat = [[MPSMatrix alloc] initWithBuffer:ctx.mps_buf_out descriptor:cDesc];
    ctx.mps_batch_m = mm;
    ctx.mps_batch_k = kk;
    ctx.mps_batch_n = nn;
  }

  if (ctx.mps_op == nil || ctx.mps_a_batch_mat == nil || ctx.mps_b_batch_mat == nil || ctx.mps_c_batch_mat == nil) {
    return runtime::Status::kDriverError;
  }

  id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
  if (!cmd) {
    return runtime::Status::kDriverError;
  }

  if (ctx.mps_a_dirty || ctx.mps_b_dirty) {
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    if (!blit) {
      return runtime::Status::kDriverError;
    }
    if (ctx.mps_a_dirty) {
      const std::size_t bytes_a = static_cast<std::size_t>(mm) * static_cast<std::size_t>(kk) * sizeof(float);
      [blit copyFromBuffer:ctx.buf_a
              sourceOffset:0
                  toBuffer:ctx.mps_buf_a
         destinationOffset:0
                      size:static_cast<NSUInteger>(bytes_a)];
      ctx.mps_a_dirty = false;
    }
    if (ctx.mps_b_dirty) {
      const std::size_t bytes_b = static_cast<std::size_t>(kk) * static_cast<std::size_t>(nn) * sizeof(float);
      [blit copyFromBuffer:ctx.buf_b
              sourceOffset:0
                  toBuffer:ctx.mps_buf_b
         destinationOffset:0
                      size:static_cast<NSUInteger>(bytes_b)];
      ctx.mps_b_dirty = false;
    }
    [blit endEncoding];
  }

  for (uint32_t i = 0; i < repeat_count; ++i) {
    [ctx.mps_op encodeToCommandBuffer:cmd
                           leftMatrix:ctx.mps_a_batch_mat
                          rightMatrix:ctx.mps_b_batch_mat
                         resultMatrix:ctx.mps_c_batch_mat];
  }
  [cmd commit];
  if (wait_for_completion) {
    [cmd waitUntilCompleted];
    return cmd.status == MTLCommandBufferStatusCompleted ? runtime::Status::kSuccess : runtime::Status::kUnknown;
  }
  return runtime::Status::kSuccess;
}

runtime::Status runMatMulMpsBatchFp16(
    MetalMatMulContext& ctx,
    uint32_t mm,
    uint32_t kk,
    uint32_t nn,
    uint32_t repeat_count,
    bool wait_for_completion) {
  if (repeat_count == 0) {
    return runtime::Status::kInvalidValue;
  }

  runtime::Status st = ensureFp16Buffers(
      static_cast<std::size_t>(mm) * static_cast<std::size_t>(kk),
      static_cast<std::size_t>(kk) * static_cast<std::size_t>(nn),
      static_cast<std::size_t>(mm) * static_cast<std::size_t>(nn));
  if (st != runtime::Status::kSuccess) {
    return st;
  }

  if (ctx.mps_op_f16 == nil || ctx.mps_f16_m != mm || ctx.mps_f16_k != kk || ctx.mps_f16_n != nn) {
    if (ctx.mps_f16_m != mm || ctx.mps_f16_k != kk || ctx.mps_f16_n != nn) {
      ctx.mps_a_f16_batch_mat = nil;
      ctx.mps_b_f16_batch_mat = nil;
      ctx.mps_c_f16_batch_mat = nil;
    }
    ctx.mps_op_f16 = [[MPSMatrixMultiplication alloc] initWithDevice:ctx.device
                                                        transposeLeft:NO
                                                       transposeRight:NO
                                                           resultRows:mm
                                                        resultColumns:nn
                                                      interiorColumns:kk
                                                                alpha:1.0f
                                                                 beta:0.0f];
    if (ctx.mps_op_f16 == nil) {
      return runtime::Status::kNotSupported;
    }
    ctx.mps_op_f16.options = matMulMpsOptions();
    ctx.mps_f16_m = mm;
    ctx.mps_f16_k = kk;
    ctx.mps_f16_n = nn;
  }

  if (ctx.mps_a_f16_batch_mat == nil || ctx.mps_b_f16_batch_mat == nil || ctx.mps_c_f16_batch_mat == nil) {
    MPSMatrixDescriptor* aDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:mm
                                                                        columns:kk
                                                                       rowBytes:kk * sizeof(uint16_t)
                                                                       dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor* bDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:kk
                                                                        columns:nn
                                                                       rowBytes:nn * sizeof(uint16_t)
                                                                       dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor* cDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:mm
                                                                        columns:nn
                                                                       rowBytes:nn * sizeof(uint16_t)
                                                                       dataType:MPSDataTypeFloat16];
    ctx.mps_a_f16_batch_mat = [[MPSMatrix alloc] initWithBuffer:ctx.mps_buf_a_f16 descriptor:aDesc];
    ctx.mps_b_f16_batch_mat = [[MPSMatrix alloc] initWithBuffer:ctx.mps_buf_b_f16 descriptor:bDesc];
    ctx.mps_c_f16_batch_mat = [[MPSMatrix alloc] initWithBuffer:ctx.mps_buf_out_f16 descriptor:cDesc];
  }

  if (ctx.mps_op_f16 == nil || ctx.mps_a_f16_batch_mat == nil || ctx.mps_b_f16_batch_mat == nil ||
      ctx.mps_c_f16_batch_mat == nil || ctx.buf_a == nil || ctx.buf_b == nil) {
    return runtime::Status::kDriverError;
  }

  if (ctx.mps_f16_a_dirty) {
    const std::size_t elems_a = static_cast<std::size_t>(mm) * static_cast<std::size_t>(kk);
    const float* src_a = static_cast<const float*>(ctx.buf_a.contents);
    uint16_t* dst_a = static_cast<uint16_t*>(ctx.mps_buf_a_f16.contents);
    if (src_a == nullptr || dst_a == nullptr) {
      return runtime::Status::kDriverError;
    }
    convertFloatToHalf(src_a, dst_a, elems_a);
    ctx.mps_f16_a_dirty = false;
  }
  if (ctx.mps_f16_b_dirty) {
    const std::size_t elems_b = static_cast<std::size_t>(kk) * static_cast<std::size_t>(nn);
    const float* src_b = static_cast<const float*>(ctx.buf_b.contents);
    uint16_t* dst_b = static_cast<uint16_t*>(ctx.mps_buf_b_f16.contents);
    if (src_b == nullptr || dst_b == nullptr) {
      return runtime::Status::kDriverError;
    }
    convertFloatToHalf(src_b, dst_b, elems_b);
    ctx.mps_f16_b_dirty = false;
  }

  id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
  if (!cmd) {
    return runtime::Status::kDriverError;
  }

  for (uint32_t i = 0; i < repeat_count; ++i) {
    [ctx.mps_op_f16 encodeToCommandBuffer:cmd
                               leftMatrix:ctx.mps_a_f16_batch_mat
                              rightMatrix:ctx.mps_b_f16_batch_mat
                             resultMatrix:ctx.mps_c_f16_batch_mat];
  }
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
    ctx.use_vec2 = cached->second.use_vec2_kernel;
    ctx.tuned_m = static_cast<std::size_t>(mm);
    ctx.tuned_k = static_cast<std::size_t>(kk);
    ctx.tuned_n = static_cast<std::size_t>(nn);
    return runtime::Status::kSuccess;
  }

  {
    uint32_t promoted_tile = 16;
    bool promoted_use_mps = false;
    bool promoted_use_vec2 = false;
    if (promotedShapeBucketPolicy(mm, kk, nn, &promoted_tile, &promoted_use_mps, &promoted_use_vec2)) {
      ctx.best_tile = promoted_tile;
      ctx.use_mps = promoted_use_mps;
      ctx.use_vec2 = promoted_use_vec2;
      ctx.tuned_m = static_cast<std::size_t>(mm);
      ctx.tuned_k = static_cast<std::size_t>(kk);
      ctx.tuned_n = static_cast<std::size_t>(nn);
      ctx.tune_cache[key] = MatMulTuneValue{promoted_tile, promoted_use_mps, promoted_use_vec2};
      ctx.tune_cache_dirty = true;
      saveMatMulTuneCacheIfDirty(ctx);
      return runtime::Status::kSuccess;
    }
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
  bool best_use_vec2 = false;

  const std::uint64_t ops = static_cast<std::uint64_t>(mm) * static_cast<std::uint64_t>(kk) * static_cast<std::uint64_t>(nn);
  const bool is_large_gemm = (ops >= matMulLargeOpsThreshold());
  const auto measure_path_ms = [&](const auto& run_sync, double* out_ms) -> runtime::Status {
    if (out_ms == nullptr) {
      return runtime::Status::kInvalidValue;
    }
    runtime::Status warm = run_sync();
    if (warm != runtime::Status::kSuccess) {
      return warm;
    }
    constexpr int kSamples = 5;
    std::array<double, kSamples> samples{};
    for (int rep = 0; rep < kSamples; ++rep) {
      auto start = std::chrono::high_resolution_clock::now();
      runtime::Status st = run_sync();
      auto end = std::chrono::high_resolution_clock::now();
      if (st != runtime::Status::kSuccess) {
        return st;
      }
      std::chrono::duration<double, std::milli> elapsed = end - start;
      samples[rep] = elapsed.count();
    }
    std::sort(samples.begin(), samples.end());
    *out_ms = samples[kSamples / 2];
    return runtime::Status::kSuccess;
  };

  if (is_large_gemm && matMulPreferMpsOnLarge()) {
    double mps_ms = 0.0;
    runtime::Status st =
        measure_path_ms([&]() { return runMatMulMps(ctx, mm, kk, nn, true); }, &mps_ms);
    if (st == runtime::Status::kSuccess) {
      best_ms = mps_ms;
      best_use_mps = true;
      best_use_vec2 = false;
    }
  }

  const bool prefer_kernel_large =
      (mm >= 2048 && nn >= 2048) || (mm >= 3072 && nn >= 3072 && kk <= 1536);
  const bool evaluate_kernel_candidates =
      (!is_large_gemm || matMulTryKernelOnLarge() || prefer_kernel_large);

  if (evaluate_kernel_candidates) {
    for (uint32_t tile : candidates) {
      id<MTLComputePipelineState> pipe = pipelineForTile(ctx, tile);
      if (!pipe || pipe.maxTotalThreadsPerThreadgroup < (tile * tile)) {
        continue;
      }
      double kernel_ms = 0.0;
      runtime::Status st = measure_path_ms(
          [&]() { return runMatMulKernel(ctx, tile, mm, kk, nn, true); }, &kernel_ms);
      if (st != runtime::Status::kSuccess) {
        continue;
      }
      if (kernel_ms < best_ms) {
        best_ms = kernel_ms;
        best_tile = tile;
        best_use_mps = false;
        best_use_vec2 = false;
      }
    }
  }

  if (evaluate_kernel_candidates && nn >= 2048 && mm >= 1024) {
    double vec2_ms = 0.0;
    runtime::Status st = measure_path_ms(
        [&]() { return runMatMulKernelVec2(ctx, mm, kk, nn, true); }, &vec2_ms);
    if (st == runtime::Status::kSuccess && vec2_ms < best_ms) {
      best_ms = vec2_ms;
      best_tile = 16;
      best_use_mps = false;
      best_use_vec2 = true;
    }
  }

  // 하드웨어 최적 GEMM(MPS) 경로도 같은 조건에서 비교해 더 빠르면 채택한다.
  {
    double mps_ms = 0.0;
    runtime::Status st =
        measure_path_ms([&]() { return runMatMulMps(ctx, mm, kk, nn, true); }, &mps_ms);
    if (st == runtime::Status::kSuccess) {
      const double mps_hysteresis_pct = matMulMpsHysteresisPct();
      const double mps_threshold = best_ms * (1.0 + mps_hysteresis_pct / 100.0);
      if (mps_ms <= mps_threshold) {
        best_ms = mps_ms;
        best_use_mps = true;
        best_use_vec2 = false;
      }
    }
  }

  ctx.best_tile = best_tile;
  ctx.use_mps = best_use_mps;
  ctx.use_vec2 = best_use_vec2;
  ctx.tuned_m = static_cast<std::size_t>(mm);
  ctx.tuned_k = static_cast<std::size_t>(kk);
  ctx.tuned_n = static_cast<std::size_t>(nn);
  ctx.tune_cache[key] = MatMulTuneValue{best_tile, best_use_mps, best_use_vec2};
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

runtime::Status matMulMetalResetTuning() {
  @autoreleasepool {
    runtime::Status st = ensureContext();
    if (st != runtime::Status::kSuccess) {
      return st;
    }
    MetalMatMulContext& ctx = getContext();
    st = waitForQueueIdle(ctx);
    if (st != runtime::Status::kSuccess) {
      return st;
    }

    ctx.best_tile = 0;
    ctx.use_mps = false;
    ctx.use_vec2 = false;
    ctx.tuned_m = 0;
    ctx.tuned_k = 0;
    ctx.tuned_n = 0;

    ctx.tune_cache.clear();
    ctx.tune_cache_loaded = false;
    ctx.tune_cache_dirty = false;
    ctx.tune_cache_path.clear();

    ctx.small_tune_cache.clear();
    ctx.small_tune_cache_loaded = false;
    ctx.small_tune_cache_dirty = false;
    ctx.small_tune_cache_path.clear();

    return runtime::Status::kSuccess;
  }
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
      ctx.mps_a_dirty = true;
      ctx.mps_f16_a_dirty = true;
    }
    if (upload_b) {
      std::memcpy(ctx.buf_b.contents, b, bytes_b);
      ctx.mps_b_dirty = true;
      ctx.mps_f16_b_dirty = true;
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
      if (ctx.use_mps) {
        st = runMatMulMps(ctx, mm, kk, nn, wait_for_completion);
      } else if (ctx.use_vec2) {
        st = runMatMulKernelVec2(ctx, mm, kk, nn, wait_for_completion);
      } else {
        st = runMatMulKernel(ctx, ctx.best_tile, mm, kk, nn, wait_for_completion);
      }
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

runtime::Status matMulMetalWithPolicyBatched(
    const float* a,
    const float* b,
    float* out,
    std::size_t m,
    std::size_t k,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize,
    std::size_t repeat_count) {
  if ((upload_a && a == nullptr) || (upload_b && b == nullptr) || (download_out && out == nullptr) || m == 0 ||
      k == 0 || n == 0 || repeat_count == 0) {
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
    const bool force_wait_for_capacity =
        (!effective_sync && (ctx.async_inflight + static_cast<uint32_t>(repeat_count) >= kMatMulMaxAsyncInflight));
    const bool wait_for_completion = effective_sync || force_wait_for_capacity;

    if (upload_a || upload_b) {
      st = waitForQueueIdle(ctx);
      if (st != runtime::Status::kSuccess) {
        return st;
      }
    }

    if (upload_a) {
      std::memcpy(ctx.buf_a.contents, a, bytes_a);
      ctx.mps_a_dirty = true;
      ctx.mps_f16_a_dirty = true;
    }
    if (upload_b) {
      std::memcpy(ctx.buf_b.contents, b, bytes_b);
      ctx.mps_b_dirty = true;
      ctx.mps_f16_b_dirty = true;
    }

    uint32_t mm = static_cast<uint32_t>(m);
    uint32_t kk = static_cast<uint32_t>(k);
    uint32_t nn = static_cast<uint32_t>(n);
    uint32_t repeat_u32 = static_cast<uint32_t>(repeat_count > static_cast<std::size_t>(UINT32_MAX) ? UINT32_MAX : repeat_count);

    if (mm <= 2) {
      for (uint32_t i = 0; i < repeat_u32; ++i) {
        st = runMatMulSmallBatch(ctx, mm, kk, nn, wait_for_completion && (i + 1 == repeat_u32));
        if (st != runtime::Status::kSuccess) {
          return st;
        }
      }
    } else {
      bool used_mps_batch_path = false;
      bool used_fp16_batch_path = false;
      st = autoTuneTile(ctx, mm, kk, nn);
      if (st != runtime::Status::kSuccess) {
        return st;
      }
      if (ctx.use_mps) {
        used_mps_batch_path = true;
        const bool try_fp16_batch =
            matMulBatchFp16Enabled() &&
            !download_out &&
            repeat_u32 >= 4 &&
            mm >= 256 && kk >= 256 && nn >= 256;
        if (try_fp16_batch) {
          runtime::Status fp16_st = runMatMulMpsBatchFp16(ctx, mm, kk, nn, repeat_u32, wait_for_completion);
          if (fp16_st == runtime::Status::kSuccess) {
            st = runtime::Status::kSuccess;
            used_fp16_batch_path = true;
          } else {
            st = runMatMulMpsBatch(ctx, mm, kk, nn, repeat_u32, wait_for_completion);
          }
        } else {
          st = runMatMulMpsBatch(ctx, mm, kk, nn, repeat_u32, wait_for_completion);
        }
      } else if (ctx.use_vec2) {
        st = runMatMulKernelVec2Batch(ctx, mm, kk, nn, repeat_u32, wait_for_completion);
      } else {
        st = runMatMulKernelBatch(ctx, ctx.best_tile, mm, kk, nn, repeat_u32, wait_for_completion);
      }
      if (st != runtime::Status::kSuccess) {
        return st;
      }

      if (download_out && used_mps_batch_path && !used_fp16_batch_path) {
        if (ctx.mps_buf_out == nil || ctx.buf_out == nil) {
          return runtime::Status::kDriverError;
        }
        id<MTLCommandBuffer> copy_cmd = [ctx.queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [copy_cmd blitCommandEncoder];
        if (!copy_cmd || !blit) {
          return runtime::Status::kDriverError;
        }
        [blit copyFromBuffer:ctx.mps_buf_out
                sourceOffset:0
                    toBuffer:ctx.buf_out
           destinationOffset:0
                        size:static_cast<NSUInteger>(bytes_out)];
        [blit endEncoding];
        [copy_cmd commit];
        [copy_cmd waitUntilCompleted];
        if (copy_cmd.status != MTLCommandBufferStatusCompleted) {
          return runtime::Status::kUnknown;
        }
      }
    }

    if (wait_for_completion) {
      ctx.async_inflight = 0;
    } else {
      ctx.async_inflight += repeat_u32;
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

}  // namespace lightning_core::detail

#else

namespace lightning_core::detail {

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

runtime::Status matMulMetalResetTuning() {
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

runtime::Status matMulMetalWithPolicyBatched(
    const float* a,
    const float* b,
    float* out,
    std::size_t m,
    std::size_t k,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize,
    std::size_t repeat_count) {
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
  (void)repeat_count;
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

}  // namespace lightning_core::detail

#endif
