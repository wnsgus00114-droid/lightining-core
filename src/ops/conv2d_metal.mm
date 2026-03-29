#include "lightning_core/core/detail/ops_backend.hpp"

#if defined(CJ_HAS_METAL) && CJ_HAS_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

namespace lightning_core::detail {

namespace {

constexpr uint32_t kConv2dMaxAsyncInflight = 64;
constexpr std::size_t kConvKernelModeCacheMaxEntries = 256;

struct MetalConv2dContext {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> pipeline_3x3s1p1 = nil;
  id<MTLComputePipelineState> pipeline_3x3s1p1_oc4 = nil;
  id<MTLComputePipelineState> pipeline_3x3s1p1_oc4_u4 = nil;
  id<MTLComputePipelineState> pipeline_3x3s1p1_oc4_tile = nil;
  id<MTLBuffer> buf_x = nil;
  id<MTLBuffer> buf_w = nil;
  id<MTLBuffer> buf_bias = nil;
  id<MTLBuffer> buf_out = nil;
  std::size_t cap_x = 0;
  std::size_t cap_w = 0;
  std::size_t cap_bias = 0;
  std::size_t cap_out = 0;
  uint32_t async_inflight = 0;
  struct ConvShapeKey {
    uint32_t n;
    uint32_t ic;
    uint32_t ih;
    uint32_t iw;
    uint32_t oc;
    uint32_t relu;
    uint32_t has_bias;
  };
  struct ConvShapeKeyHash {
    std::size_t operator()(const ConvShapeKey& k) const noexcept {
      std::size_t h = 1469598103934665603ull;
      auto mix = [&](uint32_t v) {
        h ^= static_cast<std::size_t>(v);
        h *= 1099511628211ull;
      };
      mix(k.n);
      mix(k.ic);
      mix(k.ih);
      mix(k.iw);
      mix(k.oc);
      mix(k.relu);
      mix(k.has_bias);
      return h;
    }
  };
  struct ConvShapeKeyEq {
    bool operator()(const ConvShapeKey& a, const ConvShapeKey& b) const noexcept {
      return a.n == b.n && a.ic == b.ic && a.ih == b.ih && a.iw == b.iw && a.oc == b.oc &&
             a.relu == b.relu && a.has_bias == b.has_bias;
    }
  };
  std::unordered_map<ConvShapeKey, uint8_t, ConvShapeKeyHash, ConvShapeKeyEq> kernel_mode_cache;
};

const char* kConv2dShaderSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void conv2d_nchw_3x3_s1_p1_f32(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& in_channels [[buffer(5)]],
    constant uint& in_h [[buffer(6)]],
    constant uint& in_w [[buffer(7)]],
    constant uint& out_channels [[buffer(8)]],
    constant uint& out_h [[buffer(9)]],
    constant uint& out_w [[buffer(10)]],
    constant uint& has_bias [[buffer(11)]],
    constant uint& apply_relu [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]]) {
  const uint ox = gid.x;
  const uint oy = gid.y;
  const uint z = gid.z;
  if (ox >= out_w || oy >= out_h || z >= batch * out_channels) {
    return;
  }

  const uint n = z / out_channels;
  const uint oc = z - n * out_channels;

  float acc = has_bias != 0 ? bias[oc] : 0.0f;

  for (uint ic = 0; ic < in_channels; ++ic) {
    const uint w_base = ((oc * in_channels + ic) * 3u) * 3u;
    for (uint ky = 0; ky < 3u; ++ky) {
      const int iy = static_cast<int>(oy) + static_cast<int>(ky) - 1;
      if (iy < 0 || iy >= static_cast<int>(in_h)) {
        continue;
      }
      for (uint kx = 0; kx < 3u; ++kx) {
        const int ix = static_cast<int>(ox) + static_cast<int>(kx) - 1;
        if (ix < 0 || ix >= static_cast<int>(in_w)) {
          continue;
        }
        const uint x_idx = ((n * in_channels + ic) * in_h + static_cast<uint>(iy)) * in_w + static_cast<uint>(ix);
        acc += x[x_idx] * w[w_base + ky * 3u + kx];
      }
    }
  }

  if (apply_relu != 0 && acc < 0.0f) {
    acc = 0.0f;
  }

  const uint out_idx = ((n * out_channels + oc) * out_h + oy) * out_w + ox;
  out[out_idx] = acc;
}

kernel void conv2d_nchw_3x3_s1_p1_f32_oc4(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& in_channels [[buffer(5)]],
    constant uint& in_h [[buffer(6)]],
    constant uint& in_w [[buffer(7)]],
    constant uint& out_channels [[buffer(8)]],
    constant uint& out_h [[buffer(9)]],
    constant uint& out_w [[buffer(10)]],
    constant uint& has_bias [[buffer(11)]],
    constant uint& apply_relu [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]]) {
  const uint ox = gid.x;
  const uint oy = gid.y;
  const uint z = gid.z;
  const uint out_blocks = out_channels / 4u;
  if (ox >= out_w || oy >= out_h || z >= batch * out_blocks) {
    return;
  }

  const uint n = z / out_blocks;
  const uint oc_block = z - n * out_blocks;
  const uint oc0 = oc_block * 4u;

  float4 acc = has_bias != 0
      ? float4(bias[oc0 + 0], bias[oc0 + 1], bias[oc0 + 2], bias[oc0 + 3])
      : float4(0.0f);

  for (uint ic = 0; ic < in_channels; ++ic) {
    for (uint ky = 0; ky < 3u; ++ky) {
      const int iy = static_cast<int>(oy) + static_cast<int>(ky) - 1;
      if (iy < 0 || iy >= static_cast<int>(in_h)) {
        continue;
      }
      for (uint kx = 0; kx < 3u; ++kx) {
        const int ix = static_cast<int>(ox) + static_cast<int>(kx) - 1;
        if (ix < 0 || ix >= static_cast<int>(in_w)) {
          continue;
        }

        const uint x_idx = ((n * in_channels + ic) * in_h + static_cast<uint>(iy)) * in_w + static_cast<uint>(ix);
        const float xv = x[x_idx];

        const uint w0 = (((oc0 + 0u) * in_channels + ic) * 3u + ky) * 3u + kx;
        const uint w1 = (((oc0 + 1u) * in_channels + ic) * 3u + ky) * 3u + kx;
        const uint w2 = (((oc0 + 2u) * in_channels + ic) * 3u + ky) * 3u + kx;
        const uint w3 = (((oc0 + 3u) * in_channels + ic) * 3u + ky) * 3u + kx;
        acc += xv * float4(w[w0], w[w1], w[w2], w[w3]);
      }
    }
  }

  if (apply_relu != 0) {
    acc = max(acc, float4(0.0f));
  }

  const uint base = ((n * out_channels) * out_h + oy) * out_w + ox;
  const uint plane = out_h * out_w;
  out[base + (oc0 + 0u) * plane] = acc.x;
  out[base + (oc0 + 1u) * plane] = acc.y;
  out[base + (oc0 + 2u) * plane] = acc.z;
  out[base + (oc0 + 3u) * plane] = acc.w;
}

kernel void conv2d_nchw_3x3_s1_p1_f32_oc4_u4(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& in_channels [[buffer(5)]],
    constant uint& in_h [[buffer(6)]],
    constant uint& in_w [[buffer(7)]],
    constant uint& out_channels [[buffer(8)]],
    constant uint& out_h [[buffer(9)]],
    constant uint& out_w [[buffer(10)]],
    constant uint& has_bias [[buffer(11)]],
    constant uint& apply_relu [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]]) {
  const uint ox = gid.x;
  const uint oy = gid.y;
  const uint z = gid.z;
  const uint out_blocks = out_channels / 4u;
  if (ox >= out_w || oy >= out_h || z >= batch * out_blocks) {
    return;
  }

  const uint n = z / out_blocks;
  const uint oc_block = z - n * out_blocks;
  const uint oc0 = oc_block * 4u;
  float4 acc = has_bias != 0
      ? float4(bias[oc0 + 0], bias[oc0 + 1], bias[oc0 + 2], bias[oc0 + 3])
      : float4(0.0f);

  uint ic = 0;
  const uint ic4_end = in_channels & ~3u;
  for (; ic < ic4_end; ic += 4u) {
    for (uint ky = 0; ky < 3u; ++ky) {
      const int iy = static_cast<int>(oy) + static_cast<int>(ky) - 1;
      if (iy < 0 || iy >= static_cast<int>(in_h)) {
        continue;
      }
      for (uint kx = 0; kx < 3u; ++kx) {
        const int ix = static_cast<int>(ox) + static_cast<int>(kx) - 1;
        if (ix < 0 || ix >= static_cast<int>(in_w)) {
          continue;
        }
        const uint xi = ((n * in_channels) * in_h + static_cast<uint>(iy)) * in_w + static_cast<uint>(ix);
        const float x0 = x[xi + (ic + 0u) * in_h * in_w];
        const float x1 = x[xi + (ic + 1u) * in_h * in_w];
        const float x2 = x[xi + (ic + 2u) * in_h * in_w];
        const float x3 = x[xi + (ic + 3u) * in_h * in_w];

        const uint w00 = (((oc0 + 0u) * in_channels + (ic + 0u)) * 3u + ky) * 3u + kx;
        const uint w01 = (((oc0 + 0u) * in_channels + (ic + 1u)) * 3u + ky) * 3u + kx;
        const uint w02 = (((oc0 + 0u) * in_channels + (ic + 2u)) * 3u + ky) * 3u + kx;
        const uint w03 = (((oc0 + 0u) * in_channels + (ic + 3u)) * 3u + ky) * 3u + kx;
        const uint w10 = (((oc0 + 1u) * in_channels + (ic + 0u)) * 3u + ky) * 3u + kx;
        const uint w11 = (((oc0 + 1u) * in_channels + (ic + 1u)) * 3u + ky) * 3u + kx;
        const uint w12 = (((oc0 + 1u) * in_channels + (ic + 2u)) * 3u + ky) * 3u + kx;
        const uint w13 = (((oc0 + 1u) * in_channels + (ic + 3u)) * 3u + ky) * 3u + kx;
        const uint w20 = (((oc0 + 2u) * in_channels + (ic + 0u)) * 3u + ky) * 3u + kx;
        const uint w21 = (((oc0 + 2u) * in_channels + (ic + 1u)) * 3u + ky) * 3u + kx;
        const uint w22 = (((oc0 + 2u) * in_channels + (ic + 2u)) * 3u + ky) * 3u + kx;
        const uint w23 = (((oc0 + 2u) * in_channels + (ic + 3u)) * 3u + ky) * 3u + kx;
        const uint w30 = (((oc0 + 3u) * in_channels + (ic + 0u)) * 3u + ky) * 3u + kx;
        const uint w31 = (((oc0 + 3u) * in_channels + (ic + 1u)) * 3u + ky) * 3u + kx;
        const uint w32 = (((oc0 + 3u) * in_channels + (ic + 2u)) * 3u + ky) * 3u + kx;
        const uint w33 = (((oc0 + 3u) * in_channels + (ic + 3u)) * 3u + ky) * 3u + kx;

        acc.x += x0 * w[w00] + x1 * w[w01] + x2 * w[w02] + x3 * w[w03];
        acc.y += x0 * w[w10] + x1 * w[w11] + x2 * w[w12] + x3 * w[w13];
        acc.z += x0 * w[w20] + x1 * w[w21] + x2 * w[w22] + x3 * w[w23];
        acc.w += x0 * w[w30] + x1 * w[w31] + x2 * w[w32] + x3 * w[w33];
      }
    }
  }

  for (; ic < in_channels; ++ic) {
    for (uint ky = 0; ky < 3u; ++ky) {
      const int iy = static_cast<int>(oy) + static_cast<int>(ky) - 1;
      if (iy < 0 || iy >= static_cast<int>(in_h)) {
        continue;
      }
      for (uint kx = 0; kx < 3u; ++kx) {
        const int ix = static_cast<int>(ox) + static_cast<int>(kx) - 1;
        if (ix < 0 || ix >= static_cast<int>(in_w)) {
          continue;
        }
        const uint x_idx = ((n * in_channels + ic) * in_h + static_cast<uint>(iy)) * in_w + static_cast<uint>(ix);
        const float xv = x[x_idx];
        const uint w0 = (((oc0 + 0u) * in_channels + ic) * 3u + ky) * 3u + kx;
        const uint w1 = (((oc0 + 1u) * in_channels + ic) * 3u + ky) * 3u + kx;
        const uint w2 = (((oc0 + 2u) * in_channels + ic) * 3u + ky) * 3u + kx;
        const uint w3 = (((oc0 + 3u) * in_channels + ic) * 3u + ky) * 3u + kx;
        acc += xv * float4(w[w0], w[w1], w[w2], w[w3]);
      }
    }
  }

  if (apply_relu != 0) {
    acc = max(acc, float4(0.0f));
  }
  const uint base = ((n * out_channels) * out_h + oy) * out_w + ox;
  const uint plane = out_h * out_w;
  out[base + (oc0 + 0u) * plane] = acc.x;
  out[base + (oc0 + 1u) * plane] = acc.y;
  out[base + (oc0 + 2u) * plane] = acc.z;
  out[base + (oc0 + 3u) * plane] = acc.w;
}

kernel void conv2d_nchw_3x3_s1_p1_f32_oc4_tile(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& in_channels [[buffer(5)]],
    constant uint& in_h [[buffer(6)]],
    constant uint& in_w [[buffer(7)]],
    constant uint& out_channels [[buffer(8)]],
    constant uint& out_h [[buffer(9)]],
    constant uint& out_w [[buffer(10)]],
    constant uint& has_bias [[buffer(11)]],
    constant uint& apply_relu [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  const uint ox = gid.x;
  const uint oy = gid.y;
  const uint z = gid.z;
  const uint out_blocks = out_channels / 4u;
  if (ox >= out_w || oy >= out_h || z >= batch * out_blocks) {
    return;
  }

  const uint n = z / out_blocks;
  const uint oc_block = z - n * out_blocks;
  const uint oc0 = oc_block * 4u;

  const int block_ox = static_cast<int>(ox) - static_cast<int>(lid.x);
  const int block_oy = static_cast<int>(oy) - static_cast<int>(lid.y);

  float4 acc = has_bias != 0
      ? float4(bias[oc0 + 0], bias[oc0 + 1], bias[oc0 + 2], bias[oc0 + 3])
      : float4(0.0f);

  threadgroup float tile[10][10];
  for (uint ic = 0; ic < in_channels; ++ic) {
    for (uint ty = lid.y; ty < 10u; ty += 8u) {
      for (uint tx = lid.x; tx < 10u; tx += 8u) {
        const int gx = block_ox + static_cast<int>(tx) - 1;
        const int gy = block_oy + static_cast<int>(ty) - 1;
        if (gx >= 0 && gx < static_cast<int>(in_w) && gy >= 0 && gy < static_cast<int>(in_h)) {
          const uint x_idx = ((n * in_channels + ic) * in_h + static_cast<uint>(gy)) * in_w + static_cast<uint>(gx);
          tile[ty][tx] = x[x_idx];
        } else {
          tile[ty][tx] = 0.0f;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint wbase = (oc0 * in_channels + ic) * 9u;
    const float4 k00 = float4(w[wbase + 0u], w[wbase + 9u], w[wbase + 18u], w[wbase + 27u]);
    const float4 k01 = float4(w[wbase + 1u], w[wbase + 10u], w[wbase + 19u], w[wbase + 28u]);
    const float4 k02 = float4(w[wbase + 2u], w[wbase + 11u], w[wbase + 20u], w[wbase + 29u]);
    const float4 k10 = float4(w[wbase + 3u], w[wbase + 12u], w[wbase + 21u], w[wbase + 30u]);
    const float4 k11 = float4(w[wbase + 4u], w[wbase + 13u], w[wbase + 22u], w[wbase + 31u]);
    const float4 k12 = float4(w[wbase + 5u], w[wbase + 14u], w[wbase + 23u], w[wbase + 32u]);
    const float4 k20 = float4(w[wbase + 6u], w[wbase + 15u], w[wbase + 24u], w[wbase + 33u]);
    const float4 k21 = float4(w[wbase + 7u], w[wbase + 16u], w[wbase + 25u], w[wbase + 34u]);
    const float4 k22 = float4(w[wbase + 8u], w[wbase + 17u], w[wbase + 26u], w[wbase + 35u]);

    const uint lx = lid.x;
    const uint ly = lid.y;
    acc += tile[ly + 0u][lx + 0u] * k00;
    acc += tile[ly + 0u][lx + 1u] * k01;
    acc += tile[ly + 0u][lx + 2u] * k02;
    acc += tile[ly + 1u][lx + 0u] * k10;
    acc += tile[ly + 1u][lx + 1u] * k11;
    acc += tile[ly + 1u][lx + 2u] * k12;
    acc += tile[ly + 2u][lx + 0u] * k20;
    acc += tile[ly + 2u][lx + 1u] * k21;
    acc += tile[ly + 2u][lx + 2u] * k22;

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (apply_relu != 0) {
    acc = max(acc, float4(0.0f));
  }

  const uint base = ((n * out_channels) * out_h + oy) * out_w + ox;
  const uint plane = out_h * out_w;
  out[base + (oc0 + 0u) * plane] = acc.x;
  out[base + (oc0 + 1u) * plane] = acc.y;
  out[base + (oc0 + 2u) * plane] = acc.z;
  out[base + (oc0 + 3u) * plane] = acc.w;
}
)";

id<MTLDevice> getDevice() {
  static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  return device;
}

MetalConv2dContext& getContext() {
  static MetalConv2dContext ctx;
  return ctx;
}

runtime::Status waitForConvQueueIdle(MetalConv2dContext& ctx) {
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

runtime::Status ensureContextReady(MetalConv2dContext& ctx) {
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

  if (ctx.pipeline_3x3s1p1 == nil || ctx.pipeline_3x3s1p1_oc4 == nil ||
      ctx.pipeline_3x3s1p1_oc4_u4 == nil || ctx.pipeline_3x3s1p1_oc4_tile == nil) {
    NSString* src = [NSString stringWithUTF8String:kConv2dShaderSrc];
    NSError* err = nil;
    id<MTLLibrary> lib = [ctx.device newLibraryWithSource:src options:nil error:&err];
    if (!lib) {
      return runtime::Status::kDriverError;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"conv2d_nchw_3x3_s1_p1_f32"];
    if (!fn) {
      return runtime::Status::kDriverError;
    }
    ctx.pipeline_3x3s1p1 = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
    if (!ctx.pipeline_3x3s1p1) {
      return runtime::Status::kDriverError;
    }

    err = nil;
    id<MTLFunction> fn_oc4 = [lib newFunctionWithName:@"conv2d_nchw_3x3_s1_p1_f32_oc4"];
    if (!fn_oc4) {
      return runtime::Status::kDriverError;
    }
    ctx.pipeline_3x3s1p1_oc4 = [ctx.device newComputePipelineStateWithFunction:fn_oc4 error:&err];
    if (!ctx.pipeline_3x3s1p1_oc4) {
      return runtime::Status::kDriverError;
    }

    err = nil;
    id<MTLFunction> fn_oc4_u4 = [lib newFunctionWithName:@"conv2d_nchw_3x3_s1_p1_f32_oc4_u4"];
    if (!fn_oc4_u4) {
      return runtime::Status::kDriverError;
    }
    ctx.pipeline_3x3s1p1_oc4_u4 = [ctx.device newComputePipelineStateWithFunction:fn_oc4_u4 error:&err];
    if (!ctx.pipeline_3x3s1p1_oc4_u4) {
      return runtime::Status::kDriverError;
    }

    err = nil;
    id<MTLFunction> fn_oc4_tile = [lib newFunctionWithName:@"conv2d_nchw_3x3_s1_p1_f32_oc4_tile"];
    if (!fn_oc4_tile) {
      return runtime::Status::kDriverError;
    }
    ctx.pipeline_3x3s1p1_oc4_tile = [ctx.device newComputePipelineStateWithFunction:fn_oc4_tile error:&err];
    if (!ctx.pipeline_3x3s1p1_oc4_tile) {
      return runtime::Status::kDriverError;
    }
  }

  return runtime::Status::kSuccess;
}

runtime::Status ensureBuffer(id<MTLDevice> device, id<MTLBuffer>& buf, std::size_t& cap, std::size_t need) {
  if (cap >= need) {
    return runtime::Status::kSuccess;
  }
  buf = [device newBufferWithLength:need options:MTLResourceStorageModeShared];
  if (!buf) {
    return runtime::Status::kOutOfMemory;
  }
  cap = need;
  return runtime::Status::kSuccess;
}

enum class ConvKernelMode : uint8_t {
  kScalar = 0,
  kOc4 = 1,
  kOc4UnrollIc4 = 2,
  kOc4Tile = 3,
};

bool supportsMode(ConvKernelMode mode, uint32_t ic, uint32_t oc, uint32_t ih, uint32_t iw) {
  if (mode == ConvKernelMode::kScalar) {
    return true;
  }
  if ((oc % 4u) != 0u) {
    return false;
  }
  if (mode == ConvKernelMode::kOc4) {
    return ic >= 4;
  }
  if (mode == ConvKernelMode::kOc4UnrollIc4) {
    return ic >= 8;
  }
  if (mode == ConvKernelMode::kOc4Tile) {
    return ic >= 8 && ih >= 16 && iw >= 16;
  }
  return false;
}

id<MTLComputePipelineState> pipelineForMode(MetalConv2dContext& ctx, ConvKernelMode mode) {
  switch (mode) {
    case ConvKernelMode::kOc4:
      return ctx.pipeline_3x3s1p1_oc4;
    case ConvKernelMode::kOc4UnrollIc4:
      return ctx.pipeline_3x3s1p1_oc4_u4;
    case ConvKernelMode::kOc4Tile:
      return ctx.pipeline_3x3s1p1_oc4_tile;
    case ConvKernelMode::kScalar:
    default:
      return ctx.pipeline_3x3s1p1;
  }
}

void dispatchMode(
    id<MTLComputeCommandEncoder> enc,
    id<MTLComputePipelineState> pipe,
    ConvKernelMode mode,
    uint32_t b,
    uint32_t oc,
    uint32_t oh,
    uint32_t ow) {
  NSUInteger wgx = 8;
  NSUInteger wgy = 8;
  if (mode == ConvKernelMode::kOc4 || mode == ConvKernelMode::kOc4UnrollIc4) {
    const NSUInteger simd = std::max<NSUInteger>(8, pipe.threadExecutionWidth);
    wgx = simd;
    wgy = 4;
    if (ow >= 128 && pipe.maxTotalThreadsPerThreadgroup >= simd * 8) {
      wgy = 8;
    }
  } else if (mode == ConvKernelMode::kOc4Tile) {
    wgx = 8;
    wgy = 8;
  } else {
    wgx = 8;
    wgy = 8;
  }
  NSUInteger max_threads = pipe.maxTotalThreadsPerThreadgroup;
  if (wgx * wgy > max_threads) {
    wgy = max_threads / wgx;
    if (wgy == 0) {
      wgy = 1;
    }
  }

  const bool oc4_mode = (mode != ConvKernelMode::kScalar);
  MTLSize grid = oc4_mode ? MTLSizeMake(ow, oh, b * (oc / 4u)) : MTLSizeMake(ow, oh, b * oc);
  MTLSize threads = MTLSizeMake(wgx, wgy, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:threads];
}

ConvKernelMode chooseModeWithAutotune(
    MetalConv2dContext& ctx,
    id<MTLBuffer> buf_x,
    id<MTLBuffer> buf_w,
    id<MTLBuffer> buf_bias,
    id<MTLBuffer> buf_out,
    uint32_t b,
    uint32_t ic,
    uint32_t ih,
    uint32_t iw,
    uint32_t oc,
    uint32_t oh,
    uint32_t ow,
    uint32_t has_bias,
    uint32_t relu) {
  MetalConv2dContext::ConvShapeKey key{b, ic, ih, iw, oc, relu, has_bias};
  auto it = ctx.kernel_mode_cache.find(key);
  if (it != ctx.kernel_mode_cache.end()) {
    return static_cast<ConvKernelMode>(it->second);
  }

  const ConvKernelMode candidates[] = {
      ConvKernelMode::kScalar,
      ConvKernelMode::kOc4,
      ConvKernelMode::kOc4UnrollIc4,
      ConvKernelMode::kOc4Tile,
  };
  ConvKernelMode best = ConvKernelMode::kScalar;
  double best_ms = std::numeric_limits<double>::max();
  constexpr int kTuneRuns = 6;
  constexpr int kTuneSkip = 2;

  for (ConvKernelMode mode : candidates) {
    if (!supportsMode(mode, ic, oc, ih, iw)) {
      continue;
    }
    id<MTLComputePipelineState> pipe = pipelineForMode(ctx, mode);
    if (!pipe) {
      continue;
    }

    double sum_ms = 0.0;
    int measured = 0;
    for (int rep = 0; rep < kTuneRuns; ++rep) {
      id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      if (!cmd || !enc) {
        continue;
      }
      [enc setComputePipelineState:pipe];
      [enc setBuffer:buf_x offset:0 atIndex:0];
      [enc setBuffer:buf_w offset:0 atIndex:1];
      [enc setBuffer:buf_bias offset:0 atIndex:2];
      [enc setBuffer:buf_out offset:0 atIndex:3];
      [enc setBytes:&b length:sizeof(b) atIndex:4];
      [enc setBytes:&ic length:sizeof(ic) atIndex:5];
      [enc setBytes:&ih length:sizeof(ih) atIndex:6];
      [enc setBytes:&iw length:sizeof(iw) atIndex:7];
      [enc setBytes:&oc length:sizeof(oc) atIndex:8];
      [enc setBytes:&oh length:sizeof(oh) atIndex:9];
      [enc setBytes:&ow length:sizeof(ow) atIndex:10];
      [enc setBytes:&has_bias length:sizeof(has_bias) atIndex:11];
      [enc setBytes:&relu length:sizeof(relu) atIndex:12];
      dispatchMode(enc, pipe, mode, b, oc, oh, ow);
      [enc endEncoding];

      auto t0 = std::chrono::high_resolution_clock::now();
      [cmd commit];
      [cmd waitUntilCompleted];
      if (cmd.status != MTLCommandBufferStatusCompleted) {
        continue;
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      if (rep >= kTuneSkip) {
        sum_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        ++measured;
      }
    }

    if (measured > 0) {
      const double dt_ms = sum_ms / static_cast<double>(measured);
      if (dt_ms < best_ms) {
        best_ms = dt_ms;
        best = mode;
      }
    }
  }

  if (ctx.kernel_mode_cache.size() >= kConvKernelModeCacheMaxEntries) {
    ctx.kernel_mode_cache.clear();
  }
  ctx.kernel_mode_cache[key] = static_cast<uint8_t>(best);
  return best;
}

}  // namespace

runtime::Status conv2dNchw3x3s1p1Metal(
    const float* x,
    const float* w,
    const float* bias,
    float* out,
    std::size_t batch,
    std::size_t in_channels,
    std::size_t in_h,
    std::size_t in_w,
    std::size_t out_channels,
    bool apply_relu) {
  return conv2dNchw3x3s1p1MetalWithPolicy(
      x,
      w,
      bias,
      out,
      batch,
      in_channels,
      in_h,
      in_w,
      out_channels,
      apply_relu,
      true,
      true,
      true,
      true,
      true);
}

runtime::Status conv2dNchw3x3s1p1MetalWithPolicy(
    const float* x,
    const float* w,
    const float* bias,
    float* out,
    std::size_t batch,
    std::size_t in_channels,
    std::size_t in_h,
    std::size_t in_w,
    std::size_t out_channels,
    bool apply_relu,
    bool upload_x,
    bool upload_w,
    bool upload_bias,
    bool download_out,
    bool synchronize) {
  if ((upload_x && x == nullptr) || (upload_w && w == nullptr) || (upload_bias && bias == nullptr) ||
      (download_out && out == nullptr) || batch == 0 || in_channels == 0 || in_h == 0 || in_w == 0 ||
      out_channels == 0) {
    return runtime::Status::kInvalidValue;
  }

  const std::size_t out_h = in_h;
  const std::size_t out_w = in_w;

  @autoreleasepool {
    MetalConv2dContext& ctx = getContext();
    runtime::Status st = ensureContextReady(ctx);
    if (st != runtime::Status::kSuccess) {
      return st;
    }

    const std::size_t bytes_x = batch * in_channels * in_h * in_w * sizeof(float);
    const std::size_t bytes_w = out_channels * in_channels * 3 * 3 * sizeof(float);
    const std::size_t bytes_bias = out_channels * sizeof(float);
    const std::size_t bytes_out = batch * out_channels * out_h * out_w * sizeof(float);

    st = ensureBuffer(ctx.device, ctx.buf_x, ctx.cap_x, bytes_x);
    if (st != runtime::Status::kSuccess) {
      return st;
    }
    st = ensureBuffer(ctx.device, ctx.buf_w, ctx.cap_w, bytes_w);
    if (st != runtime::Status::kSuccess) {
      return st;
    }
    st = ensureBuffer(ctx.device, ctx.buf_bias, ctx.cap_bias, bytes_bias);
    if (st != runtime::Status::kSuccess) {
      return st;
    }
    st = ensureBuffer(ctx.device, ctx.buf_out, ctx.cap_out, bytes_out);
    if (st != runtime::Status::kSuccess) {
      return st;
    }

    if ((upload_x || upload_w || upload_bias) && ctx.async_inflight > 0) {
      st = waitForConvQueueIdle(ctx);
      if (st != runtime::Status::kSuccess) {
        return st;
      }
    }

    if (upload_x) {
      std::memcpy(ctx.buf_x.contents, x, bytes_x);
    }
    if (upload_w) {
      std::memcpy(ctx.buf_w.contents, w, bytes_w);
    }
    if (upload_bias) {
      if (bias != nullptr) {
        std::memcpy(ctx.buf_bias.contents, bias, bytes_bias);
      } else {
        std::memset(ctx.buf_bias.contents, 0, bytes_bias);
      }
    } else if (bias == nullptr) {
      std::memset(ctx.buf_bias.contents, 0, bytes_bias);
    }

    const bool no_new_upload = !upload_x && !upload_w && !upload_bias;
    if (no_new_upload && (synchronize || download_out)) {
      st = waitForConvQueueIdle(ctx);
      if (st != runtime::Status::kSuccess) {
        return st;
      }
      if (download_out) {
        std::memcpy(out, ctx.buf_out.contents, bytes_out);
      }
      return runtime::Status::kSuccess;
    }

    uint32_t b = static_cast<uint32_t>(batch);
    uint32_t ic = static_cast<uint32_t>(in_channels);
    uint32_t ih = static_cast<uint32_t>(in_h);
    uint32_t iw = static_cast<uint32_t>(in_w);
    uint32_t oc = static_cast<uint32_t>(out_channels);
    uint32_t oh = static_cast<uint32_t>(out_h);
    uint32_t ow = static_cast<uint32_t>(out_w);
    uint32_t has_bias = bias != nullptr ? 1u : 0u;
    uint32_t relu = apply_relu ? 1u : 0u;

    ConvKernelMode mode = chooseModeWithAutotune(
        ctx,
        ctx.buf_x,
        ctx.buf_w,
        ctx.buf_bias,
        ctx.buf_out,
        b,
        ic,
        ih,
        iw,
        oc,
        oh,
        ow,
        has_bias,
        relu);
    id<MTLComputePipelineState> pipe = pipelineForMode(ctx, mode);
    if (!pipe) {
      return runtime::Status::kDriverError;
    }

    id<MTLCommandBuffer> command = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
    if (!command || !enc) {
      return runtime::Status::kDriverError;
    }

    [enc setComputePipelineState:pipe];
    [enc setBuffer:ctx.buf_x offset:0 atIndex:0];
    [enc setBuffer:ctx.buf_w offset:0 atIndex:1];
    [enc setBuffer:ctx.buf_bias offset:0 atIndex:2];
    [enc setBuffer:ctx.buf_out offset:0 atIndex:3];
    [enc setBytes:&b length:sizeof(b) atIndex:4];
    [enc setBytes:&ic length:sizeof(ic) atIndex:5];
    [enc setBytes:&ih length:sizeof(ih) atIndex:6];
    [enc setBytes:&iw length:sizeof(iw) atIndex:7];
    [enc setBytes:&oc length:sizeof(oc) atIndex:8];
    [enc setBytes:&oh length:sizeof(oh) atIndex:9];
    [enc setBytes:&ow length:sizeof(ow) atIndex:10];
    [enc setBytes:&has_bias length:sizeof(has_bias) atIndex:11];
    [enc setBytes:&relu length:sizeof(relu) atIndex:12];

    dispatchMode(enc, pipe, mode, b, oc, oh, ow);
    [enc endEncoding];

    [command commit];
    const bool force_wait = (!synchronize && !download_out && (ctx.async_inflight + 1 >= kConv2dMaxAsyncInflight));
    const bool wait_for_completion = synchronize || download_out || force_wait;
    if (wait_for_completion) {
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) {
        return runtime::Status::kUnknown;
      }
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

}  // namespace lightning_core::detail

#else

namespace lightning_core::detail {

runtime::Status conv2dNchw3x3s1p1Metal(
    const float* x,
    const float* w,
    const float* bias,
    float* out,
    std::size_t batch,
    std::size_t in_channels,
    std::size_t in_h,
    std::size_t in_w,
    std::size_t out_channels,
    bool apply_relu) {
  (void)x;
  (void)w;
  (void)bias;
  (void)out;
  (void)batch;
  (void)in_channels;
  (void)in_h;
  (void)in_w;
  (void)out_channels;
  (void)apply_relu;
  return runtime::Status::kNotSupported;
}

runtime::Status conv2dNchw3x3s1p1MetalWithPolicy(
    const float* x,
    const float* w,
    const float* bias,
    float* out,
    std::size_t batch,
    std::size_t in_channels,
    std::size_t in_h,
    std::size_t in_w,
    std::size_t out_channels,
    bool apply_relu,
    bool upload_x,
    bool upload_w,
    bool upload_bias,
    bool download_out,
    bool synchronize) {
  (void)x;
  (void)w;
  (void)bias;
  (void)out;
  (void)batch;
  (void)in_channels;
  (void)in_h;
  (void)in_w;
  (void)out_channels;
  (void)apply_relu;
  (void)upload_x;
  (void)upload_w;
  (void)upload_bias;
  (void)download_out;
  (void)synchronize;
  return runtime::Status::kNotSupported;
}

}  // namespace lightning_core::detail

#endif
