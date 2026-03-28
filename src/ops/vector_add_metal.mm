#include "lightning_core/core/detail/ops_backend.hpp"

#if defined(CJ_HAS_METAL) && CJ_HAS_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>

namespace lightning_core::detail {

namespace {

constexpr uint32_t kVectorAddMaxAsyncInflight = 256;

struct MetalVectorAddContext {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> pipeline_f32x4 = nil;
  id<MTLComputePipelineState> pipeline_f32_tail = nil;
  id<MTLBuffer> buf_a = nil;
  id<MTLBuffer> buf_b = nil;
  id<MTLBuffer> buf_out = nil;
  std::size_t capacity_bytes = 0;
  uint32_t async_inflight = 0;
};

const char* kMetalShaderSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void vector_add_f32x4(
    device const float4* a [[buffer(0)]],
    device const float4* b [[buffer(1)]],
    device float4* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  out[gid] = a[gid] + b[gid];
}

kernel void vector_add_f32_tail(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& base [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= count) {
    return;
  }
  const uint idx = base + gid;
  out[idx] = a[idx] + b[idx];
}
)";

id<MTLDevice> getDevice() {
  static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  return device;
}

MetalVectorAddContext& getContext() {
  static MetalVectorAddContext ctx;
  return ctx;
}

runtime::Status buildPipelines(MetalVectorAddContext& ctx) {
  NSString* source = [NSString stringWithUTF8String:kMetalShaderSrc];
  NSError* err = nil;
  id<MTLLibrary> library = [ctx.device newLibraryWithSource:source options:nil error:&err];
  if (!library) {
    return runtime::Status::kDriverError;
  }

  id<MTLFunction> fn_vec4 = [library newFunctionWithName:@"vector_add_f32x4"];
  id<MTLFunction> fn_tail = [library newFunctionWithName:@"vector_add_f32_tail"];
  if (!fn_vec4 || !fn_tail) {
    return runtime::Status::kDriverError;
  }

  ctx.pipeline_f32x4 = [ctx.device newComputePipelineStateWithFunction:fn_vec4 error:&err];
  if (!ctx.pipeline_f32x4) {
    return runtime::Status::kDriverError;
  }

  err = nil;
  ctx.pipeline_f32_tail = [ctx.device newComputePipelineStateWithFunction:fn_tail error:&err];
  if (!ctx.pipeline_f32_tail) {
    return runtime::Status::kDriverError;
  }

  return runtime::Status::kSuccess;
}

runtime::Status ensureContextReady(std::size_t bytes) {
  MetalVectorAddContext& ctx = getContext();

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

  if (ctx.pipeline_f32x4 == nil || ctx.pipeline_f32_tail == nil) {
    runtime::Status st = buildPipelines(ctx);
    if (st != runtime::Status::kSuccess) {
      return st;
    }
  }

  if (ctx.capacity_bytes < bytes) {
    ctx.buf_a = [ctx.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    ctx.buf_b = [ctx.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    ctx.buf_out = [ctx.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!ctx.buf_a || !ctx.buf_b || !ctx.buf_out) {
      return runtime::Status::kOutOfMemory;
    }
    ctx.capacity_bytes = bytes;
  }

  return runtime::Status::kSuccess;
}

runtime::Status waitForVectorQueueIdle(MetalVectorAddContext& ctx) {
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

void dispatchVectorKernel(id<MTLComputeCommandEncoder> enc,
                          id<MTLComputePipelineState> pipe,
                          std::size_t work_items,
                          NSUInteger width_mul) {
  if (work_items == 0 || pipe == nil) {
    return;
  }

  [enc setComputePipelineState:pipe];
  NSUInteger width = pipe.threadExecutionWidth;
  if (width == 0) {
    width = 32;
  }
  NSUInteger w = width * width_mul;
  NSUInteger max_threads = pipe.maxTotalThreadsPerThreadgroup;
  if (w > max_threads) {
    w = max_threads;
  }
  if (work_items >= 256 && max_threads >= 256 && w < 256) {
    w = 256;
  }
  if (w == 0) {
    w = 1;
  }

  MTLSize grid = MTLSizeMake(work_items, 1, 1);
  MTLSize threads = MTLSizeMake(w, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:threads];
}

}  // namespace

runtime::Status vectorAddMetal(const float* a, const float* b, float* out, std::size_t n) {
  return vectorAddMetalWithPolicy(a, b, out, n, true, true, true, true);
}

runtime::Status vectorAddMetalWithPolicy(
    const float* a,
    const float* b,
    float* out,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  if ((upload_a && a == nullptr) || (upload_b && b == nullptr) || (download_out && out == nullptr) || n == 0) {
    return runtime::Status::kInvalidValue;
  }

  @autoreleasepool {
    MetalVectorAddContext& ctx = getContext();
    const std::size_t bytes = n * sizeof(float);
    runtime::Status st = ensureContextReady(bytes);
    if (st != runtime::Status::kSuccess) {
      return st;
    }

    if ((upload_a || upload_b) && ctx.async_inflight > 0) {
      st = waitForVectorQueueIdle(ctx);
      if (st != runtime::Status::kSuccess) {
        return st;
      }
    }

    if (upload_a) {
      std::memcpy(ctx.buf_a.contents, a, bytes);
    }
    if (upload_b) {
      std::memcpy(ctx.buf_b.contents, b, bytes);
    }

    id<MTLCommandBuffer> command = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!command || !encoder) {
      return runtime::Status::kDriverError;
    }

    [encoder setBuffer:ctx.buf_a offset:0 atIndex:0];
    [encoder setBuffer:ctx.buf_b offset:0 atIndex:1];
    [encoder setBuffer:ctx.buf_out offset:0 atIndex:2];

    const std::size_t vec4_count = n / 4;
    const std::size_t tail_count = n % 4;

    if (vec4_count > 0) {
      dispatchVectorKernel(encoder, ctx.pipeline_f32x4, vec4_count, 4);
    }

    if (tail_count > 0) {
      const uint32_t base = static_cast<uint32_t>(vec4_count * 4);
      const uint32_t tail = static_cast<uint32_t>(tail_count);
      [encoder setBytes:&base length:sizeof(base) atIndex:3];
      [encoder setBytes:&tail length:sizeof(tail) atIndex:4];
      dispatchVectorKernel(encoder, ctx.pipeline_f32_tail, tail_count, 1);
    }

    [encoder endEncoding];

    [command commit];
    const bool force_wait = (!synchronize && !download_out && (ctx.async_inflight + 1 >= kVectorAddMaxAsyncInflight));
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
      std::memcpy(out, ctx.buf_out.contents, bytes);
    }
    return runtime::Status::kSuccess;
  }
}

runtime::Status vectorAddMetal(const double* a, const double* b, double* out, std::size_t n) {
  (void)a;
  (void)b;
  (void)out;
  (void)n;
  return runtime::Status::kNotSupported;
}

runtime::Status vectorAddMetalWithPolicy(
    const double* a,
    const double* b,
    double* out,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)a;
  (void)b;
  (void)out;
  (void)n;
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return runtime::Status::kNotSupported;
}

runtime::Status vectorAddMetal(const long double* a, const long double* b, long double* out, std::size_t n) {
  (void)a;
  (void)b;
  (void)out;
  (void)n;
  return runtime::Status::kNotSupported;
}

runtime::Status vectorAddMetalWithPolicy(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)a;
  (void)b;
  (void)out;
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

runtime::Status vectorAddMetal(const float* a, const float* b, float* out, std::size_t n) {
  (void)a;
  (void)b;
  (void)out;
  (void)n;
  return runtime::Status::kNotSupported;
}

runtime::Status vectorAddMetalWithPolicy(
    const float* a,
    const float* b,
    float* out,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)a;
  (void)b;
  (void)out;
  (void)n;
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return runtime::Status::kNotSupported;
}

runtime::Status vectorAddMetal(const double* a, const double* b, double* out, std::size_t n) {
  (void)a;
  (void)b;
  (void)out;
  (void)n;
  return runtime::Status::kNotSupported;
}

runtime::Status vectorAddMetalWithPolicy(
    const double* a,
    const double* b,
    double* out,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)a;
  (void)b;
  (void)out;
  (void)n;
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return runtime::Status::kNotSupported;
}

runtime::Status vectorAddMetal(const long double* a, const long double* b, long double* out, std::size_t n) {
  (void)a;
  (void)b;
  (void)out;
  (void)n;
  return runtime::Status::kNotSupported;
}

runtime::Status vectorAddMetalWithPolicy(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)a;
  (void)b;
  (void)out;
  (void)n;
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return runtime::Status::kNotSupported;
}

}  // namespace lightning_core::detail

#endif
