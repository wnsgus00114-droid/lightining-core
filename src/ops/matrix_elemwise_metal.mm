#include "lightning_core/core/detail/ops_backend.hpp"

#if defined(CJ_HAS_METAL) && CJ_HAS_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>

namespace lightning_core::detail {

namespace {

struct MatrixElemwiseMetalContext {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> sub_pipeline = nil;
  id<MTLComputePipelineState> sub_vec4_pipeline = nil;
  id<MTLComputePipelineState> sub_tail_pipeline = nil;
  id<MTLComputePipelineState> div_pipeline = nil;
  id<MTLComputePipelineState> div_vec4_pipeline = nil;
  id<MTLComputePipelineState> div_tail_pipeline = nil;
  id<MTLBuffer> buf_a = nil;
  id<MTLBuffer> buf_b = nil;
  id<MTLBuffer> buf_out = nil;
  std::size_t capacity_bytes = 0;
  uint32_t async_inflight = 0;
};

const char* kMetalShaderSrc = R"(
#include <metal_stdlib>
using namespace metal;

kernel void matrix_sub_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  out[gid] = a[gid] - b[gid];
}

kernel void matrix_div_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  out[gid] = a[gid] / b[gid];
}

kernel void matrix_sub_f32_tail(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& offset [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= count) {
    return;
  }
  uint idx = offset + gid;
  out[idx] = a[idx] - b[idx];
}

kernel void matrix_div_f32_tail(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& offset [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= count) {
    return;
  }
  uint idx = offset + gid;
  out[idx] = a[idx] / b[idx];
}

kernel void matrix_sub_f32_vec4(
    device const float4* a [[buffer(0)]],
    device const float4* b [[buffer(1)]],
    device float4* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  out[gid] = a[gid] - b[gid];
}

kernel void matrix_div_f32_vec4(
    device const float4* a [[buffer(0)]],
    device const float4* b [[buffer(1)]],
    device float4* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  out[gid] = a[gid] / b[gid];
}
)";

MatrixElemwiseMetalContext& getContext() {
  static MatrixElemwiseMetalContext ctx;
  return ctx;
}

runtime::Status ensureContextReady(std::size_t bytes) {
  MatrixElemwiseMetalContext& ctx = getContext();
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

    if (ctx.sub_pipeline == nil || ctx.div_pipeline == nil || ctx.sub_vec4_pipeline == nil ||
      ctx.div_vec4_pipeline == nil || ctx.sub_tail_pipeline == nil || ctx.div_tail_pipeline == nil) {
    NSString* source = [NSString stringWithUTF8String:kMetalShaderSrc];
    NSError* err = nil;
    id<MTLLibrary> library = [ctx.device newLibraryWithSource:source options:nil error:&err];
    if (!library) {
      return runtime::Status::kDriverError;
    }

    id<MTLFunction> sub_fn = [library newFunctionWithName:@"matrix_sub_f32"];
    id<MTLFunction> div_fn = [library newFunctionWithName:@"matrix_div_f32"];
    id<MTLFunction> sub_vec4_fn = [library newFunctionWithName:@"matrix_sub_f32_vec4"];
    id<MTLFunction> div_vec4_fn = [library newFunctionWithName:@"matrix_div_f32_vec4"];
    id<MTLFunction> sub_tail_fn = [library newFunctionWithName:@"matrix_sub_f32_tail"];
    id<MTLFunction> div_tail_fn = [library newFunctionWithName:@"matrix_div_f32_tail"];
    if (!sub_fn || !div_fn || !sub_vec4_fn || !div_vec4_fn || !sub_tail_fn || !div_tail_fn) {
      return runtime::Status::kDriverError;
    }

    ctx.sub_pipeline = [ctx.device newComputePipelineStateWithFunction:sub_fn error:&err];
    if (!ctx.sub_pipeline) {
      return runtime::Status::kDriverError;
    }

    ctx.div_pipeline = [ctx.device newComputePipelineStateWithFunction:div_fn error:&err];
    if (!ctx.div_pipeline) {
      return runtime::Status::kDriverError;
    }

    ctx.sub_vec4_pipeline = [ctx.device newComputePipelineStateWithFunction:sub_vec4_fn error:&err];
    if (!ctx.sub_vec4_pipeline) {
      return runtime::Status::kDriverError;
    }

    ctx.div_vec4_pipeline = [ctx.device newComputePipelineStateWithFunction:div_vec4_fn error:&err];
    if (!ctx.div_vec4_pipeline) {
      return runtime::Status::kDriverError;
    }

    ctx.sub_tail_pipeline = [ctx.device newComputePipelineStateWithFunction:sub_tail_fn error:&err];
    if (!ctx.sub_tail_pipeline) {
      return runtime::Status::kDriverError;
    }

    ctx.div_tail_pipeline = [ctx.device newComputePipelineStateWithFunction:div_tail_fn error:&err];
    if (!ctx.div_tail_pipeline) {
      return runtime::Status::kDriverError;
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

runtime::Status runKernel(
    id<MTLComputePipelineState> pipeline,
  id<MTLComputePipelineState> vec4_pipeline,
  id<MTLComputePipelineState> tail_pipeline,
    const float* a,
    const float* b,
    float* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  if (rows == 0 || cols == 0) {
    return runtime::Status::kInvalidValue;
  }
  if ((upload_a && a == nullptr) || (upload_b && b == nullptr) || (download_out && out == nullptr)) {
    return runtime::Status::kInvalidValue;
  }

  const std::size_t elem_count = rows * cols;
  const std::size_t bytes = elem_count * sizeof(float);
  const bool use_vec4 = (elem_count >= 4) && (vec4_pipeline != nil);
  const std::size_t vec4_elems = use_vec4 ? (elem_count / 4) : 0;
  const std::size_t tail_elems = use_vec4 ? (elem_count - (vec4_elems * 4)) : elem_count;

  MatrixElemwiseMetalContext& ctx = getContext();

  const bool wait_for_completion = synchronize || download_out;

  if ((upload_a || upload_b) && ctx.async_inflight > 0) {
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
  }

  if (upload_a) {
    std::memcpy(ctx.buf_a.contents, a, bytes);
  }
  if (upload_b) {
    std::memcpy(ctx.buf_b.contents, b, bytes);
  }

  id<MTLCommandBuffer> command = [ctx.queue commandBuffer];
  if (!command) {
    return runtime::Status::kDriverError;
  }

  auto dispatchLinearKernel = ^runtime::Status(id<MTLComputePipelineState> pipe, std::size_t count) {
    if (!pipe || count == 0) {
      return runtime::Status::kSuccess;
    }
    id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
    if (!enc) {
      return runtime::Status::kDriverError;
    }
    [enc setComputePipelineState:pipe];
    [enc setBuffer:ctx.buf_a offset:0 atIndex:0];
    [enc setBuffer:ctx.buf_b offset:0 atIndex:1];
    [enc setBuffer:ctx.buf_out offset:0 atIndex:2];
    NSUInteger width = pipe.threadExecutionWidth;
    if (width == 0) {
      width = 32;
    }
    NSUInteger max_threads = pipe.maxTotalThreadsPerThreadgroup;
    NSUInteger w = width * 4;
    if (w > max_threads) {
      w = max_threads;
    }
    if (w == 0) {
      w = 1;
    }
    MTLSize grid = MTLSizeMake(count, 1, 1);
    MTLSize threads = MTLSizeMake(w, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:threads];
    [enc endEncoding];
    return runtime::Status::kSuccess;
  };

  runtime::Status dispatch_st = runtime::Status::kSuccess;
  if (vec4_elems > 0) {
    dispatch_st = dispatchLinearKernel(vec4_pipeline, vec4_elems);
  }
  if (dispatch_st != runtime::Status::kSuccess) {
    return dispatch_st;
  }

  if (tail_elems > 0) {
    if (!tail_pipeline) {
      return runtime::Status::kDriverError;
    }
    id<MTLComputeCommandEncoder> tail_enc = [command computeCommandEncoder];
    if (!tail_enc) {
      return runtime::Status::kDriverError;
    }
    uint32_t tail_offset = static_cast<uint32_t>(vec4_elems * 4);
    uint32_t tail_count = static_cast<uint32_t>(tail_elems);
    [tail_enc setComputePipelineState:tail_pipeline];
    [tail_enc setBuffer:ctx.buf_a offset:0 atIndex:0];
    [tail_enc setBuffer:ctx.buf_b offset:0 atIndex:1];
    [tail_enc setBuffer:ctx.buf_out offset:0 atIndex:2];
    [tail_enc setBytes:&tail_offset length:sizeof(tail_offset) atIndex:3];
    [tail_enc setBytes:&tail_count length:sizeof(tail_count) atIndex:4];
    NSUInteger width = tail_pipeline.threadExecutionWidth;
    if (width == 0) {
      width = 32;
    }
    NSUInteger max_threads = tail_pipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger w = width * 4;
    if (w > max_threads) {
      w = max_threads;
    }
    if (w == 0) {
      w = 1;
    }
    MTLSize grid = MTLSizeMake(tail_count, 1, 1);
    MTLSize threads = MTLSizeMake(w, 1, 1);
    [tail_enc dispatchThreads:grid threadsPerThreadgroup:threads];
    [tail_enc endEncoding];
  }

  [command commit];

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

}  // namespace

runtime::Status matrixSubMetal(const float* a, const float* b, float* out, std::size_t rows, std::size_t cols) {
  return matrixSubMetalWithPolicy(a, b, out, rows, cols, true, true, true, true);
}

runtime::Status matrixSubMetal(const double* a, const double* b, double* out, std::size_t rows, std::size_t cols) {
  (void)a;
  (void)b;
  (void)out;
  (void)rows;
  (void)cols;
  return runtime::Status::kNotSupported;
}

runtime::Status matrixSubMetal(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols) {
  (void)a;
  (void)b;
  (void)out;
  (void)rows;
  (void)cols;
  return runtime::Status::kNotSupported;
}

runtime::Status matrixSubMetalWithPolicy(
    const float* a,
    const float* b,
    float* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  @autoreleasepool {
    runtime::Status st = ensureContextReady(rows * cols * sizeof(float));
    if (st != runtime::Status::kSuccess) {
      return st;
    }
    MatrixElemwiseMetalContext& ctx = getContext();
    return runKernel(
        ctx.sub_pipeline,
        ctx.sub_vec4_pipeline,
      ctx.sub_tail_pipeline,
        a,
        b,
        out,
        rows,
        cols,
        upload_a,
        upload_b,
        download_out,
        synchronize);
  }
}

runtime::Status matrixSubMetalWithPolicy(
    const double* a,
    const double* b,
    double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)a;
  (void)b;
  (void)out;
  (void)rows;
  (void)cols;
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return runtime::Status::kNotSupported;
}

runtime::Status matrixSubMetalWithPolicy(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)a;
  (void)b;
  (void)out;
  (void)rows;
  (void)cols;
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return runtime::Status::kNotSupported;
}

runtime::Status matrixDivMetal(const float* a, const float* b, float* out, std::size_t rows, std::size_t cols) {
  return matrixDivMetalWithPolicy(a, b, out, rows, cols, true, true, true, true);
}

runtime::Status matrixDivMetal(const double* a, const double* b, double* out, std::size_t rows, std::size_t cols) {
  (void)a;
  (void)b;
  (void)out;
  (void)rows;
  (void)cols;
  return runtime::Status::kNotSupported;
}

runtime::Status matrixDivMetal(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols) {
  (void)a;
  (void)b;
  (void)out;
  (void)rows;
  (void)cols;
  return runtime::Status::kNotSupported;
}

runtime::Status matrixDivMetalWithPolicy(
    const float* a,
    const float* b,
    float* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  @autoreleasepool {
    runtime::Status st = ensureContextReady(rows * cols * sizeof(float));
    if (st != runtime::Status::kSuccess) {
      return st;
    }
    MatrixElemwiseMetalContext& ctx = getContext();
    return runKernel(
        ctx.div_pipeline,
        ctx.div_vec4_pipeline,
      ctx.div_tail_pipeline,
        a,
        b,
        out,
        rows,
        cols,
        upload_a,
        upload_b,
        download_out,
        synchronize);
  }
}

runtime::Status matrixDivMetalWithPolicy(
    const double* a,
    const double* b,
    double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)a;
  (void)b;
  (void)out;
  (void)rows;
  (void)cols;
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return runtime::Status::kNotSupported;
}

runtime::Status matrixDivMetalWithPolicy(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)a;
  (void)b;
  (void)out;
  (void)rows;
  (void)cols;
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return runtime::Status::kNotSupported;
}

}  // namespace lightning_core::detail

#else

namespace lightning_core::detail {

runtime::Status matrixSubMetal(const float* a, const float* b, float* out, std::size_t rows, std::size_t cols) {
  (void)a;
  (void)b;
  (void)out;
  (void)rows;
  (void)cols;
  return runtime::Status::kNotSupported;
}

runtime::Status matrixSubMetal(const double* a, const double* b, double* out, std::size_t rows, std::size_t cols) {
  (void)a;
  (void)b;
  (void)out;
  (void)rows;
  (void)cols;
  return runtime::Status::kNotSupported;
}

runtime::Status matrixSubMetal(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols) {
  (void)a;
  (void)b;
  (void)out;
  (void)rows;
  (void)cols;
  return runtime::Status::kNotSupported;
}

runtime::Status matrixSubMetalWithPolicy(
    const float* a,
    const float* b,
    float* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)a;
  (void)b;
  (void)out;
  (void)rows;
  (void)cols;
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return runtime::Status::kNotSupported;
}

runtime::Status matrixSubMetalWithPolicy(
    const double* a,
    const double* b,
    double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)a;
  (void)b;
  (void)out;
  (void)rows;
  (void)cols;
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return runtime::Status::kNotSupported;
}

runtime::Status matrixSubMetalWithPolicy(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)a;
  (void)b;
  (void)out;
  (void)rows;
  (void)cols;
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return runtime::Status::kNotSupported;
}

runtime::Status matrixDivMetal(const float* a, const float* b, float* out, std::size_t rows, std::size_t cols) {
  (void)a;
  (void)b;
  (void)out;
  (void)rows;
  (void)cols;
  return runtime::Status::kNotSupported;
}

runtime::Status matrixDivMetal(const double* a, const double* b, double* out, std::size_t rows, std::size_t cols) {
  (void)a;
  (void)b;
  (void)out;
  (void)rows;
  (void)cols;
  return runtime::Status::kNotSupported;
}

runtime::Status matrixDivMetal(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols) {
  (void)a;
  (void)b;
  (void)out;
  (void)rows;
  (void)cols;
  return runtime::Status::kNotSupported;
}

runtime::Status matrixDivMetalWithPolicy(
    const float* a,
    const float* b,
    float* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)a;
  (void)b;
  (void)out;
  (void)rows;
  (void)cols;
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return runtime::Status::kNotSupported;
}

runtime::Status matrixDivMetalWithPolicy(
    const double* a,
    const double* b,
    double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)a;
  (void)b;
  (void)out;
  (void)rows;
  (void)cols;
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return runtime::Status::kNotSupported;
}

runtime::Status matrixDivMetalWithPolicy(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)a;
  (void)b;
  (void)out;
  (void)rows;
  (void)cols;
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return runtime::Status::kNotSupported;
}

}  // namespace lightning_core::detail

#endif
