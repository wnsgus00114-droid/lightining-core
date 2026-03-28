#include "cudajun/detail/ops_backend.hpp"

#if CJ_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace lightning_core::detail {

#if CJ_HAS_CUDA
namespace {

template <typename T>
__global__ void vectorAddKernel(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out, std::size_t n) {
  // grid-stride loop 패턴: 대형 벡터에서도 스레드가 균일하게 일을 나눠 먹는다.
  std::size_t idx = static_cast<std::size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (; idx < n; idx += stride) {
    out[idx] = a[idx] + b[idx];
  }
}

template <typename T>
runtime::Status launchVectorAdd(const T* a, const T* b, T* out, std::size_t n) {
  constexpr int kThreads = 256;
  int sm_count = 0;
  cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
  if (err != cudaSuccess || sm_count <= 0) {
    sm_count = 16;
  }
  int blocks = sm_count * 8;

  vectorAddKernel<T><<<blocks, kThreads>>>(a, b, out, n);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return runtime::Status::kUnknown;
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    return runtime::Status::kUnknown;
  }
  return runtime::Status::kSuccess;
}

}  // namespace
#endif

runtime::Status vectorAddCuda(const float* a, const float* b, float* out, std::size_t n) {
#if CJ_HAS_CUDA
  if (a == nullptr || b == nullptr || out == nullptr || n == 0) {
    return runtime::Status::kInvalidValue;
  }
  return launchVectorAdd<float>(a, b, out, n);
#else
  (void)a;
  (void)b;
  (void)out;
  (void)n;
  return runtime::Status::kNotSupported;
#endif
}

runtime::Status vectorAddCuda(const double* a, const double* b, double* out, std::size_t n) {
#if CJ_HAS_CUDA
  if (a == nullptr || b == nullptr || out == nullptr || n == 0) {
    return runtime::Status::kInvalidValue;
  }
  return launchVectorAdd<double>(a, b, out, n);
#else
  (void)a;
  (void)b;
  (void)out;
  (void)n;
  // CUDA가 꺼진 빌드에서는 명시적으로 미지원 반환.
  return runtime::Status::kNotSupported;
#endif
}

runtime::Status vectorAddCuda(const long double* a, const long double* b, long double* out, std::size_t n) {
  (void)a;
  (void)b;
  (void)out;
  (void)n;
  // 대부분의 CUDA 디바이스는 long double 커널 연산을 직접 지원하지 않으므로 명시적으로 미지원.
  return runtime::Status::kNotSupported;
}

}  // namespace lightning_core::detail
