#include "lightning_core/core/detail/ops_backend.hpp"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace lightning_core::detail {

runtime::Status vectorAddCpu(const float* a, const float* b, float* out, std::size_t n) {
  // CPU 경로 방어 체크.
  if (a == nullptr || b == nullptr || out == nullptr || n == 0) {
    return runtime::Status::kInvalidValue;
  }

  std::size_t i = 0;

#if defined(__AVX2__)
  constexpr std::size_t kWidth = 8;
  for (; i + kWidth <= n; i += kWidth) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 vc = _mm256_add_ps(va, vb);
    _mm256_storeu_ps(out + i, vc);
  }
#elif defined(__ARM_NEON)
  constexpr std::size_t kWidth = 4;
  for (; i + kWidth <= n; i += kWidth) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    float32x4_t vc = vaddq_f32(va, vb);
    vst1q_f32(out + i, vc);
  }
#endif

  for (; i < n; ++i) {
    out[i] = a[i] + b[i];
  }
  return runtime::Status::kSuccess;
}

runtime::Status vectorAddCpu(const double* a, const double* b, double* out, std::size_t n) {
  if (a == nullptr || b == nullptr || out == nullptr || n == 0) {
    return runtime::Status::kInvalidValue;
  }

  std::size_t i = 0;

#if defined(__AVX2__)
  constexpr std::size_t kWidth = 4;
  for (; i + kWidth <= n; i += kWidth) {
    __m256d va = _mm256_loadu_pd(a + i);
    __m256d vb = _mm256_loadu_pd(b + i);
    __m256d vc = _mm256_add_pd(va, vb);
    _mm256_storeu_pd(out + i, vc);
  }
#elif defined(__ARM_NEON)
  constexpr std::size_t kWidth = 2;
  for (; i + kWidth <= n; i += kWidth) {
    float64x2_t va = vld1q_f64(a + i);
    float64x2_t vb = vld1q_f64(b + i);
    float64x2_t vc = vaddq_f64(va, vb);
    vst1q_f64(out + i, vc);
  }
#endif

  for (; i < n; ++i) {
    out[i] = a[i] + b[i];
  }
  return runtime::Status::kSuccess;
}

runtime::Status vectorAddCpu(const long double* a, const long double* b, long double* out, std::size_t n) {
  if (a == nullptr || b == nullptr || out == nullptr || n == 0) {
    return runtime::Status::kInvalidValue;
  }
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = a[i] + b[i];
  }
  return runtime::Status::kSuccess;
}

runtime::Status vectorAddCpuWithPolicy(
    const float* a,
    const float* b,
    float* out,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return vectorAddCpu(a, b, out, n);
}

runtime::Status vectorAddCpuWithPolicy(
    const double* a,
    const double* b,
    double* out,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return vectorAddCpu(a, b, out, n);
}

runtime::Status vectorAddCpuWithPolicy(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return vectorAddCpu(a, b, out, n);
}

runtime::Status vectorAddCudaWithPolicy(
    const float* a,
    const float* b,
    float* out,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return vectorAddCuda(a, b, out, n);
}

runtime::Status vectorAddCudaWithPolicy(
    const double* a,
    const double* b,
    double* out,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return vectorAddCuda(a, b, out, n);
}

runtime::Status vectorAddCudaWithPolicy(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return vectorAddCuda(a, b, out, n);
}

}  // namespace lightning_core::detail
