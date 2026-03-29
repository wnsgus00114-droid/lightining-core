#include "lightning_core/core/detail/ops_backend.hpp"

#include <algorithm>
#include <limits>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

namespace lightning_core::detail {

runtime::Status matMulCpu(
    const float* a,
    const float* b,
    float* out,
    std::size_t m,
    std::size_t k,
    std::size_t n) {
  if (a == nullptr || b == nullptr || out == nullptr || m == 0 || k == 0 || n == 0) {
    return runtime::Status::kInvalidValue;
  }

#if defined(__APPLE__)
  if (m <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
      k <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
      n <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    const int mm = static_cast<int>(m);
    const int kk = static_cast<int>(k);
    const int nn = static_cast<int>(n);
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        mm,
        nn,
        kk,
        1.0f,
        a,
        kk,
        b,
        nn,
        0.0f,
        out,
        nn);
    return runtime::Status::kSuccess;
  }
#endif

  std::fill(out, out + (m * n), 0.0f);
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t p = 0; p < k; ++p) {
      float av = a[i * k + p];
      const float* brow = b + p * n;
      float* orow = out + i * n;
      for (std::size_t j = 0; j < n; ++j) {
        orow[j] += av * brow[j];
      }
    }
  }
  return runtime::Status::kSuccess;
}

runtime::Status matMulCpu(
    const double* a,
    const double* b,
    double* out,
    std::size_t m,
    std::size_t k,
    std::size_t n) {
  if (a == nullptr || b == nullptr || out == nullptr || m == 0 || k == 0 || n == 0) {
    return runtime::Status::kInvalidValue;
  }

#if defined(__APPLE__)
  if (m <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
      k <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
      n <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    const int mm = static_cast<int>(m);
    const int kk = static_cast<int>(k);
    const int nn = static_cast<int>(n);
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        mm,
        nn,
        kk,
        1.0,
        a,
        kk,
        b,
        nn,
        0.0,
        out,
        nn);
    return runtime::Status::kSuccess;
  }
#endif

  std::fill(out, out + (m * n), 0.0);
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t p = 0; p < k; ++p) {
      double av = a[i * k + p];
      const double* brow = b + p * n;
      double* orow = out + i * n;
      for (std::size_t j = 0; j < n; ++j) {
        orow[j] += av * brow[j];
      }
    }
  }
  return runtime::Status::kSuccess;
}

runtime::Status matMulCpu(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t m,
    std::size_t k,
    std::size_t n) {
  if (a == nullptr || b == nullptr || out == nullptr || m == 0 || k == 0 || n == 0) {
    return runtime::Status::kInvalidValue;
  }

  std::fill(out, out + (m * n), static_cast<long double>(0));
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t p = 0; p < k; ++p) {
      long double av = a[i * k + p];
      const long double* brow = b + p * n;
      long double* orow = out + i * n;
      for (std::size_t j = 0; j < n; ++j) {
        orow[j] += av * brow[j];
      }
    }
  }
  return runtime::Status::kSuccess;
}

runtime::Status matMulCuda(
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

runtime::Status matMulCuda(
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

runtime::Status matMulCuda(
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

runtime::Status matMulCpuWithPolicy(
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
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return matMulCpu(a, b, out, m, k, n);
}

runtime::Status matMulCpuWithPolicy(
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
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return matMulCpu(a, b, out, m, k, n);
}

runtime::Status matMulCpuWithPolicy(
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
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return matMulCpu(a, b, out, m, k, n);
}

runtime::Status matMulCudaWithPolicy(
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
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return matMulCuda(a, b, out, m, k, n);
}

runtime::Status matMulCudaWithPolicy(
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
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return matMulCuda(a, b, out, m, k, n);
}

runtime::Status matMulCudaWithPolicy(
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
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return matMulCuda(a, b, out, m, k, n);
}

}  // namespace lightning_core::detail
