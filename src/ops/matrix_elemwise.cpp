#include "lightning_core/core/detail/ops_backend.hpp"

namespace lightning_core::detail {

namespace {

template <typename T>
runtime::Status matrixSubCpuImpl(const T* a, const T* b, T* out, std::size_t rows, std::size_t cols) {
  if (a == nullptr || b == nullptr || out == nullptr || rows == 0 || cols == 0) {
    return runtime::Status::kInvalidValue;
  }
  const std::size_t n = rows * cols;
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = a[i] - b[i];
  }
  return runtime::Status::kSuccess;
}

template <typename T>
runtime::Status matrixDivCpuImpl(const T* a, const T* b, T* out, std::size_t rows, std::size_t cols) {
  if (a == nullptr || b == nullptr || out == nullptr || rows == 0 || cols == 0) {
    return runtime::Status::kInvalidValue;
  }
  const std::size_t n = rows * cols;
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = a[i] / b[i];
  }
  return runtime::Status::kSuccess;
}

template <typename T>
runtime::Status notSupportedMatrixElemwise(const T* a, const T* b, T* out, std::size_t rows, std::size_t cols) {
  (void)a;
  (void)b;
  (void)out;
  (void)rows;
  (void)cols;
  return runtime::Status::kNotSupported;
}

template <typename T>
runtime::Status matrixSubCpuWithPolicyImpl(
    const T* a,
    const T* b,
    T* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return matrixSubCpuImpl(a, b, out, rows, cols);
}

template <typename T>
runtime::Status matrixDivCpuWithPolicyImpl(
    const T* a,
    const T* b,
    T* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return matrixDivCpuImpl(a, b, out, rows, cols);
}

template <typename T>
runtime::Status matrixSubBackendWithPolicyImpl(
    const T* a,
    const T* b,
    T* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return notSupportedMatrixElemwise(a, b, out, rows, cols);
}

template <typename T>
runtime::Status matrixDivBackendWithPolicyImpl(
    const T* a,
    const T* b,
    T* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  (void)upload_a;
  (void)upload_b;
  (void)download_out;
  (void)synchronize;
  return notSupportedMatrixElemwise(a, b, out, rows, cols);
}

}  // namespace

runtime::Status matrixSubCpu(const float* a, const float* b, float* out, std::size_t rows, std::size_t cols) {
  return matrixSubCpuImpl(a, b, out, rows, cols);
}

runtime::Status matrixSubCpu(const double* a, const double* b, double* out, std::size_t rows, std::size_t cols) {
  return matrixSubCpuImpl(a, b, out, rows, cols);
}

runtime::Status matrixSubCpu(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols) {
  return matrixSubCpuImpl(a, b, out, rows, cols);
}

runtime::Status matrixSubCuda(const float* a, const float* b, float* out, std::size_t rows, std::size_t cols) {
  return notSupportedMatrixElemwise(a, b, out, rows, cols);
}

runtime::Status matrixSubCuda(const double* a, const double* b, double* out, std::size_t rows, std::size_t cols) {
  return notSupportedMatrixElemwise(a, b, out, rows, cols);
}

runtime::Status matrixSubCuda(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols) {
  return notSupportedMatrixElemwise(a, b, out, rows, cols);
}


runtime::Status matrixSubCpuWithPolicy(
    const float* a,
    const float* b,
    float* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  return matrixSubCpuWithPolicyImpl(a, b, out, rows, cols, upload_a, upload_b, download_out, synchronize);
}

runtime::Status matrixSubCpuWithPolicy(
    const double* a,
    const double* b,
    double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  return matrixSubCpuWithPolicyImpl(a, b, out, rows, cols, upload_a, upload_b, download_out, synchronize);
}

runtime::Status matrixSubCpuWithPolicy(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  return matrixSubCpuWithPolicyImpl(a, b, out, rows, cols, upload_a, upload_b, download_out, synchronize);
}

runtime::Status matrixSubCudaWithPolicy(
    const float* a,
    const float* b,
    float* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  return matrixSubBackendWithPolicyImpl(a, b, out, rows, cols, upload_a, upload_b, download_out, synchronize);
}

runtime::Status matrixSubCudaWithPolicy(
    const double* a,
    const double* b,
    double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  return matrixSubBackendWithPolicyImpl(a, b, out, rows, cols, upload_a, upload_b, download_out, synchronize);
}

runtime::Status matrixSubCudaWithPolicy(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  return matrixSubBackendWithPolicyImpl(a, b, out, rows, cols, upload_a, upload_b, download_out, synchronize);
}


runtime::Status matrixDivCpu(const float* a, const float* b, float* out, std::size_t rows, std::size_t cols) {
  return matrixDivCpuImpl(a, b, out, rows, cols);
}

runtime::Status matrixDivCpu(const double* a, const double* b, double* out, std::size_t rows, std::size_t cols) {
  return matrixDivCpuImpl(a, b, out, rows, cols);
}

runtime::Status matrixDivCpu(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols) {
  return matrixDivCpuImpl(a, b, out, rows, cols);
}

runtime::Status matrixDivCuda(const float* a, const float* b, float* out, std::size_t rows, std::size_t cols) {
  return notSupportedMatrixElemwise(a, b, out, rows, cols);
}

runtime::Status matrixDivCuda(const double* a, const double* b, double* out, std::size_t rows, std::size_t cols) {
  return notSupportedMatrixElemwise(a, b, out, rows, cols);
}

runtime::Status matrixDivCuda(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols) {
  return notSupportedMatrixElemwise(a, b, out, rows, cols);
}


runtime::Status matrixDivCpuWithPolicy(
    const float* a,
    const float* b,
    float* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  return matrixDivCpuWithPolicyImpl(a, b, out, rows, cols, upload_a, upload_b, download_out, synchronize);
}

runtime::Status matrixDivCpuWithPolicy(
    const double* a,
    const double* b,
    double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  return matrixDivCpuWithPolicyImpl(a, b, out, rows, cols, upload_a, upload_b, download_out, synchronize);
}

runtime::Status matrixDivCpuWithPolicy(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  return matrixDivCpuWithPolicyImpl(a, b, out, rows, cols, upload_a, upload_b, download_out, synchronize);
}

runtime::Status matrixDivCudaWithPolicy(
    const float* a,
    const float* b,
    float* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  return matrixDivBackendWithPolicyImpl(a, b, out, rows, cols, upload_a, upload_b, download_out, synchronize);
}

runtime::Status matrixDivCudaWithPolicy(
    const double* a,
    const double* b,
    double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  return matrixDivBackendWithPolicyImpl(a, b, out, rows, cols, upload_a, upload_b, download_out, synchronize);
}

runtime::Status matrixDivCudaWithPolicy(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize) {
  return matrixDivBackendWithPolicyImpl(a, b, out, rows, cols, upload_a, upload_b, download_out, synchronize);
}

}  // namespace lightning_core::detail
