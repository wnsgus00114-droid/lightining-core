#pragma once

#include <cstddef>

#include "lightning_core/core/runtime.hpp"

namespace lightning_core::detail {

// CPU SIMD 경로.
runtime::Status vectorAddCpu(const float* a, const float* b, float* out, std::size_t n);
runtime::Status vectorAddCpu(const double* a, const double* b, double* out, std::size_t n);
runtime::Status vectorAddCpu(const long double* a, const long double* b, long double* out, std::size_t n);

// CUDA 경로.
runtime::Status vectorAddCuda(const float* a, const float* b, float* out, std::size_t n);
runtime::Status vectorAddCuda(const double* a, const double* b, double* out, std::size_t n);
runtime::Status vectorAddCuda(const long double* a, const long double* b, long double* out, std::size_t n);

// Metal 경로 (macOS).
runtime::Status vectorAddMetal(const float* a, const float* b, float* out, std::size_t n);
runtime::Status vectorAddMetal(const double* a, const double* b, double* out, std::size_t n);
runtime::Status vectorAddMetal(const long double* a, const long double* b, long double* out, std::size_t n);

runtime::Status vectorAddCpuWithPolicy(
    const float* a,
    const float* b,
    float* out,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);
runtime::Status vectorAddCpuWithPolicy(
    const double* a,
    const double* b,
    double* out,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);
runtime::Status vectorAddCpuWithPolicy(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);

runtime::Status vectorAddCudaWithPolicy(
    const float* a,
    const float* b,
    float* out,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);
runtime::Status vectorAddCudaWithPolicy(
    const double* a,
    const double* b,
    double* out,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);
runtime::Status vectorAddCudaWithPolicy(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);

runtime::Status vectorAddMetalWithPolicy(
    const float* a,
    const float* b,
    float* out,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);
runtime::Status vectorAddMetalWithPolicy(
    const double* a,
    const double* b,
    double* out,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);
runtime::Status vectorAddMetalWithPolicy(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t n,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);

// M x K, K x N -> M x N 행렬곱 경로.
runtime::Status matMulCpu(
    const float* a,
    const float* b,
    float* out,
    std::size_t m,
    std::size_t k,
    std::size_t n);
runtime::Status matMulCpu(
    const double* a,
    const double* b,
    double* out,
    std::size_t m,
    std::size_t k,
    std::size_t n);
runtime::Status matMulCpu(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t m,
    std::size_t k,
    std::size_t n);

runtime::Status matMulCuda(
    const float* a,
    const float* b,
    float* out,
    std::size_t m,
    std::size_t k,
    std::size_t n);
runtime::Status matMulCuda(
    const double* a,
    const double* b,
    double* out,
    std::size_t m,
    std::size_t k,
    std::size_t n);
runtime::Status matMulCuda(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t m,
    std::size_t k,
    std::size_t n);

runtime::Status matMulMetal(
    const float* a,
    const float* b,
    float* out,
    std::size_t m,
    std::size_t k,
    std::size_t n);
runtime::Status matMulMetal(
    const double* a,
    const double* b,
    double* out,
    std::size_t m,
    std::size_t k,
    std::size_t n);
runtime::Status matMulMetal(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t m,
    std::size_t k,
    std::size_t n);

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
    bool synchronize);
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
    bool synchronize);
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
    bool synchronize);

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
    bool synchronize);
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
    bool synchronize);
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
    bool synchronize);

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
    bool synchronize);
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
    bool synchronize);
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
    bool synchronize);

// M x N 원소별 행렬 뺄셈/나눗셈 경로.
runtime::Status matrixSubCpu(
    const float* a,
    const float* b,
    float* out,
    std::size_t rows,
    std::size_t cols);
runtime::Status matrixSubCpu(
    const double* a,
    const double* b,
    double* out,
    std::size_t rows,
    std::size_t cols);
runtime::Status matrixSubCpu(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols);

runtime::Status matrixSubCuda(
    const float* a,
    const float* b,
    float* out,
    std::size_t rows,
    std::size_t cols);
runtime::Status matrixSubCuda(
    const double* a,
    const double* b,
    double* out,
    std::size_t rows,
    std::size_t cols);
runtime::Status matrixSubCuda(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols);

runtime::Status matrixSubMetal(
    const float* a,
    const float* b,
    float* out,
    std::size_t rows,
    std::size_t cols);
runtime::Status matrixSubMetal(
    const double* a,
    const double* b,
    double* out,
    std::size_t rows,
    std::size_t cols);
runtime::Status matrixSubMetal(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols);

runtime::Status matrixSubCpuWithPolicy(
    const float* a,
    const float* b,
    float* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);
runtime::Status matrixSubCpuWithPolicy(
    const double* a,
    const double* b,
    double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);
runtime::Status matrixSubCpuWithPolicy(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);

runtime::Status matrixSubCudaWithPolicy(
    const float* a,
    const float* b,
    float* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);
runtime::Status matrixSubCudaWithPolicy(
    const double* a,
    const double* b,
    double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);
runtime::Status matrixSubCudaWithPolicy(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);

runtime::Status matrixSubMetalWithPolicy(
    const float* a,
    const float* b,
    float* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);
runtime::Status matrixSubMetalWithPolicy(
    const double* a,
    const double* b,
    double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);
runtime::Status matrixSubMetalWithPolicy(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);

runtime::Status matrixDivCpu(
    const float* a,
    const float* b,
    float* out,
    std::size_t rows,
    std::size_t cols);
runtime::Status matrixDivCpu(
    const double* a,
    const double* b,
    double* out,
    std::size_t rows,
    std::size_t cols);
runtime::Status matrixDivCpu(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols);

runtime::Status matrixDivCuda(
    const float* a,
    const float* b,
    float* out,
    std::size_t rows,
    std::size_t cols);
runtime::Status matrixDivCuda(
    const double* a,
    const double* b,
    double* out,
    std::size_t rows,
    std::size_t cols);
runtime::Status matrixDivCuda(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols);

runtime::Status matrixDivMetal(
    const float* a,
    const float* b,
    float* out,
    std::size_t rows,
    std::size_t cols);
runtime::Status matrixDivMetal(
    const double* a,
    const double* b,
    double* out,
    std::size_t rows,
    std::size_t cols);
runtime::Status matrixDivMetal(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols);

runtime::Status matrixDivCpuWithPolicy(
    const float* a,
    const float* b,
    float* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);
runtime::Status matrixDivCpuWithPolicy(
    const double* a,
    const double* b,
    double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);
runtime::Status matrixDivCpuWithPolicy(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);

runtime::Status matrixDivCudaWithPolicy(
    const float* a,
    const float* b,
    float* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);
runtime::Status matrixDivCudaWithPolicy(
    const double* a,
    const double* b,
    double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);
runtime::Status matrixDivCudaWithPolicy(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);

runtime::Status matrixDivMetalWithPolicy(
    const float* a,
    const float* b,
    float* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);
runtime::Status matrixDivMetalWithPolicy(
    const double* a,
    const double* b,
    double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);
runtime::Status matrixDivMetalWithPolicy(
    const long double* a,
    const long double* b,
    long double* out,
    std::size_t rows,
    std::size_t cols,
    bool upload_a,
    bool upload_b,
    bool download_out,
    bool synchronize);

}  // namespace lightning_core::detail
