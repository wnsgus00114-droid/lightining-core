#include "lightning_core/core/detail/ops_backend.hpp"

namespace lightning_core::detail {

runtime::Status vectorAddCuda(const float* a, const float* b, float* out, std::size_t n) {
  // 스텁 구현: CUDA 파일이 아예 빌드되지 않는 환경에서 링킹용으로만 존재.
  // 호출되면 "지원 안 함"을 즉시 반환한다.
  (void)a;
  (void)b;
  (void)out;
  (void)n;
  return runtime::Status::kNotSupported;
}

runtime::Status vectorAddCuda(const double* a, const double* b, double* out, std::size_t n) {
  (void)a;
  (void)b;
  (void)out;
  (void)n;
  return runtime::Status::kNotSupported;
}

runtime::Status vectorAddCuda(const long double* a, const long double* b, long double* out, std::size_t n) {
  (void)a;
  (void)b;
  (void)out;
  (void)n;
  return runtime::Status::kNotSupported;
}

}  // namespace lightning_core::detail
