#include "lightning_core/core/detail/ops_backend.hpp"

namespace lightning_core::detail {

runtime::Status vectorAddCuda(const float* a, const float* b, float* out, std::size_t n) {
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
