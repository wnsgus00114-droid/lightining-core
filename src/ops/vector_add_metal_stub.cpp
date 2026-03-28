#include "cudajun/detail/ops_backend.hpp"

namespace cudajun::detail {

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

}  // namespace cudajun::detail
