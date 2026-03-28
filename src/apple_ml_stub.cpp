#include "cudajun/apple_ml.hpp"

namespace cudajun::apple {

runtime::Status benchmarkCoreMLInference(const std::string& modelPath, std::size_t n, int iters, double* avgMs) {
  (void)modelPath;
  (void)n;
  (void)iters;
  if (avgMs != nullptr) {
    *avgMs = -1.0;
  }
  return runtime::Status::kNotSupported;
}

runtime::Status benchmarkMpsGraphVectorAdd(std::size_t n, int iters, double* avgMs) {
  (void)n;
  (void)iters;
  if (avgMs != nullptr) {
    *avgMs = -1.0;
  }
  return runtime::Status::kNotSupported;
}

runtime::Status benchmarkMpsGraphTrainStep(std::size_t n, int iters, double* avgMs) {
  (void)n;
  (void)iters;
  if (avgMs != nullptr) {
    *avgMs = -1.0;
  }
  return runtime::Status::kNotSupported;
}

}  // namespace cudajun::apple
