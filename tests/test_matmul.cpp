#include <cmath>
#include <iostream>
#include <vector>

#include "lightning_core/ops.hpp"
#include "lightning_core/runtime.hpp"

namespace {

bool nearlyEqual(float a, float b) {
  return std::fabs(a - b) < 1e-4f;
}

int runCase(lightning_core::runtime::Device device) {
  // A(2x3) * B(3x2) = C(2x2)
  const std::size_t m = 2;
  const std::size_t k = 3;
  const std::size_t n = 2;

  const std::vector<float> a = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
  };
  const std::vector<float> b = {
      7.0f, 8.0f,
      9.0f, 10.0f,
      11.0f, 12.0f,
  };
  std::vector<float> c(m * n, 0.0f);

  const auto st = lightning_core::ops::matMul<float>(a.data(), b.data(), c.data(), m, k, n, device);
  if (st != lightning_core::runtime::Status::kSuccess) {
    std::cerr << "matMul failed\n";
    return 1;
  }

  const std::vector<float> expected = {
      58.0f, 64.0f,
      139.0f, 154.0f,
  };

  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (!nearlyEqual(c[i], expected[i])) {
      std::cerr << "Mismatch at " << i << " got=" << c[i] << " expected=" << expected[i] << "\n";
      return 1;
    }
  }

  return 0;
}

int runPolicyCaseMetal() {
  if (!lightning_core::runtime::isMetalAvailable()) {
    return 0;
  }

  const std::size_t m = 2;
  const std::size_t k = 3;
  const std::size_t n = 2;

  const std::vector<float> a = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
  };
  const std::vector<float> b = {
      7.0f, 8.0f,
      9.0f, 10.0f,
      11.0f, 12.0f,
  };
  std::vector<float> c(m * n, 0.0f);

  lightning_core::ops::MatMulMetalResidentSession<float> resident(m, k, n);

  auto st = resident.start(a.data(), b.data(), c.data());
  if (st != lightning_core::runtime::Status::kSuccess) {
    std::cerr << "MatMulMetalResidentSession.start failed\n";
    return 1;
  }

  st = resident.run(a.data(), b.data(), c.data());
  if (st != lightning_core::runtime::Status::kSuccess) {
    std::cerr << "MatMulMetalResidentSession.run failed\n";
    return 1;
  }

  st = resident.finish(a.data(), b.data(), c.data());
  if (st != lightning_core::runtime::Status::kSuccess) {
    std::cerr << "MatMulMetalResidentSession.finish failed\n";
    return 1;
  }

  const std::vector<float> expected = {
      58.0f, 64.0f,
      139.0f, 154.0f,
  };

  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (!nearlyEqual(c[i], expected[i])) {
      std::cerr << "Policy mismatch at " << i << " got=" << c[i] << " expected=" << expected[i] << "\n";
      return 1;
    }
  }

  return 0;
}

int runPolicyFallbackSafetyCase() {
  const std::size_t m = 2;
  const std::size_t k = 3;
  const std::size_t n = 2;
  std::vector<float> c(m * n, 0.0f);

  lightning_core::ops::MatMulIoPolicy p;
  p.upload_a = false;
  p.upload_b = false;
  p.download_out = true;
  p.synchronize = true;

  auto st = lightning_core::ops::matMulWithPolicy<float>(
      nullptr,
      nullptr,
      c.data(),
      m,
      k,
      n,
      lightning_core::runtime::Device::kCUDA,
      p);

  if (st == lightning_core::runtime::Status::kSuccess) {
    std::cerr << "Fallback safety case unexpectedly succeeded\n";
    return 1;
  }

  return 0;
}

}  // namespace

int main() {
  if (runCase(lightning_core::runtime::Device::kCPU) != 0) {
    return 1;
  }

  if (lightning_core::runtime::isMetalAvailable()) {
    if (runCase(lightning_core::runtime::Device::kMetal) != 0) {
      return 1;
    }
  }

  if (runPolicyCaseMetal() != 0) {
    return 1;
  }

  if (runPolicyFallbackSafetyCase() != 0) {
    return 1;
  }

  return 0;
}
