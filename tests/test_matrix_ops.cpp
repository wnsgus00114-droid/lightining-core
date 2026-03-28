#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "cudajun/ops.hpp"
#include "cudajun/runtime.hpp"

namespace {

bool nearlyEqual(float a, float b) {
  return std::fabs(a - b) < 1e-5f;
}

int runSubCaseCpu() {
  const std::size_t rows = 2;
  const std::size_t cols = 3;

  const std::vector<float> a = {
      10.0f, 9.0f, 8.0f,
      7.0f,  6.0f, 5.0f,
  };
  const std::vector<float> b = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
  };
  std::vector<float> out(rows * cols, 0.0f);

  auto st = cudajun::ops::matrixSub<float>(
      a.data(), b.data(), out.data(), rows, cols, cudajun::runtime::Device::kCPU);
  if (st != cudajun::runtime::Status::kSuccess) {
    std::cerr << "matrixSub CPU failed\n";
    return 1;
  }

  const std::vector<float> expected = {
      9.0f, 7.0f, 5.0f,
      3.0f, 1.0f, -1.0f,
  };

  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (!nearlyEqual(out[i], expected[i])) {
      std::cerr << "matrixSub mismatch at " << i << " got=" << out[i] << " expected=" << expected[i] << "\n";
      return 1;
    }
  }

  return 0;
}

int runDivCaseCpu() {
  const std::size_t rows = 2;
  const std::size_t cols = 3;

  const std::vector<float> a = {
      10.0f, 9.0f, 8.0f,
      7.0f,  6.0f, 5.0f,
  };
  const std::vector<float> b = {
      2.0f, 3.0f, 4.0f,
      7.0f, 2.0f, 5.0f,
  };
  std::vector<float> out(rows * cols, 0.0f);

  auto st = cudajun::ops::matrixDiv<float>(
      a.data(), b.data(), out.data(), rows, cols, cudajun::runtime::Device::kCPU);
  if (st != cudajun::runtime::Status::kSuccess) {
    std::cerr << "matrixDiv CPU failed\n";
    return 1;
  }

  const std::vector<float> expected = {
      5.0f, 3.0f, 2.0f,
      1.0f, 3.0f, 1.0f,
  };

  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (!nearlyEqual(out[i], expected[i])) {
      std::cerr << "matrixDiv mismatch at " << i << " got=" << out[i] << " expected=" << expected[i] << "\n";
      return 1;
    }
  }

  return 0;
}

int runPolicyFallbackSafetyCaseCuda() {
  const std::size_t rows = 2;
  const std::size_t cols = 3;
  std::vector<float> out(rows * cols, 0.0f);

  cudajun::ops::MatrixElementwiseIoPolicy p;
  p.upload_a = false;
  p.upload_b = false;
  p.download_out = true;
  p.synchronize = true;

  auto sub_st = cudajun::ops::matrixSubWithPolicy<float>(
      nullptr,
      nullptr,
      out.data(),
      rows,
      cols,
      cudajun::runtime::Device::kCUDA,
      p);

  if (sub_st == cudajun::runtime::Status::kSuccess) {
    std::cerr << "matrixSub fallback safety unexpectedly succeeded\n";
    return 1;
  }

  return 0;
}

int runMetalPolicyCase() {
  if (!cudajun::runtime::isMetalAvailable()) {
    return 0;
  }

  const std::size_t rows = 2;
  const std::size_t cols = 3;
  const std::vector<float> a = {
      10.0f, 9.0f, 8.0f,
      7.0f,  6.0f, 5.0f,
  };
  const std::vector<float> b = {
      2.0f, 3.0f, 4.0f,
      7.0f, 2.0f, 5.0f,
  };

  std::vector<float> sub_out(rows * cols, 0.0f);
  std::vector<float> div_out(rows * cols, 0.0f);

  cudajun::ops::MatrixElemwiseMetalResidentSession<float> resident(rows, cols);

  auto st = resident.subStart(a.data(), b.data(), sub_out.data());
  if (st != cudajun::runtime::Status::kSuccess) {
    std::cerr << "MatrixElemwiseMetalResidentSession.subStart failed\n";
    return 1;
  }

  st = resident.subRun(a.data(), b.data(), sub_out.data());
  if (st != cudajun::runtime::Status::kSuccess) {
    std::cerr << "MatrixElemwiseMetalResidentSession.subRun failed\n";
    return 1;
  }

  st = resident.subFinish(a.data(), b.data(), sub_out.data());
  if (st != cudajun::runtime::Status::kSuccess) {
    std::cerr << "MatrixElemwiseMetalResidentSession.subFinish failed\n";
    return 1;
  }

  const std::vector<float> sub_expected = {
      8.0f, 6.0f, 4.0f,
      0.0f, 4.0f, 0.0f,
  };
  for (std::size_t i = 0; i < sub_expected.size(); ++i) {
    if (!nearlyEqual(sub_out[i], sub_expected[i])) {
      std::cerr << "matrixSub Metal mismatch at " << i << " got=" << sub_out[i] << " expected=" << sub_expected[i] << "\n";
      return 1;
    }
  }

  st = resident.divStart(a.data(), b.data(), div_out.data());
  if (st != cudajun::runtime::Status::kSuccess) {
    std::cerr << "MatrixElemwiseMetalResidentSession.divStart failed\n";
    return 1;
  }

  st = resident.divRun(a.data(), b.data(), div_out.data());
  if (st != cudajun::runtime::Status::kSuccess) {
    std::cerr << "MatrixElemwiseMetalResidentSession.divRun failed\n";
    return 1;
  }

  st = resident.divFinish(a.data(), b.data(), div_out.data());
  if (st != cudajun::runtime::Status::kSuccess) {
    std::cerr << "MatrixElemwiseMetalResidentSession.divFinish failed\n";
    return 1;
  }

  const std::vector<float> div_expected = {
      5.0f, 3.0f, 2.0f,
      1.0f, 3.0f, 1.0f,
  };
  for (std::size_t i = 0; i < div_expected.size(); ++i) {
    if (!nearlyEqual(div_out[i], div_expected[i])) {
      std::cerr << "matrixDiv Metal mismatch at " << i << " got=" << div_out[i] << " expected=" << div_expected[i] << "\n";
      return 1;
    }
  }

  return 0;
}

int runPolicyValidCaseCpu() {
  const std::size_t rows = 2;
  const std::size_t cols = 2;

  const std::vector<float> a = {
      8.0f, 4.0f,
      6.0f, 2.0f,
  };
  const std::vector<float> b = {
      2.0f, 2.0f,
      3.0f, 1.0f,
  };
  std::vector<float> out(rows * cols, 0.0f);

  cudajun::ops::MatrixElementwiseIoPolicy p;
  p.upload_a = true;
  p.upload_b = true;
  p.download_out = true;
  p.synchronize = true;

  auto st = cudajun::ops::matrixDivWithPolicy<float>(
      a.data(),
      b.data(),
      out.data(),
      rows,
      cols,
      cudajun::runtime::Device::kCPU,
      p);

  if (st != cudajun::runtime::Status::kSuccess) {
    std::cerr << "matrixDivWithPolicy CPU failed\n";
    return 1;
  }

  const std::vector<float> expected = {
      4.0f, 2.0f,
      2.0f, 2.0f,
  };

  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (!nearlyEqual(out[i], expected[i])) {
      std::cerr << "matrixDivWithPolicy mismatch at " << i << " got=" << out[i] << " expected=" << expected[i] << "\n";
      return 1;
    }
  }

  return 0;
}

}  // namespace

int main() {
  if (runSubCaseCpu() != 0) {
    return 1;
  }
  if (runDivCaseCpu() != 0) {
    return 1;
  }
  if (runPolicyFallbackSafetyCaseCuda() != 0) {
    return 1;
  }
  if (runMetalPolicyCase() != 0) {
    return 1;
  }
  if (runPolicyValidCaseCpu() != 0) {
    return 1;
  }
  return 0;
}
