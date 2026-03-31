#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "lightning_core/attention.hpp"
#include "lightning_core/ops.hpp"
#include "lightning_core/runtime.hpp"

namespace {

float makeValue(std::size_t i, float scale = 1.0f) {
  const std::int64_t v = static_cast<std::int64_t>((i * 37U + 11U) % 97U) - 48;
  return static_cast<float>(v) * scale;
}

std::vector<float> makeVec(std::size_t n, float scale = 0.01f, float bias = 0.0f) {
  std::vector<float> out(n, 0.0f);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = makeValue(i, scale) + bias;
  }
  return out;
}

bool closeEnough(const std::vector<float>& ref,
                 const std::vector<float>& got,
                 float atol,
                 float rtol,
                 const char* tag) {
  if (ref.size() != got.size()) {
    std::cerr << tag << " size mismatch: ref=" << ref.size() << " got=" << got.size() << "\n";
    return false;
  }

  float max_abs = 0.0f;
  float max_rel = 0.0f;
  std::size_t worst_i = 0;

  for (std::size_t i = 0; i < ref.size(); ++i) {
    const float a = ref[i];
    const float b = got[i];
    const float abs_diff = std::fabs(a - b);
    const float denom = std::max(std::fabs(a), 1.0e-12f);
    const float rel_diff = abs_diff / denom;
    if (abs_diff > max_abs) {
      max_abs = abs_diff;
      worst_i = i;
    }
    if (rel_diff > max_rel) {
      max_rel = rel_diff;
    }
    if (abs_diff > (atol + rtol * std::fabs(a))) {
      std::cerr << tag << " mismatch at " << i << ": ref=" << a << " got=" << b
                << " abs=" << abs_diff << " rel=" << rel_diff
                << " (atol=" << atol << ", rtol=" << rtol << ")\n";
      return false;
    }
  }

  std::cout << "[parity] " << tag << " ok"
            << " max_abs=" << max_abs << " max_rel=" << max_rel
            << " worst_i=" << worst_i << "\n";
  return true;
}

int testMatMulParity() {
  using lightning_core::runtime::Device;
  using lightning_core::runtime::Status;

  constexpr std::size_t m = 4;
  constexpr std::size_t k = 7;
  constexpr std::size_t n = 5;

  const std::vector<float> a = makeVec(m * k, 0.02f);
  const std::vector<float> b = makeVec(k * n, 0.015f, 0.01f);
  std::vector<float> cpu_out(m * n, 0.0f);
  std::vector<float> metal_out(m * n, 0.0f);

  if (lightning_core::ops::matMul<float>(a.data(), b.data(), cpu_out.data(), m, k, n, Device::kCPU) != Status::kSuccess) {
    std::cerr << "matmul cpu failed\n";
    return 1;
  }
  if (lightning_core::ops::matMul<float>(a.data(), b.data(), metal_out.data(), m, k, n, Device::kMetal) !=
      Status::kSuccess) {
    std::cerr << "matmul metal failed\n";
    return 1;
  }
  return closeEnough(cpu_out, metal_out, 2.0e-3f, 3.0e-3f, "matmul") ? 0 : 1;
}

int testVectorAddParity() {
  using lightning_core::runtime::Device;
  using lightning_core::runtime::Status;

  constexpr std::size_t n = 257;
  const std::vector<float> a = makeVec(n, 0.005f, 0.2f);
  const std::vector<float> b = makeVec(n, 0.003f, -0.1f);
  std::vector<float> cpu_out(n, 0.0f);
  std::vector<float> metal_out(n, 0.0f);

  if (lightning_core::ops::vectorAdd<float>(a.data(), b.data(), cpu_out.data(), n, Device::kCPU) != Status::kSuccess) {
    std::cerr << "vector_add cpu failed\n";
    return 1;
  }
  if (lightning_core::ops::vectorAdd<float>(a.data(), b.data(), metal_out.data(), n, Device::kMetal) !=
      Status::kSuccess) {
    std::cerr << "vector_add metal failed\n";
    return 1;
  }
  return closeEnough(cpu_out, metal_out, 1.0e-4f, 1.0e-4f, "vector_add") ? 0 : 1;
}

int testMatrixElemwiseParity() {
  using lightning_core::runtime::Device;
  using lightning_core::runtime::Status;

  constexpr std::size_t rows = 9;
  constexpr std::size_t cols = 13;
  const std::size_t n = rows * cols;
  const std::vector<float> a = makeVec(n, 0.01f, 0.7f);
  std::vector<float> b = makeVec(n, 0.008f, 0.5f);
  for (float& v : b) {
    if (std::fabs(v) < 0.1f) {
      v = (v < 0.0f) ? -0.1f : 0.1f;
    }
  }

  std::vector<float> sub_cpu(n, 0.0f), sub_metal(n, 0.0f);
  std::vector<float> div_cpu(n, 0.0f), div_metal(n, 0.0f);

  if (lightning_core::ops::matrixSub<float>(a.data(), b.data(), sub_cpu.data(), rows, cols, Device::kCPU) !=
      Status::kSuccess) {
    std::cerr << "matrix_sub cpu failed\n";
    return 1;
  }
  if (lightning_core::ops::matrixSub<float>(a.data(), b.data(), sub_metal.data(), rows, cols, Device::kMetal) !=
      Status::kSuccess) {
    std::cerr << "matrix_sub metal failed\n";
    return 1;
  }
  if (!closeEnough(sub_cpu, sub_metal, 1.0e-4f, 1.0e-4f, "matrix_sub")) {
    return 1;
  }

  if (lightning_core::ops::matrixDiv<float>(a.data(), b.data(), div_cpu.data(), rows, cols, Device::kCPU) !=
      Status::kSuccess) {
    std::cerr << "matrix_div cpu failed\n";
    return 1;
  }
  if (lightning_core::ops::matrixDiv<float>(a.data(), b.data(), div_metal.data(), rows, cols, Device::kMetal) !=
      Status::kSuccess) {
    std::cerr << "matrix_div metal failed\n";
    return 1;
  }
  return closeEnough(div_cpu, div_metal, 2.0e-4f, 2.0e-4f, "matrix_div") ? 0 : 1;
}

int testAttentionParity() {
  using lightning_core::AttentionConfig;
  using lightning_core::runtime::Device;
  using lightning_core::runtime::Status;

  constexpr std::size_t seq = 16;
  constexpr std::size_t dim = 32;
  const std::size_t n = seq * dim;

  const std::vector<float> q = makeVec(n, 0.01f);
  const std::vector<float> k = makeVec(n, 0.009f, 0.02f);
  const std::vector<float> v = makeVec(n, 0.008f, -0.01f);
  std::vector<float> cpu_out(n, 0.0f);
  std::vector<float> metal_out(n, 0.0f);

  AttentionConfig cfg{seq, dim, false};
  if (lightning_core::attentionForward(q.data(), k.data(), v.data(), cpu_out.data(), cfg, Device::kCPU) !=
      Status::kSuccess) {
    std::cerr << "attention cpu failed\n";
    return 1;
  }
  if (lightning_core::attentionForward(q.data(), k.data(), v.data(), metal_out.data(), cfg, Device::kMetal) !=
      Status::kSuccess) {
    std::cerr << "attention metal failed\n";
    return 1;
  }
  return closeEnough(cpu_out, metal_out, 2.0e-2f, 2.0e-2f, "attention_forward") ? 0 : 1;
}

int testConvParity() {
  using lightning_core::runtime::Device;
  using lightning_core::runtime::Status;

  constexpr std::size_t batch = 1;
  constexpr std::size_t in_c = 3;
  constexpr std::size_t in_h = 8;
  constexpr std::size_t in_w = 8;
  constexpr std::size_t out_c = 8;
  constexpr std::size_t k = 3;

  const std::vector<float> x = makeVec(batch * in_c * in_h * in_w, 0.01f, 0.05f);
  const std::vector<float> w = makeVec(out_c * in_c * k * k, 0.01f, -0.03f);
  const std::vector<float> bias = makeVec(out_c, 0.01f, 0.02f);

  std::vector<float> cpu_out(batch * out_c * in_h * in_w, 0.0f);
  std::vector<float> metal_out(batch * out_c * in_h * in_w, 0.0f);

  if (lightning_core::ops::conv2dNchw<float>(
          x.data(),
          w.data(),
          bias.data(),
          cpu_out.data(),
          batch,
          in_c,
          in_h,
          in_w,
          out_c,
          k,
          k,
          1,
          1,
          1,
          1,
          Device::kCPU,
          false) != Status::kSuccess) {
    std::cerr << "conv cpu failed\n";
    return 1;
  }

  if (lightning_core::ops::conv2dNchw<float>(
          x.data(),
          w.data(),
          bias.data(),
          metal_out.data(),
          batch,
          in_c,
          in_h,
          in_w,
          out_c,
          k,
          k,
          1,
          1,
          1,
          1,
          Device::kMetal,
          false) != Status::kSuccess) {
    std::cerr << "conv metal failed\n";
    return 1;
  }

  return closeEnough(cpu_out, metal_out, 1.5e-2f, 1.5e-2f, "conv2d_nchw") ? 0 : 1;
}

}  // namespace

int main() {
  if (!lightning_core::runtime::isMetalAvailable()) {
    std::cout << "[parity] metal unavailable; skip parity checks\n";
    return 0;
  }

  if (testMatMulParity() != 0) {
    return 1;
  }
  if (testVectorAddParity() != 0) {
    return 1;
  }
  if (testMatrixElemwiseParity() != 0) {
    return 1;
  }
  if (testAttentionParity() != 0) {
    return 1;
  }
  if (testConvParity() != 0) {
    return 1;
  }
  return 0;
}
