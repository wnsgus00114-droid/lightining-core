#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "cudajun/ops.hpp"
#include "cudajun/runtime.hpp"

namespace {

std::size_t readSizeEnv(const char* key, std::size_t defaultValue) {
  const char* raw = std::getenv(key);
  if (raw == nullptr || raw[0] == '\0') {
    return defaultValue;
  }
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(raw, &end, 10);
  if (end == raw || *end != '\0' || parsed == 0) {
    return defaultValue;
  }
  return static_cast<std::size_t>(parsed);
}

int readIntEnv(const char* key, int defaultValue) {
  const char* raw = std::getenv(key);
  if (raw == nullptr || raw[0] == '\0') {
    return defaultValue;
  }
  char* end = nullptr;
  long parsed = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || parsed <= 0) {
    return defaultValue;
  }
  return static_cast<int>(parsed);
}

double runMatMul(
    cudajun::runtime::Device device,
    std::size_t m,
    std::size_t k,
    std::size_t n,
    int warmup,
    int iters,
    int batch) {
  std::vector<float> a(m * k, 0.25f);
  std::vector<float> b(k * n, 0.5f);
  std::vector<float> out(m * n, 0.0f);

  for (int i = 0; i < warmup; ++i) {
    for (int bidx = 0; bidx < batch; ++bidx) {
      auto st = cudajun::ops::matMul<float>(a.data(), b.data(), out.data(), m, k, n, device);
      if (st != cudajun::runtime::Status::kSuccess) {
        return -1.0;
      }
    }
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    for (int bidx = 0; bidx < batch; ++bidx) {
      auto st = cudajun::ops::matMul<float>(a.data(), b.data(), out.data(), m, k, n, device);
      if (st != cudajun::runtime::Status::kSuccess) {
        return -1.0;
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  return elapsed.count() / static_cast<double>(iters * batch);
}

void printResult(const char* label, double value) {
  if (value > 0.0) {
    std::cout << label << ": " << value << " ms\n";
  } else {
    std::cout << label << ": unavailable\n";
  }
}

}  // namespace

int main() {
  cudajun::runtime::preloadRuntimeProfileEnv();

  const std::size_t m = readSizeEnv("CJ_MM_M", 1024);
  const std::size_t k = readSizeEnv("CJ_MM_K", 1024);
  const std::size_t n = readSizeEnv("CJ_MM_N", 1024);
  const int warmup = readIntEnv("CJ_MM_WARMUP", 10);
  const int iters = readIntEnv("CJ_MM_ITERS", 30);
  const int batch = readIntEnv("CJ_MM_BATCH", 2);

  std::cout << "[bench] matmul m=" << m << " k=" << k << " n=" << n
            << " warmup=" << warmup << " iters=" << iters << " batch=" << batch << "\n";

  double cpu_ms = runMatMul(cudajun::runtime::Device::kCPU, m, k, n, warmup, iters, batch);
  printResult("CPU matmul", cpu_ms);

  double metal_ms = runMatMul(cudajun::runtime::Device::kMetal, m, k, n, warmup, iters, batch);
  printResult("Metal matmul", metal_ms);

  if (cpu_ms > 0.0 && metal_ms > 0.0) {
    std::cout << "Speedup matmul (CPU/Metal): " << (cpu_ms / metal_ms) << "x\n";
  }

  return 0;
}
