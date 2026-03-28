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

template <typename RunFn>
double runTimed(int warmup, int iters, int batch, RunFn&& fn) {
  for (int i = 0; i < warmup; ++i) {
    for (int b = 0; b < batch; ++b) {
      auto st = fn(i, b, true);
      if (st != cudajun::runtime::Status::kSuccess) {
        return -1.0;
      }
    }
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    for (int b = 0; b < batch; ++b) {
      auto st = fn(i, b, false);
      if (st != cudajun::runtime::Status::kSuccess) {
        return -1.0;
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  return elapsed.count() / static_cast<double>(iters * batch);
}

double runMatrixSub(
    cudajun::runtime::Device device,
    std::size_t rows,
    std::size_t cols,
    int warmup,
    int iters,
    int batch,
    bool resident_fastpath) {
  const std::size_t n = rows * cols;
  std::vector<float> a(n, 10.0f);
  std::vector<float> b(n, 2.0f);
  std::vector<float> out(n, 0.0f);

  if (device == cudajun::runtime::Device::kMetal && resident_fastpath) {
    cudajun::ops::MatrixElemwiseMetalResidentSession<float> session(rows, cols);
    auto fn = [&](int i, int bidx, bool is_warmup) {
      (void)is_warmup;
      if (i == 0 && bidx == 0) {
        return session.subStart(a.data(), b.data(), out.data());
      }
      return session.subRun(a.data(), b.data(), out.data());
    };

    double ms = runTimed(warmup, iters, batch, fn);
    if (ms <= 0.0) {
      return ms;
    }

    auto st = session.subFinish(a.data(), b.data(), out.data());
    if (st != cudajun::runtime::Status::kSuccess) {
      return -1.0;
    }

    return ms;
  }

  auto fn = [&](int i, int bidx, bool is_warmup) {
    (void)i;
    (void)bidx;
    (void)is_warmup;
    return cudajun::ops::matrixSub<float>(a.data(), b.data(), out.data(), rows, cols, device);
  };
  return runTimed(warmup, iters, batch, fn);
}

double runMatrixDiv(
    cudajun::runtime::Device device,
    std::size_t rows,
    std::size_t cols,
    int warmup,
    int iters,
    int batch,
    bool resident_fastpath) {
  const std::size_t n = rows * cols;
  std::vector<float> a(n, 10.0f);
  std::vector<float> b(n, 2.0f);
  std::vector<float> out(n, 0.0f);

  if (device == cudajun::runtime::Device::kMetal && resident_fastpath) {
    cudajun::ops::MatrixElemwiseMetalResidentSession<float> session(rows, cols);
    auto fn = [&](int i, int bidx, bool is_warmup) {
      (void)is_warmup;
      if (i == 0 && bidx == 0) {
        return session.divStart(a.data(), b.data(), out.data());
      }
      return session.divRun(a.data(), b.data(), out.data());
    };

    double ms = runTimed(warmup, iters, batch, fn);
    if (ms <= 0.0) {
      return ms;
    }

    auto st = session.divFinish(a.data(), b.data(), out.data());
    if (st != cudajun::runtime::Status::kSuccess) {
      return -1.0;
    }

    return ms;
  }

  auto fn = [&](int i, int bidx, bool is_warmup) {
    (void)i;
    (void)bidx;
    (void)is_warmup;
    return cudajun::ops::matrixDiv<float>(a.data(), b.data(), out.data(), rows, cols, device);
  };
  return runTimed(warmup, iters, batch, fn);
}

void printResult(const char* label, double value) {
  if (value > 0.0) {
    std::cout << label << ": " << value << " ms\n";
  } else {
    std::cout << label << ": unavailable\n";
  }
}

void printSpeedup(const char* label, double cpu_ms, double metal_ms) {
  if (cpu_ms > 0.0 && metal_ms > 0.0) {
    std::cout << label << ": " << (cpu_ms / metal_ms) << "x\n";
  }
}

}  // namespace

int main() {
  cudajun::runtime::preloadRuntimeProfileEnv();

  const std::size_t rows = readSizeEnv("CJ_ME_ROWS", 4096);
  const std::size_t cols = readSizeEnv("CJ_ME_COLS", 512);
  const int warmup = readIntEnv("CJ_ME_WARMUP", 10);
  const int iters = readIntEnv("CJ_ME_ITERS", 50);
  const int batch = readIntEnv("CJ_ME_BATCH", 4);

  std::cout << "[bench] matrix_ops rows=" << rows << " cols=" << cols
            << " warmup=" << warmup << " iters=" << iters << " batch=" << batch << "\n";

  double cpu_sub = runMatrixSub(cudajun::runtime::Device::kCPU, rows, cols, warmup, iters, batch, false);
  double metal_sub_off = runMatrixSub(cudajun::runtime::Device::kMetal, rows, cols, warmup, iters, batch, false);
  double metal_sub_on = runMatrixSub(cudajun::runtime::Device::kMetal, rows, cols, warmup, iters, batch, true);

  printResult("CPU matrixSub", cpu_sub);
  printResult("Metal matrixSub (resident=off)", metal_sub_off);
  printResult("Metal matrixSub (resident=on)", metal_sub_on);
  printSpeedup("Speedup matrixSub off (CPU/Metal)", cpu_sub, metal_sub_off);
  printSpeedup("Speedup matrixSub on (CPU/Metal)", cpu_sub, metal_sub_on);
  if (metal_sub_off > 0.0 && metal_sub_on > 0.0) {
    std::cout << "Resident gain matrixSub (off/on): " << (metal_sub_off / metal_sub_on) << "x\n";
  }

  double cpu_div = runMatrixDiv(cudajun::runtime::Device::kCPU, rows, cols, warmup, iters, batch, false);
  double metal_div_off = runMatrixDiv(cudajun::runtime::Device::kMetal, rows, cols, warmup, iters, batch, false);
  double metal_div_on = runMatrixDiv(cudajun::runtime::Device::kMetal, rows, cols, warmup, iters, batch, true);

  printResult("CPU matrixDiv", cpu_div);
  printResult("Metal matrixDiv (resident=off)", metal_div_off);
  printResult("Metal matrixDiv (resident=on)", metal_div_on);
  printSpeedup("Speedup matrixDiv off (CPU/Metal)", cpu_div, metal_div_off);
  printSpeedup("Speedup matrixDiv on (CPU/Metal)", cpu_div, metal_div_on);
  if (metal_div_off > 0.0 && metal_div_on > 0.0) {
    std::cout << "Resident gain matrixDiv (off/on): " << (metal_div_off / metal_div_on) << "x\n";
  }

  return 0;
}
