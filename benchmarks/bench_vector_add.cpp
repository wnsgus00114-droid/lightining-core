#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "lightning_core/apple_ml.hpp"
#include "lightning_core/ops.hpp"
#include "lightning_core/runtime.hpp"

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

template <typename T>
double runHostBenchmark(std::size_t n, int iters, int batch, lightning_core::runtime::Device device) {
  // 입력/출력 버퍼 준비.
  std::vector<T> a(n, static_cast<T>(1));
  std::vector<T> b(n, static_cast<T>(2));
  std::vector<T> out(n, static_cast<T>(0));

  // 반복 실행 시간 측정.
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    for (int bidx = 0; bidx < batch; ++bidx) {
      lightning_core::ops::vectorAdd<T>(a.data(), b.data(), out.data(), n, device);
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  return elapsed.count() / static_cast<double>(iters * batch);
}

double runMetalResidentBenchmark(std::size_t n, int iters, int batch) {
  std::vector<float> a(n, 1.0f);
  std::vector<float> b(n, 2.0f);
  std::vector<float> out(n, 0.0f);

  if (lightning_core::ops::vectorAddMetalResidentStart<float>(a.data(), b.data(), out.data(), n) !=
      lightning_core::runtime::Status::kSuccess) {
    return -1.0;
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    for (int bidx = 0; bidx < batch; ++bidx) {
      if (lightning_core::ops::vectorAddMetalResidentRun<float>(a.data(), b.data(), out.data(), n) !=
          lightning_core::runtime::Status::kSuccess) {
        return -1.0;
      }
    }
  }
  if (lightning_core::ops::vectorAddMetalResidentFinish<float>(a.data(), b.data(), out.data(), n) !=
      lightning_core::runtime::Status::kSuccess) {
    return -1.0;
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  return elapsed.count() / static_cast<double>(iters * batch);
}

double runMetalResidentEnqueueOnlyBenchmark(std::size_t n, int iters, int batch) {
  std::vector<float> a(n, 1.0f);
  std::vector<float> b(n, 2.0f);
  std::vector<float> out(n, 0.0f);

  if (lightning_core::ops::vectorAddMetalResidentStart<float>(a.data(), b.data(), out.data(), n) !=
      lightning_core::runtime::Status::kSuccess) {
    return -1.0;
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    for (int bidx = 0; bidx < batch; ++bidx) {
      if (lightning_core::ops::vectorAddMetalResidentRun<float>(a.data(), b.data(), out.data(), n) !=
          lightning_core::runtime::Status::kSuccess) {
        return -1.0;
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();

  if (lightning_core::ops::vectorAddMetalResidentFinish<float>(a.data(), b.data(), out.data(), n) !=
      lightning_core::runtime::Status::kSuccess) {
    return -1.0;
  }

  std::chrono::duration<double, std::milli> elapsed = end - start;
  return elapsed.count() / static_cast<double>(iters * batch);
}

template <typename T>
double runCudaBenchmark(std::size_t n, int iters, int batch) {
  // Host 측 입력 데이터.
  std::vector<T> host_a(n, static_cast<T>(1));
  std::vector<T> host_b(n, static_cast<T>(2));

  // Device 포인터들.
  T* dev_a = nullptr;
  T* dev_b = nullptr;
  T* dev_out = nullptr;

  using namespace lightning_core::runtime;
  // 디바이스 메모리 할당.
  if (mallocDevice(reinterpret_cast<void**>(&dev_a), n * sizeof(T)) != Status::kSuccess ||
      mallocDevice(reinterpret_cast<void**>(&dev_b), n * sizeof(T)) != Status::kSuccess ||
      mallocDevice(reinterpret_cast<void**>(&dev_out), n * sizeof(T)) != Status::kSuccess) {
    freeDevice(dev_a);
    freeDevice(dev_b);
    freeDevice(dev_out);
    return -1.0;
  }

  // Host -> Device 업로드.
    if (memcpy(dev_a, host_a.data(), n * sizeof(T), MemcpyKind::kHostToDevice) != Status::kSuccess ||
      memcpy(dev_b, host_b.data(), n * sizeof(T), MemcpyKind::kHostToDevice) != Status::kSuccess) {
    freeDevice(dev_a);
    freeDevice(dev_b);
    freeDevice(dev_out);
    return -1.0;
  }

  // 커널 반복 실행 시간 측정.
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    for (int bidx = 0; bidx < batch; ++bidx) {
      if (lightning_core::ops::vectorAdd<T>(dev_a, dev_b, dev_out, n, lightning_core::runtime::Device::kCUDA) != Status::kSuccess) {
        freeDevice(dev_a);
        freeDevice(dev_b);
        freeDevice(dev_out);
        return -1.0;
      }
    }
  }
  // 큐에 쌓인 작업들이 끝난 시점 기준으로 시간을 확정.
  deviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  // 누수 방지를 위해 할당한 메모리 정리.
  freeDevice(dev_a);
  freeDevice(dev_b);
  freeDevice(dev_out);

  std::chrono::duration<double, std::milli> elapsed = end - start;
  return elapsed.count() / static_cast<double>(iters * batch);
}

void report(const std::string& label, double value) {
  std::cout << label << ": " << value << " ms\n";
}

void runVectorAddSweepAndWriteCsv(int iters, int batch) {
  if (!lightning_core::runtime::isMetalAvailable()) {
    return;
  }

  std::vector<std::size_t> sizes = {
      1u << 12, 1u << 14, 1u << 16, 1u << 18, 1u << 20, 1u << 22, 1u << 24,
  };

  std::ofstream csv("build/benchmarks/vector_add_crossover.csv");
  if (!csv.is_open()) {
    return;
  }
  csv << "n,cpu_e2e_ms,metal_e2e_ms,metal_enqueue_only_ms,recommended\n";

  std::size_t crossover_n = sizes.back();
  for (std::size_t n : sizes) {
    double cpu = runHostBenchmark<float>(n, iters, batch, lightning_core::runtime::Device::kCPU);
    double metal_e2e = runHostBenchmark<float>(n, iters, batch, lightning_core::runtime::Device::kMetal);
    double metal_enqueue = runMetalResidentEnqueueOnlyBenchmark(n, iters, batch);

    const char* rec = (cpu <= metal_e2e) ? "cpu" : "metal";
    if (std::string(rec) == "metal" && crossover_n == sizes.back()) {
      crossover_n = n;
    }

    csv << n << "," << cpu << "," << metal_e2e << "," << metal_enqueue << "," << rec << "\n";
  }

  std::ofstream hint("build/benchmarks/vector_add_crossover_hint.env");
  if (hint.is_open()) {
    hint << "CJ_VECTORADD_CPU_CROSSOVER_N=" << crossover_n << "\n";
  }

  std::cout << "[sweep] wrote build/benchmarks/vector_add_crossover.csv\n";
  std::cout << "[sweep] hint CJ_VECTORADD_CPU_CROSSOVER_N=" << crossover_n << "\n";
}

}  // namespace

int main() {
  lightning_core::runtime::preloadRuntimeProfileEnv();

  // 큰 텐서/배치 비교를 위해 환경변수로 조정 가능.
  // 예: CJ_BENCH_N=16777216 CJ_BENCH_ITERS=50 CJ_BENCH_BATCH=8
  const std::size_t n = readSizeEnv("CJ_BENCH_N", 1u << 20);
  const int iters = readIntEnv("CJ_BENCH_ITERS", 100);
  const int batch = readIntEnv("CJ_BENCH_BATCH", 1);
  const int sweep = readIntEnv("CJ_BENCH_SWEEP", 0);

  std::cout << "[bench] vector_add n=" << n << " iterations=" << iters << " batch=" << batch << "\n";
  std::cout << "preferred(training)="
            << static_cast<int>(lightning_core::runtime::preferredDeviceFor(lightning_core::runtime::WorkloadKind::kTraining))
            << ", preferred(inference)="
            << static_cast<int>(lightning_core::runtime::preferredDeviceFor(lightning_core::runtime::WorkloadKind::kInference))
            << "\n";

  double cpu_f32 = runHostBenchmark<float>(n, iters, batch, lightning_core::runtime::Device::kCPU);
  double cpu_f64 = runHostBenchmark<double>(n, iters, batch, lightning_core::runtime::Device::kCPU);
  report("CPU float32", cpu_f32);
  report("CPU float64", cpu_f64);

  // CUDA 가능할 때만 비교 측정.
  if (lightning_core::runtime::isCudaAvailable()) {
    double cuda_f32 = runCudaBenchmark<float>(n, iters, batch);
    double cuda_f64 = runCudaBenchmark<double>(n, iters, batch);
    if (cuda_f32 > 0.0) {
      report("CUDA float32", cuda_f32);
      std::cout << "Speedup f32 (CPU/CUDA): " << (cpu_f32 / cuda_f32) << "x\n";
    }
    if (cuda_f64 > 0.0) {
      report("CUDA float64", cuda_f64);
      std::cout << "Speedup f64 (CPU/CUDA): " << (cpu_f64 / cuda_f64) << "x\n";
    }
  } else {
    std::cout << "CUDA unavailable; skipped CUDA benchmark.\n";
  }

  if (lightning_core::runtime::isMetalAvailable()) {
    double metal_f32_e2e = runHostBenchmark<float>(n, iters, batch, lightning_core::runtime::Device::kMetal);
    double metal_f32_resident_e2e = runMetalResidentBenchmark(n, iters, batch);
    double metal_f32_enqueue = runMetalResidentEnqueueOnlyBenchmark(n, iters, batch);
    report("Metal float32 (e2e)", metal_f32_e2e);
    if (metal_f32_resident_e2e > 0.0) {
      report("Metal float32 (resident-e2e)", metal_f32_resident_e2e);
      std::cout << "Speedup f32 (CPU/Metal resident-e2e): " << (cpu_f32 / metal_f32_resident_e2e) << "x\n";
      std::cout << "Resident gain f32 (e2e/resident-e2e): " << (metal_f32_e2e / metal_f32_resident_e2e) << "x\n";
    }
    if (metal_f32_enqueue > 0.0) {
      report("Metal float32 (enqueue-only)", metal_f32_enqueue);
      double transfer_overhead = metal_f32_e2e - metal_f32_enqueue;
      if (transfer_overhead < 0.0) {
        transfer_overhead = 0.0;
      }
      report("Metal transfer+sync overhead(est)", transfer_overhead);
    }
  } else {
    std::cout << "Metal unavailable; skipped Metal benchmark.\n";
  }

  if (sweep != 0) {
    runVectorAddSweepAndWriteCsv(std::max(20, iters / 2), std::max(1, batch));
  }

  double mps_add_ms = -1.0;
  if (lightning_core::apple::benchmarkMpsGraphVectorAdd(n, iters * batch, &mps_add_ms) == lightning_core::runtime::Status::kSuccess) {
    report("MPSGraph add float32", mps_add_ms);
  } else {
    std::cout << "MPSGraph add benchmark unavailable.\n";
  }

  double mps_train_ms = -1.0;
  if (lightning_core::apple::benchmarkMpsGraphTrainStep(n, iters * batch, &mps_train_ms) == lightning_core::runtime::Status::kSuccess) {
    report("MPSGraph train-step float32", mps_train_ms);
  } else {
    std::cout << "MPSGraph train benchmark unavailable.\n";
  }

  const char* coreml_model = std::getenv("CJ_COREML_MODEL");
  if (coreml_model != nullptr) {
    double coreml_ms = -1.0;
    if (lightning_core::apple::benchmarkCoreMLInference(coreml_model, 1024, iters * batch, &coreml_ms) ==
        lightning_core::runtime::Status::kSuccess) {
      report("CoreML inference", coreml_ms);
    } else {
      std::cout << "CoreML inference benchmark failed. Check CJ_COREML_MODEL path/input shape.\n";
    }
  } else {
    std::cout << "CoreML benchmark skipped. Set CJ_COREML_MODEL=/path/to/model.mlmodelc\n";
  }

  return 0;
}
