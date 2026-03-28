#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#include "cudajun/models/dnn_cnn_fastpath.hpp"
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

double runBlock(
    const cudajun::models::DnnCnnFastPath& fp,
    std::size_t batch,
    std::size_t in_dim,
    std::size_t out_dim,
    int warmup,
    int iters,
  bool enqueue_only,
  int resident_window) {
  std::vector<float> x(batch * in_dim, 0.1f);
  std::vector<float> w(in_dim * out_dim, 0.05f);
  std::vector<float> proj(batch * out_dim, 0.0f);
  std::vector<float> skip(batch * out_dim, 0.02f);
  std::vector<float> sub(batch * out_dim, 0.0f);
  std::vector<float> norm(batch * out_dim, 1.01f);
  std::vector<float> out(batch * out_dim, 0.0f);

  auto do_block = [&](cudajun::LoopStage stage) -> bool {
    if (fp.denseProject(x.data(), w.data(), proj.data(), batch, stage) != cudajun::runtime::Status::kSuccess) {
      return false;
    }
    if (fp.residualSub(proj.data(), skip.data(), sub.data(), batch, out_dim, stage) !=
        cudajun::runtime::Status::kSuccess) {
      return false;
    }
    if (fp.channelNormDiv(sub.data(), norm.data(), out.data(), batch, out_dim, stage) !=
        cudajun::runtime::Status::kSuccess) {
      return false;
    }
    return true;
  };

  for (int i = 0; i < warmup; ++i) {
    if (!do_block(cudajun::LoopStage::kOneShot)) {
      return -1.0;
    }
  }

  if (!enqueue_only) {
    if (resident_window < 1) {
      resident_window = 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    if (resident_window == 1) {
      for (int i = 0; i < iters; ++i) {
        if (!do_block(cudajun::LoopStage::kOneShot)) {
          return -1.0;
        }
      }
    } else {
      for (int i = 0; i < iters; ++i) {
        int pos = i % resident_window;
        int left = iters - i;
        if (pos == 0) {
          if (!do_block(cudajun::LoopStage::kStart)) {
            return -1.0;
          }
        } else if (left == 1 || pos == resident_window - 1) {
          if (!do_block(cudajun::LoopStage::kFinish)) {
            return -1.0;
          }
        } else {
          if (!do_block(cudajun::LoopStage::kRun)) {
            return -1.0;
          }
        }
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count() / static_cast<double>(iters);
  }

  if (!do_block(cudajun::LoopStage::kStart)) {
    return -1.0;
  }
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    if (!do_block(cudajun::LoopStage::kRun)) {
      return -1.0;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  if (!do_block(cudajun::LoopStage::kFinish)) {
    return -1.0;
  }

  std::chrono::duration<double, std::milli> elapsed = end - start;
  return elapsed.count() / static_cast<double>(iters);
}

int autoTuneResidentWindow(
    const cudajun::models::DnnCnnFastPath& fp,
    std::size_t batch,
    std::size_t in_dim,
    std::size_t out_dim,
    int max_window) {
  std::vector<int> candidates = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256};
  if (max_window < 1) {
    max_window = 1;
  }
  if (std::find(candidates.begin(), candidates.end(), max_window) == candidates.end()) {
    candidates.push_back(max_window);
  }
  if (max_window > 1 && std::find(candidates.begin(), candidates.end(), max_window - 1) == candidates.end()) {
    candidates.push_back(max_window - 1);
  }
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

  int best_w = 1;
  double best_ms = -1.0;
  const int probe_warmup = 2;
  const int probe_iters = (batch <= 2) ? 16 : 8;
  const int probe_trials = (batch <= 2) ? 5 : 3;
  for (int w : candidates) {
    if (w > max_window) {
      continue;
    }
    std::vector<double> trials;
    trials.reserve(static_cast<std::size_t>(probe_trials));
    for (int t = 0; t < probe_trials; ++t) {
      double ms = runBlock(fp, batch, in_dim, out_dim, probe_warmup, probe_iters, false, w);
      if (ms > 0.0) {
        trials.push_back(ms);
      }
    }
    if (trials.empty()) {
      continue;
    }
    std::sort(trials.begin(), trials.end());
    double ms = trials[trials.size() / 2];
    if (ms > 0.0 && (best_ms < 0.0 || ms < best_ms)) {
      best_ms = ms;
      best_w = w;
    }
  }
  return best_w;
}

void printResult(const char* label, double value) {
  if (value > 0.0) {
    std::cout << label << ": " << value << " ms\n";
  } else {
    std::cout << label << ": unavailable\n";
  }
}

void runSweepCsv(std::size_t in_dim, std::size_t out_dim, int warmup, int iters) {
  std::vector<std::size_t> batches = {1, 4, 16};

  std::ofstream csv("build/benchmarks/cnn_dnn_shape_sweep.csv");
  if (!csv.is_open()) {
    return;
  }
  csv << "batch,in_dim,out_dim,cpu_e2e_ms,metal_e2e_ms,metal_enqueue_only_ms\n";

  double best_metal_e2e = -1.0;
  std::size_t best_batch = 0;
  int best_window = 1;

  for (std::size_t batch : batches) {
    cudajun::models::DnnCnnFastPathConfig cfg;
    cfg.in_dim = in_dim;
    cfg.out_dim = out_dim;
    cfg.training = true;
    cfg.cnn_mode = true;

    cudajun::models::DnnCnnFastPath cpu(cfg, cudajun::runtime::Device::kCPU);
    cudajun::models::DnnCnnFastPath metal(cfg, cudajun::runtime::Device::kMetal);

    int tuned_window = autoTuneResidentWindow(metal, batch, in_dim, out_dim, (iters * 4 > 128) ? iters * 4 : 128);

    double cpu_ms = runBlock(cpu, batch, in_dim, out_dim, warmup, iters, false, 1);
    double metal_e2e = runBlock(metal, batch, in_dim, out_dim, warmup, iters, false, tuned_window);
    double metal_enqueue = runBlock(metal, batch, in_dim, out_dim, warmup, iters, true, tuned_window);

    if (metal_e2e > 0.0 && (best_metal_e2e < 0.0 || metal_e2e < best_metal_e2e)) {
      best_metal_e2e = metal_e2e;
      best_batch = batch;
      best_window = tuned_window;
    }

    csv << batch << "," << in_dim << "," << out_dim << "," << cpu_ms << "," << metal_e2e << "," << metal_enqueue
        << "\n";
  }

  std::ofstream env("build/benchmarks/cnn_dnn_runtime_profile.env");
  if (env.is_open() && best_batch > 0) {
    env << "CJ_CNN_BATCH=" << best_batch << "\n";
    env << "CJ_CNN_IN=" << in_dim << "\n";
    env << "CJ_CNN_OUT=" << out_dim << "\n";
    env << "CJ_CNN_RESIDENT_WINDOW=" << best_window << "\n";
  }

  std::cout << "[sweep] wrote build/benchmarks/cnn_dnn_shape_sweep.csv\n";
  std::cout << "[sweep] wrote build/benchmarks/cnn_dnn_runtime_profile.env\n";
}

}  // namespace

int main() {
  cudajun::runtime::preloadRuntimeProfileEnv();

  if (std::getenv("CJ_MATMUL_SMALL_TUNE_PROFILE") == nullptr) {
#if defined(_WIN32)
    (void)_putenv_s("CJ_MATMUL_SMALL_TUNE_PROFILE", "cnn_dnn");
#else
    (void)setenv("CJ_MATMUL_SMALL_TUNE_PROFILE", "cnn_dnn", 0);
#endif
  }

  const std::size_t batch = readSizeEnv("CJ_CNN_BATCH", 8);
  const std::size_t in_dim = readSizeEnv("CJ_CNN_IN", 1024);
  const std::size_t out_dim = readSizeEnv("CJ_CNN_OUT", 512);
  const int warmup = readIntEnv("CJ_CNN_WARMUP", 8);
  const int iters = readIntEnv("CJ_CNN_ITERS", 30);
  int resident_window = readIntEnv("CJ_CNN_RESIDENT_WINDOW", 0);
  const int sweep = readIntEnv("CJ_CNN_SWEEP", 0);

  cudajun::models::DnnCnnFastPathConfig cfg;
  cfg.in_dim = in_dim;
  cfg.out_dim = out_dim;
  cfg.training = true;
  cfg.cnn_mode = true;

  cudajun::models::DnnCnnFastPath cpu(cfg, cudajun::runtime::Device::kCPU);
  cudajun::models::DnnCnnFastPath metal(cfg, cudajun::runtime::Device::kMetal);

  if (resident_window <= 0) {
    resident_window = autoTuneResidentWindow(metal, batch, in_dim, out_dim, (iters * 4 > 128) ? iters * 4 : 128);
  }

  std::cout << "[bench] cnn-dnn-proxy batch=" << batch << " in=" << in_dim << " out=" << out_dim
            << " warmup=" << warmup << " iters=" << iters << " resident_window=" << resident_window << "\n";

  double cpu_e2e = runBlock(cpu, batch, in_dim, out_dim, warmup, iters, false, 1);
  double metal_e2e = runBlock(metal, batch, in_dim, out_dim, warmup, iters, false, resident_window);
  double metal_enqueue = runBlock(metal, batch, in_dim, out_dim, warmup, iters, true, resident_window);

  printResult("CPU cnn-dnn proxy (e2e)", cpu_e2e);
  printResult("Metal cnn-dnn proxy (e2e)", metal_e2e);
  printResult("Metal cnn-dnn proxy (enqueue-only)", metal_enqueue);

  if (cpu_e2e > 0.0 && metal_e2e > 0.0) {
    std::cout << "Speedup cnn-dnn proxy (CPU/Metal e2e): " << (cpu_e2e / metal_e2e) << "x\n";
  }
  if (metal_e2e > 0.0 && metal_enqueue > 0.0) {
    std::cout << "Enqueue ratio cnn-dnn proxy (e2e/enqueue): " << (metal_e2e / metal_enqueue) << "x\n";
  }

  if (sweep != 0) {
    runSweepCsv(in_dim, out_dim, warmup, iters);
  }

  return 0;
}
