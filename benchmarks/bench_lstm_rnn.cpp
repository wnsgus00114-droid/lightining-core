#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#include "cudajun/models/lstm_rnn_fastpath.hpp"

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

double runRecurrentProxy(
    const cudajun::models::LstmRnnFastPath& fp,
    std::size_t batch,
    std::size_t input_dim,
    std::size_t hidden_dim,
    int warmup,
    int iters,
    int steps,
  bool enqueue_only,
  int resident_window) {
  std::vector<float> x(batch * input_dim, 0.1f);
  std::vector<float> h(batch * hidden_dim, 0.2f);
  std::vector<float> wx(input_dim * hidden_dim, 0.05f);
  std::vector<float> wh(hidden_dim * hidden_dim, 0.04f);
  std::vector<float> hx(batch * hidden_dim, 0.0f);
  std::vector<float> hh(batch * hidden_dim, 0.0f);
  std::vector<float> out(batch * hidden_dim, 0.0f);

  auto do_step = [&](cudajun::LoopStage stage) -> bool {
    if (fp.projectInput(x.data(), wx.data(), hx.data(), batch, stage) != cudajun::runtime::Status::kSuccess) {
      return false;
    }
    if (fp.projectHidden(h.data(), wh.data(), hh.data(), batch, stage) != cudajun::runtime::Status::kSuccess) {
      return false;
    }
    if (fp.fuseRecurrent(hx.data(), hh.data(), out.data(), batch * hidden_dim, stage) !=
        cudajun::runtime::Status::kSuccess) {
      return false;
    }
    h.swap(out);
    return true;
  };

  for (int i = 0; i < warmup; ++i) {
    for (int t = 0; t < steps; ++t) {
      if (!do_step(cudajun::LoopStage::kOneShot)) {
        return -1.0;
      }
    }
  }

  if (!enqueue_only) {
    if (resident_window < 1) {
      resident_window = 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    int total = iters * steps;
    if (resident_window == 1) {
      for (int i = 0; i < iters; ++i) {
        for (int t = 0; t < steps; ++t) {
          if (!do_step(cudajun::LoopStage::kOneShot)) {
            return -1.0;
          }
        }
      }
    } else {
      for (int idx = 0; idx < total; ++idx) {
        int pos = idx % resident_window;
        int left = total - idx;
        if (pos == 0) {
          if (!do_step(cudajun::LoopStage::kStart)) {
            return -1.0;
          }
        } else if (left == 1 || pos == resident_window - 1) {
          if (!do_step(cudajun::LoopStage::kFinish)) {
            return -1.0;
          }
        } else {
          if (!do_step(cudajun::LoopStage::kRun)) {
            return -1.0;
          }
        }
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count() / static_cast<double>(iters * steps);
  }

  if (!do_step(cudajun::LoopStage::kStart)) {
    return -1.0;
  }
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    for (int t = 0; t < steps; ++t) {
      if (!do_step(cudajun::LoopStage::kRun)) {
        return -1.0;
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  if (!do_step(cudajun::LoopStage::kFinish)) {
    return -1.0;
  }

  std::chrono::duration<double, std::milli> elapsed = end - start;
  return elapsed.count() / static_cast<double>(iters * steps);
}

int autoTuneResidentWindow(
    const cudajun::models::LstmRnnFastPath& fp,
    std::size_t batch,
    std::size_t input_dim,
    std::size_t hidden_dim,
    int steps,
    int max_window) {
  std::vector<int> candidates = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024};
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
  const int probe_warmup = 1;
  const int probe_iters = (batch <= 2) ? 8 : 4;
  const int probe_trials = (batch <= 2) ? 5 : 3;
  for (int w : candidates) {
    if (w > max_window) {
      continue;
    }
    std::vector<double> trials;
    trials.reserve(static_cast<std::size_t>(probe_trials));
    for (int t = 0; t < probe_trials; ++t) {
      double ms = runRecurrentProxy(fp, batch, input_dim, hidden_dim, probe_warmup, probe_iters, steps, false, w);
      if (ms > 0.0) {
        trials.push_back(ms);
      }
    }
    if (trials.empty()) {
      continue;
    }
    std::sort(trials.begin(), trials.end());
    double ms = trials[trials.size() / 2];
    if (best_ms < 0.0 || ms < best_ms) {
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

void runSweepCsv(std::size_t input_dim, std::size_t hidden_dim, int warmup, int iters, int steps) {
  std::vector<std::size_t> batches = {1, 4, 8};

  std::ofstream csv("build/benchmarks/lstm_rnn_shape_sweep.csv");
  if (!csv.is_open()) {
    return;
  }
  csv << "batch,input_dim,hidden_dim,cpu_e2e_ms,metal_e2e_ms,metal_enqueue_only_ms\n";

  double best_metal_e2e = -1.0;
  std::size_t best_batch = 0;
  int best_window = 1;

  for (std::size_t batch : batches) {
    cudajun::models::LstmRnnFastPathConfig cfg;
    cfg.input_dim = input_dim;
    cfg.hidden_dim = hidden_dim;
    cfg.training = true;
    cfg.lstm_mode = true;

    cudajun::models::LstmRnnFastPath cpu(cfg, cudajun::runtime::Device::kCPU);
    cudajun::models::LstmRnnFastPath metal(cfg, cudajun::runtime::Device::kMetal);

    int tuned_window = autoTuneResidentWindow(metal, batch, input_dim, hidden_dim, steps, iters * steps);

    double cpu_ms = runRecurrentProxy(cpu, batch, input_dim, hidden_dim, warmup, iters, steps, false, 1);
    double metal_e2e = runRecurrentProxy(metal, batch, input_dim, hidden_dim, warmup, iters, steps, false, tuned_window);
    double metal_enqueue = runRecurrentProxy(metal, batch, input_dim, hidden_dim, warmup, iters, steps, true, tuned_window);

    if (metal_e2e > 0.0 && (best_metal_e2e < 0.0 || metal_e2e < best_metal_e2e)) {
      best_metal_e2e = metal_e2e;
      best_batch = batch;
      best_window = tuned_window;
    }

    csv << batch << "," << input_dim << "," << hidden_dim << "," << cpu_ms << "," << metal_e2e << ","
        << metal_enqueue << "\n";
  }

  std::ofstream env("build/benchmarks/lstm_rnn_runtime_profile.env");
  if (env.is_open() && best_batch > 0) {
    env << "CJ_LSTM_BATCH=" << best_batch << "\n";
    env << "CJ_LSTM_INPUT=" << input_dim << "\n";
    env << "CJ_LSTM_HIDDEN=" << hidden_dim << "\n";
    env << "CJ_LSTM_RESIDENT_WINDOW=" << best_window << "\n";
  }

  std::cout << "[sweep] wrote build/benchmarks/lstm_rnn_shape_sweep.csv\n";
  std::cout << "[sweep] wrote build/benchmarks/lstm_rnn_runtime_profile.env\n";
}

}  // namespace

int main() {
  cudajun::runtime::preloadRuntimeProfileEnv();

  if (std::getenv("CJ_MATMUL_SMALL_TUNE_PROFILE") == nullptr) {
#if defined(_WIN32)
    (void)_putenv_s("CJ_MATMUL_SMALL_TUNE_PROFILE", "lstm_rnn");
#else
    (void)setenv("CJ_MATMUL_SMALL_TUNE_PROFILE", "lstm_rnn", 0);
#endif
  }

  const std::size_t batch = readSizeEnv("CJ_LSTM_BATCH", 4);
  const std::size_t input_dim = readSizeEnv("CJ_LSTM_INPUT", 512);
  const std::size_t hidden_dim = readSizeEnv("CJ_LSTM_HIDDEN", 512);
  const int warmup = readIntEnv("CJ_LSTM_WARMUP", 4);
  const int iters = readIntEnv("CJ_LSTM_ITERS", 20);
  const int steps = readIntEnv("CJ_LSTM_STEPS", 64);
  int resident_window = readIntEnv("CJ_LSTM_RESIDENT_WINDOW", 0);
  const int sweep = readIntEnv("CJ_LSTM_SWEEP", 0);

  cudajun::models::LstmRnnFastPathConfig cfg;
  cfg.input_dim = input_dim;
  cfg.hidden_dim = hidden_dim;
  cfg.training = true;
  cfg.lstm_mode = true;

  cudajun::models::LstmRnnFastPath cpu(cfg, cudajun::runtime::Device::kCPU);
  cudajun::models::LstmRnnFastPath metal(cfg, cudajun::runtime::Device::kMetal);

  if (resident_window <= 0) {
    resident_window = autoTuneResidentWindow(metal, batch, input_dim, hidden_dim, steps, iters * steps);
  }

  std::cout << "[bench] lstm-rnn-proxy batch=" << batch << " input=" << input_dim << " hidden=" << hidden_dim
            << " steps=" << steps << " warmup=" << warmup << " iters=" << iters
            << " resident_window=" << resident_window << "\n";

  double cpu_e2e = runRecurrentProxy(cpu, batch, input_dim, hidden_dim, warmup, iters, steps, false, 1);
  double metal_e2e = runRecurrentProxy(metal, batch, input_dim, hidden_dim, warmup, iters, steps, false, resident_window);
  double metal_enqueue = runRecurrentProxy(metal, batch, input_dim, hidden_dim, warmup, iters, steps, true, resident_window);

  printResult("CPU lstm-rnn proxy (e2e)", cpu_e2e);
  printResult("Metal lstm-rnn proxy (e2e)", metal_e2e);
  printResult("Metal lstm-rnn proxy (enqueue-only)", metal_enqueue);

  if (cpu_e2e > 0.0 && metal_e2e > 0.0) {
    std::cout << "Speedup lstm-rnn proxy (CPU/Metal e2e): " << (cpu_e2e / metal_e2e) << "x\n";
  }
  if (metal_e2e > 0.0 && metal_enqueue > 0.0) {
    std::cout << "Enqueue ratio lstm-rnn proxy (e2e/enqueue): " << (metal_e2e / metal_enqueue) << "x\n";
  }

  if (sweep != 0) {
    runSweepCsv(input_dim, hidden_dim, warmup, iters, steps);
  }

  return 0;
}
