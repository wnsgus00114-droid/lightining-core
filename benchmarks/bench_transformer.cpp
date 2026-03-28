#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#include "cudajun/models/transformer_fastpath.hpp"

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

double runForward(
    const cudajun::models::TransformerFastPath& fp,
    std::size_t seq,
    std::size_t dim,
    int warmup,
    int iters,
    int batch,
  bool enqueue_only,
  int resident_window) {
  std::vector<float> q(seq * dim, 0.1f);
  std::vector<float> k(seq * dim, 0.2f);
  std::vector<float> v(seq * dim, 0.3f);
  std::vector<float> out(seq * dim, 0.0f);

  for (int i = 0; i < warmup; ++i) {
    for (int bidx = 0; bidx < batch; ++bidx) {
      auto st = fp.attentionForward(q.data(), k.data(), v.data(), out.data(), cudajun::LoopStage::kOneShot);
      if (st != cudajun::runtime::Status::kSuccess) {
        return -1.0;
      }
    }
  }

  if (!enqueue_only) {
    if (resident_window < 1) {
      resident_window = 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    if (resident_window == 1) {
      for (int i = 0; i < iters; ++i) {
        for (int bidx = 0; bidx < batch; ++bidx) {
          auto st = fp.attentionForward(q.data(), k.data(), v.data(), out.data(), cudajun::LoopStage::kOneShot);
          if (st != cudajun::runtime::Status::kSuccess) {
            return -1.0;
          }
        }
      }
    } else {
      const int total = iters * batch;
      for (int idx = 0; idx < total; ++idx) {
        const int pos = idx % resident_window;
        const int left = total - idx;
        cudajun::LoopStage stage = cudajun::LoopStage::kRun;
        if (pos == 0) {
          stage = cudajun::LoopStage::kStart;
        } else if (left == 1 || pos == resident_window - 1) {
          stage = cudajun::LoopStage::kFinish;
        }
        auto st = fp.attentionForward(q.data(), k.data(), v.data(), out.data(), stage);
        if (st != cudajun::runtime::Status::kSuccess) {
          return -1.0;
        }
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count() / static_cast<double>(iters * batch);
  }

  if (fp.attentionForward(q.data(), k.data(), v.data(), out.data(), cudajun::LoopStage::kStart) !=
      cudajun::runtime::Status::kSuccess) {
    return -1.0;
  }
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    for (int bidx = 0; bidx < batch; ++bidx) {
      auto st = fp.attentionForward(q.data(), k.data(), v.data(), out.data(), cudajun::LoopStage::kRun);
      if (st != cudajun::runtime::Status::kSuccess) {
        return -1.0;
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  if (fp.attentionForward(q.data(), k.data(), v.data(), out.data(), cudajun::LoopStage::kFinish) !=
      cudajun::runtime::Status::kSuccess) {
    return -1.0;
  }
  std::chrono::duration<double, std::milli> elapsed = end - start;
  return elapsed.count() / static_cast<double>(iters * batch);
}

int autoTuneResidentWindow(
    const cudajun::models::TransformerFastPath& fp,
    std::size_t seq,
    std::size_t dim,
    int batch,
    int max_window) {
  std::vector<int> candidates = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128};
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
  const int probe_iters = (batch <= 2 || seq <= 512) ? 12 : 6;
  const int probe_trials = (batch <= 2) ? 5 : 3;
  for (int w : candidates) {
    if (w > max_window) {
      continue;
    }
    std::vector<double> trials;
    trials.reserve(static_cast<std::size_t>(probe_trials));
    for (int t = 0; t < probe_trials; ++t) {
      double ms = runForward(fp, seq, dim, probe_warmup, probe_iters, batch, false, w);
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

void runSweepCsv(int warmup, int iters, int batch) {
  std::vector<std::size_t> seqs = {256, 512, 1024};
  std::vector<std::size_t> dims = {64, 128};

  std::ofstream csv("build/benchmarks/transformer_shape_sweep.csv");
  if (!csv.is_open()) {
    return;
  }
  csv << "seq,dim,cpu_e2e_ms,metal_e2e_ms,metal_enqueue_only_ms\n";

  double best_metal_e2e = -1.0;
  std::size_t best_seq = 0;
  std::size_t best_dim = 0;
  int best_window = 1;

  for (std::size_t seq : seqs) {
    for (std::size_t dim : dims) {
      cudajun::models::TransformerFastPathConfig cfg;
      cfg.seq_len = seq;
      cfg.head_dim = dim;
      cfg.causal = true;
      cfg.training = true;
      cudajun::models::TransformerFastPath cpu(cfg, cudajun::runtime::Device::kCPU);
      cudajun::models::TransformerFastPath metal(cfg, cudajun::runtime::Device::kMetal);

      int tuned_window = autoTuneResidentWindow(metal, seq, dim, batch, iters * batch);

      double cpu_ms = runForward(cpu, seq, dim, warmup, iters, batch, false, 1);
      double metal_e2e = runForward(metal, seq, dim, warmup, iters, batch, false, tuned_window);
      double metal_enqueue = runForward(metal, seq, dim, warmup, iters, batch, true, tuned_window);

      if (metal_e2e > 0.0 && (best_metal_e2e < 0.0 || metal_e2e < best_metal_e2e)) {
        best_metal_e2e = metal_e2e;
        best_seq = seq;
        best_dim = dim;
        best_window = tuned_window;
      }

      csv << seq << "," << dim << "," << cpu_ms << "," << metal_e2e << "," << metal_enqueue << "\n";
    }
  }

  std::ofstream env("build/benchmarks/transformer_runtime_profile.env");
  if (env.is_open() && best_seq > 0 && best_dim > 0) {
    env << "CJ_TR_SEQ=" << best_seq << "\n";
    env << "CJ_TR_DIM=" << best_dim << "\n";
    env << "CJ_TR_BATCH=" << batch << "\n";
    env << "CJ_TR_RESIDENT_WINDOW=" << best_window << "\n";
  }
  std::cout << "[sweep] wrote build/benchmarks/transformer_shape_sweep.csv\n";
  std::cout << "[sweep] wrote build/benchmarks/transformer_runtime_profile.env\n";
}

}  // namespace

int main() {
  cudajun::runtime::preloadRuntimeProfileEnv();

  const std::size_t seq = readSizeEnv("CJ_TR_SEQ", 1024);
  const std::size_t dim = readSizeEnv("CJ_TR_DIM", 64);
  const int warmup = readIntEnv("CJ_TR_WARMUP", 8);
  const int iters = readIntEnv("CJ_TR_ITERS", 30);
  const int batch = readIntEnv("CJ_TR_BATCH", 4);
  int resident_window = readIntEnv("CJ_TR_RESIDENT_WINDOW", 0);
  const int sweep = readIntEnv("CJ_TR_SWEEP", 0);

  cudajun::models::TransformerFastPathConfig cfg;
  cfg.seq_len = seq;
  cfg.head_dim = dim;
  cfg.causal = true;
  cfg.training = true;

  cudajun::models::TransformerFastPath cpu(cfg, cudajun::runtime::Device::kCPU);
  cudajun::models::TransformerFastPath metal(cfg, cudajun::runtime::Device::kMetal);

  if (resident_window <= 0) {
    resident_window = autoTuneResidentWindow(metal, seq, dim, batch, iters * batch);
  }

  std::cout << "[bench] transformer seq=" << seq << " dim=" << dim << " warmup=" << warmup
            << " iters=" << iters << " batch=" << batch << " resident_window=" << resident_window << "\n";

  double cpu_e2e = runForward(cpu, seq, dim, warmup, iters, batch, false, 1);
  double metal_e2e = runForward(metal, seq, dim, warmup, iters, batch, false, resident_window);
  double metal_enqueue = runForward(metal, seq, dim, warmup, iters, batch, true, resident_window);

  printResult("CPU transformer forward (e2e)", cpu_e2e);
  printResult("Metal transformer forward (e2e)", metal_e2e);
  printResult("Metal transformer forward (enqueue-only)", metal_enqueue);

  if (cpu_e2e > 0.0 && metal_e2e > 0.0) {
    std::cout << "Speedup transformer (CPU/Metal e2e): " << (cpu_e2e / metal_e2e) << "x\n";
  }
  if (metal_e2e > 0.0 && metal_enqueue > 0.0) {
    std::cout << "Enqueue ratio transformer (e2e/enqueue): " << (metal_e2e / metal_enqueue) << "x\n";
  }

  if (sweep != 0) {
    runSweepCsv(warmup, iters, batch);
  }

  return 0;
}
