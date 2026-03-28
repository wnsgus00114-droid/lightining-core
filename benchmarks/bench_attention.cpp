#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "cudajun/attention.hpp"
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

bool tryReadPositiveIntEnv(const char* key, int* out_value) {
  if (out_value == nullptr) {
    return false;
  }
  const char* raw = std::getenv(key);
  if (raw == nullptr || raw[0] == '\0') {
    return false;
  }
  char* end = nullptr;
  long parsed = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || parsed <= 0) {
    return false;
  }
  *out_value = static_cast<int>(parsed);
  return true;
}

int autoLossEveryForShape(const cudajun::AttentionConfig& cfg) {
  // Empirical defaults from higher-iteration validation.
  if (cfg.seq_len == 256 && cfg.head_dim == 64) {
    return 32;
  }
  if (cfg.seq_len == 256 && cfg.head_dim == 128) {
    return 16;
  }
  if (cfg.seq_len == 512 && cfg.head_dim == 64) {
    return 8;
  }
  if (cfg.seq_len == 512 && cfg.head_dim == 128) {
    return 8;
  }
  if (cfg.seq_len == 1024 && cfg.head_dim == 64) {
    return 8;
  }
  if (cfg.seq_len == 1024 && cfg.head_dim == 128) {
    return 8;
  }
  return 4;
}

void prewarmTrainAutotuneMultiShape(const cudajun::AttentionConfig& cfg) {
  std::vector<std::size_t> seq_candidates;
  if (cfg.seq_len >= 256) {
    seq_candidates.push_back(cfg.seq_len / 2);
  }
  seq_candidates.push_back(cfg.seq_len);
  if (cfg.seq_len <= 1024) {
    seq_candidates.push_back(cfg.seq_len * 2);
  }

  for (std::size_t seq : seq_candidates) {
    cudajun::AttentionConfig c{seq, cfg.head_dim, cfg.causal};
    std::size_t sd = c.seq_len * c.head_dim;
    std::vector<float> q(sd, 0.1f);
    std::vector<float> k(sd, 0.2f);
    std::vector<float> v(sd, 0.3f);
    std::vector<float> target(sd, 0.0f);
    std::vector<float> out(sd, 0.0f);
    float loss = 0.0f;

    cudajun::AttentionSession session(c, cudajun::runtime::Device::kMetal);
    cudajun::AttentionIoPolicy policy;
    policy.upload_q = true;
    policy.upload_k = true;
    policy.upload_v = true;
    policy.upload_target = true;
    policy.download_out = false;
    policy.download_v = false;
    policy.synchronize = true;
    session.setDefaultPolicy(policy);

    // compute_loss=true/false 둘 다 미리 한 번 실행해 autotune 캐시를 shape별로 채운다.
    (void)session.trainStep(q.data(), k.data(), v.data(), target.data(), out.data(), 0.001f, &loss);
    (void)session.trainStep(q.data(), k.data(), v.data(), target.data(), out.data(), 0.001f, nullptr);
  }
}

double runAttentionForward(
    cudajun::runtime::Device device,
    const cudajun::AttentionConfig& cfg,
    int warmup,
    int iters,
    int batch,
    bool resident_fastpath) {
  std::size_t s = cfg.seq_len;
  std::size_t d = cfg.head_dim;

  std::vector<float> q(s * d, 0.1f);
  std::vector<float> k(s * d, 0.2f);
  std::vector<float> v(s * d, 0.3f);
  std::vector<float> out(s * d, 0.0f);

  cudajun::AttentionSession session(cfg, device);
  cudajun::AttentionIoPolicy policy;
  if (device == cudajun::runtime::Device::kMetal && resident_fastpath) {
    policy.upload_q = true;
    policy.upload_k = true;
    policy.upload_v = true;
    policy.download_out = false;
    policy.synchronize = false;
    session.setDefaultPolicy(policy);
  }

  for (int i = 0; i < warmup; ++i) {
    for (int bidx = 0; bidx < batch; ++bidx) {
      if (device == cudajun::runtime::Device::kMetal && resident_fastpath) {
        policy.upload_q = (i == 0 && bidx == 0);
        policy.upload_k = (i == 0 && bidx == 0);
        policy.upload_v = (i == 0 && bidx == 0);
        policy.download_out = false;
        policy.synchronize = false;
        session.setDefaultPolicy(policy);
      }
      auto st = session.forward(q.data(), k.data(), v.data(), out.data());
      if (st != cudajun::runtime::Status::kSuccess) {
        return -1.0;
      }
    }
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    for (int bidx = 0; bidx < batch; ++bidx) {
      if (device == cudajun::runtime::Device::kMetal && resident_fastpath) {
        policy.upload_q = false;
        policy.upload_k = false;
        policy.upload_v = false;
        policy.download_out = false;
        policy.synchronize = false;
        session.setDefaultPolicy(policy);
      }
      auto st = session.forward(q.data(), k.data(), v.data(), out.data());
      if (st != cudajun::runtime::Status::kSuccess) {
        return -1.0;
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  if (device == cudajun::runtime::Device::kMetal && resident_fastpath) {
    policy.upload_q = false;
    policy.upload_k = false;
    policy.upload_v = false;
    policy.download_out = true;
    policy.synchronize = true;
    session.setDefaultPolicy(policy);
    auto st = session.forward(q.data(), k.data(), v.data(), out.data());
    if (st != cudajun::runtime::Status::kSuccess) {
      return -1.0;
    }
  }
  return elapsed.count() / static_cast<double>(iters * batch);
}

double runAttentionTrain(
    cudajun::runtime::Device device,
    const cudajun::AttentionConfig& cfg,
    int warmup,
    int iters,
  int batch,
  bool resident_fastpath,
  bool compute_loss,
  int loss_every) {
  std::size_t s = cfg.seq_len;
  std::size_t d = cfg.head_dim;

  std::vector<float> q(s * d, 0.1f);
  std::vector<float> k(s * d, 0.2f);
  std::vector<float> v(s * d, 0.3f);
  std::vector<float> target(s * d, 0.0f);
  std::vector<float> out(s * d, 0.0f);

  float loss = 0.0f;
  std::size_t step_counter = 0;
  cudajun::AttentionSession session(cfg, device);
  cudajun::AttentionIoPolicy policy;
  if (device == cudajun::runtime::Device::kMetal && resident_fastpath) {
    policy.upload_q = true;
    policy.upload_k = true;
    policy.upload_v = true;
    policy.upload_target = true;
    policy.download_out = false;
    policy.download_v = false;
    policy.synchronize = false;
    session.setDefaultPolicy(policy);
  }

  for (int i = 0; i < warmup; ++i) {
    for (int bidx = 0; bidx < batch; ++bidx) {
      if (device == cudajun::runtime::Device::kMetal && resident_fastpath) {
        policy.upload_q = (i == 0 && bidx == 0);
        policy.upload_k = (i == 0 && bidx == 0);
        policy.upload_v = (i == 0 && bidx == 0);
        policy.upload_target = (i == 0 && bidx == 0);
        policy.download_out = false;
        policy.download_v = false;
        policy.synchronize = false;
        session.setDefaultPolicy(policy);
      }
      bool take_loss = compute_loss && (loss_every <= 1 || (step_counter % static_cast<std::size_t>(loss_every) == 0));
      float* loss_ptr = take_loss ? &loss : nullptr;
      auto st = session.trainStep(q.data(), k.data(), v.data(), target.data(), out.data(), 0.001f, loss_ptr);
      if (st != cudajun::runtime::Status::kSuccess) {
        return -1.0;
      }
      ++step_counter;
    }
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    for (int bidx = 0; bidx < batch; ++bidx) {
      if (device == cudajun::runtime::Device::kMetal && resident_fastpath) {
        policy.upload_q = false;
        policy.upload_k = false;
        policy.upload_v = false;
        policy.upload_target = false;
        policy.download_out = false;
        policy.download_v = false;
        policy.synchronize = false;
        session.setDefaultPolicy(policy);
      }
      bool take_loss = compute_loss && (loss_every <= 1 || (step_counter % static_cast<std::size_t>(loss_every) == 0));
      float* loss_ptr = take_loss ? &loss : nullptr;
      auto st = session.trainStep(q.data(), k.data(), v.data(), target.data(), out.data(), 0.001f, loss_ptr);
      if (st != cudajun::runtime::Status::kSuccess) {
        return -1.0;
      }
      ++step_counter;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  if (device == cudajun::runtime::Device::kMetal && resident_fastpath) {
    policy.upload_q = false;
    policy.upload_k = false;
    policy.upload_v = false;
    policy.upload_target = false;
    policy.download_out = false;
    policy.download_v = true;
    policy.synchronize = true;
    session.setDefaultPolicy(policy);
    float* loss_ptr = compute_loss ? &loss : nullptr;
    auto st = session.trainStep(q.data(), k.data(), v.data(), target.data(), out.data(), 0.001f, loss_ptr);
    if (st != cudajun::runtime::Status::kSuccess) {
      return -1.0;
    }
  }
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

void printDevicePlan(const char* label, cudajun::runtime::Device device) {
  std::cout << "[plan] " << label << " impl=" << cudajun::attentionImplementationName(device)
            << " fallback=" << (cudajun::attentionUsesFallback(device) ? "yes" : "no") << "\n";
}

void runAttentionShapeSweepCsv(int warmup, int iters, int batch, bool compute_loss, int loss_every) {
  std::vector<std::size_t> seqs = {256, 512, 1024, 2048};
  std::vector<std::size_t> dims = {64, 128};

  std::ofstream csv("build/benchmarks/attention_shape_sweep.csv");
  if (!csv.is_open()) {
    return;
  }
  csv << "seq,dim,cpu_fwd_e2e_ms,metal_fwd_e2e_ms,metal_fwd_enqueue_only_ms,cpu_train_e2e_ms,metal_train_e2e_ms,metal_train_enqueue_only_ms\n";

  for (std::size_t seq : seqs) {
    for (std::size_t dim : dims) {
      cudajun::AttentionConfig cfg{seq, dim, true};
      double cpu_fwd = runAttentionForward(cudajun::runtime::Device::kCPU, cfg, warmup, iters, batch, false);
      double metal_fwd_e2e = runAttentionForward(cudajun::runtime::Device::kMetal, cfg, warmup, iters, batch, false);
      double metal_fwd_enqueue = runAttentionForward(cudajun::runtime::Device::kMetal, cfg, warmup, iters, batch, true);

      double cpu_train = runAttentionTrain(
          cudajun::runtime::Device::kCPU, cfg, warmup, iters, batch, false, compute_loss, loss_every);
      double metal_train_e2e = runAttentionTrain(
          cudajun::runtime::Device::kMetal, cfg, warmup, iters, batch, false, compute_loss, loss_every);
        double metal_train_enqueue = runAttentionTrain(
          cudajun::runtime::Device::kMetal, cfg, warmup, iters, batch, true, compute_loss, loss_every);

        csv << seq << "," << dim << "," << cpu_fwd << "," << metal_fwd_e2e << "," << metal_fwd_enqueue << ","
          << cpu_train << "," << metal_train_e2e << "," << metal_train_enqueue << "\n";
    }
  }
  std::cout << "[sweep] wrote build/benchmarks/attention_shape_sweep.csv\n";
}

}  // namespace

int main() {
  cudajun::runtime::preloadRuntimeProfileEnv();

  std::size_t seq_len = readSizeEnv("CJ_ATTN_SEQ", 1024);
  std::size_t head_dim = readSizeEnv("CJ_ATTN_DIM", 64);
  int warmup = readIntEnv("CJ_ATTN_WARMUP", 10);
  int iters = readIntEnv("CJ_ATTN_ITERS", 50);
  int batch = readIntEnv("CJ_ATTN_BATCH", 4);
  int train_compute_loss = readIntEnv("CJ_ATTN_TRAIN_COMPUTE_LOSS", 0);
  int run_sweep = readIntEnv("CJ_ATTN_SWEEP", 0);

  cudajun::AttentionConfig cfg{seq_len, head_dim, true};
  int train_loss_every = 4;
  bool user_set_loss_every = tryReadPositiveIntEnv("CJ_ATTN_LOSS_EVERY", &train_loss_every);
  if (!user_set_loss_every && train_compute_loss != 0) {
    train_loss_every = autoLossEveryForShape(cfg);
  }
  int prewarm_shapes = readIntEnv("CJ_ATTN_PREWARM_SHAPES", 1);

  if (prewarm_shapes != 0) {
    std::cout << "[plan] prewarm train-autotune multi-shape=on\n";
    prewarmTrainAutotuneMultiShape(cfg);
  } else {
    std::cout << "[plan] prewarm train-autotune multi-shape=off\n";
  }

  std::cout << "[bench] attention seq=" << seq_len << " dim=" << head_dim << " warmup=" << warmup
            << " iters=" << iters << " batch=" << batch << "\n";
  std::cout << "[plan] train compute_loss=" << (train_compute_loss != 0 ? "on" : "off") << "\n";
  if (train_compute_loss != 0) {
    std::cout << "[plan] train loss_every=" << train_loss_every << "\n";
    std::cout << "[plan] train loss_every source=" << (user_set_loss_every ? "env" : "auto") << "\n";
  }
  printDevicePlan("CPU", cudajun::runtime::Device::kCPU);
  printDevicePlan("Metal", cudajun::runtime::Device::kMetal);

    double cpu_fwd = runAttentionForward(cudajun::runtime::Device::kCPU, cfg, warmup, iters, batch, false);
    double cpu_train = runAttentionTrain(
      cudajun::runtime::Device::kCPU, cfg, warmup, iters, batch, false, train_compute_loss != 0, train_loss_every);
  printResult("CPU forward", cpu_fwd);
  printResult("CPU train", cpu_train);

    double metal_fwd_e2e =
      runAttentionForward(cudajun::runtime::Device::kMetal, cfg, warmup, iters, batch, false);
    double metal_train_e2e =
      runAttentionTrain(
          cudajun::runtime::Device::kMetal, cfg, warmup, iters, batch, false, train_compute_loss != 0, train_loss_every);
    double metal_fwd_enqueue = runAttentionForward(cudajun::runtime::Device::kMetal, cfg, warmup, iters, batch, true);
    double metal_train_enqueue = runAttentionTrain(
      cudajun::runtime::Device::kMetal, cfg, warmup, iters, batch, true, train_compute_loss != 0, train_loss_every);

    printResult("Metal forward (e2e)", metal_fwd_e2e);
    printResult("Metal train (e2e)", metal_train_e2e);
    printResult("Metal forward (enqueue-only)", metal_fwd_enqueue);
    printResult("Metal train (enqueue-only)", metal_train_enqueue);

    double metal_fwd = metal_fwd_e2e;
    double metal_train = metal_train_e2e;

  if (cpu_fwd > 0.0 && metal_fwd > 0.0) {
    std::cout << "Speedup forward (CPU/Metal): " << (cpu_fwd / metal_fwd) << "x\n";
  }
  if (cpu_train > 0.0 && metal_train > 0.0) {
    std::cout << "Speedup train (CPU/Metal): " << (cpu_train / metal_train) << "x\n";
  }

  if (metal_fwd_e2e > 0.0 && metal_fwd_enqueue > 0.0) {
    std::cout << "Enqueue ratio forward (e2e/enqueue): " << (metal_fwd_e2e / metal_fwd_enqueue) << "x\n";
    double overhead = metal_fwd_e2e - metal_fwd_enqueue;
    if (overhead < 0.0) {
      overhead = 0.0;
    }
    printResult("Metal forward transfer+sync overhead(est)", overhead);
  }
  if (metal_train_e2e > 0.0 && metal_train_enqueue > 0.0) {
    std::cout << "Enqueue ratio train (e2e/enqueue): " << (metal_train_e2e / metal_train_enqueue) << "x\n";
    double overhead = metal_train_e2e - metal_train_enqueue;
    if (overhead < 0.0) {
      overhead = 0.0;
    }
    printResult("Metal train transfer+sync overhead(est)", overhead);
  }

  if (run_sweep != 0) {
    runAttentionShapeSweepCsv(std::max(4, warmup / 2), std::max(8, iters / 2), batch, train_compute_loss != 0, train_loss_every);
  }

  return 0;
}
