#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "cudajun/models/vlm_fastpath.hpp"
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

int autoTuneResidentWindow(
    const cudajun::models::VlmFastPath& fp,
    const cudajun::models::VlmFastPathConfig& cfg,
    int batch,
    int max_window);

double runVlmProxy(
    const cudajun::models::VlmFastPath& fp,
    const cudajun::models::VlmFastPathConfig& cfg,
    int warmup,
    int iters,
    int batch,
    bool enqueue_only,
    int resident_window,
    int online_retune_every = 0) {
  std::size_t total = cfg.image_tokens + cfg.text_tokens;
  const std::size_t patch_h = cfg.patch_size;
  const std::size_t patch_w = cfg.patch_size;
  const std::size_t image_h = patch_h;
  const std::size_t image_w = cfg.image_tokens * patch_w;

  std::vector<float> image_nhwc(image_h * image_w * cfg.image_channels, 0.1f);
  std::vector<float> text_tokens(cfg.text_tokens * cfg.text_dim, 0.2f);
  std::vector<float> w_vision(cfg.vision_dim * cfg.fused_dim, 0.03f);
  std::vector<float> w_text(cfg.text_dim * cfg.fused_dim, 0.04f);
  std::vector<float> image_proj(cfg.image_tokens * cfg.fused_dim, 0.0f);
  std::vector<float> text_proj(cfg.text_tokens * cfg.fused_dim, 0.0f);

  std::vector<float> q(total * cfg.fused_dim, 0.11f);
  std::vector<float> k(total * cfg.fused_dim, 0.12f);
  std::vector<float> v(total * cfg.fused_dim, 0.13f);
  std::vector<float> out(total * cfg.fused_dim, 0.0f);
  std::vector<float> out_text(cfg.text_tokens * cfg.fused_dim, 0.0f);

  auto do_step = [&](cudajun::LoopStage stage) -> bool {
    if (fp.patchEmbedFromImage(image_nhwc.data(), image_h, image_w, w_vision.data(), image_proj.data(), stage) !=
        cudajun::runtime::Status::kSuccess) {
      return false;
    }
    if (fp.projectText(text_tokens.data(), w_text.data(), text_proj.data(), stage) !=
        cudajun::runtime::Status::kSuccess) {
      return false;
    }
    if (fp.runCrossAttentionFast(image_proj.data(), text_proj.data(), out_text.data(), stage) !=
        cudajun::runtime::Status::kSuccess) {
      return false;
    }
    if (fp.runFusionAttention(q.data(), k.data(), v.data(), out.data(), stage) !=
        cudajun::runtime::Status::kSuccess) {
      return false;
    }
    return true;
  };

  for (int i = 0; i < warmup; ++i) {
    for (int bidx = 0; bidx < batch; ++bidx) {
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
    if (resident_window == 1) {
      for (int i = 0; i < iters; ++i) {
        for (int bidx = 0; bidx < batch; ++bidx) {
          if (!do_step(cudajun::LoopStage::kOneShot)) {
            return -1.0;
          }
        }
      }
    } else {
      int total_steps = iters * batch;
      for (int idx = 0; idx < total_steps; ++idx) {
        if (online_retune_every > 0 && idx > 0 && (idx % online_retune_every) == 0) {
          int tuned = autoTuneResidentWindow(fp, cfg, batch, (iters * batch > 128) ? iters * batch : 128);
          if (tuned > 0) {
            resident_window = tuned;
          }
        }
        int pos = idx % resident_window;
        int left = total_steps - idx;
        cudajun::LoopStage stage = cudajun::LoopStage::kRun;
        if (pos == 0) {
          stage = cudajun::LoopStage::kStart;
        } else if (left == 1 || pos == resident_window - 1) {
          stage = cudajun::LoopStage::kFinish;
        }
        if (!do_step(stage)) {
          return -1.0;
        }
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count() / static_cast<double>(iters * batch);
  }

  if (!do_step(cudajun::LoopStage::kStart)) {
    return -1.0;
  }
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    for (int bidx = 0; bidx < batch; ++bidx) {
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
  return elapsed.count() / static_cast<double>(iters * batch);
}

int autoTuneResidentWindow(
    const cudajun::models::VlmFastPath& fp,
    const cudajun::models::VlmFastPathConfig& cfg,
    int batch,
    int max_window) {
  std::vector<int> candidates = {1, 2, 4, 8, 16, 32, 64, 128};
  if (max_window < 1) {
    max_window = 1;
  }

  int best_w = 1;
  double best_ms = -1.0;
  for (int w : candidates) {
    if (w > max_window) {
      continue;
    }
    std::vector<double> trials;
    trials.reserve(3);
    for (int t = 0; t < 3; ++t) {
      double ms = runVlmProxy(fp, cfg, 1, 4, batch, false, w, 0);
      if (ms > 0.0) {
        trials.push_back(ms);
      }
    }
    if (trials.empty()) {
      continue;
    }
    std::sort(trials.begin(), trials.end());
    double median = trials[trials.size() / 2];
    if (best_ms < 0.0 || median < best_ms) {
      best_ms = median;
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
  std::vector<std::size_t> img_tokens = {128, 256};
  std::vector<std::size_t> txt_tokens = {64, 128};
  std::vector<std::size_t> fused_dims = {64, 128};

  std::ofstream csv("build/benchmarks/vlm_shape_sweep.csv");
  if (!csv.is_open()) {
    return;
  }
  csv << "img_tokens,txt_tokens,fused_dim,cpu_e2e_ms,metal_e2e_ms,metal_enqueue_only_ms\n";

  double best_metal_e2e = -1.0;
  std::size_t best_img = 0;
  std::size_t best_txt = 0;
  std::size_t best_dim = 0;
  int best_window = 1;

  for (std::size_t img : img_tokens) {
    for (std::size_t txt : txt_tokens) {
      for (std::size_t dim : fused_dims) {
        cudajun::models::VlmFastPathConfig cfg;
        cfg.image_tokens = img;
        cfg.text_tokens = txt;
        cfg.image_channels = 3;
        cfg.patch_size = 16;
        cfg.vision_dim = cfg.patch_size * cfg.patch_size * cfg.image_channels;
        cfg.text_dim = 768;
        cfg.fused_dim = dim;
        cfg.training = true;

        cudajun::models::VlmFastPath cpu(cfg, cudajun::runtime::Device::kCPU);
        cudajun::models::VlmFastPath metal(cfg, cudajun::runtime::Device::kMetal);

        int w = autoTuneResidentWindow(metal, cfg, batch, (iters * batch > 128) ? iters * batch : 128);

        double cpu_ms = runVlmProxy(cpu, cfg, warmup, iters, batch, false, 1, 0);
        double metal_e2e = runVlmProxy(metal, cfg, warmup, iters, batch, false, w, 0);
        double metal_enqueue = runVlmProxy(metal, cfg, warmup, iters, batch, true, w, 0);

        if (metal_e2e > 0.0 && (best_metal_e2e < 0.0 || metal_e2e < best_metal_e2e)) {
          best_metal_e2e = metal_e2e;
          best_img = img;
          best_txt = txt;
          best_dim = dim;
          best_window = w;
        }

        csv << img << "," << txt << "," << dim << "," << cpu_ms << "," << metal_e2e << "," << metal_enqueue << "\n";
      }
    }
  }

  std::ofstream env("build/benchmarks/vlm_runtime_profile.env");
  if (env.is_open() && best_img > 0 && best_txt > 0 && best_dim > 0) {
    env << "CJ_VLM_IMG_TOKENS=" << best_img << "\n";
    env << "CJ_VLM_TEXT_TOKENS=" << best_txt << "\n";
    env << "CJ_VLM_FUSED_DIM=" << best_dim << "\n";
    env << "CJ_VLM_BATCH=" << batch << "\n";
    env << "CJ_VLM_RESIDENT_WINDOW=" << best_window << "\n";
  }

  std::cout << "[sweep] wrote build/benchmarks/vlm_shape_sweep.csv\n";
  std::cout << "[sweep] wrote build/benchmarks/vlm_runtime_profile.env\n";
}

}  // namespace

int main() {
  cudajun::runtime::preloadRuntimeProfileEnv();

  cudajun::models::VlmFastPathConfig cfg;
  cfg.image_tokens = readSizeEnv("CJ_VLM_IMG_TOKENS", 256);
  cfg.text_tokens = readSizeEnv("CJ_VLM_TEXT_TOKENS", 128);
  cfg.vision_dim = readSizeEnv("CJ_VLM_VISION_DIM", 0);
  cfg.text_dim = readSizeEnv("CJ_VLM_TEXT_DIM", 768);
  cfg.fused_dim = readSizeEnv("CJ_VLM_FUSED_DIM", 64);
  cfg.image_channels = readSizeEnv("CJ_VLM_IMAGE_CHANNELS", 3);
  cfg.patch_size = readSizeEnv("CJ_VLM_PATCH_SIZE", 16);
  cfg.training = true;

  if (cfg.vision_dim == 0) {
    cfg.vision_dim = cfg.patch_size * cfg.patch_size * cfg.image_channels;
  }

  const int warmup = readIntEnv("CJ_VLM_WARMUP", 8);
  const int iters = readIntEnv("CJ_VLM_ITERS", 20);
  const int batch = readIntEnv("CJ_VLM_BATCH", 2);
  int resident_window = readIntEnv("CJ_VLM_RESIDENT_WINDOW", 0);
  const int enable_online_retune = readIntEnv("CJ_VLM_ENABLE_ONLINE_RETUNE", 0);
  int online_retune_every = 0;
  if (enable_online_retune != 0) {
    online_retune_every = readIntEnv("CJ_VLM_ONLINE_RETUNE_EVERY", 64);
  }
  const int sweep = readIntEnv("CJ_VLM_SWEEP", 0);

  cudajun::models::VlmFastPath cpu(cfg, cudajun::runtime::Device::kCPU);
  cudajun::models::VlmFastPath metal(cfg, cudajun::runtime::Device::kMetal);

  if (resident_window <= 0) {
    resident_window = autoTuneResidentWindow(metal, cfg, batch, (iters * batch > 128) ? iters * batch : 128);
  }

  std::cout << "[bench] vlm-proxy img_tokens=" << cfg.image_tokens << " txt_tokens=" << cfg.text_tokens
            << " fused_dim=" << cfg.fused_dim << " warmup=" << warmup << " iters=" << iters << " batch=" << batch
            << " resident_window=" << resident_window << " online_retune="
            << (enable_online_retune != 0 ? "on" : "off") << " online_retune_every=" << online_retune_every << "\n";

  double cpu_e2e = runVlmProxy(cpu, cfg, warmup, iters, batch, false, 1, 0);
  double metal_e2e = runVlmProxy(metal, cfg, warmup, iters, batch, false, resident_window, online_retune_every);
  double metal_enqueue = runVlmProxy(metal, cfg, warmup, iters, batch, true, resident_window, 0);

  printResult("CPU vlm proxy (e2e)", cpu_e2e);
  printResult("Metal vlm proxy (e2e)", metal_e2e);
  printResult("Metal vlm proxy (enqueue-only)", metal_enqueue);

  if (cpu_e2e > 0.0 && metal_e2e > 0.0) {
    std::cout << "Speedup vlm proxy (CPU/Metal e2e): " << (cpu_e2e / metal_e2e) << "x\n";
  }
  if (metal_e2e > 0.0 && metal_enqueue > 0.0) {
    std::cout << "Enqueue ratio vlm proxy (e2e/enqueue): " << (metal_e2e / metal_enqueue) << "x\n";
  }

  if (sweep != 0) {
    runSweepCsv(warmup, iters, batch);
  }

  return 0;
}
