#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

#include "lightning_core/attention.hpp"

namespace {

float mse(const std::vector<float>& a, const std::vector<float>& b) {
  float acc = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    float d = a[i] - b[i];
    acc += d * d;
  }
  return acc / static_cast<float>(a.size());
}

bool approxEq(float a, float b, float tol = 1.0e-4f) {
  return std::fabs(a - b) <= tol;
}

}  // namespace

int main() {
  using namespace cudajun;

  AttentionImplementation metal_impl = attentionImplementation(runtime::Device::kMetal);
#if defined(CJ_HAS_METAL) && CJ_HAS_METAL
  if (metal_impl != AttentionImplementation::kMetalFused) {
    std::cerr << "attentionImplementation mismatch for Metal-enabled build\n";
    return 1;
  }
  if (attentionUsesFallback(runtime::Device::kMetal)) {
    std::cerr << "attentionUsesFallback must be false on Metal-enabled build\n";
    return 1;
  }
#else
  if (metal_impl != AttentionImplementation::kMetalFallbackToCpu) {
    std::cerr << "attentionImplementation mismatch for Metal-disabled build\n";
    return 1;
  }
  if (!attentionUsesFallback(runtime::Device::kMetal)) {
    std::cerr << "attentionUsesFallback must be true on Metal-disabled build\n";
    return 1;
  }
#endif

  AttentionConfig cfg{2, 2, false};

  std::vector<float> q{1.f, 0.f, 0.f, 1.f};
  std::vector<float> k{1.f, 0.f, 0.f, 1.f};
  std::vector<float> v{10.f, 0.f, 0.f, 20.f};
  std::vector<float> out(4, 0.0f);

  if (attentionForward(q.data(), k.data(), v.data(), out.data(), cfg, runtime::Device::kCPU) !=
      runtime::Status::kSuccess) {
    std::cerr << "attentionForward failed\n";
    return 1;
  }

  const std::vector<float> expected{6.6976f, 6.6048f, 3.3024f, 13.3952f};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (std::fabs(out[i] - expected[i]) > 0.05f) {
      std::cerr << "forward mismatch at " << i << ": " << out[i] << " vs " << expected[i] << "\n";
      return 1;
    }
  }

  AttentionIoPolicy post;
  post.output_scale = 0.5f;
  post.output_bias = -2.0f;
  post.output_relu = true;

  std::vector<float> out_post_cpu(4, 0.0f);
  if (attentionForwardWithPolicy(
          q.data(), k.data(), v.data(), out_post_cpu.data(), cfg, runtime::Device::kCPU, post) !=
      runtime::Status::kSuccess) {
    std::cerr << "attentionForwardWithPolicy CPU postprocess failed\n";
    return 1;
  }

  for (std::size_t i = 0; i < out_post_cpu.size(); ++i) {
    float ref = expected[i] * post.output_scale + post.output_bias;
    if (ref < 0.0f) {
      ref = 0.0f;
    }
    if (std::fabs(out_post_cpu[i] - ref) > 0.05f) {
      std::cerr << "cpu postprocess mismatch at " << i << ": " << out_post_cpu[i] << " vs " << ref << "\n";
      return 1;
    }
  }

  if (runtime::isMetalAvailable()) {
    std::vector<float> out_post_metal(4, 0.0f);
    if (attentionForwardWithPolicy(
            q.data(), k.data(), v.data(), out_post_metal.data(), cfg, runtime::Device::kMetal, post) !=
        runtime::Status::kSuccess) {
      std::cerr << "attentionForwardWithPolicy Metal postprocess failed\n";
      return 1;
    }
    for (std::size_t i = 0; i < out_post_metal.size(); ++i) {
      if (std::fabs(out_post_metal[i] - out_post_cpu[i]) > 0.06f) {
        std::cerr << "metal postprocess mismatch at " << i << ": " << out_post_metal[i]
                  << " vs cpu " << out_post_cpu[i] << "\n";
        return 1;
      }
    }
  }

#if !defined(CJ_HAS_METAL) || !CJ_HAS_METAL
  {
    std::vector<float> out_metal_fallback(4, 0.0f);
    if (attentionForwardWithPolicy(
            q.data(),
            k.data(),
            v.data(),
            out_metal_fallback.data(),
            cfg,
            runtime::Device::kMetal,
            AttentionIoPolicy{}) != runtime::Status::kSuccess) {
      std::cerr << "attentionForwardWithPolicy Metal fallback path failed\n";
      return 1;
    }
    for (std::size_t i = 0; i < out_metal_fallback.size(); ++i) {
      if (!approxEq(out_metal_fallback[i], out[i], 0.06f)) {
        std::cerr << "metal->cpu fallback output mismatch at " << i << "\n";
        return 1;
      }
    }
  }
#endif

  std::vector<float> target(4, 0.0f);
  float loss_before = mse(out, target);
  float step_loss = 0.0f;

  if (attentionTrainStep(
          q.data(),
          k.data(),
          v.data(),
          target.data(),
          out.data(),
          0.01f,
          cfg,
          runtime::Device::kCPU,
          &step_loss) != runtime::Status::kSuccess) {
    std::cerr << "attentionTrainStep failed\n";
    return 1;
  }

  std::vector<float> out_after(4, 0.0f);
  if (attentionForward(q.data(), k.data(), v.data(), out_after.data(), cfg, runtime::Device::kCPU) !=
      runtime::Status::kSuccess) {
    std::cerr << "attentionForward after step failed\n";
    return 1;
  }

  float loss_after = mse(out_after, target);
  if (loss_after > loss_before + 1e-4f) {
    std::cerr << "training step did not improve loss: before=" << loss_before << " after=" << loss_after << "\n";
    return 1;
  }

  if (step_loss <= 0.0f) {
    std::cerr << "step loss must be positive\n";
    return 1;
  }

  {
    AttentionConfig mt_cfg{4, 4, true};
    AttentionSession shared(mt_cfg, runtime::Device::kCPU);

    AttentionIoPolicy mt_policy;
    mt_policy.loss_every = 2;
    shared.setDefaultPolicy(mt_policy);

    std::vector<float> q_mt(mt_cfg.seq_len * mt_cfg.head_dim, 0.3f);
    std::vector<float> k_mt(mt_cfg.seq_len * mt_cfg.head_dim, 0.2f);
    std::vector<float> target_mt(mt_cfg.seq_len * mt_cfg.head_dim, 0.0f);

    constexpr int kThreads = 4;
    constexpr int kStepsPerThread = 16;
    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    std::vector<float> thread_last_loss(kThreads, 0.0f);

    for (int t = 0; t < kThreads; ++t) {
      workers.emplace_back([&, t]() {
        std::vector<float> v_local(mt_cfg.seq_len * mt_cfg.head_dim, 0.1f + 0.01f * static_cast<float>(t));
        std::vector<float> out_local(mt_cfg.seq_len * mt_cfg.head_dim, 0.0f);
        float loss_local = 0.0f;
        for (int s = 0; s < kStepsPerThread; ++s) {
          runtime::Status st = shared.trainStep(
              q_mt.data(),
              k_mt.data(),
              v_local.data(),
              target_mt.data(),
              out_local.data(),
              0.01f,
              &loss_local);
          if (st != runtime::Status::kSuccess) {
            loss_local = -1.0f;
            break;
          }
        }
        thread_last_loss[t] = loss_local;
      });
    }

    for (auto& w : workers) {
      w.join();
    }

    for (int t = 0; t < kThreads; ++t) {
      if (!(thread_last_loss[t] >= 0.0f) || !std::isfinite(thread_last_loss[t])) {
        std::cerr << "shared AttentionSession multi-thread run failed at thread " << t << "\n";
        return 1;
      }
    }
  }

  return 0;
}
