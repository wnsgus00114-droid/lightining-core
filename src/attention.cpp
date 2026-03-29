#include "lightning_core/core/attention.hpp"

#include "lightning_core/core/detail/attention_backend.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace {

int autoLossEveryForShape(const lightning_core::AttentionConfig& cfg) {
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

int resolveLossEvery(const lightning_core::AttentionConfig& cfg, const lightning_core::AttentionIoPolicy& policy) {
  if (policy.loss_every == 0) {
    return autoLossEveryForShape(cfg);
  }
  return std::max(1, policy.loss_every);
}

void applyForwardPostprocess(
    float* out,
    const lightning_core::AttentionConfig& cfg,
    const lightning_core::AttentionIoPolicy& policy) {
  if (out == nullptr) {
    return;
  }
  const bool identity = (policy.output_scale == 1.0f && policy.output_bias == 0.0f && !policy.output_relu);
  if (identity) {
    return;
  }
  const std::size_t n = cfg.seq_len * cfg.head_dim;
  for (std::size_t i = 0; i < n; ++i) {
    float v = out[i] * policy.output_scale + policy.output_bias;
    if (policy.output_relu && v < 0.0f) {
      v = 0.0f;
    }
    out[i] = v;
  }
}

std::mutex& attentionSessionStateMutex() {
  static std::mutex mu;
  return mu;
}

std::size_t attentionMetalOneShotCpuCrossoverOps() {
  const char* env = std::getenv("CJ_ATTN_METAL_ONESHOT_CPU_CROSSOVER_OPS");
  if (env == nullptr || env[0] == '\0') {
    // Tiny attention shapes are often launch-bound on Metal; use CPU for one-shot latency.
    return 8192;
  }
  char* end = nullptr;
  unsigned long long v = std::strtoull(env, &end, 10);
  if (end == env || *end != '\0') {
    return 8192;
  }
  return static_cast<std::size_t>(v);
}

bool useCpuCrossoverForMetalForward(const lightning_core::AttentionConfig& cfg,
                                    const lightning_core::AttentionIoPolicy& policy) {
  if (!policy.upload_q || !policy.upload_k || !policy.upload_v || !policy.download_out) {
    return false;
  }
  const std::size_t ops = cfg.seq_len * cfg.seq_len * cfg.head_dim;
  return ops <= attentionMetalOneShotCpuCrossoverOps();
}

}  // namespace

namespace lightning_core {

AttentionImplementation attentionImplementation(runtime::Device device) {
  if (device == runtime::Device::kCPU) {
    return AttentionImplementation::kCpuSimd;
  }
  if (device == runtime::Device::kMetal) {
#if defined(CJ_HAS_METAL) && CJ_HAS_METAL
    return AttentionImplementation::kMetalFused;
#else
    return AttentionImplementation::kMetalFallbackToCpu;
#endif
  }
  return AttentionImplementation::kUnavailable;
}

const char* attentionImplementationName(runtime::Device device) {
  AttentionImplementation impl = attentionImplementation(device);
  if (impl == AttentionImplementation::kCpuSimd) {
    return "custom-cpu-simd";
  }
  if (impl == AttentionImplementation::kMetalFused) {
    return "custom-metal-fused";
  }
  if (impl == AttentionImplementation::kMetalFallbackToCpu) {
    return "metal-entry->cpu-kernel";
  }
  return "unavailable";
}

bool attentionUsesFallback(runtime::Device device) {
  AttentionImplementation impl = attentionImplementation(device);
  return impl == AttentionImplementation::kMetalFallbackToCpu;
}

runtime::Status attentionForward(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const AttentionConfig& cfg,
    runtime::Device device) {
  return attentionForwardWithPolicy(q, k, v, out, cfg, device, AttentionIoPolicy{});
}

runtime::Status attentionForwardWithPolicy(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const AttentionConfig& cfg,
    runtime::Device device,
    const AttentionIoPolicy& policy) {
  if (q == nullptr || k == nullptr || v == nullptr || out == nullptr || cfg.seq_len == 0 || cfg.head_dim == 0) {
    return runtime::Status::kInvalidValue;
  }

  if (device == runtime::Device::kCPU) {
    runtime::Status st = detail::attentionForwardCpu(q, k, v, out, cfg);
    if (st == runtime::Status::kSuccess) {
      applyForwardPostprocess(out, cfg, policy);
    }
    return st;
  }
  if (device == runtime::Device::kMetal) {
    if (useCpuCrossoverForMetalForward(cfg, policy)) {
      runtime::Status st = detail::attentionForwardCpu(q, k, v, out, cfg);
      if (st == runtime::Status::kSuccess) {
        applyForwardPostprocess(out, cfg, policy);
      }
      return st;
    }
    runtime::Status st = detail::attentionForwardMetalWithPolicy(q, k, v, out, cfg, policy);
    if (st == runtime::Status::kNotSupported) {
      st = detail::attentionForwardCpu(q, k, v, out, cfg);
      if (st == runtime::Status::kSuccess) {
        applyForwardPostprocess(out, cfg, policy);
      }
      return st;
    }
    return st;
  }
  return runtime::Status::kNotSupported;
}

runtime::Status attentionTrainStep(
    const float* q,
    const float* k,
    float* v,
    const float* target,
    float* out,
    float learning_rate,
    const AttentionConfig& cfg,
    runtime::Device device,
    float* loss_out) {
  return attentionTrainStepWithPolicy(
      q, k, v, target, out, learning_rate, cfg, device, loss_out, AttentionIoPolicy{});
}

runtime::Status attentionTrainStepWithPolicy(
    const float* q,
    const float* k,
    float* v,
    const float* target,
    float* out,
    float learning_rate,
    const AttentionConfig& cfg,
    runtime::Device device,
    float* loss_out,
    const AttentionIoPolicy& policy) {
  if (q == nullptr || k == nullptr || v == nullptr || target == nullptr || out == nullptr ||
      cfg.seq_len == 0 || cfg.head_dim == 0 || learning_rate <= 0.0f) {
    return runtime::Status::kInvalidValue;
  }

  if (device == runtime::Device::kCPU) {
    return detail::attentionTrainStepCpu(q, k, v, target, out, learning_rate, cfg, loss_out);
  }
  if (device == runtime::Device::kMetal) {
    runtime::Status st =
        detail::attentionTrainStepMetalWithPolicy(q, k, v, target, out, learning_rate, cfg, loss_out, policy);
    if (st == runtime::Status::kNotSupported) {
      return detail::attentionTrainStepCpu(q, k, v, target, out, learning_rate, cfg, loss_out);
    }
    return st;
  }
  return runtime::Status::kNotSupported;
}

AttentionSession::AttentionSession(const AttentionConfig& cfg, runtime::Device device)
    : cfg_(cfg), device_(device), default_policy_{} {
  if (device_ == runtime::Device::kMetal) {
    // Session defaults target repeated execution; host downloads still force synchronization.
    default_policy_.synchronize = false;
    default_policy_.loss_every = 0;
  }
}

const AttentionConfig& AttentionSession::config() const {
  return cfg_;
}

runtime::Device AttentionSession::device() const {
  return device_;
}

void AttentionSession::setDefaultPolicy(const AttentionIoPolicy& policy) {
  default_policy_ = policy;
}

const AttentionIoPolicy& AttentionSession::defaultPolicy() const {
  return default_policy_;
}

runtime::Status AttentionSession::forward(const float* q, const float* k, const float* v, float* out) const {
  return attentionForwardWithPolicy(q, k, v, out, cfg_, device_, default_policy_);
}

runtime::Status AttentionSession::forwardWithPolicy(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const AttentionIoPolicy& policy) const {
  return attentionForwardWithPolicy(q, k, v, out, cfg_, device_, policy);
}

runtime::Status AttentionSession::trainStep(
    const float* q,
    const float* k,
    float* v,
    const float* target,
    float* out,
    float learning_rate,
    float* loss_out) const {
  return trainStepWithPolicy(q, k, v, target, out, learning_rate, loss_out, default_policy_);
}

runtime::Status AttentionSession::trainStepWithPolicy(
    const float* q,
    const float* k,
    float* v,
    const float* target,
    float* out,
    float learning_rate,
    float* loss_out,
    const AttentionIoPolicy& policy) const {
  std::lock_guard<std::mutex> lock(attentionSessionStateMutex());

  float sampled_loss = 0.0f;
  float* effective_loss_out = nullptr;
  AttentionIoPolicy effective_policy = policy;
  bool taking_loss_this_step = false;

  if (loss_out != nullptr) {
    const int loss_every = resolveLossEvery(cfg_, policy);
    taking_loss_this_step = (train_step_count_ % static_cast<std::size_t>(loss_every)) == 0;
    if (taking_loss_this_step) {
      effective_loss_out = &sampled_loss;
    } else if (!policy.download_out && !policy.download_v) {
      // If nothing is read back to host, sampled-out loss steps do not need host-side blocking.
      effective_policy.synchronize = false;
    }
  }

  runtime::Status st = attentionTrainStepWithPolicy(
      q, k, v, target, out, learning_rate, cfg_, device_, effective_loss_out, effective_policy);
  if (st != runtime::Status::kSuccess) {
    return st;
  }

  if (loss_out != nullptr) {
    if (effective_loss_out != nullptr) {
      last_loss_ = sampled_loss;
      has_last_loss_ = true;
      *loss_out = sampled_loss;
    } else if (has_last_loss_) {
      *loss_out = last_loss_;
    } else {
      *loss_out = 0.0f;
    }
  }

  ++train_step_count_;
  return runtime::Status::kSuccess;
}

}  // namespace lightning_core
