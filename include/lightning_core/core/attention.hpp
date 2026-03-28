#pragma once

#include <cstddef>

#include "lightning_core/core/runtime.hpp"

namespace lightning_core {

struct AttentionConfig {
  std::size_t seq_len;
  std::size_t head_dim;
  bool causal;
};

// Metal 경로에서 host<->device 왕복을 제어하는 옵션.
struct AttentionIoPolicy {
  bool upload_q = true;
  bool upload_k = true;
  bool upload_v = true;
  bool upload_target = true;
  bool download_out = true;
  bool download_v = true;
  bool synchronize = true;
  // If > 1 and loss_out is requested, compute loss every N train steps.
  // If set to 0, runtime chooses a shape-aware default.
  int loss_every = 1;
  // Optional forward postprocess: out = max(0, out * output_scale + output_bias) when output_relu=true.
  // This is applied to attention forward output path.
  float output_scale = 1.0f;
  float output_bias = 0.0f;
  bool output_relu = false;
};

enum class AttentionImplementation {
  kCpuSimd,
  kMetalFused,
  kMetalFallbackToCpu,
  kUnavailable,
};

AttentionImplementation attentionImplementation(runtime::Device device);
const char* attentionImplementationName(runtime::Device device);
bool attentionUsesFallback(runtime::Device device);

runtime::Status attentionForward(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const AttentionConfig& cfg,
    runtime::Device device);

runtime::Status attentionForwardWithPolicy(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const AttentionConfig& cfg,
    runtime::Device device,
    const AttentionIoPolicy& policy);

runtime::Status attentionTrainStep(
    const float* q,
    const float* k,
    float* v,
    const float* target,
    float* out,
    float learning_rate,
    const AttentionConfig& cfg,
    runtime::Device device,
    float* loss_out);

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
    const AttentionIoPolicy& policy);

  // 동일 shape/device 작업을 반복 실행할 때 설정을 묶어 재사용한다.
  class AttentionSession {
   public:
    AttentionSession(const AttentionConfig& cfg, runtime::Device device);

    const AttentionConfig& config() const;
    runtime::Device device() const;

    void setDefaultPolicy(const AttentionIoPolicy& policy);
    const AttentionIoPolicy& defaultPolicy() const;

    runtime::Status forward(const float* q, const float* k, const float* v, float* out) const;
    runtime::Status forwardWithPolicy(
      const float* q,
      const float* k,
      const float* v,
      float* out,
      const AttentionIoPolicy& policy) const;

    runtime::Status trainStep(
      const float* q,
      const float* k,
      float* v,
      const float* target,
      float* out,
      float learning_rate,
      float* loss_out) const;
    runtime::Status trainStepWithPolicy(
      const float* q,
      const float* k,
      float* v,
      const float* target,
      float* out,
      float learning_rate,
      float* loss_out,
      const AttentionIoPolicy& policy) const;

   private:
    AttentionConfig cfg_;
    runtime::Device device_;
    AttentionIoPolicy default_policy_;
      mutable std::size_t train_step_count_ = 0;
      mutable float last_loss_ = 0.0f;
      mutable bool has_last_loss_ = false;
  };

}  // namespace lightning_core
