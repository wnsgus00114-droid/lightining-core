#pragma once

#include <cstddef>

#include "lightning_core/core/attention.hpp"
#include "lightning_core/core/model_customization.hpp"
#include "lightning_core/core/ops.hpp"

namespace lightning_core::models {

struct TransformerFastPathConfig {
  std::size_t seq_len = 0;
  std::size_t head_dim = 0;
  bool causal = false;
  bool training = true;
};

class TransformerFastPath {
 public:
  TransformerFastPath(const TransformerFastPathConfig& cfg, runtime::Device device)
      : cfg_(cfg),
        device_(device),
        custom_(makeAggressiveCustomization(
            ModelFamily::kTransformer,
            cfg.training ? ExecutionMode::kTraining : ExecutionMode::kInference,
            cfg.seq_len,
            cfg.head_dim)) {
    attn_cfg_.seq_len = cfg.seq_len;
    attn_cfg_.head_dim = cfg.head_dim;
    attn_cfg_.causal = cfg.causal;
  }

  const TransformerFastPathConfig& config() const {
    return cfg_;
  }

  runtime::Device device() const {
    return device_;
  }

  const ModelCustomization& customization() const {
    return custom_;
  }

  AttentionIoPolicy attentionPolicy(LoopStage stage) const {
    return makeAttentionPolicyForLoop(custom_, stage);
  }

  ops::MatMulIoPolicy matMulPolicy(LoopStage stage) const {
    return makeMatMulPolicyForLoop(custom_, stage);
  }

  runtime::Status attentionForward(
      const float* q,
      const float* k,
      const float* v,
      float* out,
      LoopStage stage = LoopStage::kOneShot) const {
    return lightning_core::attentionForwardWithPolicy(
        q,
        k,
        v,
        out,
        attn_cfg_,
        device_,
        attentionPolicy(stage));
  }

  runtime::Status attentionTrainStep(
      const float* q,
      const float* k,
      float* v,
      const float* target,
      float* out,
      float learning_rate,
      float* loss_out,
      LoopStage stage = LoopStage::kOneShot) const {
    return lightning_core::attentionTrainStepWithPolicy(
        q,
        k,
        v,
        target,
        out,
        learning_rate,
        attn_cfg_,
        device_,
        loss_out,
        attentionPolicy(stage));
  }

 private:
  TransformerFastPathConfig cfg_;
  runtime::Device device_;
  ModelCustomization custom_;
  AttentionConfig attn_cfg_{};
};

}  // namespace lightning_core::models
