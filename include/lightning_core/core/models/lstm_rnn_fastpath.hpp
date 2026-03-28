#pragma once

#include <cstddef>

#include "lightning_core/core/model_customization.hpp"
#include "lightning_core/core/ops.hpp"
#include "lightning_core/core/runtime.hpp"

namespace lightning_core::models {

struct LstmRnnFastPathConfig {
  std::size_t input_dim = 0;
  std::size_t hidden_dim = 0;
  bool training = true;
  bool lstm_mode = true;
};

class LstmRnnFastPath {
 public:
  LstmRnnFastPath(const LstmRnnFastPathConfig& cfg, runtime::Device device)
      : cfg_(cfg),
        device_(device),
        custom_(makeAggressiveCustomization(
            cfg.lstm_mode ? ModelFamily::kLstm : ModelFamily::kRnn,
            cfg.training ? ExecutionMode::kTraining : ExecutionMode::kInference,
            0,
            cfg.hidden_dim)) {}

  const LstmRnnFastPathConfig& config() const {
    return cfg_;
  }

  runtime::Device device() const {
    return device_;
  }

  const ModelCustomization& customization() const {
    return custom_;
  }

  ops::MatMulIoPolicy recurrentProjectPolicy(LoopStage stage) const {
    return makeMatMulPolicyForLoop(custom_, stage);
  }

  ops::VectorAddIoPolicy recurrentFusePolicy(LoopStage stage) const {
    return makeVectorPolicyForLoop(custom_, stage);
  }

  runtime::Status projectInput(
      const float* x,
      const float* w_x,
      float* out,
      std::size_t batch,
      LoopStage stage = LoopStage::kOneShot) const {
    return ops::matMulWithPolicy<float>(
        x,
        w_x,
        out,
        batch,
        cfg_.input_dim,
        cfg_.hidden_dim,
        device_,
        recurrentProjectPolicy(stage));
  }

  runtime::Status projectHidden(
      const float* h,
      const float* w_h,
      float* out,
      std::size_t batch,
      LoopStage stage = LoopStage::kOneShot) const {
    return ops::matMulWithPolicy<float>(
        h,
        w_h,
        out,
        batch,
        cfg_.hidden_dim,
        cfg_.hidden_dim,
        device_,
        recurrentProjectPolicy(stage));
  }

  runtime::Status fuseRecurrent(
      const float* a,
      const float* b,
      float* out,
      std::size_t n,
      LoopStage stage = LoopStage::kOneShot) const {
    return ops::vectorAddWithPolicy<float>(a, b, out, n, device_, recurrentFusePolicy(stage));
  }

 private:
  LstmRnnFastPathConfig cfg_;
  runtime::Device device_;
  ModelCustomization custom_;
};

}  // namespace lightning_core::models
