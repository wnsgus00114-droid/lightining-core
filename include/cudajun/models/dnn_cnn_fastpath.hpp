#pragma once

#include <cstddef>

#include "cudajun/model_customization.hpp"
#include "cudajun/ops.hpp"
#include "cudajun/runtime.hpp"

namespace cudajun::models {

struct DnnCnnFastPathConfig {
  std::size_t in_dim = 0;
  std::size_t out_dim = 0;
  bool training = true;
  bool cnn_mode = false;
};

class DnnCnnFastPath {
 public:
  DnnCnnFastPath(const DnnCnnFastPathConfig& cfg, runtime::Device device)
      : cfg_(cfg),
        device_(device),
        custom_(makeAggressiveCustomization(
            cfg.cnn_mode ? ModelFamily::kCnn : ModelFamily::kDnn,
            cfg.training ? ExecutionMode::kTraining : ExecutionMode::kInference,
            0,
            cfg.out_dim)) {}

  const DnnCnnFastPathConfig& config() const {
    return cfg_;
  }

  runtime::Device device() const {
    return device_;
  }

  const ModelCustomization& customization() const {
    return custom_;
  }

  ops::MatMulIoPolicy densePolicy(LoopStage stage) const {
    return makeMatMulPolicyForLoop(custom_, stage);
  }

  ops::MatrixElementwiseIoPolicy elemwisePolicy(LoopStage stage) const {
    return makeElemwisePolicyForLoop(custom_, stage);
  }

  runtime::Status denseProject(
      const float* x,
      const float* w,
      float* out,
      std::size_t batch,
      LoopStage stage = LoopStage::kOneShot) const {
    return ops::matMulWithPolicy<float>(x, w, out, batch, cfg_.in_dim, cfg_.out_dim, device_, densePolicy(stage));
  }

  runtime::Status residualSub(
      const float* a,
      const float* b,
      float* out,
      std::size_t rows,
      std::size_t cols,
      LoopStage stage = LoopStage::kOneShot) const {
    return ops::matrixSubWithPolicy<float>(a, b, out, rows, cols, device_, elemwisePolicy(stage));
  }

  runtime::Status channelNormDiv(
      const float* a,
      const float* b,
      float* out,
      std::size_t rows,
      std::size_t cols,
      LoopStage stage = LoopStage::kOneShot) const {
    return ops::matrixDivWithPolicy<float>(a, b, out, rows, cols, device_, elemwisePolicy(stage));
  }

 private:
  DnnCnnFastPathConfig cfg_;
  runtime::Device device_;
  ModelCustomization custom_;
};

}  // namespace cudajun::models
