#pragma once

#include <cstddef>

#include "lightning_core/core/attention.hpp"
#include "lightning_core/core/ops.hpp"

namespace lightning_core {

enum class ModelFamily {
  kTransformer,
  kLstm,
  kRnn,
  kDnn,
  kCnn,
  kGcn,
  kGat,
  kVlm,
};

enum class ExecutionMode {
  kTraining,
  kInference,
};

enum class LoopStage {
  kOneShot,
  kStart,
  kRun,
  kFinish,
};

struct ModelCustomization {
  ModelFamily family = ModelFamily::kTransformer;
  ExecutionMode mode = ExecutionMode::kTraining;

  // Enable resident-style IO scheduling for repeated loops.
  bool resident_io = true;

  // Hint: benchmark and tuning should evaluate enqueue-only loop latency.
  bool prefer_enqueue_only_loop = true;

  // Vector add one-shot crossover hint for host-vs-metal routing.
  std::size_t vector_oneshot_crossover_n = (8u << 20);
  bool vector_crossover_dynamic = false;

  // Attention postprocess and train-loss cadence hints.
  int attention_loss_every = 0;
  float attention_output_scale = 1.0f;
  float attention_output_bias = 0.0f;
  bool attention_output_relu = false;
};

inline bool isTraining(ExecutionMode mode) {
  return mode == ExecutionMode::kTraining;
}

inline ModelCustomization makeAggressiveCustomization(
    ModelFamily family,
    ExecutionMode mode,
    std::size_t seq_len = 0,
    std::size_t hidden_dim = 0) {
  (void)hidden_dim;

  ModelCustomization c;
  c.family = family;
  c.mode = mode;

  switch (family) {
    case ModelFamily::kTransformer:
    case ModelFamily::kVlm:
      c.resident_io = true;
      c.prefer_enqueue_only_loop = true;
      c.vector_oneshot_crossover_n = (8u << 20);
      c.attention_loss_every = isTraining(mode) ? ((seq_len >= 1024) ? 16 : 8) : 1;
      break;

    case ModelFamily::kLstm:
    case ModelFamily::kRnn:
      c.resident_io = true;
      c.prefer_enqueue_only_loop = true;
      c.vector_oneshot_crossover_n = (1u << 20);
      c.attention_loss_every = isTraining(mode) ? 8 : 1;
      break;

    case ModelFamily::kDnn:
    case ModelFamily::kCnn:
      c.resident_io = true;
      c.prefer_enqueue_only_loop = true;
      c.vector_oneshot_crossover_n = (4u << 20);
      c.attention_loss_every = isTraining(mode) ? 4 : 1;
      break;

    case ModelFamily::kGcn:
    case ModelFamily::kGat:
      c.resident_io = true;
      c.prefer_enqueue_only_loop = true;
      c.vector_oneshot_crossover_n = (2u << 20);
      c.attention_loss_every = isTraining(mode) ? 6 : 1;
      break;
  }

  return c;
}

inline ops::MatMulIoPolicy makeMatMulPolicyForLoop(const ModelCustomization& c, LoopStage stage) {
  if (!c.resident_io || stage == LoopStage::kOneShot) {
    return ops::MatMulIoPolicy{};
  }
  if (stage == LoopStage::kStart) {
    return ops::makeMetalResidentStartPolicy();
  }
  if (stage == LoopStage::kRun) {
    return ops::makeMetalResidentRunPolicy();
  }
  return ops::makeMetalResidentFinishPolicy();
}

inline ops::MatrixElementwiseIoPolicy makeElemwisePolicyForLoop(const ModelCustomization& c, LoopStage stage) {
  if (!c.resident_io || stage == LoopStage::kOneShot) {
    return ops::MatrixElementwiseIoPolicy{};
  }
  if (stage == LoopStage::kStart) {
    return ops::makeMetalElemwiseResidentStartPolicy();
  }
  if (stage == LoopStage::kRun) {
    return ops::makeMetalElemwiseResidentRunPolicy();
  }
  return ops::makeMetalElemwiseResidentFinishPolicy();
}

inline ops::VectorAddIoPolicy makeVectorPolicyForLoop(const ModelCustomization& c, LoopStage stage) {
  if (!c.resident_io || stage == LoopStage::kOneShot) {
    return ops::VectorAddIoPolicy{};
  }
  if (stage == LoopStage::kStart) {
    return ops::makeMetalVectorResidentStartPolicy();
  }
  if (stage == LoopStage::kRun) {
    return ops::makeMetalVectorResidentRunPolicy();
  }
  return ops::makeMetalVectorResidentFinishPolicy();
}

inline AttentionIoPolicy makeAttentionPolicyForLoop(const ModelCustomization& c, LoopStage stage) {
  AttentionIoPolicy p;
  p.loss_every = c.attention_loss_every;
  p.output_scale = c.attention_output_scale;
  p.output_bias = c.attention_output_bias;
  p.output_relu = c.attention_output_relu;

  if (!c.resident_io || stage == LoopStage::kOneShot) {
    return p;
  }

  if (stage == LoopStage::kStart) {
    p.upload_q = true;
    p.upload_k = true;
    p.upload_v = true;
    p.upload_target = true;
    p.download_out = false;
    p.download_v = false;
    p.synchronize = false;
    return p;
  }
  if (stage == LoopStage::kRun) {
    p.upload_q = false;
    p.upload_k = false;
    p.upload_v = false;
    p.upload_target = false;
    p.download_out = false;
    p.download_v = false;
    p.synchronize = false;
    return p;
  }

  p.upload_q = false;
  p.upload_k = false;
  p.upload_v = false;
  p.upload_target = false;
  p.download_out = true;
  p.download_v = isTraining(c.mode);
  p.synchronize = true;
  return p;
}

}  // namespace lightning_core
