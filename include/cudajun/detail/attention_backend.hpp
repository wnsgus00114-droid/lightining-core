#pragma once

#include "cudajun/attention.hpp"

namespace cudajun::detail {

runtime::Status attentionForwardCpu(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const AttentionConfig& cfg);

runtime::Status attentionTrainStepCpu(
    const float* q,
    const float* k,
    float* v,
    const float* target,
    float* out,
    float learning_rate,
    const AttentionConfig& cfg,
    float* loss_out);

runtime::Status attentionForwardMetal(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const AttentionConfig& cfg);

runtime::Status attentionForwardMetalWithPolicy(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const AttentionConfig& cfg,
    const AttentionIoPolicy& policy);

runtime::Status attentionTrainStepMetal(
    const float* q,
    const float* k,
    float* v,
    const float* target,
    float* out,
    float learning_rate,
    const AttentionConfig& cfg,
    float* loss_out);

runtime::Status attentionTrainStepMetalWithPolicy(
    const float* q,
    const float* k,
    float* v,
    const float* target,
    float* out,
    float learning_rate,
    const AttentionConfig& cfg,
    float* loss_out,
    const AttentionIoPolicy& policy);

}  // namespace cudajun::detail
