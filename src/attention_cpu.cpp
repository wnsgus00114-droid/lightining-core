#include "cudajun/detail/attention_backend.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace cudajun::detail {

namespace {

float dotFloat(const float* a, const float* b, std::size_t n) {
#if defined(__ARM_NEON)
  std::size_t i = 0;
  float32x4_t acc = vdupq_n_f32(0.0f);
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    acc = vfmaq_f32(acc, va, vb);
  }
  float sum = vaddvq_f32(acc);
  for (; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
#else
  float sum = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
#endif
}

runtime::Status computeAttentionForward(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const AttentionConfig& cfg,
    std::vector<float>* probs_out) {
  const std::size_t s = cfg.seq_len;
  const std::size_t d = cfg.head_dim;

  std::vector<float> scores(s * s, 0.0f);
  std::vector<float> probs(s * s, 0.0f);

  const float scale = 1.0f / std::sqrt(static_cast<float>(d));

  for (std::size_t i = 0; i < s; ++i) {
    const float* qi = q + i * d;
    for (std::size_t j = 0; j < s; ++j) {
      if (cfg.causal && j > i) {
        scores[i * s + j] = -1e30f;
      } else {
        const float* kj = k + j * d;
        scores[i * s + j] = dotFloat(qi, kj, d) * scale;
      }
    }
  }

  for (std::size_t i = 0; i < s; ++i) {
    float row_max = scores[i * s];
    for (std::size_t j = 1; j < s; ++j) {
      row_max = std::max(row_max, scores[i * s + j]);
    }

    float row_sum = 0.0f;
    for (std::size_t j = 0; j < s; ++j) {
      float e = std::exp(scores[i * s + j] - row_max);
      probs[i * s + j] = e;
      row_sum += e;
    }

    float inv = 1.0f / row_sum;
    for (std::size_t j = 0; j < s; ++j) {
      probs[i * s + j] *= inv;
    }
  }

  std::fill(out, out + s * d, 0.0f);
  for (std::size_t i = 0; i < s; ++i) {
    for (std::size_t j = 0; j < s; ++j) {
      float p = probs[i * s + j];
      const float* vj = v + j * d;
      float* oi = out + i * d;
      for (std::size_t c = 0; c < d; ++c) {
        oi[c] += p * vj[c];
      }
    }
  }

  if (probs_out != nullptr) {
    *probs_out = std::move(probs);
  }

  return runtime::Status::kSuccess;
}

}  // namespace

runtime::Status attentionForwardCpu(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const AttentionConfig& cfg) {
  return computeAttentionForward(q, k, v, out, cfg, nullptr);
}

runtime::Status attentionTrainStepCpu(
    const float* q,
    const float* k,
    float* v,
    const float* target,
    float* out,
    float learning_rate,
    const AttentionConfig& cfg,
    float* loss_out) {
  const std::size_t s = cfg.seq_len;
  const std::size_t d = cfg.head_dim;

  std::vector<float> probs;
  runtime::Status st = computeAttentionForward(q, k, v, out, cfg, &probs);
  if (st != runtime::Status::kSuccess) {
    return st;
  }

  std::vector<float> grad_out(s * d, 0.0f);
  float loss = 0.0f;
  const float norm = 1.0f / static_cast<float>(s * d);

  for (std::size_t i = 0; i < s * d; ++i) {
    float diff = out[i] - target[i];
    loss += diff * diff;
    grad_out[i] = 2.0f * diff * norm;
  }

  std::vector<float> grad_v(s * d, 0.0f);
  for (std::size_t j = 0; j < s; ++j) {
    for (std::size_t i = 0; i < s; ++i) {
      float p = probs[i * s + j];
      const float* go = grad_out.data() + i * d;
      float* gv = grad_v.data() + j * d;
      for (std::size_t c = 0; c < d; ++c) {
        gv[c] += p * go[c];
      }
    }
  }

  for (std::size_t i = 0; i < s * d; ++i) {
    v[i] -= learning_rate * grad_v[i];
  }

  if (loss_out != nullptr) {
    *loss_out = loss * norm;
  }

  return runtime::Status::kSuccess;
}

}  // namespace cudajun::detail
