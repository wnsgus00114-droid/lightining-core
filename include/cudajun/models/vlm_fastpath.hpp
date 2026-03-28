#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "cudajun/attention.hpp"
#include "cudajun/model_customization.hpp"
#include "cudajun/ops.hpp"
#include "cudajun/runtime.hpp"

namespace cudajun::models {

struct VlmFastPathConfig {
  std::size_t image_tokens = 0;
  std::size_t text_tokens = 0;
  std::size_t vision_dim = 0;
  std::size_t text_dim = 0;
  std::size_t fused_dim = 0;
  std::size_t image_channels = 3;
  std::size_t patch_size = 16;
  bool training = true;
};

class VlmFastPath {
 public:
  VlmFastPath(const VlmFastPathConfig& cfg, runtime::Device device)
      : cfg_(cfg),
        device_(device),
        custom_(makeAggressiveCustomization(
            ModelFamily::kVlm,
            cfg.training ? ExecutionMode::kTraining : ExecutionMode::kInference,
            cfg.image_tokens + cfg.text_tokens,
            cfg.fused_dim)) {
    attn_cfg_.seq_len = cfg_.image_tokens + cfg_.text_tokens;
    attn_cfg_.head_dim = cfg_.fused_dim;
    attn_cfg_.causal = false;
      cross_attn_cfg_.seq_len = cfg_.image_tokens + cfg_.text_tokens;
      cross_attn_cfg_.head_dim = cfg_.fused_dim;
      cross_attn_cfg_.causal = false;
  }

  const VlmFastPathConfig& config() const {
    return cfg_;
  }

  const ModelCustomization& customization() const {
    return custom_;
  }

  ops::MatMulIoPolicy projectionPolicy(LoopStage stage) const {
    return makeMatMulPolicyForLoop(custom_, stage);
  }

  AttentionIoPolicy fusionPolicy(LoopStage stage) const {
    return makeAttentionPolicyForLoop(custom_, stage);
  }

  runtime::Status projectVision(
      const float* image_tokens,
      const float* w_vision,
      float* image_proj,
      LoopStage stage = LoopStage::kOneShot) const {
    return ops::matMulWithPolicy<float>(
        image_tokens,
        w_vision,
        image_proj,
        cfg_.image_tokens,
        cfg_.vision_dim,
        cfg_.fused_dim,
        device_,
        projectionPolicy(stage));
  }

  runtime::Status projectText(
      const float* text_tokens,
      const float* w_text,
      float* text_proj,
      LoopStage stage = LoopStage::kOneShot) const {
    return ops::matMulWithPolicy<float>(
        text_tokens,
        w_text,
        text_proj,
        cfg_.text_tokens,
        cfg_.text_dim,
        cfg_.fused_dim,
        device_,
        projectionPolicy(stage));
  }

  runtime::Status runFusionAttention(
      const float* q,
      const float* k,
      const float* v,
      float* out,
      LoopStage stage = LoopStage::kOneShot) const {
    return attentionForwardWithPolicy(q, k, v, out, attn_cfg_, device_, fusionPolicy(stage));
  }

  // Image patch embedding integration: patchify NHWC image and project patches in one call.
  runtime::Status patchEmbedFromImage(
      const float* image_nhwc,
      std::size_t image_h,
      std::size_t image_w,
      const float* w_patch,
      float* image_proj,
      LoopStage stage = LoopStage::kOneShot) const {
    if (image_nhwc == nullptr || w_patch == nullptr || image_proj == nullptr) {
      return runtime::Status::kInvalidValue;
    }
    const std::size_t ps = cfg_.patch_size;
    const std::size_t c = cfg_.image_channels;
    if (ps == 0 || c == 0 || image_h < ps || image_w < ps || image_h % ps != 0 || image_w % ps != 0) {
      return runtime::Status::kInvalidValue;
    }

    const std::size_t patch_h = image_h / ps;
    const std::size_t patch_w = image_w / ps;
    const std::size_t num_patches = patch_h * patch_w;
    const std::size_t patch_dim = ps * ps * c;
    if (num_patches != cfg_.image_tokens || patch_dim != cfg_.vision_dim) {
      return runtime::Status::kInvalidValue;
    }

    scratch_patch_tokens_.assign(num_patches * patch_dim, 0.0f);
    std::size_t pidx = 0;
    for (std::size_t py = 0; py < patch_h; ++py) {
      for (std::size_t px = 0; px < patch_w; ++px) {
        float* dst = scratch_patch_tokens_.data() + pidx * patch_dim;
        std::size_t d = 0;
        for (std::size_t iy = 0; iy < ps; ++iy) {
          for (std::size_t ix = 0; ix < ps; ++ix) {
            const std::size_t y = py * ps + iy;
            const std::size_t x = px * ps + ix;
            const float* src = image_nhwc + (y * image_w + x) * c;
            for (std::size_t ch = 0; ch < c; ++ch) {
              dst[d++] = src[ch];
            }
          }
        }
        ++pidx;
      }
    }

    return projectVision(scratch_patch_tokens_.data(), w_patch, image_proj, stage);
  }

  // Dedicated cross-attention path: packs text queries and image+text KV internally.
  runtime::Status runCrossAttentionFast(
      const float* image_proj,
      const float* text_proj,
      float* out_text,
      LoopStage stage = LoopStage::kOneShot) const {
    if (image_proj == nullptr || text_proj == nullptr || out_text == nullptr) {
      return runtime::Status::kInvalidValue;
    }
    const std::size_t s_img = cfg_.image_tokens;
    const std::size_t s_txt = cfg_.text_tokens;
    const std::size_t d = cfg_.fused_dim;
    const std::size_t s_all = s_img + s_txt;

    scratch_q_pack_.assign(s_all * d, 0.0f);
    scratch_k_pack_.assign(s_all * d, 0.0f);
    scratch_v_pack_.assign(s_all * d, 0.0f);
    scratch_out_pack_.assign(s_all * d, 0.0f);

    // Q: image prefix zeros + text queries, KV: image prefix + text prefix.
    std::copy(image_proj, image_proj + s_img * d, scratch_k_pack_.begin());
    std::copy(text_proj, text_proj + s_txt * d, scratch_k_pack_.begin() + s_img * d);
    std::copy(scratch_k_pack_.begin(), scratch_k_pack_.end(), scratch_v_pack_.begin());
    std::copy(text_proj, text_proj + s_txt * d, scratch_q_pack_.begin() + s_img * d);

    runtime::Status st = attentionForwardWithPolicy(
        scratch_q_pack_.data(),
        scratch_k_pack_.data(),
        scratch_v_pack_.data(),
        scratch_out_pack_.data(),
        cross_attn_cfg_,
        device_,
        fusionPolicy(stage));
    if (st != runtime::Status::kSuccess) {
      return st;
    }

    std::copy(scratch_out_pack_.begin() + s_img * d, scratch_out_pack_.end(), out_text);
    return runtime::Status::kSuccess;
  }

 private:
  VlmFastPathConfig cfg_;
  runtime::Device device_;
  ModelCustomization custom_;
  AttentionConfig attn_cfg_{};
  AttentionConfig cross_attn_cfg_{};
  mutable std::vector<float> scratch_patch_tokens_;
  mutable std::vector<float> scratch_q_pack_;
  mutable std::vector<float> scratch_k_pack_;
  mutable std::vector<float> scratch_v_pack_;
  mutable std::vector<float> scratch_out_pack_;
};

}  // namespace cudajun::models
