#include <iostream>
#include <vector>

#include "cudajun/models/vlm_fastpath.hpp"

int main() {
  cudajun::models::VlmFastPathConfig cfg;
  cfg.image_tokens = 2;
  cfg.text_tokens = 2;
  cfg.vision_dim = 4;
  cfg.text_dim = 2;
  cfg.fused_dim = 2;
  cfg.image_channels = 1;
  cfg.patch_size = 2;
  cfg.training = false;

  cudajun::models::VlmFastPath vlm(cfg, cudajun::runtime::Device::kCPU);

  std::vector<float> image_tokens(cfg.image_tokens * cfg.vision_dim, 1.0f);
  std::vector<float> text_tokens(cfg.text_tokens * cfg.text_dim, 1.0f);
  std::vector<float> image_nhwc(4 * 2, 1.0f);

  std::vector<float> w_vision(cfg.vision_dim * cfg.fused_dim, 0.5f);
  std::vector<float> w_text(cfg.text_dim * cfg.fused_dim, 0.25f);
  std::vector<float> w_patch(cfg.vision_dim * cfg.fused_dim, 0.5f);

  std::vector<float> image_proj(cfg.image_tokens * cfg.fused_dim, 0.0f);
  std::vector<float> text_proj(cfg.text_tokens * cfg.fused_dim, 0.0f);

  if (vlm.projectVision(image_tokens.data(), w_vision.data(), image_proj.data()) != cudajun::runtime::Status::kSuccess) {
    std::cerr << "VLM projectVision failed\n";
    return 1;
  }
  if (vlm.patchEmbedFromImage(image_nhwc.data(), 2, 4, w_patch.data(), image_proj.data()) !=
      cudajun::runtime::Status::kSuccess) {
    std::cerr << "VLM patchEmbedFromImage failed\n";
    return 1;
  }
  if (vlm.projectText(text_tokens.data(), w_text.data(), text_proj.data()) != cudajun::runtime::Status::kSuccess) {
    std::cerr << "VLM projectText failed\n";
    return 1;
  }

  std::size_t total = cfg.image_tokens + cfg.text_tokens;
  std::vector<float> q(total * cfg.fused_dim, 0.1f);
  std::vector<float> k(total * cfg.fused_dim, 0.2f);
  std::vector<float> v(total * cfg.fused_dim, 0.3f);
  std::vector<float> out(total * cfg.fused_dim, 0.0f);

  if (vlm.runFusionAttention(q.data(), k.data(), v.data(), out.data()) != cudajun::runtime::Status::kSuccess) {
    std::cerr << "VLM fusion attention failed\n";
    return 1;
  }

  std::vector<float> out_text(cfg.text_tokens * cfg.fused_dim, 0.0f);
  if (vlm.runCrossAttentionFast(image_proj.data(), text_proj.data(), out_text.data()) !=
      cudajun::runtime::Status::kSuccess) {
    std::cerr << "VLM cross attention fast path failed\n";
    return 1;
  }

  auto p = vlm.fusionPolicy(cudajun::LoopStage::kRun);
  if (p.upload_q || p.upload_k || p.upload_v || p.download_out || p.synchronize) {
    std::cerr << "VLM run fusion policy mismatch\n";
    return 1;
  }

  std::cout << "test_vlm_fastpath ok\n";
  return 0;
}
