#include <iostream>

#include "lightning_core/model_customization.hpp"
#include "lightning_core/models/dnn_cnn_fastpath.hpp"
#include "lightning_core/models/graph_fastpath.hpp"
#include "lightning_core/models/lstm_rnn_fastpath.hpp"
#include "lightning_core/models/transformer_fastpath.hpp"

int main() {
  using lightning_core::ExecutionMode;
  using lightning_core::LoopStage;
  using lightning_core::ModelFamily;

  const auto tr_train = lightning_core::makeAggressiveCustomization(
      ModelFamily::kTransformer, ExecutionMode::kTraining, 2048, 128);

  if (!tr_train.resident_io || tr_train.attention_loss_every != 16) {
    std::cerr << "Transformer training preset mismatch\n";
    return 1;
  }

  const auto attn_run = lightning_core::makeAttentionPolicyForLoop(tr_train, LoopStage::kRun);
  if (attn_run.upload_q || attn_run.upload_k || attn_run.upload_v || attn_run.upload_target ||
      attn_run.download_out || attn_run.download_v || attn_run.synchronize) {
    std::cerr << "Attention run policy mismatch\n";
    return 1;
  }

  const auto attn_finish = lightning_core::makeAttentionPolicyForLoop(tr_train, LoopStage::kFinish);
  if (attn_finish.upload_q || attn_finish.upload_k || attn_finish.upload_v || attn_finish.upload_target ||
      !attn_finish.download_out || !attn_finish.download_v || !attn_finish.synchronize) {
    std::cerr << "Attention finish policy mismatch\n";
    return 1;
  }

  const auto rnn_inf = lightning_core::makeAggressiveCustomization(
      ModelFamily::kRnn, ExecutionMode::kInference, 256, 512);
  if (rnn_inf.vector_oneshot_crossover_n != (1u << 20) || !rnn_inf.prefer_enqueue_only_loop) {
    std::cerr << "RNN inference preset mismatch\n";
    return 1;
  }

  const auto mm_start = lightning_core::makeMatMulPolicyForLoop(rnn_inf, LoopStage::kStart);
  const auto mm_run = lightning_core::makeMatMulPolicyForLoop(rnn_inf, LoopStage::kRun);
  const auto mm_finish = lightning_core::makeMatMulPolicyForLoop(rnn_inf, LoopStage::kFinish);

  if (!mm_start.upload_a || !mm_start.upload_b || mm_start.download_out || mm_start.synchronize) {
    std::cerr << "MatMul start policy mismatch\n";
    return 1;
  }
  if (mm_run.upload_a || mm_run.upload_b || mm_run.download_out || mm_run.synchronize) {
    std::cerr << "MatMul run policy mismatch\n";
    return 1;
  }
  if (mm_finish.upload_a || mm_finish.upload_b || !mm_finish.download_out || !mm_finish.synchronize) {
    std::cerr << "MatMul finish policy mismatch\n";
    return 1;
  }

  lightning_core::models::TransformerFastPathConfig tf_cfg;
  tf_cfg.seq_len = 1024;
  tf_cfg.head_dim = 64;
  tf_cfg.causal = true;
  tf_cfg.training = true;
  lightning_core::models::TransformerFastPath tf(tf_cfg, lightning_core::runtime::Device::kMetal);

  const auto tf_run = tf.attentionPolicy(LoopStage::kRun);
  if (tf_run.upload_q || tf_run.upload_k || tf_run.upload_v || tf_run.upload_target ||
      tf_run.download_out || tf_run.download_v || tf_run.synchronize) {
    std::cerr << "Transformer fastpath run policy mismatch\n";
    return 1;
  }

  const auto tf_mm_finish = tf.matMulPolicy(LoopStage::kFinish);
  if (tf_mm_finish.upload_a || tf_mm_finish.upload_b || !tf_mm_finish.download_out ||
      !tf_mm_finish.synchronize) {
    std::cerr << "Transformer fastpath matmul finish policy mismatch\n";
    return 1;
  }

  lightning_core::models::LstmRnnFastPathConfig lstm_cfg;
  lstm_cfg.input_dim = 256;
  lstm_cfg.hidden_dim = 512;
  lstm_cfg.training = true;
  lstm_cfg.lstm_mode = true;
  lightning_core::models::LstmRnnFastPath lstm(lstm_cfg, lightning_core::runtime::Device::kMetal);
  const auto lstm_proj_run = lstm.recurrentProjectPolicy(LoopStage::kRun);
  if (lstm_proj_run.upload_a || lstm_proj_run.upload_b || lstm_proj_run.download_out ||
      lstm_proj_run.synchronize) {
    std::cerr << "LSTM project run policy mismatch\n";
    return 1;
  }

  lightning_core::models::DnnCnnFastPathConfig cnn_cfg;
  cnn_cfg.in_dim = 1024;
  cnn_cfg.out_dim = 512;
  cnn_cfg.training = false;
  cnn_cfg.cnn_mode = true;
  lightning_core::models::DnnCnnFastPath cnn(cnn_cfg, lightning_core::runtime::Device::kMetal);
  const auto cnn_elem_finish = cnn.elemwisePolicy(LoopStage::kFinish);
  if (cnn_elem_finish.upload_a || cnn_elem_finish.upload_b || !cnn_elem_finish.download_out ||
      !cnn_elem_finish.synchronize) {
    std::cerr << "CNN elemwise finish policy mismatch\n";
    return 1;
  }

  const auto gcn_policy = lightning_core::models::makeGraphSparseFriendlyPolicy(
      ModelFamily::kGcn,
      ExecutionMode::kTraining,
      131072,
      1048576,
      128);
  if (!gcn_policy.resident_aggregation || gcn_policy.edge_chunk_size < 4096 ||
      gcn_policy.node_batch_size < 512) {
    std::cerr << "GCN sparse-friendly policy mismatch\n";
    return 1;
  }

  const auto gat_mm_start = lightning_core::models::makeGraphProjectionPolicy(
      lightning_core::models::makeGraphSparseFriendlyPolicy(
          ModelFamily::kGat,
          ExecutionMode::kInference,
          10000,
          50000,
          64),
      LoopStage::kStart);
  if (!gat_mm_start.upload_a || !gat_mm_start.upload_b || gat_mm_start.download_out ||
      gat_mm_start.synchronize) {
    std::cerr << "GAT graph projection start policy mismatch\n";
    return 1;
  }

  const auto gcn_cached_a = lightning_core::models::makeGraphSparseFriendlyPolicyCached(
      ModelFamily::kGcn,
      ExecutionMode::kTraining,
      2048,
      8192,
      64);
  const auto gcn_cached_b = lightning_core::models::makeGraphSparseFriendlyPolicyCached(
      ModelFamily::kGcn,
      ExecutionMode::kTraining,
      2048,
      8192,
      64);
  if (gcn_cached_a.edge_chunk_size != gcn_cached_b.edge_chunk_size ||
      gcn_cached_a.node_batch_size != gcn_cached_b.node_batch_size) {
    std::cerr << "GCN cached policy mismatch\n";
    return 1;
  }

  lightning_core::models::saveGraphPolicyCacheIfDirty();

  std::cout << "test_model_customization ok\n";
  return 0;
}
