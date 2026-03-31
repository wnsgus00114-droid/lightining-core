#include <cstring>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "lightning_core/graph.hpp"

int main() {
  using lightning_core::graph::DType;
  using lightning_core::graph::GraphIR;
  using lightning_core::graph::OpKind;
  using lightning_core::graph::TensorSpec;
  using lightning_core::runtime::Device;
  using lightning_core::runtime::Status;

  const auto& registry = lightning_core::graph::globalOperatorRegistry();
  if (registry.size() < 5) {
    std::cerr << "operator registry should contain default schemas\n";
    return 1;
  }
  const auto* matmul_schema = registry.find(OpKind::kMatMul);
  if (matmul_schema == nullptr || !matmul_schema->supportsDevice(Device::kCPU)) {
    std::cerr << "matmul schema should exist and support CPU\n";
    return 1;
  }

  GraphIR g;
  std::size_t a = 0;
  std::size_t b = 0;
  std::size_t out = 0;

  TensorSpec spec;
  spec.shape = {2, 2};
  spec.dtype = DType::kFloat32;
  spec.layout = lightning_core::Layout::kContiguous;

  if (g.addTensorSpec(spec, &a, "a", true) != Status::kSuccess ||
      g.addTensorSpec(spec, &b, "b", true) != Status::kSuccess ||
      g.addTensorSpec(spec, &out, "out", false) != Status::kSuccess) {
    std::cerr << "addTensorSpec failed\n";
    return 1;
  }

  if (g.addNode(OpKind::kMatMul, {a, b}, {out}) != Status::kSuccess) {
    std::cerr << "addNode(matmul) failed\n";
    return 1;
  }
  if (g.validate() != Status::kSuccess) {
    std::cerr << "graph validate should succeed\n";
    return 1;
  }
  lightning_core::graph::ValidationReport good_report;
  if (g.validateWithReport(&good_report) != Status::kSuccess || !good_report.ok()) {
    std::cerr << "validateWithReport should succeed for valid graph\n";
    return 1;
  }
  if (!good_report.issues.empty()) {
    std::cerr << "valid graph should not produce validation issues\n";
    return 1;
  }
  if (g.numNodes() != 1 || g.numTensors() != 3) {
    std::cerr << "graph size mismatch\n";
    return 1;
  }

  std::vector<lightning_core::graph::GraphPlanStep> plan;
  if (g.planForDevice(Device::kMetal, &plan) != Status::kSuccess) {
    std::cerr << "planForDevice should succeed with fallback when needed\n";
    return 1;
  }
  if (plan.size() != 1 || plan[0].node_id != 0) {
    std::cerr << "plan shape mismatch\n";
    return 1;
  }
  if (!lightning_core::runtime::isMetalAvailable()) {
    if (!plan[0].fallback || plan[0].assigned_device != Device::kCPU) {
      std::cerr << "expected CPU fallback when Metal is unavailable\n";
      return 1;
    }
  }

  lightning_core::graph::GraphPlannerOptions planner_options;
  planner_options.preferred_device = Device::kCUDA;
  planner_options.sync_policy.mode = lightning_core::runtime::SyncMode::kAlways;
  planner_options.sync_policy.trace_sync_boundary = true;
  planner_options.separate_fallback_segments = true;
  planner_options.insert_sync_on_device_change = true;

  std::vector<lightning_core::graph::GraphExecutionGroup> groups;
  if (g.planExecutionGroups(planner_options, &groups, &plan) != Status::kSuccess) {
    std::cerr << "planExecutionGroups should succeed\n";
    return 1;
  }
  if (groups.empty()) {
    std::cerr << "planExecutionGroups should produce at least one group\n";
    return 1;
  }
  if (plan.size() != g.numNodes()) {
    std::cerr << "planExecutionGroups step count mismatch\n";
    return 1;
  }
  for (const auto& group : groups) {
    if (!group.sync_boundary_after) {
      std::cerr << "sync_mode=always should mark sync boundary after every group\n";
      return 1;
    }
  }

  // graph execution smoke: matmul -> vector_add
  GraphIR exec_graph;
  std::size_t exec_a = 0;
  std::size_t exec_b = 0;
  std::size_t exec_mm = 0;
  std::size_t exec_bias = 0;
  std::size_t exec_out = 0;
  if (exec_graph.addTensorSpec(spec, &exec_a, "exec_a", true) != Status::kSuccess ||
      exec_graph.addTensorSpec(spec, &exec_b, "exec_b", true) != Status::kSuccess ||
      exec_graph.addTensorSpec(spec, &exec_mm, "exec_mm", false) != Status::kSuccess ||
      exec_graph.addTensorSpec(spec, &exec_bias, "exec_bias", true) != Status::kSuccess ||
      exec_graph.addTensorSpec(spec, &exec_out, "exec_out", false) != Status::kSuccess) {
    std::cerr << "exec_graph addTensorSpec failed\n";
    return 1;
  }
  if (exec_graph.addNode(OpKind::kMatMul, {exec_a, exec_b}, {exec_mm}) != Status::kSuccess ||
      exec_graph.addNode(OpKind::kVectorAdd, {exec_mm, exec_bias}, {exec_out}) != Status::kSuccess) {
    std::cerr << "exec_graph addNode failed\n";
    return 1;
  }

  lightning_core::graph::GraphPlannerOptions exec_options;
  exec_options.preferred_device = Device::kMetal;
  exec_options.sync_policy.mode = lightning_core::runtime::SyncMode::kAuto;
  exec_options.sync_policy.trace_sync_boundary = false;
  exec_options.separate_fallback_segments = true;
  exec_options.insert_sync_on_device_change = true;

  std::unordered_map<std::size_t, std::vector<float>> exec_feeds;
  exec_feeds[exec_a] = {1.0f, 2.0f, 3.0f, 4.0f};
  exec_feeds[exec_b] = {5.0f, 6.0f, 7.0f, 8.0f};
  exec_feeds[exec_bias] = {1.0f, 2.0f, 3.0f, 4.0f};

  std::unordered_map<std::size_t, std::vector<float>> exec_values;
  std::vector<lightning_core::graph::GraphExecutionGroup> exec_groups;
  std::vector<lightning_core::graph::GraphPlanStep> exec_steps;
  if (exec_graph.executeF32(exec_options, exec_feeds, &exec_values, &exec_groups, &exec_steps) !=
      Status::kSuccess) {
    std::cerr << "executeF32 should succeed for matmul->vector_add graph\n";
    return 1;
  }
  const auto out_it = exec_values.find(exec_out);
  if (out_it == exec_values.end()) {
    std::cerr << "executeF32 should produce output tensor value\n";
    return 1;
  }
  const std::vector<float> expected_out = {20.0f, 24.0f, 46.0f, 54.0f};
  if (out_it->second.size() != expected_out.size()) {
    std::cerr << "executeF32 output size mismatch\n";
    return 1;
  }
  for (std::size_t i = 0; i < expected_out.size(); ++i) {
    const float diff = out_it->second[i] - expected_out[i];
    if (diff > 1e-4f || diff < -1e-4f) {
      std::cerr << "executeF32 output mismatch at index " << i << "\n";
      return 1;
    }
  }
  if (exec_steps.size() != exec_graph.numNodes()) {
    std::cerr << "executeF32 should return one plan step per node\n";
    return 1;
  }
  if (exec_groups.empty()) {
    std::cerr << "executeF32 should return execution groups\n";
    return 1;
  }

  const lightning_core::runtime::BackendCapabilities metal_caps =
      lightning_core::runtime::backendCapabilities(Device::kMetal);
  const bool metal_compute_built = metal_caps.compute_surface;
  const bool metal_exec_available = metal_caps.compute_surface && metal_caps.available;
  if (metal_exec_available) {
    // attention dispatch validation: graph output must match direct attentionForward call.
    GraphIR attn_graph;
    std::size_t tq = 0;
    std::size_t tk = 0;
    std::size_t tv = 0;
    std::size_t to = 0;
    TensorSpec attn_spec;
    attn_spec.shape = {4, 8};
    attn_spec.dtype = DType::kFloat32;
    attn_spec.layout = lightning_core::Layout::kContiguous;
    if (attn_graph.addTensorSpec(attn_spec, &tq, "q", true) != Status::kSuccess ||
        attn_graph.addTensorSpec(attn_spec, &tk, "k", true) != Status::kSuccess ||
        attn_graph.addTensorSpec(attn_spec, &tv, "v", true) != Status::kSuccess ||
        attn_graph.addTensorSpec(attn_spec, &to, "out", false) != Status::kSuccess) {
      std::cerr << "attn_graph addTensorSpec failed\n";
      return 1;
    }
    if (attn_graph.addNode(OpKind::kAttentionForward, {tq, tk, tv}, {to}) != Status::kSuccess) {
      std::cerr << "attn_graph addNode failed\n";
      return 1;
    }

    std::vector<float> qv(32, 0.0f);
    std::vector<float> kv(32, 0.0f);
    std::vector<float> vv(32, 0.0f);
    for (std::size_t i = 0; i < qv.size(); ++i) {
      qv[i] = static_cast<float>((i % 7) + 1) * 0.05f;
      kv[i] = static_cast<float>((i % 5) + 1) * 0.03f;
      vv[i] = static_cast<float>((i % 9) + 1) * 0.04f;
    }
    std::unordered_map<std::size_t, std::vector<float>> attn_feeds;
    attn_feeds[tq] = qv;
    attn_feeds[tk] = kv;
    attn_feeds[tv] = vv;
    std::unordered_map<std::size_t, std::vector<float>> attn_values;
    if (attn_graph.executeF32(exec_options, attn_feeds, &attn_values, nullptr, nullptr) != Status::kSuccess) {
      std::cerr << "attn_graph executeF32 failed\n";
      return 1;
    }
    const auto attn_out_it = attn_values.find(to);
    if (attn_out_it == attn_values.end() || attn_out_it->second.size() != qv.size()) {
      std::cerr << "attn_graph output missing or size mismatch\n";
      return 1;
    }
    std::vector<float> attn_expected(qv.size(), 0.0f);
    lightning_core::AttentionConfig attn_cfg{4, 8, false};
    if (lightning_core::attentionForward(
            qv.data(), kv.data(), vv.data(), attn_expected.data(), attn_cfg, Device::kMetal) != Status::kSuccess) {
      std::cerr << "direct attentionForward failed\n";
      return 1;
    }
    for (std::size_t i = 0; i < attn_expected.size(); ++i) {
      const float diff = attn_out_it->second[i] - attn_expected[i];
      if (diff > 1e-3f || diff < -1e-3f) {
        std::cerr << "attn_graph output mismatch at index " << i << "\n";
        return 1;
      }
    }

    // conv dispatch validation: graph output must match direct conv2dNchw call.
    GraphIR conv_exec_graph;
    std::size_t cx = 0;
    std::size_t cw = 0;
    std::size_t cb = 0;
    std::size_t cy = 0;
    TensorSpec cx_spec;
    cx_spec.shape = {1, 3, 8, 8};
    cx_spec.dtype = DType::kFloat32;
    cx_spec.layout = lightning_core::Layout::kContiguous;
    TensorSpec cw_spec;
    cw_spec.shape = {16, 3, 3, 3};
    cw_spec.dtype = DType::kFloat32;
    cw_spec.layout = lightning_core::Layout::kContiguous;
    TensorSpec cb_spec;
    cb_spec.shape = {16};
    cb_spec.dtype = DType::kFloat32;
    cb_spec.layout = lightning_core::Layout::kContiguous;
    TensorSpec cy_spec;
    cy_spec.shape = {1, 16, 8, 8};
    cy_spec.dtype = DType::kFloat32;
    cy_spec.layout = lightning_core::Layout::kContiguous;
    if (conv_exec_graph.addTensorSpec(cx_spec, &cx, "x", true) != Status::kSuccess ||
        conv_exec_graph.addTensorSpec(cw_spec, &cw, "w", true) != Status::kSuccess ||
        conv_exec_graph.addTensorSpec(cb_spec, &cb, "b", true) != Status::kSuccess ||
        conv_exec_graph.addTensorSpec(cy_spec, &cy, "y", false) != Status::kSuccess) {
      std::cerr << "conv_exec_graph addTensorSpec failed\n";
      return 1;
    }
    if (conv_exec_graph.addNode(OpKind::kConv2dNchw3x3s1p1, {cx, cw, cb}, {cy}) != Status::kSuccess) {
      std::cerr << "conv_exec_graph addNode failed\n";
      return 1;
    }

    std::vector<float> xv(1 * 3 * 8 * 8, 0.0f);
    std::vector<float> wv(16 * 3 * 3 * 3, 0.0f);
    std::vector<float> bv(16, 0.0f);
    for (std::size_t i = 0; i < xv.size(); ++i) {
      xv[i] = static_cast<float>((i % 13) + 1) * 0.01f;
    }
    for (std::size_t i = 0; i < wv.size(); ++i) {
      wv[i] = static_cast<float>((i % 11) + 1) * 0.02f;
    }
    for (std::size_t i = 0; i < bv.size(); ++i) {
      bv[i] = static_cast<float>(i + 1) * 0.001f;
    }
    std::unordered_map<std::size_t, std::vector<float>> conv_feeds;
    conv_feeds[cx] = xv;
    conv_feeds[cw] = wv;
    conv_feeds[cb] = bv;
    std::unordered_map<std::size_t, std::vector<float>> conv_values;
    const Status conv_exec_status =
        conv_exec_graph.executeF32(exec_options, conv_feeds, &conv_values, nullptr, nullptr);
    if (conv_exec_status != Status::kSuccess) {
      std::cerr << "conv_exec_graph executeF32 failed: "
                << lightning_core::runtime::getErrorString(conv_exec_status) << "\n";
      return 1;
    }
    const auto conv_out_it = conv_values.find(cy);
    if (conv_out_it == conv_values.end() || conv_out_it->second.size() != (1 * 16 * 8 * 8)) {
      std::cerr << "conv_exec_graph output missing or size mismatch\n";
      return 1;
    }
    std::vector<float> conv_expected(1 * 16 * 8 * 8, 0.0f);
    if (lightning_core::ops::conv2dNchw<float>(
            xv.data(),
            wv.data(),
            bv.data(),
            conv_expected.data(),
            1,
            3,
            8,
            8,
            16,
            3,
            3,
            1,
            1,
            1,
            1,
            Device::kMetal,
            false) != Status::kSuccess) {
      std::cerr << "direct conv2dNchw failed\n";
      return 1;
    }
    for (std::size_t i = 0; i < conv_expected.size(); ++i) {
      const float diff = conv_out_it->second[i] - conv_expected[i];
      if (diff > 1e-3f || diff < -1e-3f) {
        std::cerr << "conv_exec_graph output mismatch at index " << i << "\n";
        return 1;
      }
    }

    // chain-level A/B check: eager(conv->pack->attn) vs graph(conv)+graph(attn) path.
    const std::size_t chain_seq = 20;
    const std::size_t chain_dim = 20;
    const std::size_t chain_need = chain_seq * chain_dim;
    const std::size_t chain_total = chain_need * 3;

    std::vector<float> chain_qkv(chain_total, 0.0f);
    std::size_t copied = std::min(conv_expected.size(), chain_total);
    std::memcpy(chain_qkv.data(), conv_expected.data(), copied * sizeof(float));
    while (copied < chain_total) {
      const std::size_t chunk = std::min(copied, chain_total - copied);
      std::memcpy(chain_qkv.data() + copied, chain_qkv.data(), chunk * sizeof(float));
      copied += chunk;
    }

    std::vector<float> eager_chain_out(chain_need, 0.0f);
    lightning_core::AttentionConfig chain_cfg{chain_seq, chain_dim, false};
    if (lightning_core::attentionForward(
            chain_qkv.data(),
            chain_qkv.data() + chain_need,
            chain_qkv.data() + (2 * chain_need),
            eager_chain_out.data(),
            chain_cfg,
            Device::kMetal) != Status::kSuccess) {
      std::cerr << "eager chain attentionForward failed\n";
      return 1;
    }

    GraphIR chain_attn_graph;
    std::size_t chain_q_id = 0;
    std::size_t chain_k_id = 0;
    std::size_t chain_v_id = 0;
    std::size_t chain_out_id = 0;
    TensorSpec chain_spec;
    chain_spec.shape = {static_cast<std::int64_t>(chain_seq), static_cast<std::int64_t>(chain_dim)};
    chain_spec.dtype = DType::kFloat32;
    chain_spec.layout = lightning_core::Layout::kContiguous;
    if (chain_attn_graph.addTensorSpec(chain_spec, &chain_q_id, "chain_q", true) != Status::kSuccess ||
        chain_attn_graph.addTensorSpec(chain_spec, &chain_k_id, "chain_k", true) != Status::kSuccess ||
        chain_attn_graph.addTensorSpec(chain_spec, &chain_v_id, "chain_v", true) != Status::kSuccess ||
        chain_attn_graph.addTensorSpec(chain_spec, &chain_out_id, "chain_out", false) != Status::kSuccess) {
      std::cerr << "chain_attn_graph addTensorSpec failed\n";
      return 1;
    }
    if (chain_attn_graph.addNode(OpKind::kAttentionForward, {chain_q_id, chain_k_id, chain_v_id}, {chain_out_id}) !=
        Status::kSuccess) {
      std::cerr << "chain_attn_graph addNode failed\n";
      return 1;
    }
    std::unordered_map<std::size_t, std::vector<float>> chain_feeds;
    chain_feeds[chain_q_id] = std::vector<float>(chain_qkv.begin(), chain_qkv.begin() + chain_need);
    chain_feeds[chain_k_id] =
        std::vector<float>(chain_qkv.begin() + chain_need, chain_qkv.begin() + (2 * chain_need));
    chain_feeds[chain_v_id] =
        std::vector<float>(chain_qkv.begin() + (2 * chain_need), chain_qkv.begin() + (3 * chain_need));
    std::unordered_map<std::size_t, std::vector<float>> chain_values;
    if (chain_attn_graph.executeF32(exec_options, chain_feeds, &chain_values, nullptr, nullptr) != Status::kSuccess) {
      std::cerr << "chain_attn_graph executeF32 failed\n";
      return 1;
    }
    const auto chain_out_it = chain_values.find(chain_out_id);
    if (chain_out_it == chain_values.end() || chain_out_it->second.size() != chain_need) {
      std::cerr << "chain_attn_graph output missing or size mismatch\n";
      return 1;
    }
    for (std::size_t i = 0; i < chain_need; ++i) {
      const float diff = chain_out_it->second[i] - eager_chain_out[i];
      if (diff > 1e-3f || diff < -1e-3f) {
        std::cerr << "chain eager-vs-graph mismatch at index " << i << "\n";
        return 1;
      }
    }
  }

  GraphIR bad_graph;
  std::size_t t0 = 0;
  std::size_t t1 = 0;
  if (bad_graph.addTensorSpec(spec, &t0) != Status::kSuccess ||
      bad_graph.addTensorSpec(spec, &t1) != Status::kSuccess) {
    std::cerr << "bad_graph addTensorSpec failed\n";
    return 1;
  }
  if (bad_graph.addNode(OpKind::kVectorAdd, {t0}, {t1}) != Status::kInvalidValue) {
    std::cerr << "vector_add with one input should fail\n";
    return 1;
  }
  if (bad_graph.addNode(OpKind::kMatMul, {t0, t1}, {t1}, {0}) != Status::kInvalidValue) {
    std::cerr << "invalid control dep on first node should fail\n";
    return 1;
  }

  // backend-capability validation pass smoke: conv2d schema is metal-only.
  GraphIR conv_graph;
  std::size_t x = 0;
  std::size_t w = 0;
  std::size_t y = 0;
  TensorSpec conv_x;
  conv_x.shape = {1, 3, 8, 8};
  conv_x.dtype = DType::kFloat32;
  conv_x.layout = lightning_core::Layout::kContiguous;
  TensorSpec conv_w;
  conv_w.shape = {16, 3, 3, 3};
  conv_w.dtype = DType::kFloat32;
  conv_w.layout = lightning_core::Layout::kContiguous;
  TensorSpec conv_y;
  conv_y.shape = {1, 16, 8, 8};
  conv_y.dtype = DType::kFloat32;
  conv_y.layout = lightning_core::Layout::kContiguous;
  if (conv_graph.addTensorSpec(conv_x, &x) != Status::kSuccess ||
      conv_graph.addTensorSpec(conv_w, &w) != Status::kSuccess ||
      conv_graph.addTensorSpec(conv_y, &y) != Status::kSuccess) {
    std::cerr << "conv_graph tensor add failed\n";
    return 1;
  }
  if (conv_graph.addNode(OpKind::kConv2dNchw3x3s1p1, {x, w}, {y}) != Status::kSuccess) {
    std::cerr << "conv_graph addNode failed\n";
    return 1;
  }
  lightning_core::graph::ValidationReport conv_report;
  const Status conv_validate_status = conv_graph.validateWithReport(&conv_report);
  if (metal_compute_built) {
    if (conv_validate_status != Status::kSuccess || !conv_report.ok()) {
      std::cerr << "conv_graph should validate when metal backend is built\n";
      return 1;
    }
  } else {
    if (conv_validate_status != Status::kNotSupported || conv_report.ok()) {
      std::cerr << "conv_graph should fail backend capability pass when metal backend is unavailable\n";
      return 1;
    }
    bool saw_backend_capability_issue = false;
    for (const auto& issue : conv_report.issues) {
      if (issue.pass == lightning_core::graph::ValidationPass::kBackendCapability) {
        saw_backend_capability_issue = true;
        break;
      }
    }
    if (!saw_backend_capability_issue) {
      std::cerr << "conv_graph should report backend capability issue\n";
      return 1;
    }
  }

  GraphIR invalid_tensor_graph;
  std::size_t invalid_id = 0;
  TensorSpec invalid_spec;
  invalid_spec.shape = {};
  invalid_spec.dtype = DType::kFloat32;
  invalid_spec.layout = lightning_core::Layout::kContiguous;
  if (invalid_tensor_graph.addTensorSpec(invalid_spec, &invalid_id) != Status::kInvalidValue) {
    std::cerr << "empty-shape tensor spec should fail\n";
    return 1;
  }

  return 0;
}
