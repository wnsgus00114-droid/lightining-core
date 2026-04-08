#include <cmath>
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

  // Capability-aware planner grouping: align flexible ops to a forced backend
  // of neighboring ops when requested.
  const lightning_core::runtime::BackendCapabilities metal_caps_for_planner =
      lightning_core::runtime::backendCapabilities(Device::kMetal);
  const bool metal_exec_available_for_planner =
      metal_caps_for_planner.compute_surface && metal_caps_for_planner.available;
  if (metal_exec_available_for_planner) {
    GraphIR grouping_graph;
    std::size_t ga = 0;
    std::size_t gb = 0;
    std::size_t gmm = 0;
    std::size_t gcx = 0;
    std::size_t gcw = 0;
    std::size_t gcb = 0;
    std::size_t gcy = 0;

    TensorSpec mat_spec;
    mat_spec.shape = {2, 2};
    mat_spec.dtype = DType::kFloat32;
    mat_spec.layout = lightning_core::Layout::kContiguous;
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

    if (grouping_graph.addTensorSpec(mat_spec, &ga, "ga", true) != Status::kSuccess ||
        grouping_graph.addTensorSpec(mat_spec, &gb, "gb", true) != Status::kSuccess ||
        grouping_graph.addTensorSpec(mat_spec, &gmm, "gmm", false) != Status::kSuccess ||
        grouping_graph.addTensorSpec(cx_spec, &gcx, "gcx", true) != Status::kSuccess ||
        grouping_graph.addTensorSpec(cw_spec, &gcw, "gcw", true) != Status::kSuccess ||
        grouping_graph.addTensorSpec(cb_spec, &gcb, "gcb", true) != Status::kSuccess ||
        grouping_graph.addTensorSpec(cy_spec, &gcy, "gcy", false) != Status::kSuccess) {
      std::cerr << "grouping_graph addTensorSpec failed\n";
      return 1;
    }
    if (grouping_graph.addNode(OpKind::kMatMul, {ga, gb}, {gmm}) != Status::kSuccess ||
        grouping_graph.addNode(OpKind::kConv2dNchw3x3s1p1, {gcx, gcw, gcb}, {gcy}) != Status::kSuccess) {
      std::cerr << "grouping_graph addNode failed\n";
      return 1;
    }

    lightning_core::graph::GraphPlannerOptions no_cap_group;
    no_cap_group.preferred_device = Device::kCPU;
    no_cap_group.sync_policy.mode = lightning_core::runtime::SyncMode::kAuto;
    no_cap_group.sync_policy.trace_sync_boundary = false;
    no_cap_group.group_by_backend_capability = false;
    no_cap_group.separate_fallback_segments = true;
    no_cap_group.insert_sync_on_device_change = true;

    std::vector<lightning_core::graph::GraphExecutionGroup> groups_no_cap;
    std::vector<lightning_core::graph::GraphPlanStep> steps_no_cap;
    if (grouping_graph.planExecutionGroups(no_cap_group, &groups_no_cap, &steps_no_cap) != Status::kSuccess) {
      std::cerr << "grouping_graph planExecutionGroups(no_cap_group) failed\n";
      return 1;
    }
    if (steps_no_cap.size() != 2 ||
        steps_no_cap[0].assigned_device != Device::kCPU ||
        steps_no_cap[1].assigned_device != Device::kMetal) {
      std::cerr << "no_cap_group expected cpu->metal assignment\n";
      return 1;
    }
    if (groups_no_cap.size() != 2 ||
        !groups_no_cap[0].sync_boundary_after ||
        !groups_no_cap[1].sync_boundary_before) {
      std::cerr << "no_cap_group expected explicit sync boundary on device transition\n";
      return 1;
    }

    lightning_core::graph::GraphPlannerOptions cap_group = no_cap_group;
    cap_group.group_by_backend_capability = true;
    std::vector<lightning_core::graph::GraphExecutionGroup> groups_cap;
    std::vector<lightning_core::graph::GraphPlanStep> steps_cap;
    if (grouping_graph.planExecutionGroups(cap_group, &groups_cap, &steps_cap) != Status::kSuccess) {
      std::cerr << "grouping_graph planExecutionGroups(cap_group) failed\n";
      return 1;
    }
    if (steps_cap.size() != 2 ||
        steps_cap[0].assigned_device != Device::kMetal ||
        steps_cap[1].assigned_device != Device::kMetal) {
      std::cerr << "cap_group expected metal->metal capability-aligned assignment\n";
      return 1;
    }
    if (groups_cap.size() != 1) {
      std::cerr << "cap_group should collapse to single backend group\n";
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

  // shape contract regression guard: invalid feed numel must fail.
  {
    std::unordered_map<std::size_t, std::vector<float>> bad_exec_feeds = exec_feeds;
    if (bad_exec_feeds[exec_a].empty()) {
      std::cerr << "bad_exec_feeds precondition failed\n";
      return 1;
    }
    bad_exec_feeds[exec_a].pop_back();
    std::unordered_map<std::size_t, std::vector<float>> bad_exec_values;
    const Status bad_exec_status =
        exec_graph.executeF32(exec_options, bad_exec_feeds, &bad_exec_values, nullptr, nullptr);
    if (bad_exec_status != Status::kInvalidValue) {
      std::cerr << "executeF32 should reject invalid feed numel for shape contract\n";
      return 1;
    }
  }

  // lifetime contract regression guard: output map must not alias caller-owned feed buffers.
  {
    lightning_core::graph::GraphPlannerOptions lifetime_options = exec_options;
    lifetime_options.preferred_device = Device::kCPU;
    std::unordered_map<std::size_t, std::vector<float>> lifetime_feeds = exec_feeds;
    std::unordered_map<std::size_t, std::vector<float>> lifetime_values;
    if (exec_graph.executeF32(lifetime_options, lifetime_feeds, &lifetime_values, nullptr, nullptr) !=
        Status::kSuccess) {
      std::cerr << "executeF32(lifetime guard) failed\n";
      return 1;
    }
    lifetime_feeds[exec_a][0] = 999.0f;
    const auto lifetime_a_it = lifetime_values.find(exec_a);
    if (lifetime_a_it == lifetime_values.end()) {
      std::cerr << "lifetime guard missing copied input tensor value\n";
      return 1;
    }
    if (lifetime_a_it->second.empty() || lifetime_a_it->second[0] == 999.0f) {
      std::cerr << "executeF32 values should not alias feed lifetime\n";
      return 1;
    }
  }

  // graph-path sync policy coverage (auto/always/never) with runtime trace verification.
  const auto run_graph_sync_policy_case = [&](
      lightning_core::runtime::SyncMode mode,
      bool trace_boundary,
      bool expect_single_node_groups) -> bool {
    lightning_core::graph::GraphPlannerOptions case_options = exec_options;
    case_options.preferred_device = Device::kCPU;
    case_options.sync_policy.mode = mode;
    case_options.sync_policy.trace_sync_boundary = trace_boundary;
    case_options.separate_fallback_segments = true;
    case_options.insert_sync_on_device_change = true;

    std::vector<lightning_core::graph::GraphExecutionGroup> case_groups;
    std::vector<lightning_core::graph::GraphPlanStep> case_steps;
    if (exec_graph.planExecutionGroups(case_options, &case_groups, &case_steps) != Status::kSuccess) {
      std::cerr << "planExecutionGroups(sync case) failed\n";
      return false;
    }
    if (case_steps.size() != exec_graph.numNodes()) {
      std::cerr << "sync case step count mismatch\n";
      return false;
    }
    if (expect_single_node_groups && case_groups.size() != exec_graph.numNodes()) {
      std::cerr << "sync_mode=always should split into single-node groups\n";
      return false;
    }
    if (mode == lightning_core::runtime::SyncMode::kNever) {
      for (const auto& group : case_groups) {
        if (group.sync_boundary_before || group.sync_boundary_after) {
          std::cerr << "sync_mode=never should clear all sync boundaries\n";
          return false;
        }
      }
    }

    std::size_t expected_apply_sync_calls = 0;
    for (const auto& group : case_groups) {
      if (group.sync_boundary_before) {
        ++expected_apply_sync_calls;
      }
      if (group.sync_boundary_after) {
        ++expected_apply_sync_calls;
      }
    }

    std::unordered_map<std::size_t, std::vector<float>> case_values;
    lightning_core::runtime::clearRuntimeTraceEvents();
    lightning_core::runtime::setRuntimeTraceEnabled(true);
    const Status exec_status = exec_graph.executeF32(case_options, exec_feeds, &case_values, nullptr, nullptr);
    lightning_core::runtime::setRuntimeTraceEnabled(false);
    if (exec_status != Status::kSuccess) {
      std::cerr << "executeF32(sync case) failed\n";
      lightning_core::runtime::clearRuntimeTraceEvents();
      return false;
    }

    std::size_t apply_sync_events = 0;
    for (const auto& ev : lightning_core::runtime::runtimeTraceEvents()) {
      if (ev.type != lightning_core::runtime::RuntimeTraceEventType::kApplySyncPolicy) {
        continue;
      }
      ++apply_sync_events;
      if (ev.detail0 != static_cast<int>(mode)) {
        std::cerr << "apply_sync_policy trace mode mismatch in graph path\n";
        lightning_core::runtime::clearRuntimeTraceEvents();
        return false;
      }
      if (ev.detail1 != (trace_boundary ? 1 : 0)) {
        std::cerr << "apply_sync_policy trace boundary flag mismatch in graph path\n";
        lightning_core::runtime::clearRuntimeTraceEvents();
        return false;
      }
    }
    lightning_core::runtime::clearRuntimeTraceEvents();

    if (apply_sync_events != expected_apply_sync_calls) {
      std::cerr << "graph sync policy apply event count mismatch: expected="
                << expected_apply_sync_calls << " got=" << apply_sync_events << "\n";
      return false;
    }
    return true;
  };

  if (!run_graph_sync_policy_case(lightning_core::runtime::SyncMode::kAuto, false, false)) {
    return 1;
  }
  if (!run_graph_sync_policy_case(lightning_core::runtime::SyncMode::kAuto, true, false)) {
    return 1;
  }
  if (!run_graph_sync_policy_case(lightning_core::runtime::SyncMode::kAlways, true, true)) {
    return 1;
  }
  if (!run_graph_sync_policy_case(lightning_core::runtime::SyncMode::kNever, true, false)) {
    return 1;
  }

  // graph parity baseline: explicit CPU planning path must match expected output.
  lightning_core::graph::GraphPlannerOptions cpu_exec_options = exec_options;
  cpu_exec_options.preferred_device = Device::kCPU;
  std::unordered_map<std::size_t, std::vector<float>> exec_values_cpu;
  if (exec_graph.executeF32(cpu_exec_options, exec_feeds, &exec_values_cpu, nullptr, nullptr) != Status::kSuccess) {
    std::cerr << "executeF32(CPU planned) should succeed for matmul->vector_add graph\n";
    return 1;
  }
  const auto cpu_out_it = exec_values_cpu.find(exec_out);
  if (cpu_out_it == exec_values_cpu.end() || cpu_out_it->second.size() != expected_out.size()) {
    std::cerr << "executeF32(CPU planned) output missing or size mismatch\n";
    return 1;
  }
  for (std::size_t i = 0; i < expected_out.size(); ++i) {
    const float diff = cpu_out_it->second[i] - expected_out[i];
    if (diff > 1e-4f || diff < -1e-4f) {
      std::cerr << "executeF32(CPU planned) output mismatch at index " << i << "\n";
      return 1;
    }
  }

  const lightning_core::runtime::BackendCapabilities metal_caps =
      lightning_core::runtime::backendCapabilities(Device::kMetal);
  const bool metal_compute_built = metal_caps.compute_surface;
  const bool metal_exec_available = metal_caps.compute_surface && metal_caps.available;
  if (metal_exec_available) {
    // graph parity: preferred-metal plan vs preferred-cpu plan should match numerically.
    if (out_it->second.size() != cpu_out_it->second.size()) {
      std::cerr << "graph parity size mismatch between metal/cpu planned runs\n";
      return 1;
    }
    for (std::size_t i = 0; i < out_it->second.size(); ++i) {
      const float diff = out_it->second[i] - cpu_out_it->second[i];
      if (diff > 1e-3f || diff < -1e-3f) {
        std::cerr << "graph parity mismatch (metal vs cpu planned) at index " << i << "\n";
        return 1;
      }
    }
  }
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

    // fallback/device-change boundary contract:
    // preferred CPU + metal-only conv should fallback to Metal, while matmul stays on CPU.
    GraphIR mixed_graph;
    std::size_t mx = 0;
    std::size_t mw = 0;
    std::size_t mb = 0;
    std::size_t my = 0;
    std::size_t ma = 0;
    std::size_t mb2 = 0;
    std::size_t mm = 0;

    TensorSpec mx_spec;
    mx_spec.shape = {1, 3, 8, 8};
    mx_spec.dtype = DType::kFloat32;
    mx_spec.layout = lightning_core::Layout::kContiguous;
    TensorSpec mw_spec;
    mw_spec.shape = {16, 3, 3, 3};
    mw_spec.dtype = DType::kFloat32;
    mw_spec.layout = lightning_core::Layout::kContiguous;
    TensorSpec mb_spec;
    mb_spec.shape = {16};
    mb_spec.dtype = DType::kFloat32;
    mb_spec.layout = lightning_core::Layout::kContiguous;
    TensorSpec my_spec;
    my_spec.shape = {1, 16, 8, 8};
    my_spec.dtype = DType::kFloat32;
    my_spec.layout = lightning_core::Layout::kContiguous;
    TensorSpec mm_spec;
    mm_spec.shape = {2, 2};
    mm_spec.dtype = DType::kFloat32;
    mm_spec.layout = lightning_core::Layout::kContiguous;

    if (mixed_graph.addTensorSpec(mx_spec, &mx, "mx", true) != Status::kSuccess ||
        mixed_graph.addTensorSpec(mw_spec, &mw, "mw", true) != Status::kSuccess ||
        mixed_graph.addTensorSpec(mb_spec, &mb, "mb", true) != Status::kSuccess ||
        mixed_graph.addTensorSpec(my_spec, &my, "my", false) != Status::kSuccess ||
        mixed_graph.addTensorSpec(mm_spec, &ma, "ma", true) != Status::kSuccess ||
        mixed_graph.addTensorSpec(mm_spec, &mb2, "mb2", true) != Status::kSuccess ||
        mixed_graph.addTensorSpec(mm_spec, &mm, "mm", false) != Status::kSuccess) {
      std::cerr << "mixed_graph addTensorSpec failed\n";
      return 1;
    }
    if (mixed_graph.addNode(OpKind::kConv2dNchw3x3s1p1, {mx, mw, mb}, {my}) != Status::kSuccess ||
        mixed_graph.addNode(OpKind::kMatMul, {ma, mb2}, {mm}) != Status::kSuccess) {
      std::cerr << "mixed_graph addNode failed\n";
      return 1;
    }

    lightning_core::graph::GraphPlannerOptions mixed_options;
    mixed_options.preferred_device = Device::kCPU;
    mixed_options.sync_policy.mode = lightning_core::runtime::SyncMode::kAuto;
    mixed_options.sync_policy.trace_sync_boundary = true;
    mixed_options.separate_fallback_segments = true;
    mixed_options.insert_sync_on_device_change = true;

    std::vector<lightning_core::graph::GraphExecutionGroup> mixed_groups;
    std::vector<lightning_core::graph::GraphPlanStep> mixed_steps;
    if (mixed_graph.planExecutionGroups(mixed_options, &mixed_groups, &mixed_steps) != Status::kSuccess) {
      std::cerr << "mixed_graph planExecutionGroups failed\n";
      return 1;
    }
    if (mixed_steps.size() != 2 || mixed_steps[0].node_id != 0 || mixed_steps[1].node_id != 1) {
      std::cerr << "mixed_graph step metadata mismatch\n";
      return 1;
    }
    if (mixed_steps[0].assigned_device != Device::kMetal || !mixed_steps[0].fallback) {
      std::cerr << "mixed_graph conv step should fallback to Metal from preferred CPU\n";
      return 1;
    }
    if (mixed_steps[1].assigned_device != Device::kCPU || mixed_steps[1].fallback) {
      std::cerr << "mixed_graph matmul step should stay on CPU without fallback\n";
      return 1;
    }
    if (mixed_groups.size() < 2) {
      std::cerr << "mixed_graph should split groups on device-change boundary\n";
      return 1;
    }
    if (!mixed_groups[0].sync_boundary_after || !mixed_groups[1].sync_boundary_before) {
      std::cerr << "mixed_graph should mark sync boundaries on device-change/fallback boundary\n";
      return 1;
    }

    std::unordered_map<std::size_t, std::vector<float>> mixed_feeds;
    mixed_feeds[mx] = xv;
    mixed_feeds[mw] = wv;
    mixed_feeds[mb] = bv;
    mixed_feeds[ma] = {1.0f, 2.0f, 3.0f, 4.0f};
    mixed_feeds[mb2] = {5.0f, 6.0f, 7.0f, 8.0f};
    std::unordered_map<std::size_t, std::vector<float>> mixed_values;
    lightning_core::runtime::clearRuntimeTraceEvents();
    lightning_core::runtime::setRuntimeTraceEnabled(true);
    if (mixed_graph.executeF32(mixed_options, mixed_feeds, &mixed_values, nullptr, nullptr) != Status::kSuccess) {
      std::cerr << "mixed_graph executeF32 failed\n";
      lightning_core::runtime::setRuntimeTraceEnabled(false);
      lightning_core::runtime::clearRuntimeTraceEvents();
      return 1;
    }
    lightning_core::runtime::setRuntimeTraceEnabled(false);

    if (mixed_values.find(my) == mixed_values.end() || mixed_values.find(mm) == mixed_values.end()) {
      std::cerr << "mixed_graph output tensors missing\n";
      lightning_core::runtime::clearRuntimeTraceEvents();
      return 1;
    }

    bool saw_conv_cpu_to_metal_fallback = false;
    bool saw_matmul_cpu_direct = false;
    std::size_t apply_sync_count = 0;
    std::size_t expected_apply_sync_count = 0;
    for (const auto& group : mixed_groups) {
      if (group.sync_boundary_before) {
        ++expected_apply_sync_count;
      }
      if (group.sync_boundary_after) {
        ++expected_apply_sync_count;
      }
    }
    for (const auto& ev : lightning_core::runtime::runtimeTraceEvents()) {
      if (ev.type == lightning_core::runtime::RuntimeTraceEventType::kApplySyncPolicy) {
        ++apply_sync_count;
      }
      if (ev.type != lightning_core::runtime::RuntimeTraceEventType::kOpDispatch) {
        continue;
      }
      lightning_core::runtime::Device requested = lightning_core::runtime::Device::kCPU;
      lightning_core::runtime::Device selected = lightning_core::runtime::Device::kCPU;
      bool fallback = false;
      if (!lightning_core::runtime::decodeRuntimeTraceDispatchDetail(
              ev.detail1, &requested, &selected, &fallback)) {
        continue;
      }
      const auto op_kind = static_cast<lightning_core::runtime::RuntimeTraceOpKind>(ev.detail0);
      if (op_kind == lightning_core::runtime::RuntimeTraceOpKind::kConv2dNchw &&
          requested == Device::kCPU &&
          selected == Device::kMetal &&
          fallback) {
        saw_conv_cpu_to_metal_fallback = true;
      }
      if (op_kind == lightning_core::runtime::RuntimeTraceOpKind::kMatMul &&
          requested == Device::kCPU &&
          selected == Device::kCPU &&
          !fallback) {
        saw_matmul_cpu_direct = true;
      }
    }
    lightning_core::runtime::clearRuntimeTraceEvents();

    if (!saw_conv_cpu_to_metal_fallback) {
      std::cerr << "mixed_graph trace missing conv CPU->Metal fallback dispatch metadata\n";
      return 1;
    }
    if (!saw_matmul_cpu_direct) {
      std::cerr << "mixed_graph trace missing matmul CPU direct dispatch metadata\n";
      return 1;
    }
    if (apply_sync_count != expected_apply_sync_count) {
      std::cerr << "mixed_graph sync boundary trace count mismatch\n";
      return 1;
    }

    // fusion pilot v1: conv+relu should fuse on eligible graph and preserve output.
    GraphIR fusion_graph;
    std::size_t fx = 0;
    std::size_t fw = 0;
    std::size_t fb = 0;
    std::size_t fmid = 0;
    std::size_t fout = 0;
    TensorSpec fx_spec;
    fx_spec.shape = {1, 3, 8, 8};
    fx_spec.dtype = DType::kFloat32;
    fx_spec.layout = lightning_core::Layout::kContiguous;
    TensorSpec fw_spec;
    fw_spec.shape = {16, 3, 3, 3};
    fw_spec.dtype = DType::kFloat32;
    fw_spec.layout = lightning_core::Layout::kContiguous;
    TensorSpec fb_spec;
    fb_spec.shape = {16};
    fb_spec.dtype = DType::kFloat32;
    fb_spec.layout = lightning_core::Layout::kContiguous;
    TensorSpec fy_spec;
    fy_spec.shape = {1, 16, 8, 8};
    fy_spec.dtype = DType::kFloat32;
    fy_spec.layout = lightning_core::Layout::kContiguous;
    if (fusion_graph.addTensorSpec(fx_spec, &fx, "fx", true) != Status::kSuccess ||
        fusion_graph.addTensorSpec(fw_spec, &fw, "fw", true) != Status::kSuccess ||
        fusion_graph.addTensorSpec(fb_spec, &fb, "fb", true) != Status::kSuccess ||
        fusion_graph.addTensorSpec(fy_spec, &fmid, "fmid", false) != Status::kSuccess ||
        fusion_graph.addTensorSpec(fy_spec, &fout, "fout", false) != Status::kSuccess) {
      std::cerr << "fusion_graph addTensorSpec failed\n";
      return 1;
    }
    if (fusion_graph.addNode(OpKind::kConv2dNchw3x3s1p1, {fx, fw, fb}, {fmid}) != Status::kSuccess ||
        fusion_graph.addNode(OpKind::kRelu, {fmid}, {fout}) != Status::kSuccess) {
      std::cerr << "fusion_graph addNode failed\n";
      return 1;
    }

    std::unordered_map<std::size_t, std::vector<float>> fusion_feeds;
    fusion_feeds[fx] = xv;
    fusion_feeds[fw] = wv;
    fusion_feeds[fb] = bv;
    std::unordered_map<std::size_t, std::vector<float>> fusion_values_on;
    std::unordered_map<std::size_t, std::vector<float>> fusion_values_off;
    lightning_core::graph::GraphPlannerOptions fusion_opts_on = exec_options;
    fusion_opts_on.enable_fusion_v1 = true;
    lightning_core::graph::GraphPlannerOptions fusion_opts_off = exec_options;
    fusion_opts_off.enable_fusion_v1 = false;
    if (fusion_graph.executeF32(fusion_opts_on, fusion_feeds, &fusion_values_on, nullptr, nullptr) != Status::kSuccess) {
      std::cerr << "fusion_graph executeF32(enable_fusion_v1=true) failed\n";
      return 1;
    }
    if (fusion_graph.executeF32(fusion_opts_off, fusion_feeds, &fusion_values_off, nullptr, nullptr) != Status::kSuccess) {
      std::cerr << "fusion_graph executeF32(enable_fusion_v1=false) failed\n";
      return 1;
    }
    const auto fusion_out_on_it = fusion_values_on.find(fout);
    const auto fusion_out_off_it = fusion_values_off.find(fout);
    if (fusion_out_on_it == fusion_values_on.end() || fusion_out_off_it == fusion_values_off.end()) {
      std::cerr << "fusion_graph output missing\n";
      return 1;
    }
    if (fusion_out_on_it->second.size() != fusion_out_off_it->second.size()) {
      std::cerr << "fusion_graph output size mismatch\n";
      return 1;
    }
    for (std::size_t i = 0; i < fusion_out_on_it->second.size(); ++i) {
      const float diff = fusion_out_on_it->second[i] - fusion_out_off_it->second[i];
      if (diff > 1e-3f || diff < -1e-3f) {
        std::cerr << "fusion_graph fused/unfused mismatch at index " << i << "\n";
        return 1;
      }
    }

    std::vector<lightning_core::graph::GraphFusionDecision> fusion_decisions_on;
    if (fusion_graph.fusionReport(fusion_opts_on, &fusion_decisions_on) != Status::kSuccess) {
      std::cerr << "fusion_graph fusionReport(enable=true) failed\n";
      return 1;
    }
    bool saw_fused_conv_relu = false;
    for (const auto& d : fusion_decisions_on) {
      if (d.pattern == lightning_core::graph::FusionPattern::kConvReluV1 && d.fused) {
        saw_fused_conv_relu = true;
        break;
      }
    }
    if (!saw_fused_conv_relu) {
      std::cerr << "fusion_graph should report fused conv+relu decision\n";
      return 1;
    }

    std::vector<lightning_core::graph::GraphFusionDecision> fusion_decisions_off;
    if (fusion_graph.fusionReport(fusion_opts_off, &fusion_decisions_off) != Status::kSuccess) {
      std::cerr << "fusion_graph fusionReport(enable=false) failed\n";
      return 1;
    }
    bool saw_disabled_reason = false;
    for (const auto& d : fusion_decisions_off) {
      if (d.pattern == lightning_core::graph::FusionPattern::kConvReluV1 &&
          !d.fused &&
          d.reason == "fusion_disabled") {
        saw_disabled_reason = true;
        break;
      }
    }
    if (!saw_disabled_reason) {
      std::cerr << "fusion_graph should report fusion_disabled reason when v1 fusion is disabled\n";
      return 1;
    }

    // fallback explanation report: non-eligible conv+relu pair should include reason.
    GraphIR no_fuse_graph;
    std::size_t nfx = 0;
    std::size_t nfw = 0;
    std::size_t nfb = 0;
    std::size_t nfmid = 0;
    std::size_t nfout = 0;
    std::size_t nfadd = 0;
    std::size_t nfadd_out = 0;
    if (no_fuse_graph.addTensorSpec(fx_spec, &nfx, "nfx", true) != Status::kSuccess ||
        no_fuse_graph.addTensorSpec(fw_spec, &nfw, "nfw", true) != Status::kSuccess ||
        no_fuse_graph.addTensorSpec(fb_spec, &nfb, "nfb", true) != Status::kSuccess ||
        no_fuse_graph.addTensorSpec(fy_spec, &nfmid, "nfmid", false) != Status::kSuccess ||
        no_fuse_graph.addTensorSpec(fy_spec, &nfout, "nfout", false) != Status::kSuccess ||
        no_fuse_graph.addTensorSpec(fy_spec, &nfadd, "nfadd", true) != Status::kSuccess ||
        no_fuse_graph.addTensorSpec(fy_spec, &nfadd_out, "nfadd_out", false) != Status::kSuccess) {
      std::cerr << "no_fuse_graph addTensorSpec failed\n";
      return 1;
    }
    if (no_fuse_graph.addNode(OpKind::kConv2dNchw3x3s1p1, {nfx, nfw, nfb}, {nfmid}) != Status::kSuccess ||
        no_fuse_graph.addNode(OpKind::kRelu, {nfmid}, {nfout}) != Status::kSuccess ||
        no_fuse_graph.addNode(OpKind::kVectorAdd, {nfmid, nfadd}, {nfadd_out}) != Status::kSuccess) {
      std::cerr << "no_fuse_graph addNode failed\n";
      return 1;
    }
    std::vector<lightning_core::graph::GraphFusionDecision> no_fuse_decisions;
    if (no_fuse_graph.fusionReport(fusion_opts_on, &no_fuse_decisions) != Status::kSuccess) {
      std::cerr << "no_fuse_graph fusionReport failed\n";
      return 1;
    }
    bool saw_multi_consumer_reason = false;
    for (const auto& d : no_fuse_decisions) {
      if (d.pattern == lightning_core::graph::FusionPattern::kConvReluV1 &&
          !d.fused &&
          d.reason == "intermediate_multi_consumer") {
        saw_multi_consumer_reason = true;
        break;
      }
    }
    if (!saw_multi_consumer_reason) {
      std::cerr << "no_fuse_graph should report intermediate_multi_consumer reason\n";
      return 1;
    }
  }

  // fusion pilot v2: matmul+bias+relu should fuse on eligible graph and
  // preserve outputs, with explicit cost-model gating reason.
  {
    GraphIR mm_fusion_graph;
    std::size_t mma = 0;
    std::size_t mmb = 0;
    std::size_t mmbias = 0;
    std::size_t mm_mid = 0;
    std::size_t mm_add = 0;
    std::size_t mm_out = 0;

    TensorSpec a_spec;
    a_spec.shape = {4, 8};
    a_spec.dtype = DType::kFloat32;
    a_spec.layout = lightning_core::Layout::kContiguous;
    TensorSpec b_spec;
    b_spec.shape = {8, 6};
    b_spec.dtype = DType::kFloat32;
    b_spec.layout = lightning_core::Layout::kContiguous;
    TensorSpec out_spec;
    out_spec.shape = {4, 6};
    out_spec.dtype = DType::kFloat32;
    out_spec.layout = lightning_core::Layout::kContiguous;

    if (mm_fusion_graph.addTensorSpec(a_spec, &mma, "mma", true) != Status::kSuccess ||
        mm_fusion_graph.addTensorSpec(b_spec, &mmb, "mmb", true) != Status::kSuccess ||
        mm_fusion_graph.addTensorSpec(out_spec, &mmbias, "mmbias", true) != Status::kSuccess ||
        mm_fusion_graph.addTensorSpec(out_spec, &mm_mid, "mm_mid", false) != Status::kSuccess ||
        mm_fusion_graph.addTensorSpec(out_spec, &mm_add, "mm_add", false) != Status::kSuccess ||
        mm_fusion_graph.addTensorSpec(out_spec, &mm_out, "mm_out", false) != Status::kSuccess) {
      std::cerr << "mm_fusion_graph addTensorSpec failed\n";
      return 1;
    }
    if (mm_fusion_graph.addNode(OpKind::kMatMul, {mma, mmb}, {mm_mid}) != Status::kSuccess ||
        mm_fusion_graph.addNode(OpKind::kVectorAdd, {mm_mid, mmbias}, {mm_add}) != Status::kSuccess ||
        mm_fusion_graph.addNode(OpKind::kRelu, {mm_add}, {mm_out}) != Status::kSuccess) {
      std::cerr << "mm_fusion_graph addNode failed\n";
      return 1;
    }

    std::unordered_map<std::size_t, std::vector<float>> mm_feeds;
    mm_feeds[mma].resize(4 * 8);
    mm_feeds[mmb].resize(8 * 6);
    mm_feeds[mmbias].resize(4 * 6);
    for (std::size_t i = 0; i < mm_feeds[mma].size(); ++i) {
      mm_feeds[mma][i] = static_cast<float>((i % 9) - 4) * 0.2f;
    }
    for (std::size_t i = 0; i < mm_feeds[mmb].size(); ++i) {
      mm_feeds[mmb][i] = static_cast<float>((i % 7) - 3) * 0.15f;
    }
    for (std::size_t i = 0; i < mm_feeds[mmbias].size(); ++i) {
      mm_feeds[mmbias][i] = static_cast<float>((i % 5) - 2) * 0.1f;
    }

    lightning_core::graph::GraphPlannerOptions mm_opts_on;
    mm_opts_on.preferred_device = Device::kCPU;
    mm_opts_on.sync_policy.mode = lightning_core::runtime::SyncMode::kAuto;
    mm_opts_on.enable_fusion_v1 = true;
    mm_opts_on.enable_fusion_cost_model_v1 = true;
    mm_opts_on.fusion_cost_min_speedup = 1.0;

    lightning_core::graph::GraphPlannerOptions mm_opts_off = mm_opts_on;
    mm_opts_off.enable_fusion_v1 = false;

    std::unordered_map<std::size_t, std::vector<float>> mm_values_on;
    std::unordered_map<std::size_t, std::vector<float>> mm_values_off;
    if (mm_fusion_graph.executeF32(mm_opts_on, mm_feeds, &mm_values_on, nullptr, nullptr) != Status::kSuccess) {
      std::cerr << "mm_fusion_graph executeF32(enable=true) failed\n";
      return 1;
    }
    if (mm_fusion_graph.executeF32(mm_opts_off, mm_feeds, &mm_values_off, nullptr, nullptr) != Status::kSuccess) {
      std::cerr << "mm_fusion_graph executeF32(enable=false) failed\n";
      return 1;
    }
    const auto mm_out_on_it = mm_values_on.find(mm_out);
    const auto mm_out_off_it = mm_values_off.find(mm_out);
    if (mm_out_on_it == mm_values_on.end() || mm_out_off_it == mm_values_off.end()) {
      std::cerr << "mm_fusion_graph output missing\n";
      return 1;
    }
    if (mm_out_on_it->second.size() != mm_out_off_it->second.size()) {
      std::cerr << "mm_fusion_graph output size mismatch\n";
      return 1;
    }
    for (std::size_t i = 0; i < mm_out_on_it->second.size(); ++i) {
      const float diff = mm_out_on_it->second[i] - mm_out_off_it->second[i];
      if (diff > 1e-4f || diff < -1e-4f) {
        std::cerr << "mm_fusion_graph fused/unfused mismatch at index " << i << "\n";
        return 1;
      }
    }

    std::vector<lightning_core::graph::GraphFusionDecision> mm_decisions_on;
    if (mm_fusion_graph.fusionReport(mm_opts_on, &mm_decisions_on) != Status::kSuccess) {
      std::cerr << "mm_fusion_graph fusionReport(enable=true) failed\n";
      return 1;
    }
    bool saw_mm_fused = false;
    for (const auto& d : mm_decisions_on) {
      if (d.pattern == lightning_core::graph::FusionPattern::kMatMulBiasReluV1 && d.fused) {
        saw_mm_fused = true;
        if (!std::isfinite(d.estimated_speedup) || d.estimated_speedup <= 0.0) {
          std::cerr << "mm_fusion_graph should expose finite cost-model speedup\n";
          return 1;
        }
        break;
      }
    }
    if (!saw_mm_fused) {
      std::cerr << "mm_fusion_graph should report fused matmul+bias+relu decision\n";
      return 1;
    }

    lightning_core::graph::GraphPlannerOptions mm_opts_reject = mm_opts_on;
    mm_opts_reject.fusion_cost_min_speedup = 1000.0;
    std::vector<lightning_core::graph::GraphFusionDecision> mm_decisions_reject;
    if (mm_fusion_graph.fusionReport(mm_opts_reject, &mm_decisions_reject) != Status::kSuccess) {
      std::cerr << "mm_fusion_graph fusionReport(cost reject) failed\n";
      return 1;
    }
    bool saw_cost_reject = false;
    for (const auto& d : mm_decisions_reject) {
      if (d.pattern == lightning_core::graph::FusionPattern::kMatMulBiasReluV1 &&
          !d.fused &&
          d.reason.rfind("cost_model_reject(", 0) == 0) {
        saw_cost_reject = true;
        break;
      }
    }
    if (!saw_cost_reject) {
      std::cerr << "mm_fusion_graph should report cost_model_reject reason under high threshold\n";
      return 1;
    }

    GraphIR mm_no_fuse_graph;
    std::size_t nfa = 0;
    std::size_t nfb = 0;
    std::size_t nfbias = 0;
    std::size_t nfmid = 0;
    std::size_t nfadd = 0;
    std::size_t nfout = 0;
    std::size_t nfextra = 0;
    if (mm_no_fuse_graph.addTensorSpec(a_spec, &nfa, "nfa", true) != Status::kSuccess ||
        mm_no_fuse_graph.addTensorSpec(b_spec, &nfb, "nfb", true) != Status::kSuccess ||
        mm_no_fuse_graph.addTensorSpec(out_spec, &nfbias, "nfbias", true) != Status::kSuccess ||
        mm_no_fuse_graph.addTensorSpec(out_spec, &nfmid, "nfmid", false) != Status::kSuccess ||
        mm_no_fuse_graph.addTensorSpec(out_spec, &nfadd, "nfadd", false) != Status::kSuccess ||
        mm_no_fuse_graph.addTensorSpec(out_spec, &nfout, "nfout", false) != Status::kSuccess ||
        mm_no_fuse_graph.addTensorSpec(out_spec, &nfextra, "nfextra", false) != Status::kSuccess) {
      std::cerr << "mm_no_fuse_graph addTensorSpec failed\n";
      return 1;
    }
    if (mm_no_fuse_graph.addNode(OpKind::kMatMul, {nfa, nfb}, {nfmid}) != Status::kSuccess ||
        mm_no_fuse_graph.addNode(OpKind::kVectorAdd, {nfmid, nfbias}, {nfadd}) != Status::kSuccess ||
        mm_no_fuse_graph.addNode(OpKind::kRelu, {nfadd}, {nfout}) != Status::kSuccess ||
        mm_no_fuse_graph.addNode(OpKind::kMatrixSub, {nfmid, nfbias}, {nfextra}) != Status::kSuccess) {
      std::cerr << "mm_no_fuse_graph addNode failed\n";
      return 1;
    }

    std::vector<lightning_core::graph::GraphFusionDecision> mm_no_fuse_decisions;
    if (mm_no_fuse_graph.fusionReport(mm_opts_on, &mm_no_fuse_decisions) != Status::kSuccess) {
      std::cerr << "mm_no_fuse_graph fusionReport failed\n";
      return 1;
    }
    bool saw_mm_multi_consumer_reason = false;
    for (const auto& d : mm_no_fuse_decisions) {
      if (d.pattern == lightning_core::graph::FusionPattern::kMatMulBiasReluV1 &&
          !d.fused &&
          d.reason == "matmul_output_multi_consumer") {
        saw_mm_multi_consumer_reason = true;
        break;
      }
    }
    if (!saw_mm_multi_consumer_reason) {
      std::cerr << "mm_no_fuse_graph should report matmul_output_multi_consumer reason\n";
      return 1;
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
  TensorSpec invalid_layout_spec;
  invalid_layout_spec.shape = {2, 2};
  invalid_layout_spec.dtype = DType::kFloat32;
  invalid_layout_spec.layout = static_cast<lightning_core::Layout>(99);
  if (invalid_tensor_graph.addTensorSpec(invalid_layout_spec, &invalid_id) != Status::kInvalidValue) {
    std::cerr << "invalid layout tensor spec should fail\n";
    return 1;
  }

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
