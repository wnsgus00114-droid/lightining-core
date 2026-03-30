#include <iostream>
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
  const bool metal_compute_built =
      lightning_core::runtime::backendCapabilities(Device::kMetal).compute_surface;
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
