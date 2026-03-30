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
