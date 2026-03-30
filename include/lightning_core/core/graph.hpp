#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "lightning_core/core/runtime.hpp"
#include "lightning_core/core/tensor.hpp"

namespace lightning_core::graph {

enum class DType {
  kFloat32 = 0,
  kFloat64
};

inline const char* dtypeName(DType dtype) {
  switch (dtype) {
    case DType::kFloat64:
      return "float64";
    case DType::kFloat32:
    default:
      return "float32";
  }
}

enum class OpKind {
  kMatMul = 0,
  kVectorAdd,
  kMatrixSub,
  kMatrixDiv,
  kAttentionForward,
  kConv2dNchw3x3s1p1
};

inline const char* opKindName(OpKind kind) {
  switch (kind) {
    case OpKind::kMatMul:
      return "matmul";
    case OpKind::kVectorAdd:
      return "vector_add";
    case OpKind::kMatrixSub:
      return "matrix_sub";
    case OpKind::kMatrixDiv:
      return "matrix_div";
    case OpKind::kAttentionForward:
      return "attention_forward";
    case OpKind::kConv2dNchw3x3s1p1:
      return "conv2d_nchw3x3s1p1";
    default:
      return "unknown";
  }
}

struct TensorSpec {
  std::vector<std::int64_t> shape;
  DType dtype{DType::kFloat32};
  Layout layout{Layout::kContiguous};
};

struct OperatorSchema {
  OpKind kind{OpKind::kMatMul};
  std::string name;
  std::size_t min_inputs{0};
  std::size_t max_inputs{0};
  std::size_t min_outputs{0};
  std::size_t max_outputs{0};
  bool supports_cpu{true};
  bool supports_cuda{false};
  bool supports_metal{false};

  bool supportsDevice(runtime::Device device) const {
    switch (device) {
      case runtime::Device::kCPU:
        return supports_cpu;
      case runtime::Device::kCUDA:
        return supports_cuda;
      case runtime::Device::kMetal:
      default:
        return supports_metal;
    }
  }
};

class OperatorRegistry {
 public:
  OperatorRegistry() {
#if defined(CJ_HAS_METAL) && CJ_HAS_METAL
    const bool metal_supported = true;
#else
    const bool metal_supported = false;
#endif
#if CJ_HAS_CUDA
    const bool cuda_supported = true;
#else
    const bool cuda_supported = false;
#endif

    (void)registerSchema(OperatorSchema{
        OpKind::kMatMul, "matmul", 2, 2, 1, 1, true, cuda_supported, metal_supported});
    (void)registerSchema(OperatorSchema{
        OpKind::kVectorAdd, "vector_add", 2, 2, 1, 1, true, cuda_supported, metal_supported});
    (void)registerSchema(OperatorSchema{
        OpKind::kMatrixSub, "matrix_sub", 2, 2, 1, 1, true, cuda_supported, metal_supported});
    (void)registerSchema(OperatorSchema{
        OpKind::kMatrixDiv, "matrix_div", 2, 2, 1, 1, true, cuda_supported, metal_supported});
    (void)registerSchema(OperatorSchema{
        OpKind::kAttentionForward, "attention_forward", 3, 3, 1, 1, true, false, metal_supported});
    (void)registerSchema(OperatorSchema{
        OpKind::kConv2dNchw3x3s1p1, "conv2d_nchw3x3s1p1", 2, 3, 1, 1, false, false, metal_supported});
  }

  runtime::Status registerSchema(const OperatorSchema& schema) {
    if (schema.name.empty() || schema.min_inputs > schema.max_inputs || schema.min_outputs > schema.max_outputs) {
      return runtime::Status::kInvalidValue;
    }
    schemas_[static_cast<int>(schema.kind)] = schema;
    return runtime::Status::kSuccess;
  }

  const OperatorSchema* find(OpKind kind) const {
    auto it = schemas_.find(static_cast<int>(kind));
    if (it == schemas_.end()) {
      return nullptr;
    }
    return &it->second;
  }

  std::vector<OperatorSchema> schemas() const {
    std::vector<OperatorSchema> out;
    out.reserve(schemas_.size());
    for (const auto& kv : schemas_) {
      out.push_back(kv.second);
    }
    return out;
  }

  std::size_t size() const {
    return schemas_.size();
  }

 private:
  std::unordered_map<int, OperatorSchema> schemas_;
};

inline OperatorRegistry& globalOperatorRegistry() {
  static OperatorRegistry registry;
  return registry;
}

struct GraphTensorValue {
  TensorSpec spec;
  std::string name;
  bool constant{false};
};

struct GraphNode {
  std::size_t id{0};
  OpKind op{OpKind::kMatMul};
  std::vector<std::size_t> inputs;
  std::vector<std::size_t> outputs;
  std::vector<std::size_t> control_deps;
  std::string name;
};

struct GraphPlanStep {
  std::size_t node_id{0};
  OpKind op{OpKind::kMatMul};
  runtime::Device assigned_device{runtime::Device::kCPU};
  bool fallback{false};
};

inline runtime::Status validateTensorSpec(const TensorSpec& spec) {
  if (!detail::isShapeValidContract(spec.shape)) {
    return runtime::Status::kInvalidValue;
  }
  if (spec.layout != Layout::kContiguous && spec.layout != Layout::kStrided) {
    return runtime::Status::kInvalidValue;
  }
  return runtime::Status::kSuccess;
}

class GraphIR {
 public:
  runtime::Status addTensorSpec(
      const TensorSpec& spec,
      std::size_t* out_id,
      const std::string& name = std::string(),
      bool constant = false) {
    if (out_id == nullptr) {
      return runtime::Status::kInvalidValue;
    }
    runtime::Status st = validateTensorSpec(spec);
    if (st != runtime::Status::kSuccess) {
      return st;
    }
    GraphTensorValue value;
    value.spec = spec;
    value.name = name;
    value.constant = constant;
    tensors_.push_back(std::move(value));
    *out_id = tensors_.size() - 1;
    return runtime::Status::kSuccess;
  }

  runtime::Status addNode(
      OpKind op,
      const std::vector<std::size_t>& inputs,
      const std::vector<std::size_t>& outputs,
      const std::vector<std::size_t>& control_deps = std::vector<std::size_t>(),
      const std::string& name = std::string()) {
    const OperatorSchema* schema = globalOperatorRegistry().find(op);
    if (schema == nullptr) {
      return runtime::Status::kNotSupported;
    }
    if (inputs.size() < schema->min_inputs || inputs.size() > schema->max_inputs ||
        outputs.size() < schema->min_outputs || outputs.size() > schema->max_outputs) {
      return runtime::Status::kInvalidValue;
    }
    for (std::size_t id : inputs) {
      if (id >= tensors_.size()) {
        return runtime::Status::kInvalidValue;
      }
    }
    for (std::size_t id : outputs) {
      if (id >= tensors_.size()) {
        return runtime::Status::kInvalidValue;
      }
    }
    const std::size_t next_id = nodes_.size();
    for (std::size_t dep : control_deps) {
      if (dep >= next_id) {
        return runtime::Status::kInvalidValue;
      }
    }
    GraphNode node;
    node.id = next_id;
    node.op = op;
    node.inputs = inputs;
    node.outputs = outputs;
    node.control_deps = control_deps;
    node.name = name;
    nodes_.push_back(std::move(node));
    return runtime::Status::kSuccess;
  }

  runtime::Status validate() const {
    for (const auto& t : tensors_) {
      runtime::Status st = validateTensorSpec(t.spec);
      if (st != runtime::Status::kSuccess) {
        return st;
      }
    }
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
      const GraphNode& node = nodes_[i];
      if (node.id != i) {
        return runtime::Status::kInvalidValue;
      }
      const OperatorSchema* schema = globalOperatorRegistry().find(node.op);
      if (schema == nullptr) {
        return runtime::Status::kNotSupported;
      }
      if (node.inputs.size() < schema->min_inputs || node.inputs.size() > schema->max_inputs ||
          node.outputs.size() < schema->min_outputs || node.outputs.size() > schema->max_outputs) {
        return runtime::Status::kInvalidValue;
      }
      for (std::size_t id : node.inputs) {
        if (id >= tensors_.size()) {
          return runtime::Status::kInvalidValue;
        }
      }
      for (std::size_t id : node.outputs) {
        if (id >= tensors_.size()) {
          return runtime::Status::kInvalidValue;
        }
      }
      for (std::size_t dep : node.control_deps) {
        if (dep >= node.id) {
          return runtime::Status::kInvalidValue;
        }
      }
    }
    return runtime::Status::kSuccess;
  }

  runtime::Status planForDevice(
      runtime::Device preferred_device,
      std::vector<GraphPlanStep>* out_steps) const {
    if (out_steps == nullptr) {
      return runtime::Status::kInvalidValue;
    }
    runtime::Status st = validate();
    if (st != runtime::Status::kSuccess) {
      return st;
    }

    out_steps->clear();
    out_steps->reserve(nodes_.size());

    for (const GraphNode& node : nodes_) {
      const OperatorSchema* schema = globalOperatorRegistry().find(node.op);
      if (schema == nullptr) {
        return runtime::Status::kNotSupported;
      }

      auto choose_if_supported = [&](runtime::Device device) -> bool {
        if (!schema->supportsDevice(device)) {
          return false;
        }
        const runtime::BackendCapabilities caps = runtime::backendCapabilities(device);
        return caps.compute_surface && caps.available;
      };

      runtime::Device assigned = preferred_device;
      if (!choose_if_supported(assigned)) {
        if (preferred_device != runtime::Device::kMetal && choose_if_supported(runtime::Device::kMetal)) {
          assigned = runtime::Device::kMetal;
        } else if (preferred_device != runtime::Device::kCUDA && choose_if_supported(runtime::Device::kCUDA)) {
          assigned = runtime::Device::kCUDA;
        } else if (preferred_device != runtime::Device::kCPU && choose_if_supported(runtime::Device::kCPU)) {
          assigned = runtime::Device::kCPU;
        } else if (choose_if_supported(preferred_device)) {
          assigned = preferred_device;
        } else {
          return runtime::Status::kNotSupported;
        }
      }

      GraphPlanStep step;
      step.node_id = node.id;
      step.op = node.op;
      step.assigned_device = assigned;
      step.fallback = (assigned != preferred_device);
      out_steps->push_back(step);
    }
    return runtime::Status::kSuccess;
  }

  const std::vector<GraphTensorValue>& tensors() const {
    return tensors_;
  }

  const std::vector<GraphNode>& nodes() const {
    return nodes_;
  }

  std::size_t numTensors() const {
    return tensors_.size();
  }

  std::size_t numNodes() const {
    return nodes_.size();
  }

 private:
  std::vector<GraphTensorValue> tensors_;
  std::vector<GraphNode> nodes_;
};

}  // namespace lightning_core::graph
