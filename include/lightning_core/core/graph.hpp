#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "lightning_core/core/attention.hpp"
#include "lightning_core/core/ops.hpp"
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

enum class ValidationPass {
  kTensorSpec = 0,
  kSchemaArity,
  kTensorReference,
  kControlDependency,
  kBackendCapability
};

inline const char* validationPassName(ValidationPass pass) {
  switch (pass) {
    case ValidationPass::kTensorSpec:
      return "tensor_spec";
    case ValidationPass::kSchemaArity:
      return "schema_arity";
    case ValidationPass::kTensorReference:
      return "tensor_reference";
    case ValidationPass::kControlDependency:
      return "control_dependency";
    case ValidationPass::kBackendCapability:
      return "backend_capability";
    default:
      return "unknown";
  }
}

struct ValidationIssue {
  ValidationPass pass{ValidationPass::kTensorSpec};
  runtime::Status status{runtime::Status::kInvalidValue};
  std::int64_t node_id{-1};
  std::int64_t tensor_id{-1};
  std::string message;
};

struct ValidationReport {
  std::vector<ValidationIssue> issues;

  bool ok() const {
    return issues.empty();
  }

  void clear() {
    issues.clear();
  }
};

struct GraphPlannerOptions {
  runtime::Device preferred_device{runtime::Device::kMetal};
  runtime::SyncPolicy sync_policy{};
  // When true, planner may pick a non-preferred but supported backend to keep
  // neighboring nodes in the same backend group and reduce host/sync churn.
  bool group_by_backend_capability{true};
  bool separate_fallback_segments{true};
  bool insert_sync_on_device_change{true};
};

struct GraphExecutionGroup {
  std::size_t group_id{0};
  runtime::Device assigned_device{runtime::Device::kCPU};
  bool fallback{false};
  bool sync_boundary_before{false};
  bool sync_boundary_after{false};
  std::vector<std::size_t> node_ids;
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
    return validateWithReport(nullptr);
  }

  runtime::Status validateWithReport(ValidationReport* out_report) const {
    ValidationReport local_report;
    ValidationReport* report = out_report == nullptr ? &local_report : out_report;
    report->clear();

    bool saw_not_supported = false;

    for (std::size_t i = 0; i < tensors_.size(); ++i) {
      runtime::Status st = validateTensorSpec(tensors_[i].spec);
      if (st != runtime::Status::kSuccess) {
        report->issues.push_back(
            ValidationIssue{ValidationPass::kTensorSpec,
                            st,
                            -1,
                            static_cast<std::int64_t>(i),
                            "invalid tensor spec"});
      }
    }

    for (std::size_t i = 0; i < nodes_.size(); ++i) {
      const GraphNode& node = nodes_[i];
      if (node.id != i) {
        report->issues.push_back(
            ValidationIssue{ValidationPass::kControlDependency,
                            runtime::Status::kInvalidValue,
                            static_cast<std::int64_t>(i),
                            -1,
                            "node id does not match insertion order"});
      }

      const OperatorSchema* schema = globalOperatorRegistry().find(node.op);
      if (schema == nullptr) {
        report->issues.push_back(
            ValidationIssue{ValidationPass::kSchemaArity,
                            runtime::Status::kNotSupported,
                            static_cast<std::int64_t>(node.id),
                            -1,
                            "operator schema not found"});
        saw_not_supported = true;
        continue;
      }

      if (node.inputs.size() < schema->min_inputs || node.inputs.size() > schema->max_inputs ||
          node.outputs.size() < schema->min_outputs || node.outputs.size() > schema->max_outputs) {
        report->issues.push_back(
            ValidationIssue{ValidationPass::kSchemaArity,
                            runtime::Status::kInvalidValue,
                            static_cast<std::int64_t>(node.id),
                            -1,
                            "node input/output arity violates schema"});
      }

      for (std::size_t id : node.inputs) {
        if (id >= tensors_.size()) {
          report->issues.push_back(
              ValidationIssue{ValidationPass::kTensorReference,
                              runtime::Status::kInvalidValue,
                              static_cast<std::int64_t>(node.id),
                              static_cast<std::int64_t>(id),
                              "input tensor id out of range"});
        }
      }
      for (std::size_t id : node.outputs) {
        if (id >= tensors_.size()) {
          report->issues.push_back(
              ValidationIssue{ValidationPass::kTensorReference,
                              runtime::Status::kInvalidValue,
                              static_cast<std::int64_t>(node.id),
                              static_cast<std::int64_t>(id),
                              "output tensor id out of range"});
        }
      }
      for (std::size_t dep : node.control_deps) {
        if (dep >= node.id) {
          report->issues.push_back(
              ValidationIssue{ValidationPass::kControlDependency,
                              runtime::Status::kInvalidValue,
                              static_cast<std::int64_t>(node.id),
                              -1,
                              "control dependency must reference prior node"});
        }
      }

      const runtime::BackendCapabilities cpu_caps = runtime::backendCapabilities(runtime::Device::kCPU);
      const runtime::BackendCapabilities cuda_caps = runtime::backendCapabilities(runtime::Device::kCUDA);
      const runtime::BackendCapabilities metal_caps = runtime::backendCapabilities(runtime::Device::kMetal);
      const bool any_supported =
          (schema->supports_cpu && cpu_caps.compute_surface) ||
          (schema->supports_cuda && cuda_caps.compute_surface) ||
          (schema->supports_metal && metal_caps.compute_surface);
      if (!any_supported) {
        report->issues.push_back(
            ValidationIssue{ValidationPass::kBackendCapability,
                            runtime::Status::kNotSupported,
                            static_cast<std::int64_t>(node.id),
                            -1,
                            "no built backend capability for operator"});
        saw_not_supported = true;
      }
    }

    if (report->ok()) {
      return runtime::Status::kSuccess;
    }
    return saw_not_supported ? runtime::Status::kNotSupported : runtime::Status::kInvalidValue;
  }

  runtime::Status planForDevice(
      runtime::Device preferred_device,
      std::vector<GraphPlanStep>* out_steps) const {
    if (out_steps == nullptr) {
      return runtime::Status::kInvalidValue;
    }
    GraphPlannerOptions options;
    options.preferred_device = preferred_device;
    options.sync_policy.mode = runtime::SyncMode::kAuto;
    options.sync_policy.trace_sync_boundary = false;
    options.separate_fallback_segments = false;
    options.insert_sync_on_device_change = true;

    std::vector<GraphExecutionGroup> groups;
    return planExecutionGroups(options, &groups, out_steps);
  }

  runtime::Status planExecutionGroups(
      const GraphPlannerOptions& options,
      std::vector<GraphExecutionGroup>* out_groups,
      std::vector<GraphPlanStep>* out_steps = nullptr) const {
    if (out_groups == nullptr) {
      return runtime::Status::kInvalidValue;
    }

    runtime::Status st = validate();
    if (st != runtime::Status::kSuccess) {
      return st;
    }

    out_groups->clear();
    if (out_steps != nullptr) {
      out_steps->clear();
      out_steps->reserve(nodes_.size());
    }

    for (std::size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
      const GraphNode& node = nodes_[node_index];
      const OperatorSchema* schema = globalOperatorRegistry().find(node.op);
      if (schema == nullptr) {
        return runtime::Status::kNotSupported;
      }

      runtime::Device assigned = runtime::Device::kCPU;
      bool fallback = false;
      st = chooseDeviceForSchema(*schema, options.preferred_device, &assigned, &fallback);
      if (st != runtime::Status::kSuccess) {
        return st;
      }

      if (options.group_by_backend_capability) {
        // Capability-aware grouping heuristic:
        // 1) keep backend continuity with previous group when safe,
        // 2) align current flexible op with next forced backend when possible.
        const bool has_prev_group = !out_groups->empty();
        const runtime::Device prev_device = has_prev_group ? out_groups->back().assigned_device : assigned;

        runtime::Device next_forced_device = assigned;
        bool has_next_forced_device = false;
        if (node_index + 1 < nodes_.size()) {
          const GraphNode& next_node = nodes_[node_index + 1];
          const OperatorSchema* next_schema = globalOperatorRegistry().find(next_node.op);
          if (next_schema == nullptr) {
            return runtime::Status::kNotSupported;
          }
          const std::vector<runtime::Device> next_candidates = availableDevicesForSchema(*next_schema);
          if (next_candidates.size() == 1) {
            next_forced_device = next_candidates.front();
            has_next_forced_device = true;
          }
        }

        const bool supports_prev = has_prev_group && schemaSupportsAvailableDevice(*schema, prev_device);
        const bool supports_next_forced =
            has_next_forced_device && schemaSupportsAvailableDevice(*schema, next_forced_device);

        if (has_prev_group && assigned != prev_device && supports_prev) {
          // Prefer continuity if this node is already fallback, or if previous
          // group is on preferred backend, or if lookahead says next op is
          // forced to previous backend.
          if (fallback || prev_device == options.preferred_device ||
              (has_next_forced_device && next_forced_device == prev_device)) {
            assigned = prev_device;
          }
        }

        if (has_next_forced_device && supports_next_forced && assigned != next_forced_device) {
          // If the next op is forced to one backend, align this flexible op to
          // the same backend to avoid an extra backend boundary.
          assigned = next_forced_device;
        }

        fallback = (assigned != options.preferred_device);
      }

      if (out_steps != nullptr) {
        GraphPlanStep step;
        step.node_id = node.id;
        step.op = node.op;
        step.assigned_device = assigned;
        step.fallback = fallback;
        out_steps->push_back(step);
      }

      const bool force_single_node_group = (options.sync_policy.mode == runtime::SyncMode::kAlways);
      bool start_new_group = out_groups->empty() || force_single_node_group;
      if (!out_groups->empty() && !force_single_node_group) {
        const GraphExecutionGroup& prev = out_groups->back();
        if (prev.assigned_device != assigned) {
          start_new_group = true;
        }
        if (options.separate_fallback_segments && prev.fallback != fallback) {
          start_new_group = true;
        }
      }

      if (start_new_group) {
        GraphExecutionGroup group;
        group.group_id = out_groups->size();
        group.assigned_device = assigned;
        group.fallback = fallback;
        const bool device_changed =
            (!out_groups->empty()) && (out_groups->back().assigned_device != assigned);
        group.sync_boundary_before =
            (!out_groups->empty()) &&
            (options.sync_policy.mode == runtime::SyncMode::kAlways ||
             options.sync_policy.trace_sync_boundary ||
             (options.insert_sync_on_device_change && device_changed));
        group.sync_boundary_after = false;
        group.node_ids.push_back(node.id);

        if (!out_groups->empty() && group.sync_boundary_before) {
          out_groups->back().sync_boundary_after = true;
        }
        out_groups->push_back(std::move(group));
      } else {
        out_groups->back().node_ids.push_back(node.id);
      }
    }

    if (options.sync_policy.mode == runtime::SyncMode::kNever) {
      for (auto& group : *out_groups) {
        group.sync_boundary_before = false;
        group.sync_boundary_after = false;
      }
    } else if (options.sync_policy.mode == runtime::SyncMode::kAlways) {
      for (auto& group : *out_groups) {
        group.sync_boundary_after = true;
      }
    } else if (options.sync_policy.trace_sync_boundary && !out_groups->empty()) {
      out_groups->back().sync_boundary_after = true;
    }

    return runtime::Status::kSuccess;
  }

  runtime::Status executeF32(
      const GraphPlannerOptions& options,
      const std::unordered_map<std::size_t, std::vector<float>>& feeds,
      std::unordered_map<std::size_t, std::vector<float>>* out_values,
      std::vector<GraphExecutionGroup>* out_groups = nullptr,
      std::vector<GraphPlanStep>* out_steps = nullptr) const {
    return executeTyped<float>(options, feeds, out_values, out_groups, out_steps);
  }

  runtime::Status executeF64(
      const GraphPlannerOptions& options,
      const std::unordered_map<std::size_t, std::vector<double>>& feeds,
      std::unordered_map<std::size_t, std::vector<double>>* out_values,
      std::vector<GraphExecutionGroup>* out_groups = nullptr,
      std::vector<GraphPlanStep>* out_steps = nullptr) const {
    return executeTyped<double>(options, feeds, out_values, out_groups, out_steps);
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
  static runtime::Status chooseDeviceForSchema(
      const OperatorSchema& schema,
      runtime::Device preferred_device,
      runtime::Device* out_assigned_device,
      bool* out_fallback) {
    if (out_assigned_device == nullptr || out_fallback == nullptr) {
      return runtime::Status::kInvalidValue;
    }
    auto choose_if_supported = [&](runtime::Device device) -> bool {
      return schemaSupportsAvailableDevice(schema, device);
    };

    runtime::Device assigned = preferred_device;
    if (!choose_if_supported(assigned)) {
      if (preferred_device != runtime::Device::kMetal && choose_if_supported(runtime::Device::kMetal)) {
        assigned = runtime::Device::kMetal;
      } else if (preferred_device != runtime::Device::kCUDA && choose_if_supported(runtime::Device::kCUDA)) {
        assigned = runtime::Device::kCUDA;
      } else if (preferred_device != runtime::Device::kCPU && choose_if_supported(runtime::Device::kCPU)) {
        assigned = runtime::Device::kCPU;
      } else if (!choose_if_supported(preferred_device)) {
        return runtime::Status::kNotSupported;
      }
    }

    *out_assigned_device = assigned;
    *out_fallback = (assigned != preferred_device);
    return runtime::Status::kSuccess;
  }

  static bool schemaSupportsAvailableDevice(const OperatorSchema& schema, runtime::Device device) {
    if (!schema.supportsDevice(device)) {
      return false;
    }
    const runtime::BackendCapabilities caps = runtime::backendCapabilities(device);
    return caps.compute_surface && caps.available;
  }

  static std::vector<runtime::Device> availableDevicesForSchema(const OperatorSchema& schema) {
    std::vector<runtime::Device> out;
    out.reserve(3);
    if (schemaSupportsAvailableDevice(schema, runtime::Device::kMetal)) {
      out.push_back(runtime::Device::kMetal);
    }
    if (schemaSupportsAvailableDevice(schema, runtime::Device::kCUDA)) {
      out.push_back(runtime::Device::kCUDA);
    }
    if (schemaSupportsAvailableDevice(schema, runtime::Device::kCPU)) {
      out.push_back(runtime::Device::kCPU);
    }
    return out;
  }

  template <typename T>
  static constexpr DType dtypeForExecution() {
    if constexpr (std::is_same_v<T, float>) {
      return DType::kFloat32;
    }
    return DType::kFloat64;
  }

  template <typename T>
  runtime::Status validateExecutionValues(
      const std::unordered_map<std::size_t, std::vector<T>>& values) const {
    const DType expected_dtype = dtypeForExecution<T>();
    for (const auto& kv : values) {
      const std::size_t tensor_id = kv.first;
      if (tensor_id >= tensors_.size()) {
        return runtime::Status::kInvalidValue;
      }
      const GraphTensorValue& tensor = tensors_[tensor_id];
      if (tensor.spec.dtype != expected_dtype) {
        return runtime::Status::kInvalidValue;
      }
      const std::size_t expected_numel = detail::numelFromShapeContract(tensor.spec.shape);
      if (expected_numel == 0 || kv.second.size() != expected_numel) {
        return runtime::Status::kInvalidValue;
      }
    }
    return runtime::Status::kSuccess;
  }

  template <typename T>
  runtime::Status executeNodeTyped(
      const GraphNode& node,
      runtime::Device assigned_device,
      std::unordered_map<std::size_t, std::vector<T>>* values) const {
    if (values == nullptr) {
      return runtime::Status::kInvalidValue;
    }
    if (node.inputs.empty() || node.outputs.empty()) {
      return runtime::Status::kInvalidValue;
    }

    auto get_value = [&](std::size_t tensor_id) -> const std::vector<T>* {
      auto it = values->find(tensor_id);
      if (it == values->end()) {
        return nullptr;
      }
      return &it->second;
    };

    auto set_output = [&](std::size_t tensor_id, std::vector<T>&& out) -> runtime::Status {
      values->insert_or_assign(tensor_id, std::move(out));
      return runtime::Status::kSuccess;
    };

    for (std::size_t tensor_id : node.inputs) {
      if (tensor_id >= tensors_.size()) {
        return runtime::Status::kInvalidValue;
      }
      const std::vector<T>* data = get_value(tensor_id);
      if (data == nullptr) {
        return runtime::Status::kInvalidValue;
      }
      const std::size_t expected_numel = detail::numelFromShapeContract(tensors_[tensor_id].spec.shape);
      if (data->size() != expected_numel) {
        return runtime::Status::kInvalidValue;
      }
    }

    for (std::size_t tensor_id : node.outputs) {
      if (tensor_id >= tensors_.size()) {
        return runtime::Status::kInvalidValue;
      }
      if (tensors_[tensor_id].spec.dtype != dtypeForExecution<T>()) {
        return runtime::Status::kInvalidValue;
      }
    }

    if (node.outputs.size() != 1) {
      return runtime::Status::kInvalidValue;
    }

    const std::size_t out_id = node.outputs[0];
    const GraphTensorValue& out_tensor = tensors_[out_id];

    switch (node.op) {
      case OpKind::kMatMul: {
        if (node.inputs.size() != 2) {
          return runtime::Status::kInvalidValue;
        }
        const std::size_t lhs_id = node.inputs[0];
        const std::size_t rhs_id = node.inputs[1];
        const GraphTensorValue& lhs_tensor = tensors_[lhs_id];
        const GraphTensorValue& rhs_tensor = tensors_[rhs_id];
        const std::vector<T>* lhs = get_value(lhs_id);
        const std::vector<T>* rhs = get_value(rhs_id);
        if (lhs == nullptr || rhs == nullptr) {
          return runtime::Status::kInvalidValue;
        }
        if (lhs_tensor.spec.shape.size() != 2 || rhs_tensor.spec.shape.size() != 2 ||
            out_tensor.spec.shape.size() != 2) {
          return runtime::Status::kInvalidValue;
        }
        const std::size_t m = static_cast<std::size_t>(lhs_tensor.spec.shape[0]);
        const std::size_t k = static_cast<std::size_t>(lhs_tensor.spec.shape[1]);
        const std::size_t rhs_k = static_cast<std::size_t>(rhs_tensor.spec.shape[0]);
        const std::size_t n = static_cast<std::size_t>(rhs_tensor.spec.shape[1]);
        const std::size_t out_m = static_cast<std::size_t>(out_tensor.spec.shape[0]);
        const std::size_t out_n = static_cast<std::size_t>(out_tensor.spec.shape[1]);
        if (k != rhs_k || out_m != m || out_n != n) {
          return runtime::Status::kInvalidValue;
        }

        std::vector<T> out(detail::numelFromShapeContract(out_tensor.spec.shape), static_cast<T>(0));
        runtime::Status st = ops::matMul<T>(lhs->data(), rhs->data(), out.data(), m, k, n, assigned_device);
        if (st != runtime::Status::kSuccess) {
          return st;
        }
        return set_output(out_id, std::move(out));
      }
      case OpKind::kVectorAdd: {
        if (node.inputs.size() != 2) {
          return runtime::Status::kInvalidValue;
        }
        const std::size_t lhs_id = node.inputs[0];
        const std::size_t rhs_id = node.inputs[1];
        const GraphTensorValue& lhs_tensor = tensors_[lhs_id];
        const GraphTensorValue& rhs_tensor = tensors_[rhs_id];
        const std::vector<T>* lhs = get_value(lhs_id);
        const std::vector<T>* rhs = get_value(rhs_id);
        if (lhs == nullptr || rhs == nullptr) {
          return runtime::Status::kInvalidValue;
        }
        const std::size_t lhs_numel = detail::numelFromShapeContract(lhs_tensor.spec.shape);
        const std::size_t rhs_numel = detail::numelFromShapeContract(rhs_tensor.spec.shape);
        const std::size_t out_numel = detail::numelFromShapeContract(out_tensor.spec.shape);
        if (lhs_numel == 0 || lhs_numel != rhs_numel || lhs_numel != out_numel) {
          return runtime::Status::kInvalidValue;
        }

        std::vector<T> out(out_numel, static_cast<T>(0));
        runtime::Status st =
            ops::vectorAdd<T>(lhs->data(), rhs->data(), out.data(), out_numel, assigned_device);
        if (st != runtime::Status::kSuccess) {
          return st;
        }
        return set_output(out_id, std::move(out));
      }
      case OpKind::kMatrixSub:
      case OpKind::kMatrixDiv: {
        if (node.inputs.size() != 2) {
          return runtime::Status::kInvalidValue;
        }
        const std::size_t lhs_id = node.inputs[0];
        const std::size_t rhs_id = node.inputs[1];
        const GraphTensorValue& lhs_tensor = tensors_[lhs_id];
        const GraphTensorValue& rhs_tensor = tensors_[rhs_id];
        const std::vector<T>* lhs = get_value(lhs_id);
        const std::vector<T>* rhs = get_value(rhs_id);
        if (lhs == nullptr || rhs == nullptr) {
          return runtime::Status::kInvalidValue;
        }
        if (lhs_tensor.spec.shape.size() != 2 || rhs_tensor.spec.shape.size() != 2 ||
            out_tensor.spec.shape.size() != 2) {
          return runtime::Status::kInvalidValue;
        }
        if (lhs_tensor.spec.shape != rhs_tensor.spec.shape || lhs_tensor.spec.shape != out_tensor.spec.shape) {
          return runtime::Status::kInvalidValue;
        }
        const std::size_t rows = static_cast<std::size_t>(out_tensor.spec.shape[0]);
        const std::size_t cols = static_cast<std::size_t>(out_tensor.spec.shape[1]);
        std::vector<T> out(detail::numelFromShapeContract(out_tensor.spec.shape), static_cast<T>(0));

        runtime::Status st = runtime::Status::kSuccess;
        if (node.op == OpKind::kMatrixSub) {
          st = ops::matrixSub<T>(lhs->data(), rhs->data(), out.data(), rows, cols, assigned_device);
        } else {
          st = ops::matrixDiv<T>(lhs->data(), rhs->data(), out.data(), rows, cols, assigned_device);
        }
        if (st != runtime::Status::kSuccess) {
          return st;
        }
        return set_output(out_id, std::move(out));
      }
      case OpKind::kAttentionForward: {
        if constexpr (!std::is_same_v<T, float>) {
          return runtime::Status::kNotSupported;
        }
        if (node.inputs.size() != 3) {
          return runtime::Status::kInvalidValue;
        }
        const std::size_t q_id = node.inputs[0];
        const std::size_t k_id = node.inputs[1];
        const std::size_t v_id = node.inputs[2];
        const GraphTensorValue& q_tensor = tensors_[q_id];
        const GraphTensorValue& k_tensor = tensors_[k_id];
        const GraphTensorValue& v_tensor = tensors_[v_id];
        const std::vector<T>* q = get_value(q_id);
        const std::vector<T>* k = get_value(k_id);
        const std::vector<T>* v = get_value(v_id);
        if (q == nullptr || k == nullptr || v == nullptr) {
          return runtime::Status::kInvalidValue;
        }
        if (q_tensor.spec.shape.size() != 2 || k_tensor.spec.shape.size() != 2 ||
            v_tensor.spec.shape.size() != 2 || out_tensor.spec.shape.size() != 2) {
          return runtime::Status::kInvalidValue;
        }
        if (q_tensor.spec.shape != k_tensor.spec.shape ||
            q_tensor.spec.shape != v_tensor.spec.shape ||
            q_tensor.spec.shape != out_tensor.spec.shape) {
          return runtime::Status::kInvalidValue;
        }

        const std::size_t seq_len = static_cast<std::size_t>(q_tensor.spec.shape[0]);
        const std::size_t head_dim = static_cast<std::size_t>(q_tensor.spec.shape[1]);
        std::vector<T> out(detail::numelFromShapeContract(out_tensor.spec.shape), static_cast<T>(0));

        AttentionConfig cfg{seq_len, head_dim, false};
        runtime::Status st = attentionForward(
            reinterpret_cast<const float*>(q->data()),
            reinterpret_cast<const float*>(k->data()),
            reinterpret_cast<const float*>(v->data()),
            reinterpret_cast<float*>(out.data()),
            cfg,
            assigned_device);
        if (st != runtime::Status::kSuccess) {
          return st;
        }
        return set_output(out_id, std::move(out));
      }
      case OpKind::kConv2dNchw3x3s1p1: {
        if (node.inputs.size() != 2 && node.inputs.size() != 3) {
          return runtime::Status::kInvalidValue;
        }
        const std::size_t x_id = node.inputs[0];
        const std::size_t w_id = node.inputs[1];
        const std::size_t bias_id = (node.inputs.size() == 3) ? node.inputs[2] : static_cast<std::size_t>(-1);
        const GraphTensorValue& x_tensor = tensors_[x_id];
        const GraphTensorValue& w_tensor = tensors_[w_id];
        const std::vector<T>* x = get_value(x_id);
        const std::vector<T>* w = get_value(w_id);
        if (x == nullptr || w == nullptr) {
          return runtime::Status::kInvalidValue;
        }
        const std::vector<T>* bias = nullptr;
        const GraphTensorValue* bias_tensor = nullptr;
        if (node.inputs.size() == 3) {
          bias_tensor = &tensors_[bias_id];
          bias = get_value(bias_id);
          if (bias == nullptr) {
            return runtime::Status::kInvalidValue;
          }
        }

        if (x_tensor.spec.shape.size() != 4 || w_tensor.spec.shape.size() != 4 || out_tensor.spec.shape.size() != 4) {
          return runtime::Status::kInvalidValue;
        }
        const std::size_t batch = static_cast<std::size_t>(x_tensor.spec.shape[0]);
        const std::size_t in_channels = static_cast<std::size_t>(x_tensor.spec.shape[1]);
        const std::size_t in_h = static_cast<std::size_t>(x_tensor.spec.shape[2]);
        const std::size_t in_w = static_cast<std::size_t>(x_tensor.spec.shape[3]);
        const std::size_t out_channels = static_cast<std::size_t>(w_tensor.spec.shape[0]);
        const std::size_t w_in_channels = static_cast<std::size_t>(w_tensor.spec.shape[1]);
        const std::size_t kernel_h = static_cast<std::size_t>(w_tensor.spec.shape[2]);
        const std::size_t kernel_w = static_cast<std::size_t>(w_tensor.spec.shape[3]);
        if (kernel_h != 3 || kernel_w != 3 || in_channels != w_in_channels) {
          return runtime::Status::kInvalidValue;
        }
        if (out_tensor.spec.shape[0] != static_cast<std::int64_t>(batch) ||
            out_tensor.spec.shape[1] != static_cast<std::int64_t>(out_channels) ||
            out_tensor.spec.shape[2] != static_cast<std::int64_t>(in_h) ||
            out_tensor.spec.shape[3] != static_cast<std::int64_t>(in_w)) {
          return runtime::Status::kInvalidValue;
        }
        const T* bias_ptr = nullptr;
        if (bias != nullptr) {
          if (bias_tensor->spec.shape.size() != 1 ||
              bias_tensor->spec.shape[0] != static_cast<std::int64_t>(out_channels)) {
            return runtime::Status::kInvalidValue;
          }
          bias_ptr = bias->data();
        }

        std::vector<T> out(detail::numelFromShapeContract(out_tensor.spec.shape), static_cast<T>(0));
        runtime::Status st = ops::conv2dNchw<T>(
            x->data(),
            w->data(),
            bias_ptr,
            out.data(),
            batch,
            in_channels,
            in_h,
            in_w,
            out_channels,
            /*kernel_h=*/3,
            /*kernel_w=*/3,
            /*stride_h=*/1,
            /*stride_w=*/1,
            /*pad_h=*/1,
            /*pad_w=*/1,
            assigned_device,
            /*apply_relu=*/false);
        if (st != runtime::Status::kSuccess) {
          return st;
        }
        return set_output(out_id, std::move(out));
      }
      default:
        return runtime::Status::kNotSupported;
    }
  }

  template <typename T>
  runtime::Status executeTyped(
      const GraphPlannerOptions& options,
      const std::unordered_map<std::size_t, std::vector<T>>& feeds,
      std::unordered_map<std::size_t, std::vector<T>>* out_values,
      std::vector<GraphExecutionGroup>* out_groups,
      std::vector<GraphPlanStep>* out_steps) const {
    if (out_values == nullptr) {
      return runtime::Status::kInvalidValue;
    }

    std::vector<GraphExecutionGroup> groups;
    std::vector<GraphPlanStep> steps;
    runtime::Status st = planExecutionGroups(options, &groups, &steps);
    if (st != runtime::Status::kSuccess) {
      return st;
    }

    std::unordered_map<std::size_t, std::vector<T>> values = feeds;
    st = validateExecutionValues(values);
    if (st != runtime::Status::kSuccess) {
      return st;
    }

    std::unordered_set<std::size_t> visited_nodes;
    visited_nodes.reserve(nodes_.size());

    for (const auto& group : groups) {
      if (group.sync_boundary_before) {
        st = runtime::deviceSynchronizeWithPolicy(options.sync_policy);
        if (st != runtime::Status::kSuccess) {
          return st;
        }
      }

      for (std::size_t node_id : group.node_ids) {
        if (node_id >= nodes_.size()) {
          return runtime::Status::kInvalidValue;
        }
        if (!visited_nodes.insert(node_id).second) {
          return runtime::Status::kInvalidValue;
        }
        st = executeNodeTyped<T>(nodes_[node_id], group.assigned_device, &values);
        if (st != runtime::Status::kSuccess) {
          return st;
        }
      }

      if (group.sync_boundary_after) {
        st = runtime::deviceSynchronizeWithPolicy(options.sync_policy);
        if (st != runtime::Status::kSuccess) {
          return st;
        }
      }
    }

    if (visited_nodes.size() != nodes_.size()) {
      return runtime::Status::kInvalidValue;
    }

    *out_values = std::move(values);
    if (out_groups != nullptr) {
      *out_groups = std::move(groups);
    }
    if (out_steps != nullptr) {
      *out_steps = std::move(steps);
    }
    return runtime::Status::kSuccess;
  }

  std::vector<GraphTensorValue> tensors_;
  std::vector<GraphNode> nodes_;
};

}  // namespace lightning_core::graph
