#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <sstream>
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
  kConv2dNchw3x3,
  kConv2dNchw3x3s1p1,
  kQkvPackRepeat,
  kRelu
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
    case OpKind::kConv2dNchw3x3:
      return "conv2d_nchw3x3";
    case OpKind::kConv2dNchw3x3s1p1:
      return "conv2d_nchw3x3s1p1";
    case OpKind::kQkvPackRepeat:
      return "qkv_pack_repeat";
    case OpKind::kRelu:
      return "relu";
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
  // Optional per-input/per-output expected ranks. If shorter than runtime arity,
  // the last element is reused.
  std::vector<int> input_ranks;
  std::vector<int> output_ranks;
  bool allow_float32{true};
  bool allow_float64{true};
  bool require_contiguous_layout{false};
  std::unordered_set<std::string> allowed_attributes;

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

    {
      OperatorSchema s;
      s.kind = OpKind::kMatMul;
      s.name = "matmul";
      s.min_inputs = 2;
      s.max_inputs = 2;
      s.min_outputs = 1;
      s.max_outputs = 1;
      s.supports_cpu = true;
      s.supports_cuda = cuda_supported;
      s.supports_metal = metal_supported;
      s.input_ranks = {2, 2};
      s.output_ranks = {2};
      s.require_contiguous_layout = true;
      s.allow_float32 = true;
      s.allow_float64 = true;
      (void)registerSchema(s);
    }
    {
      OperatorSchema s;
      s.kind = OpKind::kVectorAdd;
      s.name = "vector_add";
      s.min_inputs = 2;
      s.max_inputs = 2;
      s.min_outputs = 1;
      s.max_outputs = 1;
      s.supports_cpu = true;
      s.supports_cuda = cuda_supported;
      s.supports_metal = metal_supported;
      s.input_ranks = {-1, -1};  // rank follows producer; shape check is per-op
      s.output_ranks = {-1};
      s.require_contiguous_layout = true;
      s.allow_float32 = true;
      s.allow_float64 = true;
      (void)registerSchema(s);
    }
    {
      OperatorSchema s;
      s.kind = OpKind::kMatrixSub;
      s.name = "matrix_sub";
      s.min_inputs = 2;
      s.max_inputs = 2;
      s.min_outputs = 1;
      s.max_outputs = 1;
      s.supports_cpu = true;
      s.supports_cuda = cuda_supported;
      s.supports_metal = metal_supported;
      s.input_ranks = {2, 2};
      s.output_ranks = {2};
      s.require_contiguous_layout = true;
      s.allow_float32 = true;
      s.allow_float64 = true;
      (void)registerSchema(s);
    }
    {
      OperatorSchema s;
      s.kind = OpKind::kMatrixDiv;
      s.name = "matrix_div";
      s.min_inputs = 2;
      s.max_inputs = 2;
      s.min_outputs = 1;
      s.max_outputs = 1;
      s.supports_cpu = true;
      s.supports_cuda = cuda_supported;
      s.supports_metal = metal_supported;
      s.input_ranks = {2, 2};
      s.output_ranks = {2};
      s.require_contiguous_layout = true;
      s.allow_float32 = true;
      s.allow_float64 = true;
      (void)registerSchema(s);
    }
    {
      OperatorSchema s;
      s.kind = OpKind::kAttentionForward;
      s.name = "attention_forward";
      s.min_inputs = 3;
      s.max_inputs = 3;
      s.min_outputs = 1;
      s.max_outputs = 1;
      s.supports_cpu = true;
      s.supports_cuda = false;
      s.supports_metal = metal_supported;
      s.input_ranks = {2, 2, 2};
      s.output_ranks = {2};
      s.require_contiguous_layout = true;
      s.allow_float32 = true;
      s.allow_float64 = false;
      s.allowed_attributes = {"causal"};
      (void)registerSchema(s);
    }
    {
      OperatorSchema s;
      s.kind = OpKind::kConv2dNchw3x3;
      s.name = "conv2d_nchw3x3";
      s.min_inputs = 2;
      s.max_inputs = 3;
      s.min_outputs = 1;
      s.max_outputs = 1;
      s.supports_cpu = true;
      s.supports_cuda = false;
      s.supports_metal = metal_supported;
      s.input_ranks = {4, 4, 1};
      s.output_ranks = {4};
      s.require_contiguous_layout = true;
      s.allow_float32 = true;
      s.allow_float64 = false;
      s.allowed_attributes = {"stride_h", "stride_w", "pad_h", "pad_w", "apply_relu"};
      (void)registerSchema(s);
    }
    {
      OperatorSchema s;
      s.kind = OpKind::kConv2dNchw3x3s1p1;
      s.name = "conv2d_nchw3x3s1p1";
      s.min_inputs = 2;
      s.max_inputs = 3;
      s.min_outputs = 1;
      s.max_outputs = 1;
      s.supports_cpu = false;
      s.supports_cuda = false;
      s.supports_metal = metal_supported;
      s.input_ranks = {4, 4, 1};
      s.output_ranks = {4};
      s.require_contiguous_layout = true;
      s.allow_float32 = true;
      s.allow_float64 = false;
      s.allowed_attributes = {"stride_h", "stride_w", "pad_h", "pad_w"};
      (void)registerSchema(s);
    }
    {
      OperatorSchema s;
      s.kind = OpKind::kQkvPackRepeat;
      s.name = "qkv_pack_repeat";
      s.min_inputs = 1;
      s.max_inputs = 1;
      s.min_outputs = 3;
      s.max_outputs = 3;
      s.supports_cpu = true;
      s.supports_cuda = false;
      s.supports_metal = metal_supported;
      s.input_ranks = {4};
      s.output_ranks = {2, 2, 2};
      s.require_contiguous_layout = true;
      s.allow_float32 = true;
      s.allow_float64 = false;
      (void)registerSchema(s);
    }
    {
      OperatorSchema s;
      s.kind = OpKind::kRelu;
      s.name = "relu";
      s.min_inputs = 1;
      s.max_inputs = 1;
      s.min_outputs = 1;
      s.max_outputs = 1;
      s.supports_cpu = true;
      s.supports_cuda = cuda_supported;
      s.supports_metal = metal_supported;
      s.input_ranks = {-1};
      s.output_ranks = {-1};
      s.require_contiguous_layout = true;
      s.allow_float32 = true;
      s.allow_float64 = true;
      (void)registerSchema(s);
    }
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
  std::unordered_map<std::string, std::int64_t> attributes;
};

struct GraphPlanStep {
  std::size_t node_id{0};
  OpKind op{OpKind::kMatMul};
  runtime::Device assigned_device{runtime::Device::kCPU};
  bool fallback{false};
  std::string fallback_reason_code{"none"};
  std::string fallback_reason{"no_fallback"};
};

enum class ValidationPass {
  kSchemaContract = 0,
  kTopology,
  kAliasLifetime,
  kLayoutFlow,
  kBackendCapability
};

inline const char* validationPassName(ValidationPass pass) {
  switch (pass) {
    case ValidationPass::kSchemaContract:
      return "schema_contract";
    case ValidationPass::kTopology:
      return "topology";
    case ValidationPass::kAliasLifetime:
      return "alias_lifetime";
    case ValidationPass::kLayoutFlow:
      return "layout_flow";
    case ValidationPass::kBackendCapability:
      return "backend_capability";
    default:
      return "unknown";
  }
}

enum class ReasonCode {
  kNone = 0,
  kInvalidTensorSpec,
  kSchemaNotFound,
  kArityMismatch,
  kTensorOutOfRange,
  kControlDependencyInvalid,
  kBackendUnavailable,
  kDTypeMismatch,
  kLayoutMismatch,
  kRankMismatch,
  kShapeConstraintViolation,
  kAttributeUnsupported,
  kAliasLifetimeViolation,
  kFallbackPreferredUnsupported,
  kFallbackPreferredUnavailable,
  kFallbackPlannerOptimized
};

inline const char* reasonCodeName(ReasonCode code) {
  switch (code) {
    case ReasonCode::kNone:
      return "none";
    case ReasonCode::kInvalidTensorSpec:
      return "invalid_tensor_spec";
    case ReasonCode::kSchemaNotFound:
      return "schema_not_found";
    case ReasonCode::kArityMismatch:
      return "arity_mismatch";
    case ReasonCode::kTensorOutOfRange:
      return "tensor_out_of_range";
    case ReasonCode::kControlDependencyInvalid:
      return "control_dependency_invalid";
    case ReasonCode::kBackendUnavailable:
      return "backend_unavailable";
    case ReasonCode::kDTypeMismatch:
      return "dtype_mismatch";
    case ReasonCode::kLayoutMismatch:
      return "layout_mismatch";
    case ReasonCode::kRankMismatch:
      return "rank_mismatch";
    case ReasonCode::kShapeConstraintViolation:
      return "shape_constraint_violation";
    case ReasonCode::kAttributeUnsupported:
      return "attribute_unsupported";
    case ReasonCode::kAliasLifetimeViolation:
      return "alias_lifetime_violation";
    case ReasonCode::kFallbackPreferredUnsupported:
      return "fallback_preferred_unsupported";
    case ReasonCode::kFallbackPreferredUnavailable:
      return "fallback_preferred_unavailable";
    case ReasonCode::kFallbackPlannerOptimized:
      return "fallback_planner_optimized";
    default:
      return "unknown";
  }
}

struct ValidationIssue {
  ValidationPass pass{ValidationPass::kSchemaContract};
  runtime::Status status{runtime::Status::kInvalidValue};
  std::int64_t node_id{-1};
  std::int64_t tensor_id{-1};
  ReasonCode reason{ReasonCode::kNone};
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
  bool enable_fusion_v1{true};
  bool enable_fusion_cost_model_v1{true};
  bool enable_plan_cache{true};
  // Predicted unfused/fused ratio threshold. Fusing is allowed when:
  // estimated_speedup = unfused_cost / fused_cost >= fusion_cost_min_speedup.
  double fusion_cost_min_speedup{1.01};
  // Simple cost model v1 coefficients (nanoseconds).
  double cost_launch_overhead_ns{12000.0};
  double cost_transfer_overhead_ns{4000.0};
  double cost_elementwise_per_element_ns{0.50};
  double cost_matmul_per_mac_ns{0.0035};
  double cost_conv_per_mac_ns{0.0022};
};

struct GraphExecutionGroup {
  std::size_t group_id{0};
  runtime::Device assigned_device{runtime::Device::kCPU};
  bool fallback{false};
  std::string fallback_reason_code{"none"};
  std::string fallback_reason{"no_fallback"};
  bool sync_boundary_before{false};
  bool sync_boundary_after{false};
  std::vector<std::size_t> node_ids;
};

struct GraphPlanSummary {
  std::size_t total_nodes{0};
  std::size_t total_groups{0};
  std::size_t total_fallback_nodes{0};
  std::size_t total_fallback_groups{0};
  std::size_t sync_boundary_before_groups{0};
  std::size_t sync_boundary_after_groups{0};
  std::size_t device_switches{0};
  std::size_t planned_dispatch_groups{0};
  std::size_t cpu_nodes{0};
  std::size_t cuda_nodes{0};
  std::size_t metal_nodes{0};
  std::size_t cpu_groups{0};
  std::size_t cuda_groups{0};
  std::size_t metal_groups{0};
  bool plan_cache_enabled{false};
  bool plan_cache_hit{false};
  std::size_t plan_cache_hits_total{0};
  std::size_t plan_cache_misses_total{0};
  double plan_cache_hit_rate_pct{0.0};
};

struct GraphPlanCacheStats {
  std::size_t hits{0};
  std::size_t misses{0};
  double hit_rate_pct{0.0};
};

enum class FusionPattern {
  kConvReluV1 = 0,
  kMatMulBiasReluV1,
  kAttentionProjV1
};

inline const char* fusionPatternName(FusionPattern pattern) {
  switch (pattern) {
    case FusionPattern::kConvReluV1:
      return "conv_relu_v1";
    case FusionPattern::kMatMulBiasReluV1:
      return "matmul_bias_relu_v1";
    case FusionPattern::kAttentionProjV1:
      return "attention_proj_v1";
    default:
      return "unknown";
  }
}

struct GraphFusionDecision {
  FusionPattern pattern{FusionPattern::kConvReluV1};
  std::size_t start_node_id{0};
  std::size_t end_node_id{0};
  runtime::Device assigned_device{runtime::Device::kCPU};
  bool fused{false};
  bool cost_model_applied{false};
  double estimated_unfused_cost_ns{std::numeric_limits<double>::quiet_NaN()};
  double estimated_fused_cost_ns{std::numeric_limits<double>::quiet_NaN()};
  double estimated_speedup{std::numeric_limits<double>::quiet_NaN()};
  std::string reason;
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
    invalidatePlanCache();
    *out_id = tensors_.size() - 1;
    return runtime::Status::kSuccess;
  }

  runtime::Status addNode(
      OpKind op,
      const std::vector<std::size_t>& inputs,
      const std::vector<std::size_t>& outputs,
      const std::vector<std::size_t>& control_deps = std::vector<std::size_t>(),
      const std::string& name = std::string(),
      const std::unordered_map<std::string, std::int64_t>& attributes = std::unordered_map<std::string, std::int64_t>()) {
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
    node.attributes = attributes;
    nodes_.push_back(std::move(node));
    invalidatePlanCache();
    return runtime::Status::kSuccess;
  }

  runtime::Status validate() const {
    return validateWithReport(nullptr);
  }

  runtime::Status validateWithReport(ValidationReport* out_report) const {
    ValidationReport local_report;
    ValidationReport* report = out_report == nullptr ? &local_report : out_report;
    report->clear();

    auto push_issue = [&](ValidationPass pass,
                          runtime::Status status,
                          std::int64_t node_id,
                          std::int64_t tensor_id,
                          ReasonCode reason,
                          const std::string& message) {
      report->issues.push_back(ValidationIssue{pass, status, node_id, tensor_id, reason, message});
    };

    bool saw_not_supported = false;
    std::vector<std::size_t> writer_counts(tensors_.size(), 0);

    for (std::size_t i = 0; i < tensors_.size(); ++i) {
      runtime::Status st = validateTensorSpec(tensors_[i].spec);
      if (st != runtime::Status::kSuccess) {
        push_issue(
            ValidationPass::kSchemaContract,
            st,
            -1,
            static_cast<std::int64_t>(i),
            ReasonCode::kInvalidTensorSpec,
            "invalid tensor spec");
      }
    }

    auto tensor_valid = [&](std::size_t id) {
      return id < tensors_.size();
    };
    auto dtype_allowed = [&](const OperatorSchema& schema, DType dt) {
      if (dt == DType::kFloat32) {
        return schema.allow_float32;
      }
      return schema.allow_float64;
    };
    auto rank_expected = [](const std::vector<int>& ranks, std::size_t index) -> int {
      if (ranks.empty()) {
        return -1;
      }
      if (index < ranks.size()) {
        return ranks[index];
      }
      return ranks.back();
    };

    for (std::size_t i = 0; i < nodes_.size(); ++i) {
      const GraphNode& node = nodes_[i];
      if (node.id != i) {
        push_issue(
            ValidationPass::kTopology,
            runtime::Status::kInvalidValue,
            static_cast<std::int64_t>(i),
            -1,
            ReasonCode::kControlDependencyInvalid,
            "node id does not match insertion order");
      }

      const OperatorSchema* schema = globalOperatorRegistry().find(node.op);
      if (schema == nullptr) {
        push_issue(
            ValidationPass::kSchemaContract,
            runtime::Status::kNotSupported,
            static_cast<std::int64_t>(node.id),
            -1,
            ReasonCode::kSchemaNotFound,
            "operator schema not found");
        saw_not_supported = true;
        continue;
      }

      if (node.inputs.size() < schema->min_inputs || node.inputs.size() > schema->max_inputs ||
          node.outputs.size() < schema->min_outputs || node.outputs.size() > schema->max_outputs) {
        push_issue(
            ValidationPass::kSchemaContract,
            runtime::Status::kInvalidValue,
            static_cast<std::int64_t>(node.id),
            -1,
            ReasonCode::kArityMismatch,
            "node input/output arity violates schema");
      }

      for (const auto& kv : node.attributes) {
        if (schema->allowed_attributes.find(kv.first) == schema->allowed_attributes.end()) {
          push_issue(
              ValidationPass::kSchemaContract,
              runtime::Status::kInvalidValue,
              static_cast<std::int64_t>(node.id),
              -1,
              ReasonCode::kAttributeUnsupported,
              "attribute not allowed by schema: " + kv.first);
        }
      }

      for (std::size_t slot = 0; slot < node.inputs.size(); ++slot) {
        const std::size_t id = node.inputs[slot];
        if (!tensor_valid(id)) {
          push_issue(
              ValidationPass::kTopology,
              runtime::Status::kInvalidValue,
              static_cast<std::int64_t>(node.id),
              static_cast<std::int64_t>(id),
              ReasonCode::kTensorOutOfRange,
              "input tensor id out of range");
          continue;
        }
        const GraphTensorValue& t = tensors_[id];
        const int expected_rank = rank_expected(schema->input_ranks, slot);
        if (expected_rank >= 0 && static_cast<int>(t.spec.shape.size()) != expected_rank) {
          push_issue(
              ValidationPass::kLayoutFlow,
              runtime::Status::kInvalidValue,
              static_cast<std::int64_t>(node.id),
              static_cast<std::int64_t>(id),
              ReasonCode::kRankMismatch,
              "input tensor rank mismatch");
        }
        if (schema->require_contiguous_layout && t.spec.layout != Layout::kContiguous) {
          push_issue(
              ValidationPass::kLayoutFlow,
              runtime::Status::kInvalidValue,
              static_cast<std::int64_t>(node.id),
              static_cast<std::int64_t>(id),
              ReasonCode::kLayoutMismatch,
              "input layout must be contiguous");
        }
        if (!dtype_allowed(*schema, t.spec.dtype)) {
          push_issue(
              ValidationPass::kSchemaContract,
              runtime::Status::kInvalidValue,
              static_cast<std::int64_t>(node.id),
              static_cast<std::int64_t>(id),
              ReasonCode::kDTypeMismatch,
              "input dtype not allowed by schema");
        }
      }

      for (std::size_t slot = 0; slot < node.outputs.size(); ++slot) {
        const std::size_t id = node.outputs[slot];
        if (!tensor_valid(id)) {
          push_issue(
              ValidationPass::kTopology,
              runtime::Status::kInvalidValue,
              static_cast<std::int64_t>(node.id),
              static_cast<std::int64_t>(id),
              ReasonCode::kTensorOutOfRange,
              "output tensor id out of range");
          continue;
        }
        const GraphTensorValue& t = tensors_[id];
        const int expected_rank = rank_expected(schema->output_ranks, slot);
        if (expected_rank >= 0 && static_cast<int>(t.spec.shape.size()) != expected_rank) {
          push_issue(
              ValidationPass::kLayoutFlow,
              runtime::Status::kInvalidValue,
              static_cast<std::int64_t>(node.id),
              static_cast<std::int64_t>(id),
              ReasonCode::kRankMismatch,
              "output tensor rank mismatch");
        }
        if (schema->require_contiguous_layout && t.spec.layout != Layout::kContiguous) {
          push_issue(
              ValidationPass::kLayoutFlow,
              runtime::Status::kInvalidValue,
              static_cast<std::int64_t>(node.id),
              static_cast<std::int64_t>(id),
              ReasonCode::kLayoutMismatch,
              "output layout must be contiguous");
        }
        if (!dtype_allowed(*schema, t.spec.dtype)) {
          push_issue(
              ValidationPass::kSchemaContract,
              runtime::Status::kInvalidValue,
              static_cast<std::int64_t>(node.id),
              static_cast<std::int64_t>(id),
              ReasonCode::kDTypeMismatch,
              "output dtype not allowed by schema");
        }
        if (tensors_[id].constant) {
          push_issue(
              ValidationPass::kAliasLifetime,
              runtime::Status::kInvalidValue,
              static_cast<std::int64_t>(node.id),
              static_cast<std::int64_t>(id),
              ReasonCode::kAliasLifetimeViolation,
              "node output cannot target constant tensor");
        }
        writer_counts[id] += 1;
      }

      for (std::size_t dep : node.control_deps) {
        if (dep >= node.id) {
          push_issue(
              ValidationPass::kTopology,
              runtime::Status::kInvalidValue,
              static_cast<std::int64_t>(node.id),
              -1,
              ReasonCode::kControlDependencyInvalid,
              "control dependency must reference prior node");
        }
      }

      for (std::size_t out_id : node.outputs) {
        if (!tensor_valid(out_id)) {
          continue;
        }
        for (std::size_t in_id : node.inputs) {
          if (in_id == out_id) {
            push_issue(
                ValidationPass::kAliasLifetime,
                runtime::Status::kInvalidValue,
                static_cast<std::int64_t>(node.id),
                static_cast<std::int64_t>(out_id),
                ReasonCode::kAliasLifetimeViolation,
                "node input/output tensor id alias is not allowed");
          }
        }
      }

      auto shape_violation = [&](const std::string& msg) {
        push_issue(
            ValidationPass::kLayoutFlow,
            runtime::Status::kInvalidValue,
            static_cast<std::int64_t>(node.id),
            -1,
            ReasonCode::kShapeConstraintViolation,
            msg);
      };
      if (node.op == OpKind::kMatMul && node.inputs.size() == 2 && node.outputs.size() == 1) {
        if (tensor_valid(node.inputs[0]) && tensor_valid(node.inputs[1]) && tensor_valid(node.outputs[0])) {
          const auto& a = tensors_[node.inputs[0]].spec.shape;
          const auto& b = tensors_[node.inputs[1]].spec.shape;
          const auto& o = tensors_[node.outputs[0]].spec.shape;
          if (a.size() == 2 && b.size() == 2 && o.size() == 2) {
            if (a[1] != b[0] || o[0] != a[0] || o[1] != b[1]) {
              shape_violation("matmul shape constraint violated");
            }
          }
        }
      } else if (
          (node.op == OpKind::kVectorAdd || node.op == OpKind::kMatrixSub || node.op == OpKind::kMatrixDiv) &&
          node.inputs.size() == 2 &&
          node.outputs.size() == 1) {
        if (tensor_valid(node.inputs[0]) && tensor_valid(node.inputs[1]) && tensor_valid(node.outputs[0])) {
          const auto& a = tensors_[node.inputs[0]].spec.shape;
          const auto& b = tensors_[node.inputs[1]].spec.shape;
          const auto& o = tensors_[node.outputs[0]].spec.shape;
          if (a != b || a != o) {
            shape_violation("binary op shape constraint violated");
          }
        }
      } else if (node.op == OpKind::kRelu && node.inputs.size() == 1 && node.outputs.size() == 1) {
        if (tensor_valid(node.inputs[0]) && tensor_valid(node.outputs[0])) {
          const auto& i_shape = tensors_[node.inputs[0]].spec.shape;
          const auto& o_shape = tensors_[node.outputs[0]].spec.shape;
          if (i_shape != o_shape) {
            shape_violation("relu output shape must match input shape");
          }
        }
      } else if (node.op == OpKind::kAttentionForward && node.inputs.size() == 3 && node.outputs.size() == 1) {
        if (tensor_valid(node.inputs[0]) &&
            tensor_valid(node.inputs[1]) &&
            tensor_valid(node.inputs[2]) &&
            tensor_valid(node.outputs[0])) {
          const auto& q = tensors_[node.inputs[0]].spec.shape;
          const auto& k = tensors_[node.inputs[1]].spec.shape;
          const auto& v = tensors_[node.inputs[2]].spec.shape;
          const auto& o = tensors_[node.outputs[0]].spec.shape;
          if (q != k || q != v || q != o) {
            shape_violation("attention q/k/v/output shape constraint violated");
          }
        }
      } else if (
          (node.op == OpKind::kConv2dNchw3x3 || node.op == OpKind::kConv2dNchw3x3s1p1) &&
          (node.inputs.size() == 2 || node.inputs.size() == 3) &&
          node.outputs.size() == 1) {
        if (tensor_valid(node.inputs[0]) && tensor_valid(node.inputs[1]) && tensor_valid(node.outputs[0])) {
          const auto& x = tensors_[node.inputs[0]].spec.shape;
          const auto& w = tensors_[node.inputs[1]].spec.shape;
          const auto& y = tensors_[node.outputs[0]].spec.shape;
          if (x.size() == 4 && w.size() == 4 && y.size() == 4) {
            const auto attr_or = [&](const char* key, std::int64_t defv) -> std::int64_t {
              auto it = node.attributes.find(key);
              return it == node.attributes.end() ? defv : it->second;
            };
            const std::int64_t stride_h =
                (node.op == OpKind::kConv2dNchw3x3s1p1) ? 1 : attr_or("stride_h", 1);
            const std::int64_t stride_w =
                (node.op == OpKind::kConv2dNchw3x3s1p1) ? 1 : attr_or("stride_w", 1);
            const std::int64_t pad_h =
                (node.op == OpKind::kConv2dNchw3x3s1p1) ? 1 : attr_or("pad_h", 0);
            const std::int64_t pad_w =
                (node.op == OpKind::kConv2dNchw3x3s1p1) ? 1 : attr_or("pad_w", 0);
            if (stride_h <= 0 || stride_w <= 0 || pad_h < 0 || pad_w < 0) {
              shape_violation("conv2d_nchw3x3 attribute constraint violated");
            }
            const bool kernel_ok = (w[2] == 3 && w[3] == 3);
            const bool channel_ok = (x[1] == w[1]);
            const std::int64_t numer_h = x[2] + (2 * pad_h) - 3;
            const std::int64_t numer_w = x[3] + (2 * pad_w) - 3;
            const bool numer_ok = (numer_h >= 0 && numer_w >= 0);
            const std::int64_t expect_h = numer_ok ? (numer_h / stride_h + 1) : -1;
            const std::int64_t expect_w = numer_ok ? (numer_w / stride_w + 1) : -1;
            const bool output_ok =
                (y[0] == x[0] && y[1] == w[0] && y[2] == expect_h && y[3] == expect_w);
            if (!kernel_ok || !channel_ok || !output_ok) {
              shape_violation(
                  node.op == OpKind::kConv2dNchw3x3
                      ? "conv2d_nchw3x3 shape constraint violated"
                      : "conv2d_nchw3x3s1p1 shape constraint violated");
            }
          }
          if (node.inputs.size() == 3 && tensor_valid(node.inputs[2])) {
            const auto& b = tensors_[node.inputs[2]].spec.shape;
            if (b.size() != 1 || b[0] != w[0]) {
              shape_violation("conv2d bias shape constraint violated");
            }
          }
        }
      } else if (node.op == OpKind::kQkvPackRepeat && node.inputs.size() == 1 && node.outputs.size() == 3) {
        if (tensor_valid(node.inputs[0]) &&
            tensor_valid(node.outputs[0]) &&
            tensor_valid(node.outputs[1]) &&
            tensor_valid(node.outputs[2])) {
          const auto& in = tensors_[node.inputs[0]].spec.shape;
          const auto& q = tensors_[node.outputs[0]].spec.shape;
          const auto& k = tensors_[node.outputs[1]].spec.shape;
          const auto& v = tensors_[node.outputs[2]].spec.shape;
          if (in.size() != 4 || q.size() != 2 || k.size() != 2 || v.size() != 2 || q != k || q != v) {
            shape_violation("qkv_pack_repeat shape constraint violated");
          }
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
        push_issue(
            ValidationPass::kBackendCapability,
            runtime::Status::kNotSupported,
            static_cast<std::int64_t>(node.id),
            -1,
            ReasonCode::kBackendUnavailable,
            "no built backend capability for operator");
        saw_not_supported = true;
      }
    }

    for (std::size_t tid = 0; tid < writer_counts.size(); ++tid) {
      if (writer_counts[tid] > 1) {
        push_issue(
            ValidationPass::kAliasLifetime,
            runtime::Status::kInvalidValue,
            -1,
            static_cast<std::int64_t>(tid),
            ReasonCode::kAliasLifetimeViolation,
            "tensor has multiple writer nodes");
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

    last_plan_cache_hit_ = false;
    out_groups->clear();
    if (out_steps != nullptr) {
      out_steps->clear();
      out_steps->reserve(nodes_.size());
    }

    if (options.enable_plan_cache) {
      const std::uint64_t key = planCacheKeyForOptions(options);
      if (plan_cache_.valid &&
          plan_cache_.key == key &&
          plan_cache_.graph_revision == graph_revision_) {
        *out_groups = plan_cache_.groups;
        if (out_steps != nullptr) {
          *out_steps = plan_cache_.steps;
        }
        plan_cache_stats_hits_ += 1;
        last_plan_cache_hit_ = true;
        return runtime::Status::kSuccess;
      }
      plan_cache_stats_misses_ += 1;
    }

    for (std::size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
      const GraphNode& node = nodes_[node_index];
      const OperatorSchema* schema = globalOperatorRegistry().find(node.op);
      if (schema == nullptr) {
        return runtime::Status::kNotSupported;
      }

      runtime::Device assigned = runtime::Device::kCPU;
      bool fallback = false;
      ReasonCode fallback_reason = ReasonCode::kNone;
      st = chooseDeviceForSchema(*schema, options.preferred_device, &assigned, &fallback, &fallback_reason);
      if (st != runtime::Status::kSuccess) {
        return st;
      }

      if (options.group_by_backend_capability) {
        // Capability-aware grouping heuristic (pass-2):
        // choose the lowest boundary/sync/fallback score among candidate backends.
        const bool has_prev_group = !out_groups->empty();
        const runtime::Device prev_device = has_prev_group ? out_groups->back().assigned_device : assigned;
        const bool prev_fallback = has_prev_group ? out_groups->back().fallback : false;

        runtime::Device next_forced_device = assigned;
        bool has_next_forced_device = false;
        const OperatorSchema* next_schema = nullptr;
        if (node_index + 1 < nodes_.size()) {
          const GraphNode& next_node = nodes_[node_index + 1];
          next_schema = globalOperatorRegistry().find(next_node.op);
          if (next_schema == nullptr) {
            return runtime::Status::kNotSupported;
          }
          const std::vector<runtime::Device> next_candidates = availableDevicesForSchema(*next_schema);
          if (next_candidates.size() == 1) {
            next_forced_device = next_candidates.front();
            has_next_forced_device = true;
          }
        }

        const std::vector<runtime::Device> candidates = availableDevicesForSchema(*schema);
        if (candidates.empty()) {
          return runtime::Status::kNotSupported;
        }

        double best_score = std::numeric_limits<double>::infinity();
        runtime::Device best_device = assigned;
        bool best_fallback = fallback;

        for (runtime::Device candidate : candidates) {
          const bool candidate_fallback = (candidate != options.preferred_device);
          double score = 0.0;

          // Soft penalty for fallback to keep preferred backend when trade-off is small.
          if (candidate_fallback) {
            score += 1.25;
          } else {
            score -= 0.10;
          }

          // Previous-group continuity and sync-boundary penalty.
          if (has_prev_group && candidate != prev_device) {
            score += 2.0;
            if (options.insert_sync_on_device_change) {
              score += 4.0;
            }
          }
          if (has_prev_group && options.separate_fallback_segments && candidate_fallback != prev_fallback) {
            score += 2.0;
          }

          // Lookahead to forced backend of next op.
          if (has_next_forced_device && candidate != next_forced_device) {
            score += 3.0;
          }

          // If next node can run on this candidate backend, reward continuity.
          if (next_schema != nullptr) {
            if (schemaSupportsAvailableDevice(*next_schema, candidate)) {
              score -= 0.75;
            } else {
              score += 0.50;
            }
          }

          // Tiny tie-breaker to reduce plan churn when scores are almost equal.
          if (candidate == assigned) {
            score -= 0.05;
          }

          if (score < best_score) {
            best_score = score;
            best_device = candidate;
            best_fallback = candidate_fallback;
          }
        }

        assigned = best_device;
        fallback = best_fallback;
        if (fallback) {
          if (schema->supportsDevice(options.preferred_device) &&
              schemaSupportsAvailableDevice(*schema, options.preferred_device) &&
              assigned != options.preferred_device) {
            fallback_reason = ReasonCode::kFallbackPlannerOptimized;
          }
        } else {
          fallback_reason = ReasonCode::kNone;
        }
      }

      if (out_steps != nullptr) {
        GraphPlanStep step;
        step.node_id = node.id;
        step.op = node.op;
        step.assigned_device = assigned;
        step.fallback = fallback;
        step.fallback_reason_code = reasonCodeName(fallback_reason);
        step.fallback_reason = fallback ? ("fallback(" + std::string(reasonCodeName(fallback_reason)) + ")") : "no_fallback";
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
        group.fallback_reason_code = reasonCodeName(fallback_reason);
        group.fallback_reason = fallback ? ("fallback(" + std::string(reasonCodeName(fallback_reason)) + ")") : "no_fallback";
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

    if (options.enable_plan_cache) {
      plan_cache_.valid = true;
      plan_cache_.key = planCacheKeyForOptions(options);
      plan_cache_.graph_revision = graph_revision_;
      plan_cache_.groups = *out_groups;
      if (out_steps != nullptr) {
        plan_cache_.steps = *out_steps;
      } else {
        plan_cache_.steps.clear();
      }
    }

    return runtime::Status::kSuccess;
  }

  runtime::Status planSummary(
      const GraphPlannerOptions& options,
      GraphPlanSummary* out_summary,
      std::vector<GraphExecutionGroup>* out_groups = nullptr,
      std::vector<GraphPlanStep>* out_steps = nullptr) const {
    if (out_summary == nullptr) {
      return runtime::Status::kInvalidValue;
    }
    std::vector<GraphExecutionGroup> local_groups;
    std::vector<GraphPlanStep> local_steps;
    runtime::Status st = planExecutionGroups(options, &local_groups, &local_steps);
    if (st != runtime::Status::kSuccess) {
      return st;
    }
    buildPlanSummary(local_groups, local_steps, out_summary);
    out_summary->plan_cache_enabled = options.enable_plan_cache;
    out_summary->plan_cache_hit = last_plan_cache_hit_;
    out_summary->plan_cache_hits_total = plan_cache_stats_hits_;
    out_summary->plan_cache_misses_total = plan_cache_stats_misses_;
    const double denom = static_cast<double>(plan_cache_stats_hits_ + plan_cache_stats_misses_);
    out_summary->plan_cache_hit_rate_pct =
        (denom > 0.0) ? (static_cast<double>(plan_cache_stats_hits_) * 100.0 / denom) : 0.0;
    if (out_groups != nullptr) {
      *out_groups = std::move(local_groups);
    }
    if (out_steps != nullptr) {
      *out_steps = std::move(local_steps);
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

  runtime::Status fusionReport(
      const GraphPlannerOptions& options,
      std::vector<GraphFusionDecision>* out_decisions) const {
    if (out_decisions == nullptr) {
      return runtime::Status::kInvalidValue;
    }
    std::vector<GraphExecutionGroup> groups;
    std::vector<GraphPlanStep> steps;
    runtime::Status st = planExecutionGroups(options, &groups, &steps);
    if (st != runtime::Status::kSuccess) {
      return st;
    }

    out_decisions->clear();
    std::unordered_map<std::size_t, runtime::Device> step_device;
    step_device.reserve(steps.size());
    for (const auto& step : steps) {
      step_device[step.node_id] = step.assigned_device;
    }
    std::unordered_map<std::size_t, std::size_t> node_group;
    node_group.reserve(nodes_.size());
    for (std::size_t gi = 0; gi < groups.size(); ++gi) {
      for (std::size_t node_id : groups[gi].node_ids) {
        node_group[node_id] = gi;
      }
    }
    const std::vector<std::size_t> consumer_counts = buildTensorConsumerCounts();

    for (std::size_t i = 0; i < nodes_.size(); ++i) {
      const GraphNode& node = nodes_[i];

      if (node.op == OpKind::kConv2dNchw3x3s1p1) {
        GraphFusionDecision decision;
        decision.pattern = FusionPattern::kConvReluV1;
        decision.start_node_id = node.id;
        decision.end_node_id = node.id;
        auto step_it = step_device.find(node.id);
        decision.assigned_device = (step_it != step_device.end()) ? step_it->second : runtime::Device::kCPU;

        if (i + 1 >= nodes_.size()) {
          decision.reason = "next_op_missing";
          out_decisions->push_back(std::move(decision));
          continue;
        }

        const GraphNode& next = nodes_[i + 1];
        decision.end_node_id = next.id;
        if (next.op != OpKind::kRelu) {
          decision.reason = "next_op_not_relu";
          out_decisions->push_back(std::move(decision));
          continue;
        }

        std::string eligibility_reason;
        if (!canFuseConvReluPair(node, next, consumer_counts, &eligibility_reason)) {
          decision.reason = eligibility_reason;
          out_decisions->push_back(std::move(decision));
          continue;
        }

        const auto g0_it = node_group.find(node.id);
        const auto g1_it = node_group.find(next.id);
        if (g0_it == node_group.end() || g1_it == node_group.end() || g0_it->second != g1_it->second) {
          decision.reason = "planner_group_boundary";
          out_decisions->push_back(std::move(decision));
          continue;
        }

        const auto d1_it = step_device.find(next.id);
        const runtime::Device next_device = (d1_it != step_device.end()) ? d1_it->second : decision.assigned_device;
        if (next_device != decision.assigned_device) {
          decision.reason = "device_boundary";
          out_decisions->push_back(std::move(decision));
          continue;
        }

        if (!options.enable_fusion_v1) {
          decision.reason = "fusion_disabled";
          out_decisions->push_back(std::move(decision));
          continue;
        }

        const GraphTensorValue& x_tensor = tensors_[node.inputs[0]];
        const GraphTensorValue& w_tensor = tensors_[node.inputs[1]];
        const GraphTensorValue& conv_mid_tensor = tensors_[node.outputs[0]];
        const FusionCostEstimate estimate = estimateConvReluCost(options, x_tensor, w_tensor, conv_mid_tensor);
        if (!applyCostModelDecision(options, estimate, &decision)) {
          out_decisions->push_back(std::move(decision));
          continue;
        }

        decision.fused = true;
        decision.reason = options.enable_fusion_cost_model_v1
            ? formatCostModelPassReason(estimate.speedup)
            : "fused_conv_relu_v1";
        out_decisions->push_back(std::move(decision));
        continue;
      }

      if (node.op == OpKind::kMatMul) {
        GraphFusionDecision decision;
        decision.pattern = FusionPattern::kMatMulBiasReluV1;
        decision.start_node_id = node.id;
        decision.end_node_id = node.id;
        auto step_it = step_device.find(node.id);
        decision.assigned_device = (step_it != step_device.end()) ? step_it->second : runtime::Device::kCPU;

        if (i + 2 >= nodes_.size()) {
          decision.reason = "next_ops_missing";
          out_decisions->push_back(std::move(decision));
          continue;
        }

        const GraphNode& next = nodes_[i + 1];
        const GraphNode& next2 = nodes_[i + 2];
        decision.end_node_id = next2.id;
        if (next.op != OpKind::kVectorAdd) {
          decision.reason = "middle_op_not_vector_add";
          out_decisions->push_back(std::move(decision));
          continue;
        }
        if (next2.op != OpKind::kRelu) {
          decision.reason = "final_op_not_relu";
          out_decisions->push_back(std::move(decision));
          continue;
        }

        std::string eligibility_reason;
        if (!canFuseMatMulBiasReluTriplet(node, next, next2, consumer_counts, &eligibility_reason)) {
          decision.reason = eligibility_reason;
          out_decisions->push_back(std::move(decision));
          continue;
        }

        const auto g0_it = node_group.find(node.id);
        const auto g1_it = node_group.find(next.id);
        const auto g2_it = node_group.find(next2.id);
        if (g0_it == node_group.end() ||
            g1_it == node_group.end() ||
            g2_it == node_group.end() ||
            g0_it->second != g1_it->second ||
            g1_it->second != g2_it->second) {
          decision.reason = "planner_group_boundary";
          out_decisions->push_back(std::move(decision));
          continue;
        }

        const auto d1_it = step_device.find(next.id);
        const auto d2_it = step_device.find(next2.id);
        const runtime::Device d1 = (d1_it != step_device.end()) ? d1_it->second : decision.assigned_device;
        const runtime::Device d2 = (d2_it != step_device.end()) ? d2_it->second : decision.assigned_device;
        if (d1 != decision.assigned_device || d2 != decision.assigned_device) {
          decision.reason = "device_boundary";
          out_decisions->push_back(std::move(decision));
          continue;
        }

        if (!options.enable_fusion_v1) {
          decision.reason = "fusion_disabled";
          out_decisions->push_back(std::move(decision));
          continue;
        }

        const GraphTensorValue& lhs_tensor = tensors_[node.inputs[0]];
        const GraphTensorValue& rhs_tensor = tensors_[node.inputs[1]];
        const GraphTensorValue& vec_out_tensor = tensors_[next.outputs[0]];
        const GraphTensorValue& relu_out_tensor = tensors_[next2.outputs[0]];
        const FusionCostEstimate estimate =
            estimateMatMulBiasReluCost(options, lhs_tensor, rhs_tensor, vec_out_tensor, relu_out_tensor);
        if (!applyCostModelDecision(options, estimate, &decision)) {
          out_decisions->push_back(std::move(decision));
          continue;
        }

        decision.fused = true;
        decision.reason = options.enable_fusion_cost_model_v1
            ? formatCostModelPassReason(estimate.speedup)
            : "fused_matmul_bias_relu_v1";
        out_decisions->push_back(std::move(decision));
        continue;
      }

      if (node.op == OpKind::kAttentionForward) {
        GraphFusionDecision decision;
        decision.pattern = FusionPattern::kAttentionProjV1;
        decision.start_node_id = node.id;
        decision.end_node_id = node.id;
        auto step_it = step_device.find(node.id);
        decision.assigned_device = (step_it != step_device.end()) ? step_it->second : runtime::Device::kCPU;

        if (i + 1 >= nodes_.size()) {
          decision.reason = "next_op_missing";
          out_decisions->push_back(std::move(decision));
          continue;
        }

        const GraphNode& next = nodes_[i + 1];
        decision.end_node_id = next.id;
        if (next.op != OpKind::kMatMul) {
          decision.reason = "next_op_not_matmul";
          out_decisions->push_back(std::move(decision));
          continue;
        }

        std::string eligibility_reason;
        if (!canFuseAttentionProjPair(node, next, consumer_counts, &eligibility_reason)) {
          decision.reason = eligibility_reason;
          out_decisions->push_back(std::move(decision));
          continue;
        }

        const auto g0_it = node_group.find(node.id);
        const auto g1_it = node_group.find(next.id);
        if (g0_it == node_group.end() || g1_it == node_group.end() || g0_it->second != g1_it->second) {
          decision.reason = "planner_group_boundary";
          out_decisions->push_back(std::move(decision));
          continue;
        }

        const auto d1_it = step_device.find(next.id);
        const runtime::Device d1 = (d1_it != step_device.end()) ? d1_it->second : decision.assigned_device;
        if (d1 != decision.assigned_device) {
          decision.reason = "device_boundary";
          out_decisions->push_back(std::move(decision));
          continue;
        }

        if (!options.enable_fusion_v1) {
          decision.reason = "fusion_disabled";
          out_decisions->push_back(std::move(decision));
          continue;
        }

        const GraphTensorValue& q_tensor = tensors_[node.inputs[0]];
        const GraphTensorValue& attn_out_tensor = tensors_[node.outputs[0]];
        const GraphTensorValue& proj_weight_tensor = tensors_[next.inputs[1]];
        const GraphTensorValue& proj_out_tensor = tensors_[next.outputs[0]];
        const FusionCostEstimate estimate =
            estimateAttentionProjCost(options, q_tensor, attn_out_tensor, proj_weight_tensor, proj_out_tensor);
        if (!applyCostModelDecision(options, estimate, &decision)) {
          out_decisions->push_back(std::move(decision));
          continue;
        }

        decision.fused = true;
        decision.reason = options.enable_fusion_cost_model_v1
            ? formatCostModelPassReason(estimate.speedup)
            : "fused_attention_proj_v1";
        out_decisions->push_back(std::move(decision));
      }
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

  void clearPlanCache() const {
    plan_cache_ = PlanCacheEntry{};
    last_plan_cache_hit_ = false;
    plan_cache_stats_hits_ = 0;
    plan_cache_stats_misses_ = 0;
  }

  GraphPlanCacheStats planCacheStats() const {
    GraphPlanCacheStats out{};
    out.hits = plan_cache_stats_hits_;
    out.misses = plan_cache_stats_misses_;
    const double denom = static_cast<double>(out.hits + out.misses);
    out.hit_rate_pct = (denom > 0.0) ? (static_cast<double>(out.hits) * 100.0 / denom) : 0.0;
    return out;
  }

 private:
  struct PlanCacheEntry {
    bool valid{false};
    std::size_t graph_revision{0};
    std::uint64_t key{0};
    std::vector<GraphExecutionGroup> groups;
    std::vector<GraphPlanStep> steps;
  };

  static std::uint64_t hashCombine(std::uint64_t seed, std::uint64_t v) {
    return seed ^ (v + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U));
  }

  static std::uint64_t hashDouble(double v) {
    union Bits {
      double d;
      std::uint64_t u;
    };
    Bits bits{};
    bits.d = v;
    return bits.u;
  }

  std::uint64_t planCacheKeyForOptions(const GraphPlannerOptions& options) const {
    std::uint64_t seed = 1469598103934665603ULL;
    seed = hashCombine(seed, static_cast<std::uint64_t>(options.preferred_device));
    seed = hashCombine(seed, static_cast<std::uint64_t>(options.sync_policy.mode));
    seed = hashCombine(seed, options.sync_policy.trace_sync_boundary ? 1ULL : 0ULL);
    seed = hashCombine(seed, options.group_by_backend_capability ? 1ULL : 0ULL);
    seed = hashCombine(seed, options.separate_fallback_segments ? 1ULL : 0ULL);
    seed = hashCombine(seed, options.insert_sync_on_device_change ? 1ULL : 0ULL);
    seed = hashCombine(seed, options.enable_fusion_v1 ? 1ULL : 0ULL);
    seed = hashCombine(seed, options.enable_fusion_cost_model_v1 ? 1ULL : 0ULL);
    seed = hashCombine(seed, options.enable_plan_cache ? 1ULL : 0ULL);
    seed = hashCombine(seed, hashDouble(options.fusion_cost_min_speedup));
    seed = hashCombine(seed, hashDouble(options.cost_launch_overhead_ns));
    seed = hashCombine(seed, hashDouble(options.cost_transfer_overhead_ns));
    seed = hashCombine(seed, hashDouble(options.cost_elementwise_per_element_ns));
    seed = hashCombine(seed, hashDouble(options.cost_matmul_per_mac_ns));
    seed = hashCombine(seed, hashDouble(options.cost_conv_per_mac_ns));
    return seed;
  }

  void invalidatePlanCache() {
    graph_revision_ += 1;
    plan_cache_.valid = false;
    last_plan_cache_hit_ = false;
  }

  static void incrementDeviceNodeCount(runtime::Device device, GraphPlanSummary* summary) {
    if (summary == nullptr) {
      return;
    }
    switch (device) {
      case runtime::Device::kCPU:
        summary->cpu_nodes += 1;
        break;
      case runtime::Device::kCUDA:
        summary->cuda_nodes += 1;
        break;
      case runtime::Device::kMetal:
      default:
        summary->metal_nodes += 1;
        break;
    }
  }

  static void incrementDeviceGroupCount(runtime::Device device, GraphPlanSummary* summary) {
    if (summary == nullptr) {
      return;
    }
    switch (device) {
      case runtime::Device::kCPU:
        summary->cpu_groups += 1;
        break;
      case runtime::Device::kCUDA:
        summary->cuda_groups += 1;
        break;
      case runtime::Device::kMetal:
      default:
        summary->metal_groups += 1;
        break;
    }
  }

  static void buildPlanSummary(
      const std::vector<GraphExecutionGroup>& groups,
      const std::vector<GraphPlanStep>& steps,
      GraphPlanSummary* out_summary) {
    if (out_summary == nullptr) {
      return;
    }
    GraphPlanSummary summary{};
    summary.total_nodes = steps.size();
    summary.total_groups = groups.size();
    summary.planned_dispatch_groups = groups.size();

    runtime::Device prev_group_device = runtime::Device::kCPU;
    bool has_prev_group = false;
    for (const auto& group : groups) {
      if (group.fallback) {
        summary.total_fallback_groups += 1;
      }
      if (group.sync_boundary_before) {
        summary.sync_boundary_before_groups += 1;
      }
      if (group.sync_boundary_after) {
        summary.sync_boundary_after_groups += 1;
      }
      incrementDeviceGroupCount(group.assigned_device, &summary);
      if (has_prev_group && prev_group_device != group.assigned_device) {
        summary.device_switches += 1;
      }
      prev_group_device = group.assigned_device;
      has_prev_group = true;
    }

    for (const auto& step : steps) {
      if (step.fallback) {
        summary.total_fallback_nodes += 1;
      }
      incrementDeviceNodeCount(step.assigned_device, &summary);
    }

    *out_summary = summary;
  }

  struct FusionCostEstimate {
    double unfused_ns{std::numeric_limits<double>::quiet_NaN()};
    double fused_ns{std::numeric_limits<double>::quiet_NaN()};
    double speedup{std::numeric_limits<double>::quiet_NaN()};
  };

  static std::string formatCostModelRejectReason(double speedup, double threshold) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(3);
    oss << "cost_model_reject(speedup=" << speedup << "<" << threshold << ")";
    return oss.str();
  }

  static std::string formatCostModelPassReason(double speedup) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(3);
    oss << "cost_model_pass(speedup=" << speedup << ")";
    return oss.str();
  }

  static bool applyCostModelDecision(
      const GraphPlannerOptions& options,
      const FusionCostEstimate& estimate,
      GraphFusionDecision* decision,
      bool* out_enabled_and_rejected = nullptr) {
    if (decision == nullptr) {
      return false;
    }
    decision->estimated_unfused_cost_ns = estimate.unfused_ns;
    decision->estimated_fused_cost_ns = estimate.fused_ns;
    decision->estimated_speedup = estimate.speedup;
    decision->cost_model_applied = options.enable_fusion_cost_model_v1;

    if (!options.enable_fusion_cost_model_v1) {
      return true;
    }

    const bool valid = std::isfinite(estimate.speedup) && estimate.speedup > 0.0;
    const bool rejected = (!valid) || (estimate.speedup < options.fusion_cost_min_speedup);
    if (out_enabled_and_rejected != nullptr) {
      *out_enabled_and_rejected = rejected;
    }
    if (rejected) {
      decision->reason = formatCostModelRejectReason(estimate.speedup, options.fusion_cost_min_speedup);
      decision->fused = false;
      return false;
    }
    return true;
  }

  static FusionCostEstimate estimateConvReluCost(
      const GraphPlannerOptions& options,
      const GraphTensorValue& x_tensor,
      const GraphTensorValue& w_tensor,
      const GraphTensorValue& conv_mid_tensor) {
    FusionCostEstimate out;
    if (x_tensor.spec.shape.size() != 4 || w_tensor.spec.shape.size() != 4 || conv_mid_tensor.spec.shape.size() != 4) {
      return out;
    }
    const double batch = static_cast<double>(x_tensor.spec.shape[0]);
    const double in_channels = static_cast<double>(x_tensor.spec.shape[1]);
    const double out_h = static_cast<double>(conv_mid_tensor.spec.shape[2]);
    const double out_w = static_cast<double>(conv_mid_tensor.spec.shape[3]);
    const double out_channels = static_cast<double>(w_tensor.spec.shape[0]);
    const double kernel_h = static_cast<double>(w_tensor.spec.shape[2]);
    const double kernel_w = static_cast<double>(w_tensor.spec.shape[3]);
    const double conv_macs = batch * out_h * out_w * out_channels * in_channels * kernel_h * kernel_w;
    const double relu_elements = static_cast<double>(detail::numelFromShapeContract(conv_mid_tensor.spec.shape));
    const double conv_core_ns = conv_macs * options.cost_conv_per_mac_ns;
    const double relu_core_ns = relu_elements * options.cost_elementwise_per_element_ns;

    out.unfused_ns = (2.0 * options.cost_launch_overhead_ns) + options.cost_transfer_overhead_ns + conv_core_ns + relu_core_ns;
    out.fused_ns = options.cost_launch_overhead_ns + conv_core_ns + (0.55 * relu_core_ns);
    out.speedup = (out.fused_ns > 0.0) ? (out.unfused_ns / out.fused_ns) : std::numeric_limits<double>::quiet_NaN();
    return out;
  }

  static FusionCostEstimate estimateMatMulBiasReluCost(
      const GraphPlannerOptions& options,
      const GraphTensorValue& lhs_tensor,
      const GraphTensorValue& rhs_tensor,
      const GraphTensorValue& vec_out_tensor,
      const GraphTensorValue& relu_out_tensor) {
    FusionCostEstimate out;
    if (lhs_tensor.spec.shape.size() != 2 ||
        rhs_tensor.spec.shape.size() != 2 ||
        vec_out_tensor.spec.shape.size() != 2 ||
        relu_out_tensor.spec.shape.size() != 2) {
      return out;
    }
    const double m = static_cast<double>(lhs_tensor.spec.shape[0]);
    const double k = static_cast<double>(lhs_tensor.spec.shape[1]);
    const double n = static_cast<double>(rhs_tensor.spec.shape[1]);
    const double mm_macs = m * k * n;
    const double bias_elements = static_cast<double>(detail::numelFromShapeContract(vec_out_tensor.spec.shape));
    const double relu_elements = static_cast<double>(detail::numelFromShapeContract(relu_out_tensor.spec.shape));
    const double mm_core_ns = mm_macs * options.cost_matmul_per_mac_ns;
    const double bias_core_ns = bias_elements * options.cost_elementwise_per_element_ns;
    const double relu_core_ns = relu_elements * options.cost_elementwise_per_element_ns;

    out.unfused_ns =
        (3.0 * options.cost_launch_overhead_ns) + (2.0 * options.cost_transfer_overhead_ns) + mm_core_ns + bias_core_ns + relu_core_ns;
    out.fused_ns =
        options.cost_launch_overhead_ns + mm_core_ns + (0.65 * bias_core_ns) + (0.50 * relu_core_ns);
    out.speedup = (out.fused_ns > 0.0) ? (out.unfused_ns / out.fused_ns) : std::numeric_limits<double>::quiet_NaN();
    return out;
  }

  static FusionCostEstimate estimateAttentionProjCost(
      const GraphPlannerOptions& options,
      const GraphTensorValue& q_tensor,
      const GraphTensorValue& attn_out_tensor,
      const GraphTensorValue& proj_weight_tensor,
      const GraphTensorValue& proj_out_tensor) {
    FusionCostEstimate out;
    if (q_tensor.spec.shape.size() != 2 ||
        attn_out_tensor.spec.shape.size() != 2 ||
        proj_weight_tensor.spec.shape.size() != 2 ||
        proj_out_tensor.spec.shape.size() != 2) {
      return out;
    }
    const double seq = static_cast<double>(q_tensor.spec.shape[0]);
    const double d = static_cast<double>(q_tensor.spec.shape[1]);
    const double proj_out = static_cast<double>(proj_out_tensor.spec.shape[1]);
    const double attn_macs = (2.0 * seq * seq * d) + (seq * seq * d);
    const double proj_macs = seq * d * proj_out;
    const double softmax_elems = seq * seq;
    const double attn_core_ns =
        (attn_macs * options.cost_matmul_per_mac_ns * 1.10) +
        (softmax_elems * options.cost_elementwise_per_element_ns * 1.35);
    const double proj_core_ns = proj_macs * options.cost_matmul_per_mac_ns;

    out.unfused_ns =
        (2.0 * options.cost_launch_overhead_ns) + options.cost_transfer_overhead_ns + attn_core_ns + proj_core_ns;
    out.fused_ns = options.cost_launch_overhead_ns + attn_core_ns + proj_core_ns;
    out.speedup = (out.fused_ns > 0.0) ? (out.unfused_ns / out.fused_ns) : std::numeric_limits<double>::quiet_NaN();
    return out;
  }

  std::vector<std::size_t> buildTensorConsumerCounts() const {
    std::vector<std::size_t> counts(tensors_.size(), 0);
    for (const auto& node : nodes_) {
      for (std::size_t input_id : node.inputs) {
        if (input_id < counts.size()) {
          counts[input_id] += 1;
        }
      }
    }
    return counts;
  }

  bool canFuseConvReluPair(const GraphNode& conv_node,
                           const GraphNode& relu_node,
                           const std::vector<std::size_t>& consumer_counts,
                           std::string* out_reason) const {
    auto set_reason = [&](const char* reason) {
      if (out_reason != nullptr) {
        *out_reason = reason;
      }
    };

    if (conv_node.op != OpKind::kConv2dNchw3x3s1p1) {
      set_reason("conv_pattern_mismatch");
      return false;
    }
    if (relu_node.op != OpKind::kRelu) {
      set_reason("next_op_not_relu");
      return false;
    }
    if (conv_node.id + 1 != relu_node.id) {
      set_reason("non_adjacent_node");
      return false;
    }
    if (conv_node.inputs.size() != 2 && conv_node.inputs.size() != 3) {
      set_reason("conv_arity_mismatch");
      return false;
    }
    if (conv_node.outputs.size() != 1 || relu_node.inputs.size() != 1 || relu_node.outputs.size() != 1) {
      set_reason("relu_arity_mismatch");
      return false;
    }
    const std::size_t conv_out_id = conv_node.outputs[0];
    if (conv_out_id >= tensors_.size()) {
      set_reason("intermediate_out_of_range");
      return false;
    }
    if (relu_node.inputs[0] != conv_out_id) {
      set_reason("intermediate_tensor_mismatch");
      return false;
    }
    if (conv_out_id >= consumer_counts.size() || consumer_counts[conv_out_id] != 1) {
      set_reason("intermediate_multi_consumer");
      return false;
    }
    const std::size_t relu_out_id = relu_node.outputs[0];
    if (relu_out_id >= tensors_.size()) {
      set_reason("relu_output_out_of_range");
      return false;
    }
    const GraphTensorValue& conv_mid_tensor = tensors_[conv_out_id];
    const GraphTensorValue& relu_out_tensor = tensors_[relu_out_id];
    if (conv_mid_tensor.spec.dtype != relu_out_tensor.spec.dtype ||
        conv_mid_tensor.spec.layout != relu_out_tensor.spec.layout ||
        conv_mid_tensor.spec.shape != relu_out_tensor.spec.shape) {
      set_reason("relu_output_spec_mismatch");
      return false;
    }
    if (!relu_node.control_deps.empty()) {
      set_reason("relu_control_dependency_present");
      return false;
    }
    set_reason("eligible");
    return true;
  }

  bool canFuseMatMulBiasReluTriplet(const GraphNode& matmul_node,
                                    const GraphNode& bias_node,
                                    const GraphNode& relu_node,
                                    const std::vector<std::size_t>& consumer_counts,
                                    std::string* out_reason) const {
    auto set_reason = [&](const char* reason) {
      if (out_reason != nullptr) {
        *out_reason = reason;
      }
    };

    if (matmul_node.op != OpKind::kMatMul) {
      set_reason("matmul_pattern_mismatch");
      return false;
    }
    if (bias_node.op != OpKind::kVectorAdd) {
      set_reason("middle_op_not_vector_add");
      return false;
    }
    if (relu_node.op != OpKind::kRelu) {
      set_reason("final_op_not_relu");
      return false;
    }
    if (matmul_node.id + 1 != bias_node.id || bias_node.id + 1 != relu_node.id) {
      set_reason("non_adjacent_node");
      return false;
    }
    if (matmul_node.inputs.size() != 2 ||
        matmul_node.outputs.size() != 1 ||
        bias_node.inputs.size() != 2 ||
        bias_node.outputs.size() != 1 ||
        relu_node.inputs.size() != 1 ||
        relu_node.outputs.size() != 1) {
      set_reason("arity_mismatch");
      return false;
    }

    const std::size_t mm_out_id = matmul_node.outputs[0];
    const std::size_t vec_out_id = bias_node.outputs[0];
    const std::size_t relu_out_id = relu_node.outputs[0];
    const std::size_t bias_a = bias_node.inputs[0];
    const std::size_t bias_b = bias_node.inputs[1];

    if (mm_out_id >= tensors_.size() || vec_out_id >= tensors_.size() || relu_out_id >= tensors_.size()) {
      set_reason("intermediate_out_of_range");
      return false;
    }
    if (bias_a != mm_out_id && bias_b != mm_out_id) {
      set_reason("matmul_output_not_used_by_bias_add");
      return false;
    }
    const std::size_t bias_tensor_id = (bias_a == mm_out_id) ? bias_b : bias_a;
    if (bias_tensor_id >= tensors_.size()) {
      set_reason("bias_tensor_out_of_range");
      return false;
    }
    if (relu_node.inputs[0] != vec_out_id) {
      set_reason("relu_input_not_bias_output");
      return false;
    }
    if (mm_out_id >= consumer_counts.size() || consumer_counts[mm_out_id] != 1) {
      set_reason("matmul_output_multi_consumer");
      return false;
    }
    if (vec_out_id >= consumer_counts.size() || consumer_counts[vec_out_id] != 1) {
      set_reason("bias_output_multi_consumer");
      return false;
    }

    const GraphTensorValue& lhs_tensor = tensors_[matmul_node.inputs[0]];
    const GraphTensorValue& rhs_tensor = tensors_[matmul_node.inputs[1]];
    const GraphTensorValue& mm_out_tensor = tensors_[mm_out_id];
    const GraphTensorValue& bias_tensor = tensors_[bias_tensor_id];
    const GraphTensorValue& vec_out_tensor = tensors_[vec_out_id];
    const GraphTensorValue& relu_out_tensor = tensors_[relu_out_id];

    if (lhs_tensor.spec.shape.size() != 2 ||
        rhs_tensor.spec.shape.size() != 2 ||
        mm_out_tensor.spec.shape.size() != 2 ||
        vec_out_tensor.spec.shape.size() != 2 ||
        relu_out_tensor.spec.shape.size() != 2) {
      set_reason("tensor_rank_mismatch");
      return false;
    }
    if (lhs_tensor.spec.shape[1] != rhs_tensor.spec.shape[0]) {
      set_reason("matmul_inner_dim_mismatch");
      return false;
    }
    if (mm_out_tensor.spec.shape[0] != lhs_tensor.spec.shape[0] ||
        mm_out_tensor.spec.shape[1] != rhs_tensor.spec.shape[1]) {
      set_reason("matmul_output_shape_mismatch");
      return false;
    }
    if (bias_tensor.spec.shape != mm_out_tensor.spec.shape ||
        vec_out_tensor.spec.shape != mm_out_tensor.spec.shape ||
        relu_out_tensor.spec.shape != mm_out_tensor.spec.shape) {
      set_reason("bias_or_relu_shape_mismatch");
      return false;
    }
    if (lhs_tensor.spec.dtype != rhs_tensor.spec.dtype ||
        lhs_tensor.spec.dtype != mm_out_tensor.spec.dtype ||
        lhs_tensor.spec.dtype != bias_tensor.spec.dtype ||
        lhs_tensor.spec.dtype != vec_out_tensor.spec.dtype ||
        lhs_tensor.spec.dtype != relu_out_tensor.spec.dtype) {
      set_reason("dtype_mismatch");
      return false;
    }
    if (lhs_tensor.spec.layout != Layout::kContiguous ||
        rhs_tensor.spec.layout != Layout::kContiguous ||
        mm_out_tensor.spec.layout != Layout::kContiguous ||
        bias_tensor.spec.layout != Layout::kContiguous ||
        vec_out_tensor.spec.layout != Layout::kContiguous ||
        relu_out_tensor.spec.layout != Layout::kContiguous) {
      set_reason("layout_not_contiguous");
      return false;
    }
    if (!bias_node.control_deps.empty() || !relu_node.control_deps.empty()) {
      set_reason("control_dependency_present");
      return false;
    }
    set_reason("eligible");
    return true;
  }

  bool canFuseAttentionProjPair(const GraphNode& attn_node,
                                const GraphNode& proj_node,
                                const std::vector<std::size_t>& consumer_counts,
                                std::string* out_reason) const {
    auto set_reason = [&](const char* reason) {
      if (out_reason != nullptr) {
        *out_reason = reason;
      }
    };

    if (attn_node.op != OpKind::kAttentionForward) {
      set_reason("attention_pattern_mismatch");
      return false;
    }
    if (proj_node.op != OpKind::kMatMul) {
      set_reason("next_op_not_matmul");
      return false;
    }
    if (attn_node.id + 1 != proj_node.id) {
      set_reason("non_adjacent_node");
      return false;
    }
    if (attn_node.inputs.size() != 3 ||
        attn_node.outputs.size() != 1 ||
        proj_node.inputs.size() != 2 ||
        proj_node.outputs.size() != 1) {
      set_reason("arity_mismatch");
      return false;
    }

    const std::size_t attn_out_id = attn_node.outputs[0];
    const std::size_t proj_lhs = proj_node.inputs[0];
    const std::size_t proj_rhs = proj_node.inputs[1];
    const std::size_t proj_out_id = proj_node.outputs[0];
    if (attn_out_id >= tensors_.size() || proj_rhs >= tensors_.size() || proj_out_id >= tensors_.size()) {
      set_reason("tensor_out_of_range");
      return false;
    }
    if (proj_lhs != attn_out_id) {
      set_reason("matmul_input_not_attention_output");
      return false;
    }
    if (attn_out_id >= consumer_counts.size() || consumer_counts[attn_out_id] != 1) {
      set_reason("attention_output_multi_consumer");
      return false;
    }

    const GraphTensorValue& q_tensor = tensors_[attn_node.inputs[0]];
    const GraphTensorValue& k_tensor = tensors_[attn_node.inputs[1]];
    const GraphTensorValue& v_tensor = tensors_[attn_node.inputs[2]];
    const GraphTensorValue& attn_out_tensor = tensors_[attn_out_id];
    const GraphTensorValue& proj_weight_tensor = tensors_[proj_rhs];
    const GraphTensorValue& proj_out_tensor = tensors_[proj_out_id];
    if (q_tensor.spec.shape.size() != 2 ||
        k_tensor.spec.shape.size() != 2 ||
        v_tensor.spec.shape.size() != 2 ||
        attn_out_tensor.spec.shape.size() != 2 ||
        proj_weight_tensor.spec.shape.size() != 2 ||
        proj_out_tensor.spec.shape.size() != 2) {
      set_reason("tensor_rank_mismatch");
      return false;
    }
    if (q_tensor.spec.shape != k_tensor.spec.shape ||
        q_tensor.spec.shape != v_tensor.spec.shape ||
        q_tensor.spec.shape != attn_out_tensor.spec.shape) {
      set_reason("attention_shape_mismatch");
      return false;
    }
    if (proj_weight_tensor.spec.shape[0] != attn_out_tensor.spec.shape[1] ||
        proj_out_tensor.spec.shape[0] != attn_out_tensor.spec.shape[0] ||
        proj_out_tensor.spec.shape[1] != proj_weight_tensor.spec.shape[1]) {
      set_reason("projection_shape_mismatch");
      return false;
    }
    if (q_tensor.spec.dtype != k_tensor.spec.dtype ||
        q_tensor.spec.dtype != v_tensor.spec.dtype ||
        q_tensor.spec.dtype != attn_out_tensor.spec.dtype ||
        q_tensor.spec.dtype != proj_weight_tensor.spec.dtype ||
        q_tensor.spec.dtype != proj_out_tensor.spec.dtype) {
      set_reason("dtype_mismatch");
      return false;
    }
    if (q_tensor.spec.layout != Layout::kContiguous ||
        k_tensor.spec.layout != Layout::kContiguous ||
        v_tensor.spec.layout != Layout::kContiguous ||
        attn_out_tensor.spec.layout != Layout::kContiguous ||
        proj_weight_tensor.spec.layout != Layout::kContiguous ||
        proj_out_tensor.spec.layout != Layout::kContiguous) {
      set_reason("layout_not_contiguous");
      return false;
    }
    if (!proj_node.control_deps.empty()) {
      set_reason("projection_control_dependency_present");
      return false;
    }
    set_reason("eligible");
    return true;
  }

  static runtime::Status chooseDeviceForSchema(
      const OperatorSchema& schema,
      runtime::Device preferred_device,
      runtime::Device* out_assigned_device,
      bool* out_fallback,
      ReasonCode* out_fallback_reason = nullptr) {
    if (out_assigned_device == nullptr || out_fallback == nullptr) {
      return runtime::Status::kInvalidValue;
    }
    auto choose_if_supported = [&](runtime::Device device) -> bool {
      return schemaSupportsAvailableDevice(schema, device);
    };

    runtime::Device assigned = preferred_device;
    const bool preferred_declared = schema.supportsDevice(preferred_device);
    const bool preferred_available = choose_if_supported(preferred_device);
    ReasonCode fallback_reason = ReasonCode::kNone;
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

    if (assigned != preferred_device) {
      if (!preferred_declared) {
        fallback_reason = ReasonCode::kFallbackPreferredUnsupported;
      } else if (!preferred_available) {
        fallback_reason = ReasonCode::kFallbackPreferredUnavailable;
      } else {
        fallback_reason = ReasonCode::kFallbackPlannerOptimized;
      }
    }

    *out_assigned_device = assigned;
    *out_fallback = (assigned != preferred_device);
    if (out_fallback_reason != nullptr) {
      *out_fallback_reason = fallback_reason;
    }
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

    if (node.op == OpKind::kQkvPackRepeat) {
      if (node.outputs.size() != 3) {
        return runtime::Status::kInvalidValue;
      }
    } else if (node.outputs.size() != 1) {
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
      case OpKind::kConv2dNchw3x3:
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
        const auto attr_or = [&](const char* key, std::int64_t defv) -> std::int64_t {
          auto it = node.attributes.find(key);
          return it == node.attributes.end() ? defv : it->second;
        };
        const std::int64_t stride_h_i64 =
            (node.op == OpKind::kConv2dNchw3x3s1p1) ? 1 : attr_or("stride_h", 1);
        const std::int64_t stride_w_i64 =
            (node.op == OpKind::kConv2dNchw3x3s1p1) ? 1 : attr_or("stride_w", 1);
        const std::int64_t pad_h_i64 =
            (node.op == OpKind::kConv2dNchw3x3s1p1) ? 1 : attr_or("pad_h", 0);
        const std::int64_t pad_w_i64 =
            (node.op == OpKind::kConv2dNchw3x3s1p1) ? 1 : attr_or("pad_w", 0);
        const bool apply_relu = (node.op == OpKind::kConv2dNchw3x3) && (attr_or("apply_relu", 0) != 0);
        if (stride_h_i64 <= 0 || stride_w_i64 <= 0 || pad_h_i64 < 0 || pad_w_i64 < 0) {
          return runtime::Status::kInvalidValue;
        }
        const std::size_t stride_h = static_cast<std::size_t>(stride_h_i64);
        const std::size_t stride_w = static_cast<std::size_t>(stride_w_i64);
        const std::size_t pad_h = static_cast<std::size_t>(pad_h_i64);
        const std::size_t pad_w = static_cast<std::size_t>(pad_w_i64);
        if (in_h + (2 * pad_h) < 3 || in_w + (2 * pad_w) < 3) {
          return runtime::Status::kInvalidValue;
        }
        const std::size_t out_h = (in_h + (2 * pad_h) - 3) / stride_h + 1;
        const std::size_t out_w = (in_w + (2 * pad_w) - 3) / stride_w + 1;
        if (out_tensor.spec.shape[0] != static_cast<std::int64_t>(batch) ||
            out_tensor.spec.shape[1] != static_cast<std::int64_t>(out_channels) ||
            out_tensor.spec.shape[2] != static_cast<std::int64_t>(out_h) ||
            out_tensor.spec.shape[3] != static_cast<std::int64_t>(out_w)) {
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
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            assigned_device,
            apply_relu);
        if (st != runtime::Status::kSuccess) {
          return st;
        }
        return set_output(out_id, std::move(out));
      }
      case OpKind::kQkvPackRepeat: {
        if constexpr (!std::is_same_v<T, float>) {
          return runtime::Status::kNotSupported;
        }
        if (node.inputs.size() != 1 || node.outputs.size() != 3) {
          return runtime::Status::kInvalidValue;
        }
        const std::size_t in_id = node.inputs[0];
        const std::size_t q_id = node.outputs[0];
        const std::size_t k_id = node.outputs[1];
        const std::size_t v_id = node.outputs[2];
        const GraphTensorValue& in_tensor = tensors_[in_id];
        const GraphTensorValue& q_tensor = tensors_[q_id];
        const GraphTensorValue& k_tensor = tensors_[k_id];
        const GraphTensorValue& v_tensor = tensors_[v_id];
        const std::vector<T>* in = get_value(in_id);
        if (in == nullptr) {
          return runtime::Status::kInvalidValue;
        }
        if (in_tensor.spec.shape.size() != 4 ||
            q_tensor.spec.shape.size() != 2 ||
            k_tensor.spec.shape.size() != 2 ||
            v_tensor.spec.shape.size() != 2) {
          return runtime::Status::kInvalidValue;
        }
        if (q_tensor.spec.shape != k_tensor.spec.shape || q_tensor.spec.shape != v_tensor.spec.shape) {
          return runtime::Status::kInvalidValue;
        }
        const std::size_t need = detail::numelFromShapeContract(q_tensor.spec.shape);
        if (need == 0) {
          return runtime::Status::kInvalidValue;
        }
        const std::size_t total = need * 3;
        std::vector<T> packed(total, static_cast<T>(0));
        const std::size_t copied0 = std::min(in->size(), total);
        if (copied0 == 0) {
          return runtime::Status::kInvalidValue;
        }
        std::copy_n(in->data(), copied0, packed.data());
        std::size_t copied = copied0;
        while (copied < total) {
          const std::size_t chunk = std::min(copied, total - copied);
          std::copy_n(packed.data(), chunk, packed.data() + copied);
          copied += chunk;
        }
        std::vector<T> q(need, static_cast<T>(0));
        std::vector<T> k(need, static_cast<T>(0));
        std::vector<T> v(need, static_cast<T>(0));
        std::copy_n(packed.data(), need, q.data());
        std::copy_n(packed.data() + need, need, k.data());
        std::copy_n(packed.data() + (2 * need), need, v.data());
        values->insert_or_assign(q_id, std::move(q));
        values->insert_or_assign(k_id, std::move(k));
        values->insert_or_assign(v_id, std::move(v));
        return runtime::Status::kSuccess;
      }
      case OpKind::kRelu: {
        if (node.inputs.size() != 1) {
          return runtime::Status::kInvalidValue;
        }
        const std::size_t in_id = node.inputs[0];
        const GraphTensorValue& in_tensor = tensors_[in_id];
        const std::vector<T>* in = get_value(in_id);
        if (in == nullptr) {
          return runtime::Status::kInvalidValue;
        }
        const std::size_t in_numel = detail::numelFromShapeContract(in_tensor.spec.shape);
        const std::size_t out_numel = detail::numelFromShapeContract(out_tensor.spec.shape);
        if (in_numel == 0 ||
            in_numel != out_numel ||
            in_tensor.spec.shape != out_tensor.spec.shape ||
            in_tensor.spec.layout != out_tensor.spec.layout) {
          return runtime::Status::kInvalidValue;
        }
        std::vector<T> out(out_numel, static_cast<T>(0));
        for (std::size_t i = 0; i < out_numel; ++i) {
          const T v = (*in)[i];
          out[i] = (v < static_cast<T>(0)) ? static_cast<T>(0) : v;
        }
        return set_output(out_id, std::move(out));
      }
      default:
        return runtime::Status::kNotSupported;
    }
  }

  template <typename T>
  runtime::Status executeConvReluFusedNodeTyped(
      const GraphNode& conv_node,
      const GraphNode& relu_node,
      runtime::Device assigned_device,
      const std::vector<std::size_t>& consumer_counts,
      std::unordered_map<std::size_t, std::vector<T>>* values) const {
    if (values == nullptr) {
      return runtime::Status::kInvalidValue;
    }
    std::string fuse_reason;
    if (!canFuseConvReluPair(conv_node, relu_node, consumer_counts, &fuse_reason)) {
      return runtime::Status::kInvalidValue;
    }

    auto get_value = [&](std::size_t tensor_id) -> const std::vector<T>* {
      auto it = values->find(tensor_id);
      if (it == values->end()) {
        return nullptr;
      }
      return &it->second;
    };

    const std::size_t x_id = conv_node.inputs[0];
    const std::size_t w_id = conv_node.inputs[1];
    const std::size_t bias_id = (conv_node.inputs.size() == 3) ? conv_node.inputs[2] : static_cast<std::size_t>(-1);
    const std::size_t conv_out_id = conv_node.outputs[0];
    const std::size_t relu_out_id = relu_node.outputs[0];

    if (x_id >= tensors_.size() ||
        w_id >= tensors_.size() ||
        conv_out_id >= tensors_.size() ||
        relu_out_id >= tensors_.size()) {
      return runtime::Status::kInvalidValue;
    }
    if (bias_id != static_cast<std::size_t>(-1) && bias_id >= tensors_.size()) {
      return runtime::Status::kInvalidValue;
    }

    const GraphTensorValue& x_tensor = tensors_[x_id];
    const GraphTensorValue& w_tensor = tensors_[w_id];
    const GraphTensorValue& conv_mid_tensor = tensors_[conv_out_id];
    const GraphTensorValue& out_tensor = tensors_[relu_out_id];

    const std::vector<T>* x = get_value(x_id);
    const std::vector<T>* w = get_value(w_id);
    if (x == nullptr || w == nullptr) {
      return runtime::Status::kInvalidValue;
    }
    const std::vector<T>* bias = nullptr;
    const GraphTensorValue* bias_tensor = nullptr;
    if (bias_id != static_cast<std::size_t>(-1)) {
      bias_tensor = &tensors_[bias_id];
      bias = get_value(bias_id);
      if (bias == nullptr) {
        return runtime::Status::kInvalidValue;
      }
    }

    if (x_tensor.spec.shape.size() != 4 ||
        w_tensor.spec.shape.size() != 4 ||
        conv_mid_tensor.spec.shape.size() != 4 ||
        out_tensor.spec.shape.size() != 4) {
      return runtime::Status::kInvalidValue;
    }
    if (conv_mid_tensor.spec.shape != out_tensor.spec.shape) {
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
        /*apply_relu=*/true);
    if (st != runtime::Status::kSuccess) {
      return st;
    }

    values->insert_or_assign(relu_out_id, out);
    values->insert_or_assign(conv_out_id, std::move(out));
    return runtime::Status::kSuccess;
  }

  template <typename T>
  runtime::Status executeMatMulBiasReluFusedNodeTyped(
      const GraphNode& matmul_node,
      const GraphNode& bias_node,
      const GraphNode& relu_node,
      runtime::Device assigned_device,
      const std::vector<std::size_t>& consumer_counts,
      std::unordered_map<std::size_t, std::vector<T>>* values) const {
    if (values == nullptr) {
      return runtime::Status::kInvalidValue;
    }
    std::string fuse_reason;
    if (!canFuseMatMulBiasReluTriplet(matmul_node, bias_node, relu_node, consumer_counts, &fuse_reason)) {
      return runtime::Status::kInvalidValue;
    }

    auto get_value = [&](std::size_t tensor_id) -> const std::vector<T>* {
      auto it = values->find(tensor_id);
      if (it == values->end()) {
        return nullptr;
      }
      return &it->second;
    };

    const std::size_t lhs_id = matmul_node.inputs[0];
    const std::size_t rhs_id = matmul_node.inputs[1];
    const std::size_t mm_out_id = matmul_node.outputs[0];
    const std::size_t add_a_id = bias_node.inputs[0];
    const std::size_t add_b_id = bias_node.inputs[1];
    const std::size_t vec_out_id = bias_node.outputs[0];
    const std::size_t relu_out_id = relu_node.outputs[0];
    const std::size_t bias_id = (add_a_id == mm_out_id) ? add_b_id : add_a_id;

    if (lhs_id >= tensors_.size() ||
        rhs_id >= tensors_.size() ||
        bias_id >= tensors_.size() ||
        mm_out_id >= tensors_.size() ||
        vec_out_id >= tensors_.size() ||
        relu_out_id >= tensors_.size()) {
      return runtime::Status::kInvalidValue;
    }

    const GraphTensorValue& lhs_tensor = tensors_[lhs_id];
    const GraphTensorValue& rhs_tensor = tensors_[rhs_id];
    const GraphTensorValue& mm_out_tensor = tensors_[mm_out_id];
    const GraphTensorValue& bias_tensor = tensors_[bias_id];
    const GraphTensorValue& vec_out_tensor = tensors_[vec_out_id];
    const GraphTensorValue& relu_out_tensor = tensors_[relu_out_id];

    if (lhs_tensor.spec.shape.size() != 2 ||
        rhs_tensor.spec.shape.size() != 2 ||
        mm_out_tensor.spec.shape.size() != 2 ||
        vec_out_tensor.spec.shape.size() != 2 ||
        relu_out_tensor.spec.shape.size() != 2 ||
        bias_tensor.spec.shape.size() != 2) {
      return runtime::Status::kInvalidValue;
    }

    const std::size_t m = static_cast<std::size_t>(lhs_tensor.spec.shape[0]);
    const std::size_t k = static_cast<std::size_t>(lhs_tensor.spec.shape[1]);
    const std::size_t rhs_k = static_cast<std::size_t>(rhs_tensor.spec.shape[0]);
    const std::size_t n = static_cast<std::size_t>(rhs_tensor.spec.shape[1]);
    if (k != rhs_k) {
      return runtime::Status::kInvalidValue;
    }
    if (mm_out_tensor.spec.shape[0] != static_cast<std::int64_t>(m) ||
        mm_out_tensor.spec.shape[1] != static_cast<std::int64_t>(n) ||
        vec_out_tensor.spec.shape != mm_out_tensor.spec.shape ||
        relu_out_tensor.spec.shape != mm_out_tensor.spec.shape ||
        bias_tensor.spec.shape != mm_out_tensor.spec.shape) {
      return runtime::Status::kInvalidValue;
    }

    const std::vector<T>* lhs = get_value(lhs_id);
    const std::vector<T>* rhs = get_value(rhs_id);
    const std::vector<T>* bias = get_value(bias_id);
    if (lhs == nullptr || rhs == nullptr || bias == nullptr) {
      return runtime::Status::kInvalidValue;
    }

    const std::size_t out_numel = detail::numelFromShapeContract(mm_out_tensor.spec.shape);
    if (lhs->size() != m * k || rhs->size() != k * n || bias->size() != out_numel) {
      return runtime::Status::kInvalidValue;
    }

    std::vector<T> mm_out(out_numel, static_cast<T>(0));
    runtime::Status st = ops::matMul<T>(lhs->data(), rhs->data(), mm_out.data(), m, k, n, assigned_device);
    if (st != runtime::Status::kSuccess) {
      return st;
    }

    std::vector<T> add_out(out_numel, static_cast<T>(0));
    std::vector<T> relu_out(out_numel, static_cast<T>(0));
    for (std::size_t i = 0; i < out_numel; ++i) {
      const T summed = mm_out[i] + (*bias)[i];
      add_out[i] = summed;
      relu_out[i] = (summed < static_cast<T>(0)) ? static_cast<T>(0) : summed;
    }

    values->insert_or_assign(mm_out_id, mm_out);
    values->insert_or_assign(vec_out_id, add_out);
    values->insert_or_assign(relu_out_id, std::move(relu_out));
    return runtime::Status::kSuccess;
  }

  template <typename T>
  runtime::Status executeAttentionProjFusedNodeTyped(
      const GraphNode& attn_node,
      const GraphNode& proj_node,
      runtime::Device assigned_device,
      const std::vector<std::size_t>& consumer_counts,
      std::unordered_map<std::size_t, std::vector<T>>* values) const {
    if constexpr (!std::is_same_v<T, float>) {
      return runtime::Status::kNotSupported;
    }
    if (values == nullptr) {
      return runtime::Status::kInvalidValue;
    }
    std::string fuse_reason;
    if (!canFuseAttentionProjPair(attn_node, proj_node, consumer_counts, &fuse_reason)) {
      return runtime::Status::kInvalidValue;
    }

    auto get_value = [&](std::size_t tensor_id) -> const std::vector<T>* {
      auto it = values->find(tensor_id);
      if (it == values->end()) {
        return nullptr;
      }
      return &it->second;
    };

    const std::size_t q_id = attn_node.inputs[0];
    const std::size_t k_id = attn_node.inputs[1];
    const std::size_t v_id = attn_node.inputs[2];
    const std::size_t attn_out_id = attn_node.outputs[0];
    const std::size_t proj_w_id = proj_node.inputs[1];
    const std::size_t proj_out_id = proj_node.outputs[0];
    if (q_id >= tensors_.size() ||
        k_id >= tensors_.size() ||
        v_id >= tensors_.size() ||
        attn_out_id >= tensors_.size() ||
        proj_w_id >= tensors_.size() ||
        proj_out_id >= tensors_.size()) {
      return runtime::Status::kInvalidValue;
    }

    const GraphTensorValue& q_tensor = tensors_[q_id];
    const GraphTensorValue& attn_out_tensor = tensors_[attn_out_id];
    const GraphTensorValue& proj_weight_tensor = tensors_[proj_w_id];
    const GraphTensorValue& proj_out_tensor = tensors_[proj_out_id];
    if (q_tensor.spec.shape.size() != 2 ||
        attn_out_tensor.spec.shape.size() != 2 ||
        proj_weight_tensor.spec.shape.size() != 2 ||
        proj_out_tensor.spec.shape.size() != 2) {
      return runtime::Status::kInvalidValue;
    }

    const std::vector<T>* q = get_value(q_id);
    const std::vector<T>* k = get_value(k_id);
    const std::vector<T>* v = get_value(v_id);
    const std::vector<T>* proj_w = get_value(proj_w_id);
    if (q == nullptr || k == nullptr || v == nullptr || proj_w == nullptr) {
      return runtime::Status::kInvalidValue;
    }

    const std::size_t seq = static_cast<std::size_t>(q_tensor.spec.shape[0]);
    const std::size_t head_dim = static_cast<std::size_t>(q_tensor.spec.shape[1]);
    const std::size_t proj_out_cols = static_cast<std::size_t>(proj_weight_tensor.spec.shape[1]);
    const std::size_t attn_numel = detail::numelFromShapeContract(attn_out_tensor.spec.shape);
    const std::size_t proj_numel = detail::numelFromShapeContract(proj_out_tensor.spec.shape);
    if (q->size() != attn_numel ||
        k->size() != attn_numel ||
        v->size() != attn_numel ||
        proj_w->size() != (head_dim * proj_out_cols) ||
        proj_numel != (seq * proj_out_cols)) {
      return runtime::Status::kInvalidValue;
    }

    std::vector<T> attn_out(attn_numel, static_cast<T>(0));
    std::vector<T> proj_out(proj_numel, static_cast<T>(0));
    AttentionConfig cfg{seq, head_dim, false};
    runtime::Status st = attentionForward(
        reinterpret_cast<const float*>(q->data()),
        reinterpret_cast<const float*>(k->data()),
        reinterpret_cast<const float*>(v->data()),
        reinterpret_cast<float*>(attn_out.data()),
        cfg,
        assigned_device);
    if (st != runtime::Status::kSuccess) {
      return st;
    }
    st = ops::matMul<T>(
        attn_out.data(),
        proj_w->data(),
        proj_out.data(),
        seq,
        head_dim,
        proj_out_cols,
        assigned_device);
    if (st != runtime::Status::kSuccess) {
      return st;
    }

    values->insert_or_assign(attn_out_id, attn_out);
    values->insert_or_assign(proj_out_id, std::move(proj_out));
    return runtime::Status::kSuccess;
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
    const std::vector<std::size_t> consumer_counts = buildTensorConsumerCounts();

    for (const auto& group : groups) {
      if (group.sync_boundary_before) {
        st = runtime::deviceSynchronizeWithPolicy(options.sync_policy);
        if (st != runtime::Status::kSuccess) {
          return st;
        }
      }

      for (std::size_t gi = 0; gi < group.node_ids.size(); ++gi) {
        const std::size_t node_id = group.node_ids[gi];
        if (node_id >= nodes_.size()) {
          return runtime::Status::kInvalidValue;
        }
        if (visited_nodes.find(node_id) != visited_nodes.end()) {
          return runtime::Status::kInvalidValue;
        }
        const GraphNode& node = nodes_[node_id];
        if (options.enable_fusion_v1 && node.op == OpKind::kConv2dNchw3x3s1p1 && gi + 1 < group.node_ids.size()) {
          const std::size_t next_node_id = group.node_ids[gi + 1];
          if (next_node_id < nodes_.size()) {
            const GraphNode& next_node = nodes_[next_node_id];
            std::string fuse_reason;
            if (canFuseConvReluPair(node, next_node, consumer_counts, &fuse_reason)) {
              bool allow_fusion = true;
              if (options.enable_fusion_cost_model_v1) {
                const FusionCostEstimate estimate = estimateConvReluCost(
                    options, tensors_[node.inputs[0]], tensors_[node.inputs[1]], tensors_[node.outputs[0]]);
                allow_fusion = std::isfinite(estimate.speedup) && estimate.speedup >= options.fusion_cost_min_speedup;
              }
              if (allow_fusion) {
                if (visited_nodes.find(next_node_id) != visited_nodes.end()) {
                  return runtime::Status::kInvalidValue;
                }
                st = executeConvReluFusedNodeTyped<T>(
                    node, next_node, group.assigned_device, consumer_counts, &values);
                if (st != runtime::Status::kSuccess) {
                  return st;
                }
                visited_nodes.insert(node_id);
                visited_nodes.insert(next_node_id);
                gi += 1;
                continue;
              }
            }
          }
        }

        if (options.enable_fusion_v1 && node.op == OpKind::kMatMul && gi + 2 < group.node_ids.size()) {
          const std::size_t next_node_id = group.node_ids[gi + 1];
          const std::size_t next2_node_id = group.node_ids[gi + 2];
          if (next_node_id < nodes_.size() && next2_node_id < nodes_.size()) {
            const GraphNode& next_node = nodes_[next_node_id];
            const GraphNode& next2_node = nodes_[next2_node_id];
            std::string fuse_reason;
            if (canFuseMatMulBiasReluTriplet(node, next_node, next2_node, consumer_counts, &fuse_reason)) {
              bool allow_fusion = true;
              if (options.enable_fusion_cost_model_v1) {
                const FusionCostEstimate estimate = estimateMatMulBiasReluCost(
                    options,
                    tensors_[node.inputs[0]],
                    tensors_[node.inputs[1]],
                    tensors_[next_node.outputs[0]],
                    tensors_[next2_node.outputs[0]]);
                allow_fusion = std::isfinite(estimate.speedup) && estimate.speedup >= options.fusion_cost_min_speedup;
              }
              if (allow_fusion) {
                if (visited_nodes.find(next_node_id) != visited_nodes.end() ||
                    visited_nodes.find(next2_node_id) != visited_nodes.end()) {
                  return runtime::Status::kInvalidValue;
                }
                st = executeMatMulBiasReluFusedNodeTyped<T>(
                    node, next_node, next2_node, group.assigned_device, consumer_counts, &values);
                if (st != runtime::Status::kSuccess) {
                  return st;
                }
                visited_nodes.insert(node_id);
                visited_nodes.insert(next_node_id);
                visited_nodes.insert(next2_node_id);
                gi += 2;
                continue;
              }
            }
          }
        }

        if (options.enable_fusion_v1 && node.op == OpKind::kAttentionForward && gi + 1 < group.node_ids.size()) {
          const std::size_t next_node_id = group.node_ids[gi + 1];
          if (next_node_id < nodes_.size()) {
            const GraphNode& next_node = nodes_[next_node_id];
            std::string fuse_reason;
            if (canFuseAttentionProjPair(node, next_node, consumer_counts, &fuse_reason)) {
              bool allow_fusion = true;
              if (options.enable_fusion_cost_model_v1) {
                const FusionCostEstimate estimate = estimateAttentionProjCost(
                    options,
                    tensors_[node.inputs[0]],
                    tensors_[node.outputs[0]],
                    tensors_[next_node.inputs[1]],
                    tensors_[next_node.outputs[0]]);
                allow_fusion = std::isfinite(estimate.speedup) && estimate.speedup >= options.fusion_cost_min_speedup;
              }
              if (allow_fusion) {
                if (visited_nodes.find(next_node_id) != visited_nodes.end()) {
                  return runtime::Status::kInvalidValue;
                }
                st = executeAttentionProjFusedNodeTyped<T>(
                    node, next_node, group.assigned_device, consumer_counts, &values);
                if (st != runtime::Status::kSuccess) {
                  return st;
                }
                visited_nodes.insert(node_id);
                visited_nodes.insert(next_node_id);
                gi += 1;
                continue;
              }
            }
          }
        }

        st = executeNodeTyped<T>(node, group.assigned_device, &values);
        if (st != runtime::Status::kSuccess) {
          return st;
        }
        visited_nodes.insert(node_id);
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
  std::size_t graph_revision_{0};
  mutable PlanCacheEntry plan_cache_{};
  mutable bool last_plan_cache_hit_{false};
  mutable std::size_t plan_cache_stats_hits_{0};
  mutable std::size_t plan_cache_stats_misses_{0};
};

}  // namespace lightning_core::graph
