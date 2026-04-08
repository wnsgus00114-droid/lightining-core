#include "bind_common.hpp"

#include <unordered_map>

namespace {

lc::graph::OpKind parseOpKind(const std::string& op) {
  if (op == "matmul") {
    return lc::graph::OpKind::kMatMul;
  }
  if (op == "vector_add") {
    return lc::graph::OpKind::kVectorAdd;
  }
  if (op == "matrix_sub") {
    return lc::graph::OpKind::kMatrixSub;
  }
  if (op == "matrix_div") {
    return lc::graph::OpKind::kMatrixDiv;
  }
  if (op == "attention_forward") {
    return lc::graph::OpKind::kAttentionForward;
  }
  if (op == "conv2d_nchw3x3s1p1") {
    return lc::graph::OpKind::kConv2dNchw3x3s1p1;
  }
  if (op == "relu") {
    return lc::graph::OpKind::kRelu;
  }
  throw std::invalid_argument(
      "op must be one of: matmul, vector_add, matrix_sub, matrix_div, attention_forward, conv2d_nchw3x3s1p1, relu");
}

const char* toOpString(lc::graph::OpKind op) {
  return lc::graph::opKindName(op);
}

lc::graph::DType parseDType(const std::string& dtype) {
  if (dtype == "float32") {
    return lc::graph::DType::kFloat32;
  }
  if (dtype == "float64") {
    return lc::graph::DType::kFloat64;
  }
  throw std::invalid_argument("dtype must be 'float32' or 'float64'");
}

lc::Layout parseLayout(const std::string& layout) {
  if (layout == "contiguous") {
    return lc::Layout::kContiguous;
  }
  if (layout == "strided") {
    return lc::Layout::kStrided;
  }
  throw std::invalid_argument("layout must be 'contiguous' or 'strided'");
}

lc::runtime::SyncMode parseSyncMode(const std::string& mode) {
  if (mode == "auto") {
    return lc::runtime::SyncMode::kAuto;
  }
  if (mode == "always") {
    return lc::runtime::SyncMode::kAlways;
  }
  if (mode == "never") {
    return lc::runtime::SyncMode::kNever;
  }
  throw std::invalid_argument("sync_mode must be 'auto', 'always', or 'never'");
}

const char* toLayoutString(lc::Layout layout) {
  return layout == lc::Layout::kContiguous ? "contiguous" : "strided";
}

py::dict toSchemaDict(const lc::graph::OperatorSchema& schema) {
  py::dict out;
  out["op"] = toOpString(schema.kind);
  out["name"] = schema.name;
  out["min_inputs"] = schema.min_inputs;
  out["max_inputs"] = schema.max_inputs;
  out["min_outputs"] = schema.min_outputs;
  out["max_outputs"] = schema.max_outputs;
  out["supports_cpu"] = schema.supports_cpu;
  out["supports_cuda"] = schema.supports_cuda;
  out["supports_metal"] = schema.supports_metal;
  return out;
}

py::dict toTensorValueDict(const lc::graph::GraphTensorValue& t) {
  py::dict out;
  out["name"] = t.name;
  out["constant"] = t.constant;
  out["shape"] = t.spec.shape;
  out["dtype"] = lc::graph::dtypeName(t.spec.dtype);
  out["layout"] = toLayoutString(t.spec.layout);
  return out;
}

py::dict toNodeDict(const lc::graph::GraphNode& node) {
  py::dict out;
  out["id"] = node.id;
  out["op"] = toOpString(node.op);
  out["name"] = node.name;
  out["inputs"] = node.inputs;
  out["outputs"] = node.outputs;
  out["control_deps"] = node.control_deps;
  return out;
}

py::dict toValidationIssueDict(const lc::graph::ValidationIssue& issue) {
  py::dict out;
  out["pass"] = lc::graph::validationPassName(issue.pass);
  out["status"] = lc::runtime::getErrorString(issue.status);
  out["node_id"] = issue.node_id;
  out["tensor_id"] = issue.tensor_id;
  out["message"] = issue.message;
  return out;
}

py::list toPlanStepsList(const std::vector<lc::graph::GraphPlanStep>& steps) {
  py::list out;
  for (const auto& step : steps) {
    py::dict row;
    row["node_id"] = step.node_id;
    row["op"] = toOpString(step.op);
    row["assigned_device"] = toString(step.assigned_device);
    row["fallback"] = step.fallback;
    out.append(row);
  }
  return out;
}

py::list toExecutionGroupsList(const std::vector<lc::graph::GraphExecutionGroup>& groups) {
  py::list out;
  for (const auto& group : groups) {
    py::dict row;
    row["group_id"] = group.group_id;
    row["assigned_device"] = toString(group.assigned_device);
    row["fallback"] = group.fallback;
    row["sync_boundary_before"] = group.sync_boundary_before;
    row["sync_boundary_after"] = group.sync_boundary_after;
    row["node_ids"] = group.node_ids;
    out.append(row);
  }
  return out;
}

py::list toFusionDecisionList(const std::vector<lc::graph::GraphFusionDecision>& decisions) {
  py::list out;
  for (const auto& d : decisions) {
    py::dict row;
    row["pattern"] = lc::graph::fusionPatternName(d.pattern);
    row["start_node_id"] = d.start_node_id;
    row["end_node_id"] = d.end_node_id;
    row["assigned_device"] = toString(d.assigned_device);
    row["fused"] = d.fused;
    row["cost_model_applied"] = d.cost_model_applied;
    row["estimated_unfused_cost_ns"] = d.estimated_unfused_cost_ns;
    row["estimated_fused_cost_ns"] = d.estimated_fused_cost_ns;
    row["estimated_speedup"] = d.estimated_speedup;
    row["reason"] = d.reason;
    out.append(row);
  }
  return out;
}

lc::graph::GraphPlannerOptions parseGraphPlannerOptions(
    const std::string& preferred_device,
    const std::string& sync_mode,
    bool trace_sync_boundary,
    bool group_by_backend_capability,
    bool separate_fallback_segments,
    bool insert_sync_on_device_change,
    bool enable_fusion_v1,
    bool enable_fusion_cost_model_v1,
    double fusion_cost_min_speedup) {
  lc::graph::GraphPlannerOptions options;
  options.preferred_device = parseDevice(preferred_device);
  options.sync_policy.mode = parseSyncMode(sync_mode);
  options.sync_policy.trace_sync_boundary = trace_sync_boundary;
  options.group_by_backend_capability = group_by_backend_capability;
  options.separate_fallback_segments = separate_fallback_segments;
  options.insert_sync_on_device_change = insert_sync_on_device_change;
  options.enable_fusion_v1 = enable_fusion_v1;
  options.enable_fusion_cost_model_v1 = enable_fusion_cost_model_v1;
  options.fusion_cost_min_speedup = fusion_cost_min_speedup;
  return options;
}

template <typename T>
std::unordered_map<std::size_t, std::vector<T>> parseValueFeedDict(const py::dict& feed_dict) {
  std::unordered_map<std::size_t, std::vector<T>> feeds;
  feeds.reserve(feed_dict.size());
  for (const auto& item : feed_dict) {
    const std::size_t tensor_id = py::cast<std::size_t>(item.first);
    py::array_t<T, py::array::c_style | py::array::forcecast> arr =
        py::cast<py::array_t<T, py::array::c_style | py::array::forcecast>>(item.second);
    feeds[tensor_id] = toVector<T>(arr);
  }
  return feeds;
}

template <typename T>
py::dict toValueDict(const std::unordered_map<std::size_t, std::vector<T>>& values) {
  py::dict out;
  for (const auto& kv : values) {
    out[py::int_(kv.first)] = toNumpy<T>(kv.second);
  }
  return out;
}

}  // namespace

void bindGraph(py::module_& m) {
  m.def("graph_validation_passes", [] {
    py::list out;
    out.append(lc::graph::validationPassName(lc::graph::ValidationPass::kTensorSpec));
    out.append(lc::graph::validationPassName(lc::graph::ValidationPass::kSchemaArity));
    out.append(lc::graph::validationPassName(lc::graph::ValidationPass::kTensorReference));
    out.append(lc::graph::validationPassName(lc::graph::ValidationPass::kControlDependency));
    out.append(lc::graph::validationPassName(lc::graph::ValidationPass::kBackendCapability));
    return out;
  });

  m.def("graph_registry_size", [] {
    return lc::graph::globalOperatorRegistry().size();
  });

  m.def("graph_registry_schemas", [] {
    py::list out;
    const auto schemas = lc::graph::globalOperatorRegistry().schemas();
    for (const auto& schema : schemas) {
      out.append(toSchemaDict(schema));
    }
    return out;
  });

  m.def("graph_registry_schema", [](const std::string& op) {
    const lc::graph::OperatorSchema* schema = lc::graph::globalOperatorRegistry().find(parseOpKind(op));
    if (schema == nullptr) {
      throw std::runtime_error("operator schema not found");
    }
    return toSchemaDict(*schema);
  });

  py::class_<lc::graph::GraphIR>(m, "GraphIR")
      .def(py::init<>())
      .def("add_tensor",
           [](lc::graph::GraphIR& g,
              const std::vector<std::int64_t>& shape,
              const std::string& dtype,
              const std::string& layout,
              const std::string& name,
              bool constant) {
             lc::graph::TensorSpec spec;
             spec.shape = shape;
             spec.dtype = parseDType(dtype);
             spec.layout = parseLayout(layout);
             std::size_t id = 0;
             throwIfNotSuccess(g.addTensorSpec(spec, &id, name, constant));
             return id;
           },
           py::arg("shape"),
           py::arg("dtype") = "float32",
           py::arg("layout") = "contiguous",
           py::arg("name") = "",
           py::arg("constant") = false)
      .def("add_node",
           [](lc::graph::GraphIR& g,
              const std::string& op,
              const std::vector<std::size_t>& inputs,
              const std::vector<std::size_t>& outputs,
              const std::vector<std::size_t>& control_deps,
              const std::string& name) {
             throwIfNotSuccess(g.addNode(parseOpKind(op), inputs, outputs, control_deps, name));
           },
           py::arg("op"),
           py::arg("inputs"),
           py::arg("outputs"),
           py::arg("control_deps") = std::vector<std::size_t>{},
           py::arg("name") = "")
      .def("validate", [](const lc::graph::GraphIR& g) { throwIfNotSuccess(g.validate()); })
      .def("validate_report", [](const lc::graph::GraphIR& g) {
        lc::graph::ValidationReport report;
        const lc::runtime::Status st = g.validateWithReport(&report);
        py::dict out;
        out["ok"] = (st == lc::runtime::Status::kSuccess);
        out["status"] = lc::runtime::getErrorString(st);
        py::list issues;
        for (const auto& issue : report.issues) {
          issues.append(toValidationIssueDict(issue));
        }
        out["issues"] = issues;
        return out;
      })
      .def("num_tensors", &lc::graph::GraphIR::numTensors)
      .def("num_nodes", &lc::graph::GraphIR::numNodes)
      .def("tensors", [](const lc::graph::GraphIR& g) {
        py::list out;
        for (const auto& t : g.tensors()) {
          out.append(toTensorValueDict(t));
        }
        return out;
      })
      .def("nodes", [](const lc::graph::GraphIR& g) {
        py::list out;
        for (const auto& node : g.nodes()) {
          out.append(toNodeDict(node));
        }
        return out;
      })
      .def("plan",
           [](const lc::graph::GraphIR& g, const std::string& preferred_device) {
             std::vector<lc::graph::GraphPlanStep> steps;
             throwIfNotSuccess(g.planForDevice(parseDevice(preferred_device), &steps));
             py::list out;
             for (const auto& step : steps) {
               py::dict row;
               row["node_id"] = step.node_id;
               row["op"] = toOpString(step.op);
               row["assigned_device"] = toString(step.assigned_device);
               row["fallback"] = step.fallback;
               out.append(row);
             }
             return out;
           },
           py::arg("preferred_device") = "metal")
      .def("plan_groups",
           [](const lc::graph::GraphIR& g,
              const std::string& preferred_device,
              const std::string& sync_mode,
              bool trace_sync_boundary,
              bool group_by_backend_capability,
              bool separate_fallback_segments,
              bool insert_sync_on_device_change,
              bool enable_fusion_v1,
              bool enable_fusion_cost_model_v1,
              double fusion_cost_min_speedup) {
             const lc::graph::GraphPlannerOptions options = parseGraphPlannerOptions(
                 preferred_device,
                 sync_mode,
                 trace_sync_boundary,
                 group_by_backend_capability,
                 separate_fallback_segments,
                 insert_sync_on_device_change,
                 enable_fusion_v1,
                 enable_fusion_cost_model_v1,
                 fusion_cost_min_speedup);

             std::vector<lc::graph::GraphExecutionGroup> groups;
             std::vector<lc::graph::GraphPlanStep> steps;
             throwIfNotSuccess(g.planExecutionGroups(options, &groups, &steps));

             py::dict out;
             out["groups"] = toExecutionGroupsList(groups);
             out["steps"] = toPlanStepsList(steps);
             return out;
           },
           py::arg("preferred_device") = "metal",
           py::arg("sync_mode") = "auto",
           py::arg("trace_sync_boundary") = false,
           py::arg("group_by_backend_capability") = true,
           py::arg("separate_fallback_segments") = true,
           py::arg("insert_sync_on_device_change") = true,
           py::arg("enable_fusion_v1") = true,
           py::arg("enable_fusion_cost_model_v1") = true,
           py::arg("fusion_cost_min_speedup") = 1.01)
      .def("fusion_report",
           [](const lc::graph::GraphIR& g,
              const std::string& preferred_device,
              const std::string& sync_mode,
              bool trace_sync_boundary,
              bool group_by_backend_capability,
              bool separate_fallback_segments,
              bool insert_sync_on_device_change,
              bool enable_fusion_v1,
              bool enable_fusion_cost_model_v1,
              double fusion_cost_min_speedup) {
             const lc::graph::GraphPlannerOptions options = parseGraphPlannerOptions(
                 preferred_device,
                 sync_mode,
                 trace_sync_boundary,
                 group_by_backend_capability,
                 separate_fallback_segments,
                 insert_sync_on_device_change,
                 enable_fusion_v1,
                 enable_fusion_cost_model_v1,
                 fusion_cost_min_speedup);
             std::vector<lc::graph::GraphFusionDecision> decisions;
             throwIfNotSuccess(g.fusionReport(options, &decisions));
             return toFusionDecisionList(decisions);
           },
           py::arg("preferred_device") = "metal",
           py::arg("sync_mode") = "auto",
           py::arg("trace_sync_boundary") = false,
           py::arg("group_by_backend_capability") = true,
           py::arg("separate_fallback_segments") = true,
           py::arg("insert_sync_on_device_change") = true,
           py::arg("enable_fusion_v1") = true,
           py::arg("enable_fusion_cost_model_v1") = true,
           py::arg("fusion_cost_min_speedup") = 1.01)
      .def("execute_f32",
           [](const lc::graph::GraphIR& g,
              const py::dict& feeds,
              const std::string& preferred_device,
              const std::string& sync_mode,
              bool trace_sync_boundary,
              bool group_by_backend_capability,
              bool separate_fallback_segments,
              bool insert_sync_on_device_change,
              bool enable_fusion_v1,
              bool enable_fusion_cost_model_v1,
              double fusion_cost_min_speedup) {
             const lc::graph::GraphPlannerOptions options = parseGraphPlannerOptions(
                 preferred_device,
                 sync_mode,
                 trace_sync_boundary,
                 group_by_backend_capability,
                 separate_fallback_segments,
                 insert_sync_on_device_change,
                 enable_fusion_v1,
                 enable_fusion_cost_model_v1,
                 fusion_cost_min_speedup);
             std::unordered_map<std::size_t, std::vector<float>> values;
             std::vector<lc::graph::GraphExecutionGroup> groups;
             std::vector<lc::graph::GraphPlanStep> steps;
             throwIfNotSuccess(g.executeF32(
                 options, parseValueFeedDict<float>(feeds), &values, &groups, &steps));

             py::dict out;
             out["values"] = toValueDict<float>(values);
             out["groups"] = toExecutionGroupsList(groups);
             out["steps"] = toPlanStepsList(steps);
             return out;
           },
           py::arg("feeds"),
           py::arg("preferred_device") = "metal",
           py::arg("sync_mode") = "auto",
           py::arg("trace_sync_boundary") = false,
           py::arg("group_by_backend_capability") = true,
           py::arg("separate_fallback_segments") = true,
           py::arg("insert_sync_on_device_change") = true,
           py::arg("enable_fusion_v1") = true,
           py::arg("enable_fusion_cost_model_v1") = true,
           py::arg("fusion_cost_min_speedup") = 1.01)
      .def("execute_f64",
           [](const lc::graph::GraphIR& g,
              const py::dict& feeds,
              const std::string& preferred_device,
              const std::string& sync_mode,
              bool trace_sync_boundary,
              bool group_by_backend_capability,
              bool separate_fallback_segments,
              bool insert_sync_on_device_change,
              bool enable_fusion_v1,
              bool enable_fusion_cost_model_v1,
              double fusion_cost_min_speedup) {
             const lc::graph::GraphPlannerOptions options = parseGraphPlannerOptions(
                 preferred_device,
                 sync_mode,
                 trace_sync_boundary,
                 group_by_backend_capability,
                 separate_fallback_segments,
                 insert_sync_on_device_change,
                 enable_fusion_v1,
                 enable_fusion_cost_model_v1,
                 fusion_cost_min_speedup);
             std::unordered_map<std::size_t, std::vector<double>> values;
             std::vector<lc::graph::GraphExecutionGroup> groups;
             std::vector<lc::graph::GraphPlanStep> steps;
             throwIfNotSuccess(g.executeF64(
                 options, parseValueFeedDict<double>(feeds), &values, &groups, &steps));

             py::dict out;
             out["values"] = toValueDict<double>(values);
             out["groups"] = toExecutionGroupsList(groups);
             out["steps"] = toPlanStepsList(steps);
             return out;
           },
           py::arg("feeds"),
           py::arg("preferred_device") = "metal",
           py::arg("sync_mode") = "auto",
           py::arg("trace_sync_boundary") = false,
           py::arg("group_by_backend_capability") = true,
           py::arg("separate_fallback_segments") = true,
           py::arg("insert_sync_on_device_change") = true,
           py::arg("enable_fusion_v1") = true,
           py::arg("enable_fusion_cost_model_v1") = true,
           py::arg("fusion_cost_min_speedup") = 1.01);
}
