#include "bind_common.hpp"

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
  throw std::invalid_argument(
      "op must be one of: matmul, vector_add, matrix_sub, matrix_div, attention_forward, conv2d_nchw3x3s1p1");
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

}  // namespace

void bindGraph(py::module_& m) {
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
           py::arg("preferred_device") = "metal");
}
