#include "bind_common.hpp"

void bindRuntime(py::module_& m) {
  m.def("cuda_available", [] { return lc::runtime::isCudaAvailable(); });
  m.def("metal_available", [] { return lc::runtime::isMetalAvailable(); });
  m.def("backend_name", [] { return lc::runtime::backendName(); });
  m.def("memory_model_name",
        [] { return std::string(lc::runtime::memoryModelName(lc::runtime::deviceMemoryModel())); });
  m.def("runtime_trace_enable", [](bool enabled) { lc::runtime::setRuntimeTraceEnabled(enabled); }, py::arg("enabled"));
  m.def("runtime_trace_enabled", [] { return lc::runtime::isRuntimeTraceEnabled(); });
  m.def("runtime_trace_clear", [] { lc::runtime::clearRuntimeTraceEvents(); });
  m.def("runtime_trace_capacity", [] { return lc::runtime::runtimeTraceEventCapacity(); });
  m.def("runtime_trace_events", [] {
    py::list out;
    const auto events = lc::runtime::runtimeTraceEvents();
    for (const auto& ev : events) {
      py::dict row;
      row["type"] = lc::runtime::runtimeTraceEventTypeName(ev.type);
      row["status"] = lc::runtime::getErrorString(ev.status);
      row["timestamp_ns"] = ev.timestamp_ns;
      row["size_bytes"] = ev.size_bytes;
      row["detail0"] = ev.detail0;
      row["detail1"] = ev.detail1;
      out.append(row);
    }
    return out;
  });
}
