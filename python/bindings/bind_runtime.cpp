#include "bind_common.hpp"

namespace {

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
  throw std::invalid_argument("sync mode must be 'auto', 'always', or 'never'");
}

py::dict toBackendCapabilitiesDict(const lc::runtime::BackendCapabilities& caps) {
  py::dict out;
  out["device"] = toString(caps.device);
  out["built"] = caps.built;
  out["available"] = caps.available;
  out["compute_surface"] = caps.compute_surface;
  out["memory_surface"] = caps.memory_surface;
  out["sync_surface"] = caps.sync_surface;
  out["profiling_surface"] = caps.profiling_surface;
  out["runtime_trace_surface"] = caps.runtime_trace_surface;
  out["sync_policy_surface"] = caps.sync_policy_surface;
  out["memory_model"] = lc::runtime::memoryModelName(caps.memory_model);
  return out;
}

}  // namespace

void bindRuntime(py::module_& m) {
  m.def("cuda_available", [] { return lc::runtime::isCudaAvailable(); });
  m.def("metal_available", [] { return lc::runtime::isMetalAvailable(); });
  m.def("backend_name", [] { return lc::runtime::backendName(); });
  m.def("memory_model_name",
        [] { return std::string(lc::runtime::memoryModelName(lc::runtime::deviceMemoryModel())); });
  m.def("runtime_sync_policy_set",
        [](const std::string& mode, bool trace_sync_boundary) {
          lc::runtime::SyncPolicy policy;
          policy.mode = parseSyncMode(mode);
          policy.trace_sync_boundary = trace_sync_boundary;
          lc::runtime::setDefaultSyncPolicy(policy);
        },
        py::arg("mode") = "auto",
        py::arg("trace_sync_boundary") = false);
  m.def("runtime_sync_policy_get", [] {
    const auto policy = lc::runtime::defaultSyncPolicy();
    py::dict out;
    out["mode"] = lc::runtime::syncModeName(policy.mode);
    out["trace_sync_boundary"] = policy.trace_sync_boundary;
    return out;
  });
  m.def("runtime_sync_apply",
        [](const std::string& mode, bool trace_sync_boundary) {
          lc::runtime::SyncPolicy policy;
          policy.mode = parseSyncMode(mode);
          policy.trace_sync_boundary = trace_sync_boundary;
          throwIfNotSuccess(lc::runtime::deviceSynchronizeWithPolicy(policy));
        },
        py::arg("mode") = "auto",
        py::arg("trace_sync_boundary") = false);
  m.def("runtime_sync_apply_default", [] { throwIfNotSuccess(lc::runtime::applyDefaultSyncPolicy()); });
  m.def("runtime_backend_capabilities",
        [](const std::string& device) {
          return toBackendCapabilitiesDict(lc::runtime::backendCapabilities(parseDevice(device)));
        },
        py::arg("device"));
  m.def("runtime_active_backend_capabilities",
        [] { return toBackendCapabilitiesDict(lc::runtime::activeBackendCapabilities()); });
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
