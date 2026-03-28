#include "bind_common.hpp"

void bindRuntime(py::module_& m) {
  m.def("cuda_available", [] { return lc::runtime::isCudaAvailable(); });
  m.def("metal_available", [] { return lc::runtime::isMetalAvailable(); });
  m.def("backend_name", [] { return lc::runtime::backendName(); });
  m.def("memory_model_name",
        [] { return std::string(lc::runtime::memoryModelName(lc::runtime::deviceMemoryModel())); });
}
