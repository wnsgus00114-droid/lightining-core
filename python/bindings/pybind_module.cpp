#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cudajun/runtime.hpp"
#include "cudajun/tensor.hpp"

namespace py = pybind11;

namespace {

// Python 문자열 장치명 -> C++ Device enum 변환.
cudajun::Device parseDevice(const std::string& name) {
  if (name == "cpu") {
    return cudajun::Device::kCPU;
  }
  if (name == "cuda") {
    return cudajun::Device::kCUDA;
  }
  if (name == "metal") {
    return cudajun::Device::kMetal;
  }
  throw std::invalid_argument("device must be 'cpu', 'cuda', or 'metal'");
}

// C++ Device enum -> Python 문자열 변환.
std::string toString(cudajun::Device device) {
  if (device == cudajun::Device::kCUDA) {
    return "cuda";
  }
  if (device == cudajun::Device::kMetal) {
    return "metal";
  }
  return "cpu";
}

}  // namespace

template <typename TensorType, typename Scalar>
void bindTensorType(py::module_& m, const char* name) {
  py::class_<TensorType>(m, name)
      .def(py::init([](const std::vector<std::int64_t>& shape, const std::string& device) {
             return TensorType(shape, parseDevice(device));
           }),
           py::arg("shape"), py::arg("device") = "cpu")
      .def("shape", &TensorType::shape)
      .def("device", [](const TensorType& t) { return toString(t.device()); })
      .def("numel", &TensorType::numel)
      .def("from_list", [](TensorType& t, const std::vector<Scalar>& values) {
        auto status = t.fromHost(values);
        if (status != cudajun::runtime::Status::kSuccess) {
          throw std::runtime_error(cudajun::runtime::getErrorString(status));
        }
      })
      .def("to_list", [](const TensorType& t) {
        std::vector<Scalar> out;
        auto status = t.toHost(&out);
        if (status != cudajun::runtime::Status::kSuccess) {
          throw std::runtime_error(cudajun::runtime::getErrorString(status));
        }
        return out;
      })
      .def("add", [](const TensorType& lhs, const TensorType& rhs) {
        TensorType out(lhs.shape(), lhs.device());
        auto status = lhs.add(rhs, &out);
        if (status != cudajun::runtime::Status::kSuccess) {
          throw std::runtime_error(cudajun::runtime::getErrorString(status));
        }
        return out;
      });
}

PYBIND11_MODULE(lightining_core, m) {
  // 모듈 docstring.
  m.doc() = "Lightining Core python bindings";

  bindTensorType<cudajun::Tensor, float>(m, "Tensor");
  bindTensorType<cudajun::Tensor64, double>(m, "Tensor64");

  // 런타임 상태 유틸 함수.
  m.def("cuda_available", [] { return cudajun::runtime::isCudaAvailable(); });
  m.def("backend_name", [] { return cudajun::runtime::backendName(); });
}
