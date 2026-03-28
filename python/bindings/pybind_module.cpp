#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cudajun/attention.hpp"
#include "cudajun/ops.hpp"
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

template <typename ViewType>
void bindTensorViewType(py::module_& m, const char* name) {
  py::class_<ViewType>(m, name)
      .def("shape", &ViewType::shape)
      .def("strides", &ViewType::strides)
      .def("rank", &ViewType::rank)
      .def("numel", &ViewType::numel)
      .def("is_contiguous", &ViewType::isContiguous)
      .def("dtype", &ViewType::dtypeName)
      .def("offset_elements", &ViewType::offsetElements)
      .def("device", [](const ViewType& t) { return toString(t.device()); });
}

template <typename TensorType, typename Scalar>
void bindTensorType(py::module_& m, const char* name) {
  py::class_<TensorType>(m, name)
      .def(py::init([](const std::vector<std::int64_t>& shape, const std::string& device) {
             return TensorType(shape, parseDevice(device));
           }),
           py::arg("shape"), py::arg("device") = "cpu")
      .def("shape", &TensorType::shape)
      .def("strides", &TensorType::strides)
      .def("rank", &TensorType::rank)
      .def("is_contiguous", &TensorType::isContiguous)
      .def("dtype", &TensorType::dtypeName)
      .def("device", [](const TensorType& t) { return toString(t.device()); })
      .def("numel", &TensorType::numel)
      .def("reshape", [](TensorType& t, const std::vector<std::int64_t>& shape) {
        auto status = t.reshape(shape);
        if (status != cudajun::runtime::Status::kSuccess) {
          throw std::runtime_error(cudajun::runtime::getErrorString(status));
        }
      })
      .def("layout", [](const TensorType& t) {
        return t.layout() == cudajun::Layout::kContiguous ? "contiguous" : "strided";
      })
      .def("view", [](const TensorType& t, const std::vector<std::int64_t>& shape) {
        cudajun::TensorViewT<Scalar> v;
        auto status = t.view(shape, &v);
        if (status != cudajun::runtime::Status::kSuccess) {
          throw std::runtime_error(cudajun::runtime::getErrorString(status));
        }
        return v;
      })
      .def("slice", [](const TensorType& t, std::size_t axis, std::int64_t start, std::int64_t end) {
        cudajun::TensorViewT<Scalar> v;
        auto status = t.slice(axis, start, end, &v);
        if (status != cudajun::runtime::Status::kSuccess) {
          throw std::runtime_error(cudajun::runtime::getErrorString(status));
        }
        return v;
      })
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

  py::class_<cudajun::ops::MatMulIoPolicy>(m, "MatMulIoPolicy")
      .def(py::init<>())
      .def_readwrite("upload_a", &cudajun::ops::MatMulIoPolicy::upload_a)
      .def_readwrite("upload_b", &cudajun::ops::MatMulIoPolicy::upload_b)
      .def_readwrite("download_out", &cudajun::ops::MatMulIoPolicy::download_out)
      .def_readwrite("synchronize", &cudajun::ops::MatMulIoPolicy::synchronize);

  py::class_<cudajun::ops::MatMulMetalResidentSession<float>>(m, "MatMulMetalResidentSession")
      .def(py::init<std::size_t, std::size_t, std::size_t>())
      .def("start",
           [](const cudajun::ops::MatMulMetalResidentSession<float>& s,
              const std::vector<float>& a,
              const std::vector<float>& b) {
             std::vector<float> out(a.size(), 0.0f);
             auto status = s.start(a.data(), b.data(), out.data());
             if (status != cudajun::runtime::Status::kSuccess) {
               throw std::runtime_error(cudajun::runtime::getErrorString(status));
             }
           })
      .def("run",
           [](const cudajun::ops::MatMulMetalResidentSession<float>& s,
              const std::vector<float>& a,
              const std::vector<float>& b) {
             std::vector<float> out(a.size(), 0.0f);
             auto status = s.run(a.data(), b.data(), out.data());
             if (status != cudajun::runtime::Status::kSuccess) {
               throw std::runtime_error(cudajun::runtime::getErrorString(status));
             }
           })
      .def("finish",
           [](const cudajun::ops::MatMulMetalResidentSession<float>& s,
              const std::vector<float>& a,
              const std::vector<float>& b,
              std::size_t out_size) {
             std::vector<float> out(out_size, 0.0f);
             auto status = s.finish(a.data(), b.data(), out.data());
             if (status != cudajun::runtime::Status::kSuccess) {
               throw std::runtime_error(cudajun::runtime::getErrorString(status));
             }
             return out;
           });

  bindTensorViewType<cudajun::TensorView>(m, "TensorView");
  bindTensorViewType<cudajun::Tensor64View>(m, "Tensor64View");
  bindTensorType<cudajun::Tensor, float>(m, "Tensor");
  bindTensorType<cudajun::Tensor64, double>(m, "Tensor64");

  // 런타임 상태 유틸 함수.
  m.def("cuda_available", [] { return cudajun::runtime::isCudaAvailable(); });
  m.def("metal_available", [] { return cudajun::runtime::isMetalAvailable(); });
  m.def("backend_name", [] { return cudajun::runtime::backendName(); });
  m.def("memory_model_name",
        [] { return std::string(cudajun::runtime::memoryModelName(cudajun::runtime::deviceMemoryModel())); });

  m.def("matmul",
        [](const std::vector<float>& a,
           const std::vector<float>& b,
           std::size_t m_rows,
           std::size_t k_inner,
           std::size_t n_cols,
           const std::string& device_name) {
          if (a.size() != m_rows * k_inner || b.size() != k_inner * n_cols) {
            throw std::invalid_argument("a or b shape mismatch");
          }
          std::vector<float> out(m_rows * n_cols, 0.0f);
          auto status = cudajun::ops::matMul<float>(
              a.data(), b.data(), out.data(), m_rows, k_inner, n_cols, parseDevice(device_name));
          if (status != cudajun::runtime::Status::kSuccess) {
            throw std::runtime_error(cudajun::runtime::getErrorString(status));
          }
          return out;
        },
        py::arg("a"),
        py::arg("b"),
        py::arg("m"),
        py::arg("k"),
        py::arg("n"),
        py::arg("device") = "metal");

  m.def("matmul_with_policy",
        [](const std::vector<float>& a,
           const std::vector<float>& b,
           std::size_t m_rows,
           std::size_t k_inner,
           std::size_t n_cols,
           const std::string& device_name,
           const cudajun::ops::MatMulIoPolicy& policy) {
          if (a.size() != m_rows * k_inner || b.size() != k_inner * n_cols) {
            throw std::invalid_argument("a or b shape mismatch");
          }
          std::vector<float> out(m_rows * n_cols, 0.0f);
          auto status = cudajun::ops::matMulWithPolicy<float>(
              a.data(), b.data(), out.data(), m_rows, k_inner, n_cols, parseDevice(device_name), policy);
          if (status != cudajun::runtime::Status::kSuccess) {
            throw std::runtime_error(cudajun::runtime::getErrorString(status));
          }
          return out;
        },
        py::arg("a"),
        py::arg("b"),
        py::arg("m"),
        py::arg("k"),
        py::arg("n"),
        py::arg("device") = "metal",
        py::arg("policy") = cudajun::ops::MatMulIoPolicy{});

  m.def("attention_forward",
        [](const std::vector<float>& q,
           const std::vector<float>& k,
           const std::vector<float>& v,
           std::size_t seq_len,
           std::size_t head_dim,
           bool causal,
           const std::string& device_name) {
          const std::size_t expected = seq_len * head_dim;
          if (q.size() != expected || k.size() != expected || v.size() != expected) {
            throw std::invalid_argument("q/k/v length must match seq_len * head_dim");
          }
          cudajun::AttentionConfig cfg{seq_len, head_dim, causal};
          std::vector<float> out(expected, 0.0f);
          const auto status = cudajun::attentionForward(
              q.data(), k.data(), v.data(), out.data(), cfg, parseDevice(device_name));
          if (status != cudajun::runtime::Status::kSuccess) {
            throw std::runtime_error(cudajun::runtime::getErrorString(status));
          }
          return out;
        },
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("seq_len"),
        py::arg("head_dim"),
        py::arg("causal") = false,
        py::arg("device") = "metal");
}
