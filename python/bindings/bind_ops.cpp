#include "bind_common.hpp"

namespace {

std::vector<float> matmulVector(const std::vector<float>& a,
                                const std::vector<float>& b,
                                std::size_t m_rows,
                                std::size_t k_inner,
                                std::size_t n_cols,
                                lc::Device device,
                                const lc::ops::MatMulIoPolicy& policy) {
  if (a.size() != m_rows * k_inner || b.size() != k_inner * n_cols) {
    throw std::invalid_argument("a or b shape mismatch");
  }
  std::vector<float> out(m_rows * n_cols, 0.0f);
  throwIfNotSuccess(lc::ops::matMulWithPolicy<float>(
      a.data(), b.data(), out.data(), m_rows, k_inner, n_cols, device, policy));
  return out;
}

}  // namespace

void bindOps(py::module_& m) {
  py::class_<lc::ops::MatMulIoPolicy>(m, "MatMulIoPolicy")
      .def(py::init<>())
      .def_readwrite("upload_a", &lc::ops::MatMulIoPolicy::upload_a)
      .def_readwrite("upload_b", &lc::ops::MatMulIoPolicy::upload_b)
      .def_readwrite("download_out", &lc::ops::MatMulIoPolicy::download_out)
      .def_readwrite("synchronize", &lc::ops::MatMulIoPolicy::synchronize);

  py::class_<lc::ops::MatrixElementwiseIoPolicy>(m, "MatrixElementwiseIoPolicy")
      .def(py::init<>())
      .def_readwrite("upload_a", &lc::ops::MatrixElementwiseIoPolicy::upload_a)
      .def_readwrite("upload_b", &lc::ops::MatrixElementwiseIoPolicy::upload_b)
      .def_readwrite("download_out", &lc::ops::MatrixElementwiseIoPolicy::download_out)
      .def_readwrite("synchronize", &lc::ops::MatrixElementwiseIoPolicy::synchronize);

  py::class_<lc::ops::VectorAddIoPolicy>(m, "VectorAddIoPolicy")
      .def(py::init<>())
      .def_readwrite("upload_a", &lc::ops::VectorAddIoPolicy::upload_a)
      .def_readwrite("upload_b", &lc::ops::VectorAddIoPolicy::upload_b)
      .def_readwrite("download_out", &lc::ops::VectorAddIoPolicy::download_out)
      .def_readwrite("synchronize", &lc::ops::VectorAddIoPolicy::synchronize);

  py::class_<lc::ops::MatMulMetalResidentSession<float>>(m, "MatMulMetalResidentSession")
      .def(py::init<std::size_t, std::size_t, std::size_t>())
      .def("start", [](const lc::ops::MatMulMetalResidentSession<float>& s,
                        const std::vector<float>& a,
                        const std::vector<float>& b,
                        std::size_t out_size) {
        std::vector<float> out(out_size, 0.0f);
        throwIfNotSuccess(s.start(a.data(), b.data(), out.data()));
        return out;
      })
      .def("run", [](const lc::ops::MatMulMetalResidentSession<float>& s,
                      const std::vector<float>& a,
                      const std::vector<float>& b,
                      std::size_t out_size) {
        std::vector<float> out(out_size, 0.0f);
        throwIfNotSuccess(s.run(a.data(), b.data(), out.data()));
        return out;
      })
      .def("finish", [](const lc::ops::MatMulMetalResidentSession<float>& s,
                         const std::vector<float>& a,
                         const std::vector<float>& b,
                         std::size_t out_size) {
        std::vector<float> out(out_size, 0.0f);
        throwIfNotSuccess(s.finish(a.data(), b.data(), out.data()));
        return out;
      })
      .def("start_into",
           [](const lc::ops::MatMulMetalResidentSession<float>& s,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
              py::array_t<float, py::array::c_style | py::array::forcecast>& out) {
             throwIfNotSuccess(s.start(a.data(), b.data(), out.mutable_data()));
           })
      .def("run_into",
           [](const lc::ops::MatMulMetalResidentSession<float>& s,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
              py::array_t<float, py::array::c_style | py::array::forcecast>& out) {
             throwIfNotSuccess(s.run(a.data(), b.data(), out.mutable_data()));
           })
      .def("finish_into",
           [](const lc::ops::MatMulMetalResidentSession<float>& s,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
              py::array_t<float, py::array::c_style | py::array::forcecast>& out) {
             throwIfNotSuccess(s.finish(a.data(), b.data(), out.mutable_data()));
           });

  py::class_<lc::ops::MatrixElemwiseMetalResidentSession<float>>(m, "MatrixElemwiseMetalResidentSession")
      .def(py::init<std::size_t, std::size_t>())
      .def("sub_start", [](const lc::ops::MatrixElemwiseMetalResidentSession<float>& s,
                            const std::vector<float>& a,
                            const std::vector<float>& b,
                            std::size_t out_size) {
        std::vector<float> out(out_size, 0.0f);
        throwIfNotSuccess(s.subStart(a.data(), b.data(), out.data()));
        return out;
      })
      .def("sub_run", [](const lc::ops::MatrixElemwiseMetalResidentSession<float>& s,
                          const std::vector<float>& a,
                          const std::vector<float>& b,
                          std::size_t out_size) {
        std::vector<float> out(out_size, 0.0f);
        throwIfNotSuccess(s.subRun(a.data(), b.data(), out.data()));
        return out;
      })
      .def("sub_finish", [](const lc::ops::MatrixElemwiseMetalResidentSession<float>& s,
                             const std::vector<float>& a,
                             const std::vector<float>& b,
                             std::size_t out_size) {
        std::vector<float> out(out_size, 0.0f);
        throwIfNotSuccess(s.subFinish(a.data(), b.data(), out.data()));
        return out;
      })
      .def("div_start", [](const lc::ops::MatrixElemwiseMetalResidentSession<float>& s,
                            const std::vector<float>& a,
                            const std::vector<float>& b,
                            std::size_t out_size) {
        std::vector<float> out(out_size, 0.0f);
        throwIfNotSuccess(s.divStart(a.data(), b.data(), out.data()));
        return out;
      })
      .def("div_run", [](const lc::ops::MatrixElemwiseMetalResidentSession<float>& s,
                          const std::vector<float>& a,
                          const std::vector<float>& b,
                          std::size_t out_size) {
        std::vector<float> out(out_size, 0.0f);
        throwIfNotSuccess(s.divRun(a.data(), b.data(), out.data()));
        return out;
      })
      .def("div_finish", [](const lc::ops::MatrixElemwiseMetalResidentSession<float>& s,
                             const std::vector<float>& a,
                             const std::vector<float>& b,
                             std::size_t out_size) {
        std::vector<float> out(out_size, 0.0f);
        throwIfNotSuccess(s.divFinish(a.data(), b.data(), out.data()));
        return out;
      })
      .def("sub", [](const lc::ops::MatrixElemwiseMetalResidentSession<float>& s,
                      const std::vector<float>& a,
                      const std::vector<float>& b,
                      std::size_t out_size) {
        std::vector<float> out(out_size, 0.0f);
        throwIfNotSuccess(s.subRun(a.data(), b.data(), out.data()));
        return out;
      })
      .def("div", [](const lc::ops::MatrixElemwiseMetalResidentSession<float>& s,
                      const std::vector<float>& a,
                      const std::vector<float>& b,
                      std::size_t out_size) {
        std::vector<float> out(out_size, 0.0f);
        throwIfNotSuccess(s.divRun(a.data(), b.data(), out.data()));
        return out;
      })
      .def("sub_start_into",
           [](const lc::ops::MatrixElemwiseMetalResidentSession<float>& s,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
              py::array_t<float, py::array::c_style | py::array::forcecast>& out) {
             throwIfNotSuccess(s.subStart(a.data(), b.data(), out.mutable_data()));
           })
      .def("sub_run_into",
           [](const lc::ops::MatrixElemwiseMetalResidentSession<float>& s,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
              py::array_t<float, py::array::c_style | py::array::forcecast>& out) {
             throwIfNotSuccess(s.subRun(a.data(), b.data(), out.mutable_data()));
           })
      .def("sub_finish_into",
           [](const lc::ops::MatrixElemwiseMetalResidentSession<float>& s,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
              py::array_t<float, py::array::c_style | py::array::forcecast>& out) {
             throwIfNotSuccess(s.subFinish(a.data(), b.data(), out.mutable_data()));
           })
      .def("div_start_into",
           [](const lc::ops::MatrixElemwiseMetalResidentSession<float>& s,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
              py::array_t<float, py::array::c_style | py::array::forcecast>& out) {
             throwIfNotSuccess(s.divStart(a.data(), b.data(), out.mutable_data()));
           })
      .def("div_run_into",
           [](const lc::ops::MatrixElemwiseMetalResidentSession<float>& s,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
              py::array_t<float, py::array::c_style | py::array::forcecast>& out) {
             throwIfNotSuccess(s.divRun(a.data(), b.data(), out.mutable_data()));
           })
      .def("div_finish_into",
           [](const lc::ops::MatrixElemwiseMetalResidentSession<float>& s,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
              py::array_t<float, py::array::c_style | py::array::forcecast>& out) {
             throwIfNotSuccess(s.divFinish(a.data(), b.data(), out.mutable_data()));
           });

  py::class_<lc::ops::VectorAddMetalResidentSession<float>>(m, "VectorAddMetalResidentSession")
      .def(py::init<std::size_t>())
      .def("start", [](const lc::ops::VectorAddMetalResidentSession<float>& s,
                        const std::vector<float>& a,
                        const std::vector<float>& b,
                        std::size_t out_size) {
        std::vector<float> out(out_size, 0.0f);
        throwIfNotSuccess(s.start(a.data(), b.data(), out.data()));
        return out;
      })
      .def("run", [](const lc::ops::VectorAddMetalResidentSession<float>& s,
                      const std::vector<float>& a,
                      const std::vector<float>& b,
                      std::size_t out_size) {
        std::vector<float> out(out_size, 0.0f);
        throwIfNotSuccess(s.run(a.data(), b.data(), out.data()));
        return out;
      })
      .def("finish", [](const lc::ops::VectorAddMetalResidentSession<float>& s,
                         const std::vector<float>& a,
                         const std::vector<float>& b,
                         std::size_t out_size) {
        std::vector<float> out(out_size, 0.0f);
        throwIfNotSuccess(s.finish(a.data(), b.data(), out.data()));
        return out;
      })
      .def("add", [](const lc::ops::VectorAddMetalResidentSession<float>& s,
                      const std::vector<float>& a,
                      const std::vector<float>& b,
                      std::size_t out_size) {
        std::vector<float> out(out_size, 0.0f);
        throwIfNotSuccess(s.run(a.data(), b.data(), out.data()));
        return out;
      })
      .def("start_into",
           [](const lc::ops::VectorAddMetalResidentSession<float>& s,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
              py::array_t<float, py::array::c_style | py::array::forcecast>& out) {
             throwIfNotSuccess(s.start(a.data(), b.data(), out.mutable_data()));
           })
      .def("run_into",
           [](const lc::ops::VectorAddMetalResidentSession<float>& s,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
              py::array_t<float, py::array::c_style | py::array::forcecast>& out) {
             throwIfNotSuccess(s.run(a.data(), b.data(), out.mutable_data()));
           })
      .def("finish_into",
           [](const lc::ops::VectorAddMetalResidentSession<float>& s,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
              py::array_t<float, py::array::c_style | py::array::forcecast>& out) {
             throwIfNotSuccess(s.finish(a.data(), b.data(), out.mutable_data()));
           });

  m.def("matmul",
        [](const std::vector<float>& a,
           const std::vector<float>& b,
           std::size_t m_rows,
           std::size_t k_inner,
           std::size_t n_cols,
           const std::string& device_name) {
          return matmulVector(a, b, m_rows, k_inner, n_cols, parseDevice(device_name), lc::ops::MatMulIoPolicy{});
        },
        py::arg("a"),
        py::arg("b"),
        py::arg("m"),
        py::arg("k"),
        py::arg("n"),
        py::arg("device") = "metal");

  m.def("matmul",
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
           const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
           std::size_t m_rows,
           std::size_t k_inner,
           std::size_t n_cols,
           const std::string& device_name) {
          if (static_cast<std::size_t>(a.size()) != m_rows * k_inner || static_cast<std::size_t>(b.size()) != k_inner * n_cols) {
            throw std::invalid_argument("a or b shape mismatch");
          }
          py::array_t<float> out(m_rows * n_cols);
          throwIfNotSuccess(lc::ops::matMulWithPolicy<float>(
              a.data(), b.data(), out.mutable_data(), m_rows, k_inner, n_cols, parseDevice(device_name), lc::ops::MatMulIoPolicy{}));
          return out;
        },
        py::arg("a"),
        py::arg("b"),
        py::arg("m"),
        py::arg("k"),
        py::arg("n"),
        py::arg("device") = "metal");

  m.def("matmul_np",
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
           const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
           std::size_t m_rows,
           std::size_t k_inner,
           std::size_t n_cols,
           const std::string& device_name) {
          if (static_cast<std::size_t>(a.size()) != m_rows * k_inner || static_cast<std::size_t>(b.size()) != k_inner * n_cols) {
            throw std::invalid_argument("a or b shape mismatch");
          }
          py::array_t<float> out(m_rows * n_cols);
          throwIfNotSuccess(lc::ops::matMulWithPolicy<float>(
              a.data(), b.data(), out.mutable_data(), m_rows, k_inner, n_cols, parseDevice(device_name), lc::ops::MatMulIoPolicy{}));
          return out;
        },
        py::arg("a"),
        py::arg("b"),
        py::arg("m"),
        py::arg("k"),
        py::arg("n"),
        py::arg("device") = "metal");

  m.def("matmul_np_into",
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
           const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
           py::array_t<float, py::array::c_style | py::array::forcecast>& out,
           std::size_t m_rows,
           std::size_t k_inner,
           std::size_t n_cols,
           const std::string& device_name) {
          if (static_cast<std::size_t>(a.size()) != m_rows * k_inner || static_cast<std::size_t>(b.size()) != k_inner * n_cols) {
            throw std::invalid_argument("a or b shape mismatch");
          }
          if (static_cast<std::size_t>(out.size()) != m_rows * n_cols) {
            throw std::invalid_argument("out shape mismatch");
          }
          throwIfNotSuccess(lc::ops::matMulWithPolicy<float>(
              a.data(), b.data(), out.mutable_data(), m_rows, k_inner, n_cols, parseDevice(device_name), lc::ops::MatMulIoPolicy{}));
        },
        py::arg("a"),
        py::arg("b"),
        py::arg("out"),
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
           const lc::ops::MatMulIoPolicy& policy) {
          return matmulVector(a, b, m_rows, k_inner, n_cols, parseDevice(device_name), policy);
        },
        py::arg("a"),
        py::arg("b"),
        py::arg("m"),
        py::arg("k"),
        py::arg("n"),
        py::arg("device") = "metal",
        py::arg("policy") = lc::ops::MatMulIoPolicy{});

  m.def("matmul_with_policy",
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
           const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
           std::size_t m_rows,
           std::size_t k_inner,
           std::size_t n_cols,
           const std::string& device_name,
           const lc::ops::MatMulIoPolicy& policy) {
          if (static_cast<std::size_t>(a.size()) != m_rows * k_inner || static_cast<std::size_t>(b.size()) != k_inner * n_cols) {
            throw std::invalid_argument("a or b shape mismatch");
          }
          py::array_t<float> out(m_rows * n_cols);
          throwIfNotSuccess(lc::ops::matMulWithPolicy<float>(
              a.data(), b.data(), out.mutable_data(), m_rows, k_inner, n_cols, parseDevice(device_name), policy));
          return out;
        },
        py::arg("a"),
        py::arg("b"),
        py::arg("m"),
        py::arg("k"),
        py::arg("n"),
        py::arg("device") = "metal",
        py::arg("policy") = lc::ops::MatMulIoPolicy{});

  m.def("matmul_np_with_policy",
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
           const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
           std::size_t m_rows,
           std::size_t k_inner,
           std::size_t n_cols,
           const std::string& device_name,
           const lc::ops::MatMulIoPolicy& policy) {
          if (static_cast<std::size_t>(a.size()) != m_rows * k_inner || static_cast<std::size_t>(b.size()) != k_inner * n_cols) {
            throw std::invalid_argument("a or b shape mismatch");
          }
          py::array_t<float> out(m_rows * n_cols);
          throwIfNotSuccess(lc::ops::matMulWithPolicy<float>(
              a.data(), b.data(), out.mutable_data(), m_rows, k_inner, n_cols, parseDevice(device_name), policy));
          return out;
        },
        py::arg("a"),
        py::arg("b"),
        py::arg("m"),
        py::arg("k"),
        py::arg("n"),
        py::arg("device") = "metal",
        py::arg("policy") = lc::ops::MatMulIoPolicy{});

  m.def("matmul_np_into_with_policy",
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
           const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
           py::array_t<float, py::array::c_style | py::array::forcecast>& out,
           std::size_t m_rows,
           std::size_t k_inner,
           std::size_t n_cols,
           const std::string& device_name,
           const lc::ops::MatMulIoPolicy& policy) {
          if (static_cast<std::size_t>(a.size()) != m_rows * k_inner || static_cast<std::size_t>(b.size()) != k_inner * n_cols) {
            throw std::invalid_argument("a or b shape mismatch");
          }
          if (static_cast<std::size_t>(out.size()) != m_rows * n_cols) {
            throw std::invalid_argument("out shape mismatch");
          }
          throwIfNotSuccess(lc::ops::matMulWithPolicy<float>(
              a.data(), b.data(), out.mutable_data(), m_rows, k_inner, n_cols, parseDevice(device_name), policy));
        },
        py::arg("a"),
        py::arg("b"),
        py::arg("out"),
        py::arg("m"),
        py::arg("k"),
        py::arg("n"),
        py::arg("device") = "metal",
        py::arg("policy") = lc::ops::MatMulIoPolicy{});

  m.def("vector_add",
        [](const std::vector<float>& a, const std::vector<float>& b, const std::string& device_name) {
          if (a.size() != b.size()) {
            throw std::invalid_argument("a and b sizes must match");
          }
          std::vector<float> out(a.size(), 0.0f);
          throwIfNotSuccess(lc::ops::vectorAdd<float>(a.data(), b.data(), out.data(), a.size(), parseDevice(device_name)));
          return out;
        },
        py::arg("a"),
        py::arg("b"),
        py::arg("device") = "metal");

  m.def("vector_add",
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
           const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
           const std::string& device_name) {
          if (a.size() != b.size()) {
            throw std::invalid_argument("a and b sizes must match");
          }
          py::array_t<float> out(a.size());
          throwIfNotSuccess(lc::ops::vectorAdd<float>(a.data(), b.data(), out.mutable_data(), static_cast<std::size_t>(a.size()), parseDevice(device_name)));
          return out;
        },
        py::arg("a"),
        py::arg("b"),
        py::arg("device") = "metal");

  m.def("matrix_sub",
        [](const std::vector<float>& a,
           const std::vector<float>& b,
           std::size_t rows,
           std::size_t cols,
           const std::string& device_name) {
          if (a.size() != rows * cols || b.size() != rows * cols) {
            throw std::invalid_argument("a or b shape mismatch");
          }
          std::vector<float> out(rows * cols, 0.0f);
          throwIfNotSuccess(lc::ops::matrixSub<float>(a.data(), b.data(), out.data(), rows, cols, parseDevice(device_name)));
          return out;
        },
        py::arg("a"),
        py::arg("b"),
        py::arg("rows"),
        py::arg("cols"),
        py::arg("device") = "metal");

  m.def("matrix_sub",
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
           const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
           std::size_t rows,
           std::size_t cols,
           const std::string& device_name) {
          if (static_cast<std::size_t>(a.size()) != rows * cols || static_cast<std::size_t>(b.size()) != rows * cols) {
            throw std::invalid_argument("a or b shape mismatch");
          }
          py::array_t<float> out(rows * cols);
          throwIfNotSuccess(lc::ops::matrixSub<float>(a.data(), b.data(), out.mutable_data(), rows, cols, parseDevice(device_name)));
          return out;
        },
        py::arg("a"),
        py::arg("b"),
        py::arg("rows"),
        py::arg("cols"),
        py::arg("device") = "metal");
}
