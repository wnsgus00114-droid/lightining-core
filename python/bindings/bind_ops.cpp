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

void requireMatrix2D(const py::buffer_info& info,
                     const char* name,
                     std::size_t* rows_out,
                     std::size_t* cols_out) {
  if (rows_out == nullptr || cols_out == nullptr) {
    throw std::invalid_argument("internal error: null shape output");
  }
  if (info.ndim != 2) {
    throw std::invalid_argument(std::string(name) + " must be a 2D float32 array");
  }
  if (info.shape.size() != 2) {
    throw std::invalid_argument(std::string(name) + " shape metadata is invalid");
  }
  *rows_out = static_cast<std::size_t>(info.shape[0]);
  *cols_out = static_cast<std::size_t>(info.shape[1]);
}

void requireMatrixOut2DOrFlat(const py::buffer_info& info,
                              std::size_t rows,
                              std::size_t cols,
                              const char* name) {
  if (info.ndim == 1) {
    const std::size_t expected = rows * cols;
    if (static_cast<std::size_t>(info.shape[0]) != expected) {
      throw std::invalid_argument(std::string(name) + " size mismatch for flattened output");
    }
    return;
  }
  if (info.ndim != 2 || info.shape.size() != 2) {
    throw std::invalid_argument(std::string(name) + " must be 2D or flattened 1D float32 array");
  }
  if (static_cast<std::size_t>(info.shape[0]) != rows || static_cast<std::size_t>(info.shape[1]) != cols) {
    throw std::invalid_argument(std::string(name) + " shape mismatch");
  }
}

struct Conv2dMetalResidentSessionF32 {
  std::size_t batch;
  std::size_t in_channels;
  std::size_t in_h;
  std::size_t in_w;
  std::size_t out_channels;
  std::size_t kernel_h;
  std::size_t kernel_w;
  std::size_t stride_h;
  std::size_t stride_w;
  std::size_t pad_h;
  std::size_t pad_w;
  bool apply_relu;

  lc::runtime::Status start(const float* x, const float* w, const float* bias, float* out) const {
    return lc::ops::conv2dNchwMetalResidentStart<float>(
        x,
        w,
        bias,
        out,
        batch,
        in_channels,
        in_h,
        in_w,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        apply_relu);
  }

  lc::runtime::Status run(const float* x, const float* w, const float* bias, float* out) const {
    return lc::ops::conv2dNchwMetalResidentRun<float>(
        x,
        w,
        bias,
        out,
        batch,
        in_channels,
        in_h,
        in_w,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        apply_relu);
  }

  lc::runtime::Status sync(const float* x, const float* w, const float* bias, float* out) const {
    return lc::ops::conv2dNchwMetalResidentSync<float>(
        x,
        w,
        bias,
        out,
        batch,
        in_channels,
        in_h,
        in_w,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        apply_relu);
  }

  lc::runtime::Status finish(const float* x, const float* w, const float* bias, float* out) const {
    return lc::ops::conv2dNchwMetalResidentFinish<float>(
        x,
        w,
        bias,
        out,
        batch,
        in_channels,
        in_h,
        in_w,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        apply_relu);
  }

  lc::runtime::Status runBatchSync(
      const float* x,
      const float* w,
      const float* bias,
      float* out,
      std::size_t repeat_count) const {
    if (repeat_count == 0) {
      return lc::runtime::Status::kInvalidValue;
    }
    for (std::size_t i = 0; i < repeat_count; ++i) {
      lc::runtime::Status st = run(x, w, bias, out);
      if (st != lc::runtime::Status::kSuccess) {
        return st;
      }
    }
    return sync(x, w, bias, out);
  }
};

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
      .def("sync", [](const lc::ops::MatMulMetalResidentSession<float>& s,
                       const std::vector<float>& a,
                       const std::vector<float>& b,
                       std::size_t out_size) {
        std::vector<float> out(out_size, 0.0f);
        throwIfNotSuccess(s.sync(a.data(), b.data(), out.data()));
        return out;
      })
      .def("run_batch_sync",
           [](const lc::ops::MatMulMetalResidentSession<float>& s,
              const std::vector<float>& a,
              const std::vector<float>& b,
              std::size_t out_size,
              std::size_t repeat_count) {
             std::vector<float> out(out_size, 0.0f);
             throwIfNotSuccess(s.runBatchSync(a.data(), b.data(), out.data(), repeat_count));
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
         })
       .def("sync_into",
         [](const lc::ops::MatMulMetalResidentSession<float>& s,
            const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
            const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
            py::array_t<float, py::array::c_style | py::array::forcecast>& out) {
           throwIfNotSuccess(s.sync(a.data(), b.data(), out.mutable_data()));
          })
      .def("run_batch_sync_into",
          [](const lc::ops::MatMulMetalResidentSession<float>& s,
            const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
            const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
            py::array_t<float, py::array::c_style | py::array::forcecast>& out,
            std::size_t repeat_count) {
           throwIfNotSuccess(s.runBatchSync(a.data(), b.data(), out.mutable_data(), repeat_count));
           })
      .def("run_batch_sync_no_download_into",
          [](const lc::ops::MatMulMetalResidentSession<float>& s,
            const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
            const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
            py::array_t<float, py::array::c_style | py::array::forcecast>& out,
            std::size_t repeat_count) {
           throwIfNotSuccess(s.runBatchSyncNoDownload(a.data(), b.data(), out.mutable_data(), repeat_count));
           })
      .def("run_batch_sync_cached_no_download",
          [](const lc::ops::MatMulMetalResidentSession<float>& s,
             std::size_t repeat_count) {
            throwIfNotSuccess(s.runBatchSyncCachedNoDownload(repeat_count));
          });

  py::class_<Conv2dMetalResidentSessionF32>(m, "Conv2dMetalResidentSession")
      .def(py::init<std::size_t,
                    std::size_t,
                    std::size_t,
                    std::size_t,
                    std::size_t,
                    std::size_t,
                    std::size_t,
                    std::size_t,
                    std::size_t,
                    std::size_t,
                    std::size_t,
                    bool>(),
           py::arg("batch"),
           py::arg("in_channels"),
           py::arg("in_h"),
           py::arg("in_w"),
           py::arg("out_channels"),
           py::arg("kernel_h"),
           py::arg("kernel_w"),
           py::arg("stride_h") = 1,
           py::arg("stride_w") = 1,
           py::arg("pad_h") = 0,
           py::arg("pad_w") = 0,
           py::arg("apply_relu") = true)
      .def("start_into",
           [](const Conv2dMetalResidentSessionF32& s,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& x,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& w,
              py::object bias_obj,
              py::array_t<float, py::array::c_style | py::array::forcecast>& out) {
             const float* bias_ptr = nullptr;
             py::array_t<float, py::array::c_style | py::array::forcecast> bias_arr;
             if (!bias_obj.is_none()) {
               bias_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(bias_obj);
               bias_ptr = bias_arr.data();
             }
             throwIfNotSuccess(s.start(x.data(), w.data(), bias_ptr, out.mutable_data()));
           })
      .def("run_into",
           [](const Conv2dMetalResidentSessionF32& s,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& x,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& w,
              py::object bias_obj,
              py::array_t<float, py::array::c_style | py::array::forcecast>& out) {
             const float* bias_ptr = nullptr;
             py::array_t<float, py::array::c_style | py::array::forcecast> bias_arr;
             if (!bias_obj.is_none()) {
               bias_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(bias_obj);
               bias_ptr = bias_arr.data();
             }
             throwIfNotSuccess(s.run(x.data(), w.data(), bias_ptr, out.mutable_data()));
           })
      .def("sync_into",
           [](const Conv2dMetalResidentSessionF32& s,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& x,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& w,
              py::object bias_obj,
              py::array_t<float, py::array::c_style | py::array::forcecast>& out) {
             const float* bias_ptr = nullptr;
             py::array_t<float, py::array::c_style | py::array::forcecast> bias_arr;
             if (!bias_obj.is_none()) {
               bias_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(bias_obj);
               bias_ptr = bias_arr.data();
             }
             throwIfNotSuccess(s.sync(x.data(), w.data(), bias_ptr, out.mutable_data()));
           })
      .def("finish_into",
           [](const Conv2dMetalResidentSessionF32& s,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& x,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& w,
              py::object bias_obj,
              py::array_t<float, py::array::c_style | py::array::forcecast>& out) {
             const float* bias_ptr = nullptr;
             py::array_t<float, py::array::c_style | py::array::forcecast> bias_arr;
             if (!bias_obj.is_none()) {
               bias_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(bias_obj);
               bias_ptr = bias_arr.data();
             }
             throwIfNotSuccess(s.finish(x.data(), w.data(), bias_ptr, out.mutable_data()));
           })
      .def("run_batch_sync_into",
           [](const Conv2dMetalResidentSessionF32& s,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& x,
              const py::array_t<float, py::array::c_style | py::array::forcecast>& w,
              py::object bias_obj,
              py::array_t<float, py::array::c_style | py::array::forcecast>& out,
              std::size_t repeat_count) {
             const float* bias_ptr = nullptr;
             py::array_t<float, py::array::c_style | py::array::forcecast> bias_arr;
             if (!bias_obj.is_none()) {
               bias_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(bias_obj);
               bias_ptr = bias_arr.data();
             }
             throwIfNotSuccess(s.runBatchSync(x.data(), w.data(), bias_ptr, out.mutable_data(), repeat_count));
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

  m.def("conv2d_nchw",
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>& x,
           const py::array_t<float, py::array::c_style | py::array::forcecast>& w,
           py::object bias_obj,
           std::size_t stride_h,
           std::size_t stride_w,
           std::size_t pad_h,
           std::size_t pad_w,
           const std::string& device_name) {
          auto xbuf = x.request();
          auto wbuf = w.request();
          if (xbuf.ndim != 4 || wbuf.ndim != 4) {
            throw std::invalid_argument("x and w must be 4D arrays (NCHW and OIHW)");
          }

          const std::size_t batch = static_cast<std::size_t>(xbuf.shape[0]);
          const std::size_t in_channels = static_cast<std::size_t>(xbuf.shape[1]);
          const std::size_t in_h = static_cast<std::size_t>(xbuf.shape[2]);
          const std::size_t in_w = static_cast<std::size_t>(xbuf.shape[3]);

          const std::size_t out_channels = static_cast<std::size_t>(wbuf.shape[0]);
          const std::size_t w_in_channels = static_cast<std::size_t>(wbuf.shape[1]);
          const std::size_t kernel_h = static_cast<std::size_t>(wbuf.shape[2]);
          const std::size_t kernel_w = static_cast<std::size_t>(wbuf.shape[3]);

          if (in_channels != w_in_channels) {
            throw std::invalid_argument("x channels and w in_channels mismatch");
          }
          if (stride_h == 0 || stride_w == 0) {
            throw std::invalid_argument("stride must be > 0");
          }
          if (in_h + (2 * pad_h) < kernel_h || in_w + (2 * pad_w) < kernel_w) {
            throw std::invalid_argument("invalid padding/kernel for input size");
          }

          const std::size_t out_h = (in_h + (2 * pad_h) - kernel_h) / stride_h + 1;
          const std::size_t out_w = (in_w + (2 * pad_w) - kernel_w) / stride_w + 1;

          py::array_t<float> out({batch, out_channels, out_h, out_w});

          const float* bias_ptr = nullptr;
          py::array_t<float, py::array::c_style | py::array::forcecast> bias_arr;
          if (!bias_obj.is_none()) {
            bias_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(bias_obj);
            if (static_cast<std::size_t>(bias_arr.size()) != out_channels) {
              throw std::invalid_argument("bias shape mismatch");
            }
            bias_ptr = bias_arr.data();
          }

          throwIfNotSuccess(lc::ops::conv2dNchw<float>(
              x.data(),
              w.data(),
              bias_ptr,
              out.mutable_data(),
              batch,
              in_channels,
              in_h,
              in_w,
              out_channels,
              kernel_h,
              kernel_w,
              stride_h,
              stride_w,
              pad_h,
              pad_w,
              parseDevice(device_name)));
          return out;
        },
        py::arg("x"),
        py::arg("w"),
        py::arg("bias") = py::none(),
        py::arg("stride_h") = 1,
        py::arg("stride_w") = 1,
        py::arg("pad_h") = 0,
        py::arg("pad_w") = 0,
        py::arg("device") = "metal");

  m.def("conv2d_nchw_into",
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>& x,
           const py::array_t<float, py::array::c_style | py::array::forcecast>& w,
           py::object bias_obj,
           py::array_t<float, py::array::c_style | py::array::forcecast>& out,
           std::size_t stride_h,
           std::size_t stride_w,
           std::size_t pad_h,
           std::size_t pad_w,
           const std::string& device_name) {
          auto xbuf = x.request();
          auto wbuf = w.request();
          auto obuf = out.request();
          if (xbuf.ndim != 4 || wbuf.ndim != 4 || obuf.ndim != 4) {
            throw std::invalid_argument("x, w, out must be 4D arrays");
          }

          const std::size_t batch = static_cast<std::size_t>(xbuf.shape[0]);
          const std::size_t in_channels = static_cast<std::size_t>(xbuf.shape[1]);
          const std::size_t in_h = static_cast<std::size_t>(xbuf.shape[2]);
          const std::size_t in_w = static_cast<std::size_t>(xbuf.shape[3]);

          const std::size_t out_channels = static_cast<std::size_t>(wbuf.shape[0]);
          const std::size_t w_in_channels = static_cast<std::size_t>(wbuf.shape[1]);
          const std::size_t kernel_h = static_cast<std::size_t>(wbuf.shape[2]);
          const std::size_t kernel_w = static_cast<std::size_t>(wbuf.shape[3]);

          if (in_channels != w_in_channels) {
            throw std::invalid_argument("x channels and w in_channels mismatch");
          }
          if (stride_h == 0 || stride_w == 0) {
            throw std::invalid_argument("stride must be > 0");
          }
          if (in_h + (2 * pad_h) < kernel_h || in_w + (2 * pad_w) < kernel_w) {
            throw std::invalid_argument("invalid padding/kernel for input size");
          }

          const std::size_t out_h = (in_h + (2 * pad_h) - kernel_h) / stride_h + 1;
          const std::size_t out_w = (in_w + (2 * pad_w) - kernel_w) / stride_w + 1;
          if (static_cast<std::size_t>(obuf.shape[0]) != batch ||
              static_cast<std::size_t>(obuf.shape[1]) != out_channels ||
              static_cast<std::size_t>(obuf.shape[2]) != out_h ||
              static_cast<std::size_t>(obuf.shape[3]) != out_w) {
            throw std::invalid_argument("out shape mismatch");
          }

          const float* bias_ptr = nullptr;
          py::array_t<float, py::array::c_style | py::array::forcecast> bias_arr;
          if (!bias_obj.is_none()) {
            bias_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(bias_obj);
            if (static_cast<std::size_t>(bias_arr.size()) != out_channels) {
              throw std::invalid_argument("bias shape mismatch");
            }
            bias_ptr = bias_arr.data();
          }

          throwIfNotSuccess(lc::ops::conv2dNchw<float>(
              x.data(),
              w.data(),
              bias_ptr,
              out.mutable_data(),
              batch,
              in_channels,
              in_h,
              in_w,
              out_channels,
              kernel_h,
              kernel_w,
              stride_h,
              stride_w,
              pad_h,
              pad_w,
              parseDevice(device_name)));
        },
        py::arg("x"),
        py::arg("w"),
        py::arg("bias"),
        py::arg("out"),
        py::arg("stride_h") = 1,
        py::arg("stride_w") = 1,
        py::arg("pad_h") = 0,
        py::arg("pad_w") = 0,
        py::arg("device") = "metal");

  m.def("conv2d_nchw_metal_resident",
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>& x,
           const py::array_t<float, py::array::c_style | py::array::forcecast>& w,
           py::object bias_obj,
           py::array_t<float, py::array::c_style | py::array::forcecast>& out,
           std::size_t stride_h,
           std::size_t stride_w,
           std::size_t pad_h,
           std::size_t pad_w,
           const std::string& phase,
           bool apply_relu) {
          auto xbuf = x.request();
          auto wbuf = w.request();
          auto obuf = out.request();
          if (xbuf.ndim != 4 || wbuf.ndim != 4 || obuf.ndim != 4) {
            throw std::invalid_argument("x, w, out must be 4D arrays");
          }

          const std::size_t batch = static_cast<std::size_t>(xbuf.shape[0]);
          const std::size_t in_channels = static_cast<std::size_t>(xbuf.shape[1]);
          const std::size_t in_h = static_cast<std::size_t>(xbuf.shape[2]);
          const std::size_t in_w = static_cast<std::size_t>(xbuf.shape[3]);

          const std::size_t out_channels = static_cast<std::size_t>(wbuf.shape[0]);
          const std::size_t w_in_channels = static_cast<std::size_t>(wbuf.shape[1]);
          const std::size_t kernel_h = static_cast<std::size_t>(wbuf.shape[2]);
          const std::size_t kernel_w = static_cast<std::size_t>(wbuf.shape[3]);

          if (in_channels != w_in_channels) {
            throw std::invalid_argument("x channels and w in_channels mismatch");
          }
          if (stride_h == 0 || stride_w == 0) {
            throw std::invalid_argument("stride must be > 0");
          }
          if (in_h + (2 * pad_h) < kernel_h || in_w + (2 * pad_w) < kernel_w) {
            throw std::invalid_argument("invalid padding/kernel for input size");
          }

          const std::size_t out_h = (in_h + (2 * pad_h) - kernel_h) / stride_h + 1;
          const std::size_t out_w = (in_w + (2 * pad_w) - kernel_w) / stride_w + 1;
          if (static_cast<std::size_t>(obuf.shape[0]) != batch ||
              static_cast<std::size_t>(obuf.shape[1]) != out_channels ||
              static_cast<std::size_t>(obuf.shape[2]) != out_h ||
              static_cast<std::size_t>(obuf.shape[3]) != out_w) {
            throw std::invalid_argument("out shape mismatch");
          }

          const float* bias_ptr = nullptr;
          py::array_t<float, py::array::c_style | py::array::forcecast> bias_arr;
          if (!bias_obj.is_none()) {
            bias_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(bias_obj);
            if (static_cast<std::size_t>(bias_arr.size()) != out_channels) {
              throw std::invalid_argument("bias shape mismatch");
            }
            bias_ptr = bias_arr.data();
          }

          if (phase == "start") {
            throwIfNotSuccess(lc::ops::conv2dNchwMetalResidentStart<float>(
                x.data(),
                w.data(),
                bias_ptr,
                out.mutable_data(),
                batch,
                in_channels,
                in_h,
                in_w,
                out_channels,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                apply_relu));
            return;
          }
          if (phase == "run") {
            throwIfNotSuccess(lc::ops::conv2dNchwMetalResidentRun<float>(
                x.data(),
                w.data(),
                bias_ptr,
                out.mutable_data(),
                batch,
                in_channels,
                in_h,
                in_w,
                out_channels,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                apply_relu));
            return;
          }
          if (phase.rfind("run_batch_sync", 0) == 0) {
            std::size_t repeat_count = 8;
            if (phase.size() > 14 && phase[14] == ':') {
              try {
                repeat_count = static_cast<std::size_t>(std::stoull(phase.substr(15)));
              } catch (...) {
                throw std::invalid_argument("run_batch_sync count parse failed");
              }
              if (repeat_count == 0) {
                throw std::invalid_argument("run_batch_sync repeat_count must be > 0");
              }
            }

            for (std::size_t i = 0; i < repeat_count; ++i) {
              throwIfNotSuccess(lc::ops::conv2dNchwMetalResidentRun<float>(
                  x.data(),
                  w.data(),
                  bias_ptr,
                  out.mutable_data(),
                  batch,
                  in_channels,
                  in_h,
                  in_w,
                  out_channels,
                  kernel_h,
                  kernel_w,
                  stride_h,
                  stride_w,
                  pad_h,
                  pad_w,
                  apply_relu));
            }
            throwIfNotSuccess(lc::ops::conv2dNchwMetalResidentSync<float>(
                x.data(),
                w.data(),
                bias_ptr,
                out.mutable_data(),
                batch,
                in_channels,
                in_h,
                in_w,
                out_channels,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                apply_relu));
            return;
          }
          if (phase == "sync") {
            throwIfNotSuccess(lc::ops::conv2dNchwMetalResidentSync<float>(
                x.data(),
                w.data(),
                bias_ptr,
                out.mutable_data(),
                batch,
                in_channels,
                in_h,
                in_w,
                out_channels,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                apply_relu));
            return;
          }
          if (phase == "finish") {
            throwIfNotSuccess(lc::ops::conv2dNchwMetalResidentFinish<float>(
                x.data(),
                w.data(),
                bias_ptr,
                out.mutable_data(),
                batch,
                in_channels,
                in_h,
                in_w,
                out_channels,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                apply_relu));
            return;
          }
          throw std::invalid_argument("phase must be one of: start, run, run_batch_sync[:N], sync, finish");
        },
        py::arg("x"),
        py::arg("w"),
        py::arg("bias"),
        py::arg("out"),
        py::arg("stride_h") = 1,
        py::arg("stride_w") = 1,
        py::arg("pad_h") = 0,
        py::arg("pad_w") = 0,
        py::arg("phase") = "run",
        py::arg("apply_relu") = false);

        m.def("matmul_reset_tuning", []() {
          lc::runtime::Status st = lc::ops::matMulMetalResetTuning();
          if (st != lc::runtime::Status::kSuccess && st != lc::runtime::Status::kNotSupported) {
            throwIfNotSuccess(st);
          }
        });

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

  m.def("matmul2d",
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
           const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
           const std::string& device_name) {
          std::size_t m_rows = 0;
          std::size_t k_inner = 0;
          std::size_t b_rows = 0;
          std::size_t n_cols = 0;
          requireMatrix2D(a.request(), "a", &m_rows, &k_inner);
          requireMatrix2D(b.request(), "b", &b_rows, &n_cols);
          if (k_inner != b_rows) {
            throw std::invalid_argument("matmul2d shape mismatch: a.shape[1] must equal b.shape[0]");
          }

          py::array_t<float> out({static_cast<py::ssize_t>(m_rows), static_cast<py::ssize_t>(n_cols)});
          throwIfNotSuccess(lc::ops::matMulWithPolicy<float>(
              a.data(), b.data(), out.mutable_data(), m_rows, k_inner, n_cols, parseDevice(device_name), lc::ops::MatMulIoPolicy{}));
          return out;
        },
        py::arg("a"),
        py::arg("b"),
        py::arg("device") = "metal");

  m.def("matmul2d_into",
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
           const py::array_t<float, py::array::c_style | py::array::forcecast>& b,
           py::array_t<float, py::array::c_style | py::array::forcecast>& out,
           const std::string& device_name) {
          std::size_t m_rows = 0;
          std::size_t k_inner = 0;
          std::size_t b_rows = 0;
          std::size_t n_cols = 0;
          requireMatrix2D(a.request(), "a", &m_rows, &k_inner);
          requireMatrix2D(b.request(), "b", &b_rows, &n_cols);
          if (k_inner != b_rows) {
            throw std::invalid_argument("matmul2d_into shape mismatch: a.shape[1] must equal b.shape[0]");
          }
          requireMatrixOut2DOrFlat(out.request(), m_rows, n_cols, "out");

          throwIfNotSuccess(lc::ops::matMulWithPolicy<float>(
              a.data(), b.data(), out.mutable_data(), m_rows, k_inner, n_cols, parseDevice(device_name), lc::ops::MatMulIoPolicy{}));
        },
        py::arg("a"),
        py::arg("b"),
        py::arg("out"),
        py::arg("device") = "metal");

  m.def("matmul2d_resident_session",
        [](const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
           const py::array_t<float, py::array::c_style | py::array::forcecast>& b) {
          std::size_t m_rows = 0;
          std::size_t k_inner = 0;
          std::size_t b_rows = 0;
          std::size_t n_cols = 0;
          requireMatrix2D(a.request(), "a", &m_rows, &k_inner);
          requireMatrix2D(b.request(), "b", &b_rows, &n_cols);
          if (k_inner != b_rows) {
            throw std::invalid_argument("matmul2d_resident_session shape mismatch: a.shape[1] must equal b.shape[0]");
          }
          return lc::ops::MatMulMetalResidentSession<float>(m_rows, k_inner, n_cols);
        },
        py::arg("a"),
        py::arg("b"));
}
