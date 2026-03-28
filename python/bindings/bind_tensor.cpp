#include "bind_common.hpp"

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

template <typename TensorType, typename ViewType, typename Scalar>
void bindTensorType(py::module_& m, const char* name) {
  py::class_<TensorType>(m, name)
      .def(py::init([](const std::vector<std::int64_t>& shape, const std::string& device) {
             return TensorType(shape, parseDevice(device));
           }),
           py::arg("shape"),
           py::arg("device") = "cpu")
      .def("shape", &TensorType::shape)
      .def("strides", &TensorType::strides)
      .def("rank", &TensorType::rank)
      .def("is_contiguous", &TensorType::isContiguous)
      .def("dtype", &TensorType::dtypeName)
      .def("device", [](const TensorType& t) { return toString(t.device()); })
      .def("numel", &TensorType::numel)
      .def("reshape", [](TensorType& t, const std::vector<std::int64_t>& shape) {
        throwIfNotSuccess(t.reshape(shape));
      })
      .def("layout", [](const TensorType& t) {
        return t.layout() == lc::Layout::kContiguous ? "contiguous" : "strided";
      })
      .def("view", [](const TensorType& t, const std::vector<std::int64_t>& shape) {
        ViewType v;
        throwIfNotSuccess(t.view(shape, &v));
        return v;
      })
      .def("slice", [](const TensorType& t, std::size_t axis, std::int64_t start, std::int64_t end) {
        ViewType v;
        throwIfNotSuccess(t.slice(axis, start, end, &v));
        return v;
      })
      .def("to_host_view", [](const TensorType& t, const ViewType& v) {
        std::vector<Scalar> out;
        throwIfNotSuccess(t.toHostView(v, &out));
        return out;
      })
      .def("to_host_view_numpy", [](const TensorType& t, const ViewType& v) {
        std::vector<Scalar> out;
        throwIfNotSuccess(t.toHostView(v, &out));
        return toNumpy(out);
      })
      .def("slice_copy", [](const TensorType& t, std::size_t axis, std::int64_t start, std::int64_t end) {
        std::vector<Scalar> out;
        throwIfNotSuccess(t.sliceCopy(axis, start, end, &out));
        return out;
      })
      .def("slice_copy_numpy", [](const TensorType& t, std::size_t axis, std::int64_t start, std::int64_t end) {
        std::vector<Scalar> out;
        throwIfNotSuccess(t.sliceCopy(axis, start, end, &out));
        return toNumpy(out);
      })
      .def("read_strided",
           [](const TensorType& t,
              const std::vector<std::int64_t>& out_shape,
              const std::vector<std::int64_t>& source_strides,
              std::size_t offset_elements) {
             std::vector<Scalar> out;
             throwIfNotSuccess(t.readStrided(out_shape, source_strides, offset_elements, &out));
             return out;
           },
           py::arg("out_shape"),
           py::arg("source_strides"),
           py::arg("offset_elements") = 0)
      .def("from_list", [](TensorType& t, const std::vector<Scalar>& values) {
        throwIfNotSuccess(t.fromHost(values));
      })
      .def("from_numpy",
           [](TensorType& t, const py::array_t<Scalar, py::array::c_style | py::array::forcecast>& values) {
             throwIfNotSuccess(t.fromHost(toVector(values)));
           })
      .def("to_list", [](const TensorType& t) {
        std::vector<Scalar> out;
        throwIfNotSuccess(t.toHost(&out));
        return out;
      })
      .def("to_numpy", [](const TensorType& t) {
        std::vector<Scalar> out;
        throwIfNotSuccess(t.toHost(&out));
        return toNumpy(out);
      })
      .def("add", [](const TensorType& lhs, const TensorType& rhs) {
        TensorType out(lhs.shape(), lhs.device());
        throwIfNotSuccess(lhs.add(rhs, &out));
        return out;
      });
}

void bindTensor(py::module_& m) {
  bindTensorViewType<lc::TensorView>(m, "TensorView");
  bindTensorViewType<lc::Tensor64View>(m, "Tensor64View");
  bindTensorType<lc::Tensor, lc::TensorView, float>(m, "Tensor");
  bindTensorType<lc::Tensor64, lc::Tensor64View, double>(m, "Tensor64");
}
