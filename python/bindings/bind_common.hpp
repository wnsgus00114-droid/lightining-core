#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lightning_core/attention.hpp"
#include "lightning_core/ops.hpp"
#include "lightning_core/runtime.hpp"
#include "lightning_core/tensor.hpp"

namespace py = pybind11;
namespace lc = lightning_core;

inline lc::Device parseDevice(const std::string& name) {
  if (name == "cpu") {
    return lc::Device::kCPU;
  }
  if (name == "cuda") {
    return lc::Device::kCUDA;
  }
  if (name == "metal") {
    return lc::Device::kMetal;
  }
  throw std::invalid_argument("device must be 'cpu', 'cuda', or 'metal'");
}

inline std::string toString(lc::Device device) {
  if (device == lc::Device::kCUDA) {
    return "cuda";
  }
  if (device == lc::Device::kMetal) {
    return "metal";
  }
  return "cpu";
}

inline void throwIfNotSuccess(lc::runtime::Status status) {
  if (status != lc::runtime::Status::kSuccess) {
    throw std::runtime_error(lc::runtime::getErrorString(status));
  }
}

template <typename T>
inline std::vector<T> toVector(const py::array_t<T, py::array::c_style | py::array::forcecast>& arr) {
  std::vector<T> out(static_cast<std::size_t>(arr.size()));
  std::memcpy(out.data(), arr.data(), sizeof(T) * out.size());
  return out;
}

template <typename T>
inline py::array_t<T> toNumpy(const std::vector<T>& values) {
  py::array_t<T> out(values.size());
  if (!values.empty()) {
    std::memcpy(out.mutable_data(), values.data(), sizeof(T) * values.size());
  }
  return out;
}

template <typename T>
inline void requireSize(const py::array_t<T, py::array::c_style | py::array::forcecast>& arr,
                        std::size_t expected,
                        const char* name) {
  if (static_cast<std::size_t>(arr.size()) != expected) {
    throw std::invalid_argument(std::string(name) + " size mismatch");
  }
}

void bindTensor(py::module_& m);
void bindRuntime(py::module_& m);
void bindOps(py::module_& m);
void bindAttention(py::module_& m);
