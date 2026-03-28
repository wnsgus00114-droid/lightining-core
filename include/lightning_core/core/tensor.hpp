#pragma once

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <type_traits>
#include <vector>

#include "lightning_core/core/ops.hpp"
#include "lightning_core/core/runtime.hpp"

namespace lightning_core {

using Device = runtime::Device;

enum class Layout {
  kContiguous = 0,
  kStrided
};

template <typename T>
class TensorT;

template <typename T>
class TensorViewT {
 public:
  TensorViewT() = default;

  const std::vector<std::int64_t>& shape() const {
    return shape_;
  }

  const std::vector<std::int64_t>& strides() const {
    return strides_;
  }

  std::size_t rank() const {
    return shape_.size();
  }

  std::size_t numel() const {
    if (shape_.empty()) {
      return 0;
    }
    std::size_t total = 1;
    for (std::int64_t dim : shape_) {
      total *= static_cast<std::size_t>(dim);
    }
    return total;
  }

  bool isContiguous() const {
    return layout_ == Layout::kContiguous;
  }

  Layout layout() const {
    return layout_;
  }

  Device device() const {
    return device_;
  }

  std::size_t offsetElements() const {
    return offset_elements_;
  }

  const char* dtypeName() const {
    if constexpr (std::is_same_v<T, float>) {
      return "float32";
    }
    if constexpr (std::is_same_v<T, double>) {
      return "float64";
    }
    return "longdouble";
  }

 private:
  template <typename U>
  friend class TensorT;

  TensorViewT(
      const std::vector<std::int64_t>& shape,
      const std::vector<std::int64_t>& strides,
      Device device,
      std::size_t offset_elements,
      Layout layout)
      : shape_(shape),
        strides_(strides),
        device_(device),
        offset_elements_(offset_elements),
        layout_(layout) {}

  std::vector<std::int64_t> shape_;
  std::vector<std::int64_t> strides_;
  Device device_ = Device::kCPU;
  std::size_t offset_elements_ = 0;
  Layout layout_ = Layout::kContiguous;
};

// float 계열 전용 텐서 템플릿.
template <typename T>
class TensorT {
  static_assert(std::is_floating_point_v<T>, "TensorT<T> supports floating-point types only");

 public:
  TensorT(const std::vector<std::int64_t>& shape, Device device = Device::kCPU)
      : shape_(shape), strides_(computeDefaultStrides(shape)), device_(device), device_storage_(nullptr) {
    allocateStorage();
  }

  ~TensorT() {
    releaseStorage();
  }

  TensorT(const TensorT&) = delete;
  TensorT& operator=(const TensorT&) = delete;

  TensorT(TensorT&& other) noexcept
      : shape_(std::move(other.shape_)),
        strides_(std::move(other.strides_)),
        device_(other.device_),
        cpu_storage_(std::move(other.cpu_storage_)),
        device_storage_(other.device_storage_) {
    other.device_storage_ = nullptr;
  }

  TensorT& operator=(TensorT&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    releaseStorage();
    shape_ = std::move(other.shape_);
    strides_ = std::move(other.strides_);
    device_ = other.device_;
    cpu_storage_ = std::move(other.cpu_storage_);
    device_storage_ = other.device_storage_;
    other.device_storage_ = nullptr;
    return *this;
  }

  std::size_t numel() const {
    if (shape_.empty() || !isShapeValid(shape_)) {
      return 0;
    }
    std::size_t total = 1;
    for (std::int64_t dim : shape_) {
      total *= static_cast<std::size_t>(dim);
    }
    return total;
  }

  const std::vector<std::int64_t>& shape() const {
    return shape_;
  }

  const std::vector<std::int64_t>& strides() const {
    return strides_;
  }

  std::size_t rank() const {
    return shape_.size();
  }

  bool isContiguous() const {
    return strides_ == computeDefaultStrides(shape_);
  }

  Layout layout() const {
    return isContiguous() ? Layout::kContiguous : Layout::kStrided;
  }

  const char* dtypeName() const {
    if constexpr (std::is_same_v<T, float>) {
      return "float32";
    }
    if constexpr (std::is_same_v<T, double>) {
      return "float64";
    }
    return "longdouble";
  }

  Device device() const {
    return device_;
  }

  runtime::Status reshape(const std::vector<std::int64_t>& new_shape) {
    if (!isShapeValid(new_shape)) {
      return runtime::Status::kInvalidValue;
    }
    std::size_t new_numel = 1;
    for (std::int64_t d : new_shape) {
      new_numel *= static_cast<std::size_t>(d);
    }
    if (new_numel != numel()) {
      return runtime::Status::kInvalidValue;
    }
    shape_ = new_shape;
    strides_ = computeDefaultStrides(new_shape);
    return runtime::Status::kSuccess;
  }

  runtime::Status view(const std::vector<std::int64_t>& new_shape, TensorViewT<T>* out) const {
    if (out == nullptr || !isContiguous() || !isShapeValid(new_shape)) {
      return runtime::Status::kInvalidValue;
    }
    std::size_t new_numel = 1;
    for (std::int64_t d : new_shape) {
      new_numel *= static_cast<std::size_t>(d);
    }
    if (new_numel != numel()) {
      return runtime::Status::kInvalidValue;
    }
    *out = TensorViewT<T>(new_shape, computeDefaultStrides(new_shape), device_, 0, Layout::kContiguous);
    return runtime::Status::kSuccess;
  }

  runtime::Status slice(
      std::size_t axis,
      std::int64_t start,
      std::int64_t end,
      TensorViewT<T>* out) const {
    if (out == nullptr || axis >= shape_.size() || start < 0 || end <= start || end > shape_[axis]) {
      return runtime::Status::kInvalidValue;
    }
    std::vector<std::int64_t> sliced_shape = shape_;
    sliced_shape[axis] = end - start;
    std::vector<std::int64_t> sliced_strides = strides_;
    const std::size_t offset = static_cast<std::size_t>(start * strides_[axis]);
    *out = TensorViewT<T>(sliced_shape, sliced_strides, device_, offset, Layout::kStrided);
    return runtime::Status::kSuccess;
  }

  runtime::Status toHostView(const TensorViewT<T>& view, std::vector<T>* out) const {
    if (view.device() != device_) {
      return runtime::Status::kInvalidValue;
    }
    return readStrided(view.shape(), view.strides(), view.offsetElements(), out);
  }

  runtime::Status sliceCopy(
      std::size_t axis,
      std::int64_t start,
      std::int64_t end,
      std::vector<T>* out) const {
    TensorViewT<T> v;
    runtime::Status st = slice(axis, start, end, &v);
    if (st != runtime::Status::kSuccess) {
      return st;
    }
    return toHostView(v, out);
  }

  runtime::Status readStrided(
      const std::vector<std::int64_t>& out_shape,
      const std::vector<std::int64_t>& source_strides,
      std::size_t offset_elements,
      std::vector<T>* out) const {
    if (out == nullptr || !isShapeValid(out_shape) || out_shape.size() != source_strides.size()) {
      return runtime::Status::kInvalidValue;
    }
    for (std::int64_t s : source_strides) {
      if (s <= 0) {
        return runtime::Status::kInvalidValue;
      }
    }

    std::size_t out_numel = 1;
    for (std::int64_t d : out_shape) {
      out_numel *= static_cast<std::size_t>(d);
    }

    std::size_t max_source_index = offset_elements;
    for (std::size_t i = 0; i < out_shape.size(); ++i) {
      max_source_index +=
          static_cast<std::size_t>(out_shape[i] - 1) * static_cast<std::size_t>(source_strides[i]);
    }
    if (max_source_index >= numel()) {
      return runtime::Status::kInvalidValue;
    }

    std::vector<T> host_values;
    const T* src = nullptr;
    if (device_ == Device::kCPU) {
      src = cpu_storage_.data();
    } else {
      runtime::Status st = toHost(&host_values);
      if (st != runtime::Status::kSuccess) {
        return st;
      }
      src = host_values.data();
    }

    out->assign(out_numel, static_cast<T>(0));

    std::vector<std::size_t> linear_strides(out_shape.size(), 1);
    for (std::size_t i = out_shape.size(); i > 1; --i) {
      linear_strides[i - 2] = linear_strides[i - 1] * static_cast<std::size_t>(out_shape[i - 1]);
    }

    for (std::size_t linear = 0; linear < out_numel; ++linear) {
      std::size_t rem = linear;
      std::size_t source_index = offset_elements;
      for (std::size_t axis = 0; axis < out_shape.size(); ++axis) {
        const std::size_t idx = rem / linear_strides[axis];
        rem %= linear_strides[axis];
        source_index += idx * static_cast<std::size_t>(source_strides[axis]);
      }
      (*out)[linear] = src[source_index];
    }

    return runtime::Status::kSuccess;
  }

  runtime::Status fromHost(const std::vector<T>& values) {
    if (values.size() != numel()) {
      return runtime::Status::kInvalidValue;
    }
    if (device_ == Device::kCPU) {
      cpu_storage_ = values;
      return runtime::Status::kSuccess;
    }
    return runtime::memcpy(
        device_storage_, values.data(), sizeof(T) * values.size(), runtime::MemcpyKind::kHostToDevice);
  }

  runtime::Status toHost(std::vector<T>* out) const {
    if (out == nullptr) {
      return runtime::Status::kInvalidValue;
    }
    out->resize(numel());
    if (device_ == Device::kCPU) {
      *out = cpu_storage_;
      return runtime::Status::kSuccess;
    }
    return runtime::memcpy(
        out->data(), device_storage_, sizeof(T) * out->size(), runtime::MemcpyKind::kDeviceToHost);
  }

  runtime::Status add(const TensorT& other, TensorT* out) const {
    if (out == nullptr || other.numel() != numel() || out->numel() != numel()) {
      return runtime::Status::kInvalidValue;
    }
    if (other.device_ != device_ || out->device_ != device_) {
      return runtime::Status::kInvalidValue;
    }

    if (device_ == Device::kCPU) {
      return ops::vectorAdd<T>(
          cpu_storage_.data(), other.cpu_storage_.data(), out->cpu_storage_.data(), numel(), device_);
    }
    return ops::vectorAdd<T>(
      static_cast<const T*>(device_storage_),
      static_cast<const T*>(other.device_storage_),
      static_cast<T*>(out->device_storage_),
        numel(),
        device_);
  }

  T* mutableCpuData() {
    return cpu_storage_.data();
  }

  const T* cpuData() const {
    return cpu_storage_.data();
  }

  void* rawDevicePtr() {
    return device_storage_;
  }

  const void* rawDevicePtr() const {
    return device_storage_;
  }

 private:
  static bool isShapeValid(const std::vector<std::int64_t>& shape) {
    if (shape.empty()) {
      return false;
    }
    for (std::int64_t d : shape) {
      if (d <= 0) {
        return false;
      }
    }
    return true;
  }

  static std::vector<std::int64_t> computeDefaultStrides(const std::vector<std::int64_t>& shape) {
    std::vector<std::int64_t> s(shape.size(), 1);
    if (shape.empty()) {
      return s;
    }
    for (std::size_t i = shape.size(); i > 1; --i) {
      s[i - 2] = s[i - 1] * shape[i - 1];
    }
    return s;
  }

  runtime::Status allocateStorage() {
    if (!isShapeValid(shape_)) {
      return runtime::Status::kInvalidValue;
    }
    if (device_ == Device::kCPU) {
      cpu_storage_.assign(numel(), static_cast<T>(0));
      return runtime::Status::kSuccess;
    }
    return runtime::mallocDevice(&device_storage_, sizeof(T) * numel());
  }

  void releaseStorage() {
    if (device_ != Device::kCPU && device_storage_ != nullptr) {
      runtime::freeDevice(device_storage_);
      device_storage_ = nullptr;
    }
  }

  std::vector<std::int64_t> shape_;
  std::vector<std::int64_t> strides_;
  Device device_;
  std::vector<T> cpu_storage_;
  void* device_storage_;
};

// 기존 코드 호환용 기본 타입 alias.
using Tensor = TensorT<float>;
using Tensor64 = TensorT<double>;
using TensorLong = TensorT<long double>;
using TensorView = TensorViewT<float>;
using Tensor64View = TensorViewT<double>;
using TensorLongView = TensorViewT<long double>;

}  // namespace lightning_core
