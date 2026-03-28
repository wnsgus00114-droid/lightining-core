#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "cudajun/ops.hpp"
#include "cudajun/runtime.hpp"

namespace cudajun {

using Device = runtime::Device;

// float 계열 전용 텐서 템플릿.
template <typename T>
class TensorT {
  static_assert(std::is_floating_point_v<T>, "TensorT<T> supports floating-point types only");

 public:
  TensorT(const std::vector<std::int64_t>& shape, Device device = Device::kCPU)
      : shape_(shape), device_(device), cuda_storage_(nullptr) {
    allocateStorage();
  }

  ~TensorT() {
    releaseStorage();
  }

  TensorT(const TensorT&) = delete;
  TensorT& operator=(const TensorT&) = delete;

  TensorT(TensorT&& other) noexcept
      : shape_(std::move(other.shape_)),
        device_(other.device_),
        cpu_storage_(std::move(other.cpu_storage_)),
        cuda_storage_(other.cuda_storage_) {
    other.cuda_storage_ = nullptr;
  }

  TensorT& operator=(TensorT&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    releaseStorage();
    shape_ = std::move(other.shape_);
    device_ = other.device_;
    cpu_storage_ = std::move(other.cpu_storage_);
    cuda_storage_ = other.cuda_storage_;
    other.cuda_storage_ = nullptr;
    return *this;
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

  const std::vector<std::int64_t>& shape() const {
    return shape_;
  }

  Device device() const {
    return device_;
  }

  runtime::Status fromHost(const std::vector<T>& values) {
    if (values.size() != numel()) {
      return runtime::Status::kInvalidValue;
    }
    // Metal 경로는 현재 Host staging 방식으로 운영한다.
    if (device_ == Device::kCPU || device_ == Device::kMetal) {
      cpu_storage_ = values;
      return runtime::Status::kSuccess;
    }
    return runtime::memcpy(
        cuda_storage_, values.data(), sizeof(T) * values.size(), runtime::MemcpyKind::kHostToDevice);
  }

  runtime::Status toHost(std::vector<T>* out) const {
    if (out == nullptr) {
      return runtime::Status::kInvalidValue;
    }
    out->resize(numel());
    if (device_ == Device::kCPU || device_ == Device::kMetal) {
      *out = cpu_storage_;
      return runtime::Status::kSuccess;
    }
    return runtime::memcpy(
        out->data(), cuda_storage_, sizeof(T) * out->size(), runtime::MemcpyKind::kDeviceToHost);
  }

  runtime::Status add(const TensorT& other, TensorT* out) const {
    if (out == nullptr || other.numel() != numel() || out->numel() != numel()) {
      return runtime::Status::kInvalidValue;
    }
    if (other.device_ != device_ || out->device_ != device_) {
      return runtime::Status::kInvalidValue;
    }

    if (device_ == Device::kCPU || device_ == Device::kMetal) {
      return ops::vectorAdd<T>(
          cpu_storage_.data(), other.cpu_storage_.data(), out->cpu_storage_.data(), numel(), device_);
    }
    return ops::vectorAdd<T>(
        static_cast<const T*>(cuda_storage_),
        static_cast<const T*>(other.cuda_storage_),
        static_cast<T*>(out->cuda_storage_),
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
    return cuda_storage_;
  }

  const void* rawDevicePtr() const {
    return cuda_storage_;
  }

 private:
  runtime::Status allocateStorage() {
    if (device_ == Device::kCPU || device_ == Device::kMetal) {
      cpu_storage_.assign(numel(), static_cast<T>(0));
      return runtime::Status::kSuccess;
    }
    if (numel() == 0) {
      return runtime::Status::kInvalidValue;
    }
    return runtime::mallocDevice(&cuda_storage_, sizeof(T) * numel());
  }

  void releaseStorage() {
    if (device_ == Device::kCUDA && cuda_storage_ != nullptr) {
      runtime::freeDevice(cuda_storage_);
      cuda_storage_ = nullptr;
    }
  }

  std::vector<std::int64_t> shape_;
  Device device_;
  std::vector<T> cpu_storage_;
  void* cuda_storage_;
};

// 기존 코드 호환용 기본 타입 alias.
using Tensor = TensorT<float>;
using Tensor64 = TensorT<double>;
using TensorLong = TensorT<long double>;

}  // namespace cudajun
