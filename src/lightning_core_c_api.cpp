#include "lightning_core/lightning_core.h"

#include <string>

#include "cudajun/runtime.hpp"

namespace {

lcError_t toLcError(cudajun::runtime::Status status) {
  switch (status) {
    case cudajun::runtime::Status::kSuccess:
      return LC_SUCCESS;
    case cudajun::runtime::Status::kNotInitialized:
      return LC_NOT_INITIALIZED;
    case cudajun::runtime::Status::kInvalidValue:
      return LC_INVALID_VALUE;
    case cudajun::runtime::Status::kOutOfMemory:
      return LC_OUT_OF_MEMORY;
    case cudajun::runtime::Status::kNotSupported:
      return LC_NOT_SUPPORTED;
    case cudajun::runtime::Status::kDriverError:
      return LC_DRIVER_ERROR;
    case cudajun::runtime::Status::kUnknown:
    default:
      return LC_UNKNOWN;
  }
}

cudajun::runtime::MemcpyKind toMemcpyKind(lcMemcpyKind kind) {
  switch (kind) {
    case LC_MEMCPY_HOST_TO_HOST:
      return cudajun::runtime::MemcpyKind::kHostToHost;
    case LC_MEMCPY_HOST_TO_DEVICE:
      return cudajun::runtime::MemcpyKind::kHostToDevice;
    case LC_MEMCPY_DEVICE_TO_HOST:
      return cudajun::runtime::MemcpyKind::kDeviceToHost;
    case LC_MEMCPY_DEVICE_TO_DEVICE:
      return cudajun::runtime::MemcpyKind::kDeviceToDevice;
    default:
      return cudajun::runtime::MemcpyKind::kHostToHost;
  }
}

lcDeviceKind toDeviceKind(cudajun::runtime::Device device) {
  switch (device) {
    case cudajun::runtime::Device::kCPU:
      return LC_DEVICE_CPU;
    case cudajun::runtime::Device::kCUDA:
      return LC_DEVICE_CUDA;
    case cudajun::runtime::Device::kMetal:
    default:
      return LC_DEVICE_METAL;
  }
}

lcMemoryModel toLcMemoryModel(cudajun::runtime::MemoryModel model) {
  if (model == cudajun::runtime::MemoryModel::kNativeDevice) {
    return LC_MEMORY_NATIVE_DEVICE;
  }
  return LC_MEMORY_HOST_MANAGED_COMPAT;
}

}  // namespace

extern "C" {

lcError_t lcMalloc(void** ptr, size_t size_bytes) {
  return toLcError(cudajun::runtime::mallocDevice(ptr, size_bytes));
}

lcError_t lcFree(void* ptr) {
  return toLcError(cudajun::runtime::freeDevice(ptr));
}

lcError_t lcMemcpy(void* dst, const void* src, size_t size_bytes, lcMemcpyKind kind) {
  return toLcError(cudajun::runtime::memcpy(dst, src, size_bytes, toMemcpyKind(kind)));
}

lcError_t lcDeviceSynchronize(void) {
  return toLcError(cudajun::runtime::deviceSynchronize());
}

lcError_t lcGetDeviceCount(int* count) {
  return toLcError(cudajun::runtime::getDeviceCount(count));
}

lcError_t lcGetPreferredDeviceForInference(lcDeviceKind* device) {
  if (device == nullptr) {
    return LC_INVALID_VALUE;
  }
  *device = toDeviceKind(cudajun::runtime::preferredDeviceFor(cudajun::runtime::WorkloadKind::kInference));
  return LC_SUCCESS;
}

lcError_t lcGetPreferredDeviceForTraining(lcDeviceKind* device) {
  if (device == nullptr) {
    return LC_INVALID_VALUE;
  }
  *device = toDeviceKind(cudajun::runtime::preferredDeviceFor(cudajun::runtime::WorkloadKind::kTraining));
  return LC_SUCCESS;
}

int lcIsCudaAvailable(void) {
  return cudajun::runtime::isCudaAvailable() ? 1 : 0;
}

int lcIsMetalAvailable(void) {
  return cudajun::runtime::isMetalAvailable() ? 1 : 0;
}

lcMemoryModel lcGetMemoryModel(void) {
  return toLcMemoryModel(cudajun::runtime::deviceMemoryModel());
}

const char* lcGetMemoryModelName(lcMemoryModel model) {
  if (model == LC_MEMORY_NATIVE_DEVICE) {
    return cudajun::runtime::memoryModelName(cudajun::runtime::MemoryModel::kNativeDevice);
  }
  return cudajun::runtime::memoryModelName(cudajun::runtime::MemoryModel::kHostManagedCompat);
}

const char* lcBackendName(void) {
  static std::string backend;
  backend = cudajun::runtime::backendName();
  return backend.c_str();
}

const char* lcGetErrorString(lcError_t error) {
  switch (error) {
    case LC_SUCCESS:
      return cudajun::runtime::getErrorString(cudajun::runtime::Status::kSuccess);
    case LC_NOT_INITIALIZED:
      return cudajun::runtime::getErrorString(cudajun::runtime::Status::kNotInitialized);
    case LC_INVALID_VALUE:
      return cudajun::runtime::getErrorString(cudajun::runtime::Status::kInvalidValue);
    case LC_OUT_OF_MEMORY:
      return cudajun::runtime::getErrorString(cudajun::runtime::Status::kOutOfMemory);
    case LC_NOT_SUPPORTED:
      return cudajun::runtime::getErrorString(cudajun::runtime::Status::kNotSupported);
    case LC_DRIVER_ERROR:
      return cudajun::runtime::getErrorString(cudajun::runtime::Status::kDriverError);
    case LC_UNKNOWN:
    default:
      return cudajun::runtime::getErrorString(cudajun::runtime::Status::kUnknown);
  }
}

}  // extern "C"
