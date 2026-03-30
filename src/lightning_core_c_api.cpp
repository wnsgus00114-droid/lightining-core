#include "lightning_core/lightning_core.h"

#include <string>

#include "lightning_core/core/runtime.hpp"

namespace {

lcError_t toLcError(lightning_core::runtime::Status status) {
  switch (status) {
    case lightning_core::runtime::Status::kSuccess:
      return LC_SUCCESS;
    case lightning_core::runtime::Status::kNotInitialized:
      return LC_NOT_INITIALIZED;
    case lightning_core::runtime::Status::kInvalidValue:
      return LC_INVALID_VALUE;
    case lightning_core::runtime::Status::kOutOfMemory:
      return LC_OUT_OF_MEMORY;
    case lightning_core::runtime::Status::kNotSupported:
      return LC_NOT_SUPPORTED;
    case lightning_core::runtime::Status::kDriverError:
      return LC_DRIVER_ERROR;
    case lightning_core::runtime::Status::kUnknown:
    default:
      return LC_UNKNOWN;
  }
}

lightning_core::runtime::MemcpyKind toMemcpyKind(lcMemcpyKind kind) {
  switch (kind) {
    case LC_MEMCPY_HOST_TO_HOST:
      return lightning_core::runtime::MemcpyKind::kHostToHost;
    case LC_MEMCPY_HOST_TO_DEVICE:
      return lightning_core::runtime::MemcpyKind::kHostToDevice;
    case LC_MEMCPY_DEVICE_TO_HOST:
      return lightning_core::runtime::MemcpyKind::kDeviceToHost;
    case LC_MEMCPY_DEVICE_TO_DEVICE:
      return lightning_core::runtime::MemcpyKind::kDeviceToDevice;
    default:
      return lightning_core::runtime::MemcpyKind::kHostToHost;
  }
}

lcDeviceKind toDeviceKind(lightning_core::runtime::Device device) {
  switch (device) {
    case lightning_core::runtime::Device::kCPU:
      return LC_DEVICE_CPU;
    case lightning_core::runtime::Device::kCUDA:
      return LC_DEVICE_CUDA;
    case lightning_core::runtime::Device::kMetal:
    default:
      return LC_DEVICE_METAL;
  }
}

lightning_core::runtime::Device fromLcDeviceKind(lcDeviceKind device) {
  switch (device) {
    case LC_DEVICE_CPU:
      return lightning_core::runtime::Device::kCPU;
    case LC_DEVICE_CUDA:
      return lightning_core::runtime::Device::kCUDA;
    case LC_DEVICE_METAL:
    default:
      return lightning_core::runtime::Device::kMetal;
  }
}

lcMemoryModel toLcMemoryModel(lightning_core::runtime::MemoryModel model) {
  if (model == lightning_core::runtime::MemoryModel::kNativeDevice) {
    return LC_MEMORY_NATIVE_DEVICE;
  }
  return LC_MEMORY_HOST_MANAGED_COMPAT;
}

lightning_core::runtime::SyncMode toSyncMode(lcSyncMode mode) {
  switch (mode) {
    case LC_SYNC_ALWAYS:
      return lightning_core::runtime::SyncMode::kAlways;
    case LC_SYNC_NEVER:
      return lightning_core::runtime::SyncMode::kNever;
    case LC_SYNC_AUTO:
    default:
      return lightning_core::runtime::SyncMode::kAuto;
  }
}

lcSyncMode toLcSyncMode(lightning_core::runtime::SyncMode mode) {
  switch (mode) {
    case lightning_core::runtime::SyncMode::kAlways:
      return LC_SYNC_ALWAYS;
    case lightning_core::runtime::SyncMode::kNever:
      return LC_SYNC_NEVER;
    case lightning_core::runtime::SyncMode::kAuto:
    default:
      return LC_SYNC_AUTO;
  }
}

lightning_core::runtime::SyncPolicy toSyncPolicy(lcSyncPolicy policy) {
  lightning_core::runtime::SyncPolicy out;
  out.mode = toSyncMode(policy.mode);
  out.trace_sync_boundary = (policy.trace_sync_boundary != 0);
  return out;
}

lcSyncPolicy toLcSyncPolicy(lightning_core::runtime::SyncPolicy policy) {
  lcSyncPolicy out;
  out.mode = toLcSyncMode(policy.mode);
  out.trace_sync_boundary = policy.trace_sync_boundary ? 1 : 0;
  return out;
}

lcBackendCapabilities toLcBackendCapabilities(lightning_core::runtime::BackendCapabilities caps) {
  lcBackendCapabilities out;
  out.device = toDeviceKind(caps.device);
  out.built = caps.built ? 1 : 0;
  out.available = caps.available ? 1 : 0;
  out.compute_surface = caps.compute_surface ? 1 : 0;
  out.memory_surface = caps.memory_surface ? 1 : 0;
  out.sync_surface = caps.sync_surface ? 1 : 0;
  out.profiling_surface = caps.profiling_surface ? 1 : 0;
  out.runtime_trace_surface = caps.runtime_trace_surface ? 1 : 0;
  out.sync_policy_surface = caps.sync_policy_surface ? 1 : 0;
  out.memory_model = toLcMemoryModel(caps.memory_model);
  return out;
}

}  // namespace

extern "C" {

lcError_t lcMalloc(void** ptr, size_t size_bytes) {
  return toLcError(lightning_core::runtime::mallocDevice(ptr, size_bytes));
}

lcError_t lcFree(void* ptr) {
  return toLcError(lightning_core::runtime::freeDevice(ptr));
}

lcError_t lcMemcpy(void* dst, const void* src, size_t size_bytes, lcMemcpyKind kind) {
  return toLcError(lightning_core::runtime::memcpy(dst, src, size_bytes, toMemcpyKind(kind)));
}

lcError_t lcDeviceSynchronize(void) {
  return toLcError(lightning_core::runtime::deviceSynchronize());
}

lcError_t lcGetDeviceCount(int* count) {
  return toLcError(lightning_core::runtime::getDeviceCount(count));
}

lcError_t lcGetPreferredDeviceForInference(lcDeviceKind* device) {
  if (device == nullptr) {
    return LC_INVALID_VALUE;
  }
  *device = toDeviceKind(lightning_core::runtime::preferredDeviceFor(lightning_core::runtime::WorkloadKind::kInference));
  return LC_SUCCESS;
}

lcError_t lcGetPreferredDeviceForTraining(lcDeviceKind* device) {
  if (device == nullptr) {
    return LC_INVALID_VALUE;
  }
  *device = toDeviceKind(lightning_core::runtime::preferredDeviceFor(lightning_core::runtime::WorkloadKind::kTraining));
  return LC_SUCCESS;
}

int lcIsCudaAvailable(void) {
  return lightning_core::runtime::isCudaAvailable() ? 1 : 0;
}

int lcIsMetalAvailable(void) {
  return lightning_core::runtime::isMetalAvailable() ? 1 : 0;
}

lcMemoryModel lcGetMemoryModel(void) {
  return toLcMemoryModel(lightning_core::runtime::deviceMemoryModel());
}

const char* lcGetMemoryModelName(lcMemoryModel model) {
  if (model == LC_MEMORY_NATIVE_DEVICE) {
    return lightning_core::runtime::memoryModelName(lightning_core::runtime::MemoryModel::kNativeDevice);
  }
  return lightning_core::runtime::memoryModelName(lightning_core::runtime::MemoryModel::kHostManagedCompat);
}

lcError_t lcSetDefaultSyncPolicy(lcSyncPolicy policy) {
  lightning_core::runtime::setDefaultSyncPolicy(toSyncPolicy(policy));
  return LC_SUCCESS;
}

lcSyncPolicy lcGetDefaultSyncPolicy(void) {
  return toLcSyncPolicy(lightning_core::runtime::defaultSyncPolicy());
}

lcError_t lcApplySyncPolicy(lcSyncPolicy policy) {
  return toLcError(lightning_core::runtime::deviceSynchronizeWithPolicy(toSyncPolicy(policy)));
}

lcError_t lcApplyDefaultSyncPolicy(void) {
  return toLcError(lightning_core::runtime::applyDefaultSyncPolicy());
}

lcError_t lcGetBackendCapabilities(lcDeviceKind device, lcBackendCapabilities* out_caps) {
  if (out_caps == nullptr) {
    return LC_INVALID_VALUE;
  }
  *out_caps = toLcBackendCapabilities(lightning_core::runtime::backendCapabilities(fromLcDeviceKind(device)));
  return LC_SUCCESS;
}

lcError_t lcGetActiveBackendCapabilities(lcBackendCapabilities* out_caps) {
  if (out_caps == nullptr) {
    return LC_INVALID_VALUE;
  }
  *out_caps = toLcBackendCapabilities(lightning_core::runtime::activeBackendCapabilities());
  return LC_SUCCESS;
}

const char* lcBackendName(void) {
  static std::string backend;
  backend = lightning_core::runtime::backendName();
  return backend.c_str();
}

const char* lcGetErrorString(lcError_t error) {
  switch (error) {
    case LC_SUCCESS:
      return lightning_core::runtime::getErrorString(lightning_core::runtime::Status::kSuccess);
    case LC_NOT_INITIALIZED:
      return lightning_core::runtime::getErrorString(lightning_core::runtime::Status::kNotInitialized);
    case LC_INVALID_VALUE:
      return lightning_core::runtime::getErrorString(lightning_core::runtime::Status::kInvalidValue);
    case LC_OUT_OF_MEMORY:
      return lightning_core::runtime::getErrorString(lightning_core::runtime::Status::kOutOfMemory);
    case LC_NOT_SUPPORTED:
      return lightning_core::runtime::getErrorString(lightning_core::runtime::Status::kNotSupported);
    case LC_DRIVER_ERROR:
      return lightning_core::runtime::getErrorString(lightning_core::runtime::Status::kDriverError);
    case LC_UNKNOWN:
    default:
      return lightning_core::runtime::getErrorString(lightning_core::runtime::Status::kUnknown);
  }
}

}  // extern "C"
