#pragma once

#include <stddef.h>

#define LC_API_VERSION_MAJOR 0
#define LC_API_VERSION_MINOR 6
#define LC_API_VERSION_PATCH 0
#define LC_API_VERSION_STRING "0.6.0-rc0"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum lcError {
  LC_SUCCESS = 0,
  LC_NOT_INITIALIZED,
  LC_INVALID_VALUE,
  LC_OUT_OF_MEMORY,
  LC_NOT_SUPPORTED,
  LC_DRIVER_ERROR,
  LC_UNKNOWN
} lcError_t;

typedef enum lcMemcpyKind {
  LC_MEMCPY_HOST_TO_HOST = 0,
  LC_MEMCPY_HOST_TO_DEVICE,
  LC_MEMCPY_DEVICE_TO_HOST,
  LC_MEMCPY_DEVICE_TO_DEVICE
} lcMemcpyKind;

typedef enum lcDeviceKind {
  LC_DEVICE_CPU = 0,
  LC_DEVICE_CUDA,
  LC_DEVICE_METAL
} lcDeviceKind;

typedef enum lcMemoryModel {
  LC_MEMORY_NATIVE_DEVICE = 0,
  LC_MEMORY_HOST_MANAGED_COMPAT
} lcMemoryModel;

typedef enum lcSyncMode {
  LC_SYNC_AUTO = 0,
  LC_SYNC_ALWAYS,
  LC_SYNC_NEVER
} lcSyncMode;

typedef struct lcSyncPolicy {
  lcSyncMode mode;
  int trace_sync_boundary;
} lcSyncPolicy;

typedef struct lcBackendCapabilities {
  lcDeviceKind device;
  int built;
  int available;
  int compute_surface;
  int memory_surface;
  int sync_surface;
  int profiling_surface;
  int runtime_trace_surface;
  int sync_policy_surface;
  lcMemoryModel memory_model;
} lcBackendCapabilities;

typedef struct lcComputeInterfaceContract {
  int available;
  int op_dispatch_trace_surface;
  const char* driver_tag;
} lcComputeInterfaceContract;

typedef struct lcMemoryInterfaceContract {
  int available;
  int allocator_surface;
  int memcpy_surface;
  lcMemoryModel memory_model;
  const char* driver_tag;
} lcMemoryInterfaceContract;

typedef struct lcSyncInterfaceContract {
  int available;
  int sync_policy_surface;
  int trace_sync_boundary_surface;
  const char* driver_tag;
} lcSyncInterfaceContract;

typedef struct lcProfilerInterfaceContract {
  int available;
  int runtime_trace_surface;
  int op_dispatch_trace_surface;
  size_t trace_capacity;
  const char* driver_tag;
} lcProfilerInterfaceContract;

typedef struct lcBackendInterfaceContract {
  lcDeviceKind device;
  lcBackendCapabilities capabilities;
  lcComputeInterfaceContract compute;
  lcMemoryInterfaceContract memory;
  lcSyncInterfaceContract sync;
  lcProfilerInterfaceContract profiler;
} lcBackendInterfaceContract;

typedef enum lcStructId {
  LC_STRUCT_BACKEND_CAPABILITIES = 1,
  LC_STRUCT_COMPUTE_INTERFACE_CONTRACT = 2,
  LC_STRUCT_MEMORY_INTERFACE_CONTRACT = 3,
  LC_STRUCT_SYNC_INTERFACE_CONTRACT = 4,
  LC_STRUCT_PROFILER_INTERFACE_CONTRACT = 5,
  LC_STRUCT_BACKEND_INTERFACE_CONTRACT = 6
} lcStructId;

lcError_t lcMalloc(void** ptr, size_t size_bytes);
lcError_t lcFree(void* ptr);
lcError_t lcMemcpy(void* dst, const void* src, size_t size_bytes, lcMemcpyKind kind);
lcError_t lcDeviceSynchronize(void);
lcError_t lcGetDeviceCount(int* count);
lcError_t lcGetPreferredDeviceForInference(lcDeviceKind* device);
lcError_t lcGetPreferredDeviceForTraining(lcDeviceKind* device);
int lcIsCudaAvailable(void);
int lcIsMetalAvailable(void);
lcMemoryModel lcGetMemoryModel(void);
const char* lcGetMemoryModelName(lcMemoryModel model);
lcError_t lcSetDefaultSyncPolicy(lcSyncPolicy policy);
lcSyncPolicy lcGetDefaultSyncPolicy(void);
lcError_t lcApplySyncPolicy(lcSyncPolicy policy);
lcError_t lcApplyDefaultSyncPolicy(void);
lcError_t lcGetBackendCapabilities(lcDeviceKind device, lcBackendCapabilities* out_caps);
lcError_t lcGetActiveBackendCapabilities(lcBackendCapabilities* out_caps);
lcError_t lcGetBackendInterfaceContract(lcDeviceKind device, lcBackendInterfaceContract* out_contract);
lcError_t lcGetActiveBackendInterfaceContract(lcBackendInterfaceContract* out_contract);
lcError_t lcGetApiVersion(int* out_major, int* out_minor, int* out_patch);
const char* lcGetApiVersionString(void);
size_t lcGetStructSize(lcStructId struct_id);
int lcCheckStructSize(lcStructId struct_id, size_t observed_size);
const char* lcBackendName(void);
const char* lcGetErrorString(lcError_t error);

#ifdef __cplusplus
}
#endif
