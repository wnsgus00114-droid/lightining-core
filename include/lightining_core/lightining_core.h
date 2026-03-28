#pragma once

#include <stddef.h>

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

lcError_t lcMalloc(void** ptr, size_t size_bytes);
lcError_t lcFree(void* ptr);
lcError_t lcMemcpy(void* dst, const void* src, size_t size_bytes, lcMemcpyKind kind);
lcError_t lcDeviceSynchronize(void);
lcError_t lcGetDeviceCount(int* count);
lcError_t lcGetPreferredDeviceForInference(lcDeviceKind* device);
lcError_t lcGetPreferredDeviceForTraining(lcDeviceKind* device);
int lcIsCudaAvailable(void);
int lcIsMetalAvailable(void);
const char* lcBackendName(void);
const char* lcGetErrorString(lcError_t error);

#ifdef __cplusplus
}
#endif
