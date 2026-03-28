#include <stdio.h>
#include <string.h>

#include "lightining_core/lightining_core.h"

int main(void) {
  float host_in[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  float host_out[8] = {0};

  void* device_ptr = NULL;
  lcError_t err = lcMalloc(&device_ptr, sizeof(host_in));
  if (err != LC_SUCCESS) {
    fprintf(stderr, "lcMalloc failed: %s\n", lcGetErrorString(err));
    return 1;
  }

  err = lcMemcpy(device_ptr, host_in, sizeof(host_in), LC_MEMCPY_HOST_TO_DEVICE);
  if (err != LC_SUCCESS) {
    fprintf(stderr, "lcMemcpy H2D failed: %s\n", lcGetErrorString(err));
    lcFree(device_ptr);
    return 1;
  }

  err = lcMemcpy(host_out, device_ptr, sizeof(host_out), LC_MEMCPY_DEVICE_TO_HOST);
  if (err != LC_SUCCESS) {
    fprintf(stderr, "lcMemcpy D2H failed: %s\n", lcGetErrorString(err));
    lcFree(device_ptr);
    return 1;
  }

  err = lcDeviceSynchronize();
  if (err != LC_SUCCESS) {
    fprintf(stderr, "lcDeviceSynchronize failed: %s\n", lcGetErrorString(err));
    lcFree(device_ptr);
    return 1;
  }

  if (memcmp(host_in, host_out, sizeof(host_in)) != 0) {
    fprintf(stderr, "data mismatch\n");
    lcFree(device_ptr);
    return 1;
  }

  printf("backend=%s, metal=%d, cuda=%d\n", lcBackendName(), lcIsMetalAvailable(), lcIsCudaAvailable());
  printf("Lightining Core C API example: OK\n");

  lcFree(device_ptr);
  return 0;
}
