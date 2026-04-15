#include <cstring>
#include <iostream>

#include "lightning_core/lightning_core.h"

int main() {
  int major = -1;
  int minor = -1;
  int patch = -1;
  if (lcGetApiVersion(&major, &minor, &patch) != LC_SUCCESS) {
    std::cerr << "lcGetApiVersion failed\n";
    return 1;
  }
  if (major != LC_API_VERSION_MAJOR || minor != LC_API_VERSION_MINOR || patch != LC_API_VERSION_PATCH) {
    std::cerr << "version mismatch: " << major << "." << minor << "." << patch << "\n";
    return 1;
  }

  const char* vstr = lcGetApiVersionString();
  if (vstr == nullptr || std::strlen(vstr) == 0) {
    std::cerr << "version string missing\n";
    return 1;
  }

  if (lcCheckStructSize(LC_STRUCT_BACKEND_CAPABILITIES, sizeof(lcBackendCapabilities)) != 1) {
    std::cerr << "struct size check failed for lcBackendCapabilities\n";
    return 1;
  }
  if (lcCheckStructSize(LC_STRUCT_BACKEND_INTERFACE_CONTRACT, sizeof(lcBackendInterfaceContract)) != 1) {
    std::cerr << "struct size check failed for lcBackendInterfaceContract\n";
    return 1;
  }
  if (lcGetStructSize(LC_STRUCT_BACKEND_INTERFACE_CONTRACT) == 0) {
    std::cerr << "struct size query returned 0\n";
    return 1;
  }

  std::cout << "c api abi freeze smoke: ok\n";
  return 0;
}
