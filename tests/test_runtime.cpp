#include <iostream>

#include "lightning_core/runtime.hpp"

int main() {
  // runtime 네임스페이스 심볼을 자주 쓰니까 짧게 가져온다.
  using namespace lightning_core::runtime;

  // 1) 디바이스 조회 테스트: CUDA가 없으면 NotSupported여도 정상 시나리오.
  int device_count = -1;
  const Status status = getDeviceCount(&device_count);

  if (status == Status::kSuccess) {
    std::cout << "CUDA devices: " << device_count << "\n";
  } else {
    std::cout << "CUDA unavailable in this build/runtime: " << getErrorString(status) << "\n";
  }

  std::cout << "Metal available: " << (isMetalAvailable() ? "yes" : "no") << "\n";
  std::cout << "Memory model: " << memoryModelName(deviceMemoryModel()) << "\n";
  std::cout << "Preferred(training): " << static_cast<int>(preferredDeviceFor(WorkloadKind::kTraining)) << "\n";
  std::cout << "Preferred(inference): " << static_cast<int>(preferredDeviceFor(WorkloadKind::kInference)) << "\n";

  // 2) malloc/free 스모크 테스트.
  // CUDA가 있으면 실제 할당/해제 확인, 없으면 NotSupported를 기대한다.
  void* ptr = nullptr;
  Status alloc_status = mallocDevice(&ptr, 1024);
  if (alloc_status == Status::kSuccess) {
    Status free_status = freeDevice(ptr);
    if (free_status != Status::kSuccess) {
      std::cerr << "freeDevice failed: " << getErrorString(free_status) << "\n";
      return 1;
    }
  } else if (alloc_status != Status::kNotSupported) {
    std::cerr << "mallocDevice failed unexpectedly: " << getErrorString(alloc_status) << "\n";
    return 1;
  }

  // 3) runtime trace 스모크 테스트.
  clearRuntimeTraceEvents();
  setRuntimeTraceEnabled(true);
  (void)backendName();
  (void)isMetalAvailable();
  (void)isCudaAvailable();
  (void)getDeviceCount(&device_count);
  setRuntimeTraceEnabled(false);

  const auto events = runtimeTraceEvents();
  if (events.empty()) {
    std::cerr << "runtime trace should contain events, but it is empty\n";
    return 1;
  }
  bool saw_backend_name = false;
  for (const auto& ev : events) {
    if (ev.type == RuntimeTraceEventType::kBackendName) {
      saw_backend_name = true;
      break;
    }
  }
  if (!saw_backend_name) {
    std::cerr << "runtime trace missing backend_name event\n";
    return 1;
  }
  clearRuntimeTraceEvents();

  return 0;
}
