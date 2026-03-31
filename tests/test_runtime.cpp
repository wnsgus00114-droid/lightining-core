#include <iostream>
#include <vector>

#include "lightning_core/ops.hpp"
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

  // op-dispatch trace smoke: a tiny matmul should emit op_dispatch metadata.
  setRuntimeTraceEnabled(true);
  const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> b = {5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> c(4, 0.0f);
  if (lightning_core::ops::matMul<float>(a.data(), b.data(), c.data(), 2, 2, 2, Device::kCPU) != Status::kSuccess) {
    std::cerr << "matMul CPU failed in runtime trace smoke\n";
    return 1;
  }
  setRuntimeTraceEnabled(false);
  bool saw_op_dispatch = false;
  bool saw_matmul_dispatch = false;
  for (const auto& ev : runtimeTraceEvents()) {
    if (ev.type == RuntimeTraceEventType::kOpDispatch) {
      saw_op_dispatch = true;
      if (static_cast<RuntimeTraceOpKind>(ev.detail0) == RuntimeTraceOpKind::kMatMul) {
        saw_matmul_dispatch = true;
      }
    }
  }
  if (!saw_op_dispatch || !saw_matmul_dispatch) {
    std::cerr << "runtime trace missing op_dispatch/matmul metadata\n";
    return 1;
  }
  clearRuntimeTraceEvents();

  // 4) 기본 sync policy 제어 스모크 테스트.
  SyncPolicy never_policy;
  never_policy.mode = SyncMode::kNever;
  never_policy.trace_sync_boundary = true;
  setDefaultSyncPolicy(never_policy);
  const SyncPolicy loaded_policy = defaultSyncPolicy();
  if (loaded_policy.mode != SyncMode::kNever || !loaded_policy.trace_sync_boundary) {
    std::cerr << "default sync policy round-trip mismatch\n";
    return 1;
  }
  if (applyDefaultSyncPolicy() != Status::kSuccess) {
    std::cerr << "applyDefaultSyncPolicy(never) should succeed\n";
    return 1;
  }

  SyncPolicy always_policy;
  always_policy.mode = SyncMode::kAlways;
  always_policy.trace_sync_boundary = true;
  clearRuntimeTraceEvents();
  setRuntimeTraceEnabled(true);
  const Status sync_status = deviceSynchronizeWithPolicy(always_policy);
  setRuntimeTraceEnabled(false);
  if (sync_status != Status::kSuccess && sync_status != Status::kNotSupported) {
    std::cerr << "deviceSynchronizeWithPolicy(always) unexpected status: " << getErrorString(sync_status)
              << "\n";
    return 1;
  }
  bool saw_apply_sync = false;
  for (const auto& ev : runtimeTraceEvents()) {
    if (ev.type == RuntimeTraceEventType::kApplySyncPolicy) {
      saw_apply_sync = true;
      break;
    }
  }
  if (!saw_apply_sync) {
    std::cerr << "runtime trace missing apply_sync_policy event\n";
    return 1;
  }
  clearRuntimeTraceEvents();

  SyncPolicy auto_policy;
  auto_policy.mode = SyncMode::kAuto;
  auto_policy.trace_sync_boundary = false;
  setDefaultSyncPolicy(auto_policy);

  // 5) backend capability 계약 스모크 테스트.
  const BackendCapabilities cpu_caps = backendCapabilities(Device::kCPU);
  if (!cpu_caps.available || !cpu_caps.compute_surface) {
    std::cerr << "CPU capability contract invalid\n";
    return 1;
  }
  if (!cpu_caps.runtime_trace_surface || !cpu_caps.sync_policy_surface) {
    std::cerr << "CPU capability missing runtime control surfaces\n";
    return 1;
  }

  const BackendCapabilities active_caps = activeBackendCapabilities();
  if (!active_caps.available) {
    std::cerr << "active backend should be available\n";
    return 1;
  }

  if (isMetalAvailable()) {
    const BackendCapabilities metal_caps = backendCapabilities(Device::kMetal);
    if (!metal_caps.built || !metal_caps.available || !metal_caps.compute_surface || !metal_caps.memory_surface) {
      std::cerr << "Metal capability contract invalid\n";
      return 1;
    }
  }

  return 0;
}
