#include "lightning_core/core/runtime.hpp"

#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>

#if CJ_HAS_CUDA
#include <cuda_runtime.h>
#endif

#if defined(CJ_HAS_METAL) && CJ_HAS_METAL && defined(__APPLE__)
extern "C" void* MTLCreateSystemDefaultDevice(void);
#endif

namespace lightning_core::runtime {

namespace {

std::string trimCopy(const std::string& s) {
  std::size_t b = 0;
  while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b])) != 0) {
    ++b;
  }
  std::size_t e = s.size();
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])) != 0) {
    --e;
  }
  return s.substr(b, e - b);
}

bool setEnvIfMissing(const std::string& key, const std::string& value) {
  const char* existing = std::getenv(key.c_str());
  if (existing != nullptr && existing[0] != '\0') {
    return false;
  }
#if defined(_WIN32)
  return _putenv_s(key.c_str(), value.c_str()) == 0;
#else
  return ::setenv(key.c_str(), value.c_str(), 0) == 0;
#endif
}

bool envValueMeansFalse(const char* raw) {
  if (raw == nullptr) {
    return false;
  }
  std::string value = trimCopy(raw);
  if (value.empty()) {
    return false;
  }
  for (char& ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value == "0" || value == "false" || value == "off" || value == "no";
}

constexpr std::size_t kRuntimeTraceCapacity = 4096;
std::atomic<bool> g_runtime_trace_enabled{false};
std::mutex g_runtime_trace_mu;
std::vector<RuntimeTraceEvent> g_runtime_trace_events;
std::size_t g_runtime_trace_write_idx = 0;
std::atomic<int> g_default_sync_mode{static_cast<int>(SyncMode::kAuto)};
std::atomic<bool> g_default_sync_trace_boundary{false};

std::uint64_t nowSteadyNs() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

void recordRuntimeTraceEvent(RuntimeTraceEventType type,
                             Status status,
                             std::size_t size_bytes = 0,
                             int detail0 = 0,
                             int detail1 = 0) {
  if (!g_runtime_trace_enabled.load(std::memory_order_relaxed)) {
    return;
  }
  RuntimeTraceEvent ev;
  ev.type = type;
  ev.status = status;
  ev.timestamp_ns = nowSteadyNs();
  ev.size_bytes = size_bytes;
  ev.detail0 = detail0;
  ev.detail1 = detail1;

  std::lock_guard<std::mutex> lock(g_runtime_trace_mu);
  if (g_runtime_trace_events.size() < kRuntimeTraceCapacity) {
    g_runtime_trace_events.push_back(ev);
    return;
  }
  g_runtime_trace_events[g_runtime_trace_write_idx] = ev;
  g_runtime_trace_write_idx = (g_runtime_trace_write_idx + 1) % kRuntimeTraceCapacity;
}

std::string resolveRuntimeProfilePath() {
  if (const char* raw = std::getenv("CJ_RUNTIME_PROFILE_ENV_FILE")) {
    if (raw[0] != '\0') {
      return std::string(raw);
    }
  }

  // Try several cwd-relative paths because benchmarks may be launched from
  // workspace root, build/, or build/benchmarks.
  static constexpr std::array<const char*, 7> kCandidates = {
      "build/benchmarks/model_runtime_profile.env",
      "../build/benchmarks/model_runtime_profile.env",
      "../../build/benchmarks/model_runtime_profile.env",
      "model_runtime_profile.env",
      "../model_runtime_profile.env",
      "benchmarks/model_runtime_profile.env",
      "../benchmarks/model_runtime_profile.env"};

  for (const char* candidate : kCandidates) {
    std::ifstream in(candidate);
    if (in.good()) {
      return std::string(candidate);
    }
  }
  return std::string();
}

void ensureRuntimeProfileLoaded() {
  static std::once_flag once;
  std::call_once(once, []() {
    const char* autoload = std::getenv("CJ_RUNTIME_PROFILE_AUTOLOAD");
    if (envValueMeansFalse(autoload)) {
      return;
    }

    std::string path = resolveRuntimeProfilePath();
    if (path.empty()) {
      return;
    }

    std::ifstream in(path);
    if (!in.is_open()) {
      return;
    }

    std::string line;
    while (std::getline(in, line)) {
      std::string t = trimCopy(line);
      if (t.empty() || t[0] == '#') {
        continue;
      }
      std::size_t eq = t.find('=');
      if (eq == std::string::npos || eq == 0 || eq + 1 >= t.size()) {
        continue;
      }
      std::string key = trimCopy(t.substr(0, eq));
      std::string val = trimCopy(t.substr(eq + 1));
      if (key.empty() || val.empty()) {
        continue;
      }
      // Only import runtime tuning variables from this profile file.
      if (key.rfind("CJ_", 0) != 0) {
        continue;
      }
      (void)setEnvIfMissing(key, val);
    }
  });
}

#if CJ_HAS_CUDA
// CUDA 에러 코드를 우리 Status enum으로 변환한다.
// 상위 레이어는 CUDA 헤더를 몰라도 되게 만드는 번역기 역할.
Status fromCudaError(cudaError_t err) {
  switch (err) {
    case cudaSuccess:
      return Status::kSuccess;
    case cudaErrorInvalidValue:
      return Status::kInvalidValue;
    case cudaErrorMemoryAllocation:
      return Status::kOutOfMemory;
    case cudaErrorInitializationError:
      return Status::kNotInitialized;
    case cudaErrorInsufficientDriver:
      return Status::kDriverError;
    default:
      return Status::kUnknown;
  }
}
#endif

}  // namespace

Status mallocDevice(void** ptr, std::size_t size_bytes) {
  // 포인터나 크기가 이상하면 즉시 컷.
  if (ptr == nullptr || size_bytes == 0) {
    recordRuntimeTraceEvent(RuntimeTraceEventType::kMallocDevice, Status::kInvalidValue, size_bytes);
    return Status::kInvalidValue;
  }
  Status st = Status::kUnknown;
#if CJ_HAS_CUDA
  // CUDA 빌드에서는 진짜 cudaMalloc 호출.
  st = fromCudaError(cudaMalloc(ptr, size_bytes));
#elif defined(CJ_HAS_METAL) && CJ_HAS_METAL
  // Metal-only builds expose CUDA-like allocation semantics via host-managed memory.
  *ptr = std::malloc(size_bytes);
  if (*ptr == nullptr) {
    st = Status::kOutOfMemory;
  } else {
    st = Status::kSuccess;
  }
#else
  // CPU-only 빌드에서는 명시적으로 미지원 처리.
  (void)size_bytes;
  *ptr = nullptr;
  st = Status::kNotSupported;
#endif
  recordRuntimeTraceEvent(RuntimeTraceEventType::kMallocDevice, st, size_bytes);
  return st;
}

Status freeDevice(void* ptr) {
  Status st = Status::kUnknown;
#if CJ_HAS_CUDA
  // nullptr free는 성공으로 취급(일반적인 CUDA/C 표준 관습).
  if (ptr == nullptr) {
    st = Status::kSuccess;
  } else {
    st = fromCudaError(cudaFree(ptr));
  }
#elif defined(CJ_HAS_METAL) && CJ_HAS_METAL
  std::free(ptr);
  st = Status::kSuccess;
#else
  (void)ptr;
  st = Status::kNotSupported;
#endif
  recordRuntimeTraceEvent(RuntimeTraceEventType::kFreeDevice, st);
  return st;
}

Status memcpy(void* dst, const void* src, std::size_t size_bytes, MemcpyKind kind) {
  // 기본 방어 로직.
  if (dst == nullptr || src == nullptr) {
    recordRuntimeTraceEvent(RuntimeTraceEventType::kMemcpy, Status::kInvalidValue, size_bytes,
                            static_cast<int>(kind));
    return Status::kInvalidValue;
  }

  Status st = Status::kUnknown;
#if CJ_HAS_CUDA
  // 우리 enum -> cudaMemcpyKind 변환.
  cudaMemcpyKind cuda_kind = cudaMemcpyDefault;
  switch (kind) {
    case MemcpyKind::kHostToHost:
      cuda_kind = cudaMemcpyHostToHost;
      break;
    case MemcpyKind::kHostToDevice:
      cuda_kind = cudaMemcpyHostToDevice;
      break;
    case MemcpyKind::kDeviceToHost:
      cuda_kind = cudaMemcpyDeviceToHost;
      break;
    case MemcpyKind::kDeviceToDevice:
      cuda_kind = cudaMemcpyDeviceToDevice;
      break;
  }
  // 실제 복사 실행.
  st = fromCudaError(cudaMemcpy(dst, src, size_bytes, cuda_kind));
#elif defined(CJ_HAS_METAL) && CJ_HAS_METAL
  (void)kind;
  std::memcpy(dst, src, size_bytes);
  st = Status::kSuccess;
#else
  // CPU-only에서는 HostToHost만 memcpy 허용.
  if (kind != MemcpyKind::kHostToHost) {
    st = Status::kNotSupported;
  } else {
    std::memcpy(dst, src, size_bytes);
    st = Status::kSuccess;
  }
#endif
  recordRuntimeTraceEvent(RuntimeTraceEventType::kMemcpy, st, size_bytes, static_cast<int>(kind));
  return st;
}

Status deviceSynchronize() {
  Status st = Status::kUnknown;
#if CJ_HAS_CUDA
  // 커널 런치 비동기 특성 때문에 필요할 때 전체 동기화.
  st = fromCudaError(cudaDeviceSynchronize());
#elif defined(CJ_HAS_METAL) && CJ_HAS_METAL
  st = Status::kSuccess;
#else
  st = Status::kNotSupported;
#endif
  recordRuntimeTraceEvent(RuntimeTraceEventType::kDeviceSynchronize, st);
  return st;
}

Status getDeviceCount(int* count) {
  // 출력 포인터 검증.
  if (count == nullptr) {
    recordRuntimeTraceEvent(RuntimeTraceEventType::kGetDeviceCount, Status::kInvalidValue);
    return Status::kInvalidValue;
  }
  Status st = Status::kUnknown;
#if CJ_HAS_CUDA
  st = fromCudaError(cudaGetDeviceCount(count));
#elif defined(CJ_HAS_METAL) && CJ_HAS_METAL
  *count = 1;
  st = Status::kSuccess;
#else
  *count = 0;
  st = Status::kNotSupported;
#endif
  recordRuntimeTraceEvent(RuntimeTraceEventType::kGetDeviceCount, st, 0, *count);
  return st;
}

bool isCudaAvailable() {
  ensureRuntimeProfileLoaded();
  bool available = false;
#if CJ_HAS_CUDA
  // 런타임에서 실제 디바이스 개수까지 확인.
  int count = 0;
  available = (getDeviceCount(&count) == Status::kSuccess && count > 0);
#else
  available = false;
#endif
  recordRuntimeTraceEvent(RuntimeTraceEventType::kIsCudaAvailable,
                          Status::kSuccess,
                          0,
                          available ? 1 : 0);
  return available;
}

bool isMetalAvailable() {
  ensureRuntimeProfileLoaded();
  bool available = false;
#if defined(CJ_HAS_METAL) && CJ_HAS_METAL
  static bool probed = false;
  static bool cached_available = false;
  if (!probed) {
    cached_available = (MTLCreateSystemDefaultDevice() != nullptr);
    probed = true;
  }
  available = cached_available;
#else
  available = false;
#endif
  recordRuntimeTraceEvent(RuntimeTraceEventType::kIsMetalAvailable,
                          Status::kSuccess,
                          0,
                          available ? 1 : 0);
  return available;
}

Device preferredDeviceFor(WorkloadKind workload) {
  ensureRuntimeProfileLoaded();
  Device selected = Device::kCPU;
  if (workload == WorkloadKind::kInference) {
    if (isMetalAvailable()) {
      selected = Device::kMetal;
    } else if (isCudaAvailable()) {
      selected = Device::kCUDA;
    } else {
      selected = Device::kCPU;
    }
  } else {
    // 학습은 현재 커스텀 커널/그래디언트 경로 제약 때문에 Metal/CUDA를 우선한다.
    if (isMetalAvailable()) {
      selected = Device::kMetal;
    } else if (isCudaAvailable()) {
      selected = Device::kCUDA;
    } else {
      selected = Device::kCPU;
    }
  }
  recordRuntimeTraceEvent(RuntimeTraceEventType::kPreferredDeviceFor,
                          Status::kSuccess,
                          0,
                          static_cast<int>(workload),
                          static_cast<int>(selected));
  return selected;
}

std::string backendName() {
  ensureRuntimeProfileLoaded();
  // 아주 단순하지만 디버깅 로그에서 꽤 유용한 정보.
  std::string name = "cpu";
  if (isMetalAvailable()) {
    name = "metal";
  } else if (isCudaAvailable()) {
    name = "cuda";
  }
  recordRuntimeTraceEvent(RuntimeTraceEventType::kBackendName, Status::kSuccess);
  return name;
}

MemoryModel deviceMemoryModel() {
#if CJ_HAS_CUDA
  return MemoryModel::kNativeDevice;
#elif defined(CJ_HAS_METAL) && CJ_HAS_METAL
  return MemoryModel::kHostManagedCompat;
#else
  return MemoryModel::kHostManagedCompat;
#endif
}

const char* memoryModelName(MemoryModel model) {
  switch (model) {
    case MemoryModel::kNativeDevice:
      return "native-device";
    case MemoryModel::kHostManagedCompat:
    default:
      return "host-managed-compat";
  }
}


void preloadRuntimeProfileEnv() {
  ensureRuntimeProfileLoaded();
}

const char* syncModeName(SyncMode mode) {
  switch (mode) {
    case SyncMode::kAlways:
      return "always";
    case SyncMode::kNever:
      return "never";
    case SyncMode::kAuto:
    default:
      return "auto";
  }
}

void setDefaultSyncPolicy(const SyncPolicy& policy) {
  g_default_sync_mode.store(static_cast<int>(policy.mode), std::memory_order_relaxed);
  g_default_sync_trace_boundary.store(policy.trace_sync_boundary, std::memory_order_relaxed);
}

SyncPolicy defaultSyncPolicy() {
  SyncPolicy policy;
  const int mode_raw = g_default_sync_mode.load(std::memory_order_relaxed);
  switch (mode_raw) {
    case static_cast<int>(SyncMode::kAlways):
      policy.mode = SyncMode::kAlways;
      break;
    case static_cast<int>(SyncMode::kNever):
      policy.mode = SyncMode::kNever;
      break;
    case static_cast<int>(SyncMode::kAuto):
    default:
      policy.mode = SyncMode::kAuto;
      break;
  }
  policy.trace_sync_boundary = g_default_sync_trace_boundary.load(std::memory_order_relaxed);
  return policy;
}

Status deviceSynchronizeWithPolicy(const SyncPolicy& policy) {
  Status st = Status::kSuccess;
  switch (policy.mode) {
    case SyncMode::kNever:
      st = Status::kSuccess;
      break;
    case SyncMode::kAlways:
      st = deviceSynchronize();
      break;
    case SyncMode::kAuto:
    default:
      if (isCudaAvailable() || isMetalAvailable()) {
        st = deviceSynchronize();
      } else {
        st = Status::kSuccess;
      }
      break;
  }

  recordRuntimeTraceEvent(RuntimeTraceEventType::kApplySyncPolicy,
                          st,
                          0,
                          static_cast<int>(policy.mode),
                          policy.trace_sync_boundary ? 1 : 0);
  return st;
}

Status applyDefaultSyncPolicy() {
  return deviceSynchronizeWithPolicy(defaultSyncPolicy());
}

void setRuntimeTraceEnabled(bool enabled) {
  g_runtime_trace_enabled.store(enabled, std::memory_order_relaxed);
}

bool isRuntimeTraceEnabled() {
  return g_runtime_trace_enabled.load(std::memory_order_relaxed);
}

void clearRuntimeTraceEvents() {
  std::lock_guard<std::mutex> lock(g_runtime_trace_mu);
  g_runtime_trace_events.clear();
  g_runtime_trace_write_idx = 0;
}

std::vector<RuntimeTraceEvent> runtimeTraceEvents() {
  std::lock_guard<std::mutex> lock(g_runtime_trace_mu);
  if (g_runtime_trace_events.size() < kRuntimeTraceCapacity || g_runtime_trace_write_idx == 0) {
    return g_runtime_trace_events;
  }

  std::vector<RuntimeTraceEvent> ordered;
  ordered.reserve(g_runtime_trace_events.size());
  for (std::size_t i = g_runtime_trace_write_idx; i < g_runtime_trace_events.size(); ++i) {
    ordered.push_back(g_runtime_trace_events[i]);
  }
  for (std::size_t i = 0; i < g_runtime_trace_write_idx; ++i) {
    ordered.push_back(g_runtime_trace_events[i]);
  }
  return ordered;
}

std::size_t runtimeTraceEventCapacity() {
  return kRuntimeTraceCapacity;
}

const char* runtimeTraceEventTypeName(RuntimeTraceEventType type) {
  switch (type) {
    case RuntimeTraceEventType::kMallocDevice:
      return "malloc_device";
    case RuntimeTraceEventType::kFreeDevice:
      return "free_device";
    case RuntimeTraceEventType::kMemcpy:
      return "memcpy";
    case RuntimeTraceEventType::kDeviceSynchronize:
      return "device_synchronize";
    case RuntimeTraceEventType::kApplySyncPolicy:
      return "apply_sync_policy";
    case RuntimeTraceEventType::kGetDeviceCount:
      return "get_device_count";
    case RuntimeTraceEventType::kIsCudaAvailable:
      return "is_cuda_available";
    case RuntimeTraceEventType::kIsMetalAvailable:
      return "is_metal_available";
    case RuntimeTraceEventType::kPreferredDeviceFor:
      return "preferred_device_for";
    case RuntimeTraceEventType::kBackendName:
    default:
      return "backend_name";
  }
}

const char* getErrorString(Status status) {
  // 로그 가독성용 문자열 맵.
  switch (status) {
    case Status::kSuccess:
      return "success";
    case Status::kNotInitialized:
      return "not initialized";
    case Status::kInvalidValue:
      return "invalid value";
    case Status::kOutOfMemory:
      return "out of memory";
    case Status::kNotSupported:
      return "not supported";
    case Status::kDriverError:
      return "driver error";
    case Status::kUnknown:
    default:
      return "unknown error";
  }
}

}  // namespace lightning_core::runtime
