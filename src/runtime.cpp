#include "cudajun/runtime.hpp"

#include <array>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>

#if CJ_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace cudajun::runtime {

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
    return Status::kInvalidValue;
  }
#if CJ_HAS_CUDA
  // CUDA 빌드에서는 진짜 cudaMalloc 호출.
  return fromCudaError(cudaMalloc(ptr, size_bytes));
#elif defined(CJ_HAS_METAL) && CJ_HAS_METAL
  // Metal-only builds expose CUDA-like allocation semantics via host-managed memory.
  *ptr = std::malloc(size_bytes);
  if (*ptr == nullptr) {
    return Status::kOutOfMemory;
  }
  return Status::kSuccess;
#else
  // CPU-only 빌드에서는 명시적으로 미지원 처리.
  (void)size_bytes;
  *ptr = nullptr;
  return Status::kNotSupported;
#endif
}

Status freeDevice(void* ptr) {
#if CJ_HAS_CUDA
  // nullptr free는 성공으로 취급(일반적인 CUDA/C 표준 관습).
  if (ptr == nullptr) {
    return Status::kSuccess;
  }
  return fromCudaError(cudaFree(ptr));
#elif defined(CJ_HAS_METAL) && CJ_HAS_METAL
  std::free(ptr);
  return Status::kSuccess;
#else
  (void)ptr;
  return Status::kNotSupported;
#endif
}

Status memcpy(void* dst, const void* src, std::size_t size_bytes, MemcpyKind kind) {
  // 기본 방어 로직.
  if (dst == nullptr || src == nullptr) {
    return Status::kInvalidValue;
  }

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
  return fromCudaError(cudaMemcpy(dst, src, size_bytes, cuda_kind));
#elif defined(CJ_HAS_METAL) && CJ_HAS_METAL
  (void)kind;
  std::memcpy(dst, src, size_bytes);
  return Status::kSuccess;
#else
  // CPU-only에서는 HostToHost만 memcpy 허용.
  if (kind != MemcpyKind::kHostToHost) {
    return Status::kNotSupported;
  }
  std::memcpy(dst, src, size_bytes);
  return Status::kSuccess;
#endif
}

Status deviceSynchronize() {
#if CJ_HAS_CUDA
  // 커널 런치 비동기 특성 때문에 필요할 때 전체 동기화.
  return fromCudaError(cudaDeviceSynchronize());
#elif defined(CJ_HAS_METAL) && CJ_HAS_METAL
  return Status::kSuccess;
#else
  return Status::kNotSupported;
#endif
}

Status getDeviceCount(int* count) {
  // 출력 포인터 검증.
  if (count == nullptr) {
    return Status::kInvalidValue;
  }
#if CJ_HAS_CUDA
  return fromCudaError(cudaGetDeviceCount(count));
#elif defined(CJ_HAS_METAL) && CJ_HAS_METAL
  *count = 1;
  return Status::kSuccess;
#else
  *count = 0;
  return Status::kNotSupported;
#endif
}

bool isCudaAvailable() {
  ensureRuntimeProfileLoaded();
#if CJ_HAS_CUDA
  // 런타임에서 실제 디바이스 개수까지 확인.
  int count = 0;
  return getDeviceCount(&count) == Status::kSuccess && count > 0;
#else
  return false;
#endif
}

bool isMetalAvailable() {
  ensureRuntimeProfileLoaded();
#if defined(CJ_HAS_METAL) && CJ_HAS_METAL
  return true;
#else
  return false;
#endif
}

Device preferredDeviceFor(WorkloadKind workload) {
  ensureRuntimeProfileLoaded();
  if (workload == WorkloadKind::kInference) {
    if (isMetalAvailable()) {
      return Device::kMetal;
    }
    if (isCudaAvailable()) {
      return Device::kCUDA;
    }
    return Device::kCPU;
  }

  // 학습은 현재 커스텀 커널/그래디언트 경로 제약 때문에 Metal/CUDA를 우선한다.
  if (isMetalAvailable()) {
    return Device::kMetal;
  }
  if (isCudaAvailable()) {
    return Device::kCUDA;
  }
  return Device::kCPU;
}

std::string backendName() {
  ensureRuntimeProfileLoaded();
  // 아주 단순하지만 디버깅 로그에서 꽤 유용한 정보.
  if (isMetalAvailable()) {
    return "metal";
  }
  if (isCudaAvailable()) {
    return "cuda";
  }
  return "cpu";
}


void preloadRuntimeProfileEnv() {
  ensureRuntimeProfileLoaded();
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

}  // namespace cudajun::runtime
