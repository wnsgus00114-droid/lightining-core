#pragma once

#include <cstddef>
#include <string>

namespace cudajun::runtime {

// 연산을 실행할 대상 장치.
enum class Device {
  kCPU = 0,
  kCUDA,
  kMetal
};

enum class WorkloadKind {
  kTraining = 0,
  kInference
};

// 런타임 계층의 공통 상태 코드.
// CUDA가 있든 없든 동일한 enum을 쓰면 상위 코드가 단순해진다.
enum class Status {
  kSuccess = 0,
  kNotInitialized,
  kInvalidValue,
  kOutOfMemory,
  kNotSupported,
  kDriverError,
  kUnknown
};

// 메모리 복사 방향.
// CUDA 스타일 API 감성을 유지하려고 kind를 명시적으로 받는다.
enum class MemcpyKind {
  kHostToHost = 0,
  kHostToDevice,
  kDeviceToHost,
  kDeviceToDevice
};

// 실제 백엔드에서 device allocation이 어떤 방식으로 구현되는지 설명한다.
enum class MemoryModel {
  kNativeDevice = 0,
  kHostManagedCompat
};

// 디바이스 메모리 할당.
// CUDA 환경이면 cudaMalloc, 아니면 NotSupported를 반환한다.
Status mallocDevice(void** ptr, std::size_t size_bytes);

// 디바이스 메모리 해제.
// nullptr는 안전하게 no-op 처리되는 방향을 따른다.
Status freeDevice(void* ptr);

// 방향(kind) 기반 메모리 복사.
// CPU-only 빌드에서는 HostToHost만 허용한다.
Status memcpy(void* dst, const void* src, std::size_t size_bytes, MemcpyKind kind);

// 디바이스 동기화(커널 완료 대기).
Status deviceSynchronize();

// 가용 디바이스 개수 조회.
Status getDeviceCount(int* count);

// 실제 CUDA 런타임 사용 가능 여부.
bool isCudaAvailable();

// Metal 백엔드 사용 가능 여부.
bool isMetalAvailable();

// workload 성격에 맞는 기본 실행 디바이스 추천.
Device preferredDeviceFor(WorkloadKind workload);

// 현재 백엔드 이름("metal"/"cuda"/"cpu").
std::string backendName();

// 현재 빌드/런타임의 device memory 모델.
MemoryModel deviceMemoryModel();

// 메모리 모델 설명 문자열.
const char* memoryModelName(MemoryModel model);

// Load runtime tuning profile env file once (if configured/found).
// Existing environment variables are kept as-is.
void preloadRuntimeProfileEnv();

// Status를 사람이 읽을 수 있는 문자열로 변환.
const char* getErrorString(Status status);

}  // namespace cudajun::runtime
