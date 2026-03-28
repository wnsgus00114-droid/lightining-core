#pragma once

#include <cstddef>
#include <cstdlib>

namespace lightning_core::ops {

struct MatMulIoPolicy {
  bool upload_a = true;
  bool upload_b = true;
  bool download_out = true;
  bool synchronize = true;
};

struct MatrixElementwiseIoPolicy {
  bool upload_a = true;
  bool upload_b = true;
  bool download_out = true;
  bool synchronize = true;
};

struct VectorAddIoPolicy {
  bool upload_a = true;
  bool upload_b = true;
  bool download_out = true;
  bool synchronize = true;
};

inline MatMulIoPolicy makeMetalResidentStartPolicy() {
  MatMulIoPolicy p;
  p.upload_a = true;
  p.upload_b = true;
  p.download_out = false;
  p.synchronize = false;
  return p;
}

inline MatMulIoPolicy makeMetalResidentRunPolicy() {
  MatMulIoPolicy p;
  p.upload_a = false;
  p.upload_b = false;
  p.download_out = false;
  p.synchronize = false;
  return p;
}

inline MatMulIoPolicy makeMetalResidentFinishPolicy() {
  MatMulIoPolicy p;
  p.upload_a = false;
  p.upload_b = false;
  p.download_out = true;
  p.synchronize = true;
  return p;
}

inline MatMulIoPolicy makeMetalResidentSyncPolicy() {
  MatMulIoPolicy p;
  p.upload_a = false;
  p.upload_b = false;
  p.download_out = false;
  p.synchronize = true;
  return p;
}

inline MatrixElementwiseIoPolicy makeMetalElemwiseResidentStartPolicy() {
  MatrixElementwiseIoPolicy p;
  p.upload_a = true;
  p.upload_b = true;
  p.download_out = false;
  p.synchronize = false;
  return p;
}

inline MatrixElementwiseIoPolicy makeMetalElemwiseResidentRunPolicy() {
  MatrixElementwiseIoPolicy p;
  p.upload_a = false;
  p.upload_b = false;
  p.download_out = false;
  p.synchronize = false;
  return p;
}

inline MatrixElementwiseIoPolicy makeMetalElemwiseResidentFinishPolicy() {
  MatrixElementwiseIoPolicy p;
  p.upload_a = false;
  p.upload_b = false;
  p.download_out = true;
  p.synchronize = true;
  return p;
}

inline VectorAddIoPolicy makeMetalVectorResidentStartPolicy() {
  VectorAddIoPolicy p;
  p.upload_a = true;
  p.upload_b = true;
  p.download_out = false;
  p.synchronize = false;
  return p;
}

inline VectorAddIoPolicy makeMetalVectorResidentRunPolicy() {
  VectorAddIoPolicy p;
  p.upload_a = false;
  p.upload_b = false;
  p.download_out = false;
  p.synchronize = false;
  return p;
}

inline VectorAddIoPolicy makeMetalVectorResidentFinishPolicy() {
  VectorAddIoPolicy p;
  p.upload_a = false;
  p.upload_b = false;
  p.download_out = true;
  p.synchronize = true;
  return p;
}

inline std::size_t vectorAddOneShotCrossoverN() {
  const bool dynamic_refresh = []() {
    if (const char* raw = std::getenv("CJ_VECTORADD_CPU_CROSSOVER_DYNAMIC")) {
      return raw[0] == '1';
    }
    return false;
  }();

  if (dynamic_refresh) {
    std::size_t crossover_n = (8u << 20);
    if (const char* raw = std::getenv("CJ_VECTORADD_CPU_CROSSOVER_N")) {
      char* end = nullptr;
      unsigned long long parsed = std::strtoull(raw, &end, 10);
      if (end != raw && *end == '\0' && parsed > 0) {
        crossover_n = static_cast<std::size_t>(parsed);
      }
    }
    return crossover_n;
  }

  static const std::size_t value = []() {
    std::size_t crossover_n = (8u << 20);
    if (const char* raw = std::getenv("CJ_VECTORADD_CPU_CROSSOVER_N")) {
      char* end = nullptr;
      unsigned long long parsed = std::strtoull(raw, &end, 10);
      if (end != raw && *end == '\0' && parsed > 0) {
        crossover_n = static_cast<std::size_t>(parsed);
      }
    }
    return crossover_n;
  }();
  return value;
}

}  // namespace lightning_core::ops
