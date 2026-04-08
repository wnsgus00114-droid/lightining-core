#include <cmath>
#include <iostream>
#include <vector>

#include "lightning_core/tensor.hpp"

namespace {

using Status = lightning_core::runtime::Status;

bool nearlyEqual(float a, float b) {
  float diff = a - b;
  if (diff < 0.0f) {
    diff = -diff;
  }
  return diff < 1e-6f;
}

int runCpuContractFreeze() {
  using Tensor = lightning_core::TensorT<float>;
  using TensorView = lightning_core::TensorViewT<float>;

  // Shape contract: empty or non-positive dims must be invalid.
  Tensor invalid_empty(std::vector<std::int64_t>{}, lightning_core::Device::kCPU);
  if (invalid_empty.validateContract() != Status::kInvalidValue) {
    std::cerr << "empty shape contract should be invalid\n";
    return 1;
  }
  Tensor invalid_non_positive(std::vector<std::int64_t>{2, -1}, lightning_core::Device::kCPU);
  if (invalid_non_positive.validateContract() != Status::kInvalidValue) {
    std::cerr << "non-positive dim contract should be invalid\n";
    return 1;
  }

  Tensor t(std::vector<std::int64_t>{2, 3}, lightning_core::Device::kCPU);
  if (t.validateContract() != Status::kSuccess) {
    std::cerr << "valid tensor contract should succeed\n";
    return 1;
  }

  // Lifetime contract: tensor must own copied data, not alias caller vector lifetime.
  std::vector<float> src{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  if (t.fromHost(src) != Status::kSuccess) {
    std::cerr << "fromHost failed\n";
    return 1;
  }
  src[0] = 777.0f;

  std::vector<float> host_values;
  if (t.toHost(&host_values) != Status::kSuccess) {
    std::cerr << "toHost failed\n";
    return 1;
  }
  if (host_values.empty() || nearlyEqual(host_values[0], 777.0f)) {
    std::cerr << "tensor storage should not alias caller source vector\n";
    return 1;
  }

  // Lifetime contract: toHost output is a detached copy.
  host_values[1] = 888.0f;
  std::vector<float> host_values2;
  if (t.toHost(&host_values2) != Status::kSuccess) {
    std::cerr << "second toHost failed\n";
    return 1;
  }
  if (host_values2.size() < 2 || nearlyEqual(host_values2[1], 888.0f)) {
    std::cerr << "toHost result should not alias mutable caller buffer\n";
    return 1;
  }

  // Layout contract: slice should produce strided view with valid non-zero offset.
  TensorView sliced;
  if (t.slice(0, 1, 2, &sliced) != Status::kSuccess) {
    std::cerr << "slice should succeed\n";
    return 1;
  }
  if (sliced.layout() != lightning_core::Layout::kStrided || sliced.offsetElements() != 3) {
    std::cerr << "slice layout/offset contract invalid\n";
    return 1;
  }
  if (t.validateViewContract(sliced) != Status::kSuccess) {
    std::cerr << "slice view contract should be valid\n";
    return 1;
  }

  std::vector<float> slice_values;
  if (t.toHostView(sliced, &slice_values) != Status::kSuccess) {
    std::cerr << "toHostView failed\n";
    return 1;
  }
  if (slice_values.size() != 3 || !nearlyEqual(slice_values[0], 4.0f) || !nearlyEqual(slice_values[2], 6.0f)) {
    std::cerr << "toHostView values mismatch\n";
    return 1;
  }

  // Alias contract: views are metadata aliases over tensor storage (read path reflects updates).
  if (t.fromHost(std::vector<float>{10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f}) != Status::kSuccess) {
    std::cerr << "fromHost update failed\n";
    return 1;
  }
  std::vector<float> slice_values_updated;
  if (t.toHostView(sliced, &slice_values_updated) != Status::kSuccess) {
    std::cerr << "toHostView after update failed\n";
    return 1;
  }
  if (slice_values_updated.size() != 3 || !nearlyEqual(slice_values_updated[0], 13.0f) ||
      !nearlyEqual(slice_values_updated[2], 15.0f)) {
    std::cerr << "slice view should reflect latest tensor storage values\n";
    return 1;
  }

  // Shape/layout regression guards.
  TensorView flat;
  if (t.view(std::vector<std::int64_t>{6}, &flat) != Status::kSuccess) {
    std::cerr << "flatten view should succeed\n";
    return 1;
  }
  if (t.view(std::vector<std::int64_t>{5}, &flat) != Status::kInvalidValue) {
    std::cerr << "mismatched view shape should fail\n";
    return 1;
  }
  if (t.readStrided(std::vector<std::int64_t>{2, 3}, std::vector<std::int64_t>{3}, 0, &slice_values) !=
      Status::kInvalidValue) {
    std::cerr << "readStrided rank mismatch should fail\n";
    return 1;
  }
  if (t.readStrided(std::vector<std::int64_t>{2, 3}, std::vector<std::int64_t>{3, 0}, 0, &slice_values) !=
      Status::kInvalidValue) {
    std::cerr << "readStrided zero stride should fail\n";
    return 1;
  }
  if (t.readStrided(std::vector<std::int64_t>{2, 3}, std::vector<std::int64_t>{3, -1}, 0, &slice_values) !=
      Status::kInvalidValue) {
    std::cerr << "readStrided negative stride should fail\n";
    return 1;
  }
  if (flat.validateContractForStorage(2) != Status::kInvalidValue) {
    std::cerr << "view contract should fail for insufficient storage bounds\n";
    return 1;
  }

  if (lightning_core::runtime::isMetalAvailable()) {
    lightning_core::TensorT<float> metal_t(std::vector<std::int64_t>{2, 3}, lightning_core::Device::kMetal);
    if (metal_t.validateViewContract(flat) != Status::kInvalidValue) {
      std::cerr << "cross-device view contract should fail\n";
      return 1;
    }
  }

  return 0;
}

}  // namespace

int main() {
  return runCpuContractFreeze();
}
