#include <cmath>
#include <iostream>
#include <type_traits>
#include <vector>

#include "cudajun/tensor.hpp"

template <typename T>
bool nearlyEqual(T a, T b) {
  long double diff = static_cast<long double>(a) - static_cast<long double>(b);
  if (diff < 0) {
    diff = -diff;
  }
  return diff < static_cast<long double>(1e-6);
}

template <typename T>
int runCase(cudajun::Device device) {
  using TensorT = cudajun::TensorT<T>;

  TensorT a({2, 2}, device);
  TensorT b({2, 2}, device);
  TensorT out({2, 2}, device);

  if (a.rank() != 2 || !a.isContiguous()) {
    std::cerr << "rank/contiguous check failed\n";
    return 1;
  }
  const auto& s = a.strides();
  if (s.size() != 2 || s[0] != 2 || s[1] != 1) {
    std::cerr << "stride check failed\n";
    return 1;
  }

  if (a.fromHost({static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4)}) !=
      cudajun::runtime::Status::kSuccess) {
    std::cerr << "fromHost failed for tensor a\n";
    return 1;
  }
  if (b.fromHost({static_cast<T>(10), static_cast<T>(20), static_cast<T>(30), static_cast<T>(40)}) !=
      cudajun::runtime::Status::kSuccess) {
    std::cerr << "fromHost failed for tensor b\n";
    return 1;
  }

  if (a.add(b, &out) != cudajun::runtime::Status::kSuccess) {
    std::cerr << "Tensor::add failed\n";
    return 1;
  }

  std::vector<T> values;
  if (out.toHost(&values) != cudajun::runtime::Status::kSuccess) {
    std::cerr << "toHost failed\n";
    return 1;
  }

  const std::vector<T> expected{static_cast<T>(11), static_cast<T>(22), static_cast<T>(33), static_cast<T>(44)};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (!nearlyEqual(values[i], expected[i])) {
      std::cerr << "Mismatch at " << i << "\n";
      return 1;
    }
  }

  if (out.reshape({4}) != cudajun::runtime::Status::kSuccess) {
    std::cerr << "reshape to 1D failed\n";
    return 1;
  }
  if (out.rank() != 1 || out.numel() != 4) {
    std::cerr << "reshape state invalid\n";
    return 1;
  }
  if (out.reshape({3}) != cudajun::runtime::Status::kInvalidValue) {
    std::cerr << "reshape mismatch should fail\n";
    return 1;
  }

  cudajun::TensorViewT<T> v;
  if (a.view({4}, &v) != cudajun::runtime::Status::kSuccess) {
    std::cerr << "view should succeed\n";
    return 1;
  }
  if (v.rank() != 1 || !v.isContiguous() || v.numel() != 4) {
    std::cerr << "view metadata invalid\n";
    return 1;
  }

  cudajun::TensorViewT<T> sview;
  if (a.slice(0, 0, 1, &sview) != cudajun::runtime::Status::kSuccess) {
    std::cerr << "slice should succeed\n";
    return 1;
  }
  if (sview.rank() != 2 || sview.shape()[0] != 1 || sview.shape()[1] != 2) {
    std::cerr << "slice shape invalid\n";
    return 1;
  }
  if (sview.offsetElements() != 0) {
    std::cerr << "slice offset invalid\n";
    return 1;
  }
  if (a.slice(0, 1, 3, &sview) != cudajun::runtime::Status::kInvalidValue) {
    std::cerr << "slice out-of-range should fail\n";
    return 1;
  }

  std::vector<T> slice_values;
  if (a.toHostView(sview, &slice_values) != cudajun::runtime::Status::kSuccess) {
    std::cerr << "toHostView failed\n";
    return 1;
  }
  const std::vector<T> slice_expected{static_cast<T>(1), static_cast<T>(2)};
  if (slice_values.size() != slice_expected.size()) {
    std::cerr << "toHostView size mismatch\n";
    return 1;
  }
  for (std::size_t i = 0; i < slice_expected.size(); ++i) {
    if (!nearlyEqual(slice_values[i], slice_expected[i])) {
      std::cerr << "toHostView value mismatch\n";
      return 1;
    }
  }

  std::vector<T> copy_values;
  if (a.sliceCopy(1, 0, 1, &copy_values) != cudajun::runtime::Status::kSuccess) {
    std::cerr << "sliceCopy failed\n";
    return 1;
  }
  const std::vector<T> copy_expected{static_cast<T>(1), static_cast<T>(3)};
  for (std::size_t i = 0; i < copy_expected.size(); ++i) {
    if (!nearlyEqual(copy_values[i], copy_expected[i])) {
      std::cerr << "sliceCopy value mismatch\n";
      return 1;
    }
  }

  std::vector<T> strided_values;
  if (a.readStrided({2}, {2}, 0, &strided_values) != cudajun::runtime::Status::kSuccess) {
    std::cerr << "readStrided failed\n";
    return 1;
  }
  for (std::size_t i = 0; i < copy_expected.size(); ++i) {
    if (!nearlyEqual(strided_values[i], copy_expected[i])) {
      std::cerr << "readStrided value mismatch\n";
      return 1;
    }
  }

  return 0;
}

int main() {
  using Device = cudajun::Device;

  if (runCase<float>(Device::kCPU) != 0) {
    return 1;
  }
  if (runCase<double>(Device::kCPU) != 0) {
    return 1;
  }
  if (runCase<long double>(Device::kCPU) != 0) {
    return 1;
  }

  if (cudajun::runtime::isMetalAvailable()) {
    if (runCase<float>(Device::kMetal) != 0) {
      return 1;
    }
  }

  return 0;
}
