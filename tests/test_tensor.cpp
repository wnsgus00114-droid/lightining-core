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

  TensorT a({4}, device);
  TensorT b({4}, device);
  TensorT out({4}, device);

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
