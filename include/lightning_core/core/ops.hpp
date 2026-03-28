#pragma once

#include <cstddef>
#include <type_traits>

#include "lightning_core/core/detail/ops_backend.hpp"
#include "lightning_core/core/ops/policy.hpp"
#include "lightning_core/core/runtime.hpp"

namespace lightning_core::ops {

template <typename T>
runtime::Status matMulWithPolicy(
		const T* a,
		const T* b,
		T* out,
		std::size_t m,
		std::size_t k,
		std::size_t n,
		runtime::Device device,
		const MatMulIoPolicy& policy);

template <typename T>
runtime::Status matrixSubWithPolicy(
		const T* a,
		const T* b,
		T* out,
		std::size_t rows,
		std::size_t cols,
		runtime::Device device,
		const MatrixElementwiseIoPolicy& policy);

template <typename T>
runtime::Status matrixDivWithPolicy(
		const T* a,
		const T* b,
		T* out,
		std::size_t rows,
		std::size_t cols,
		runtime::Device device,
		const MatrixElementwiseIoPolicy& policy);

template <typename T>
runtime::Status vectorAddWithPolicy(
		const T* a,
		const T* b,
		T* out,
		std::size_t n,
		runtime::Device device,
		const VectorAddIoPolicy& policy);

template <typename T>
runtime::Status vectorAdd(const T* a, const T* b, T* out, std::size_t n, runtime::Device device) {
	return vectorAddWithPolicy(a, b, out, n, device, VectorAddIoPolicy{});
}

template <typename T>
runtime::Status vectorAddMetalResidentStart(const T* a, const T* b, T* out, std::size_t n) {
	return vectorAddWithPolicy(a, b, out, n, runtime::Device::kMetal, makeMetalVectorResidentStartPolicy());
}

template <typename T>
runtime::Status vectorAddMetalResidentRun(const T* a, const T* b, T* out, std::size_t n) {
	return vectorAddWithPolicy(a, b, out, n, runtime::Device::kMetal, makeMetalVectorResidentRunPolicy());
}

template <typename T>
runtime::Status vectorAddMetalResidentFinish(const T* a, const T* b, T* out, std::size_t n) {
	return vectorAddWithPolicy(a, b, out, n, runtime::Device::kMetal, makeMetalVectorResidentFinishPolicy());
}

template <typename T>
runtime::Status vectorAddWithPolicy(
		const T* a,
		const T* b,
		T* out,
		std::size_t n,
		runtime::Device device,
		const VectorAddIoPolicy& policy) {
	static_assert(std::is_floating_point_v<T>, "vectorAdd<T> supports floating-point types only");
	if ((policy.upload_a && a == nullptr) || (policy.upload_b && b == nullptr) ||
			(policy.download_out && out == nullptr) || n == 0) {
		return runtime::Status::kInvalidValue;
	}

	if (device == runtime::Device::kCPU) {
		if (a == nullptr || b == nullptr || out == nullptr) {
			return runtime::Status::kInvalidValue;
		}
		return detail::vectorAddCpuWithPolicy(
				a,
				b,
				out,
				n,
				policy.upload_a,
				policy.upload_b,
				policy.download_out,
				policy.synchronize);
	}
	if (device == runtime::Device::kCUDA) {
		runtime::Status st = detail::vectorAddCudaWithPolicy(
				a,
				b,
				out,
				n,
				policy.upload_a,
				policy.upload_b,
				policy.download_out,
				policy.synchronize);
		if (st == runtime::Status::kNotSupported) {
			if (a == nullptr || b == nullptr || out == nullptr) {
				return runtime::Status::kNotSupported;
			}
			return detail::vectorAddCpu(a, b, out, n);
		}
		return st;
	}
	if (device == runtime::Device::kMetal) {
		// One-shot add (upload+download+sync) is transfer-bound for small/medium tensors.
		// Use a crossover threshold to decide CPU vs Metal for end-to-end latency.
		if (policy.upload_a && policy.upload_b && policy.download_out && policy.synchronize) {
			if (a == nullptr || b == nullptr || out == nullptr) {
				return runtime::Status::kInvalidValue;
			}
			const std::size_t crossover_n = vectorAddOneShotCrossoverN();
			if (n <= crossover_n) {
				return detail::vectorAddCpu(a, b, out, n);
			}
		}
		runtime::Status st = detail::vectorAddMetalWithPolicy(
				a,
				b,
				out,
				n,
				policy.upload_a,
				policy.upload_b,
				policy.download_out,
				policy.synchronize);
		if (st == runtime::Status::kNotSupported) {
			if (a == nullptr || b == nullptr || out == nullptr) {
				return runtime::Status::kNotSupported;
			}
			return detail::vectorAddCpu(a, b, out, n);
		}
		return st;
	}
	return runtime::Status::kInvalidValue;
}

template <typename T>
runtime::Status matMul(
		const T* a,
		const T* b,
		T* out,
		std::size_t m,
		std::size_t k,
		std::size_t n,
		runtime::Device device) {
	return matMulWithPolicy(a, b, out, m, k, n, device, MatMulIoPolicy{});
}

template <typename T>
runtime::Status matMulMetalResidentStart(
		const T* a,
		const T* b,
		T* out,
		std::size_t m,
		std::size_t k,
		std::size_t n) {
	return matMulWithPolicy(a, b, out, m, k, n, runtime::Device::kMetal, makeMetalResidentStartPolicy());
}

template <typename T>
runtime::Status matMulMetalResidentRun(
		const T* a,
		const T* b,
		T* out,
		std::size_t m,
		std::size_t k,
		std::size_t n) {
	return matMulWithPolicy(a, b, out, m, k, n, runtime::Device::kMetal, makeMetalResidentRunPolicy());
}

template <typename T>
runtime::Status matMulMetalResidentFinish(
		const T* a,
		const T* b,
		T* out,
		std::size_t m,
		std::size_t k,
		std::size_t n) {
	return matMulWithPolicy(a, b, out, m, k, n, runtime::Device::kMetal, makeMetalResidentFinishPolicy());
}

template <typename T>
runtime::Status matMulMetalResidentSync(
		const T* a,
		const T* b,
		T* out,
		std::size_t m,
		std::size_t k,
		std::size_t n) {
	return matMulWithPolicy(a, b, out, m, k, n, runtime::Device::kMetal, makeMetalResidentSyncPolicy());
}

template <typename T>
runtime::Status matMulWithPolicy(
		const T* a,
		const T* b,
		T* out,
		std::size_t m,
		std::size_t k,
		std::size_t n,
		runtime::Device device,
		const MatMulIoPolicy& policy) {
	static_assert(std::is_floating_point_v<T>, "matMul<T> supports floating-point types only");
	if ((policy.upload_a && a == nullptr) || (policy.upload_b && b == nullptr) ||
			(policy.download_out && out == nullptr) || m == 0 || k == 0 || n == 0) {
		return runtime::Status::kInvalidValue;
	}

	if (device == runtime::Device::kCPU) {
		if (a == nullptr || b == nullptr || out == nullptr) {
			return runtime::Status::kInvalidValue;
		}
		return detail::matMulCpuWithPolicy(
				a,
				b,
				out,
				m,
				k,
				n,
				policy.upload_a,
				policy.upload_b,
				policy.download_out,
				policy.synchronize);
	}
	if (device == runtime::Device::kCUDA) {
		runtime::Status st = detail::matMulCudaWithPolicy(
				a,
				b,
				out,
				m,
				k,
				n,
				policy.upload_a,
				policy.upload_b,
				policy.download_out,
				policy.synchronize);
		if (st == runtime::Status::kNotSupported) {
			if (a == nullptr || b == nullptr || out == nullptr) {
				return runtime::Status::kNotSupported;
			}
			return detail::matMulCpu(a, b, out, m, k, n);
		}
		return st;
	}
	if (device == runtime::Device::kMetal) {
		runtime::Status st = detail::matMulMetalWithPolicy(
				a,
				b,
				out,
				m,
				k,
				n,
				policy.upload_a,
				policy.upload_b,
				policy.download_out,
				policy.synchronize);
		if (st == runtime::Status::kNotSupported) {
			if (a == nullptr || b == nullptr || out == nullptr) {
				return runtime::Status::kNotSupported;
			}
			return detail::matMulCpu(a, b, out, m, k, n);
		}
		return st;
	}
	return runtime::Status::kInvalidValue;
}

template <typename T>
runtime::Status matrixSub(
		const T* a,
		const T* b,
		T* out,
		std::size_t rows,
		std::size_t cols,
		runtime::Device device) {
	return matrixSubWithPolicy(a, b, out, rows, cols, device, MatrixElementwiseIoPolicy{});
}

template <typename T>
runtime::Status matrixDiv(
		const T* a,
		const T* b,
		T* out,
		std::size_t rows,
		std::size_t cols,
		runtime::Device device) {
	return matrixDivWithPolicy(a, b, out, rows, cols, device, MatrixElementwiseIoPolicy{});
}

template <typename T>
runtime::Status matrixSubMetalResidentStart(
		const T* a,
		const T* b,
		T* out,
		std::size_t rows,
		std::size_t cols) {
	return matrixSubWithPolicy(
			a,
			b,
			out,
			rows,
			cols,
			runtime::Device::kMetal,
			makeMetalElemwiseResidentStartPolicy());
}

template <typename T>
runtime::Status matrixSubMetalResidentRun(
		const T* a,
		const T* b,
		T* out,
		std::size_t rows,
		std::size_t cols) {
	return matrixSubWithPolicy(
			a,
			b,
			out,
			rows,
			cols,
			runtime::Device::kMetal,
			makeMetalElemwiseResidentRunPolicy());
}

template <typename T>
runtime::Status matrixSubMetalResidentFinish(
		const T* a,
		const T* b,
		T* out,
		std::size_t rows,
		std::size_t cols) {
	return matrixSubWithPolicy(
			a,
			b,
			out,
			rows,
			cols,
			runtime::Device::kMetal,
			makeMetalElemwiseResidentFinishPolicy());
}

template <typename T>
runtime::Status matrixDivMetalResidentStart(
		const T* a,
		const T* b,
		T* out,
		std::size_t rows,
		std::size_t cols) {
	return matrixDivWithPolicy(
			a,
			b,
			out,
			rows,
			cols,
			runtime::Device::kMetal,
			makeMetalElemwiseResidentStartPolicy());
}

template <typename T>
runtime::Status matrixDivMetalResidentRun(
		const T* a,
		const T* b,
		T* out,
		std::size_t rows,
		std::size_t cols) {
	return matrixDivWithPolicy(
			a,
			b,
			out,
			rows,
			cols,
			runtime::Device::kMetal,
			makeMetalElemwiseResidentRunPolicy());
}

template <typename T>
runtime::Status matrixDivMetalResidentFinish(
		const T* a,
		const T* b,
		T* out,
		std::size_t rows,
		std::size_t cols) {
	return matrixDivWithPolicy(
			a,
			b,
			out,
			rows,
			cols,
			runtime::Device::kMetal,
			makeMetalElemwiseResidentFinishPolicy());
}

template <typename T>
class MatMulMetalResidentSession {
 public:
	explicit MatMulMetalResidentSession(std::size_t m, std::size_t k, std::size_t n)
			: m_(m), k_(k), n_(n) {
		static_assert(std::is_floating_point_v<T>, "MatMulMetalResidentSession<T> supports floating-point types only");
	}

	runtime::Status start(const T* a, const T* b, T* out) const {
		return matMulMetalResidentStart<T>(a, b, out, m_, k_, n_);
	}

	runtime::Status run(const T* a, const T* b, T* out) const {
		return matMulMetalResidentRun<T>(a, b, out, m_, k_, n_);
	}

	runtime::Status finish(const T* a, const T* b, T* out) const {
		return matMulMetalResidentFinish<T>(a, b, out, m_, k_, n_);
	}

	runtime::Status sync(const T* a, const T* b, T* out) const {
		return matMulMetalResidentSync<T>(a, b, out, m_, k_, n_);
	}

 private:
	std::size_t m_;
	std::size_t k_;
	std::size_t n_;
};

template <typename T>
class MatrixElemwiseMetalResidentSession {
 public:
	explicit MatrixElemwiseMetalResidentSession(std::size_t rows, std::size_t cols)
			: rows_(rows), cols_(cols) {
		static_assert(std::is_floating_point_v<T>, "MatrixElemwiseMetalResidentSession<T> supports floating-point types only");
	}

	runtime::Status subStart(const T* a, const T* b, T* out) const {
		return matrixSubMetalResidentStart<T>(a, b, out, rows_, cols_);
	}

	runtime::Status subRun(const T* a, const T* b, T* out) const {
		return matrixSubMetalResidentRun<T>(a, b, out, rows_, cols_);
	}

	runtime::Status subFinish(const T* a, const T* b, T* out) const {
		return matrixSubMetalResidentFinish<T>(a, b, out, rows_, cols_);
	}

	runtime::Status divStart(const T* a, const T* b, T* out) const {
		return matrixDivMetalResidentStart<T>(a, b, out, rows_, cols_);
	}

	runtime::Status divRun(const T* a, const T* b, T* out) const {
		return matrixDivMetalResidentRun<T>(a, b, out, rows_, cols_);
	}

	runtime::Status divFinish(const T* a, const T* b, T* out) const {
		return matrixDivMetalResidentFinish<T>(a, b, out, rows_, cols_);
	}

 private:
	std::size_t rows_;
	std::size_t cols_;
};

template <typename T>
class VectorAddMetalResidentSession {
 public:
	explicit VectorAddMetalResidentSession(std::size_t n) : n_(n) {
		static_assert(std::is_floating_point_v<T>, "VectorAddMetalResidentSession<T> supports floating-point types only");
	}

	runtime::Status start(const T* a, const T* b, T* out) const {
		return vectorAddMetalResidentStart<T>(a, b, out, n_);
	}

	runtime::Status run(const T* a, const T* b, T* out) const {
		return vectorAddMetalResidentRun<T>(a, b, out, n_);
	}

	runtime::Status finish(const T* a, const T* b, T* out) const {
		return vectorAddMetalResidentFinish<T>(a, b, out, n_);
	}

	runtime::Status runBatch(const T* const* a_list, const T* const* b_list, T* const* out_list, std::size_t count) const {
		if (a_list == nullptr || b_list == nullptr || out_list == nullptr || count == 0) {
			return runtime::Status::kInvalidValue;
		}
		for (std::size_t i = 0; i < count; ++i) {
			runtime::Status st = run(a_list[i], b_list[i], out_list[i]);
			if (st != runtime::Status::kSuccess) {
				return st;
			}
		}
		return runtime::Status::kSuccess;
	}

 private:
	std::size_t n_;
};

template <typename T>
runtime::Status matrixSubWithPolicy(
		const T* a,
		const T* b,
		T* out,
		std::size_t rows,
		std::size_t cols,
		runtime::Device device,
		const MatrixElementwiseIoPolicy& policy) {
	static_assert(std::is_floating_point_v<T>, "matrixSub<T> supports floating-point types only");
	if ((policy.upload_a && a == nullptr) || (policy.upload_b && b == nullptr) ||
			(policy.download_out && out == nullptr) || rows == 0 || cols == 0) {
		return runtime::Status::kInvalidValue;
	}

	if (device == runtime::Device::kCPU) {
		if (a == nullptr || b == nullptr || out == nullptr) {
			return runtime::Status::kInvalidValue;
		}
		return detail::matrixSubCpuWithPolicy(
				a,
				b,
				out,
				rows,
				cols,
				policy.upload_a,
				policy.upload_b,
				policy.download_out,
				policy.synchronize);
	}
	if (device == runtime::Device::kCUDA) {
		runtime::Status st = detail::matrixSubCudaWithPolicy(
				a,
				b,
				out,
				rows,
				cols,
				policy.upload_a,
				policy.upload_b,
				policy.download_out,
				policy.synchronize);
		if (st == runtime::Status::kNotSupported) {
			if (a == nullptr || b == nullptr || out == nullptr) {
				return runtime::Status::kNotSupported;
			}
			return detail::matrixSubCpu(a, b, out, rows, cols);
		}
		return st;
	}
	if (device == runtime::Device::kMetal) {
		runtime::Status st = detail::matrixSubMetalWithPolicy(
				a,
				b,
				out,
				rows,
				cols,
				policy.upload_a,
				policy.upload_b,
				policy.download_out,
				policy.synchronize);
		if (st == runtime::Status::kNotSupported) {
			if (a == nullptr || b == nullptr || out == nullptr) {
				return runtime::Status::kNotSupported;
			}
			return detail::matrixSubCpu(a, b, out, rows, cols);
		}
		return st;
	}
	return runtime::Status::kInvalidValue;
}

template <typename T>
runtime::Status matrixDivWithPolicy(
		const T* a,
		const T* b,
		T* out,
		std::size_t rows,
		std::size_t cols,
		runtime::Device device,
		const MatrixElementwiseIoPolicy& policy) {
	static_assert(std::is_floating_point_v<T>, "matrixDiv<T> supports floating-point types only");
	if ((policy.upload_a && a == nullptr) || (policy.upload_b && b == nullptr) ||
			(policy.download_out && out == nullptr) || rows == 0 || cols == 0) {
		return runtime::Status::kInvalidValue;
	}

	if (device == runtime::Device::kCPU) {
		if (a == nullptr || b == nullptr || out == nullptr) {
			return runtime::Status::kInvalidValue;
		}
		return detail::matrixDivCpuWithPolicy(
				a,
				b,
				out,
				rows,
				cols,
				policy.upload_a,
				policy.upload_b,
				policy.download_out,
				policy.synchronize);
	}
	if (device == runtime::Device::kCUDA) {
		runtime::Status st = detail::matrixDivCudaWithPolicy(
				a,
				b,
				out,
				rows,
				cols,
				policy.upload_a,
				policy.upload_b,
				policy.download_out,
				policy.synchronize);
		if (st == runtime::Status::kNotSupported) {
			if (a == nullptr || b == nullptr || out == nullptr) {
				return runtime::Status::kNotSupported;
			}
			return detail::matrixDivCpu(a, b, out, rows, cols);
		}
		return st;
	}
	if (device == runtime::Device::kMetal) {
		runtime::Status st = detail::matrixDivMetalWithPolicy(
				a,
				b,
				out,
				rows,
				cols,
				policy.upload_a,
				policy.upload_b,
				policy.download_out,
				policy.synchronize);
		if (st == runtime::Status::kNotSupported) {
			if (a == nullptr || b == nullptr || out == nullptr) {
				return runtime::Status::kNotSupported;
			}
			return detail::matrixDivCpu(a, b, out, rows, cols);
		}
		return st;
	}
	return runtime::Status::kInvalidValue;
}

}  // namespace lightning_core::ops
