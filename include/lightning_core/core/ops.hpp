#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <vector>

#include "lightning_core/core/detail/ops_backend.hpp"
#include "lightning_core/core/ops/policy.hpp"
#include "lightning_core/core/runtime.hpp"

namespace lightning_core::ops {

inline runtime::Status matMulMetalResetTuning() {
	return detail::matMulMetalResetTuning();
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
runtime::Status matMulMetalResidentBatchSync(
		const T* a,
		const T* b,
		T* out,
		std::size_t m,
		std::size_t k,
		std::size_t n,
		std::size_t repeat_count) {
	if (repeat_count == 0) {
		return runtime::Status::kInvalidValue;
	}
	if (a == nullptr || b == nullptr || out == nullptr) {
		return runtime::Status::kInvalidValue;
	}

	if constexpr (std::is_same_v<T, float>) {
		runtime::Status st = detail::matMulMetalWithPolicyBatched(
				a,
				b,
				out,
				m,
				k,
				n,
				false,
				false,
				false,
				true,
				repeat_count);
		if (st == runtime::Status::kNotSupported) {
			for (std::size_t i = 0; i < repeat_count; ++i) {
				st = detail::matMulCpu(a, b, out, m, k, n);
				if (st != runtime::Status::kSuccess) {
					return st;
				}
			}
			return runtime::Status::kSuccess;
		}
		return st;
	}

	for (std::size_t i = 0; i < repeat_count; ++i) {
		runtime::Status st = detail::matMulCpu(a, b, out, m, k, n);
		if (st != runtime::Status::kSuccess) {
			return st;
		}
	}
	return runtime::Status::kSuccess;
}

template <typename T>
runtime::Status matMulMetalResidentBatchNoDownload(
		const T* a,
		const T* b,
		T* out,
		std::size_t m,
		std::size_t k,
		std::size_t n,
		std::size_t repeat_count) {
	if (repeat_count == 0) {
		return runtime::Status::kInvalidValue;
	}

	if constexpr (std::is_same_v<T, float>) {
		runtime::Status st = detail::matMulMetalWithPolicyBatched(
				a,
				b,
				out,
				m,
				k,
				n,
				false,
				false,
				false,
				true,
				repeat_count);
		if (st == runtime::Status::kNotSupported) {
			if (a == nullptr || b == nullptr || out == nullptr) {
				return runtime::Status::kNotSupported;
			}
			for (std::size_t i = 0; i < repeat_count; ++i) {
				st = detail::matMulCpu(a, b, out, m, k, n);
				if (st != runtime::Status::kSuccess) {
					return st;
				}
			}
			return runtime::Status::kSuccess;
		}
		return st;
	}

	if (a == nullptr || b == nullptr || out == nullptr) {
		return runtime::Status::kInvalidValue;
	}
	for (std::size_t i = 0; i < repeat_count; ++i) {
		runtime::Status st = detail::matMulCpu(a, b, out, m, k, n);
		if (st != runtime::Status::kSuccess) {
			return st;
		}
	}
	return runtime::Status::kSuccess;
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
runtime::Status conv2dNchw(
		const T* x,
		const T* w,
		const T* bias,
		T* out,
		std::size_t batch,
		std::size_t in_channels,
		std::size_t in_h,
		std::size_t in_w,
		std::size_t out_channels,
		std::size_t kernel_h,
		std::size_t kernel_w,
		std::size_t stride_h,
		std::size_t stride_w,
		std::size_t pad_h,
		std::size_t pad_w,
		runtime::Device device,
		bool apply_relu = false);

template <typename T>
runtime::Status conv2dNchwWithPolicy(
		const T* x,
		const T* w,
		const T* bias,
		T* out,
		std::size_t batch,
		std::size_t in_channels,
		std::size_t in_h,
		std::size_t in_w,
		std::size_t out_channels,
		std::size_t kernel_h,
		std::size_t kernel_w,
		std::size_t stride_h,
		std::size_t stride_w,
		std::size_t pad_h,
		std::size_t pad_w,
		runtime::Device device,
		const Conv2dIoPolicy& policy,
		bool apply_relu = false) {
	static_assert(std::is_floating_point_v<T>, "conv2dNchwWithPolicy<T> supports floating-point types only");
	if ((policy.upload_x && x == nullptr) || (policy.upload_w && w == nullptr) ||
			(policy.upload_bias && bias == nullptr) || (policy.download_out && out == nullptr) ||
			batch == 0 || in_channels == 0 || in_h == 0 || in_w == 0 || out_channels == 0 ||
			kernel_h == 0 || kernel_w == 0 || stride_h == 0 || stride_w == 0) {
		return runtime::Status::kInvalidValue;
	}
	if (in_h + (2 * pad_h) < kernel_h || in_w + (2 * pad_w) < kernel_w) {
		return runtime::Status::kInvalidValue;
	}

	if constexpr (std::is_same_v<T, float>) {
		const bool metal_conv3x3_shape =
				device == runtime::Device::kMetal &&
				kernel_h == 3 && kernel_w == 3 &&
				stride_h == 1 && stride_w == 1 &&
				pad_h == 1 && pad_w == 1 &&
				batch <= 64 && in_channels <= 64 && out_channels <= 128;
		if (metal_conv3x3_shape) {
			return detail::conv2dNchw3x3s1p1MetalWithPolicy(
					reinterpret_cast<const float*>(x),
					reinterpret_cast<const float*>(w),
					reinterpret_cast<const float*>(bias),
					reinterpret_cast<float*>(out),
					batch,
					in_channels,
					in_h,
					in_w,
					out_channels,
					apply_relu,
					policy.upload_x,
					policy.upload_w,
					policy.upload_bias,
					policy.download_out,
					policy.synchronize);
		}
	}

	if (!policy.upload_x || !policy.upload_w || !policy.upload_bias || !policy.download_out || !policy.synchronize) {
		return runtime::Status::kNotSupported;
	}

	return conv2dNchw<T>(
			x,
			w,
			bias,
			out,
			batch,
			in_channels,
			in_h,
			in_w,
			out_channels,
			kernel_h,
			kernel_w,
			stride_h,
			stride_w,
			pad_h,
			pad_w,
			device,
			apply_relu);
}

template <typename T>
runtime::Status conv2dNchwMetalResidentStart(
		const T* x,
		const T* w,
		const T* bias,
		T* out,
		std::size_t batch,
		std::size_t in_channels,
		std::size_t in_h,
		std::size_t in_w,
		std::size_t out_channels,
		std::size_t kernel_h,
		std::size_t kernel_w,
		std::size_t stride_h,
		std::size_t stride_w,
		std::size_t pad_h,
		std::size_t pad_w,
		bool apply_relu = false) {
	runtime::Status st = conv2dNchwWithPolicy(
			x,
			w,
			bias,
			out,
			batch,
			in_channels,
			in_h,
			in_w,
			out_channels,
			kernel_h,
			kernel_w,
			stride_h,
			stride_w,
			pad_h,
			pad_w,
			runtime::Device::kMetal,
			makeMetalConvResidentStartPolicy(),
			apply_relu);
	if (st == runtime::Status::kNotSupported) {
		if (x == nullptr || w == nullptr || bias == nullptr || out == nullptr) {
			return st;
		}
		return conv2dNchw<T>(
				x,
				w,
				bias,
				out,
				batch,
				in_channels,
				in_h,
				in_w,
				out_channels,
				kernel_h,
				kernel_w,
				stride_h,
				stride_w,
				pad_h,
				pad_w,
				runtime::Device::kCPU,
				apply_relu);
	}
	return st;
}

template <typename T>
runtime::Status conv2dNchwMetalResidentRun(
		const T* x,
		const T* w,
		const T* bias,
		T* out,
		std::size_t batch,
		std::size_t in_channels,
		std::size_t in_h,
		std::size_t in_w,
		std::size_t out_channels,
		std::size_t kernel_h,
		std::size_t kernel_w,
		std::size_t stride_h,
		std::size_t stride_w,
		std::size_t pad_h,
		std::size_t pad_w,
		bool apply_relu = false) {
	runtime::Status st = conv2dNchwWithPolicy(
			x,
			w,
			bias,
			out,
			batch,
			in_channels,
			in_h,
			in_w,
			out_channels,
			kernel_h,
			kernel_w,
			stride_h,
			stride_w,
			pad_h,
			pad_w,
			runtime::Device::kMetal,
			makeMetalConvResidentRunPolicy(),
			apply_relu);
	if (st == runtime::Status::kNotSupported) {
		if (x == nullptr || w == nullptr || bias == nullptr || out == nullptr) {
			return st;
		}
		return conv2dNchw<T>(
				x,
				w,
				bias,
				out,
				batch,
				in_channels,
				in_h,
				in_w,
				out_channels,
				kernel_h,
				kernel_w,
				stride_h,
				stride_w,
				pad_h,
				pad_w,
				runtime::Device::kCPU,
				apply_relu);
	}
	return st;
}

template <typename T>
runtime::Status conv2dNchwMetalResidentSync(
		const T* x,
		const T* w,
		const T* bias,
		T* out,
		std::size_t batch,
		std::size_t in_channels,
		std::size_t in_h,
		std::size_t in_w,
		std::size_t out_channels,
		std::size_t kernel_h,
		std::size_t kernel_w,
		std::size_t stride_h,
		std::size_t stride_w,
		std::size_t pad_h,
		std::size_t pad_w,
		bool apply_relu = false) {
	runtime::Status st = conv2dNchwWithPolicy(
			x,
			w,
			bias,
			out,
			batch,
			in_channels,
			in_h,
			in_w,
			out_channels,
			kernel_h,
			kernel_w,
			stride_h,
			stride_w,
			pad_h,
			pad_w,
			runtime::Device::kMetal,
			makeMetalConvResidentSyncPolicy(),
			apply_relu);
	if (st == runtime::Status::kNotSupported) {
		// CPU fallback path is synchronous by nature.
		return runtime::Status::kSuccess;
	}
	return st;
}

template <typename T>
runtime::Status conv2dNchwMetalResidentFinish(
		const T* x,
		const T* w,
		const T* bias,
		T* out,
		std::size_t batch,
		std::size_t in_channels,
		std::size_t in_h,
		std::size_t in_w,
		std::size_t out_channels,
		std::size_t kernel_h,
		std::size_t kernel_w,
		std::size_t stride_h,
		std::size_t stride_w,
		std::size_t pad_h,
		std::size_t pad_w,
		bool apply_relu = false) {
	runtime::Status st = conv2dNchwWithPolicy(
			x,
			w,
			bias,
			out,
			batch,
			in_channels,
			in_h,
			in_w,
			out_channels,
			kernel_h,
			kernel_w,
			stride_h,
			stride_w,
			pad_h,
			pad_w,
			runtime::Device::kMetal,
			makeMetalConvResidentFinishPolicy(),
			apply_relu);
	if (st == runtime::Status::kNotSupported) {
		if (x == nullptr || w == nullptr || bias == nullptr || out == nullptr) {
			return st;
		}
		return conv2dNchw<T>(
				x,
				w,
				bias,
				out,
				batch,
				in_channels,
				in_h,
				in_w,
				out_channels,
				kernel_h,
				kernel_w,
				stride_h,
				stride_w,
				pad_h,
				pad_w,
				runtime::Device::kCPU,
				apply_relu);
	}
	return st;
}

template <typename T>
runtime::Status conv2dNchw(
		const T* x,
		const T* w,
		const T* bias,
		T* out,
		std::size_t batch,
		std::size_t in_channels,
		std::size_t in_h,
		std::size_t in_w,
		std::size_t out_channels,
		std::size_t kernel_h,
		std::size_t kernel_w,
		std::size_t stride_h,
		std::size_t stride_w,
		std::size_t pad_h,
		std::size_t pad_w,
		runtime::Device device,
		bool apply_relu) {
	static_assert(std::is_floating_point_v<T>, "conv2dNchw<T> supports floating-point types only");
	if (x == nullptr || w == nullptr || out == nullptr ||
				batch == 0 || in_channels == 0 || in_h == 0 || in_w == 0 ||
				out_channels == 0 || kernel_h == 0 || kernel_w == 0 ||
				stride_h == 0 || stride_w == 0) {
		return runtime::Status::kInvalidValue;
	}
	if (in_h + (2 * pad_h) < kernel_h || in_w + (2 * pad_w) < kernel_w) {
		return runtime::Status::kInvalidValue;
	}

	const std::size_t out_h = (in_h + (2 * pad_h) - kernel_h) / stride_h + 1;
	const std::size_t out_w = (in_w + (2 * pad_w) - kernel_w) / stride_w + 1;
	const std::size_t k_inner = in_channels * kernel_h * kernel_w;
	const std::size_t m_rows = batch * out_h * out_w;
	const std::size_t cols_size = m_rows * k_inner;
	const std::size_t out2d_size = m_rows * out_channels;
	if constexpr (std::is_same_v<T, float>) {
		const bool metal_conv3x3_shape =
				device == runtime::Device::kMetal &&
				kernel_h == 3 && kernel_w == 3 &&
				stride_h == 1 && stride_w == 1 &&
				pad_h == 1 && pad_w == 1 &&
				batch <= 64 && in_channels <= 64 && out_channels <= 128;
		if (metal_conv3x3_shape) {
			runtime::Status metal_st = detail::conv2dNchw3x3s1p1Metal(
					reinterpret_cast<const float*>(x),
					reinterpret_cast<const float*>(w),
					reinterpret_cast<const float*>(bias),
					reinterpret_cast<float*>(out),
					batch,
					in_channels,
					in_h,
					in_w,
					out_channels,
					apply_relu);
			if (metal_st == runtime::Status::kSuccess) {
				return runtime::Status::kSuccess;
			}
		}
	}
	const bool torchstrong_direct_shape =
			device == runtime::Device::kMetal &&
			kernel_h == 3 && kernel_w == 3 &&
			stride_h == 1 && stride_w == 1 &&
			pad_h == 1 && pad_w == 1 &&
			in_channels <= 4 && out_channels <= 32 &&
			batch <= 64;
 	const bool torchstrong_im2col_shape =
			device == runtime::Device::kMetal &&
			kernel_h == 3 && kernel_w == 3 &&
			stride_h == 1 && stride_w == 1 &&
			pad_h == 1 && pad_w == 1 &&
			in_channels <= 16 && out_channels <= 32 &&
			batch <= 64;

	if (torchstrong_direct_shape) {
		const std::size_t pad_h_i = in_h + 2;
		const std::size_t pad_w_i = in_w + 2;
		const std::size_t per_ch_pad = pad_h_i * pad_w_i;
		const std::size_t per_n_pad = in_channels * per_ch_pad;
		const std::size_t x_pad_size = batch * per_n_pad;

		static thread_local std::vector<T> x_pad_buf;
		if (x_pad_buf.size() < x_pad_size) {
			x_pad_buf.resize(x_pad_size);
		}
		T* x_pad = x_pad_buf.data();

		for (std::size_t n_idx = 0; n_idx < batch; ++n_idx) {
			for (std::size_t c = 0; c < in_channels; ++c) {
				const T* src = x + ((n_idx * in_channels + c) * in_h * in_w);
				T* dst = x_pad + (n_idx * per_n_pad + c * per_ch_pad);
				// Torch-strong paths are fixed to pad=1; zero only border rows/cols instead of clearing full buffer.
				dst[0] = static_cast<T>(0);
				std::fill(dst + 1, dst + pad_w_i, static_cast<T>(0));
				for (std::size_t y = 0; y < in_h; ++y) {
					T* row = dst + ((y + 1) * pad_w_i);
					row[0] = static_cast<T>(0);
					std::memcpy(row + 1, src + (y * in_w), sizeof(T) * in_w);
					row[pad_w_i - 1] = static_cast<T>(0);
				}
				T* bottom = dst + ((pad_h_i - 1) * pad_w_i);
				std::fill(bottom, bottom + pad_w_i, static_cast<T>(0));
			}
		}

		for (std::size_t n_idx = 0; n_idx < batch; ++n_idx) {
			for (std::size_t oc = 0; oc < out_channels; ++oc) {
				const T b0 = (bias != nullptr ? bias[oc] : static_cast<T>(0));
				for (std::size_t oy = 0; oy < out_h; ++oy) {
					for (std::size_t ox = 0; ox < out_w; ++ox) {
						T acc = b0;
						for (std::size_t c = 0; c < in_channels; ++c) {
							const T* p = x_pad + (n_idx * per_n_pad + c * per_ch_pad + oy * pad_w_i + ox);
							const T* wf = w + ((oc * in_channels + c) * 9);
							acc += p[0] * wf[0] + p[1] * wf[1] + p[2] * wf[2]
									 + p[pad_w_i + 0] * wf[3] + p[pad_w_i + 1] * wf[4] + p[pad_w_i + 2] * wf[5]
									 + p[(2 * pad_w_i) + 0] * wf[6] + p[(2 * pad_w_i) + 1] * wf[7] + p[(2 * pad_w_i) + 2] * wf[8];
						}
						if (apply_relu && acc < static_cast<T>(0)) {
							acc = static_cast<T>(0);
						}
						out[((n_idx * out_channels + oc) * out_h + oy) * out_w + ox] = acc;
					}
				}
			}
		}
		return runtime::Status::kSuccess;
	}

	static thread_local std::vector<T> cols_buf;
	static thread_local std::vector<T> out2d_buf;
	if (cols_buf.size() < cols_size) {
		cols_buf.resize(cols_size);
	}
	if (out2d_buf.size() < out2d_size) {
		out2d_buf.resize(out2d_size);
	}
	T* cols = cols_buf.data();
	T* out2d = out2d_buf.data();

	struct PackedWeightCacheEntry {
		const T* ptr = nullptr;
		std::array<std::size_t, 4> shape{};
		std::vector<T> packed;
		std::size_t age = 0;
	};
	static thread_local std::vector<PackedWeightCacheEntry> w_cache;
	static thread_local std::size_t w_cache_clock = 0;
	constexpr std::size_t kPackedWeightCacheCap = 8;

	auto get_packed_weight = [&]() -> const T* {
		const std::array<std::size_t, 4> shape = {in_channels, out_channels, kernel_h, kernel_w};
		++w_cache_clock;
		PackedWeightCacheEntry* hit = nullptr;
		for (auto& e : w_cache) {
			if (e.ptr == w && e.shape == shape) {
				hit = &e;
				break;
			}
		}
		if (hit == nullptr) {
			if (w_cache.size() < kPackedWeightCacheCap) {
				w_cache.push_back(PackedWeightCacheEntry{});
				hit = &w_cache.back();
			} else {
				hit = &w_cache.front();
				for (auto& e : w_cache) {
					if (e.age < hit->age) {
						hit = &e;
					}
				}
			}
			hit->ptr = w;
			hit->shape = shape;
			hit->packed.assign(k_inner * out_channels, static_cast<T>(0));
			for (std::size_t oc = 0; oc < out_channels; ++oc) {
				for (std::size_t c = 0; c < in_channels; ++c) {
					for (std::size_t ky = 0; ky < kernel_h; ++ky) {
						for (std::size_t kx = 0; kx < kernel_w; ++kx) {
							const std::size_t col = ((c * kernel_h) + ky) * kernel_w + kx;
							const std::size_t w_idx = (((oc * in_channels + c) * kernel_h + ky) * kernel_w + kx);
							hit->packed[col * out_channels + oc] = w[w_idx];
						}
					}
				}
			}
		}
		hit->age = w_cache_clock;
		return hit->packed.data();
	};

	const T* w_col = get_packed_weight();
	if (torchstrong_im2col_shape) {
		const std::size_t pad_h_i = in_h + 2;
		const std::size_t pad_w_i = in_w + 2;
		const std::size_t per_ch_pad = pad_h_i * pad_w_i;
		const std::size_t per_n_pad = in_channels * per_ch_pad;
		const std::size_t x_pad_size = batch * per_n_pad;
		static thread_local std::vector<T> x_pad_buf;
		if (x_pad_buf.size() < x_pad_size) {
			x_pad_buf.resize(x_pad_size);
		}
		T* x_pad = x_pad_buf.data();

		for (std::size_t n_idx = 0; n_idx < batch; ++n_idx) {
			for (std::size_t c = 0; c < in_channels; ++c) {
				const T* src = x + ((n_idx * in_channels + c) * in_h * in_w);
				T* dst = x_pad + (n_idx * per_n_pad + c * per_ch_pad);
				dst[0] = static_cast<T>(0);
				std::fill(dst + 1, dst + pad_w_i, static_cast<T>(0));
				for (std::size_t y = 0; y < in_h; ++y) {
					T* row = dst + ((y + 1) * pad_w_i);
					row[0] = static_cast<T>(0);
					std::memcpy(row + 1, src + (y * in_w), sizeof(T) * in_w);
					row[pad_w_i - 1] = static_cast<T>(0);
				}
				T* bottom = dst + ((pad_h_i - 1) * pad_w_i);
				std::fill(bottom, bottom + pad_w_i, static_cast<T>(0));
			}
		}

		for (std::size_t n_idx = 0; n_idx < batch; ++n_idx) {
			for (std::size_t oy = 0; oy < out_h; ++oy) {
				for (std::size_t ox = 0; ox < out_w; ++ox) {
					const std::size_t row = (n_idx * out_h + oy) * out_w + ox;
					T* row_dst = cols + (row * k_inner);
					for (std::size_t c = 0; c < in_channels; ++c) {
						const T* p = x_pad + (n_idx * per_n_pad + c * per_ch_pad + oy * pad_w_i + ox);
						T* dst = row_dst + (c * 9);
						const T* p1 = p + pad_w_i;
						const T* p2 = p1 + pad_w_i;
						// Keep stores contiguous so the compiler can schedule two 4-value groups efficiently.
						dst[0] = p[0];
						dst[1] = p[1];
						dst[2] = p[2];
						dst[3] = p1[0];
						dst[4] = p1[1];
						dst[5] = p1[2];
						dst[6] = p2[0];
						dst[7] = p2[1];
						dst[8] = p2[2];
					}
				}
			}
		}
	} else {
		std::fill(cols_buf.begin(), cols_buf.begin() + static_cast<std::ptrdiff_t>(cols_size), static_cast<T>(0));
		for (std::size_t n_idx = 0; n_idx < batch; ++n_idx) {
			for (std::size_t oy = 0; oy < out_h; ++oy) {
				const std::size_t in_y_base = oy * stride_h;
				for (std::size_t ox = 0; ox < out_w; ++ox) {
					const std::size_t in_x_base = ox * stride_w;
					const std::size_t row = (n_idx * out_h + oy) * out_w + ox;
					for (std::size_t c = 0; c < in_channels; ++c) {
						for (std::size_t ky = 0; ky < kernel_h; ++ky) {
							const std::size_t y_nom = in_y_base + ky;
							const bool y_valid = y_nom >= pad_h && (y_nom - pad_h) < in_h;
							const std::size_t y_in = y_nom - pad_h;
							for (std::size_t kx = 0; kx < kernel_w; ++kx) {
								const std::size_t x_nom = in_x_base + kx;
								const bool x_valid = x_nom >= pad_w && (x_nom - pad_w) < in_w;
								const std::size_t x_in = x_nom - pad_w;
								const std::size_t col = ((c * kernel_h) + ky) * kernel_w + kx;
								if (y_valid && x_valid) {
									const std::size_t x_idx = ((n_idx * in_channels + c) * in_h + y_in) * in_w + x_in;
									cols[row * k_inner + col] = x[x_idx];
								}
							}
						}
					}
				}
			}
		}
	}
	runtime::Status mm_st = matMulWithPolicy<T>(
				cols,
				w_col,
				out2d,
				m_rows,
				k_inner,
				out_channels,
				device,
				MatMulIoPolicy{});
	if (mm_st != runtime::Status::kSuccess) {
		return mm_st;
	}

	for (std::size_t n_idx = 0; n_idx < batch; ++n_idx) {
		for (std::size_t oy = 0; oy < out_h; ++oy) {
			for (std::size_t ox = 0; ox < out_w; ++ox) {
				const std::size_t row = (n_idx * out_h + oy) * out_w + ox;
				for (std::size_t oc = 0; oc < out_channels; ++oc) {
					T v = out2d[row * out_channels + oc] + (bias != nullptr ? bias[oc] : static_cast<T>(0));
					if (apply_relu && v < static_cast<T>(0)) {
						v = static_cast<T>(0);
					}
					out[((n_idx * out_channels + oc) * out_h + oy) * out_w + ox] = v;
				}
			}
		}
	}

	return runtime::Status::kSuccess;
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

	runtime::Status runBatchSync(const T* a, const T* b, T* out, std::size_t repeat_count) const {
		return matMulMetalResidentBatchSync<T>(a, b, out, m_, k_, n_, repeat_count);
	}

	runtime::Status runBatchSyncNoDownload(const T* a, const T* b, T* out, std::size_t repeat_count) const {
		return matMulMetalResidentBatchNoDownload<T>(a, b, out, m_, k_, n_, repeat_count);
	}

	runtime::Status runBatchSyncCachedNoDownload(std::size_t repeat_count) const {
		return matMulMetalResidentBatchNoDownload<T>(nullptr, nullptr, nullptr, m_, k_, n_, repeat_count);
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
