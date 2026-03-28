#pragma once

#include <cstddef>
#include <string>

#include "cudajun/runtime.hpp"

namespace cudajun::apple {

// CoreML 모델을 사용한 추론 벤치 (ANE 선호).
runtime::Status benchmarkCoreMLInference(const std::string& modelPath, std::size_t n, int iters, double* avgMs);

// 기존 MPSGraph 기반 vector add 벤치.
runtime::Status benchmarkMpsGraphVectorAdd(std::size_t n, int iters, double* avgMs);

// MPSGraph 기반 학습 스텝(가중치 업데이트) 벤치.
runtime::Status benchmarkMpsGraphTrainStep(std::size_t n, int iters, double* avgMs);

}  // namespace cudajun::apple
