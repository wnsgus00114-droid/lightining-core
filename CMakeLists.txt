cmake_minimum_required(VERSION 3.23)
project(LightningCore VERSION 0.1.12 LANGUAGES CXX)

if(NOT APPLE)
  message(FATAL_ERROR "Lightning Core currently supports macOS only.")
endif()

# CMake의 언어 감지 유틸을 끌어온다.
# 여기서 CUDA 감지할 때 쓰는데, 있으면 행복하고 없으면 CPU로 우회하면 그만임 ㅋ
include(CheckLanguage)

# 옵션들은 "오늘 어디까지 빌드할래?" 스위치들이다.
set(CJ_ENABLE_CUDA OFF CACHE BOOL "Enable native NVIDIA CUDA runtime path" FORCE)
set(CJ_ENABLE_METAL ON CACHE BOOL "Enable native Apple Metal runtime path" FORCE)
if(DEFINED SKBUILD)
  option(CJ_BUILD_TESTS "Build unit tests" OFF)
  option(CJ_BUILD_BENCHMARKS "Build benchmark binaries" OFF)
  option(CJ_BUILD_PYTHON "Build Python bindings (pybind11)" ON)
  option(CJ_BUILD_EXAMPLES "Build C/C++ examples" OFF)
else()
  option(CJ_BUILD_TESTS "Build unit tests" ON)
  option(CJ_BUILD_BENCHMARKS "Build benchmark binaries" ON)
  option(CJ_BUILD_PYTHON "Build Python bindings (pybind11)" ON)
  option(CJ_BUILD_EXAMPLES "Build C/C++ examples" ON)
endif()

# C++17로 통일해서 플랫폼별 컴파일러 성격 차이를 줄인다.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 기본은 보수적으로 OFF, 감지 성공하면 ON으로 전환한다.
set(CJ_HAS_CUDA OFF)
set(CJ_HAS_METAL OFF)
if(CJ_ENABLE_CUDA)
  # CUDA 컴파일러가 있는지 먼저 살핀다.
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    # 여기 들어오면 .cu도 진짜로 빌드 가능한 상태다.
    enable_language(CUDA)
    set(CJ_HAS_CUDA ON)
    message(STATUS "CUDA compiler detected: ${CMAKE_CUDA_COMPILER}")
  else()
    # macOS 같은 환경에서 자주 보게 되는 경고. 정상 동작이며 CPU 폴백으로 간다.
    message(WARNING "CUDA compiler not found. Building CPU-only path.")
  endif()
endif()

if(APPLE AND CJ_ENABLE_METAL)
  check_language(OBJCXX)
  if(CMAKE_OBJCXX_COMPILER)
    enable_language(OBJCXX)
    set(CJ_HAS_METAL ON)
    message(STATUS "Metal backend enabled with Objective-C++ compiler: ${CMAKE_OBJCXX_COMPILER}")
  else()
    message(WARNING "Objective-C++ compiler not found. Metal backend disabled.")
  endif()
endif()

# 공통 소스: 런타임, 텐서, 디스패치, CPU 구현.
set(CJ_SOURCES
  src/runtime.cpp
  src/lightning_core_c_api.cpp
  src/attention.cpp
  src/attention_cpu.cpp
  src/ops/matmul_cpu.cpp
  src/ops/matrix_elemwise.cpp
  src/ops/vector_add_cpu.cpp
)

if(APPLE)
  list(APPEND CJ_SOURCES src/apple_ml.mm)
else()
  list(APPEND CJ_SOURCES src/apple_ml_stub.cpp)
endif()

if(CJ_HAS_CUDA)
  # CUDA 가능 시 커널 파일 포함.
  list(APPEND CJ_SOURCES src/ops/vector_add_cuda.cu)
else()
  # CUDA 미지원 시 링커 에러 안 나게 스텁 구현 포함.
  list(APPEND CJ_SOURCES src/ops/vector_add_stub.cpp)
endif()

if(CJ_HAS_METAL)
  list(APPEND CJ_SOURCES src/attention_metal.mm)
  list(APPEND CJ_SOURCES src/ops/conv2d_metal.mm)
  list(APPEND CJ_SOURCES src/ops/matmul_metal.mm)
  list(APPEND CJ_SOURCES src/ops/matrix_elemwise_metal.mm)
  list(APPEND CJ_SOURCES src/ops/vector_add_metal.mm)
else()
  list(APPEND CJ_SOURCES src/attention_metal.mm)
  list(APPEND CJ_SOURCES src/ops/conv2d_metal.mm)
  list(APPEND CJ_SOURCES src/ops/matmul_metal.mm)
  list(APPEND CJ_SOURCES src/ops/matrix_elemwise_metal.mm)
  list(APPEND CJ_SOURCES src/ops/vector_add_metal_stub.cpp)
endif()

# 코어 라이브러리 타깃과 네임스페이스 별칭.
add_library(lightning_core_core STATIC ${CJ_SOURCES})
add_library(lightning_core::lightning_core ALIAS lightning_core_core)
set_target_properties(lightning_core_core PROPERTIES OUTPUT_NAME lightning_core)

target_compile_options(lightning_core_core PRIVATE
  $<$<CONFIG:Release>:-O3 -ffast-math -funroll-loops>
)

# 공개 헤더 경로를 내보내서 외부 타깃에서도 include 가능하게 한다.
target_include_directories(lightning_core_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(CJ_HAS_CUDA)
  # 코드에서 #if CJ_HAS_CUDA로 분기할 수 있도록 매크로 주입.
  target_compile_definitions(lightning_core_core PUBLIC CJ_HAS_CUDA=1)
  set_target_properties(lightning_core_core PROPERTIES
    # CUDA separable compilation: 커널/호스트 코드를 파일 단위로 나눠도 링크 가능.
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
  )
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # 아키텍처 미지정이면 로컬에 맞춰 자동 추정.
    set_target_properties(lightning_core_core PROPERTIES CUDA_ARCHITECTURES "native")
  endif()
  # CUDA 런타임(cudart) 링크.
  target_link_libraries(lightning_core_core PUBLIC CUDA::cudart)
else()
  # CPU 빌드에서도 코드 경로가 명확히 갈리도록 0을 박아둔다.
  target_compile_definitions(lightning_core_core PUBLIC CJ_HAS_CUDA=0)
endif()

if(CJ_HAS_METAL)
  target_compile_definitions(lightning_core_core PUBLIC CJ_HAS_METAL=1)
  target_link_libraries(lightning_core_core PUBLIC "-framework Metal" "-framework Foundation")
else()
  target_compile_definitions(lightning_core_core PUBLIC CJ_HAS_METAL=0)
endif()

if(APPLE)
  target_link_libraries(lightning_core_core PUBLIC
    "-framework Accelerate"
    "-framework CoreML"
    "-framework MetalPerformanceShaders"
    "-framework MetalPerformanceShadersGraph"
  )
endif()

# 플랫폼 식별 매크로: OS별 조건 컴파일용.
if(WIN32)
  target_compile_definitions(lightning_core_core PUBLIC CJ_PLATFORM_WINDOWS=1)
elseif(APPLE)
  target_compile_definitions(lightning_core_core PUBLIC CJ_PLATFORM_MACOS=1)
elseif(UNIX)
  target_compile_definitions(lightning_core_core PUBLIC CJ_PLATFORM_LINUX=1)
endif()

# 선택 옵션에 따라 하위 모듈들을 순차적으로 연결한다.
if(CJ_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(CJ_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(CJ_BUILD_PYTHON)
  add_subdirectory(python)
endif()

if(CJ_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
