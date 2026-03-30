# pybind11이 있으면 파이썬 모듈까지 빌드하고,
# 없으면 경고만 남기고 전체 빌드는 계속 진행한다.
set(PYBIND11_FINDPYTHON ON)
find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
if(NOT Python3_Development.Module_FOUND)
  find_package(Python3 COMPONENTS Interpreter Development QUIET)
endif()

# pybind11_DIR가 비어 있으면 현재 Python 환경(venv 우선)에서 자동 탐색한다.
if(NOT DEFINED pybind11_DIR OR "${pybind11_DIR}" STREQUAL "" OR pybind11_DIR MATCHES "NOTFOUND")
  if(DEFINED pybind11_DIR AND pybind11_DIR MATCHES "NOTFOUND")
    unset(pybind11_DIR CACHE)
  endif()

  set(_lc_python_hints)

  # 흔한 로컬 구조: repo/.venv 또는 repo 상위/.venv
  set(_lc_repo_venv_python "${CMAKE_SOURCE_DIR}/.venv/bin/python")
  if(EXISTS "${_lc_repo_venv_python}")
    list(APPEND _lc_python_hints "${_lc_repo_venv_python}")
  endif()

  set(_lc_parent_venv_python "${CMAKE_SOURCE_DIR}/../.venv/bin/python")
  if(EXISTS "${_lc_parent_venv_python}")
    list(APPEND _lc_python_hints "${_lc_parent_venv_python}")
  endif()

  if(DEFINED ENV{VIRTUAL_ENV})
    set(_lc_venv_python "$ENV{VIRTUAL_ENV}/bin/python")
    if(EXISTS "${_lc_venv_python}")
      list(APPEND _lc_python_hints "${_lc_venv_python}")
    endif()
  endif()

  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    list(APPEND _lc_python_hints "${Python3_EXECUTABLE}")
  endif()

  list(REMOVE_DUPLICATES _lc_python_hints)

  foreach(_lc_python IN LISTS _lc_python_hints)
    execute_process(
      COMMAND "${_lc_python}" -c "import pybind11; print(pybind11.get_cmake_dir())"
      RESULT_VARIABLE _lc_pybind11_result
      OUTPUT_VARIABLE _lc_pybind11_dir
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
     if(_lc_pybind11_result EQUAL 0
       AND NOT _lc_pybind11_dir MATCHES "NOTFOUND"
       AND IS_DIRECTORY "${_lc_pybind11_dir}")
      set(pybind11_DIR "${_lc_pybind11_dir}" CACHE PATH "Path to pybind11 CMake config")
      message(STATUS "Auto-detected pybind11_DIR: ${pybind11_DIR}")
      break()
    endif()
  endforeach()
endif()

find_package(pybind11 CONFIG QUIET)

if(pybind11_FOUND)
  # 파이썬 import 가능한 확장 모듈 생성.
  pybind11_add_module(lightning_core MODULE
    bindings/pybind_module.cpp
    bindings/bind_runtime.cpp
    bindings/bind_tensor.cpp
    bindings/bind_ops.cpp
    bindings/bind_graph.cpp
    bindings/bind_attention.cpp
    bindings/bind_integrated.cpp
  )
  target_link_libraries(lightning_core PRIVATE lightning_core::lightning_core)
  target_include_directories(lightning_core PRIVATE ${CMAKE_SOURCE_DIR}/include)
  install(TARGETS lightning_core LIBRARY DESTINATION .)
  message(STATUS "pybind11 found. Python module target enabled.")
else()
  if(DEFINED SKBUILD)
    message(FATAL_ERROR "pybind11 is required for pip/scikit-build package builds.")
  endif()
  message(WARNING "pybind11 not found. Skipping Python module build.")
endif()
