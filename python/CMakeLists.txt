# pybind11이 있으면 파이썬 모듈까지 빌드하고,
# 없으면 경고만 남기고 전체 빌드는 계속 진행한다.
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG QUIET)

if(pybind11_FOUND)
  # 파이썬 import 가능한 확장 모듈 생성.
  pybind11_add_module(lightining_core MODULE bindings/pybind_module.cpp)
  target_link_libraries(lightining_core PRIVATE cudajun::cudajun)
  target_include_directories(lightining_core PRIVATE ${CMAKE_SOURCE_DIR}/include)
  install(TARGETS lightining_core LIBRARY DESTINATION .)
  message(STATUS "pybind11 found. Python module target enabled.")
else()
  if(DEFINED SKBUILD)
    message(FATAL_ERROR "pybind11 is required for pip/scikit-build package builds.")
  endif()
  message(WARNING "pybind11 not found. Skipping Python module build.")
endif()
