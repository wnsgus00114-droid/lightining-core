# 런타임 API 스모크 테스트.
add_executable(test_runtime test_runtime.cpp)
target_link_libraries(test_runtime PRIVATE lightning_core::lightning_core)
add_test(NAME test_runtime COMMAND test_runtime)

# 텐서 연산 정확도 테스트.
add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE lightning_core::lightning_core)
add_test(NAME test_tensor COMMAND test_tensor)

# Shape/layout/lifetime/alias contract freeze regression test.
add_executable(test_tensor_contract_freeze test_tensor_contract_freeze.cpp)
target_link_libraries(test_tensor_contract_freeze PRIVATE lightning_core::lightning_core)
add_test(NAME test_tensor_contract_freeze COMMAND test_tensor_contract_freeze)

# attention forward/train 스모크 테스트.
add_executable(test_attention test_attention.cpp)
target_link_libraries(test_attention PRIVATE lightning_core::lightning_core)
add_test(NAME test_attention COMMAND test_attention)

# 기본 행렬곱 정확도 테스트.
add_executable(test_matmul test_matmul.cpp)
target_link_libraries(test_matmul PRIVATE lightning_core::lightning_core)
add_test(NAME test_matmul COMMAND test_matmul)

# 행렬 원소별 뺄셈/나눗셈 및 정책 경로 테스트.
add_executable(test_matrix_ops test_matrix_ops.cpp)
target_link_libraries(test_matrix_ops PRIVATE lightning_core::lightning_core)
add_test(NAME test_matrix_ops COMMAND test_matrix_ops)

# 모델 패밀리별 커스텀 프리셋/정책 매핑 테스트.
add_executable(test_model_customization test_model_customization.cpp)
target_link_libraries(test_model_customization PRIVATE lightning_core::lightning_core)
add_test(NAME test_model_customization COMMAND test_model_customization)

# VLM fast-path smoke test.
add_executable(test_vlm_fastpath test_vlm_fastpath.cpp)
target_link_libraries(test_vlm_fastpath PRIVATE lightning_core::lightning_core)
add_test(NAME test_vlm_fastpath COMMAND test_vlm_fastpath)

# Operator registry + minimal graph IR contract test.
add_executable(test_graph_ir test_graph_ir.cpp)
target_link_libraries(test_graph_ir PRIVATE lightning_core::lightning_core)
add_test(NAME test_graph_ir COMMAND test_graph_ir)

# Metal/CPU backend parity test for core ops.
add_executable(test_backend_parity test_backend_parity.cpp)
target_link_libraries(test_backend_parity PRIVATE lightning_core::lightning_core)
add_test(NAME test_backend_parity COMMAND test_backend_parity)

# C API/ABI version + struct-size freeze smoke test.
add_executable(test_c_api_abi_freeze test_c_api_abi_freeze.cpp)
target_link_libraries(test_c_api_abi_freeze PRIVATE lightning_core::lightning_core)
add_test(NAME test_c_api_abi_freeze COMMAND test_c_api_abi_freeze)
