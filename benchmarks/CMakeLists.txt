# vector add 마이크로 벤치 바이너리.
add_executable(bench_vector_add bench_vector_add.cpp)
target_link_libraries(bench_vector_add PRIVATE lightning_core::lightning_core)

# attention 전용 벤치 바이너리.
add_executable(bench_attention bench_attention.cpp)
target_link_libraries(bench_attention PRIVATE lightning_core::lightning_core)

# 기본 행렬곱 벤치 바이너리.
add_executable(bench_matmul bench_matmul.cpp)
target_link_libraries(bench_matmul PRIVATE lightning_core::lightning_core)

# 행렬 원소별 뺄셈/나눗셈 벤치 바이너리.
add_executable(bench_matrix_ops bench_matrix_ops.cpp)
target_link_libraries(bench_matrix_ops PRIVATE lightning_core::lightning_core)

# 모델 단위 벤치 바이너리.
add_executable(bench_transformer bench_transformer.cpp)
target_link_libraries(bench_transformer PRIVATE lightning_core::lightning_core)

add_executable(bench_lstm_rnn bench_lstm_rnn.cpp)
target_link_libraries(bench_lstm_rnn PRIVATE lightning_core::lightning_core)

add_executable(bench_cnn_dnn bench_cnn_dnn.cpp)
target_link_libraries(bench_cnn_dnn PRIVATE lightning_core::lightning_core)

add_executable(bench_vlm bench_vlm.cpp)
target_link_libraries(bench_vlm PRIVATE lightning_core::lightning_core)
