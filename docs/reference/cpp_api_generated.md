# C/C++ API Reference (Generated)

Generated from public headers under `include/lightning_core/`.

Regenerate:
`python scripts/generate_api_reference_docs.py`

| Metric | Count |
| --- | --- |
| Headers scanned | 7 |
| Types (enum/struct/class) | 48 |
| C++ functions | 38 |
| C API functions | 21 |

## Header Set

- [include/lightning_core/core/runtime.hpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp)
- [include/lightning_core/core/attention.hpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/attention.hpp)
- [include/lightning_core/core/graph.hpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp)
- [include/lightning_core/core/tensor.hpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/tensor.hpp)
- [include/lightning_core/core/ops/policy.hpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/ops/policy.hpp)
- [include/lightning_core/core/ops.hpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/ops.hpp)
- [include/lightning_core/lightning_core.h](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h)

## C API Functions

| Function | Source |
| --- | --- |
| `lcApplyDefaultSyncPolicy` | [include/lightning_core/lightning_core.h:113](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L113) |
| `lcApplySyncPolicy` | [include/lightning_core/lightning_core.h:112](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L112) |
| `lcBackendName` | [include/lightning_core/lightning_core.h:118](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L118) |
| `lcDeviceSynchronize` | [include/lightning_core/lightning_core.h:102](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L102) |
| `lcFree` | [include/lightning_core/lightning_core.h:100](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L100) |
| `lcGetActiveBackendCapabilities` | [include/lightning_core/lightning_core.h:115](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L115) |
| `lcGetActiveBackendInterfaceContract` | [include/lightning_core/lightning_core.h:117](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L117) |
| `lcGetBackendCapabilities` | [include/lightning_core/lightning_core.h:114](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L114) |
| `lcGetBackendInterfaceContract` | [include/lightning_core/lightning_core.h:116](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L116) |
| `lcGetDefaultSyncPolicy` | [include/lightning_core/lightning_core.h:111](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L111) |
| `lcGetDeviceCount` | [include/lightning_core/lightning_core.h:103](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L103) |
| `lcGetErrorString` | [include/lightning_core/lightning_core.h:119](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L119) |
| `lcGetMemoryModel` | [include/lightning_core/lightning_core.h:108](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L108) |
| `lcGetMemoryModelName` | [include/lightning_core/lightning_core.h:109](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L109) |
| `lcGetPreferredDeviceForInference` | [include/lightning_core/lightning_core.h:104](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L104) |
| `lcGetPreferredDeviceForTraining` | [include/lightning_core/lightning_core.h:105](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L105) |
| `lcIsCudaAvailable` | [include/lightning_core/lightning_core.h:106](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L106) |
| `lcIsMetalAvailable` | [include/lightning_core/lightning_core.h:107](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L107) |
| `lcMalloc` | [include/lightning_core/lightning_core.h:99](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L99) |
| `lcMemcpy` | [include/lightning_core/lightning_core.h:101](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L101) |
| `lcSetDefaultSyncPolicy` | [include/lightning_core/lightning_core.h:110](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L110) |

## Types

| Name | Kind | Source |
| --- | --- | --- |
| `AttentionConfig` | struct | [include/lightning_core/core/attention.hpp:9](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/attention.hpp#L9) |
| `AttentionImplementation` | enum class | [include/lightning_core/core/attention.hpp:34](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/attention.hpp#L34) |
| `AttentionIoPolicy` | struct | [include/lightning_core/core/attention.hpp:16](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/attention.hpp#L16) |
| `AttentionSession` | class | [include/lightning_core/core/attention.hpp:86](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/attention.hpp#L86) |
| `DType` | enum class | [include/lightning_core/core/graph.hpp:19](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L19) |
| `FusionPattern` | enum class | [include/lightning_core/core/graph.hpp:250](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L250) |
| `GraphExecutionGroup` | struct | [include/lightning_core/core/graph.hpp:241](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L241) |
| `GraphFusionDecision` | struct | [include/lightning_core/core/graph.hpp:263](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L263) |
| `GraphIR` | class | [include/lightning_core/core/graph.hpp:282](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L282) |
| `GraphNode` | struct | [include/lightning_core/core/graph.hpp:169](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L169) |
| `GraphPlanStep` | struct | [include/lightning_core/core/graph.hpp:178](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L178) |
| `GraphPlannerOptions` | struct | [include/lightning_core/core/graph.hpp:230](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L230) |
| `GraphTensorValue` | struct | [include/lightning_core/core/graph.hpp:163](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L163) |
| `OpKind` | enum class | [include/lightning_core/core/graph.hpp:34](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L34) |
| `OperatorRegistry` | class | [include/lightning_core/core/graph.hpp:95](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L95) |
| `OperatorSchema` | struct | [include/lightning_core/core/graph.hpp:71](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L71) |
| `TensorSpec` | struct | [include/lightning_core/core/graph.hpp:65](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L65) |
| `ValidationIssue` | struct | [include/lightning_core/core/graph.hpp:210](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L210) |
| `ValidationPass` | enum class | [include/lightning_core/core/graph.hpp:185](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L185) |
| `ValidationReport` | struct | [include/lightning_core/core/graph.hpp:218](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L218) |
| `MatMulMetalResidentSession` | class | [include/lightning_core/core/ops.hpp:1340](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/ops.hpp#L1340) |
| `MatrixElemwiseMetalResidentSession` | class | [include/lightning_core/core/ops.hpp:1382](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/ops.hpp#L1382) |
| `PackedWeightCacheEntry` | struct | [include/lightning_core/core/ops.hpp:1044](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/ops.hpp#L1044) |
| `VectorAddMetalResidentSession` | class | [include/lightning_core/core/ops.hpp:1419](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/ops.hpp#L1419) |
| `Conv2dIoPolicy` | struct | [include/lightning_core/core/ops/policy.hpp:29](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/ops/policy.hpp#L29) |
| `MatMulIoPolicy` | struct | [include/lightning_core/core/ops/policy.hpp:8](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/ops/policy.hpp#L8) |
| `MatrixElementwiseIoPolicy` | struct | [include/lightning_core/core/ops/policy.hpp:15](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/ops/policy.hpp#L15) |
| `VectorAddIoPolicy` | struct | [include/lightning_core/core/ops/policy.hpp:22](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/ops/policy.hpp#L22) |
| `BackendCapabilities` | struct | [include/lightning_core/core/runtime.hpp:99](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L99) |
| `BackendInterfaceContract` | struct | [include/lightning_core/core/runtime.hpp:146](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L146) |
| `ComputeInterfaceContract` | struct | [include/lightning_core/core/runtime.hpp:113](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L113) |
| `Device` | enum class | [include/lightning_core/core/runtime.hpp:11](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L11) |
| `MemcpyKind` | enum class | [include/lightning_core/core/runtime.hpp:36](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L36) |
| `MemoryInterfaceContract` | struct | [include/lightning_core/core/runtime.hpp:120](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L120) |
| `MemoryModel` | enum class | [include/lightning_core/core/runtime.hpp:44](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L44) |
| `ProfilerInterfaceContract` | struct | [include/lightning_core/core/runtime.hpp:137](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L137) |
| `RuntimeTraceEvent` | struct | [include/lightning_core/core/runtime.hpp:76](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L76) |
| `RuntimeTraceEventType` | enum class | [include/lightning_core/core/runtime.hpp:50](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L50) |
| `RuntimeTraceOpKind` | enum class | [include/lightning_core/core/runtime.hpp:65](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L65) |
| `Status` | enum class | [include/lightning_core/core/runtime.hpp:24](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L24) |
| `SyncInterfaceContract` | struct | [include/lightning_core/core/runtime.hpp:129](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L129) |
| `SyncMode` | enum class | [include/lightning_core/core/runtime.hpp:86](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L86) |
| `SyncPolicy` | struct | [include/lightning_core/core/runtime.hpp:93](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L93) |
| `WorkloadKind` | enum class | [include/lightning_core/core/runtime.hpp:17](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L17) |
| `Layout` | enum class | [include/lightning_core/core/tensor.hpp:17](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/tensor.hpp#L17) |
| `TensorT` | class | [include/lightning_core/core/tensor.hpp:97](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/tensor.hpp#L97) |
| `TensorT` | class | [include/lightning_core/core/tensor.hpp:190](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/tensor.hpp#L190) |
| `TensorViewT` | class | [include/lightning_core/core/tensor.hpp:100](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/tensor.hpp#L100) |

## C++ Function Symbols

| Function | Source |
| --- | --- |
| `attentionImplementation` | [include/lightning_core/core/attention.hpp:41](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/attention.hpp#L41) |
| `attentionImplementationName` | [include/lightning_core/core/attention.hpp:42](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/attention.hpp#L42) |
| `attentionUsesFallback` | [include/lightning_core/core/attention.hpp:43](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/attention.hpp#L43) |
| `config` | [include/lightning_core/core/attention.hpp:90](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/attention.hpp#L90) |
| `defaultPolicy` | [include/lightning_core/core/attention.hpp:94](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/attention.hpp#L94) |
| `device` | [include/lightning_core/core/attention.hpp:91](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/attention.hpp#L91) |
| `forward` | [include/lightning_core/core/attention.hpp:96](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/attention.hpp#L96) |
| `setDefaultPolicy` | [include/lightning_core/core/attention.hpp:93](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/attention.hpp#L93) |
| `activeBackendCapabilities` | [include/lightning_core/core/runtime.hpp:214](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L214) |
| `activeBackendInterfaceContract` | [include/lightning_core/core/runtime.hpp:220](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L220) |
| `applyDefaultSyncPolicy` | [include/lightning_core/core/runtime.hpp:205](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L205) |
| `backendCapabilities` | [include/lightning_core/core/runtime.hpp:211](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L211) |
| `backendInterfaceContract` | [include/lightning_core/core/runtime.hpp:217](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L217) |
| `backendName` | [include/lightning_core/core/runtime.hpp:183](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L183) |
| `clearRuntimeTraceEvents` | [include/lightning_core/core/runtime.hpp:229](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L229) |
| `defaultSyncPolicy` | [include/lightning_core/core/runtime.hpp:199](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L199) |
| `deviceMemoryModel` | [include/lightning_core/core/runtime.hpp:186](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L186) |
| `deviceSynchronize` | [include/lightning_core/core/runtime.hpp:168](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L168) |
| `deviceSynchronizeWithPolicy` | [include/lightning_core/core/runtime.hpp:202](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L202) |
| `encodeRuntimeTraceDispatchDetail` | [include/lightning_core/core/runtime.hpp:244](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L244) |
| `freeDevice` | [include/lightning_core/core/runtime.hpp:161](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L161) |
| `getDeviceCount` | [include/lightning_core/core/runtime.hpp:171](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L171) |
| `getErrorString` | [include/lightning_core/core/runtime.hpp:260](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L260) |
| `isCudaAvailable` | [include/lightning_core/core/runtime.hpp:174](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L174) |
| `isMetalAvailable` | [include/lightning_core/core/runtime.hpp:177](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L177) |
| `isRuntimeTraceEnabled` | [include/lightning_core/core/runtime.hpp:226](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L226) |
| `mallocDevice` | [include/lightning_core/core/runtime.hpp:157](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L157) |
| `memcpy` | [include/lightning_core/core/runtime.hpp:165](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L165) |
| `memoryModelName` | [include/lightning_core/core/runtime.hpp:189](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L189) |
| `preferredDeviceFor` | [include/lightning_core/core/runtime.hpp:180](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L180) |
| `preloadRuntimeProfileEnv` | [include/lightning_core/core/runtime.hpp:193](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L193) |
| `runtimeTraceEventCapacity` | [include/lightning_core/core/runtime.hpp:235](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L235) |
| `runtimeTraceEventTypeName` | [include/lightning_core/core/runtime.hpp:238](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L238) |
| `runtimeTraceEvents` | [include/lightning_core/core/runtime.hpp:232](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L232) |
| `runtimeTraceOpKindName` | [include/lightning_core/core/runtime.hpp:241](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L241) |
| `setDefaultSyncPolicy` | [include/lightning_core/core/runtime.hpp:196](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L196) |
| `setRuntimeTraceEnabled` | [include/lightning_core/core/runtime.hpp:223](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L223) |
| `syncModeName` | [include/lightning_core/core/runtime.hpp:208](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp#L208) |
