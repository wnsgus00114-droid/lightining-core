# C/C++ API Reference (Generated)

Generated from public headers under `include/lightning_core/`.

Regenerate:
`python scripts/generate_api_reference_docs.py`

| Metric | Count |
| --- | --- |
| Headers scanned | 7 |
| Types (enum/struct/class) | 56 |
| C++ functions | 38 |
| C API functions | 25 |

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
| `lcApplyDefaultSyncPolicy` | [include/lightning_core/lightning_core.h:127](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L127) |
| `lcApplySyncPolicy` | [include/lightning_core/lightning_core.h:126](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L126) |
| `lcBackendName` | [include/lightning_core/lightning_core.h:136](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L136) |
| `lcCheckStructSize` | [include/lightning_core/lightning_core.h:135](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L135) |
| `lcDeviceSynchronize` | [include/lightning_core/lightning_core.h:116](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L116) |
| `lcFree` | [include/lightning_core/lightning_core.h:114](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L114) |
| `lcGetActiveBackendCapabilities` | [include/lightning_core/lightning_core.h:129](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L129) |
| `lcGetActiveBackendInterfaceContract` | [include/lightning_core/lightning_core.h:131](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L131) |
| `lcGetApiVersion` | [include/lightning_core/lightning_core.h:132](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L132) |
| `lcGetApiVersionString` | [include/lightning_core/lightning_core.h:133](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L133) |
| `lcGetBackendCapabilities` | [include/lightning_core/lightning_core.h:128](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L128) |
| `lcGetBackendInterfaceContract` | [include/lightning_core/lightning_core.h:130](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L130) |
| `lcGetDefaultSyncPolicy` | [include/lightning_core/lightning_core.h:125](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L125) |
| `lcGetDeviceCount` | [include/lightning_core/lightning_core.h:117](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L117) |
| `lcGetErrorString` | [include/lightning_core/lightning_core.h:137](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L137) |
| `lcGetMemoryModel` | [include/lightning_core/lightning_core.h:122](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L122) |
| `lcGetMemoryModelName` | [include/lightning_core/lightning_core.h:123](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L123) |
| `lcGetPreferredDeviceForInference` | [include/lightning_core/lightning_core.h:118](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L118) |
| `lcGetPreferredDeviceForTraining` | [include/lightning_core/lightning_core.h:119](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L119) |
| `lcGetStructSize` | [include/lightning_core/lightning_core.h:134](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L134) |
| `lcIsCudaAvailable` | [include/lightning_core/lightning_core.h:120](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L120) |
| `lcIsMetalAvailable` | [include/lightning_core/lightning_core.h:121](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L121) |
| `lcMalloc` | [include/lightning_core/lightning_core.h:113](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L113) |
| `lcMemcpy` | [include/lightning_core/lightning_core.h:115](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L115) |
| `lcSetDefaultSyncPolicy` | [include/lightning_core/lightning_core.h:124](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h#L124) |

## Types

| Name | Kind | Source |
| --- | --- | --- |
| `AttentionConfig` | struct | [include/lightning_core/core/attention.hpp:9](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/attention.hpp#L9) |
| `AttentionImplementation` | enum class | [include/lightning_core/core/attention.hpp:34](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/attention.hpp#L34) |
| `AttentionIoPolicy` | struct | [include/lightning_core/core/attention.hpp:16](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/attention.hpp#L16) |
| `AttentionSession` | class | [include/lightning_core/core/attention.hpp:86](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/attention.hpp#L86) |
| `CostProfileCoefficients` | struct | [include/lightning_core/core/graph.hpp:1751](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L1751) |
| `DType` | enum class | [include/lightning_core/core/graph.hpp:24](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L24) |
| `FusionCostEstimate` | struct | [include/lightning_core/core/graph.hpp:2131](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L2131) |
| `FusionPassKind` | enum class | [include/lightning_core/core/graph.hpp:570](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L570) |
| `FusionPattern` | enum class | [include/lightning_core/core/graph.hpp:548](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L548) |
| `GraphExecutionGroup` | struct | [include/lightning_core/core/graph.hpp:501](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L501) |
| `GraphFusionDecision` | struct | [include/lightning_core/core/graph.hpp:592](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L592) |
| `GraphIR` | class | [include/lightning_core/core/graph.hpp:618](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L618) |
| `GraphNode` | struct | [include/lightning_core/core/graph.hpp:339](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L339) |
| `GraphPlanCacheStats` | struct | [include/lightning_core/core/graph.hpp:542](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L542) |
| `GraphPlanStep` | struct | [include/lightning_core/core/graph.hpp:349](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L349) |
| `GraphPlanSummary` | struct | [include/lightning_core/core/graph.hpp:512](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L512) |
| `GraphPlannerOptions` | struct | [include/lightning_core/core/graph.hpp:462](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L462) |
| `GraphTensorValue` | struct | [include/lightning_core/core/graph.hpp:333](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L333) |
| `OpKind` | enum class | [include/lightning_core/core/graph.hpp:39](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L39) |
| `OperatorRegistry` | class | [include/lightning_core/core/graph.hpp:114](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L114) |
| `OperatorSchema` | struct | [include/lightning_core/core/graph.hpp:82](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L82) |
| `PlanCacheEntry` | struct | [include/lightning_core/core/graph.hpp:1761](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L1761) |
| `PlanCostTelemetry` | struct | [include/lightning_core/core/graph.hpp:1740](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L1740) |
| `ReasonCode` | enum class | [include/lightning_core/core/graph.hpp:383](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L383) |
| `TensorSpec` | struct | [include/lightning_core/core/graph.hpp:76](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L76) |
| `ValidationIssue` | struct | [include/lightning_core/core/graph.hpp:441](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L441) |
| `ValidationPass` | enum class | [include/lightning_core/core/graph.hpp:358](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L358) |
| `ValidationReport` | struct | [include/lightning_core/core/graph.hpp:450](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp#L450) |
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
