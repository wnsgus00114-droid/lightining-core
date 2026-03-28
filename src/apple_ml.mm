#include "lightning_core/core/apple_ml.hpp"

#if defined(CJ_PLATFORM_MACOS) && CJ_PLATFORM_MACOS

#import <CoreML/CoreML.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <chrono>
#include <vector>

namespace lightning_core::apple {

namespace {

NSURL* resolveModelURL(const std::string& path, NSError** err) {
  NSString* p = [NSString stringWithUTF8String:path.c_str()];
  NSURL* inputURL = [NSURL fileURLWithPath:p];
  if ([[inputURL pathExtension] isEqualToString:@"mlmodel"]) {
    NSURL* compiled = [MLModel compileModelAtURL:inputURL error:err];
    return compiled;
  }
  return inputURL;
}

MLFeatureValue* makeDefaultFeatureValue(MLFeatureDescription* desc, std::size_t fallbackN, NSError** err) {
  switch (desc.type) {
    case MLFeatureTypeInt64:
      return [MLFeatureValue featureValueWithInt64:0];
    case MLFeatureTypeDouble:
      return [MLFeatureValue featureValueWithDouble:0.0];
    case MLFeatureTypeString:
      return [MLFeatureValue featureValueWithString:@""];
    case MLFeatureTypeDictionary: {
      return [MLFeatureValue featureValueWithDictionary:@{} error:err];
    }
    case MLFeatureTypeMultiArray: {
      MLMultiArrayConstraint* c = desc.multiArrayConstraint;
      MLMultiArrayDataType dt = c != nil ? c.dataType : MLMultiArrayDataTypeFloat32;
      NSArray<NSNumber*>* shape = c != nil ? c.shape : nil;
      if (shape == nil || shape.count == 0) {
        shape = @[ @(fallbackN) ];
      }

      NSMutableArray<NSNumber*>* fixedShape = [NSMutableArray arrayWithCapacity:shape.count];
      for (NSNumber* d in shape) {
        long long v = d.longLongValue;
        if (v <= 0) {
          [fixedShape addObject:@(fallbackN > 0 ? fallbackN : 1)];
        } else {
          [fixedShape addObject:@(v)];
        }
      }

      MLMultiArray* arr = [[MLMultiArray alloc] initWithShape:fixedShape dataType:dt error:err];
      if (arr == nil) {
        return nil;
      }

      // 기본 입력값은 0으로 채운다.
      std::size_t total = 1;
      for (NSNumber* d in fixedShape) {
        total *= static_cast<std::size_t>(d.unsignedLongLongValue);
      }

      if (dt == MLMultiArrayDataTypeFloat32) {
        float* p = static_cast<float*>(arr.dataPointer);
        for (std::size_t i = 0; i < total; ++i) p[i] = 0.0f;
      } else if (dt == MLMultiArrayDataTypeDouble) {
        double* p = static_cast<double*>(arr.dataPointer);
        for (std::size_t i = 0; i < total; ++i) p[i] = 0.0;
      } else if (dt == MLMultiArrayDataTypeInt32) {
        int32_t* p = static_cast<int32_t*>(arr.dataPointer);
        for (std::size_t i = 0; i < total; ++i) p[i] = 0;
      }

      return [MLFeatureValue featureValueWithMultiArray:arr];
    }
    case MLFeatureTypeImage:
    case MLFeatureTypeSequence:
#if defined(MLFeatureTypeState)
    case MLFeatureTypeState:
#endif
      // 이미지/시퀀스/상태는 모델별 제약이 커서 기본값 생성이 어렵다.
      if (desc.isOptional) {
        return [MLFeatureValue undefinedFeatureValueWithType:desc.type];
      }
      return nil;
    case MLFeatureTypeInvalid:
    default:
      return nil;
  }
}

runtime::Status makeTensorData(id<MTLDevice> device,
                               const std::vector<float>& src,
                               MPSGraphTensorData** outData,
                               MPSShape* shape) {
  if (outData == nullptr) {
    return runtime::Status::kInvalidValue;
  }
  MPSGraphDevice* gdev = [MPSGraphDevice deviceWithMTLDevice:device];
  NSData* raw = [NSData dataWithBytes:src.data() length:src.size() * sizeof(float)];
  MPSGraphTensorData* data = [[MPSGraphTensorData alloc] initWithDevice:gdev
                                                                    data:raw
                                                                   shape:shape
                                                                dataType:MPSDataTypeFloat32];
  if (!data) {
    return runtime::Status::kOutOfMemory;
  }
  *outData = data;
  return runtime::Status::kSuccess;
}

}  // namespace

runtime::Status benchmarkCoreMLInference(const std::string& modelPath, std::size_t n, int iters, double* avgMs) {
  if (modelPath.empty() || n == 0 || iters <= 0 || avgMs == nullptr) {
    return runtime::Status::kInvalidValue;
  }

  @autoreleasepool {
    NSError* err = nil;
    NSURL* modelURL = resolveModelURL(modelPath, &err);
    if (!modelURL) {
      return runtime::Status::kNotSupported;
    }

    MLModelConfiguration* cfg = [[MLModelConfiguration alloc] init];
    if (@available(macOS 13.0, *)) {
      cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    } else {
      cfg.computeUnits = MLComputeUnitsAll;
    }

    MLModel* model = [MLModel modelWithContentsOfURL:modelURL configuration:cfg error:&err];
    if (!model) {
      return runtime::Status::kNotSupported;
    }

    NSDictionary<NSString*, MLFeatureDescription*>* inputs = model.modelDescription.inputDescriptionsByName;
    if (inputs.count == 0) {
      return runtime::Status::kInvalidValue;
    }

    NSMutableDictionary<NSString*, MLFeatureValue*>* feed = [NSMutableDictionary dictionaryWithCapacity:inputs.count];
    for (NSString* key in inputs) {
      MLFeatureDescription* desc = inputs[key];
      MLFeatureValue* value = makeDefaultFeatureValue(desc, n, &err);
      if (value == nil) {
        return runtime::Status::kNotSupported;
      }
      feed[key] = value;
    }

    MLDictionaryFeatureProvider* provider =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:feed error:&err];
    if (!provider) {
      return runtime::Status::kInvalidValue;
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
      id<MLFeatureProvider> outProvider = [model predictionFromFeatures:provider error:&err];
      if (!outProvider) {
        return runtime::Status::kDriverError;
      }
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    *avgMs = elapsed.count() / static_cast<double>(iters);
    return runtime::Status::kSuccess;
  }
}

runtime::Status benchmarkMpsGraphVectorAdd(std::size_t n, int iters, double* avgMs) {
  if (n == 0 || iters <= 0 || avgMs == nullptr) {
    return runtime::Status::kInvalidValue;
  }

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      return runtime::Status::kNotSupported;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
      return runtime::Status::kDriverError;
    }

    MPSGraph* graph = [[MPSGraph alloc] init];
    MPSShape* shape = @[ @(n) ];

    MPSGraphTensor* a = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"a"];
    MPSGraphTensor* b = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"b"];
    MPSGraphTensor* c = [graph additionWithPrimaryTensor:a secondaryTensor:b name:@"c"];

    std::vector<float> hostA(n, 1.0f);
    std::vector<float> hostB(n, 2.0f);

    MPSGraphTensorData* dataA = nil;
    MPSGraphTensorData* dataB = nil;
    if (makeTensorData(device, hostA, &dataA, shape) != runtime::Status::kSuccess ||
        makeTensorData(device, hostB, &dataB, shape) != runtime::Status::kSuccess) {
      return runtime::Status::kOutOfMemory;
    }

    MPSGraphTensorDataDictionary* feeds = @{a : dataA, b : dataB};

    MPSGraphTensorData* lastOutData = nil;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
      MPSGraphTensorDataDictionary* results =
          [graph runWithMTLCommandQueue:queue feeds:feeds targetTensors:@[ c ] targetOperations:nil];
      lastOutData = results[c];
      if (!lastOutData) {
        return runtime::Status::kDriverError;
      }
    }
    if (!lastOutData) {
      return runtime::Status::kDriverError;
    }
    MPSNDArray* ndarray = [lastOutData mpsndarray];
    if (!ndarray) {
      return runtime::Status::kDriverError;
    }
    std::vector<float> tmp(n, 0.0f);
    [ndarray readBytes:tmp.data() strideBytes:nil];
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    *avgMs = elapsed.count() / static_cast<double>(iters);
    return runtime::Status::kSuccess;
  }
}

runtime::Status benchmarkMpsGraphTrainStep(std::size_t n, int iters, double* avgMs) {
  if (n == 0 || iters <= 0 || avgMs == nullptr) {
    return runtime::Status::kInvalidValue;
  }

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      return runtime::Status::kNotSupported;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
      return runtime::Status::kDriverError;
    }

    MPSGraph* graph = [[MPSGraph alloc] init];
    MPSShape* shape = @[ @(n) ];

    MPSGraphTensor* w = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"w"];
    MPSGraphTensor* g = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"g"];
    MPSGraphTensor* lr = [graph constantWithScalar:0.001 shape:shape dataType:MPSDataTypeFloat32];
    MPSGraphTensor* scaled = [graph multiplicationWithPrimaryTensor:g secondaryTensor:lr name:@"scaled"];
    MPSGraphTensor* wNew = [graph subtractionWithPrimaryTensor:w secondaryTensor:scaled name:@"w_new"];

    std::vector<float> hostW(n, 1.0f);
    std::vector<float> hostG(n, 0.1f);

    MPSGraphTensorData* dataG = nil;
    if (makeTensorData(device, hostG, &dataG, shape) != runtime::Status::kSuccess) {
      return runtime::Status::kOutOfMemory;
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
      MPSGraphTensorData* dataW = nil;
      if (makeTensorData(device, hostW, &dataW, shape) != runtime::Status::kSuccess) {
        return runtime::Status::kOutOfMemory;
      }

      MPSGraphTensorDataDictionary* feeds = @{w : dataW, g : dataG};
      MPSGraphTensorDataDictionary* results =
          [graph runWithMTLCommandQueue:queue feeds:feeds targetTensors:@[ wNew ] targetOperations:nil];
      MPSGraphTensorData* outData = results[wNew];
      if (!outData) {
        return runtime::Status::kDriverError;
      }
      MPSNDArray* ndarray = [outData mpsndarray];
      if (!ndarray) {
        return runtime::Status::kDriverError;
      }
      [ndarray readBytes:hostW.data() strideBytes:nil];
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    *avgMs = elapsed.count() / static_cast<double>(iters);
    return runtime::Status::kSuccess;
  }
}

}  // namespace lightning_core::apple

#endif
