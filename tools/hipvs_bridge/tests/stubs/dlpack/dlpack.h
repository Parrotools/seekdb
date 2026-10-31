#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum { kDLCPU=1, kDLCUDA=2 } DLDeviceType;
typedef enum { kDLInt=0, kDLUInt=1, kDLFloat=2 } DLDataTypeCode;
typedef struct { DLDeviceType device_type; int32_t device_id; } DLDevice;
typedef struct { uint8_t code, bits; uint16_t lanes; } DLDataType;
typedef struct { void *data; DLDevice device; int32_t ndim; DLDataType dtype; int64_t *shape, *strides; uint64_t byte_offset; } DLTensor;
typedef struct DLManagedTensor { DLTensor dl_tensor; void *manager_ctx; void (*deleter)(struct DLManagedTensor*); } DLManagedTensor;
#ifdef __cplusplus
}
#endif
