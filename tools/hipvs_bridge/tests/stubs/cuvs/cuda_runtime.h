#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum { cudaSuccess=0, cudaErrorUnknown=1 } cudaError_t;
typedef enum { cudaMemcpyDefault } cudaMemcpyKind;
cudaError_t cudaMemcpy(void*,const void*,size_t,cudaMemcpyKind);
#ifdef __cplusplus
}
#endif
