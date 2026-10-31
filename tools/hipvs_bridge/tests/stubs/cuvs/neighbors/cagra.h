#pragma once
#include <cuvs/core/c_api.h>
#include <dlpack/dlpack.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum { NN_DESCENT } cuvsCagraGraphBuildAlgo;
typedef struct { cuvsCagraGraphBuildAlgo build_algo; size_t graph_degree, intermediate_graph_degree, nn_descent_niter; } cuvsCagraIndexParams;
typedef cuvsCagraIndexParams* cuvsCagraIndexParams_t;
typedef struct { size_t itopk_size, search_width; } cuvsCagraSearchParams;
typedef cuvsCagraSearchParams* cuvsCagraSearchParams_t;
typedef struct { int64_t n, dim; } cuvsCagraIndex;
typedef cuvsCagraIndex* cuvsCagraIndex_t;
typedef enum { NO_FILTER } cuvsFilterType;
typedef struct { cuvsFilterType type; uintptr_t addr; } cuvsFilter;
cuvsError_t cuvsCagraIndexParamsCreate(cuvsCagraIndexParams_t*);
cuvsError_t cuvsCagraIndexParamsDestroy(cuvsCagraIndexParams_t);
cuvsError_t cuvsCagraSearchParamsCreate(cuvsCagraSearchParams_t*);
cuvsError_t cuvsCagraSearchParamsDestroy(cuvsCagraSearchParams_t);
cuvsError_t cuvsCagraIndexCreate(cuvsCagraIndex_t*);
cuvsError_t cuvsCagraIndexDestroy(cuvsCagraIndex_t);
cuvsError_t cuvsCagraBuild(cuvsResources_t,cuvsCagraIndexParams_t,DLManagedTensor*,cuvsCagraIndex_t);
cuvsError_t cuvsCagraSearch(cuvsResources_t,cuvsCagraSearchParams_t,cuvsCagraIndex_t,DLManagedTensor*,DLManagedTensor*,DLManagedTensor*,cuvsFilter);
cuvsError_t cuvsCagraSerialize(cuvsResources_t,const char*,cuvsCagraIndex_t,bool);
cuvsError_t cuvsCagraDeserialize(cuvsResources_t,const char*,cuvsCagraIndex_t);
cuvsError_t cuvsCagraIndexGetSize(cuvsCagraIndex_t,int64_t*);
cuvsError_t cuvsCagraIndexGetDims(cuvsCagraIndex_t,int64_t*);
#ifdef __cplusplus
}
#endif
