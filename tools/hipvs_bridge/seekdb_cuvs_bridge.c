/* seekdb <-> hipVS bridge: plain-C CAGRA kNN; all cuVS/ROCm deps live here. */
#include "seekdb_cuvs_bridge.h"
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>

typedef struct {
  cuvsResources_t res;
  cuvsCagraIndexParams_t ip;
  cuvsCagraIndex_t idx;
  long n, dim;
  void *query, *ids, *dist;
  size_t query_bytes, ids_bytes, dist_bytes;
  cuvsCagraSearchParams_t sp;
  size_t default_itopk;
  pthread_mutex_t lock;
} sk_cuvs_index;

/* Validate before multiplication, allocation, or passing shapes into cuVS. */
static int sk_bytes(long rows, long cols, size_t width, size_t* bytes)
{
  if (rows <= 0 || cols <= 0 || (uint64_t)cols > SIZE_MAX / width ||
      (uint64_t)rows > SIZE_MAX / width / (uint64_t)cols) return 0;
  *bytes = (size_t)rows * (size_t)cols * width;
  return 1;
}

static void sk_release_workspace(sk_cuvs_index* h)
{
  if (h->query) cuvsRMMFree(h->res, h->query, h->query_bytes);
  if (h->ids) cuvsRMMFree(h->res, h->ids, h->ids_bytes);
  if (h->dist) cuvsRMMFree(h->res, h->dist, h->dist_bytes);
  h->query = h->ids = h->dist = NULL;
  h->query_bytes = h->ids_bytes = h->dist_bytes = 0;
}

/* Free before growing to avoid doubling the peak workspace. On failure the
 * buffer stays empty and the handle can be retried. Sizes are always bytes. */
static int sk_reserve(sk_cuvs_index* h, void** buffer, size_t* capacity, size_t bytes)
{
  void* next = NULL;
  if (*capacity >= bytes) return 1;
  if (*buffer) {
    if (cuvsRMMFree(h->res, *buffer, *capacity) != CUVS_SUCCESS) return 0;
    *buffer = NULL;
    *capacity = 0;
  }
  if (cuvsRMMAlloc(h->res, &next, bytes) != CUVS_SUCCESS) return 0;
  *buffer = next;
  *capacity = bytes;
  return 1;
}

void seekdb_cuvs_free(void* handle)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return;
  sk_release_workspace(h);
  if (h->sp) cuvsCagraSearchParamsDestroy(h->sp);
  if (h->idx) cuvsCagraIndexDestroy(h->idx);
  if (h->ip) cuvsCagraIndexParamsDestroy(h->ip);
  if (h->res) cuvsResourcesDestroy(h->res);
  pthread_mutex_destroy(&h->lock);
  free(h);
}

static sk_cuvs_index* sk_create(void)
{
  sk_cuvs_index* h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  if (pthread_mutex_init(&h->lock, NULL) != 0) { free(h); return NULL; }
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) goto fail;
  return h;
fail:
  seekdb_cuvs_free(h);
  return NULL;
}

void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  size_t bytes = 0;
  sk_cuvs_index* h;
  DLManagedTensor dset;
  int64_t shape[2] = {n, dim};
  if (!base || (uint64_t)n > UINT32_MAX || !sk_bytes(n, dim, sizeof(float), &bytes)) return NULL;
  h = sk_create();
  if (!h) return NULL;
  memset(&dset, 0, sizeof(dset));
  dset.dl_tensor.data = (void*)base;
  dset.dl_tensor.device.device_type = kDLCPU;
  dset.dl_tensor.ndim = 2;
  dset.dl_tensor.dtype.code = kDLFloat;
  dset.dl_tensor.dtype.bits = 32;
  dset.dl_tensor.dtype.lanes = 1;
  dset.dl_tensor.shape = shape;
  /* Preserve the V5 graph-quality settings until paired GPU recall is measured. */
  h->ip->build_algo = NN_DESCENT;
  h->ip->graph_degree = 128;
  h->ip->intermediate_graph_degree = 256;
  h->ip->nn_descent_niter = 100;
  if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) != CUVS_SUCCESS ||
      cuvsStreamSync(h->res) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
    seekdb_cuvs_free(h);
    return NULL;
  }
  h->n = n;
  h->dim = dim;
  return h;
}

int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  size_t query_bytes = 0, output_bytes = 0;
  int rc = 0;
  if (!h || !query || !out_ids || topk > h->n ||
      (uint64_t)topk > SIZE_MAX / 4 ||
      !sk_bytes(nq, h->dim, sizeof(float), &query_bytes) ||
      !sk_bytes(nq, topk, sizeof(unsigned int), &output_bytes)) return 1;
  if (pthread_mutex_lock(&h->lock) != 0) return 2;
  if (!h->sp) {
    if (cuvsCagraSearchParamsCreate(&h->sp) != CUVS_SUCCESS) { rc = 14; goto done; }
    h->default_itopk = h->sp->itopk_size;
  }
  if (!sk_reserve(h, &h->query, &h->query_bytes, query_bytes)) { rc = 10; goto done; }
  if (!sk_reserve(h, &h->ids, &h->ids_bytes, output_bytes)) { rc = 11; goto done; }
  if (!sk_reserve(h, &h->dist, &h->dist_bytes, output_bytes)) { rc = 12; goto done; }
  if (cudaMemcpy(h->query, query, query_bytes, cudaMemcpyDefault) != cudaSuccess) { rc = 13; goto done; }
  {
    DLManagedTensor qt, nt, dt;
    memset(&qt, 0, sizeof(qt));
    int64_t qs[2] = {nq, h->dim}, os[2] = {nq, topk};
    size_t required_itopk = (size_t)topk * 4;
    if (required_itopk < 512) required_itopk = 512;
    if (required_itopk < h->default_itopk) required_itopk = h->default_itopk;
    /* Reset each call: a previous larger k must not change this query's work. */
    h->sp->itopk_size = required_itopk;
    h->sp->search_width = 4;
    qt.dl_tensor.data = h->query;
    qt.dl_tensor.device.device_type = kDLCUDA;
    qt.dl_tensor.ndim = 2;
    qt.dl_tensor.dtype.code = kDLFloat;
    qt.dl_tensor.dtype.bits = 32;
    qt.dl_tensor.dtype.lanes = 1;
    qt.dl_tensor.shape = qs;
    nt = qt;
    nt.dl_tensor.data = h->ids;
    nt.dl_tensor.dtype.code = kDLUInt;
    nt.dl_tensor.shape = os;
    dt = qt;
    dt.dl_tensor.data = h->dist;
    dt.dl_tensor.shape = os;
    cuvsFilter filter;
    filter.type = NO_FILTER;
    filter.addr = (uintptr_t)0;
    if (cuvsCagraSearch(h->res, h->sp, h->idx, &qt, &nt, &dt, filter) != CUVS_SUCCESS) { rc = 15; goto done; }
    if (cuvsStreamSync(h->res) != CUVS_SUCCESS) { rc = 18; goto done; }
    if (cudaMemcpy(out_ids, h->ids, output_bytes, cudaMemcpyDefault) != cudaSuccess) { rc = 16; goto done; }
    if (out_dist && cudaMemcpy(out_dist, h->dist, output_bytes, cudaMemcpyDefault) != cudaSuccess) { rc = 17; goto done; }
  }
done:
  /* Drain any work queued before a reported search/copy failure before the
   * workspace may be released or used by another thread. */
  if (rc) cuvsStreamSync(h->res);
  /* Retain at most 8 MiB per handle. Large one-off batches release workspace;
   * ordinary single-row SQL keeps only the few KiB it actually needs. */
  if (h->query_bytes > 4UL * 1024 * 1024 || h->ids_bytes > 2UL * 1024 * 1024 ||
      h->dist_bytes > 2UL * 1024 * 1024) sk_release_workspace(h);
  if (rc) fprintf(stderr, "[seekdb_cuvs] search rc=%d: %s\n", rc, cuvsGetLastErrorText());
  pthread_mutex_unlock(&h->lock);
  return rc;
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  int rc;
  if (!h || !filename) return 1;
  if (pthread_mutex_lock(&h->lock) != 0) return 1;
  rc = cuvsCagraSerialize(h->res, filename, h->idx, true) != CUVS_SUCCESS;
  if (!rc) rc = cuvsStreamSync(h->res) != CUVS_SUCCESS;
  if (rc) fprintf(stderr, "[seekdb_cuvs] serialize failed: %s\n", cuvsGetLastErrorText());
  pthread_mutex_unlock(&h->lock);
  return rc;
}

void* seekdb_cuvs_deserialize(const char* filename, long n, long dim)
{
  size_t bytes = 0;
  int64_t actual_n = 0, actual_dim = 0;
  sk_cuvs_index* h;
  if (!filename || (uint64_t)n > UINT32_MAX || !sk_bytes(n, dim, sizeof(float), &bytes)) return NULL;
  h = sk_create();
  if (!h) return NULL;
  if (cuvsCagraDeserialize(h->res, filename, h->idx) != CUVS_SUCCESS ||
      cuvsStreamSync(h->res) != CUVS_SUCCESS ||
      cuvsCagraIndexGetSize(h->idx, &actual_n) != CUVS_SUCCESS ||
      cuvsCagraIndexGetDims(h->idx, &actual_dim) != CUVS_SUCCESS ||
      actual_n != n || actual_dim != dim) {
    fprintf(stderr, "[seekdb_cuvs] deserialize failed or shape mismatch: %s\n", cuvsGetLastErrorText());
    seekdb_cuvs_free(h);
    return NULL;
  }
  h->n = n;
  h->dim = dim;
  return h;
}

int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                          const float* query, long nq, long topk,
                          unsigned int* out_ids)
{
  void* h = seekdb_cuvs_build(base, n, dim);
  int rc;
  if (!h) return 10;
  rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}
