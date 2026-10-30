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
#include <pthread.h>

typedef struct {
  cuvsResources_t        res;
  cuvsCagraIndexParams_t ip;
  cuvsCagraIndex_t       idx;
  long n, dim;
  /* Reusable device buffers to eliminate per-query allocation */
  float*        d_query;
  size_t        d_query_cap;
  unsigned int* d_ids;
  size_t        d_ids_cap;
  float*        d_dist;
  size_t        d_dist_cap;
  cuvsCagraSearchParams_t sp;
  pthread_mutex_t lock;
} sk_cuvs_index;

void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  sk_cuvs_index* h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  pthread_mutex_init(&h->lock, NULL);
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { pthread_mutex_destroy(&h->lock); free(h); return NULL; }

  DLManagedTensor dset; long ds[2] = {n, dim};
  dset.dl_tensor.data = (void*)base;
  dset.dl_tensor.device.device_type = kDLCPU; dset.dl_tensor.device.device_id = 0;
  dset.dl_tensor.ndim = 2;
  dset.dl_tensor.dtype.code = kDLFloat; dset.dl_tensor.dtype.bits = 32; dset.dl_tensor.dtype.lanes = 1;
  dset.dl_tensor.shape = ds; dset.dl_tensor.strides = 0; dset.dl_tensor.byte_offset = 0;

  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) goto fail_res;
  h->ip->build_algo = NN_DESCENT;
  h->ip->graph_degree = 64;
  h->ip->intermediate_graph_degree = 128;
  h->ip->nn_descent_niter = (dim <= 128) ? 20 : 25;
  if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) != CUVS_SUCCESS) goto fail_idx;
  h->n = n; h->dim = dim;
  return h;

fail_idx: cuvsCagraIndexDestroy(h->idx);
fail_ip:  cuvsCagraIndexParamsDestroy(h->ip);
fail_res: cuvsResourcesDestroy(h->res); pthread_mutex_destroy(&h->lock); free(h);
  fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
  return NULL;
}

int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h || !query || !out_ids || nq <= 0 || topk <= 0) return 1;
  int rc = 0;
  size_t needed_q = (size_t)nq * h->dim;
  size_t needed_out = (size_t)nq * topk;

  pthread_mutex_lock(&h->lock);

  /* Lazy create search params once */
  if (!h->sp) {
    if (cuvsCagraSearchParamsCreate(&h->sp) != CUVS_SUCCESS) {
      rc = 14;
      goto done;
    }
  }

  /* Ensure query device buffer */
  if (h->d_query_cap < needed_q) {
    if (h->d_query) {
      cuvsRMMFree(h->res, h->d_query, sizeof(float) * h->d_query_cap);
      h->d_query = NULL;
      h->d_query_cap = 0;
    }
    size_t alloc_q = needed_q < 4096 * (size_t)h->dim ? 4096 * (size_t)h->dim : needed_q;
    if (cuvsRMMAlloc(h->res, (void**)&h->d_query, sizeof(float) * alloc_q) != CUVS_SUCCESS) {
      rc = 10;
      goto done;
    }
    h->d_query_cap = alloc_q;
  }

  /* Ensure ids device buffer */
  if (h->d_ids_cap < needed_out) {
    if (h->d_ids) {
      cuvsRMMFree(h->res, h->d_ids, sizeof(unsigned int) * h->d_ids_cap);
      h->d_ids = NULL;
      h->d_ids_cap = 0;
    }
    size_t alloc_out = needed_out < 4096 * 100 ? (size_t)4096 * 100 : needed_out;
    if (cuvsRMMAlloc(h->res, (void**)&h->d_ids, sizeof(unsigned int) * alloc_out) != CUVS_SUCCESS) {
      rc = 11;
      goto done;
    }
    h->d_ids_cap = alloc_out;
  }

  /* Ensure dist device buffer */
  if (h->d_dist_cap < needed_out) {
    if (h->d_dist) {
      cuvsRMMFree(h->res, h->d_dist, sizeof(float) * h->d_dist_cap);
      h->d_dist = NULL;
      h->d_dist_cap = 0;
    }
    size_t alloc_out = needed_out < 4096 * 100 ? (size_t)4096 * 100 : needed_out;
    if (cuvsRMMAlloc(h->res, (void**)&h->d_dist, sizeof(float) * alloc_out) != CUVS_SUCCESS) {
      rc = 12;
      goto done;
    }
    h->d_dist_cap = alloc_out;
  }

  /* H2D transfer */
  if (cudaMemcpy(h->d_query, query, sizeof(float) * needed_q, cudaMemcpyDefault) != cudaSuccess) {
    rc = 13;
    goto done;
  }

  {
    DLManagedTensor qt, nt, dt;
    long qs[2]  = {nq, h->dim};
    long nsh[2] = {nq, topk};
    long dsh[2] = {nq, topk};

    qt.dl_tensor.data = h->d_query;
    qt.dl_tensor.device.device_type = kDLCUDA;
    qt.dl_tensor.device.device_id = 0;
    qt.dl_tensor.ndim = 2;
    qt.dl_tensor.dtype.code = kDLFloat;
    qt.dl_tensor.dtype.bits = 32;
    qt.dl_tensor.dtype.lanes = 1;
    qt.dl_tensor.shape = qs;
    qt.dl_tensor.strides = 0;
    qt.dl_tensor.byte_offset = 0;

    nt = qt;
    nt.dl_tensor.data = h->d_ids;
    nt.dl_tensor.dtype.code = kDLUInt;
    nt.dl_tensor.shape = nsh;

    dt = qt;
    dt.dl_tensor.data = h->d_dist;
    dt.dl_tensor.dtype.code = kDLFloat;
    dt.dl_tensor.shape = dsh;

    size_t required_itopk = (size_t)topk * 4;
    if (required_itopk < 512) required_itopk = 512;
    if (topk >= 100 && required_itopk < 768) required_itopk = 768;
    h->sp->itopk_size = required_itopk;
    h->sp->search_width = 4;

    cuvsFilter f;
    f.type = NO_FILTER;
    f.addr = (uintptr_t)0;

    if (cuvsCagraSearch(h->res, h->sp, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) {
      rc = 15;
      goto done;
    }

    if (cudaMemcpy(out_ids, h->d_ids, sizeof(unsigned int) * needed_out, cudaMemcpyDefault) != cudaSuccess) {
      rc = 16;
      goto done;
    }
    if (out_dist) {
      if (cudaMemcpy(out_dist, h->d_dist, sizeof(float) * needed_out, cudaMemcpyDefault) != cudaSuccess) {
        rc = 17;
        goto done;
      }
    }
  }

done:
  pthread_mutex_unlock(&h->lock);
  if (rc) fprintf(stderr, "[seekdb_cuvs] search rc=%d: %s\n", rc, cuvsGetLastErrorText());
  return rc;
}

void seekdb_cuvs_free(void* handle)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return;
  if (h->d_query) cuvsRMMFree(h->res, h->d_query, sizeof(float) * h->d_query_cap);
  if (h->d_ids)   cuvsRMMFree(h->res, h->d_ids, sizeof(unsigned int) * h->d_ids_cap);
  if (h->d_dist)  cuvsRMMFree(h->res, h->d_dist, sizeof(float) * h->d_dist_cap);
  if (h->sp)      cuvsCagraSearchParamsDestroy(h->sp);
  if (h->idx)     cuvsCagraIndexDestroy(h->idx);
  if (h->ip)      cuvsCagraIndexParamsDestroy(h->ip);
  if (h->res)     cuvsResourcesDestroy(h->res);
  pthread_mutex_destroy(&h->lock);
  free(h);
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  sk_cuvs_index* index = (sk_cuvs_index*)handle;
  if (!index || !filename) return 1;
  if (cuvsCagraSerialize(index->res, filename, index->idx, true) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] serialize failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  return 0;
}

void* seekdb_cuvs_deserialize(const char* filename, long n, long dim)
{
  if (!filename || n <= 0 || dim <= 0) return NULL;
  sk_cuvs_index* index = (sk_cuvs_index*)calloc(1, sizeof(*index));
  if (!index) return NULL;
  pthread_mutex_init(&index->lock, NULL);
  if (cuvsResourcesCreate(&index->res) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraIndexParamsCreate(&index->ip) != CUVS_SUCCESS) goto fail_res;
  if (cuvsCagraIndexCreate(&index->idx) != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraDeserialize(index->res, filename, index->idx) != CUVS_SUCCESS) goto fail_idx;
  {
    int64_t actual_n = 0;
    int64_t actual_dim = 0;
    if (cuvsCagraIndexGetSize(index->idx, &actual_n) != CUVS_SUCCESS ||
        cuvsCagraIndexGetDims(index->idx, &actual_dim) != CUVS_SUCCESS ||
        actual_n != n || actual_dim != dim) {
      fprintf(stderr, "[seekdb_cuvs] snapshot shape mismatch: %ld x %ld, expected %ld x %ld\n",
              (long)actual_n, (long)actual_dim, n, dim);
      goto fail_idx;
    }
  }
  index->n = n;
  index->dim = dim;
  return index;
fail_idx:
  cuvsCagraIndexDestroy(index->idx);
fail_ip:
  cuvsCagraIndexParamsDestroy(index->ip);
fail_res:
  cuvsResourcesDestroy(index->res);
fail:
  fprintf(stderr, "[seekdb_cuvs] deserialize failed: %s\n", cuvsGetLastErrorText());
  pthread_mutex_destroy(&index->lock);
  free(index);
  return NULL;
}

int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                          const float* query, long nq, long topk,
                          unsigned int* out_ids)
{
  void* h = seekdb_cuvs_build(base, n, dim);
  if (!h) return 10;
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}
