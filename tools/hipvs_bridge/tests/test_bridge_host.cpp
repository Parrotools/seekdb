// Tests the real bridge control flow with a host-only cuVS/runtime double.
// These tests do not execute GPU kernels and cannot establish ANN quality.
#include "../seekdb_cuvs_bridge.h"
#include <cuvs/neighbors/cagra.h>
#include <cuvs/cuda_runtime.h>
#include <atomic>
#include <cassert>
#include <climits>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <thread>
#include <vector>

static std::map<void*,size_t> allocations;
static size_t alloc_calls=0, peak_bytes=0, live_bytes=0, resource_count=0, param_count=0, index_count=0;
static int fail_alloc_after=-1, fail_copy_after=-1;
static bool fail_build=false, fail_sync=false, fail_load=false, fail_search=false;
static std::atomic<int> in_search{0};
static size_t last_itopk=0;
static bool benchmark_mode=false;
static void balanced() { assert(allocations.empty()); assert(resource_count==0); assert(param_count==0); assert(index_count==0); }
extern "C" {
cuvsError_t cuvsResourcesCreate(cuvsResources_t* r) { *r=++resource_count; return CUVS_SUCCESS; }
cuvsError_t cuvsResourcesDestroy(cuvsResources_t) { --resource_count; return CUVS_SUCCESS; }
cuvsError_t cuvsStreamSync(cuvsResources_t) { return fail_sync?CUVS_ERROR:CUVS_SUCCESS; }
const char* cuvsGetLastErrorText() { return "injected host-test failure"; }
cuvsError_t cuvsRMMAlloc(cuvsResources_t,void** p,size_t n) {
  if(fail_alloc_after==0) { fail_alloc_after=-1; return CUVS_ERROR; }
  if(fail_alloc_after>0) --fail_alloc_after;
  *p=std::malloc(n); assert(*p); allocations[*p]=n; ++alloc_calls; live_bytes+=n;
  if(live_bytes>peak_bytes) peak_bytes=live_bytes;
  return CUVS_SUCCESS;
}
cuvsError_t cuvsRMMFree(cuvsResources_t,void* p,size_t n) {
  assert(allocations.count(p)==1 && allocations[p]==n);
  live_bytes-=n; allocations.erase(p); std::free(p); return CUVS_SUCCESS;
}
cuvsError_t cuvsCagraIndexParamsCreate(cuvsCagraIndexParams_t* p) { *p=new cuvsCagraIndexParams{}; ++param_count; return CUVS_SUCCESS; }
cuvsError_t cuvsCagraIndexParamsDestroy(cuvsCagraIndexParams_t p) { delete p; --param_count; return CUVS_SUCCESS; }
cuvsError_t cuvsCagraSearchParamsCreate(cuvsCagraSearchParams_t* p) { *p=new cuvsCagraSearchParams{128,1}; ++param_count; return CUVS_SUCCESS; }
cuvsError_t cuvsCagraSearchParamsDestroy(cuvsCagraSearchParams_t p) { delete p; --param_count; return CUVS_SUCCESS; }
cuvsError_t cuvsCagraIndexCreate(cuvsCagraIndex_t* p) { *p=new cuvsCagraIndex{}; ++index_count; return CUVS_SUCCESS; }
cuvsError_t cuvsCagraIndexDestroy(cuvsCagraIndex_t p) { delete p; --index_count; return CUVS_SUCCESS; }
cuvsError_t cuvsCagraBuild(cuvsResources_t,cuvsCagraIndexParams_t p,DLManagedTensor* t,cuvsCagraIndex_t idx) {
  assert(p->graph_degree==128 && p->intermediate_graph_degree==256 && p->nn_descent_niter==100);
  if (!benchmark_mode) assert(t->manager_ctx==nullptr && t->deleter==nullptr);
  idx->n=t->dl_tensor.shape[0]; idx->dim=t->dl_tensor.shape[1]; return fail_build?CUVS_ERROR:CUVS_SUCCESS;
}
cuvsError_t cuvsCagraSearch(cuvsResources_t,cuvsCagraSearchParams_t p,cuvsCagraIndex_t idx,DLManagedTensor* q,DLManagedTensor* ids,DLManagedTensor* d,cuvsFilter) {
  assert(in_search.fetch_add(1)==0); std::this_thread::yield();
  if (!benchmark_mode) assert(q->manager_ctx==nullptr && q->deleter==nullptr);
  last_itopk=p->itopk_size; assert(p->search_width==4);
  for(int64_t row=0; row<q->dl_tensor.shape[0]; ++row) {
    for(int64_t k=0; k<ids->dl_tensor.shape[1]; ++k) {
      auto pos=row*ids->dl_tensor.shape[1]+k;
      static_cast<unsigned*>(ids->dl_tensor.data)[pos]=static_cast<unsigned>((row+k)%idx->n);
      static_cast<float*>(d->dl_tensor.data)[pos]=static_cast<float>(k);
    }
  }
  --in_search; return fail_search?CUVS_ERROR:CUVS_SUCCESS;
}
cuvsError_t cuvsCagraSerialize(cuvsResources_t,const char* path,cuvsCagraIndex_t idx,bool include) {
  assert(include); FILE* f=std::fopen(path,"wb"); if(!f) return CUVS_ERROR;
  bool ok=std::fwrite(idx,sizeof(*idx),1,f)==1; std::fclose(f); return ok?CUVS_SUCCESS:CUVS_ERROR;
}
cuvsError_t cuvsCagraDeserialize(cuvsResources_t,const char* path,cuvsCagraIndex_t idx) {
  if(fail_load) return CUVS_ERROR;
  FILE* f=std::fopen(path,"rb"); if(!f) return CUVS_ERROR;
  bool ok=std::fread(idx,sizeof(*idx),1,f)==1;std::fclose(f);return ok?CUVS_SUCCESS:CUVS_ERROR;
}
cuvsError_t cuvsCagraIndexGetSize(cuvsCagraIndex_t idx,int64_t* n) { *n=idx->n;return CUVS_SUCCESS; }
cuvsError_t cuvsCagraIndexGetDims(cuvsCagraIndex_t idx,int64_t* n) { *n=idx->dim;return CUVS_SUCCESS; }
cudaError_t cudaMemcpy(void* to,const void* from,size_t n,cudaMemcpyKind) {
  if(fail_copy_after==0) { fail_copy_after=-1;return cudaErrorUnknown; }
  if(fail_copy_after>0) --fail_copy_after;
  std::memcpy(to,from,n);return cudaSuccess;
}
}
int main(int argc,char** argv) {
  assert(argc==2); std::vector<float> base(1024*32), queries(8*32);
  if (std::strcmp(argv[1], "--benchmark")==0) {
    benchmark_mode=true;
    void* h=seekdb_cuvs_build(base.data(),1024,32);assert(h);
    unsigned ids[100];float dist[100];
    const auto start=std::chrono::steady_clock::now();
    for(int i=0;i<4096;++i) assert(seekdb_cuvs_search(h,queries.data(),1,100,ids,dist)==0);
    const double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    seekdb_cuvs_free(h);balanced();
    std::printf("{\"queries\":4096,\"mock_alloc_calls\":%zu,\"peak_mock_bytes\":%zu,\"host_mock_seconds\":%.9f}\n",alloc_calls,peak_bytes,sec);
    return 0;
  }
  assert(!seekdb_cuvs_build(nullptr,1024,32));
  assert(!seekdb_cuvs_build(base.data(),0,32));
  assert(!seekdb_cuvs_build(base.data(),1024,0));
  assert(!seekdb_cuvs_build(base.data(),LONG_MAX,32));
  fail_build=true; assert(!seekdb_cuvs_build(base.data(),1024,32));fail_build=false;balanced();
  void* h=seekdb_cuvs_build(base.data(),1024,32); assert(h);
  unsigned ids[2048];float dist[2048];
  assert(seekdb_cuvs_search(h,queries.data(),0,10,ids,dist)!=0);
  assert(seekdb_cuvs_search(h,queries.data(),1,1025,ids,dist)!=0);
  assert(seekdb_cuvs_search(h,queries.data(),LONG_MAX,100,ids,dist)!=0);
  // Every allocation failure must leave a retryable handle and exact free sizes.
  for(int fail=0;fail<3;++fail) {
    void* fresh=seekdb_cuvs_build(base.data(),1024,32);assert(fresh);
    fail_alloc_after=fail; assert(seekdb_cuvs_search(fresh,queries.data(),1,10,ids,dist)!=0);
    fail_alloc_after=-1;assert(seekdb_cuvs_search(fresh,queries.data(),1,10,ids,dist)==0);
    seekdb_cuvs_free(fresh);
  }
  assert(seekdb_cuvs_search(h,queries.data(),1,10,ids,dist)==0);assert(ids[0]==0 && dist[9]==9);
  size_t warm_allocs=alloc_calls;
  for(int i=0;i<100;++i) assert(seekdb_cuvs_search(h,queries.data(),1,10,ids,dist)==0);
  assert(alloc_calls==warm_allocs); // Actual allocation elimination on warm path.
  assert(seekdb_cuvs_search(h,queries.data(),8,200,ids,dist)==0); assert(last_itopk==800);
  assert(seekdb_cuvs_search(h,queries.data(),1,10,ids,dist)==0); assert(last_itopk==512);
  for(int copy=0;copy<3;++copy) {
    fail_copy_after=copy;assert(seekdb_cuvs_search(h,queries.data(),1,10,ids,dist)!=0);
    assert(seekdb_cuvs_search(h,queries.data(),1,10,ids,dist)==0);
  }
  fail_sync=true; assert(seekdb_cuvs_search(h,queries.data(),1,10,ids,dist)!=0);fail_sync=false;
  fail_search=true; assert(seekdb_cuvs_search(h,queries.data(),1,10,ids,dist)!=0);fail_search=false;
  std::vector<std::thread> threads;
  for(int i=0;i<4;++i) threads.emplace_back([&]{unsigned a[10];float b[10];for(int j=0;j<100;++j) assert(seekdb_cuvs_search(h,queries.data(),1,10,a,b)==0);});
  for(auto& t:threads) t.join();
  {
    std::vector<float> big_queries(40000*32), big_dist(40000);
    std::vector<unsigned> big_ids(40000);
    assert(seekdb_cuvs_search(h,big_queries.data(),40000,1,big_ids.data(),big_dist.data())==0);
    assert(allocations.empty()); // A large one-off request must not pin VRAM.
    assert(seekdb_cuvs_search(h,queries.data(),1,10,ids,dist)==0);
  }
  assert(seekdb_cuvs_serialize(h,argv[1])==0);seekdb_cuvs_free(h);balanced();
  assert(!seekdb_cuvs_deserialize(argv[1],1023,32));balanced();
  fail_load=true;assert(!seekdb_cuvs_deserialize(argv[1],1024,32));fail_load=false;balanced();
  h=seekdb_cuvs_deserialize(argv[1],1024,32);assert(h);
  assert(seekdb_cuvs_search(h,queries.data(),1,10,ids,nullptr)==0);
  seekdb_cuvs_free(h);seekdb_cuvs_free(nullptr);balanced();
  std::printf("PASS host bridge: validation, reuse, growth, retries, copy/sync errors, concurrency, reload; peak_mock_bytes=%zu\n",peak_bytes);
}
