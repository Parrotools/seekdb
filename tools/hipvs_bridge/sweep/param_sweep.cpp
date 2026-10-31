// Bridge-level CAGRA parameter sweep on the REAL hipVS/cuVS implementation.
// Host tests exercise control flow only; recall and throughput need a GPU.
// One invocation measures ONE configuration (given via OBCUVS_* environment
// variables, the same overrides seekdb_cuvs_bridge.c reads) and prints one
// CSV row: algo,gd,igd,niter,itopk_min,sw,build_s,qps10,qps100,recall10,recall100
//
// usage: param_sweep <base.fvec> <queries.fvec> [truth.ivec] [nq_limit]
// Ground truth (top-100 L2) is brute-forced on CPU threads and cached to
// <queries>.truth.ivec when no truth file is supplied.
#include "../seekdb_cuvs_bridge.h"
#include <pthread.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace {

struct Matrix {
  std::vector<float> data;
  long rows = 0, cols = 0;
};

// fvec format: [int32 dim][dim x float32] repeated.
bool load_fvec(const char* path, Matrix* out) {
  FILE* f = std::fopen(path, "rb");
  if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return false; }
  int32_t dim = 0;
  if (std::fread(&dim, sizeof(dim), 1, f) != 1 || dim <= 0 || dim > 4096) { std::fclose(f); return false; }
  out->cols = dim;
  out->rows = 0;
  while (true) {
    int32_t next = 0;
    if (std::fread(&next, sizeof(next), 1, f) != 1) break;
    if (next != dim) { std::fclose(f); return false; }
    out->data.resize(out->data.size() + static_cast<size_t>(dim));
    if (std::fread(out->data.data() + static_cast<size_t>(out->rows) * dim, sizeof(float),
                   static_cast<size_t>(dim), f) != static_cast<size_t>(dim)) { std::fclose(f); return false; }
    ++out->rows;
  }
  std::fclose(f);
  return out->rows > 0;
}

// ivec format: [int32 dim][dim x int32] repeated.
bool load_ivec(const char* path, std::vector<int32_t>* out, long* rows, int32_t* cols) {
  FILE* f = std::fopen(path, "rb");
  if (!f) return false;
  int32_t dim = 0;
  if (std::fread(&dim, sizeof(dim), 1, f) != 1 || dim <= 0) { std::fclose(f); return false; }
  *cols = dim;
  *rows = 0;
  while (true) {
    int32_t next = 0;
    if (std::fread(&next, sizeof(next), 1, f) != 1) break;
    if (next != dim) { std::fclose(f); return false; }
    out->resize(out->size() + static_cast<size_t>(dim));
    if (std::fread(out->data() + static_cast<size_t>(*rows) * dim, sizeof(int32_t),
                   static_cast<size_t>(dim), f) != static_cast<size_t>(dim)) { std::fclose(f); return false; }
    ++*rows;
  }
  std::fclose(f);
  return *rows > 0;
}

void store_ivec(const char* path, const std::vector<int32_t>& ids, long rows, int32_t cols) {
  FILE* f = std::fopen(path, "wb");
  if (!f) return;
  for (long r = 0; r < rows; ++r) {
    std::fwrite(&cols, sizeof(cols), 1, f);
    std::fwrite(ids.data() + static_cast<size_t>(r) * cols, sizeof(int32_t), static_cast<size_t>(cols), f);
  }
  std::fclose(f);
}

struct BruteJob {
  const Matrix* base;
  const Matrix* queries;
  std::vector<int32_t>* truth;
  int32_t k;
  long begin, end;
};

void* brute_job(void* arg) {
  BruteJob* j = static_cast<BruteJob*>(arg);
  const long d = j->base->cols;
  std::vector<std::pair<float, int32_t>> best;
  best.reserve(static_cast<size_t>(j->k) + 1);
  for (long q = j->begin; q < j->end; ++q) {
    const float* query = j->queries->data.data() + static_cast<size_t>(q) * d;
    best.clear();
    float worst = -1.0f;
    for (long r = 0; r < j->base->rows; ++r) {
      const float* row = j->base->data.data() + static_cast<size_t>(r) * d;
      float sum = 0.0f;
      for (long c = 0; c < d; ++c) { const float diff = query[c] - row[c]; sum += diff * diff; }
      if (static_cast<long>(best.size()) < j->k) {
        best.emplace_back(sum, static_cast<int32_t>(r));
        if (sum > worst) worst = sum;
      } else if (sum < worst) {
        auto it = std::max_element(best.begin(), best.end());
        *it = {sum, static_cast<int32_t>(r)};
        worst = it->first;
      }
    }
    std::sort(best.begin(), best.end());
    for (int32_t i = 0; i < j->k; ++i)
      (*j->truth)[static_cast<size_t>(q) * j->k + i] = best[static_cast<size_t>(i)].second;
  }
  return nullptr;
}

double steady_seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

size_t env_size(const char* name) {
  const char* raw = std::getenv(name);
  return raw ? std::strtoull(raw, nullptr, 10) : 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: param_sweep <base.fvec> <queries.fvec> [truth.ivec] [nq_limit]\n");
    return 2;
  }
  Matrix base, queries;
  if (!load_fvec(argv[1], &base) || !load_fvec(argv[2], &queries)) return 2;
  if (queries.cols != base.cols) { std::fprintf(stderr, "dim mismatch\n"); return 2; }
  long nq = queries.rows;
  if (argc >= 5) nq = std::min(nq, strtol(argv[4], nullptr, 10));
  const int32_t k_truth = 100;
  if (nq < k_truth) { std::fprintf(stderr, "nq < %d\n", k_truth); return 2; }

  std::vector<int32_t> truth(static_cast<size_t>(nq) * k_truth);
  long truth_rows = 0;
  int32_t truth_cols = 0;
  const bool cached = argc >= 4 && load_ivec(argv[3], &truth, &truth_rows, &truth_cols);
  if (!cached || truth_rows < nq || truth_cols != k_truth) {
    const int threads = 32;
    std::vector<pthread_t> tids(static_cast<size_t>(threads));
    std::vector<BruteJob> jobs(static_cast<size_t>(threads));
    const long chunk = (nq + threads - 1) / threads;
    for (int t = 0; t < threads; ++t) {
      jobs[static_cast<size_t>(t)] = {&base, &queries, &truth, k_truth,
                                      std::min<long>(t * chunk, nq), std::min<long>((t + 1) * chunk, nq)};
      pthread_create(&tids[static_cast<size_t>(t)], nullptr, brute_job, &jobs[static_cast<size_t>(t)]);
    }
    for (int t = 0; t < threads; ++t) pthread_join(tids[static_cast<size_t>(t)], nullptr);
    if (argc >= 4) store_ivec(argv[3], truth, nq, k_truth);
  }
  // Recall compares sets, so give every truth row a sorted ID order for
  // binary_search regardless of whether it was brute-forced or loaded.
  for (long q = 0; q < nq; ++q) {
    std::sort(truth.begin() + static_cast<size_t>(q) * k_truth,
              truth.begin() + static_cast<size_t>(q + 1) * k_truth);
  }

  const double t0 = steady_seconds();
  void* index = seekdb_cuvs_build(base.data.data(), base.rows, base.cols);
  const double build_s = index ? steady_seconds() - t0 : -1.0;
  if (!index) { std::printf("build failed\n"); return 1; }

  double qps[2] = {0.0, 0.0};
  double recall[2] = {0.0, 0.0};
  std::vector<unsigned> ids(static_cast<size_t>(k_truth));
  std::vector<float> dist(static_cast<size_t>(k_truth));
  for (int mode = 0; mode < 2; ++mode) {
    const long topk = mode == 0 ? 10 : 100;
    double hits = 0.0;
    const double s0 = steady_seconds();
    for (long q = 0; q < nq; ++q) {
      if (seekdb_cuvs_search(index, queries.data.data() + static_cast<size_t>(q) * queries.cols,
                             1, topk, ids.data(), dist.data()) != 0) {
        std::fprintf(stderr, "search failed at q=%ld\n", q);
        seekdb_cuvs_free(index);
        return 1;
      }
      std::sort(ids.begin(), ids.begin() + topk);
      for (long i = 0; i < topk; ++i) {
        if (std::binary_search(truth.begin() + static_cast<size_t>(q) * k_truth,
                               truth.begin() + static_cast<size_t>(q) * k_truth + topk,
                               static_cast<int32_t>(ids[static_cast<size_t>(i)]))) hits += 1.0;
      }
    }
    const double seconds = steady_seconds() - s0;
    qps[mode] = seconds > 0 ? static_cast<double>(nq) / seconds : 0.0;
    recall[mode] = hits / (static_cast<double>(nq) * static_cast<double>(topk));
  }
  seekdb_cuvs_free(index);
  std::printf("%zu,%zu,%zu,%zu,%zu,%zu,%.3f,%.1f,%.1f,%.6f,%.6f\n",
              env_size("OBCUVS_BUILD_ALGO"), env_size("OBCUVS_GRAPH_DEGREE"),
              env_size("OBCUVS_INTERMEDIATE_DEGREE"), env_size("OBCUVS_NNDESCENT_NITER"),
              env_size("OBCUVS_ITOPK_MIN"), env_size("OBCUVS_SEARCH_WIDTH"),
              build_s, qps[0], qps[1], recall[0], recall[1]);
  return 0;
}
