/* Clustered synthetic vector generator for the bridge-level parameter sweep.
 * Uniform random data makes ANN recall look artificially good, so this emits
 * a Gaussian mixture: recall deltas measured on it transfer to real datasets
 * better than on uniform noise. It never leaves the experiment machine.
 *
 * usage: gen_synthetic <out_prefix> <n> <dim> <nq> <clusters> <seed>
 * writes <out_prefix>.base.fvec and <out_prefix>.queries.fvec
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static uint64_t s_state;
static uint64_t next_u64(void) {
  s_state ^= s_state << 13;
  s_state ^= s_state >> 7;
  s_state ^= s_state << 17;
  return s_state;
}
static double next_uniform(void) { return (double)(next_u64() >> 11) / 9007199254740992.0; }
static double next_normal(void) {
  double u1 = next_uniform(), u2 = next_uniform();
  if (u1 < 1e-12) u1 = 1e-12;
  return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static int write_clustered_fvec(const char* path, long rows, int32_t dim, uint64_t seed,
                                const float* centroids, long clusters, double spread) {
  FILE* f = fopen(path, "wb");
  if (!f) return -1;
  s_state = seed ? seed : 0x9E3779B97F4A7C15ull;
  float* row = malloc(sizeof(float) * (size_t)dim);
  if (!row) { fclose(f); return -1; }
  for (long r = 0; r < rows; ++r) {
    const float* center = centroids + (size_t)(next_u64() % (uint64_t)clusters) * (size_t)dim;
    fwrite(&dim, sizeof(dim), 1, f);
    for (int32_t c = 0; c < dim; ++c) row[c] = center[c] + (float)(next_normal() * spread);
    fwrite(row, sizeof(float), (size_t)dim, f);
  }
  free(row);
  fclose(f);
  return 0;
}

int main(int argc, char** argv) {
  if (argc != 7) {
    fprintf(stderr, "usage: %s <out_prefix> <n> <dim> <nq> <clusters> <seed>\n", argv[0]);
    return 2;
  }
  const char* prefix = argv[1];
  const long n = strtol(argv[2], NULL, 10);
  const int32_t dim = (int32_t)strtol(argv[3], NULL, 10);
  const long nq = strtol(argv[4], NULL, 10);
  const long clusters = strtol(argv[5], NULL, 10);
  const uint64_t seed = strtoull(argv[6], NULL, 10);
  if (n <= 0 || dim <= 0 || nq <= 0 || clusters <= 0) return 2;

  s_state = seed ? seed : 0x9E3779B97F4A7C15ull;
  float* centroids = malloc(sizeof(float) * (size_t)clusters * (size_t)dim);
  if (!centroids) return 1;
  for (long i = 0; i < clusters * (long)dim; ++i) centroids[i] = (float)(next_normal() * 3.0);

  char path[4096];
  snprintf(path, sizeof(path), "%s.base.fvec", prefix);
  if (write_clustered_fvec(path, n, dim, seed + 1, centroids, clusters, 1.0) != 0) return 1;
  snprintf(path, sizeof(path), "%s.queries.fvec", prefix);
  if (write_clustered_fvec(path, nq, dim, seed + 2, centroids, clusters, 1.0) != 0) return 1;
  free(centroids);
  return 0;
}
