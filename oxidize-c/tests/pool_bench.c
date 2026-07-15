#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../src/tensor.h"

static void touch(void* ctx, size_t begin, size_t end) {
  float* values = ctx;
  for (size_t i = begin; i < end; ++i) values[i] += 1.0f;
}

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int physical_cores(void) {
  cpu_set_t allowed;
  if (sched_getaffinity(0, sizeof allowed, &allowed) != 0) return 0;
  int packages[CPU_SETSIZE], cores[CPU_SETSIZE], count = 0, topology_cpus = 0;
  int available = CPU_COUNT(&allowed);
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &allowed)) continue;
    char path[128];
    int package = -1, core = -1;
    snprintf(path, sizeof path,
             "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
    FILE* f = fopen(path, "r");
    if (f) {
      if (fscanf(f, "%d", &package) != 1) package = -1;
      fclose(f);
    }
    snprintf(path, sizeof path,
             "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
    f = fopen(path, "r");
    if (f) {
      if (fscanf(f, "%d", &core) != 1) core = -1;
      fclose(f);
    }
    if (package < 0 || core < 0) continue;
    ++topology_cpus;
    int seen = 0;
    for (int i = 0; i < count; ++i)
      if (packages[i] == package && cores[i] == core) seen = 1;
    if (!seen) {
      packages[count] = package;
      cores[count] = core;
      ++count;
    }
  }
  return topology_cpus == available ? count : 0;
}

int main(int argc, char** argv) {
  int threads = argc > 1 ? atoi(argv[1]) : 48;
  int iterations = argc > 2 ? atoi(argv[2]) : 400;
  size_t rows = argc > 3 ? (size_t)strtoull(argv[3], NULL, 10) : 48;
  float* values = calloc(rows, sizeof(*values));
  if (!values) return 2;

  int expected_physical = physical_cores();
  oc_pool_init(threads);
  int actual_threads = oc_pool_size();
  oc_numa_replicate(values, rows * sizeof(*values));
  for (int i = 0; i < 10; ++i) oc_parallel_for(rows, touch, values);
  double start = now_seconds();
  for (int i = 0; i < iterations; ++i) oc_parallel_for(rows, touch, values);
  double elapsed = now_seconds() - start;
  oc_pool_free();

  double dispatch_us = elapsed * 1e6 / (double)iterations;
  printf("requested=%d actual=%d rows=%zu iterations=%d dispatch=%.2f us total=%.3f s\n",
         threads, actual_threads, rows, iterations, dispatch_us, elapsed);
  free(values);
  if (threads <= 0 && expected_physical > 0 && actual_threads != expected_physical) return 1;
  return 0;
}
