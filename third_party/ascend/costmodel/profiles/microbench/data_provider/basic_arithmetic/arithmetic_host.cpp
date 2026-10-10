// Fixed arithmetic host: all samples, fixed mean, no profile writes/reset.
#include "runtime/runtime/rt.h"
#include <acl/acl.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#ifndef ARITH_ROUTE
#define ARITH_ROUTE 0
#endif
#ifndef ARITH_OP
#define ARITH_OP 0
#endif
static_assert(ARITH_ROUTE == 0 || ARITH_ROUTE == 1, "invalid route");
static_assert(ARITH_OP >= 0 && ARITH_OP <= 3, "invalid operation");
constexpr bool SUM_OUTPUT = ARITH_ROUTE == 1;
constexpr int NT = 1024, C = 8, ACTIVE = ARITH_ROUTE ? NT : 64, K = 20;
const char *const OPS[] = {"add", "sub", "mul", "max"};
void check(int rc, const char *where) {
  if (rc) {
    std::fprintf(stderr, "ERROR %s=%d\n", where, rc);
    std::exit(1);
  }
}
template <class T> uint32_t bits(T value) {
  static_assert(sizeof(T) == 4, "32-bit inputs only");
  uint32_t result;
  std::memcpy(&result, &value, 4);
  return result;
}
template <class T> T value(uint32_t word) {
  T result;
  std::memcpy(&result, &word, 4);
  return result;
}
std::vector<uint32_t> seeds() {
  std::vector<uint32_t> v(3 * C * ACTIVE, 0x7fc00000u);
  for (int c = 0; c < C; ++c)
    for (int t = 0; t < ACTIVE; ++t) {
      int i = c * ACTIVE + t;
      float x, d;
      if (ARITH_ROUTE) {
        x = (-2.f + 0.005f * t) + 0.01f * c;
        d = ARITH_OP == 0
                ? (t & 1 ? -1.f : 1.f) * (0.0125f + 0.0001f * (t % 11))
            : ARITH_OP == 1 ? 0.020f + 0.0001f * (t % 7)
            : ARITH_OP == 3 ? 0.5f + 0.001f * (t % 13)
                            : 1.0001f + 0.00001f * (t % 7);
      } else {
        x = (ARITH_OP == 3 ? -2.f : 1.f) + 0.015f * i;
        d = ARITH_OP == 0   ? 0.015f + 0.0001f * (t % 13)
            : ARITH_OP == 2 ? 1.000001f + 0.0000001f * (t % 7)
                            : 1.001f + 0.00001f * (t % 7);
      }
      v[i] = bits(x);
      v[C * ACTIVE + i] = bits(d);
    }
  return v;
}
int validate(const std::vector<uint32_t> &input,
             const std::vector<uint32_t> &got, int iterations) {
  int bad = 0;
  double max_relative = 0;
  for (int c = 0; c < (SUM_OUTPUT ? 1 : C); ++c)
    for (int t = 0; t < ACTIVE; ++t) {
      int i = c * ACTIVE + t;
      float expected = 0;
      for (int chain = 0; chain < (SUM_OUTPUT ? C : 1); ++chain) {
        int index = SUM_OUTPUT ? chain * ACTIVE + t : i;
        float x = value<float>(input[index]),
              d = value<float>(input[C * ACTIVE + index]);
        for (int r = 0; r < iterations * (ARITH_ROUTE ? 1 : K); ++r) {
          if (ARITH_OP == 0)
            x += d;
          else if (ARITH_OP == 1)
            x -= d;
          else if (ARITH_OP == 2)
            x *= d;
          else
            x = std::fmax(x, d);
        }
        if (chain == 0)
          expected = x;
        else
          expected += x;
      }
      float actual = value<float>(got[2 * C * ACTIVE + i]);
      double error = std::fabs(double(actual) - expected);
      double relative = error / std::fmax(std::fabs(double(expected)), 1.0);
      if (relative > max_relative)
        max_relative = relative;
      if (!std::isfinite(expected) || !std::isfinite(actual) ||
          error > 3e-5 + 3e-5 * std::fabs(double(expected)))
        ++bad;
    }
  std::printf("CHECK iterations=%d logical_elements=%d mismatches=%d "
              "max_scaled_error=%.12g\n",
              iterations, C * ACTIVE, bad, max_relative);
  return bad;
}
int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const char *device = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
  if (!device || !*device) {
    std::fprintf(stderr, "Explicit idle ASCEND_RT_VISIBLE_DEVICES required\n");
    return 2;
  }
  for (const char *p = device; *p; ++p)
    if (*p < '0' || *p > '9') {
      std::fprintf(stderr, "Select exactly one explicit physical device\n");
      return 2;
    }
  std::ifstream f("arithmetic.o", std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "Cannot read arithmetic.o\n");
    return 2;
  }
  f.seekg(0, std::ios::end);
  const auto size = f.tellg();
  if (size <= 0)
    return 2;
  std::vector<char> elf(static_cast<size_t>(size));
  f.seekg(0);
  f.read(elf.data(), elf.size());
  if (!f)
    return 2;
  check(aclInit(nullptr), "aclInit");
  check(rtSetDevice(0), "device0");
  rtDevBinary_t bin{};
  bin.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  bin.data = elf.data();
  bin.length = elf.size();
  void *handle = nullptr;
  check(rtDevBinaryRegister(&bin, &handle), "register");
  const char *name = "measure";
  check(rtFunctionRegister(handle, name, name, (void *)name, 0), "function");
  rtStream_t stream;
  check(rtStreamCreate(&stream, 0), "stream");
  auto input = seeds();
  const size_t bytes = input.size() * sizeof(uint32_t);
  void *gm = nullptr, *out = nullptr;
  check(rtMalloc(&gm, bytes, RT_MEMORY_HBM, 0), "gm");
  check(rtMalloc(&out, sizeof(uint64_t), RT_MEMORY_HBM, 0), "counter");
  std::vector<uint32_t> got(input.size());
  struct Args {
    void *out;
    void *gm;
    int K;
    int iterations;
  };
  static_assert(sizeof(Args) == 24, "CCE pointer,pointer,int,int ABI");
  auto run = [&](int iterations) {
    // Copies and result validation are outside the device's timed interval.
    check(rtMemcpy(gm, bytes, input.data(), bytes, RT_MEMCPY_HOST_TO_DEVICE),
          "seed");
    Args args{out, gm, K, iterations};
    rtArgsEx_t arg{};
    arg.args = &args;
    arg.argsSize = sizeof(args);
    rtTaskCfgInfo_t cfg{};
    cfg.localMemorySize = 192 * 1024;
    check(rtKernelLaunchWithFlagV2((void *)name, 1, &arg, nullptr, stream, 0,
                                   &cfg),
          "launch");
    check(rtStreamSynchronize(stream), "sync");
    uint64_t elapsed = 0;
    check(rtMemcpy(&elapsed, sizeof(elapsed), out, sizeof(elapsed),
                   RT_MEMCPY_DEVICE_TO_HOST),
          "ticks");
    check(rtMemcpy(got.data(), bytes, gm, bytes, RT_MEMCPY_DEVICE_TO_HOST),
          "result");
    if (!elapsed || validate(input, got, iterations))
      std::exit(3);
    return elapsed;
  };
  std::array<int, 3> points = ARITH_ROUTE ? std::array<int, 3>{400, 800, 1200}
                                          : std::array<int, 3>{200, 400, 600};
  std::printf("CONFIG physical=%s logical=0 route=%s op=%s NT=%d chains=%d "
              "active=%d K=%d batches=7 mean_only=1\n",
              device, ARITH_ROUTE ? "simt" : "simd", OPS[ARITH_OP], NT, C,
              ACTIVE, K);
  for (int j = 0; j < 3; ++j) {
    auto ticks = run(points[j]);
    std::printf("WARM point=%d ticks=%llu\n", points[j],
                (unsigned long long)ticks);
  }
  std::array<std::array<uint64_t, 3>, 7> samples{};
  for (int batch = 0; batch < 7; ++batch)
    for (int j = 0; j < 3; ++j) {
      int point = (batch + j) % 3;
      samples[batch][point] = run(points[point]);
      std::printf("RAW batch=%d order=%d point=%d ticks=%llu\n", batch, j,
                  points[point], (unsigned long long)samples[batch][point]);
    }
  std::array<double, 3> means{};
  for (int p = 0; p < 3; ++p) {
    for (int b = 0; b < 7; ++b)
      means[p] += samples[b][p] / 7.0;
    std::printf("MEAN point=%d ticks=%.12f\n", points[p], means[p]);
  }
  const double norm = double(K) * (points[2] - points[0]);
  double incremental = (means[2] - means[0]) / norm;
  double residual = means[1] - (means[0] + means[2]) / 2;
  std::printf("SUMMARY mean_increment_ticks_per_iteration=%.12f "
              "mean_midpoint_residual_ticks=%.12f denominator=%.0f "
              "elements_per_iteration=%d\n",
              incremental, residual, norm, C * ACTIVE);
  std::printf("RATE tick_per_target=%.12f logical_elements_per_tick=%.12f\n",
              incremental / (C * (ARITH_ROUTE ? ACTIVE : 1)),
              C * ACTIVE / incremental);
  std::puts(
      "SCOPE fixed low-unroll effective throughput includes loop/control "
      "and completion; SYS_CNT ticks, not frontend issue or kernel latency.");
  check(rtFree(gm), "freegm");
  check(rtFree(out), "freecounter");
  check(rtStreamDestroy(stream), "destroystream");
  check(aclFinalize(), "finalize");
  return 0;
}
