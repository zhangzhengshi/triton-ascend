// Host for rem_simd.cce: GM seeds, sentinels, return-code checks, interleaved
// three-point slope, and read-back of lanes 0..7 against a host simulation.
// Usage: host <mode>; reads rem_simd.o from the current directory.
#include "runtime/runtime/rt.h"
#include <acl/acl.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <vector>
using namespace std;
static const int GMN = 1024, W = 64;
static char *readBin(const char *f, uint32_t *sz) {
  ifstream s(f, ios::binary); s.seekg(0, ios::end); size_t n = s.tellg(); s.seekg(0);
  char *b = new char[n]; s.read(b, n); *sz = n; return b;
}
struct Args { void *out; void *gm; int K; int a; int b; int iters; int mode; };
static vector<int32_t> G(GMN), RG(GMN);
static int RTERR = 0, SENT = 0;
static void fill() {
  for (int j = 0; j < 4; j++) for (int i = 0; i < W; i++) G[j * W + i] = 1000 + 37 * i + 11 * j;
  for (int i = 0; i < W; i++) G[4 * W + i] = 3 + (i % 5);
  float zero = 0.0f, one = 1.0f; int32_t zb, ob; memcpy(&zb, &zero, 4); memcpy(&ob, &one, 4);
  for (int j = 0; j < 4; j++) for (int i = 0; i < W; i++) G[320 + j * W + i] = zb;
  for (int i = 0; i < W; i++) G[320 + 4 * W + i] = ob;
  for (int i = 0; i < W; i++) G[640 + i] = 0x7fffffff;
}
static long long runK(const char *fn, rtStream_t s, void *dout, void *gm, int K, int iters, int mode) {
  fill();
  long long sent = -1;
  int e0 = rtMemcpy(gm, GMN * 4, G.data(), GMN * 4, RT_MEMCPY_HOST_TO_DEVICE);
  int e0b = rtMemcpy(dout, 8, &sent, 8, RT_MEMCPY_HOST_TO_DEVICE);
  Args a{dout, gm, K, 0, 0, iters, mode};
  rtArgsEx_t ai = {}; ai.args = &a; ai.argsSize = sizeof(a);
  rtTaskCfgInfo_t c = {}; c.localMemorySize = 192 * 1024;
  int e1 = rtKernelLaunchWithFlagV2((void *)fn, 1, &ai, 0, s, 0, &c);
  int e2 = rtStreamSynchronize(s);
  long long cyc = -1;
  int e3 = rtMemcpy(&cyc, 8, dout, 8, RT_MEMCPY_DEVICE_TO_HOST);
  int e4 = rtMemcpy(RG.data(), GMN * 4, gm, GMN * 4, RT_MEMCPY_DEVICE_TO_HOST);
  if (e0 || e0b || e1 || e2 || e3 || e4) { RTERR++; if (RTERR <= 2) printf("  RTERR h2d=%d/%d launch=%d sync=%d d2h=%d/%d\n", e0, e0b, e1, e2, e3, e4); }
  if (cyc < 0) SENT++;
  return cyc;
}
// Lane i of a0 after K launches of `iters` iterations, 4 steps per iteration.
static int32_t sim(int mode, int K, int iters, int i) {
  int32_t a = 1000 + 37 * i, k = 3 + (i % 5);
  long steps = (long)K * iters * 4;
  for (long st = 0; st < steps; st++) {
    a = (int32_t)((uint32_t)a + (uint32_t)k);
    if (mode >= 1) a = a % 7;
  }
  return a;
}
int main(int argc, char **argv) {
  int md = argc > 1 ? atoi(argv[1]) : 0;
  aclInit(0); rtSetDevice(0);
  uint32_t sz; char *buf = readBin("rem_simd.o", &sz);
  rtDevBinary_t bin; bin.data = buf; bin.length = sz; bin.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC; bin.version = 0;
  void *h = 0; rtDevBinaryRegister(&bin, &h);
  const char *fn = "measure"; rtFunctionRegister(h, fn, fn, (void *)fn, 0);
  rtStream_t s; rtStreamCreate(&s, 0);
  void *dout, *gm; rtMalloc(&dout, 72, RT_MEMORY_HBM, 0); rtMalloc(&gm, GMN * 4, RT_MEMORY_HBM, 0);
  runK(fn, s, dout, gm, 4, 50, md);
  const int K = 20, I1 = 200, Im = 400, I2 = 600;
  long long c1 = LLONG_MAX, cm = LLONG_MAX, c2 = LLONG_MAX;
  for (int r = 0; r < 7; r++) { // interleaved so drift does not bias the slope
    c1 = min(c1, runK(fn, s, dout, gm, K, I1, md));
    cm = min(cm, runK(fn, s, dout, gm, K, Im, md));
    c2 = min(c2, runK(fn, s, dout, gm, K, I2, md));
  }
  double cps = (double)(c2 - c1) / ((double)(I2 - I1) * K * 4.0); // cycles per step
  double lin = c2 > c1 ? ((double)cm - 0.5 * (c1 + c2)) / (double)(c2 - c1) : 1e9;
  // Read-back after RK*RI*4 steps of a = (a + k) % 7 is (1000 + 37i + steps*k) mod 7.
  // With RI = 200 the step count is 5 mod 7, which makes lanes 0..7 collapse to
  // two values; RI = 201 gives 1 mod 7 and five distinct values.
  const int RK = 20, RI = 201;
  runK(fn, s, dout, gm, RK, RI, md);
  bool ok = true; set<int32_t> d;
  for (int i = 0; i < 8; i++) { if (sim(md, RK, RI, i) != RG[640 + i]) ok = false; d.insert(RG[640 + i]); }
  printf("RESULT mode=%d c1=%lld cm=%lld c2=%lld cyc_per_step=%.5f lin=%.4f distinct8=%zu rterr=%d sentinel_hits=%d readback=",
         md, c1, cm, c2, cps, lin, d.size(), RTERR, SENT);
  for (int i = 0; i < 8; i++) printf("%d%s", RG[640 + i], i < 7 ? "," : "");
  printf(" expect=");
  for (int i = 0; i < 8; i++) printf("%d%s", sim(md, RK, RI, i), i < 7 ? "," : "");
  printf("\n");
  printf("GATE runtime=%s readback=%s nontrivial=%s linear=%s\n",
         (RTERR == 0 && SENT == 0) ? "PASS" : "FAIL", ok ? "PASS" : "FAIL",
         d.size() >= 3 ? "PASS" : "FAIL", fabs(lin) < 0.03 ? "PASS" : "FAIL");
  rtFree(dout); rtFree(gm); rtStreamDestroy(s); rtDeviceReset(0); aclFinalize();
  return 0;
}
