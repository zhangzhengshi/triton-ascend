// Host for rem_simt.cce. Usage: host <mode>; reads rem_simt.o from the current
// directory. Gates are evaluated here and printed on GATE.
//   mode 0  terms are `x != 0`
//   mode 1  terms are `(x % 7) != 0`
// cyc_per_op of mode 1 minus mode 0 is the cost of one remainder-by-7 tested
// for zero, per element.
#include "runtime/runtime/rt.h"
#include <acl/acl.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <vector>
using namespace std;

static const int NT = 1024;                 // 32 warps x 32 lanes
static const uint32_t SENTINEL = 0xA5A5A5A5u;
static const int I1 = 400, I2 = 1200, K = 20;   // tput.cce's iteration counts
static const int OPS_PER_ITER = 8;

// The kernel takes the mode in the slot the other probes use for t1.
struct Args { void *out; void *gm; int K; int iters; uint32_t mode; uint32_t unused1; int unused2; };

// Mirrors rem_vf in rem_simt.cce.
static uint32_t remHash(int lane, int warp, int it, bool withRem) {
  uint32_t b = (uint32_t)(lane + warp + 1);
  int32_t base = (int32_t)(b * 37u);
  uint32_t h = (uint32_t)base;
  for (int i = 0; i < it; i++) {
    uint32_t s = 0;
    for (int j = 0; j < 8; j++) {
      int32_t x = (int32_t)((((uint32_t)(i + j) & 31u) << 2) ^ (uint32_t)((base + 101 * j) & 1023));
      s += withRem ? (uint32_t)((x % 7) != 0) : (uint32_t)(x != 0);
    }
    h = h * 3u + s;
  }
  return h;
}

static int RTERR = 0, OUTSENT = 0;
static const char *FN_MEASURE = "measure";
static const char *FN_VERIFY = "verify";

static long long runK(const char *fn, rtStream_t s, void *dout, void *gm, int k, int iters, uint32_t mode) {
  long long cyc = -1;
  int e0 = rtMemcpy(dout, 8, &cyc, 8, RT_MEMCPY_HOST_TO_DEVICE);
  Args a{dout, gm, k, iters, mode, 0, 0};
  rtArgsEx_t ai = {}; ai.args = &a; ai.argsSize = sizeof(a);
  rtTaskCfgInfo_t c = {}; c.localMemorySize = 192 * 1024;
  int e1 = rtKernelLaunchWithFlagV2((void *)fn, 1, &ai, 0, s, 0, &c);
  int e2 = rtStreamSynchronize(s);
  int e3 = rtMemcpy(&cyc, 8, dout, 8, RT_MEMCPY_DEVICE_TO_HOST);
  if (e0 || e1 || e2 || e3) { RTERR++; if (RTERR <= 2) printf("  RTERR h2d=%d launch=%d sync=%d d2h=%d\n", e0, e1, e2, e3); }
  if (cyc < 0) OUTSENT++;
  return cyc;
}

int main(int argc, char **argv) {
  int md = argc > 1 ? atoi(argv[1]) : 0;
  if (md != 0 && md != 1) { printf("mode must be 0 or 1\n"); return 2; }
  aclInit(0); rtSetDevice(0);
  ifstream f("rem_simt.o", ios::binary); f.seekg(0, ios::end); size_t n = f.tellg(); f.seekg(0);
  vector<char> buf(n); f.read(buf.data(), n);
  rtDevBinary_t bin; bin.data = buf.data(); bin.length = n; bin.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC; bin.version = 0;
  void *h = 0; rtDevBinaryRegister(&bin, &h);
  rtFunctionRegister(h, FN_MEASURE, FN_MEASURE, (void *)FN_MEASURE, 0);
  rtFunctionRegister(h, FN_VERIFY, FN_VERIFY, (void *)FN_VERIFY, 0);
  rtStream_t s; rtStreamCreate(&s, 0);
  void *dout, *gm; rtMalloc(&dout, 72, RT_MEMORY_HBM, 0); rtMalloc(&gm, NT * 4, RT_MEMORY_HBM, 0);

  int Im = (I1 + I2) / 2;
  runK(FN_MEASURE, s, dout, gm, 2, I1, (uint32_t)md); // warmup
  long long c1 = LLONG_MAX, cm = LLONG_MAX, c2 = LLONG_MAX;
  for (int r = 0; r < 7; r++) { // interleaved so drift does not bias the slope
    c1 = min(c1, runK(FN_MEASURE, s, dout, gm, K, I1, (uint32_t)md));
    cm = min(cm, runK(FN_MEASURE, s, dout, gm, K, Im, (uint32_t)md));
    c2 = min(c2, runK(FN_MEASURE, s, dout, gm, K, I2, (uint32_t)md));
  }
  double cpi = (double)(c2 - c1) / ((double)(I2 - I1) * K); // cycles per iteration, all threads
  double cpo = cpi / (NT * OPS_PER_ITER);
  double lin = (c2 > c1) ? ((double)cm - 0.5 * (c1 + c2)) / (double)(c2 - c1) : 1e9;

  vector<uint32_t> R(NT, SENTINEL);
  rtMemcpy(gm, NT * 4, R.data(), NT * 4, RT_MEMCPY_HOST_TO_DEVICE);
  runK(FN_VERIFY, s, dout, gm, 1, I2, (uint32_t)md);
  rtMemcpy(R.data(), NT * 4, gm, NT * 4, RT_MEMCPY_DEVICE_TO_HOST);
  int mismatch = 0, sentinel = 0; set<uint32_t> distinct;
  for (int w = 0; w < 32; w++) for (int l = 0; l < 32; l++) {
    uint32_t got = R[w * 32 + l], want = remHash(l, w, I2, md == 1);
    if (got != want) mismatch++;
    if (got == SENTINEL && want != SENTINEL) sentinel++;
    distinct.insert(got);
  }

  printf("RESULT mode=%d I1=%d Im=%d I2=%d K=%d c1=%lld cm=%lld c2=%lld "
         "cyc_per_iter=%.3f ops_per_iter=%d cyc_per_op=%.6f ops_per_cyc=%.2f lin=%.4f "
         "mismatch=%d sentinel=%d distinct=%zu rterr=%d outsent=%d\n",
         md, I1, Im, I2, K, c1, cm, c2, cpi, OPS_PER_ITER, cpo, 1.0 / cpo, lin,
         mismatch, sentinel, distinct.size(), RTERR, OUTSENT);
  printf("READBACK got=");
  for (int i = 0; i < 4; i++) printf("%08x%s", R[i], i < 3 ? "," : "");
  printf(" want=");
  for (int i = 0; i < 4; i++) printf("%08x%s", remHash(i, 0, I2, md == 1), i < 3 ? "," : "");
  printf("\n");
  printf("GATE runtime=%s readback=%s nontrivial=%s linear=%s\n",
         (RTERR == 0 && OUTSENT == 0) ? "PASS" : "FAIL",
         (mismatch == 0 && sentinel == 0) ? "PASS" : "FAIL",
         distinct.size() >= 32 ? "PASS" : "FAIL",
         (lin > -0.03 && lin < 0.03) ? "PASS" : "FAIL");
  rtFree(dout); rtFree(gm); rtStreamDestroy(s); rtDeviceReset(0); aclFinalize();
  return 0;
}
