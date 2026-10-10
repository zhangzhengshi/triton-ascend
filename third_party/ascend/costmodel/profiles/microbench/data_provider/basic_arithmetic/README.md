# FP32 ADD/SUB/MUL/MAX CCE probes

Fixed configuration: eight independent chains, one target per chain per runtime
iteration, `K=20`, SIMD vector width 64, SIMT 32 requested warps (1024 threads).
Iteration points are 200/400/600 for SIMD and 400/800/1200 for SIMT.

## Adopted hardware measurements

These are the accepted fixed-low-unroll records from the 2026-10-09 v33 campaign,
not a new measurement of this consolidated source. Sources:
`low_unroll_calibration_results.json` (SIMD eight-chain rows) and
`low_unroll_profile_update.json` (SIMT arithmetic costs) in the original
calibration workspace. Its `throughput_revalidation.md` records acceptance.

| op | SIMD tick/vector64 | SIMD element-op/tick | SIMT tick/scalar-op | SIMT scalar-op/tick |
|---|---:|---:|---:|---:|
| add | 0.303016 | 211.209969 | 0.009542816162109 | 104.790869 |
| sub | 0.302922 | 211.275510 | 0.009767997741699 | 102.375126 |
| mul | 0.303094 | 211.155615 | 0.009836799621582 | 101.659080 |
| max | 0.302953 | 211.253891 | 0.008876686096191 | 112.654654 |

The SIMD ADD/SUB raw campaign was
`/tmp/low_unroll_simd_addsub_npu6_20261009/results.json`; MUL/MAX were
`/tmp/low_unroll_simd_targets_npu6_20261009/results.json`. SIMT records are the
32-warp, eight-chain, unroll-one entries in the accepted results JSON.
These describe effective throughput of the fixed loop, including its control
cost; they are neither front-end issue rates nor whole-kernel latency.

The route profile stores SUB/MUL/MAX as absolute rates. ADD retains the shared
raw references (SIMD 3.3 vector/tick, SIMT 141 scalar-op/tick) with
`factor = shared_reference * measured_tick_per_target`. Its effective rate is
`shared_reference / factor`. The current loader inherits only the raw throughput
for `relative_to`, so this ADD correction leaves all other operation costs
unchanged. No shared microbenchmark catalog entry is modified.

## Reproduction

The four files consolidate only these arithmetic loops and runtime plumbing;
they retain one target per chain per iteration and the campaign inputs. The SIMD
recurrence continues across K calls; SIMT reloads the base each call and returns
the left-associated sum of its eight chains. Compilation of this consolidation
produces a new ELF, so the old instruction audits are not audits of the new ELF.

Inspect `npu-smi info` and select an idle physical device before running:

```sh
python3 run.py --template-root /path/to/AscendNPU-IR/ascend/lib/Templates \
  --route simt --op sub --device 6 --output /tmp/basic-simt-sub
```

`--route` accepts `simd`/`simt`; `--op` accepts `add`/`sub`/`mul`/`max`.
`--compile-only` builds without launching hardware. Keep a fresh output directory
for every configuration; the script records commands, build output and samples.

Initialization and output readback are outside the device SYS_CNT interval. Each
work call completes with `pipe_barrier(PIPE_ALL)` before the next call or end
counter. The host checks FP32 outputs against CPU recurrences, warms all three
points, then collects seven interleaved batches and reports every sample.
For arithmetic-mean ticks `T(N)`:

```
increment_ticks = (T(N_high) - T(N_low)) / (K * (N_high - N_low))
midpoint_residual = T(N_mid) - (T(N_low) + T(N_high)) / 2
SIMD tick/vector = increment_ticks / 8
SIMT tick/scalar = increment_ticks / (8 * 1024)
element_ops/tick = (8 * active_lanes) / increment_ticks
```

Review midpoint linearity and repeat stability, plus the generated instruction
counts, before adopting new results. The runner never rewrites a profile. This
commit adopts existing hardware data and does not claim new Triton validation.
