# INT32 remainder probes (`i32.rem`)

Evidence for the two `i32.rem` entries in
`../../simd_simt/david_v100_simd_simt_v1.json`.  `arith.remsi` is the one
integer operation whose lowering cost is far from the generic issue estimate,
so it is priced explicitly.  Both sides are measured as an **increment over a
baseline mode that differs only by the remainder**, which keeps loop, address
and read-back work out of the number.

```bash
bash -lc './build_and_run.sh rem_simd rem_simt'   # builds both, runs mode 0
./rem_simd_host <mode>    # mode 0..4
./rem_simt_host <mode>    # mode 0..1
```

The mode is a kernel argument, so one object serves all modes.

## SIMD - `rem_simd.cce`

Triton lowers `arith.remsi` to one always-inline `vmod_int32_t` call per
64-lane chunk.  All modes run the same loop on one s32 register:

| mode | body | passive live vectors |
|---|---|---|
| 0 | `vadd` x4 | 0 |
| 1 | `vadd` x4 + one `vmod` by 7 | 0 |
| 2 / 3 / 4 | same as 1 | 2 / 4 / 6 |

A passive vector is loaded before the loop and stored after it, never touched
inside.  The Triton SIMD lowering of `lane % 7` in `masked_gather` keeps six
64-lane vectors alive across its `vmod` call, so modes 2-4 ask whether that
slows the call down.

Measured (`cyc_per_step`, one AIV):

| mode | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| cycles | 1.8188 | 35.7692 | 35.7771 | 35.7745 | 35.7722 |

One `vmod` therefore costs **35.7692 - 1.8188 = 33.95 SYS_CNT cycles**, and two
to six passive live vectors change it by under 0.03%.  Against
`simd.f32.add.throughput` (3.3 vector instructions per cycle, i.e. 0.3030 cycle
per vadd) that is **112x**, the factor in the profile.

**Not modeled**: several `vmod` chains active at the same time cost more per
call (75 and 166 cycles per step for 2 and 4 chains).

## SIMT - `rem_simt.cce`

Mirrors the SIMT lowering of `(lane % 7) != 0` in `masked_gather`: the dividend
is a masked, shifted index xor a per-lane constant, and the remainder is only
compared with zero, so the backend may use a divisibility test rather than a
full remainder.

| mode | term |
|---|---|
| 0 | `x != 0` |
| 1 | `(x % 7) != 0` |

Mode 1 minus mode 0 is the price of one remainder-by-7 tested for zero.  SIMT
timings on this target are code-layout sensitive, so the probe is built under
seven alignment settings and the difference is taken within a layout
(`cyc_per_op`, cycles per element):

| layout | L0 | Laf64 | Laf128 | Laf256 | Lal64 | Lal128 | Lal256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mode 0 | 0.035521 | 0.035522 | 0.035523 | 0.035522 | 0.035521 | 0.035521 | 0.035523 |
| mode 1 | 0.041899 | 0.042086 | 0.042091 | 0.042092 | 0.041959 | 0.041962 | 0.041960 |
| increment | 0.006378 | 0.006564 | 0.006568 | 0.006570 | 0.006438 | 0.006441 | 0.006437 |

Against `simt.f32.add.throughput` (141 scalar ops per cycle, i.e. 0.007092
cycle per element) the increment is **0.90 to 0.93**; the profile uses **0.9**.
Cheaper than an integer add, consistent with a multiply-and-compare
divisibility test.

**Not covered**: a remainder whose value is used, or a runtime divisor.

Build flags for the layout sweep, on top of the `ccec` line in
`build_commands.md`: `-falign-functions=64|128|256`, `-falign-loops=64|128|256`.
