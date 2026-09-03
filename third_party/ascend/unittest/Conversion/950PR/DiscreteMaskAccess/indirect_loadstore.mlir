// RUN: triton-opt '--discrete-mask-access-conversion=compile-on-910-95=True compile-mode=simd_simt_template' --split-input-file %s | FileCheck %s

// CHECK-LABEL: tt.func @discrete_load
// CHECK: tt.load {{.*}} {MixCompileDiscreteMask} : tensor<1024x!tt.ptr<i32>>
tt.func @discrete_load(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
  %cst = arith.constant dense<0> : tensor<1024xi32>
  %cst_0 = arith.constant dense<200> : tensor<1024xi32>
  %cst_1 = arith.constant dense<400> : tensor<1024xi32>
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %1 = arith.cmpi slt, %0, %cst_0 : tensor<1024xi32>
  %2 = arith.cmpi sgt, %0, %cst_1 : tensor<1024xi32>
  %3 = arith.ori %1, %2 : tensor<1024xi1>
  %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %5 = tt.addptr %4, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %7 = tt.addptr %6, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %8 = tt.load %5, %3, %cst : tensor<1024x!tt.ptr<i32>>
  tt.store %7, %8 : tensor<1024x!tt.ptr<i32>>
  tt.return
}

// CHECK-LABEL: tt.func @discrete_store
// CHECK: tt.store {{.*}} {MixCompileDiscreteMask} : tensor<1024x!tt.ptr<i32>>
tt.func @discrete_store(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
  %cst = arith.constant dense<0> : tensor<1024xi32>
  %cst_0 = arith.constant dense<200> : tensor<1024xi32>
  %cst_1 = arith.constant dense<400> : tensor<1024xi32>
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %1 = arith.cmpi slt, %0, %cst_0 : tensor<1024xi32>
  %2 = arith.cmpi sgt, %0, %cst_1 : tensor<1024xi32>
  %3 = arith.ori %1, %2 : tensor<1024xi1>
  %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %5 = tt.addptr %4, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %7 = tt.addptr %6, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %8 = tt.load %5 : tensor<1024x!tt.ptr<i32>>
  tt.store %7, %8, %3 : tensor<1024x!tt.ptr<i32>>
  tt.return
}

// CHECK-LABEL: tt.func @and_mask_bad_order
// CHECK: tt.load {{.*}} {MixCompileDiscreteMask} : tensor<1024x!tt.ptr<i32>>
tt.func @and_mask_bad_order(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %n_elements: i32, %stride: i32) {
  %cst = arith.constant dense<0> : tensor<1024xi32>
  %czero = arith.constant 0 : i32
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %n = tt.splat %n_elements : i32 -> tensor<1024xi32>
  %in_bounds = arith.cmpi slt, %0, %n : tensor<1024xi32>
  %s = tt.splat %stride : i32 -> tensor<1024xi32>
  %r = arith.remsi %0, %s : tensor<1024xi32>
  %zero = tt.splat %czero : i32 -> tensor<1024xi32>
  %mod_ne = arith.cmpi ne, %r, %zero : tensor<1024xi32>
  %mask = arith.andi %in_bounds, %mod_ne : tensor<1024xi1>
  %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %2 = tt.addptr %1, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %3 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %4 = tt.addptr %3, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %5 = tt.load %2, %mask, %cst : tensor<1024x!tt.ptr<i32>>
  tt.store %4, %5 : tensor<1024x!tt.ptr<i32>>
  tt.return
}
