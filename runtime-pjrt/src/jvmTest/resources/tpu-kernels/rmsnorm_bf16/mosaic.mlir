module {
  func.func @tlaloc_rmsnorm(%arg0: memref<16x256xbf16, #tpu.memory_space<vmem>>, %arg1: memref<1x256xbf16, #tpu.memory_space<vmem>>, %arg2: memref<16x256xbf16, #tpu.memory_space<vmem>>) attributes {dimension_semantics = [], scalar_prefetch = 0 : i64, scratch_operands = 0 : i64, tpu.core_type = #tpu.core_type<tc>} {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %0 = vector.load %arg0[%c0, %c0_0] : memref<16x256xbf16, #tpu.memory_space<vmem>>, vector<16x256xbf16>
    %1 = arith.extf %0 : vector<16x256xbf16> to vector<16x256xf32>
    %2 = arith.mulf %1, %1 : vector<16x256xf32>
    %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
    %3 = vector.multi_reduction <add>, %2, %cst [1] : vector<16x256xf32> to vector<16xf32>
    %4 = vector.shape_cast %3 : vector<16xf32> to vector<16x1xf32>
    %cst_1 = arith.constant 2.560000e+02 : f32
    %5 = vector.broadcast %cst_1 : f32 to vector<16x1xf32>
    %6 = arith.divf %4, %5 : vector<16x1xf32>
    %cst_2 = arith.constant 9.99999997E-7 : f32
    %7 = vector.broadcast %cst_2 : f32 to vector<16x1xf32>
    %8 = arith.addf %6, %7 : vector<16x1xf32>
    %9 = math.rsqrt %8 : vector<16x1xf32>
    %10 = vector.broadcast %9 : vector<16x1xf32> to vector<16x256xf32>
    %11 = arith.mulf %1, %10 : vector<16x256xf32>
    %c0_3 = arith.constant 0 : index
    %c0_4 = arith.constant 0 : index
    %12 = vector.load %arg1[%c0_3, %c0_4] : memref<1x256xbf16, #tpu.memory_space<vmem>>, vector<1x256xbf16>
    %13 = arith.extf %12 : vector<1x256xbf16> to vector<1x256xf32>
    %14 = vector.broadcast %13 : vector<1x256xf32> to vector<16x256xf32>
    %15 = arith.mulf %11, %14 : vector<16x256xf32>
    %16 = arith.truncf %15 : vector<16x256xf32> to vector<16x256xbf16>
    %c0_5 = arith.constant 0 : index
    %c0_6 = arith.constant 0 : index
    %17 = vector.load %arg2[%c0_5, %c0_6] : memref<16x256xbf16, #tpu.memory_space<vmem>>, vector<16x256xbf16>
    tpu.vector_store %arg2[%c0_5, %c0_6], %16 {strides = array<i32>} : memref<16x256xbf16, #tpu.memory_space<vmem>>, vector<16x256xbf16>, 
    return
  }
}
