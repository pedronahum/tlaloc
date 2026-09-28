module {
  func.func @tlaloc_rmsnorm(%arg0: memref<16x256xf32, #tpu.memory_space<vmem>>, %arg1: memref<1x256xf32, #tpu.memory_space<vmem>>, %arg2: memref<16x256xf32, #tpu.memory_space<vmem>>) attributes {dimension_semantics = [], scalar_prefetch = 0 : i64, scratch_operands = 0 : i64, tpu.core_type = #tpu.core_type<tc>} {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %0 = vector.load %arg0[%c0, %c0_0] : memref<16x256xf32, #tpu.memory_space<vmem>>, vector<16x256xf32>
    %1 = arith.mulf %0, %0 : vector<16x256xf32>
    %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
    %2 = vector.multi_reduction <add>, %1, %cst [1] : vector<16x256xf32> to vector<16xf32>
    %3 = vector.shape_cast %2 : vector<16xf32> to vector<16x1xf32>
    %cst_1 = arith.constant 2.560000e+02 : f32
    %4 = vector.broadcast %cst_1 : f32 to vector<16x1xf32>
    %5 = arith.divf %3, %4 : vector<16x1xf32>
    %cst_2 = arith.constant 9.99999997E-7 : f32
    %6 = vector.broadcast %cst_2 : f32 to vector<16x1xf32>
    %7 = arith.addf %5, %6 : vector<16x1xf32>
    %8 = math.rsqrt %7 : vector<16x1xf32>
    %9 = vector.broadcast %8 : vector<16x1xf32> to vector<16x256xf32>
    %10 = arith.mulf %0, %9 : vector<16x256xf32>
    %c0_3 = arith.constant 0 : index
    %c0_4 = arith.constant 0 : index
    %11 = vector.load %arg1[%c0_3, %c0_4] : memref<1x256xf32, #tpu.memory_space<vmem>>, vector<1x256xf32>
    %12 = vector.broadcast %11 : vector<1x256xf32> to vector<16x256xf32>
    %13 = arith.mulf %10, %12 : vector<16x256xf32>
    %c0_5 = arith.constant 0 : index
    %c0_6 = arith.constant 0 : index
    %14 = vector.load %arg2[%c0_5, %c0_6] : memref<16x256xf32, #tpu.memory_space<vmem>>, vector<16x256xf32>
    tpu.vector_store %arg2[%c0_5, %c0_6], %13 {strides = array<i32>} : memref<16x256xf32, #tpu.memory_space<vmem>>, vector<16x256xf32>, 
    return
  }
}
