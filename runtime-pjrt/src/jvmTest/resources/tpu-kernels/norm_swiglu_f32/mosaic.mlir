module {
  func.func @tlaloc_norm_swiglu(%arg0: memref<8x128xf32, #tpu.memory_space<vmem>>, %arg1: memref<1x128xf32, #tpu.memory_space<vmem>>, %arg2: memref<128x256xf32, #tpu.memory_space<vmem>>, %arg3: memref<128x256xf32, #tpu.memory_space<vmem>>, %arg4: memref<8x128xf32, #tpu.memory_space<vmem>>, %arg5: memref<8x256xf32, #tpu.memory_space<vmem>>) attributes {dimension_semantics = [], scalar_prefetch = 0 : i64, scratch_operands = 0 : i64, tpu.core_type = #tpu.core_type<tc>} {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %0 = vector.load %arg0[%c0, %c0_0] : memref<8x128xf32, #tpu.memory_space<vmem>>, vector<8x128xf32>
    %1 = arith.mulf %0, %0 : vector<8x128xf32>
    %cst = arith.constant dense<0.000000e+00> : vector<8xf32>
    %2 = vector.multi_reduction <add>, %1, %cst [1] : vector<8x128xf32> to vector<8xf32>
    %3 = vector.shape_cast %2 : vector<8xf32> to vector<8x1xf32>
    %cst_1 = arith.constant 1.280000e+02 : f32
    %4 = vector.broadcast %cst_1 : f32 to vector<8x1xf32>
    %5 = arith.divf %3, %4 : vector<8x1xf32>
    %cst_2 = arith.constant 9.99999997E-7 : f32
    %6 = vector.broadcast %cst_2 : f32 to vector<8x1xf32>
    %7 = arith.addf %5, %6 : vector<8x1xf32>
    %8 = math.rsqrt %7 : vector<8x1xf32>
    %9 = vector.broadcast %8 : vector<8x1xf32> to vector<8x128xf32>
    %10 = arith.mulf %0, %9 : vector<8x128xf32>
    %c0_3 = arith.constant 0 : index
    %c0_4 = arith.constant 0 : index
    %11 = vector.load %arg1[%c0_3, %c0_4] : memref<1x128xf32, #tpu.memory_space<vmem>>, vector<1x128xf32>
    %12 = vector.broadcast %11 : vector<1x128xf32> to vector<8x128xf32>
    %13 = arith.mulf %10, %12 : vector<8x128xf32>
    %c0_5 = arith.constant 0 : index
    %c0_6 = arith.constant 0 : index
    %14 = vector.load %arg4[%c0_5, %c0_6] : memref<8x128xf32, #tpu.memory_space<vmem>>, vector<8x128xf32>
    tpu.vector_store %arg4[%c0_5, %c0_6], %13 {strides = array<i32>} : memref<8x128xf32, #tpu.memory_space<vmem>>, vector<8x128xf32>, 
    %c0_7 = arith.constant 0 : index
    %c0_8 = arith.constant 0 : index
    %15 = vector.load %arg2[%c0_7, %c0_8] : memref<128x256xf32, #tpu.memory_space<vmem>>, vector<128x256xf32>
    %cst_9 = arith.constant dense<0.000000e+00> : vector<8x256xf32>
    %16 = tpu.matmul %13, %15, %cst_9 {dimension_numbers = #tpu.dot_dimension_numbers<[1], [0], [0], [1], [0, 0, 1, 1], [], []>, precision = #tpu.contract_precision<fp32>} : vector<8x128xf32>, vector<128x256xf32>, vector<8x256xf32> -> vector<8x256xf32>
    %c0_10 = arith.constant 0 : index
    %c0_11 = arith.constant 0 : index
    %17 = vector.load %arg3[%c0_10, %c0_11] : memref<128x256xf32, #tpu.memory_space<vmem>>, vector<128x256xf32>
    %cst_12 = arith.constant dense<0.000000e+00> : vector<8x256xf32>
    %18 = tpu.matmul %13, %17, %cst_12 {dimension_numbers = #tpu.dot_dimension_numbers<[1], [0], [0], [1], [0, 0, 1, 1], [], []>, precision = #tpu.contract_precision<fp32>} : vector<8x128xf32>, vector<128x256xf32>, vector<8x256xf32> -> vector<8x256xf32>
    %19 = arith.negf %16 : vector<8x256xf32>
    %20 = math.exp %19 : vector<8x256xf32>
    %cst_13 = arith.constant 1.000000e+00 : f32
    %21 = vector.broadcast %cst_13 : f32 to vector<8x256xf32>
    %22 = arith.addf %21, %20 : vector<8x256xf32>
    %23 = arith.divf %21, %22 : vector<8x256xf32>
    %24 = arith.mulf %16, %23 : vector<8x256xf32>
    %25 = arith.mulf %24, %18 : vector<8x256xf32>
    %c0_14 = arith.constant 0 : index
    %c0_15 = arith.constant 0 : index
    %26 = vector.load %arg5[%c0_14, %c0_15] : memref<8x256xf32, #tpu.memory_space<vmem>>, vector<8x256xf32>
    tpu.vector_store %arg5[%c0_14, %c0_15], %25 {strides = array<i32>} : memref<8x256xf32, #tpu.memory_space<vmem>>, vector<8x256xf32>, 
    return
  }
}
