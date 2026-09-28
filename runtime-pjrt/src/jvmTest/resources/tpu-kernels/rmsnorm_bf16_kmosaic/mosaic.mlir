module {
  func.func @tlaloc_rmsnorm(%x_ref: memref<16x256xbf16, #tpu.memory_space<vmem>>, %w_ref: memref<1x256xbf16, #tpu.memory_space<vmem>>, %o_ref: memref<16x256xbf16, #tpu.memory_space<vmem>>) attributes {dimension_semantics = [], scalar_prefetch = 0 : i64, scratch_operands = 0 : i64, tpu.core_type = #tpu.core_type<tc>} {
    %c0 = arith.constant 0 : index
    %x_in = vector.load %x_ref[%c0, %c0] : memref<16x256xbf16, #tpu.memory_space<vmem>>, vector<16x256xbf16>
    %x = arith.extf %x_in : vector<16x256xbf16> to vector<16x256xf32>
    %sq = arith.mulf %x, %x : vector<16x256xf32>
    %zero = arith.constant dense<0.000000e+00> : vector<16xf32>
    %sum = vector.multi_reduction <add>, %sq, %zero [1] : vector<16x256xf32> to vector<16xf32>
    %sum_col = vector.shape_cast %sum : vector<16xf32> to vector<16x1xf32>
    %n = arith.constant 0x43800000 : f32
    %n_col = vector.broadcast %n : f32 to vector<16x1xf32>
    %mean = arith.divf %sum_col, %n_col : vector<16x1xf32>
    %eps = arith.constant 0x358637BD : f32
    %eps_col = vector.broadcast %eps : f32 to vector<16x1xf32>
    %var = arith.addf %mean, %eps_col : vector<16x1xf32>
    %inv = math.rsqrt %var : vector<16x1xf32>
    %inv_full = vector.broadcast %inv : vector<16x1xf32> to vector<16x256xf32>
    %normed = arith.mulf %x, %inv_full : vector<16x256xf32>
    %w_in = vector.load %w_ref[%c0, %c0] : memref<1x256xbf16, #tpu.memory_space<vmem>>, vector<1x256xbf16>
    %w = arith.extf %w_in : vector<1x256xbf16> to vector<1x256xf32>
    %w_full = vector.broadcast %w : vector<1x256xf32> to vector<16x256xf32>
    %y = arith.mulf %normed, %w_full : vector<16x256xf32>
    %y_out = arith.truncf %y : vector<16x256xf32> to vector<16x256xbf16>
    tpu.vector_store %o_ref[%c0, %c0], %y_out {strides = array<i32>} : memref<16x256xbf16, #tpu.memory_space<vmem>>, vector<16x256xbf16>,
    return
  }
}
