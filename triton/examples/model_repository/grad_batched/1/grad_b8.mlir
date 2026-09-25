func.func @batched_sumsq_b8_grad(%0: tensor<8x2x2xf32>) -> tensor<8x2x2xf32> {
  %1 = stablehlo.constant dense<1.0> : tensor<f32>
  %3 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<8x2x2xf32>
  %5 = stablehlo.transpose %0, dims = [0, 2, 1] : (tensor<8x2x2xf32>) -> tensor<8x2x2xf32>
  %7 = stablehlo.dot_general %3, %5, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x2x2xf32>, tensor<8x2x2xf32>) -> tensor<8x2x2xf32>
  %8 = stablehlo.dot_general %5, %3, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x2x2xf32>, tensor<8x2x2xf32>) -> tensor<8x2x2xf32>
  %9 = stablehlo.add %7, %8 : tensor<8x2x2xf32>
  return %9 : tensor<8x2x2xf32>
}
