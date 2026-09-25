func.func @matmul_sumsq_grad(%0: tensor<2x2xf32>) -> (tensor<f32>, tensor<2x2xf32>) {
  %1 = stablehlo.dot_general %0, %0, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %s10 = stablehlo.constant dense<0.0> : tensor<f32>
  %2 = stablehlo.reduce(%1 init: %s10) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<f32>
  %3 = stablehlo.constant dense<1.0> : tensor<f32>
  %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
  %5 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %7 = stablehlo.dot_general %4, %5, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %8 = stablehlo.dot_general %5, %4, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %9 = stablehlo.add %7, %8 : tensor<2x2xf32>
  return %2, %9 : tensor<f32>, tensor<2x2xf32>
}
