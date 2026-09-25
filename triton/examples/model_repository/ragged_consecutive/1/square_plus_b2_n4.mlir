func.func @square_plus_b2_n4(%0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %1 = stablehlo.multiply %0, %0 : tensor<2x4xf32>
  %2 = stablehlo.add %1, %0 : tensor<2x4xf32>
  return %2 : tensor<2x4xf32>
}
