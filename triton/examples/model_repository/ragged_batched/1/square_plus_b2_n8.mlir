func.func @square_plus_b2_n8(%0: tensor<2x8xf32>) -> tensor<2x8xf32> {
  %1 = stablehlo.multiply %0, %0 : tensor<2x8xf32>
  %2 = stablehlo.add %1, %0 : tensor<2x8xf32>
  return %2 : tensor<2x8xf32>
}
