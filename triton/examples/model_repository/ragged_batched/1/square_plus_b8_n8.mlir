func.func @square_plus_b8_n8(%0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %1 = stablehlo.multiply %0, %0 : tensor<8x8xf32>
  %2 = stablehlo.add %1, %0 : tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
}
