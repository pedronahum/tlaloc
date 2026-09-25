func.func @large_square_plus(%0: tensor<4194304xf32>) -> tensor<4194304xf32> {
  %1 = stablehlo.multiply %0, %0 : tensor<4194304xf32>
  %2 = stablehlo.add %1, %0 : tensor<4194304xf32>
  return %2 : tensor<4194304xf32>
}
