func.func @square_plus_4(%0: tensor<4xf32>) -> tensor<4xf32> {
  %1 = stablehlo.multiply %0, %0 : tensor<4xf32>
  %2 = stablehlo.add %1, %0 : tensor<4xf32>
  return %2 : tensor<4xf32>
}
