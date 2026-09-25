func.func @square_plus_8(%0: tensor<8xf32>) -> tensor<8xf32> {
  %1 = stablehlo.multiply %0, %0 : tensor<8xf32>
  %2 = stablehlo.add %1, %0 : tensor<8xf32>
  return %2 : tensor<8xf32>
}
