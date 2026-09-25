func.func @int64_bool(%0: tensor<4xi64>, %1: tensor<4xi1>) -> (tensor<4xi64>, tensor<4xi1>, tensor<4xi1>) {
  %2 = stablehlo.multiply %0, %0 : tensor<4xi64>
  %3 = stablehlo.add %2, %0 : tensor<4xi64>
  %4 = stablehlo.compare  GT, %2, %0,  SIGNED : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
  %5 = stablehlo.and %4, %1 : tensor<4xi1>
  %6 = stablehlo.not %1 : tensor<4xi1>
  return %3, %5, %6 : tensor<4xi64>, tensor<4xi1>, tensor<4xi1>
}
