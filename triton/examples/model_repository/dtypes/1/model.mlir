func.func @dtypes(%0: tensor<4xf64>, %3: tensor<4xbf16>, %6: tensor<4xi32>) -> (tensor<4xf64>, tensor<4xbf16>, tensor<4xi32>) {
  %1 = stablehlo.multiply %0, %0 : tensor<4xf64>
  %2 = stablehlo.add %1, %0 : tensor<4xf64>
  %4 = stablehlo.multiply %3, %3 : tensor<4xbf16>
  %5 = stablehlo.add %4, %3 : tensor<4xbf16>
  %7 = stablehlo.multiply %6, %6 : tensor<4xi32>
  %8 = stablehlo.add %7, %6 : tensor<4xi32>
  return %2, %5, %8 : tensor<4xf64>, tensor<4xbf16>, tensor<4xi32>
}
