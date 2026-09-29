module {
  func.func @main(%0: tensor<16x4xf32>, %1: tensor<16xf32>, %5: tensor<f32>) -> tensor<4xf32> {
    %2 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<16xf32>) -> tensor<16x4xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<16x4xf32>
    %s8 = stablehlo.constant dense<0.0> : tensor<f32>
    %4 = stablehlo.reduce(%3 init: %s8) applies stablehlo.add across dimensions = [0] : (tensor<16x4xf32>, tensor<f32>) -> tensor<4xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %7 = stablehlo.divide %4, %6 : tensor<4xf32>
    return %7 : tensor<4xf32>
  }
}
