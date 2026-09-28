module {
  func.func @main(%0: tensor<4xf32>, %1: tensor<4xf32>) -> tensor<4xf32> {
    %2 = stablehlo.constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf32>
    %3 = stablehlo.subtract %1, %2 : tensor<4xf32>
    %4 = stablehlo.constant dense<[1.0E9, 1.0E9, 1.0E9, 1.0E9]> : tensor<4xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<4xf32>
    %6 = stablehlo.add %0, %5 : tensor<4xf32>
    %s8 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s9 = stablehlo.reduce(%6 init: %s8) applies stablehlo.maximum across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %s10 = stablehlo.broadcast_in_dim %s9, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %s11 = stablehlo.subtract %6, %s10 : tensor<4xf32>
    %s12 = stablehlo.exponential %s11 : tensor<4xf32>
    %s13 = stablehlo.constant dense<0.0> : tensor<f32>
    %s14 = stablehlo.reduce(%s12 init: %s13) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %s15 = stablehlo.broadcast_in_dim %s14, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %7 = stablehlo.divide %s12, %s15 : tensor<4xf32>
    return %7 : tensor<4xf32>
  }
}
