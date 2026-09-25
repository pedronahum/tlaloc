// Written by hand, not emitted by Tlaloc (Tlaloc has no f16, i8 or u8 dtype).
func.func @dtypes_small(%x: tensor<4xf16>, %y: tensor<4xi8>, %z: tensor<4xui8>) -> (tensor<4xf16>, tensor<4xi8>, tensor<4xui8>) {
  %0 = stablehlo.multiply %x, %x : tensor<4xf16>
  %1 = stablehlo.add %0, %x : tensor<4xf16>
  %2 = stablehlo.multiply %y, %y : tensor<4xi8>
  %3 = stablehlo.add %2, %y : tensor<4xi8>
  %4 = stablehlo.multiply %z, %z : tensor<4xui8>
  %5 = stablehlo.add %4, %z : tensor<4xui8>
  return %1, %3, %5 : tensor<4xf16>, tensor<4xi8>, tensor<4xui8>
}
