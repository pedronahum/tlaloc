module {
  func.func @main(%0: tensor<1x6xi32>, %1: tensor<1x6xi32>, %2: tensor<1x4xi32>, %3: tensor<1xi32>, %4: tensor<6xi32>, %5: tensor<1x4xi32>, %6: tensor<6xi32>, %7: tensor<17x4x2x8xf32> {tf.aliasing_output = 1 : i32}, %8: tensor<17x4x2x8xf32> {tf.aliasing_output = 2 : i32}, %9: tensor<17x4x2x8xf32> {tf.aliasing_output = 3 : i32}, %10: tensor<17x4x2x8xf32> {tf.aliasing_output = 4 : i32}, %11: tensor<65x4x2x8xf32> {tf.aliasing_output = 5 : i32}, %12: tensor<65x4x2x8xf32> {tf.aliasing_output = 6 : i32}, %13: tensor<64x32xf32>, %14: tensor<32xf32>, %15: tensor<32x32xf32>, %16: tensor<32x16xf32>, %17: tensor<32x16xf32>, %18: tensor<32x32xf32>, %19: tensor<32xf32>, %20: tensor<32x48xf32>, %21: tensor<32x48xf32>, %22: tensor<48x32xf32>, %23: tensor<32xf32>, %24: tensor<32x32xf32>, %25: tensor<32x16xf32>, %26: tensor<32x16xf32>, %27: tensor<32x32xf32>, %28: tensor<32xf32>, %29: tensor<32x48xf32>, %30: tensor<32x48xf32>, %31: tensor<48x32xf32>, %32: tensor<32xf32>, %33: tensor<32x32xf32>, %34: tensor<32x16xf32>, %35: tensor<32x16xf32>, %36: tensor<32x32xf32>, %37: tensor<32xf32>, %38: tensor<32x48xf32>, %39: tensor<32x48xf32>, %40: tensor<48x32xf32>, %41: tensor<32xf32>, %42: tensor<32x64xf32>) -> (tensor<1x1x64xf32>, tensor<17x4x2x8xf32>, tensor<17x4x2x8xf32>, tensor<17x4x2x8xf32>, tensor<17x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>) {
    %43 = stablehlo.reshape %1 : (tensor<1x6xi32>) -> tensor<6xi32>
    %44 = stablehlo.constant dense<[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.5403023, 0.9950042, 0.99995, 0.9999995, 0.5403023, 0.9950042, 0.99995, 0.9999995], [-0.41614684, 0.9800666, 0.9998, 0.999998, -0.41614684, 0.9800666, 0.9998, 0.999998], [-0.9899925, 0.9553365, 0.99955004, 0.9999955, -0.9899925, 0.9553365, 0.99955004, 0.9999955], [-0.6536436, 0.921061, 0.9992001, 0.999992, -0.6536436, 0.921061, 0.9992001, 0.999992], [0.2836622, 0.87758255, 0.99875027, 0.9999875, 0.2836622, 0.87758255, 0.99875027, 0.9999875], [0.96017027, 0.8253356, 0.99820054, 0.999982, 0.96017027, 0.8253356, 0.99820054, 0.999982], [0.75390226, 0.7648422, 0.997551, 0.9999755, 0.75390226, 0.7648422, 0.997551, 0.9999755], [-0.14550003, 0.6967067, 0.99680173, 0.999968, -0.14550003, 0.6967067, 0.99680173, 0.999968], [-0.91113025, 0.62161, 0.9959527, 0.9999595, -0.91113025, 0.62161, 0.9959527, 0.9999595], [-0.8390715, 0.5403023, 0.9950042, 0.99995, -0.8390715, 0.5403023, 0.9950042, 0.99995], [0.004425698, 0.45359612, 0.9939561, 0.9999395, 0.004425698, 0.45359612, 0.9939561, 0.9999395], [0.84385395, 0.36235777, 0.99280864, 0.999928, 0.84385395, 0.36235777, 0.99280864, 0.999928], [0.9074468, 0.26749882, 0.9915619, 0.9999155, 0.9074468, 0.26749882, 0.9915619, 0.9999155], [0.13673721, 0.16996714, 0.990216, 0.999902, 0.13673721, 0.16996714, 0.990216, 0.999902], [-0.7596879, 0.0707372, 0.9887711, 0.9998875, -0.7596879, 0.0707372, 0.9887711, 0.9998875]]> : tensor<16x8xf32>
    %45 = stablehlo.constant dense<[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.84147096, 0.099833414, 0.009999833, 9.999998E-4, 0.84147096, 0.099833414, 0.009999833, 9.999998E-4], [0.9092974, 0.19866933, 0.019998666, 0.0019999987, 0.9092974, 0.19866933, 0.019998666, 0.0019999987], [0.14112, 0.29552022, 0.029995501, 0.0029999956, 0.14112, 0.29552022, 0.029995501, 0.0029999956], [-0.7568025, 0.38941833, 0.039989334, 0.0039999895, -0.7568025, 0.38941833, 0.039989334, 0.0039999895], [-0.9589243, 0.47942555, 0.04997917, 0.0049999794, -0.9589243, 0.47942555, 0.04997917, 0.0049999794], [-0.2794155, 0.5646425, 0.059964005, 0.005999964, -0.2794155, 0.5646425, 0.059964005, 0.005999964], [0.6569866, 0.64421767, 0.06994285, 0.006999943, 0.6569866, 0.64421767, 0.06994285, 0.006999943], [0.98935825, 0.7173561, 0.0799147, 0.007999915, 0.98935825, 0.7173561, 0.0799147, 0.007999915], [0.4121185, 0.7833269, 0.08987855, 0.008999879, 0.4121185, 0.7833269, 0.08987855, 0.008999879], [-0.5440211, 0.84147096, 0.099833414, 0.009999833, -0.5440211, 0.84147096, 0.099833414, 0.009999833], [-0.9999902, 0.89120734, 0.1097783, 0.010999778, -0.9999902, 0.89120734, 0.1097783, 0.010999778], [-0.53657293, 0.9320391, 0.119712204, 0.011999712, -0.53657293, 0.9320391, 0.119712204, 0.011999712], [0.42016703, 0.9635582, 0.12963414, 0.012999634, 0.42016703, 0.9635582, 0.12963414, 0.012999634], [0.9906074, 0.98544973, 0.13954312, 0.013999542, 0.9906074, 0.98544973, 0.13954312, 0.013999542], [0.65028787, 0.997495, 0.14943813, 0.014999437, 0.65028787, 0.997495, 0.14943813, 0.014999437]]> : tensor<16x8xf32>
    %46 = "stablehlo.gather"(%44, %43) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 8>}> : (tensor<16x8xf32>, tensor<6xi32>) -> tensor<6x8xf32>
    %47 = "stablehlo.gather"(%45, %43) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 8>}> : (tensor<16x8xf32>, tensor<6xi32>) -> tensor<6x8xf32>
    %48 = stablehlo.reshape %46 : (tensor<6x8xf32>) -> tensor<6x1x8xf32>
    %49 = stablehlo.broadcast_in_dim %48, dims = [0, 1, 2] : (tensor<6x1x8xf32>) -> tensor<6x4x8xf32>
    %50 = stablehlo.reshape %47 : (tensor<6x8xf32>) -> tensor<6x1x8xf32>
    %51 = stablehlo.broadcast_in_dim %50, dims = [0, 1, 2] : (tensor<6x1x8xf32>) -> tensor<6x4x8xf32>
    %52 = stablehlo.reshape %46 : (tensor<6x8xf32>) -> tensor<6x1x8xf32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [0, 1, 2] : (tensor<6x1x8xf32>) -> tensor<6x2x8xf32>
    %54 = stablehlo.reshape %47 : (tensor<6x8xf32>) -> tensor<6x1x8xf32>
    %55 = stablehlo.broadcast_in_dim %54, dims = [0, 1, 2] : (tensor<6x1x8xf32>) -> tensor<6x2x8xf32>
    %56 = stablehlo.constant dense<1> : tensor<6xi32>
    %57 = stablehlo.add %43, %56 : tensor<6xi32>
    %58 = "stablehlo.gather"(%13, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 32>}> : (tensor<64x32xf32>, tensor<1x6xi32>) -> tensor<1x6x32xf32>
    %59 = stablehlo.reshape %58 : (tensor<1x6x32xf32>) -> tensor<6x32xf32>
    %60 = stablehlo.constant dense<[[1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6]]> : tensor<6x1xf32>
    %61 = stablehlo.multiply %59, %59 : tensor<6x32xf32>
    %s219 = stablehlo.constant dense<0.0> : tensor<f32>
    %s220 = stablehlo.reduce(%61 init: %s219) applies stablehlo.add across dimensions = [1] : (tensor<6x32xf32>, tensor<f32>) -> tensor<6xf32>
    %s221 = stablehlo.constant dense<32.0> : tensor<6xf32>
    %s222 = stablehlo.divide %s220, %s221 : tensor<6xf32>
    %62 = stablehlo.broadcast_in_dim %s222, dims = [0] : (tensor<6xf32>) -> tensor<6x1xf32>
    %63 = stablehlo.add %62, %60 : tensor<6x1xf32>
    %64 = stablehlo.rsqrt %63 : tensor<6x1xf32>
    %s223 = stablehlo.broadcast_in_dim %64, dims = [0, 1] : (tensor<6x1xf32>) -> tensor<6x32xf32>
    %65 = stablehlo.multiply %59, %s223 : tensor<6x32xf32>
    %66 = stablehlo.reshape %14 : (tensor<32xf32>) -> tensor<1x32xf32>
    %67 = stablehlo.broadcast_in_dim %66, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<6x32xf32>
    %68 = stablehlo.multiply %65, %67 : tensor<6x32xf32>
    %69 = stablehlo.dot_general %68, %15, contracting_dims = [1] x [0] : (tensor<6x32xf32>, tensor<32x32xf32>) -> tensor<6x32xf32>
    %70 = stablehlo.dot_general %68, %16, contracting_dims = [1] x [0] : (tensor<6x32xf32>, tensor<32x16xf32>) -> tensor<6x16xf32>
    %71 = stablehlo.dot_general %68, %17, contracting_dims = [1] x [0] : (tensor<6x32xf32>, tensor<32x16xf32>) -> tensor<6x16xf32>
    %72 = stablehlo.reshape %69 : (tensor<6x32xf32>) -> tensor<6x4x8xf32>
    %73 = stablehlo.reshape %70 : (tensor<6x16xf32>) -> tensor<6x2x8xf32>
    %74 = stablehlo.reshape %71 : (tensor<6x16xf32>) -> tensor<6x2x8xf32>
    %75 = stablehlo.multiply %72, %49 : tensor<6x4x8xf32>
    %76 = stablehlo.slice %72 [0:6, 0:4, 4:8] : (tensor<6x4x8xf32>) -> tensor<6x4x4xf32>
    %77 = stablehlo.slice %72 [0:6, 0:4, 0:4] : (tensor<6x4x8xf32>) -> tensor<6x4x4xf32>
    %78 = stablehlo.negate %76 : tensor<6x4x4xf32>
    %79 = stablehlo.concatenate %78, %77, dim = 2 : (tensor<6x4x4xf32>, tensor<6x4x4xf32>) -> tensor<6x4x8xf32>
    %80 = stablehlo.multiply %79, %51 : tensor<6x4x8xf32>
    %81 = stablehlo.add %75, %80 : tensor<6x4x8xf32>
    %82 = stablehlo.multiply %73, %53 : tensor<6x2x8xf32>
    %83 = stablehlo.slice %73 [0:6, 0:2, 4:8] : (tensor<6x2x8xf32>) -> tensor<6x2x4xf32>
    %84 = stablehlo.slice %73 [0:6, 0:2, 0:4] : (tensor<6x2x8xf32>) -> tensor<6x2x4xf32>
    %85 = stablehlo.negate %83 : tensor<6x2x4xf32>
    %86 = stablehlo.concatenate %85, %84, dim = 2 : (tensor<6x2x4xf32>, tensor<6x2x4xf32>) -> tensor<6x2x8xf32>
    %87 = stablehlo.multiply %86, %55 : tensor<6x2x8xf32>
    %88 = stablehlo.add %82, %87 : tensor<6x2x8xf32>
    %s224 = stablehlo.reshape %7 : (tensor<17x4x2x8xf32>) -> tensor<68x2x8xf32>
    %s225 = "stablehlo.scatter"(%s224, %6, %88) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s226: tensor<f32>, %s227: tensor<f32>):
       stablehlo.return %s227 : tensor<f32>
     }) : (tensor<68x2x8xf32>, tensor<6xi32>, tensor<6x2x8xf32>) -> tensor<68x2x8xf32>
    %89 = stablehlo.reshape %s225 : (tensor<68x2x8xf32>) -> tensor<17x4x2x8xf32>
    %s228 = stablehlo.reshape %8 : (tensor<17x4x2x8xf32>) -> tensor<68x2x8xf32>
    %s229 = "stablehlo.scatter"(%s228, %6, %74) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s230: tensor<f32>, %s231: tensor<f32>):
       stablehlo.return %s231 : tensor<f32>
     }) : (tensor<68x2x8xf32>, tensor<6xi32>, tensor<6x2x8xf32>) -> tensor<68x2x8xf32>
    %90 = stablehlo.reshape %s229 : (tensor<68x2x8xf32>) -> tensor<17x4x2x8xf32>
    %s232 = "stablehlo.gather"(%89, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<17x4x2x8xf32>, tensor<1x4xi32>) -> tensor<1x4x4x2x8xf32>
    %s233 = stablehlo.reshape %s232 : (tensor<1x4x4x2x8xf32>) -> tensor<1x16x2x8xf32>
    %s234 = "stablehlo.gather"(%90, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<17x4x2x8xf32>, tensor<1x4xi32>) -> tensor<1x4x4x2x8xf32>
    %s235 = stablehlo.reshape %s234 : (tensor<1x4x4x2x8xf32>) -> tensor<1x16x2x8xf32>
    %s236 = stablehlo.reshape %81 : (tensor<6x4x8xf32>) -> tensor<1x6x2x2x8xf32>
    %s237 = stablehlo.dot_general %s236, %s233, batching_dims = [0, 2] x [0, 2], contracting_dims = [4] x [3], precision = [HIGHEST, HIGHEST] : (tensor<1x6x2x2x8xf32>, tensor<1x16x2x8xf32>) -> tensor<1x2x6x2x16xf32>
    %s238 = stablehlo.constant dense<0.35355338> : tensor<f32>
    %s239 = stablehlo.broadcast_in_dim %s238, dims = [] : (tensor<f32>) -> tensor<1x2x6x2x16xf32>
    %s240 = stablehlo.multiply %s237, %s239 : tensor<1x2x6x2x16xf32>
    %s241 = stablehlo.iota dim = 4 : tensor<1x2x6x2x16xi32>
    %s242 = stablehlo.reshape %57 : (tensor<6xi32>) -> tensor<1x6xi32>
    %s243 = stablehlo.broadcast_in_dim %s242, dims = [0, 2] : (tensor<1x6xi32>) -> tensor<1x2x6x2x16xi32>
    %s245 = stablehlo.compare LT, %s241, %s243, SIGNED : (tensor<1x2x6x2x16xi32>, tensor<1x2x6x2x16xi32>) -> tensor<1x2x6x2x16xi1>
    %s246 = stablehlo.constant dense<8> : tensor<i32>
    %s247 = stablehlo.broadcast_in_dim %s246, dims = [] : (tensor<i32>) -> tensor<1x2x6x2x16xi32>
    %s248 = stablehlo.subtract %s243, %s247 : tensor<1x2x6x2x16xi32>
    %s249 = stablehlo.compare GE, %s241, %s248, SIGNED : (tensor<1x2x6x2x16xi32>, tensor<1x2x6x2x16xi32>) -> tensor<1x2x6x2x16xi1>
    %s244 = stablehlo.and %s245, %s249 : tensor<1x2x6x2x16xi1>
    %s250 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s251 = stablehlo.broadcast_in_dim %s250, dims = [] : (tensor<f32>) -> tensor<1x2x6x2x16xf32>
    %s252 = stablehlo.select %s244, %s240, %s251 : tensor<1x2x6x2x16xi1>, tensor<1x2x6x2x16xf32>
    %s253 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s254 = stablehlo.reduce(%s252 init: %s253) applies stablehlo.maximum across dimensions = [4] : (tensor<1x2x6x2x16xf32>, tensor<f32>) -> tensor<1x2x6x2xf32>
    %s255 = stablehlo.broadcast_in_dim %s254, dims = [0, 1, 2, 3] : (tensor<1x2x6x2xf32>) -> tensor<1x2x6x2x16xf32>
    %s256 = stablehlo.subtract %s252, %s255 : tensor<1x2x6x2x16xf32>
    %s257 = stablehlo.exponential %s256 : tensor<1x2x6x2x16xf32>
    %s258 = stablehlo.constant dense<0.0> : tensor<f32>
    %s259 = stablehlo.reduce(%s257 init: %s258) applies stablehlo.add across dimensions = [4] : (tensor<1x2x6x2x16xf32>, tensor<f32>) -> tensor<1x2x6x2xf32>
    %s260 = stablehlo.broadcast_in_dim %s259, dims = [0, 1, 2, 3] : (tensor<1x2x6x2xf32>) -> tensor<1x2x6x2x16xf32>
    %s261 = stablehlo.divide %s257, %s260 : tensor<1x2x6x2x16xf32>
    %s262 = stablehlo.dot_general %s261, %s235, batching_dims = [0, 1] x [0, 2], contracting_dims = [4] x [1], precision = [HIGHEST, HIGHEST] : (tensor<1x2x6x2x16xf32>, tensor<1x16x2x8xf32>) -> tensor<1x2x6x2x8xf32>
    %s263 = stablehlo.transpose %s262, dims = [0, 2, 1, 3, 4] : (tensor<1x2x6x2x8xf32>) -> tensor<1x6x2x2x8xf32>
    %91 = stablehlo.reshape %s263 : (tensor<1x6x2x2x8xf32>) -> tensor<6x4x8xf32>
    %92 = stablehlo.reshape %91 : (tensor<6x4x8xf32>) -> tensor<6x32xf32>
    %93 = stablehlo.dot_general %92, %18, contracting_dims = [1] x [0] : (tensor<6x32xf32>, tensor<32x32xf32>) -> tensor<6x32xf32>
    %94 = stablehlo.add %59, %93 : tensor<6x32xf32>
    %95 = stablehlo.multiply %94, %94 : tensor<6x32xf32>
    %s264 = stablehlo.constant dense<0.0> : tensor<f32>
    %s265 = stablehlo.reduce(%95 init: %s264) applies stablehlo.add across dimensions = [1] : (tensor<6x32xf32>, tensor<f32>) -> tensor<6xf32>
    %s266 = stablehlo.constant dense<32.0> : tensor<6xf32>
    %s267 = stablehlo.divide %s265, %s266 : tensor<6xf32>
    %96 = stablehlo.broadcast_in_dim %s267, dims = [0] : (tensor<6xf32>) -> tensor<6x1xf32>
    %97 = stablehlo.add %96, %60 : tensor<6x1xf32>
    %98 = stablehlo.rsqrt %97 : tensor<6x1xf32>
    %s268 = stablehlo.broadcast_in_dim %98, dims = [0, 1] : (tensor<6x1xf32>) -> tensor<6x32xf32>
    %99 = stablehlo.multiply %94, %s268 : tensor<6x32xf32>
    %100 = stablehlo.reshape %19 : (tensor<32xf32>) -> tensor<1x32xf32>
    %101 = stablehlo.broadcast_in_dim %100, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<6x32xf32>
    %102 = stablehlo.multiply %99, %101 : tensor<6x32xf32>
    %103 = stablehlo.dot_general %102, %20, contracting_dims = [1] x [0] : (tensor<6x32xf32>, tensor<32x48xf32>) -> tensor<6x48xf32>
    %104 = stablehlo.dot_general %102, %21, contracting_dims = [1] x [0] : (tensor<6x32xf32>, tensor<32x48xf32>) -> tensor<6x48xf32>
    %s269 = stablehlo.logistic %103 : tensor<6x48xf32>
    %105 = stablehlo.multiply %103, %s269 : tensor<6x48xf32>
    %106 = stablehlo.multiply %105, %104 : tensor<6x48xf32>
    %107 = stablehlo.dot_general %106, %22, contracting_dims = [1] x [0] : (tensor<6x48xf32>, tensor<48x32xf32>) -> tensor<6x32xf32>
    %108 = stablehlo.add %94, %107 : tensor<6x32xf32>
    %109 = stablehlo.multiply %108, %108 : tensor<6x32xf32>
    %s270 = stablehlo.constant dense<0.0> : tensor<f32>
    %s271 = stablehlo.reduce(%109 init: %s270) applies stablehlo.add across dimensions = [1] : (tensor<6x32xf32>, tensor<f32>) -> tensor<6xf32>
    %s272 = stablehlo.constant dense<32.0> : tensor<6xf32>
    %s273 = stablehlo.divide %s271, %s272 : tensor<6xf32>
    %110 = stablehlo.broadcast_in_dim %s273, dims = [0] : (tensor<6xf32>) -> tensor<6x1xf32>
    %111 = stablehlo.add %110, %60 : tensor<6x1xf32>
    %112 = stablehlo.rsqrt %111 : tensor<6x1xf32>
    %s274 = stablehlo.broadcast_in_dim %112, dims = [0, 1] : (tensor<6x1xf32>) -> tensor<6x32xf32>
    %113 = stablehlo.multiply %108, %s274 : tensor<6x32xf32>
    %114 = stablehlo.reshape %23 : (tensor<32xf32>) -> tensor<1x32xf32>
    %115 = stablehlo.broadcast_in_dim %114, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<6x32xf32>
    %116 = stablehlo.multiply %113, %115 : tensor<6x32xf32>
    %117 = stablehlo.dot_general %116, %24, contracting_dims = [1] x [0] : (tensor<6x32xf32>, tensor<32x32xf32>) -> tensor<6x32xf32>
    %118 = stablehlo.dot_general %116, %25, contracting_dims = [1] x [0] : (tensor<6x32xf32>, tensor<32x16xf32>) -> tensor<6x16xf32>
    %119 = stablehlo.dot_general %116, %26, contracting_dims = [1] x [0] : (tensor<6x32xf32>, tensor<32x16xf32>) -> tensor<6x16xf32>
    %120 = stablehlo.reshape %117 : (tensor<6x32xf32>) -> tensor<6x4x8xf32>
    %121 = stablehlo.reshape %118 : (tensor<6x16xf32>) -> tensor<6x2x8xf32>
    %122 = stablehlo.reshape %119 : (tensor<6x16xf32>) -> tensor<6x2x8xf32>
    %123 = stablehlo.multiply %120, %49 : tensor<6x4x8xf32>
    %124 = stablehlo.slice %120 [0:6, 0:4, 4:8] : (tensor<6x4x8xf32>) -> tensor<6x4x4xf32>
    %125 = stablehlo.slice %120 [0:6, 0:4, 0:4] : (tensor<6x4x8xf32>) -> tensor<6x4x4xf32>
    %126 = stablehlo.negate %124 : tensor<6x4x4xf32>
    %127 = stablehlo.concatenate %126, %125, dim = 2 : (tensor<6x4x4xf32>, tensor<6x4x4xf32>) -> tensor<6x4x8xf32>
    %128 = stablehlo.multiply %127, %51 : tensor<6x4x8xf32>
    %129 = stablehlo.add %123, %128 : tensor<6x4x8xf32>
    %130 = stablehlo.multiply %121, %53 : tensor<6x2x8xf32>
    %131 = stablehlo.slice %121 [0:6, 0:2, 4:8] : (tensor<6x2x8xf32>) -> tensor<6x2x4xf32>
    %132 = stablehlo.slice %121 [0:6, 0:2, 0:4] : (tensor<6x2x8xf32>) -> tensor<6x2x4xf32>
    %133 = stablehlo.negate %131 : tensor<6x2x4xf32>
    %134 = stablehlo.concatenate %133, %132, dim = 2 : (tensor<6x2x4xf32>, tensor<6x2x4xf32>) -> tensor<6x2x8xf32>
    %135 = stablehlo.multiply %134, %55 : tensor<6x2x8xf32>
    %136 = stablehlo.add %130, %135 : tensor<6x2x8xf32>
    %s275 = stablehlo.reshape %9 : (tensor<17x4x2x8xf32>) -> tensor<68x2x8xf32>
    %s276 = "stablehlo.scatter"(%s275, %6, %136) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s277: tensor<f32>, %s278: tensor<f32>):
       stablehlo.return %s278 : tensor<f32>
     }) : (tensor<68x2x8xf32>, tensor<6xi32>, tensor<6x2x8xf32>) -> tensor<68x2x8xf32>
    %137 = stablehlo.reshape %s276 : (tensor<68x2x8xf32>) -> tensor<17x4x2x8xf32>
    %s279 = stablehlo.reshape %10 : (tensor<17x4x2x8xf32>) -> tensor<68x2x8xf32>
    %s280 = "stablehlo.scatter"(%s279, %6, %122) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s281: tensor<f32>, %s282: tensor<f32>):
       stablehlo.return %s282 : tensor<f32>
     }) : (tensor<68x2x8xf32>, tensor<6xi32>, tensor<6x2x8xf32>) -> tensor<68x2x8xf32>
    %138 = stablehlo.reshape %s280 : (tensor<68x2x8xf32>) -> tensor<17x4x2x8xf32>
    %s283 = "stablehlo.gather"(%137, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<17x4x2x8xf32>, tensor<1x4xi32>) -> tensor<1x4x4x2x8xf32>
    %s284 = stablehlo.reshape %s283 : (tensor<1x4x4x2x8xf32>) -> tensor<1x16x2x8xf32>
    %s285 = "stablehlo.gather"(%138, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<17x4x2x8xf32>, tensor<1x4xi32>) -> tensor<1x4x4x2x8xf32>
    %s286 = stablehlo.reshape %s285 : (tensor<1x4x4x2x8xf32>) -> tensor<1x16x2x8xf32>
    %s287 = stablehlo.reshape %129 : (tensor<6x4x8xf32>) -> tensor<1x6x2x2x8xf32>
    %s288 = stablehlo.dot_general %s287, %s284, batching_dims = [0, 2] x [0, 2], contracting_dims = [4] x [3], precision = [HIGHEST, HIGHEST] : (tensor<1x6x2x2x8xf32>, tensor<1x16x2x8xf32>) -> tensor<1x2x6x2x16xf32>
    %s289 = stablehlo.constant dense<0.35355338> : tensor<f32>
    %s290 = stablehlo.broadcast_in_dim %s289, dims = [] : (tensor<f32>) -> tensor<1x2x6x2x16xf32>
    %s291 = stablehlo.multiply %s288, %s290 : tensor<1x2x6x2x16xf32>
    %s292 = stablehlo.iota dim = 4 : tensor<1x2x6x2x16xi32>
    %s293 = stablehlo.reshape %57 : (tensor<6xi32>) -> tensor<1x6xi32>
    %s294 = stablehlo.broadcast_in_dim %s293, dims = [0, 2] : (tensor<1x6xi32>) -> tensor<1x2x6x2x16xi32>
    %s296 = stablehlo.compare LT, %s292, %s294, SIGNED : (tensor<1x2x6x2x16xi32>, tensor<1x2x6x2x16xi32>) -> tensor<1x2x6x2x16xi1>
    %s297 = stablehlo.constant dense<8> : tensor<i32>
    %s298 = stablehlo.broadcast_in_dim %s297, dims = [] : (tensor<i32>) -> tensor<1x2x6x2x16xi32>
    %s299 = stablehlo.subtract %s294, %s298 : tensor<1x2x6x2x16xi32>
    %s300 = stablehlo.compare GE, %s292, %s299, SIGNED : (tensor<1x2x6x2x16xi32>, tensor<1x2x6x2x16xi32>) -> tensor<1x2x6x2x16xi1>
    %s295 = stablehlo.and %s296, %s300 : tensor<1x2x6x2x16xi1>
    %s301 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s302 = stablehlo.broadcast_in_dim %s301, dims = [] : (tensor<f32>) -> tensor<1x2x6x2x16xf32>
    %s303 = stablehlo.select %s295, %s291, %s302 : tensor<1x2x6x2x16xi1>, tensor<1x2x6x2x16xf32>
    %s304 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s305 = stablehlo.reduce(%s303 init: %s304) applies stablehlo.maximum across dimensions = [4] : (tensor<1x2x6x2x16xf32>, tensor<f32>) -> tensor<1x2x6x2xf32>
    %s306 = stablehlo.broadcast_in_dim %s305, dims = [0, 1, 2, 3] : (tensor<1x2x6x2xf32>) -> tensor<1x2x6x2x16xf32>
    %s307 = stablehlo.subtract %s303, %s306 : tensor<1x2x6x2x16xf32>
    %s308 = stablehlo.exponential %s307 : tensor<1x2x6x2x16xf32>
    %s309 = stablehlo.constant dense<0.0> : tensor<f32>
    %s310 = stablehlo.reduce(%s308 init: %s309) applies stablehlo.add across dimensions = [4] : (tensor<1x2x6x2x16xf32>, tensor<f32>) -> tensor<1x2x6x2xf32>
    %s311 = stablehlo.broadcast_in_dim %s310, dims = [0, 1, 2, 3] : (tensor<1x2x6x2xf32>) -> tensor<1x2x6x2x16xf32>
    %s312 = stablehlo.divide %s308, %s311 : tensor<1x2x6x2x16xf32>
    %s313 = stablehlo.dot_general %s312, %s286, batching_dims = [0, 1] x [0, 2], contracting_dims = [4] x [1], precision = [HIGHEST, HIGHEST] : (tensor<1x2x6x2x16xf32>, tensor<1x16x2x8xf32>) -> tensor<1x2x6x2x8xf32>
    %s314 = stablehlo.transpose %s313, dims = [0, 2, 1, 3, 4] : (tensor<1x2x6x2x8xf32>) -> tensor<1x6x2x2x8xf32>
    %139 = stablehlo.reshape %s314 : (tensor<1x6x2x2x8xf32>) -> tensor<6x4x8xf32>
    %140 = stablehlo.reshape %139 : (tensor<6x4x8xf32>) -> tensor<6x32xf32>
    %141 = stablehlo.dot_general %140, %27, contracting_dims = [1] x [0] : (tensor<6x32xf32>, tensor<32x32xf32>) -> tensor<6x32xf32>
    %142 = stablehlo.add %108, %141 : tensor<6x32xf32>
    %143 = stablehlo.multiply %142, %142 : tensor<6x32xf32>
    %s315 = stablehlo.constant dense<0.0> : tensor<f32>
    %s316 = stablehlo.reduce(%143 init: %s315) applies stablehlo.add across dimensions = [1] : (tensor<6x32xf32>, tensor<f32>) -> tensor<6xf32>
    %s317 = stablehlo.constant dense<32.0> : tensor<6xf32>
    %s318 = stablehlo.divide %s316, %s317 : tensor<6xf32>
    %144 = stablehlo.broadcast_in_dim %s318, dims = [0] : (tensor<6xf32>) -> tensor<6x1xf32>
    %145 = stablehlo.add %144, %60 : tensor<6x1xf32>
    %146 = stablehlo.rsqrt %145 : tensor<6x1xf32>
    %s319 = stablehlo.broadcast_in_dim %146, dims = [0, 1] : (tensor<6x1xf32>) -> tensor<6x32xf32>
    %147 = stablehlo.multiply %142, %s319 : tensor<6x32xf32>
    %148 = stablehlo.reshape %28 : (tensor<32xf32>) -> tensor<1x32xf32>
    %149 = stablehlo.broadcast_in_dim %148, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<6x32xf32>
    %150 = stablehlo.multiply %147, %149 : tensor<6x32xf32>
    %151 = stablehlo.dot_general %150, %29, contracting_dims = [1] x [0] : (tensor<6x32xf32>, tensor<32x48xf32>) -> tensor<6x48xf32>
    %152 = stablehlo.dot_general %150, %30, contracting_dims = [1] x [0] : (tensor<6x32xf32>, tensor<32x48xf32>) -> tensor<6x48xf32>
    %s320 = stablehlo.logistic %151 : tensor<6x48xf32>
    %153 = stablehlo.multiply %151, %s320 : tensor<6x48xf32>
    %154 = stablehlo.multiply %153, %152 : tensor<6x48xf32>
    %155 = stablehlo.dot_general %154, %31, contracting_dims = [1] x [0] : (tensor<6x48xf32>, tensor<48x32xf32>) -> tensor<6x32xf32>
    %156 = stablehlo.add %142, %155 : tensor<6x32xf32>
    %157 = stablehlo.multiply %156, %156 : tensor<6x32xf32>
    %s321 = stablehlo.constant dense<0.0> : tensor<f32>
    %s322 = stablehlo.reduce(%157 init: %s321) applies stablehlo.add across dimensions = [1] : (tensor<6x32xf32>, tensor<f32>) -> tensor<6xf32>
    %s323 = stablehlo.constant dense<32.0> : tensor<6xf32>
    %s324 = stablehlo.divide %s322, %s323 : tensor<6xf32>
    %158 = stablehlo.broadcast_in_dim %s324, dims = [0] : (tensor<6xf32>) -> tensor<6x1xf32>
    %159 = stablehlo.add %158, %60 : tensor<6x1xf32>
    %160 = stablehlo.rsqrt %159 : tensor<6x1xf32>
    %s325 = stablehlo.broadcast_in_dim %160, dims = [0, 1] : (tensor<6x1xf32>) -> tensor<6x32xf32>
    %161 = stablehlo.multiply %156, %s325 : tensor<6x32xf32>
    %162 = stablehlo.reshape %32 : (tensor<32xf32>) -> tensor<1x32xf32>
    %163 = stablehlo.broadcast_in_dim %162, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<6x32xf32>
    %164 = stablehlo.multiply %161, %163 : tensor<6x32xf32>
    %165 = stablehlo.dot_general %164, %33, contracting_dims = [1] x [0] : (tensor<6x32xf32>, tensor<32x32xf32>) -> tensor<6x32xf32>
    %166 = stablehlo.dot_general %164, %34, contracting_dims = [1] x [0] : (tensor<6x32xf32>, tensor<32x16xf32>) -> tensor<6x16xf32>
    %167 = stablehlo.dot_general %164, %35, contracting_dims = [1] x [0] : (tensor<6x32xf32>, tensor<32x16xf32>) -> tensor<6x16xf32>
    %168 = stablehlo.reshape %165 : (tensor<6x32xf32>) -> tensor<6x4x8xf32>
    %169 = stablehlo.reshape %166 : (tensor<6x16xf32>) -> tensor<6x2x8xf32>
    %170 = stablehlo.reshape %167 : (tensor<6x16xf32>) -> tensor<6x2x8xf32>
    %171 = stablehlo.multiply %168, %49 : tensor<6x4x8xf32>
    %172 = stablehlo.slice %168 [0:6, 0:4, 4:8] : (tensor<6x4x8xf32>) -> tensor<6x4x4xf32>
    %173 = stablehlo.slice %168 [0:6, 0:4, 0:4] : (tensor<6x4x8xf32>) -> tensor<6x4x4xf32>
    %174 = stablehlo.negate %172 : tensor<6x4x4xf32>
    %175 = stablehlo.concatenate %174, %173, dim = 2 : (tensor<6x4x4xf32>, tensor<6x4x4xf32>) -> tensor<6x4x8xf32>
    %176 = stablehlo.multiply %175, %51 : tensor<6x4x8xf32>
    %177 = stablehlo.add %171, %176 : tensor<6x4x8xf32>
    %178 = stablehlo.multiply %169, %53 : tensor<6x2x8xf32>
    %179 = stablehlo.slice %169 [0:6, 0:2, 4:8] : (tensor<6x2x8xf32>) -> tensor<6x2x4xf32>
    %180 = stablehlo.slice %169 [0:6, 0:2, 0:4] : (tensor<6x2x8xf32>) -> tensor<6x2x4xf32>
    %181 = stablehlo.negate %179 : tensor<6x2x4xf32>
    %182 = stablehlo.concatenate %181, %180, dim = 2 : (tensor<6x2x4xf32>, tensor<6x2x4xf32>) -> tensor<6x2x8xf32>
    %183 = stablehlo.multiply %182, %55 : tensor<6x2x8xf32>
    %184 = stablehlo.add %178, %183 : tensor<6x2x8xf32>
    %s326 = stablehlo.reshape %11 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s327 = "stablehlo.scatter"(%s326, %4, %184) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s328: tensor<f32>, %s329: tensor<f32>):
       stablehlo.return %s329 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<6xi32>, tensor<6x2x8xf32>) -> tensor<260x2x8xf32>
    %185 = stablehlo.reshape %s327 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s330 = stablehlo.reshape %12 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s331 = "stablehlo.scatter"(%s330, %4, %170) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s332: tensor<f32>, %s333: tensor<f32>):
       stablehlo.return %s333 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<6xi32>, tensor<6x2x8xf32>) -> tensor<260x2x8xf32>
    %186 = stablehlo.reshape %s331 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s334 = "stablehlo.gather"(%185, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<1x4xi32>) -> tensor<1x4x4x2x8xf32>
    %s335 = stablehlo.reshape %s334 : (tensor<1x4x4x2x8xf32>) -> tensor<1x16x2x8xf32>
    %s336 = "stablehlo.gather"(%186, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<1x4xi32>) -> tensor<1x4x4x2x8xf32>
    %s337 = stablehlo.reshape %s336 : (tensor<1x4x4x2x8xf32>) -> tensor<1x16x2x8xf32>
    %s338 = stablehlo.reshape %177 : (tensor<6x4x8xf32>) -> tensor<1x6x2x2x8xf32>
    %s339 = stablehlo.dot_general %s338, %s335, batching_dims = [0, 2] x [0, 2], contracting_dims = [4] x [3], precision = [HIGHEST, HIGHEST] : (tensor<1x6x2x2x8xf32>, tensor<1x16x2x8xf32>) -> tensor<1x2x6x2x16xf32>
    %s340 = stablehlo.constant dense<0.35355338> : tensor<f32>
    %s341 = stablehlo.broadcast_in_dim %s340, dims = [] : (tensor<f32>) -> tensor<1x2x6x2x16xf32>
    %s342 = stablehlo.multiply %s339, %s341 : tensor<1x2x6x2x16xf32>
    %s343 = stablehlo.iota dim = 4 : tensor<1x2x6x2x16xi32>
    %s344 = stablehlo.reshape %57 : (tensor<6xi32>) -> tensor<1x6xi32>
    %s345 = stablehlo.broadcast_in_dim %s344, dims = [0, 2] : (tensor<1x6xi32>) -> tensor<1x2x6x2x16xi32>
    %s346 = stablehlo.compare LT, %s343, %s345, SIGNED : (tensor<1x2x6x2x16xi32>, tensor<1x2x6x2x16xi32>) -> tensor<1x2x6x2x16xi1>
    %s347 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s348 = stablehlo.broadcast_in_dim %s347, dims = [] : (tensor<f32>) -> tensor<1x2x6x2x16xf32>
    %s349 = stablehlo.select %s346, %s342, %s348 : tensor<1x2x6x2x16xi1>, tensor<1x2x6x2x16xf32>
    %s350 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s351 = stablehlo.reduce(%s349 init: %s350) applies stablehlo.maximum across dimensions = [4] : (tensor<1x2x6x2x16xf32>, tensor<f32>) -> tensor<1x2x6x2xf32>
    %s352 = stablehlo.broadcast_in_dim %s351, dims = [0, 1, 2, 3] : (tensor<1x2x6x2xf32>) -> tensor<1x2x6x2x16xf32>
    %s353 = stablehlo.subtract %s349, %s352 : tensor<1x2x6x2x16xf32>
    %s354 = stablehlo.exponential %s353 : tensor<1x2x6x2x16xf32>
    %s355 = stablehlo.constant dense<0.0> : tensor<f32>
    %s356 = stablehlo.reduce(%s354 init: %s355) applies stablehlo.add across dimensions = [4] : (tensor<1x2x6x2x16xf32>, tensor<f32>) -> tensor<1x2x6x2xf32>
    %s357 = stablehlo.broadcast_in_dim %s356, dims = [0, 1, 2, 3] : (tensor<1x2x6x2xf32>) -> tensor<1x2x6x2x16xf32>
    %s358 = stablehlo.divide %s354, %s357 : tensor<1x2x6x2x16xf32>
    %s359 = stablehlo.dot_general %s358, %s337, batching_dims = [0, 1] x [0, 2], contracting_dims = [4] x [1], precision = [HIGHEST, HIGHEST] : (tensor<1x2x6x2x16xf32>, tensor<1x16x2x8xf32>) -> tensor<1x2x6x2x8xf32>
    %s360 = stablehlo.transpose %s359, dims = [0, 2, 1, 3, 4] : (tensor<1x2x6x2x8xf32>) -> tensor<1x6x2x2x8xf32>
    %187 = stablehlo.reshape %s360 : (tensor<1x6x2x2x8xf32>) -> tensor<6x4x8xf32>
    %188 = stablehlo.reshape %187 : (tensor<6x4x8xf32>) -> tensor<6x32xf32>
    %189 = stablehlo.dot_general %188, %36, contracting_dims = [1] x [0] : (tensor<6x32xf32>, tensor<32x32xf32>) -> tensor<6x32xf32>
    %190 = stablehlo.add %156, %189 : tensor<6x32xf32>
    %191 = stablehlo.multiply %190, %190 : tensor<6x32xf32>
    %s361 = stablehlo.constant dense<0.0> : tensor<f32>
    %s362 = stablehlo.reduce(%191 init: %s361) applies stablehlo.add across dimensions = [1] : (tensor<6x32xf32>, tensor<f32>) -> tensor<6xf32>
    %s363 = stablehlo.constant dense<32.0> : tensor<6xf32>
    %s364 = stablehlo.divide %s362, %s363 : tensor<6xf32>
    %192 = stablehlo.broadcast_in_dim %s364, dims = [0] : (tensor<6xf32>) -> tensor<6x1xf32>
    %193 = stablehlo.add %192, %60 : tensor<6x1xf32>
    %194 = stablehlo.rsqrt %193 : tensor<6x1xf32>
    %s365 = stablehlo.broadcast_in_dim %194, dims = [0, 1] : (tensor<6x1xf32>) -> tensor<6x32xf32>
    %195 = stablehlo.multiply %190, %s365 : tensor<6x32xf32>
    %196 = stablehlo.reshape %37 : (tensor<32xf32>) -> tensor<1x32xf32>
    %197 = stablehlo.broadcast_in_dim %196, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<6x32xf32>
    %198 = stablehlo.multiply %195, %197 : tensor<6x32xf32>
    %199 = stablehlo.dot_general %198, %38, contracting_dims = [1] x [0] : (tensor<6x32xf32>, tensor<32x48xf32>) -> tensor<6x48xf32>
    %200 = stablehlo.dot_general %198, %39, contracting_dims = [1] x [0] : (tensor<6x32xf32>, tensor<32x48xf32>) -> tensor<6x48xf32>
    %s366 = stablehlo.logistic %199 : tensor<6x48xf32>
    %201 = stablehlo.multiply %199, %s366 : tensor<6x48xf32>
    %202 = stablehlo.multiply %201, %200 : tensor<6x48xf32>
    %203 = stablehlo.dot_general %202, %40, contracting_dims = [1] x [0] : (tensor<6x48xf32>, tensor<48x32xf32>) -> tensor<6x32xf32>
    %204 = stablehlo.add %190, %203 : tensor<6x32xf32>
    %205 = stablehlo.reshape %204 : (tensor<6x32xf32>) -> tensor<1x6x32xf32>
    %206 = stablehlo.slice %205 [0:1, 5:6, 0:32] : (tensor<1x6x32xf32>) -> tensor<1x1x32xf32>
    %207 = stablehlo.reshape %206 : (tensor<1x1x32xf32>) -> tensor<1x32xf32>
    %208 = stablehlo.constant dense<[[1.0E-6]]> : tensor<1x1xf32>
    %209 = stablehlo.multiply %207, %207 : tensor<1x32xf32>
    %s367 = stablehlo.constant dense<0.0> : tensor<f32>
    %s368 = stablehlo.reduce(%209 init: %s367) applies stablehlo.add across dimensions = [1] : (tensor<1x32xf32>, tensor<f32>) -> tensor<1xf32>
    %s369 = stablehlo.constant dense<32.0> : tensor<1xf32>
    %s370 = stablehlo.divide %s368, %s369 : tensor<1xf32>
    %210 = stablehlo.broadcast_in_dim %s370, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %211 = stablehlo.add %210, %208 : tensor<1x1xf32>
    %212 = stablehlo.rsqrt %211 : tensor<1x1xf32>
    %s371 = stablehlo.broadcast_in_dim %212, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x32xf32>
    %213 = stablehlo.multiply %207, %s371 : tensor<1x32xf32>
    %214 = stablehlo.reshape %41 : (tensor<32xf32>) -> tensor<1x32xf32>
    %215 = stablehlo.broadcast_in_dim %214, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32xf32>
    %216 = stablehlo.multiply %213, %215 : tensor<1x32xf32>
    %217 = stablehlo.dot_general %216, %42, contracting_dims = [1] x [0] : (tensor<1x32xf32>, tensor<32x64xf32>) -> tensor<1x64xf32>
    %218 = stablehlo.reshape %217 : (tensor<1x64xf32>) -> tensor<1x1x64xf32>
    return %218, %89, %90, %137, %138, %185, %186 : tensor<1x1x64xf32>, tensor<17x4x2x8xf32>, tensor<17x4x2x8xf32>, tensor<17x4x2x8xf32>, tensor<17x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>
  }
}
