module {
  func.func @main(%0: tensor<1x16xi32>, %1: tensor<1x16xi32>, %2: tensor<1x4xi32>, %3: tensor<1xi32>, %4: tensor<16xi32>, %5: tensor<65x4x2x8xf32> {tf.aliasing_output = 1 : i32}, %6: tensor<65x4x2x8xf32> {tf.aliasing_output = 2 : i32}, %7: tensor<65x4x2x8xf32> {tf.aliasing_output = 3 : i32}, %8: tensor<65x4x2x8xf32> {tf.aliasing_output = 4 : i32}, %9: tensor<65x4x2x8xf32> {tf.aliasing_output = 5 : i32}, %10: tensor<65x4x2x8xf32> {tf.aliasing_output = 6 : i32}, %11: tensor<64x32xf32>, %12: tensor<32xf32>, %13: tensor<32x32xf32>, %14: tensor<32x16xf32>, %15: tensor<32x16xf32>, %16: tensor<32x32xf32>, %17: tensor<32xf32>, %18: tensor<32x48xf32>, %19: tensor<32x48xf32>, %20: tensor<48x32xf32>, %21: tensor<32xf32>, %22: tensor<32x32xf32>, %23: tensor<32x16xf32>, %24: tensor<32x16xf32>, %25: tensor<32x32xf32>, %26: tensor<32xf32>, %27: tensor<32x48xf32>, %28: tensor<32x48xf32>, %29: tensor<48x32xf32>, %30: tensor<32xf32>, %31: tensor<32x32xf32>, %32: tensor<32x16xf32>, %33: tensor<32x16xf32>, %34: tensor<32x32xf32>, %35: tensor<32xf32>, %36: tensor<32x48xf32>, %37: tensor<32x48xf32>, %38: tensor<48x32xf32>, %39: tensor<32xf32>, %40: tensor<32x64xf32>) -> (tensor<1x1x64xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>) {
    %41 = stablehlo.reshape %1 : (tensor<1x16xi32>) -> tensor<16xi32>
    %42 = stablehlo.constant dense<[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.5403023, 0.9950042, 0.99995, 0.9999995, 0.5403023, 0.9950042, 0.99995, 0.9999995], [-0.41614684, 0.9800666, 0.9998, 0.999998, -0.41614684, 0.9800666, 0.9998, 0.999998], [-0.9899925, 0.9553365, 0.99955004, 0.9999955, -0.9899925, 0.9553365, 0.99955004, 0.9999955], [-0.6536436, 0.921061, 0.9992001, 0.999992, -0.6536436, 0.921061, 0.9992001, 0.999992], [0.2836622, 0.87758255, 0.99875027, 0.9999875, 0.2836622, 0.87758255, 0.99875027, 0.9999875], [0.96017027, 0.8253356, 0.99820054, 0.999982, 0.96017027, 0.8253356, 0.99820054, 0.999982], [0.75390226, 0.7648422, 0.997551, 0.9999755, 0.75390226, 0.7648422, 0.997551, 0.9999755], [-0.14550003, 0.6967067, 0.99680173, 0.999968, -0.14550003, 0.6967067, 0.99680173, 0.999968], [-0.91113025, 0.62161, 0.9959527, 0.9999595, -0.91113025, 0.62161, 0.9959527, 0.9999595], [-0.8390715, 0.5403023, 0.9950042, 0.99995, -0.8390715, 0.5403023, 0.9950042, 0.99995], [0.004425698, 0.45359612, 0.9939561, 0.9999395, 0.004425698, 0.45359612, 0.9939561, 0.9999395], [0.84385395, 0.36235777, 0.99280864, 0.999928, 0.84385395, 0.36235777, 0.99280864, 0.999928], [0.9074468, 0.26749882, 0.9915619, 0.9999155, 0.9074468, 0.26749882, 0.9915619, 0.9999155], [0.13673721, 0.16996714, 0.990216, 0.999902, 0.13673721, 0.16996714, 0.990216, 0.999902], [-0.7596879, 0.0707372, 0.9887711, 0.9998875, -0.7596879, 0.0707372, 0.9887711, 0.9998875]]> : tensor<16x8xf32>
    %43 = stablehlo.constant dense<[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.84147096, 0.099833414, 0.009999833, 9.999998E-4, 0.84147096, 0.099833414, 0.009999833, 9.999998E-4], [0.9092974, 0.19866933, 0.019998666, 0.0019999987, 0.9092974, 0.19866933, 0.019998666, 0.0019999987], [0.14112, 0.29552022, 0.029995501, 0.0029999956, 0.14112, 0.29552022, 0.029995501, 0.0029999956], [-0.7568025, 0.38941833, 0.039989334, 0.0039999895, -0.7568025, 0.38941833, 0.039989334, 0.0039999895], [-0.9589243, 0.47942555, 0.04997917, 0.0049999794, -0.9589243, 0.47942555, 0.04997917, 0.0049999794], [-0.2794155, 0.5646425, 0.059964005, 0.005999964, -0.2794155, 0.5646425, 0.059964005, 0.005999964], [0.6569866, 0.64421767, 0.06994285, 0.006999943, 0.6569866, 0.64421767, 0.06994285, 0.006999943], [0.98935825, 0.7173561, 0.0799147, 0.007999915, 0.98935825, 0.7173561, 0.0799147, 0.007999915], [0.4121185, 0.7833269, 0.08987855, 0.008999879, 0.4121185, 0.7833269, 0.08987855, 0.008999879], [-0.5440211, 0.84147096, 0.099833414, 0.009999833, -0.5440211, 0.84147096, 0.099833414, 0.009999833], [-0.9999902, 0.89120734, 0.1097783, 0.010999778, -0.9999902, 0.89120734, 0.1097783, 0.010999778], [-0.53657293, 0.9320391, 0.119712204, 0.011999712, -0.53657293, 0.9320391, 0.119712204, 0.011999712], [0.42016703, 0.9635582, 0.12963414, 0.012999634, 0.42016703, 0.9635582, 0.12963414, 0.012999634], [0.9906074, 0.98544973, 0.13954312, 0.013999542, 0.9906074, 0.98544973, 0.13954312, 0.013999542], [0.65028787, 0.997495, 0.14943813, 0.014999437, 0.65028787, 0.997495, 0.14943813, 0.014999437]]> : tensor<16x8xf32>
    %44 = "stablehlo.gather"(%42, %41) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 8>}> : (tensor<16x8xf32>, tensor<16xi32>) -> tensor<16x8xf32>
    %45 = "stablehlo.gather"(%43, %41) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 8>}> : (tensor<16x8xf32>, tensor<16xi32>) -> tensor<16x8xf32>
    %46 = stablehlo.reshape %44 : (tensor<16x8xf32>) -> tensor<16x1x8xf32>
    %47 = stablehlo.broadcast_in_dim %46, dims = [0, 1, 2] : (tensor<16x1x8xf32>) -> tensor<16x4x8xf32>
    %48 = stablehlo.reshape %45 : (tensor<16x8xf32>) -> tensor<16x1x8xf32>
    %49 = stablehlo.broadcast_in_dim %48, dims = [0, 1, 2] : (tensor<16x1x8xf32>) -> tensor<16x4x8xf32>
    %50 = stablehlo.reshape %44 : (tensor<16x8xf32>) -> tensor<16x1x8xf32>
    %51 = stablehlo.broadcast_in_dim %50, dims = [0, 1, 2] : (tensor<16x1x8xf32>) -> tensor<16x2x8xf32>
    %52 = stablehlo.reshape %45 : (tensor<16x8xf32>) -> tensor<16x1x8xf32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [0, 1, 2] : (tensor<16x1x8xf32>) -> tensor<16x2x8xf32>
    %54 = stablehlo.constant dense<1> : tensor<16xi32>
    %55 = stablehlo.add %41, %54 : tensor<16xi32>
    %56 = "stablehlo.gather"(%11, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 32>}> : (tensor<64x32xf32>, tensor<1x16xi32>) -> tensor<1x16x32xf32>
    %57 = stablehlo.reshape %56 : (tensor<1x16x32xf32>) -> tensor<16x32xf32>
    %58 = stablehlo.constant dense<[[1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6]]> : tensor<16x1xf32>
    %59 = stablehlo.multiply %57, %57 : tensor<16x32xf32>
    %s217 = stablehlo.constant dense<0.0> : tensor<f32>
    %s218 = stablehlo.reduce(%59 init: %s217) applies stablehlo.add across dimensions = [1] : (tensor<16x32xf32>, tensor<f32>) -> tensor<16xf32>
    %s219 = stablehlo.constant dense<32.0> : tensor<16xf32>
    %s220 = stablehlo.divide %s218, %s219 : tensor<16xf32>
    %60 = stablehlo.broadcast_in_dim %s220, dims = [0] : (tensor<16xf32>) -> tensor<16x1xf32>
    %61 = stablehlo.add %60, %58 : tensor<16x1xf32>
    %62 = stablehlo.rsqrt %61 : tensor<16x1xf32>
    %s221 = stablehlo.broadcast_in_dim %62, dims = [0, 1] : (tensor<16x1xf32>) -> tensor<16x32xf32>
    %63 = stablehlo.multiply %57, %s221 : tensor<16x32xf32>
    %64 = stablehlo.reshape %12 : (tensor<32xf32>) -> tensor<1x32xf32>
    %65 = stablehlo.broadcast_in_dim %64, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<16x32xf32>
    %66 = stablehlo.multiply %63, %65 : tensor<16x32xf32>
    %67 = stablehlo.dot_general %66, %13, contracting_dims = [1] x [0] : (tensor<16x32xf32>, tensor<32x32xf32>) -> tensor<16x32xf32>
    %68 = stablehlo.dot_general %66, %14, contracting_dims = [1] x [0] : (tensor<16x32xf32>, tensor<32x16xf32>) -> tensor<16x16xf32>
    %69 = stablehlo.dot_general %66, %15, contracting_dims = [1] x [0] : (tensor<16x32xf32>, tensor<32x16xf32>) -> tensor<16x16xf32>
    %70 = stablehlo.reshape %67 : (tensor<16x32xf32>) -> tensor<16x4x8xf32>
    %71 = stablehlo.reshape %68 : (tensor<16x16xf32>) -> tensor<16x2x8xf32>
    %72 = stablehlo.reshape %69 : (tensor<16x16xf32>) -> tensor<16x2x8xf32>
    %73 = stablehlo.multiply %70, %47 : tensor<16x4x8xf32>
    %74 = stablehlo.slice %70 [0:16, 0:4, 4:8] : (tensor<16x4x8xf32>) -> tensor<16x4x4xf32>
    %75 = stablehlo.slice %70 [0:16, 0:4, 0:4] : (tensor<16x4x8xf32>) -> tensor<16x4x4xf32>
    %76 = stablehlo.negate %74 : tensor<16x4x4xf32>
    %77 = stablehlo.concatenate %76, %75, dim = 2 : (tensor<16x4x4xf32>, tensor<16x4x4xf32>) -> tensor<16x4x8xf32>
    %78 = stablehlo.multiply %77, %49 : tensor<16x4x8xf32>
    %79 = stablehlo.add %73, %78 : tensor<16x4x8xf32>
    %80 = stablehlo.multiply %71, %51 : tensor<16x2x8xf32>
    %81 = stablehlo.slice %71 [0:16, 0:2, 4:8] : (tensor<16x2x8xf32>) -> tensor<16x2x4xf32>
    %82 = stablehlo.slice %71 [0:16, 0:2, 0:4] : (tensor<16x2x8xf32>) -> tensor<16x2x4xf32>
    %83 = stablehlo.negate %81 : tensor<16x2x4xf32>
    %84 = stablehlo.concatenate %83, %82, dim = 2 : (tensor<16x2x4xf32>, tensor<16x2x4xf32>) -> tensor<16x2x8xf32>
    %85 = stablehlo.multiply %84, %53 : tensor<16x2x8xf32>
    %86 = stablehlo.add %80, %85 : tensor<16x2x8xf32>
    %s222 = stablehlo.reshape %5 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s223 = "stablehlo.scatter"(%s222, %4, %86) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s224: tensor<f32>, %s225: tensor<f32>):
       stablehlo.return %s225 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<16xi32>, tensor<16x2x8xf32>) -> tensor<260x2x8xf32>
    %87 = stablehlo.reshape %s223 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s226 = stablehlo.reshape %6 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s227 = "stablehlo.scatter"(%s226, %4, %72) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s228: tensor<f32>, %s229: tensor<f32>):
       stablehlo.return %s229 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<16xi32>, tensor<16x2x8xf32>) -> tensor<260x2x8xf32>
    %88 = stablehlo.reshape %s227 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s230 = "stablehlo.gather"(%87, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<1x4xi32>) -> tensor<1x4x4x2x8xf32>
    %s231 = stablehlo.reshape %s230 : (tensor<1x4x4x2x8xf32>) -> tensor<1x16x2x8xf32>
    %s232 = "stablehlo.gather"(%88, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<1x4xi32>) -> tensor<1x4x4x2x8xf32>
    %s233 = stablehlo.reshape %s232 : (tensor<1x4x4x2x8xf32>) -> tensor<1x16x2x8xf32>
    %s234 = stablehlo.reshape %79 : (tensor<16x4x8xf32>) -> tensor<1x16x2x2x8xf32>
    %s235 = stablehlo.dot_general %s234, %s231, batching_dims = [0, 2] x [0, 2], contracting_dims = [4] x [3], precision = [HIGHEST, HIGHEST] : (tensor<1x16x2x2x8xf32>, tensor<1x16x2x8xf32>) -> tensor<1x2x16x2x16xf32>
    %s236 = stablehlo.constant dense<0.35355338> : tensor<f32>
    %s237 = stablehlo.broadcast_in_dim %s236, dims = [] : (tensor<f32>) -> tensor<1x2x16x2x16xf32>
    %s238 = stablehlo.multiply %s235, %s237 : tensor<1x2x16x2x16xf32>
    %s239 = stablehlo.iota dim = 4 : tensor<1x2x16x2x16xi32>
    %s240 = stablehlo.reshape %55 : (tensor<16xi32>) -> tensor<1x16xi32>
    %s241 = stablehlo.broadcast_in_dim %s240, dims = [0, 2] : (tensor<1x16xi32>) -> tensor<1x2x16x2x16xi32>
    %s243 = stablehlo.compare LT, %s239, %s241, SIGNED : (tensor<1x2x16x2x16xi32>, tensor<1x2x16x2x16xi32>) -> tensor<1x2x16x2x16xi1>
    %s244 = stablehlo.constant dense<8> : tensor<i32>
    %s245 = stablehlo.broadcast_in_dim %s244, dims = [] : (tensor<i32>) -> tensor<1x2x16x2x16xi32>
    %s246 = stablehlo.subtract %s241, %s245 : tensor<1x2x16x2x16xi32>
    %s247 = stablehlo.compare GE, %s239, %s246, SIGNED : (tensor<1x2x16x2x16xi32>, tensor<1x2x16x2x16xi32>) -> tensor<1x2x16x2x16xi1>
    %s242 = stablehlo.and %s243, %s247 : tensor<1x2x16x2x16xi1>
    %s248 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s249 = stablehlo.broadcast_in_dim %s248, dims = [] : (tensor<f32>) -> tensor<1x2x16x2x16xf32>
    %s250 = stablehlo.select %s242, %s238, %s249 : tensor<1x2x16x2x16xi1>, tensor<1x2x16x2x16xf32>
    %s251 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s252 = stablehlo.reduce(%s250 init: %s251) applies stablehlo.maximum across dimensions = [4] : (tensor<1x2x16x2x16xf32>, tensor<f32>) -> tensor<1x2x16x2xf32>
    %s253 = stablehlo.broadcast_in_dim %s252, dims = [0, 1, 2, 3] : (tensor<1x2x16x2xf32>) -> tensor<1x2x16x2x16xf32>
    %s254 = stablehlo.subtract %s250, %s253 : tensor<1x2x16x2x16xf32>
    %s255 = stablehlo.exponential %s254 : tensor<1x2x16x2x16xf32>
    %s256 = stablehlo.constant dense<0.0> : tensor<f32>
    %s257 = stablehlo.reduce(%s255 init: %s256) applies stablehlo.add across dimensions = [4] : (tensor<1x2x16x2x16xf32>, tensor<f32>) -> tensor<1x2x16x2xf32>
    %s258 = stablehlo.broadcast_in_dim %s257, dims = [0, 1, 2, 3] : (tensor<1x2x16x2xf32>) -> tensor<1x2x16x2x16xf32>
    %s259 = stablehlo.divide %s255, %s258 : tensor<1x2x16x2x16xf32>
    %s260 = stablehlo.dot_general %s259, %s233, batching_dims = [0, 1] x [0, 2], contracting_dims = [4] x [1], precision = [HIGHEST, HIGHEST] : (tensor<1x2x16x2x16xf32>, tensor<1x16x2x8xf32>) -> tensor<1x2x16x2x8xf32>
    %s261 = stablehlo.transpose %s260, dims = [0, 2, 1, 3, 4] : (tensor<1x2x16x2x8xf32>) -> tensor<1x16x2x2x8xf32>
    %89 = stablehlo.reshape %s261 : (tensor<1x16x2x2x8xf32>) -> tensor<16x4x8xf32>
    %90 = stablehlo.reshape %89 : (tensor<16x4x8xf32>) -> tensor<16x32xf32>
    %91 = stablehlo.dot_general %90, %16, contracting_dims = [1] x [0] : (tensor<16x32xf32>, tensor<32x32xf32>) -> tensor<16x32xf32>
    %92 = stablehlo.add %57, %91 : tensor<16x32xf32>
    %93 = stablehlo.multiply %92, %92 : tensor<16x32xf32>
    %s262 = stablehlo.constant dense<0.0> : tensor<f32>
    %s263 = stablehlo.reduce(%93 init: %s262) applies stablehlo.add across dimensions = [1] : (tensor<16x32xf32>, tensor<f32>) -> tensor<16xf32>
    %s264 = stablehlo.constant dense<32.0> : tensor<16xf32>
    %s265 = stablehlo.divide %s263, %s264 : tensor<16xf32>
    %94 = stablehlo.broadcast_in_dim %s265, dims = [0] : (tensor<16xf32>) -> tensor<16x1xf32>
    %95 = stablehlo.add %94, %58 : tensor<16x1xf32>
    %96 = stablehlo.rsqrt %95 : tensor<16x1xf32>
    %s266 = stablehlo.broadcast_in_dim %96, dims = [0, 1] : (tensor<16x1xf32>) -> tensor<16x32xf32>
    %97 = stablehlo.multiply %92, %s266 : tensor<16x32xf32>
    %98 = stablehlo.reshape %17 : (tensor<32xf32>) -> tensor<1x32xf32>
    %99 = stablehlo.broadcast_in_dim %98, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<16x32xf32>
    %100 = stablehlo.multiply %97, %99 : tensor<16x32xf32>
    %101 = stablehlo.dot_general %100, %18, contracting_dims = [1] x [0] : (tensor<16x32xf32>, tensor<32x48xf32>) -> tensor<16x48xf32>
    %102 = stablehlo.dot_general %100, %19, contracting_dims = [1] x [0] : (tensor<16x32xf32>, tensor<32x48xf32>) -> tensor<16x48xf32>
    %s267 = stablehlo.logistic %101 : tensor<16x48xf32>
    %103 = stablehlo.multiply %101, %s267 : tensor<16x48xf32>
    %104 = stablehlo.multiply %103, %102 : tensor<16x48xf32>
    %105 = stablehlo.dot_general %104, %20, contracting_dims = [1] x [0] : (tensor<16x48xf32>, tensor<48x32xf32>) -> tensor<16x32xf32>
    %106 = stablehlo.add %92, %105 : tensor<16x32xf32>
    %107 = stablehlo.multiply %106, %106 : tensor<16x32xf32>
    %s268 = stablehlo.constant dense<0.0> : tensor<f32>
    %s269 = stablehlo.reduce(%107 init: %s268) applies stablehlo.add across dimensions = [1] : (tensor<16x32xf32>, tensor<f32>) -> tensor<16xf32>
    %s270 = stablehlo.constant dense<32.0> : tensor<16xf32>
    %s271 = stablehlo.divide %s269, %s270 : tensor<16xf32>
    %108 = stablehlo.broadcast_in_dim %s271, dims = [0] : (tensor<16xf32>) -> tensor<16x1xf32>
    %109 = stablehlo.add %108, %58 : tensor<16x1xf32>
    %110 = stablehlo.rsqrt %109 : tensor<16x1xf32>
    %s272 = stablehlo.broadcast_in_dim %110, dims = [0, 1] : (tensor<16x1xf32>) -> tensor<16x32xf32>
    %111 = stablehlo.multiply %106, %s272 : tensor<16x32xf32>
    %112 = stablehlo.reshape %21 : (tensor<32xf32>) -> tensor<1x32xf32>
    %113 = stablehlo.broadcast_in_dim %112, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<16x32xf32>
    %114 = stablehlo.multiply %111, %113 : tensor<16x32xf32>
    %115 = stablehlo.dot_general %114, %22, contracting_dims = [1] x [0] : (tensor<16x32xf32>, tensor<32x32xf32>) -> tensor<16x32xf32>
    %116 = stablehlo.dot_general %114, %23, contracting_dims = [1] x [0] : (tensor<16x32xf32>, tensor<32x16xf32>) -> tensor<16x16xf32>
    %117 = stablehlo.dot_general %114, %24, contracting_dims = [1] x [0] : (tensor<16x32xf32>, tensor<32x16xf32>) -> tensor<16x16xf32>
    %118 = stablehlo.reshape %115 : (tensor<16x32xf32>) -> tensor<16x4x8xf32>
    %119 = stablehlo.reshape %116 : (tensor<16x16xf32>) -> tensor<16x2x8xf32>
    %120 = stablehlo.reshape %117 : (tensor<16x16xf32>) -> tensor<16x2x8xf32>
    %121 = stablehlo.multiply %118, %47 : tensor<16x4x8xf32>
    %122 = stablehlo.slice %118 [0:16, 0:4, 4:8] : (tensor<16x4x8xf32>) -> tensor<16x4x4xf32>
    %123 = stablehlo.slice %118 [0:16, 0:4, 0:4] : (tensor<16x4x8xf32>) -> tensor<16x4x4xf32>
    %124 = stablehlo.negate %122 : tensor<16x4x4xf32>
    %125 = stablehlo.concatenate %124, %123, dim = 2 : (tensor<16x4x4xf32>, tensor<16x4x4xf32>) -> tensor<16x4x8xf32>
    %126 = stablehlo.multiply %125, %49 : tensor<16x4x8xf32>
    %127 = stablehlo.add %121, %126 : tensor<16x4x8xf32>
    %128 = stablehlo.multiply %119, %51 : tensor<16x2x8xf32>
    %129 = stablehlo.slice %119 [0:16, 0:2, 4:8] : (tensor<16x2x8xf32>) -> tensor<16x2x4xf32>
    %130 = stablehlo.slice %119 [0:16, 0:2, 0:4] : (tensor<16x2x8xf32>) -> tensor<16x2x4xf32>
    %131 = stablehlo.negate %129 : tensor<16x2x4xf32>
    %132 = stablehlo.concatenate %131, %130, dim = 2 : (tensor<16x2x4xf32>, tensor<16x2x4xf32>) -> tensor<16x2x8xf32>
    %133 = stablehlo.multiply %132, %53 : tensor<16x2x8xf32>
    %134 = stablehlo.add %128, %133 : tensor<16x2x8xf32>
    %s273 = stablehlo.reshape %7 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s274 = "stablehlo.scatter"(%s273, %4, %134) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s275: tensor<f32>, %s276: tensor<f32>):
       stablehlo.return %s276 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<16xi32>, tensor<16x2x8xf32>) -> tensor<260x2x8xf32>
    %135 = stablehlo.reshape %s274 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s277 = stablehlo.reshape %8 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s278 = "stablehlo.scatter"(%s277, %4, %120) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s279: tensor<f32>, %s280: tensor<f32>):
       stablehlo.return %s280 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<16xi32>, tensor<16x2x8xf32>) -> tensor<260x2x8xf32>
    %136 = stablehlo.reshape %s278 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s281 = "stablehlo.gather"(%135, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<1x4xi32>) -> tensor<1x4x4x2x8xf32>
    %s282 = stablehlo.reshape %s281 : (tensor<1x4x4x2x8xf32>) -> tensor<1x16x2x8xf32>
    %s283 = "stablehlo.gather"(%136, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<1x4xi32>) -> tensor<1x4x4x2x8xf32>
    %s284 = stablehlo.reshape %s283 : (tensor<1x4x4x2x8xf32>) -> tensor<1x16x2x8xf32>
    %s285 = stablehlo.reshape %127 : (tensor<16x4x8xf32>) -> tensor<1x16x2x2x8xf32>
    %s286 = stablehlo.dot_general %s285, %s282, batching_dims = [0, 2] x [0, 2], contracting_dims = [4] x [3], precision = [HIGHEST, HIGHEST] : (tensor<1x16x2x2x8xf32>, tensor<1x16x2x8xf32>) -> tensor<1x2x16x2x16xf32>
    %s287 = stablehlo.constant dense<0.35355338> : tensor<f32>
    %s288 = stablehlo.broadcast_in_dim %s287, dims = [] : (tensor<f32>) -> tensor<1x2x16x2x16xf32>
    %s289 = stablehlo.multiply %s286, %s288 : tensor<1x2x16x2x16xf32>
    %s290 = stablehlo.iota dim = 4 : tensor<1x2x16x2x16xi32>
    %s291 = stablehlo.reshape %55 : (tensor<16xi32>) -> tensor<1x16xi32>
    %s292 = stablehlo.broadcast_in_dim %s291, dims = [0, 2] : (tensor<1x16xi32>) -> tensor<1x2x16x2x16xi32>
    %s294 = stablehlo.compare LT, %s290, %s292, SIGNED : (tensor<1x2x16x2x16xi32>, tensor<1x2x16x2x16xi32>) -> tensor<1x2x16x2x16xi1>
    %s295 = stablehlo.constant dense<8> : tensor<i32>
    %s296 = stablehlo.broadcast_in_dim %s295, dims = [] : (tensor<i32>) -> tensor<1x2x16x2x16xi32>
    %s297 = stablehlo.subtract %s292, %s296 : tensor<1x2x16x2x16xi32>
    %s298 = stablehlo.compare GE, %s290, %s297, SIGNED : (tensor<1x2x16x2x16xi32>, tensor<1x2x16x2x16xi32>) -> tensor<1x2x16x2x16xi1>
    %s293 = stablehlo.and %s294, %s298 : tensor<1x2x16x2x16xi1>
    %s299 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s300 = stablehlo.broadcast_in_dim %s299, dims = [] : (tensor<f32>) -> tensor<1x2x16x2x16xf32>
    %s301 = stablehlo.select %s293, %s289, %s300 : tensor<1x2x16x2x16xi1>, tensor<1x2x16x2x16xf32>
    %s302 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s303 = stablehlo.reduce(%s301 init: %s302) applies stablehlo.maximum across dimensions = [4] : (tensor<1x2x16x2x16xf32>, tensor<f32>) -> tensor<1x2x16x2xf32>
    %s304 = stablehlo.broadcast_in_dim %s303, dims = [0, 1, 2, 3] : (tensor<1x2x16x2xf32>) -> tensor<1x2x16x2x16xf32>
    %s305 = stablehlo.subtract %s301, %s304 : tensor<1x2x16x2x16xf32>
    %s306 = stablehlo.exponential %s305 : tensor<1x2x16x2x16xf32>
    %s307 = stablehlo.constant dense<0.0> : tensor<f32>
    %s308 = stablehlo.reduce(%s306 init: %s307) applies stablehlo.add across dimensions = [4] : (tensor<1x2x16x2x16xf32>, tensor<f32>) -> tensor<1x2x16x2xf32>
    %s309 = stablehlo.broadcast_in_dim %s308, dims = [0, 1, 2, 3] : (tensor<1x2x16x2xf32>) -> tensor<1x2x16x2x16xf32>
    %s310 = stablehlo.divide %s306, %s309 : tensor<1x2x16x2x16xf32>
    %s311 = stablehlo.dot_general %s310, %s284, batching_dims = [0, 1] x [0, 2], contracting_dims = [4] x [1], precision = [HIGHEST, HIGHEST] : (tensor<1x2x16x2x16xf32>, tensor<1x16x2x8xf32>) -> tensor<1x2x16x2x8xf32>
    %s312 = stablehlo.transpose %s311, dims = [0, 2, 1, 3, 4] : (tensor<1x2x16x2x8xf32>) -> tensor<1x16x2x2x8xf32>
    %137 = stablehlo.reshape %s312 : (tensor<1x16x2x2x8xf32>) -> tensor<16x4x8xf32>
    %138 = stablehlo.reshape %137 : (tensor<16x4x8xf32>) -> tensor<16x32xf32>
    %139 = stablehlo.dot_general %138, %25, contracting_dims = [1] x [0] : (tensor<16x32xf32>, tensor<32x32xf32>) -> tensor<16x32xf32>
    %140 = stablehlo.add %106, %139 : tensor<16x32xf32>
    %141 = stablehlo.multiply %140, %140 : tensor<16x32xf32>
    %s313 = stablehlo.constant dense<0.0> : tensor<f32>
    %s314 = stablehlo.reduce(%141 init: %s313) applies stablehlo.add across dimensions = [1] : (tensor<16x32xf32>, tensor<f32>) -> tensor<16xf32>
    %s315 = stablehlo.constant dense<32.0> : tensor<16xf32>
    %s316 = stablehlo.divide %s314, %s315 : tensor<16xf32>
    %142 = stablehlo.broadcast_in_dim %s316, dims = [0] : (tensor<16xf32>) -> tensor<16x1xf32>
    %143 = stablehlo.add %142, %58 : tensor<16x1xf32>
    %144 = stablehlo.rsqrt %143 : tensor<16x1xf32>
    %s317 = stablehlo.broadcast_in_dim %144, dims = [0, 1] : (tensor<16x1xf32>) -> tensor<16x32xf32>
    %145 = stablehlo.multiply %140, %s317 : tensor<16x32xf32>
    %146 = stablehlo.reshape %26 : (tensor<32xf32>) -> tensor<1x32xf32>
    %147 = stablehlo.broadcast_in_dim %146, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<16x32xf32>
    %148 = stablehlo.multiply %145, %147 : tensor<16x32xf32>
    %149 = stablehlo.dot_general %148, %27, contracting_dims = [1] x [0] : (tensor<16x32xf32>, tensor<32x48xf32>) -> tensor<16x48xf32>
    %150 = stablehlo.dot_general %148, %28, contracting_dims = [1] x [0] : (tensor<16x32xf32>, tensor<32x48xf32>) -> tensor<16x48xf32>
    %s318 = stablehlo.logistic %149 : tensor<16x48xf32>
    %151 = stablehlo.multiply %149, %s318 : tensor<16x48xf32>
    %152 = stablehlo.multiply %151, %150 : tensor<16x48xf32>
    %153 = stablehlo.dot_general %152, %29, contracting_dims = [1] x [0] : (tensor<16x48xf32>, tensor<48x32xf32>) -> tensor<16x32xf32>
    %154 = stablehlo.add %140, %153 : tensor<16x32xf32>
    %155 = stablehlo.multiply %154, %154 : tensor<16x32xf32>
    %s319 = stablehlo.constant dense<0.0> : tensor<f32>
    %s320 = stablehlo.reduce(%155 init: %s319) applies stablehlo.add across dimensions = [1] : (tensor<16x32xf32>, tensor<f32>) -> tensor<16xf32>
    %s321 = stablehlo.constant dense<32.0> : tensor<16xf32>
    %s322 = stablehlo.divide %s320, %s321 : tensor<16xf32>
    %156 = stablehlo.broadcast_in_dim %s322, dims = [0] : (tensor<16xf32>) -> tensor<16x1xf32>
    %157 = stablehlo.add %156, %58 : tensor<16x1xf32>
    %158 = stablehlo.rsqrt %157 : tensor<16x1xf32>
    %s323 = stablehlo.broadcast_in_dim %158, dims = [0, 1] : (tensor<16x1xf32>) -> tensor<16x32xf32>
    %159 = stablehlo.multiply %154, %s323 : tensor<16x32xf32>
    %160 = stablehlo.reshape %30 : (tensor<32xf32>) -> tensor<1x32xf32>
    %161 = stablehlo.broadcast_in_dim %160, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<16x32xf32>
    %162 = stablehlo.multiply %159, %161 : tensor<16x32xf32>
    %163 = stablehlo.dot_general %162, %31, contracting_dims = [1] x [0] : (tensor<16x32xf32>, tensor<32x32xf32>) -> tensor<16x32xf32>
    %164 = stablehlo.dot_general %162, %32, contracting_dims = [1] x [0] : (tensor<16x32xf32>, tensor<32x16xf32>) -> tensor<16x16xf32>
    %165 = stablehlo.dot_general %162, %33, contracting_dims = [1] x [0] : (tensor<16x32xf32>, tensor<32x16xf32>) -> tensor<16x16xf32>
    %166 = stablehlo.reshape %163 : (tensor<16x32xf32>) -> tensor<16x4x8xf32>
    %167 = stablehlo.reshape %164 : (tensor<16x16xf32>) -> tensor<16x2x8xf32>
    %168 = stablehlo.reshape %165 : (tensor<16x16xf32>) -> tensor<16x2x8xf32>
    %169 = stablehlo.multiply %166, %47 : tensor<16x4x8xf32>
    %170 = stablehlo.slice %166 [0:16, 0:4, 4:8] : (tensor<16x4x8xf32>) -> tensor<16x4x4xf32>
    %171 = stablehlo.slice %166 [0:16, 0:4, 0:4] : (tensor<16x4x8xf32>) -> tensor<16x4x4xf32>
    %172 = stablehlo.negate %170 : tensor<16x4x4xf32>
    %173 = stablehlo.concatenate %172, %171, dim = 2 : (tensor<16x4x4xf32>, tensor<16x4x4xf32>) -> tensor<16x4x8xf32>
    %174 = stablehlo.multiply %173, %49 : tensor<16x4x8xf32>
    %175 = stablehlo.add %169, %174 : tensor<16x4x8xf32>
    %176 = stablehlo.multiply %167, %51 : tensor<16x2x8xf32>
    %177 = stablehlo.slice %167 [0:16, 0:2, 4:8] : (tensor<16x2x8xf32>) -> tensor<16x2x4xf32>
    %178 = stablehlo.slice %167 [0:16, 0:2, 0:4] : (tensor<16x2x8xf32>) -> tensor<16x2x4xf32>
    %179 = stablehlo.negate %177 : tensor<16x2x4xf32>
    %180 = stablehlo.concatenate %179, %178, dim = 2 : (tensor<16x2x4xf32>, tensor<16x2x4xf32>) -> tensor<16x2x8xf32>
    %181 = stablehlo.multiply %180, %53 : tensor<16x2x8xf32>
    %182 = stablehlo.add %176, %181 : tensor<16x2x8xf32>
    %s324 = stablehlo.reshape %9 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s325 = "stablehlo.scatter"(%s324, %4, %182) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s326: tensor<f32>, %s327: tensor<f32>):
       stablehlo.return %s327 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<16xi32>, tensor<16x2x8xf32>) -> tensor<260x2x8xf32>
    %183 = stablehlo.reshape %s325 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s328 = stablehlo.reshape %10 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s329 = "stablehlo.scatter"(%s328, %4, %168) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s330: tensor<f32>, %s331: tensor<f32>):
       stablehlo.return %s331 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<16xi32>, tensor<16x2x8xf32>) -> tensor<260x2x8xf32>
    %184 = stablehlo.reshape %s329 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s332 = "stablehlo.gather"(%183, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<1x4xi32>) -> tensor<1x4x4x2x8xf32>
    %s333 = stablehlo.reshape %s332 : (tensor<1x4x4x2x8xf32>) -> tensor<1x16x2x8xf32>
    %s334 = "stablehlo.gather"(%184, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<1x4xi32>) -> tensor<1x4x4x2x8xf32>
    %s335 = stablehlo.reshape %s334 : (tensor<1x4x4x2x8xf32>) -> tensor<1x16x2x8xf32>
    %s336 = stablehlo.reshape %175 : (tensor<16x4x8xf32>) -> tensor<1x16x2x2x8xf32>
    %s337 = stablehlo.dot_general %s336, %s333, batching_dims = [0, 2] x [0, 2], contracting_dims = [4] x [3], precision = [HIGHEST, HIGHEST] : (tensor<1x16x2x2x8xf32>, tensor<1x16x2x8xf32>) -> tensor<1x2x16x2x16xf32>
    %s338 = stablehlo.constant dense<0.35355338> : tensor<f32>
    %s339 = stablehlo.broadcast_in_dim %s338, dims = [] : (tensor<f32>) -> tensor<1x2x16x2x16xf32>
    %s340 = stablehlo.multiply %s337, %s339 : tensor<1x2x16x2x16xf32>
    %s341 = stablehlo.iota dim = 4 : tensor<1x2x16x2x16xi32>
    %s342 = stablehlo.reshape %55 : (tensor<16xi32>) -> tensor<1x16xi32>
    %s343 = stablehlo.broadcast_in_dim %s342, dims = [0, 2] : (tensor<1x16xi32>) -> tensor<1x2x16x2x16xi32>
    %s344 = stablehlo.compare LT, %s341, %s343, SIGNED : (tensor<1x2x16x2x16xi32>, tensor<1x2x16x2x16xi32>) -> tensor<1x2x16x2x16xi1>
    %s345 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s346 = stablehlo.broadcast_in_dim %s345, dims = [] : (tensor<f32>) -> tensor<1x2x16x2x16xf32>
    %s347 = stablehlo.select %s344, %s340, %s346 : tensor<1x2x16x2x16xi1>, tensor<1x2x16x2x16xf32>
    %s348 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s349 = stablehlo.reduce(%s347 init: %s348) applies stablehlo.maximum across dimensions = [4] : (tensor<1x2x16x2x16xf32>, tensor<f32>) -> tensor<1x2x16x2xf32>
    %s350 = stablehlo.broadcast_in_dim %s349, dims = [0, 1, 2, 3] : (tensor<1x2x16x2xf32>) -> tensor<1x2x16x2x16xf32>
    %s351 = stablehlo.subtract %s347, %s350 : tensor<1x2x16x2x16xf32>
    %s352 = stablehlo.exponential %s351 : tensor<1x2x16x2x16xf32>
    %s353 = stablehlo.constant dense<0.0> : tensor<f32>
    %s354 = stablehlo.reduce(%s352 init: %s353) applies stablehlo.add across dimensions = [4] : (tensor<1x2x16x2x16xf32>, tensor<f32>) -> tensor<1x2x16x2xf32>
    %s355 = stablehlo.broadcast_in_dim %s354, dims = [0, 1, 2, 3] : (tensor<1x2x16x2xf32>) -> tensor<1x2x16x2x16xf32>
    %s356 = stablehlo.divide %s352, %s355 : tensor<1x2x16x2x16xf32>
    %s357 = stablehlo.dot_general %s356, %s335, batching_dims = [0, 1] x [0, 2], contracting_dims = [4] x [1], precision = [HIGHEST, HIGHEST] : (tensor<1x2x16x2x16xf32>, tensor<1x16x2x8xf32>) -> tensor<1x2x16x2x8xf32>
    %s358 = stablehlo.transpose %s357, dims = [0, 2, 1, 3, 4] : (tensor<1x2x16x2x8xf32>) -> tensor<1x16x2x2x8xf32>
    %185 = stablehlo.reshape %s358 : (tensor<1x16x2x2x8xf32>) -> tensor<16x4x8xf32>
    %186 = stablehlo.reshape %185 : (tensor<16x4x8xf32>) -> tensor<16x32xf32>
    %187 = stablehlo.dot_general %186, %34, contracting_dims = [1] x [0] : (tensor<16x32xf32>, tensor<32x32xf32>) -> tensor<16x32xf32>
    %188 = stablehlo.add %154, %187 : tensor<16x32xf32>
    %189 = stablehlo.multiply %188, %188 : tensor<16x32xf32>
    %s359 = stablehlo.constant dense<0.0> : tensor<f32>
    %s360 = stablehlo.reduce(%189 init: %s359) applies stablehlo.add across dimensions = [1] : (tensor<16x32xf32>, tensor<f32>) -> tensor<16xf32>
    %s361 = stablehlo.constant dense<32.0> : tensor<16xf32>
    %s362 = stablehlo.divide %s360, %s361 : tensor<16xf32>
    %190 = stablehlo.broadcast_in_dim %s362, dims = [0] : (tensor<16xf32>) -> tensor<16x1xf32>
    %191 = stablehlo.add %190, %58 : tensor<16x1xf32>
    %192 = stablehlo.rsqrt %191 : tensor<16x1xf32>
    %s363 = stablehlo.broadcast_in_dim %192, dims = [0, 1] : (tensor<16x1xf32>) -> tensor<16x32xf32>
    %193 = stablehlo.multiply %188, %s363 : tensor<16x32xf32>
    %194 = stablehlo.reshape %35 : (tensor<32xf32>) -> tensor<1x32xf32>
    %195 = stablehlo.broadcast_in_dim %194, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<16x32xf32>
    %196 = stablehlo.multiply %193, %195 : tensor<16x32xf32>
    %197 = stablehlo.dot_general %196, %36, contracting_dims = [1] x [0] : (tensor<16x32xf32>, tensor<32x48xf32>) -> tensor<16x48xf32>
    %198 = stablehlo.dot_general %196, %37, contracting_dims = [1] x [0] : (tensor<16x32xf32>, tensor<32x48xf32>) -> tensor<16x48xf32>
    %s364 = stablehlo.logistic %197 : tensor<16x48xf32>
    %199 = stablehlo.multiply %197, %s364 : tensor<16x48xf32>
    %200 = stablehlo.multiply %199, %198 : tensor<16x48xf32>
    %201 = stablehlo.dot_general %200, %38, contracting_dims = [1] x [0] : (tensor<16x48xf32>, tensor<48x32xf32>) -> tensor<16x32xf32>
    %202 = stablehlo.add %188, %201 : tensor<16x32xf32>
    %203 = stablehlo.reshape %202 : (tensor<16x32xf32>) -> tensor<1x16x32xf32>
    %204 = stablehlo.slice %203 [0:1, 15:16, 0:32] : (tensor<1x16x32xf32>) -> tensor<1x1x32xf32>
    %205 = stablehlo.reshape %204 : (tensor<1x1x32xf32>) -> tensor<1x32xf32>
    %206 = stablehlo.constant dense<[[1.0E-6]]> : tensor<1x1xf32>
    %207 = stablehlo.multiply %205, %205 : tensor<1x32xf32>
    %s365 = stablehlo.constant dense<0.0> : tensor<f32>
    %s366 = stablehlo.reduce(%207 init: %s365) applies stablehlo.add across dimensions = [1] : (tensor<1x32xf32>, tensor<f32>) -> tensor<1xf32>
    %s367 = stablehlo.constant dense<32.0> : tensor<1xf32>
    %s368 = stablehlo.divide %s366, %s367 : tensor<1xf32>
    %208 = stablehlo.broadcast_in_dim %s368, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %209 = stablehlo.add %208, %206 : tensor<1x1xf32>
    %210 = stablehlo.rsqrt %209 : tensor<1x1xf32>
    %s369 = stablehlo.broadcast_in_dim %210, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x32xf32>
    %211 = stablehlo.multiply %205, %s369 : tensor<1x32xf32>
    %212 = stablehlo.reshape %39 : (tensor<32xf32>) -> tensor<1x32xf32>
    %213 = stablehlo.broadcast_in_dim %212, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32xf32>
    %214 = stablehlo.multiply %211, %213 : tensor<1x32xf32>
    %215 = stablehlo.dot_general %214, %40, contracting_dims = [1] x [0] : (tensor<1x32xf32>, tensor<32x64xf32>) -> tensor<1x64xf32>
    %216 = stablehlo.reshape %215 : (tensor<1x64xf32>) -> tensor<1x1x64xf32>
    return %216, %87, %88, %135, %136, %183, %184 : tensor<1x1x64xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>
  }
}
