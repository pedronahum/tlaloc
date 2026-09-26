module {
  func.func @main(%0: tensor<1x32xi32>, %1: tensor<1x32xi32>, %2: tensor<1x8xi32>, %3: tensor<1xi32>, %4: tensor<32xi32>, %5: tensor<1x8xi32>, %6: tensor<32xi32>, %7: tensor<13x4x2x8xf32> {tf.aliasing_output = 1 : i32}, %8: tensor<13x4x2x8xf32> {tf.aliasing_output = 2 : i32}, %9: tensor<13x4x2x8xf32> {tf.aliasing_output = 3 : i32}, %10: tensor<13x4x2x8xf32> {tf.aliasing_output = 4 : i32}, %11: tensor<65x4x2x8xf32> {tf.aliasing_output = 5 : i32}, %12: tensor<65x4x2x8xf32> {tf.aliasing_output = 6 : i32}, %13: tensor<64x32xf32>, %14: tensor<32xf32>, %15: tensor<32x32xf32>, %16: tensor<32x16xf32>, %17: tensor<32x16xf32>, %18: tensor<32x32xf32>, %19: tensor<32xf32>, %20: tensor<32x48xf32>, %21: tensor<32x48xf32>, %22: tensor<48x32xf32>, %23: tensor<32xf32>, %24: tensor<32x32xf32>, %25: tensor<32x16xf32>, %26: tensor<32x16xf32>, %27: tensor<32x32xf32>, %28: tensor<32xf32>, %29: tensor<32x48xf32>, %30: tensor<32x48xf32>, %31: tensor<48x32xf32>, %32: tensor<32xf32>, %33: tensor<32x32xf32>, %34: tensor<32x16xf32>, %35: tensor<32x16xf32>, %36: tensor<32x32xf32>, %37: tensor<32xf32>, %38: tensor<32x48xf32>, %39: tensor<32x48xf32>, %40: tensor<48x32xf32>, %41: tensor<32xf32>, %42: tensor<32x64xf32>) -> (tensor<1x1x64xf32>, tensor<13x4x2x8xf32>, tensor<13x4x2x8xf32>, tensor<13x4x2x8xf32>, tensor<13x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>) {
    %43 = stablehlo.reshape %1 : (tensor<1x32xi32>) -> tensor<32xi32>
    %44 = stablehlo.constant dense<[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.5403023, 0.9950042, 0.99995, 0.9999995, 0.5403023, 0.9950042, 0.99995, 0.9999995], [-0.41614684, 0.9800666, 0.9998, 0.999998, -0.41614684, 0.9800666, 0.9998, 0.999998], [-0.9899925, 0.9553365, 0.99955004, 0.9999955, -0.9899925, 0.9553365, 0.99955004, 0.9999955], [-0.6536436, 0.921061, 0.9992001, 0.999992, -0.6536436, 0.921061, 0.9992001, 0.999992], [0.2836622, 0.87758255, 0.99875027, 0.9999875, 0.2836622, 0.87758255, 0.99875027, 0.9999875], [0.96017027, 0.8253356, 0.99820054, 0.999982, 0.96017027, 0.8253356, 0.99820054, 0.999982], [0.75390226, 0.7648422, 0.997551, 0.9999755, 0.75390226, 0.7648422, 0.997551, 0.9999755], [-0.14550003, 0.6967067, 0.99680173, 0.999968, -0.14550003, 0.6967067, 0.99680173, 0.999968], [-0.91113025, 0.62161, 0.9959527, 0.9999595, -0.91113025, 0.62161, 0.9959527, 0.9999595], [-0.8390715, 0.5403023, 0.9950042, 0.99995, -0.8390715, 0.5403023, 0.9950042, 0.99995], [0.004425698, 0.45359612, 0.9939561, 0.9999395, 0.004425698, 0.45359612, 0.9939561, 0.9999395], [0.84385395, 0.36235777, 0.99280864, 0.999928, 0.84385395, 0.36235777, 0.99280864, 0.999928], [0.9074468, 0.26749882, 0.9915619, 0.9999155, 0.9074468, 0.26749882, 0.9915619, 0.9999155], [0.13673721, 0.16996714, 0.990216, 0.999902, 0.13673721, 0.16996714, 0.990216, 0.999902], [-0.7596879, 0.0707372, 0.9887711, 0.9998875, -0.7596879, 0.0707372, 0.9887711, 0.9998875], [-0.9576595, -0.029199522, 0.98722726, 0.999872, -0.9576595, -0.029199522, 0.98722726, 0.999872], [-0.27516335, -0.1288445, 0.9855848, 0.9998555, -0.27516335, -0.1288445, 0.9855848, 0.9998555], [0.6603167, -0.22720209, 0.9838437, 0.999838, 0.6603167, -0.22720209, 0.9838437, 0.999838], [0.9887046, -0.32328957, 0.9820042, 0.9998195, 0.9887046, -0.32328957, 0.9820042, 0.9998195], [0.40808207, -0.41614684, 0.9800666, 0.9998, 0.40808207, -0.41614684, 0.9800666, 0.9998], [-0.54772925, -0.5048461, 0.9780309, 0.9997795, -0.54772925, -0.5048461, 0.9780309, 0.9997795], [-0.99996084, -0.5885011, 0.97589743, 0.999758, -0.99996084, -0.5885011, 0.97589743, 0.999758], [-0.53283304, -0.66627604, 0.97366637, 0.99973553, -0.53283304, -0.66627604, 0.97366637, 0.99973553], [0.42417902, -0.73739374, 0.971338, 0.999712, 0.42417902, -0.73739374, 0.971338, 0.999712], [0.99120283, -0.8011436, 0.9689124, 0.9996875, 0.99120283, -0.8011436, 0.9689124, 0.9996875], [0.6469193, -0.8568888, 0.96638995, 0.99966204, 0.6469193, -0.8568888, 0.96638995, 0.99966204], [-0.29213881, -0.90407217, 0.9637709, 0.9996355, -0.29213881, -0.90407217, 0.9637709, 0.9996355], [-0.9626059, -0.94222236, 0.96105546, 0.99960804, -0.9626059, -0.94222236, 0.96105546, 0.99960804], [-0.74805754, -0.9709582, 0.95824385, 0.99957955, -0.74805754, -0.9709582, 0.95824385, 0.99957955], [0.15425146, -0.9899925, 0.9553365, 0.99955004, 0.15425146, -0.9899925, 0.9553365, 0.99955004], [0.91474235, -0.99913514, 0.95233357, 0.9995195, 0.91474235, -0.99913514, 0.95233357, 0.9995195]]> : tensor<32x8xf32>
    %45 = stablehlo.constant dense<[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.84147096, 0.099833414, 0.009999833, 9.999998E-4, 0.84147096, 0.099833414, 0.009999833, 9.999998E-4], [0.9092974, 0.19866933, 0.019998666, 0.0019999987, 0.9092974, 0.19866933, 0.019998666, 0.0019999987], [0.14112, 0.29552022, 0.029995501, 0.0029999956, 0.14112, 0.29552022, 0.029995501, 0.0029999956], [-0.7568025, 0.38941833, 0.039989334, 0.0039999895, -0.7568025, 0.38941833, 0.039989334, 0.0039999895], [-0.9589243, 0.47942555, 0.04997917, 0.0049999794, -0.9589243, 0.47942555, 0.04997917, 0.0049999794], [-0.2794155, 0.5646425, 0.059964005, 0.005999964, -0.2794155, 0.5646425, 0.059964005, 0.005999964], [0.6569866, 0.64421767, 0.06994285, 0.006999943, 0.6569866, 0.64421767, 0.06994285, 0.006999943], [0.98935825, 0.7173561, 0.0799147, 0.007999915, 0.98935825, 0.7173561, 0.0799147, 0.007999915], [0.4121185, 0.7833269, 0.08987855, 0.008999879, 0.4121185, 0.7833269, 0.08987855, 0.008999879], [-0.5440211, 0.84147096, 0.099833414, 0.009999833, -0.5440211, 0.84147096, 0.099833414, 0.009999833], [-0.9999902, 0.89120734, 0.1097783, 0.010999778, -0.9999902, 0.89120734, 0.1097783, 0.010999778], [-0.53657293, 0.9320391, 0.119712204, 0.011999712, -0.53657293, 0.9320391, 0.119712204, 0.011999712], [0.42016703, 0.9635582, 0.12963414, 0.012999634, 0.42016703, 0.9635582, 0.12963414, 0.012999634], [0.9906074, 0.98544973, 0.13954312, 0.013999542, 0.9906074, 0.98544973, 0.13954312, 0.013999542], [0.65028787, 0.997495, 0.14943813, 0.014999437, 0.65028787, 0.997495, 0.14943813, 0.014999437], [-0.2879033, 0.9995736, 0.15931821, 0.015999317, -0.2879033, 0.9995736, 0.15931821, 0.015999317], [-0.96139747, 0.9916648, 0.16918235, 0.016999181, -0.96139747, 0.9916648, 0.16918235, 0.016999181], [-0.75098723, 0.9738476, 0.17902957, 0.017999029, -0.75098723, 0.9738476, 0.17902957, 0.017999029], [0.1498772, 0.9463001, 0.1888589, 0.018998858, 0.1498772, 0.9463001, 0.1888589, 0.018998858], [0.9129453, 0.9092974, 0.19866933, 0.019998666, 0.9129453, 0.9092974, 0.19866933, 0.019998666], [0.8366556, 0.86320937, 0.2084599, 0.020998457, 0.8366556, 0.86320937, 0.2084599, 0.020998457], [-0.008851309, 0.8084964, 0.21822962, 0.021998225, -0.008851309, 0.8084964, 0.21822962, 0.021998225], [-0.84622043, 0.7457052, 0.22797753, 0.022997972, -0.84622043, 0.7457052, 0.22797753, 0.022997972], [-0.9055784, 0.6754632, 0.23770262, 0.023997696, -0.9055784, 0.6754632, 0.23770262, 0.023997696], [-0.13235176, 0.5984721, 0.24740396, 0.024997396, -0.13235176, 0.5984721, 0.24740396, 0.024997396], [0.76255846, 0.5155014, 0.25708055, 0.02599707, 0.76255846, 0.5155014, 0.25708055, 0.02599707], [0.95637596, 0.42737988, 0.26673144, 0.026996719, 0.95637596, 0.42737988, 0.26673144, 0.026996719], [0.2709058, 0.33498815, 0.27635565, 0.02799634, 0.2709058, 0.33498815, 0.27635565, 0.02799634], [-0.6636339, 0.23924933, 0.2859522, 0.028995935, -0.6636339, 0.23924933, 0.2859522, 0.028995935], [-0.9880316, 0.14112, 0.29552022, 0.029995501, -0.9880316, 0.14112, 0.29552022, 0.029995501], [-0.40403765, 0.041580662, 0.30505863, 0.030995036, -0.40403765, 0.041580662, 0.30505863, 0.030995036]]> : tensor<32x8xf32>
    %46 = "stablehlo.gather"(%44, %43) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 8>}> : (tensor<32x8xf32>, tensor<32xi32>) -> tensor<32x8xf32>
    %47 = "stablehlo.gather"(%45, %43) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 8>}> : (tensor<32x8xf32>, tensor<32xi32>) -> tensor<32x8xf32>
    %48 = stablehlo.reshape %46 : (tensor<32x8xf32>) -> tensor<32x1x8xf32>
    %49 = stablehlo.broadcast_in_dim %48, dims = [0, 1, 2] : (tensor<32x1x8xf32>) -> tensor<32x4x8xf32>
    %50 = stablehlo.reshape %47 : (tensor<32x8xf32>) -> tensor<32x1x8xf32>
    %51 = stablehlo.broadcast_in_dim %50, dims = [0, 1, 2] : (tensor<32x1x8xf32>) -> tensor<32x4x8xf32>
    %52 = stablehlo.reshape %46 : (tensor<32x8xf32>) -> tensor<32x1x8xf32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [0, 1, 2] : (tensor<32x1x8xf32>) -> tensor<32x2x8xf32>
    %54 = stablehlo.reshape %47 : (tensor<32x8xf32>) -> tensor<32x1x8xf32>
    %55 = stablehlo.broadcast_in_dim %54, dims = [0, 1, 2] : (tensor<32x1x8xf32>) -> tensor<32x2x8xf32>
    %56 = stablehlo.slice %5 [0:1, 0:3] : (tensor<1x8xi32>) -> tensor<1x3xi32>
    %57 = stablehlo.constant dense<1> : tensor<32xi32>
    %58 = stablehlo.add %43, %57 : tensor<32xi32>
    %59 = "stablehlo.gather"(%13, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 32>}> : (tensor<64x32xf32>, tensor<1x32xi32>) -> tensor<1x32x32xf32>
    %60 = stablehlo.reshape %59 : (tensor<1x32x32xf32>) -> tensor<32x32xf32>
    %61 = stablehlo.constant dense<[[1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6]]> : tensor<32x1xf32>
    %62 = stablehlo.multiply %60, %60 : tensor<32x32xf32>
    %s220 = stablehlo.constant dense<0.0> : tensor<f32>
    %s221 = stablehlo.reduce(%62 init: %s220) applies stablehlo.add across dimensions = [1] : (tensor<32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %s222 = stablehlo.constant dense<32.0> : tensor<32xf32>
    %s223 = stablehlo.divide %s221, %s222 : tensor<32xf32>
    %63 = stablehlo.broadcast_in_dim %s223, dims = [0] : (tensor<32xf32>) -> tensor<32x1xf32>
    %64 = stablehlo.add %63, %61 : tensor<32x1xf32>
    %65 = stablehlo.rsqrt %64 : tensor<32x1xf32>
    %s224 = stablehlo.broadcast_in_dim %65, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x32xf32>
    %66 = stablehlo.multiply %60, %s224 : tensor<32x32xf32>
    %67 = stablehlo.reshape %14 : (tensor<32xf32>) -> tensor<1x32xf32>
    %68 = stablehlo.broadcast_in_dim %67, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<32x32xf32>
    %69 = stablehlo.multiply %66, %68 : tensor<32x32xf32>
    %70 = stablehlo.dot_general %69, %15, contracting_dims = [1] x [0] : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %71 = stablehlo.dot_general %69, %16, contracting_dims = [1] x [0] : (tensor<32x32xf32>, tensor<32x16xf32>) -> tensor<32x16xf32>
    %72 = stablehlo.dot_general %69, %17, contracting_dims = [1] x [0] : (tensor<32x32xf32>, tensor<32x16xf32>) -> tensor<32x16xf32>
    %73 = stablehlo.reshape %70 : (tensor<32x32xf32>) -> tensor<32x4x8xf32>
    %74 = stablehlo.reshape %71 : (tensor<32x16xf32>) -> tensor<32x2x8xf32>
    %75 = stablehlo.reshape %72 : (tensor<32x16xf32>) -> tensor<32x2x8xf32>
    %76 = stablehlo.multiply %73, %49 : tensor<32x4x8xf32>
    %77 = stablehlo.slice %73 [0:32, 0:4, 4:8] : (tensor<32x4x8xf32>) -> tensor<32x4x4xf32>
    %78 = stablehlo.slice %73 [0:32, 0:4, 0:4] : (tensor<32x4x8xf32>) -> tensor<32x4x4xf32>
    %79 = stablehlo.negate %77 : tensor<32x4x4xf32>
    %80 = stablehlo.concatenate %79, %78, dim = 2 : (tensor<32x4x4xf32>, tensor<32x4x4xf32>) -> tensor<32x4x8xf32>
    %81 = stablehlo.multiply %80, %51 : tensor<32x4x8xf32>
    %82 = stablehlo.add %76, %81 : tensor<32x4x8xf32>
    %83 = stablehlo.multiply %74, %53 : tensor<32x2x8xf32>
    %84 = stablehlo.slice %74 [0:32, 0:2, 4:8] : (tensor<32x2x8xf32>) -> tensor<32x2x4xf32>
    %85 = stablehlo.slice %74 [0:32, 0:2, 0:4] : (tensor<32x2x8xf32>) -> tensor<32x2x4xf32>
    %86 = stablehlo.negate %84 : tensor<32x2x4xf32>
    %87 = stablehlo.concatenate %86, %85, dim = 2 : (tensor<32x2x4xf32>, tensor<32x2x4xf32>) -> tensor<32x2x8xf32>
    %88 = stablehlo.multiply %87, %55 : tensor<32x2x8xf32>
    %89 = stablehlo.add %83, %88 : tensor<32x2x8xf32>
    %s225 = stablehlo.reshape %7 : (tensor<13x4x2x8xf32>) -> tensor<52x2x8xf32>
    %s226 = "stablehlo.scatter"(%s225, %6, %89) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s227: tensor<f32>, %s228: tensor<f32>):
       stablehlo.return %s228 : tensor<f32>
     }) : (tensor<52x2x8xf32>, tensor<32xi32>, tensor<32x2x8xf32>) -> tensor<52x2x8xf32>
    %90 = stablehlo.reshape %s226 : (tensor<52x2x8xf32>) -> tensor<13x4x2x8xf32>
    %s229 = stablehlo.reshape %8 : (tensor<13x4x2x8xf32>) -> tensor<52x2x8xf32>
    %s230 = "stablehlo.scatter"(%s229, %6, %75) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s231: tensor<f32>, %s232: tensor<f32>):
       stablehlo.return %s232 : tensor<f32>
     }) : (tensor<52x2x8xf32>, tensor<32xi32>, tensor<32x2x8xf32>) -> tensor<52x2x8xf32>
    %91 = stablehlo.reshape %s230 : (tensor<52x2x8xf32>) -> tensor<13x4x2x8xf32>
    %s233 = "stablehlo.gather"(%90, %56) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<13x4x2x8xf32>, tensor<1x3xi32>) -> tensor<1x3x4x2x8xf32>
    %s234 = stablehlo.reshape %s233 : (tensor<1x3x4x2x8xf32>) -> tensor<1x12x2x8xf32>
    %s235 = "stablehlo.gather"(%91, %56) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<13x4x2x8xf32>, tensor<1x3xi32>) -> tensor<1x3x4x2x8xf32>
    %s236 = stablehlo.reshape %s235 : (tensor<1x3x4x2x8xf32>) -> tensor<1x12x2x8xf32>
    %s237 = stablehlo.reshape %82 : (tensor<32x4x8xf32>) -> tensor<1x32x2x2x8xf32>
    %s238 = stablehlo.dot_general %s237, %s234, batching_dims = [0, 2] x [0, 2], contracting_dims = [4] x [3], algorithm = <lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false> : (tensor<1x32x2x2x8xf32>, tensor<1x12x2x8xf32>) -> tensor<1x2x32x2x12xf32>
    %s239 = stablehlo.constant dense<0.35355338> : tensor<f32>
    %s240 = stablehlo.broadcast_in_dim %s239, dims = [] : (tensor<f32>) -> tensor<1x2x32x2x12xf32>
    %s241 = stablehlo.multiply %s238, %s240 : tensor<1x2x32x2x12xf32>
    %s242 = stablehlo.iota dim = 4 : tensor<1x2x32x2x12xi32>
    %s243 = stablehlo.reshape %58 : (tensor<32xi32>) -> tensor<1x32xi32>
    %s244 = stablehlo.broadcast_in_dim %s243, dims = [0, 2] : (tensor<1x32xi32>) -> tensor<1x2x32x2x12xi32>
    %s248 = stablehlo.constant dense<11> : tensor<i32>
    %s249 = stablehlo.broadcast_in_dim %s248, dims = [] : (tensor<i32>) -> tensor<1x2x32x2x12xi32>
    %s245 = stablehlo.add %s244, %s249 : tensor<1x2x32x2x12xi32>
    %s246 = stablehlo.subtract %s245, %s242 : tensor<1x2x32x2x12xi32>
    %s250 = stablehlo.constant dense<12> : tensor<i32>
    %s251 = stablehlo.broadcast_in_dim %s250, dims = [] : (tensor<i32>) -> tensor<1x2x32x2x12xi32>
    %s247 = stablehlo.remainder %s246, %s251 : tensor<1x2x32x2x12xi32>
    %s252 = stablehlo.constant dense<8> : tensor<i32>
    %s253 = stablehlo.broadcast_in_dim %s252, dims = [] : (tensor<i32>) -> tensor<1x2x32x2x12xi32>
    %s254 = stablehlo.compare LT, %s247, %s253, SIGNED : (tensor<1x2x32x2x12xi32>, tensor<1x2x32x2x12xi32>) -> tensor<1x2x32x2x12xi1>
    %s255 = stablehlo.compare LT, %s247, %s244, SIGNED : (tensor<1x2x32x2x12xi32>, tensor<1x2x32x2x12xi32>) -> tensor<1x2x32x2x12xi1>
    %s256 = stablehlo.and %s254, %s255 : tensor<1x2x32x2x12xi1>
    %s257 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s258 = stablehlo.broadcast_in_dim %s257, dims = [] : (tensor<f32>) -> tensor<1x2x32x2x12xf32>
    %s259 = stablehlo.select %s256, %s241, %s258 : tensor<1x2x32x2x12xi1>, tensor<1x2x32x2x12xf32>
    %s260 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s261 = stablehlo.reduce(%s259 init: %s260) applies stablehlo.maximum across dimensions = [4] : (tensor<1x2x32x2x12xf32>, tensor<f32>) -> tensor<1x2x32x2xf32>
    %s262 = stablehlo.broadcast_in_dim %s261, dims = [0, 1, 2, 3] : (tensor<1x2x32x2xf32>) -> tensor<1x2x32x2x12xf32>
    %s263 = stablehlo.subtract %s259, %s262 : tensor<1x2x32x2x12xf32>
    %s264 = stablehlo.exponential %s263 : tensor<1x2x32x2x12xf32>
    %s265 = stablehlo.constant dense<0.0> : tensor<f32>
    %s266 = stablehlo.reduce(%s264 init: %s265) applies stablehlo.add across dimensions = [4] : (tensor<1x2x32x2x12xf32>, tensor<f32>) -> tensor<1x2x32x2xf32>
    %s267 = stablehlo.broadcast_in_dim %s266, dims = [0, 1, 2, 3] : (tensor<1x2x32x2xf32>) -> tensor<1x2x32x2x12xf32>
    %s268 = stablehlo.divide %s264, %s267 : tensor<1x2x32x2x12xf32>
    %s269 = stablehlo.dot_general %s268, %s236, batching_dims = [0, 1] x [0, 2], contracting_dims = [4] x [1], algorithm = <lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false> : (tensor<1x2x32x2x12xf32>, tensor<1x12x2x8xf32>) -> tensor<1x2x32x2x8xf32>
    %s270 = stablehlo.transpose %s269, dims = [0, 2, 1, 3, 4] : (tensor<1x2x32x2x8xf32>) -> tensor<1x32x2x2x8xf32>
    %92 = stablehlo.reshape %s270 : (tensor<1x32x2x2x8xf32>) -> tensor<32x4x8xf32>
    %93 = stablehlo.reshape %92 : (tensor<32x4x8xf32>) -> tensor<32x32xf32>
    %94 = stablehlo.dot_general %93, %18, contracting_dims = [1] x [0] : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %95 = stablehlo.add %60, %94 : tensor<32x32xf32>
    %96 = stablehlo.multiply %95, %95 : tensor<32x32xf32>
    %s271 = stablehlo.constant dense<0.0> : tensor<f32>
    %s272 = stablehlo.reduce(%96 init: %s271) applies stablehlo.add across dimensions = [1] : (tensor<32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %s273 = stablehlo.constant dense<32.0> : tensor<32xf32>
    %s274 = stablehlo.divide %s272, %s273 : tensor<32xf32>
    %97 = stablehlo.broadcast_in_dim %s274, dims = [0] : (tensor<32xf32>) -> tensor<32x1xf32>
    %98 = stablehlo.add %97, %61 : tensor<32x1xf32>
    %99 = stablehlo.rsqrt %98 : tensor<32x1xf32>
    %s275 = stablehlo.broadcast_in_dim %99, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x32xf32>
    %100 = stablehlo.multiply %95, %s275 : tensor<32x32xf32>
    %101 = stablehlo.reshape %19 : (tensor<32xf32>) -> tensor<1x32xf32>
    %102 = stablehlo.broadcast_in_dim %101, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<32x32xf32>
    %103 = stablehlo.multiply %100, %102 : tensor<32x32xf32>
    %104 = stablehlo.dot_general %103, %20, contracting_dims = [1] x [0] : (tensor<32x32xf32>, tensor<32x48xf32>) -> tensor<32x48xf32>
    %105 = stablehlo.dot_general %103, %21, contracting_dims = [1] x [0] : (tensor<32x32xf32>, tensor<32x48xf32>) -> tensor<32x48xf32>
    %s276 = stablehlo.logistic %104 : tensor<32x48xf32>
    %106 = stablehlo.multiply %104, %s276 : tensor<32x48xf32>
    %107 = stablehlo.multiply %106, %105 : tensor<32x48xf32>
    %108 = stablehlo.dot_general %107, %22, contracting_dims = [1] x [0] : (tensor<32x48xf32>, tensor<48x32xf32>) -> tensor<32x32xf32>
    %109 = stablehlo.add %95, %108 : tensor<32x32xf32>
    %110 = stablehlo.multiply %109, %109 : tensor<32x32xf32>
    %s277 = stablehlo.constant dense<0.0> : tensor<f32>
    %s278 = stablehlo.reduce(%110 init: %s277) applies stablehlo.add across dimensions = [1] : (tensor<32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %s279 = stablehlo.constant dense<32.0> : tensor<32xf32>
    %s280 = stablehlo.divide %s278, %s279 : tensor<32xf32>
    %111 = stablehlo.broadcast_in_dim %s280, dims = [0] : (tensor<32xf32>) -> tensor<32x1xf32>
    %112 = stablehlo.add %111, %61 : tensor<32x1xf32>
    %113 = stablehlo.rsqrt %112 : tensor<32x1xf32>
    %s281 = stablehlo.broadcast_in_dim %113, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x32xf32>
    %114 = stablehlo.multiply %109, %s281 : tensor<32x32xf32>
    %115 = stablehlo.reshape %23 : (tensor<32xf32>) -> tensor<1x32xf32>
    %116 = stablehlo.broadcast_in_dim %115, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<32x32xf32>
    %117 = stablehlo.multiply %114, %116 : tensor<32x32xf32>
    %118 = stablehlo.dot_general %117, %24, contracting_dims = [1] x [0] : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %119 = stablehlo.dot_general %117, %25, contracting_dims = [1] x [0] : (tensor<32x32xf32>, tensor<32x16xf32>) -> tensor<32x16xf32>
    %120 = stablehlo.dot_general %117, %26, contracting_dims = [1] x [0] : (tensor<32x32xf32>, tensor<32x16xf32>) -> tensor<32x16xf32>
    %121 = stablehlo.reshape %118 : (tensor<32x32xf32>) -> tensor<32x4x8xf32>
    %122 = stablehlo.reshape %119 : (tensor<32x16xf32>) -> tensor<32x2x8xf32>
    %123 = stablehlo.reshape %120 : (tensor<32x16xf32>) -> tensor<32x2x8xf32>
    %124 = stablehlo.multiply %121, %49 : tensor<32x4x8xf32>
    %125 = stablehlo.slice %121 [0:32, 0:4, 4:8] : (tensor<32x4x8xf32>) -> tensor<32x4x4xf32>
    %126 = stablehlo.slice %121 [0:32, 0:4, 0:4] : (tensor<32x4x8xf32>) -> tensor<32x4x4xf32>
    %127 = stablehlo.negate %125 : tensor<32x4x4xf32>
    %128 = stablehlo.concatenate %127, %126, dim = 2 : (tensor<32x4x4xf32>, tensor<32x4x4xf32>) -> tensor<32x4x8xf32>
    %129 = stablehlo.multiply %128, %51 : tensor<32x4x8xf32>
    %130 = stablehlo.add %124, %129 : tensor<32x4x8xf32>
    %131 = stablehlo.multiply %122, %53 : tensor<32x2x8xf32>
    %132 = stablehlo.slice %122 [0:32, 0:2, 4:8] : (tensor<32x2x8xf32>) -> tensor<32x2x4xf32>
    %133 = stablehlo.slice %122 [0:32, 0:2, 0:4] : (tensor<32x2x8xf32>) -> tensor<32x2x4xf32>
    %134 = stablehlo.negate %132 : tensor<32x2x4xf32>
    %135 = stablehlo.concatenate %134, %133, dim = 2 : (tensor<32x2x4xf32>, tensor<32x2x4xf32>) -> tensor<32x2x8xf32>
    %136 = stablehlo.multiply %135, %55 : tensor<32x2x8xf32>
    %137 = stablehlo.add %131, %136 : tensor<32x2x8xf32>
    %s282 = stablehlo.reshape %9 : (tensor<13x4x2x8xf32>) -> tensor<52x2x8xf32>
    %s283 = "stablehlo.scatter"(%s282, %6, %137) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s284: tensor<f32>, %s285: tensor<f32>):
       stablehlo.return %s285 : tensor<f32>
     }) : (tensor<52x2x8xf32>, tensor<32xi32>, tensor<32x2x8xf32>) -> tensor<52x2x8xf32>
    %138 = stablehlo.reshape %s283 : (tensor<52x2x8xf32>) -> tensor<13x4x2x8xf32>
    %s286 = stablehlo.reshape %10 : (tensor<13x4x2x8xf32>) -> tensor<52x2x8xf32>
    %s287 = "stablehlo.scatter"(%s286, %6, %123) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s288: tensor<f32>, %s289: tensor<f32>):
       stablehlo.return %s289 : tensor<f32>
     }) : (tensor<52x2x8xf32>, tensor<32xi32>, tensor<32x2x8xf32>) -> tensor<52x2x8xf32>
    %139 = stablehlo.reshape %s287 : (tensor<52x2x8xf32>) -> tensor<13x4x2x8xf32>
    %s290 = "stablehlo.gather"(%138, %56) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<13x4x2x8xf32>, tensor<1x3xi32>) -> tensor<1x3x4x2x8xf32>
    %s291 = stablehlo.reshape %s290 : (tensor<1x3x4x2x8xf32>) -> tensor<1x12x2x8xf32>
    %s292 = "stablehlo.gather"(%139, %56) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<13x4x2x8xf32>, tensor<1x3xi32>) -> tensor<1x3x4x2x8xf32>
    %s293 = stablehlo.reshape %s292 : (tensor<1x3x4x2x8xf32>) -> tensor<1x12x2x8xf32>
    %s294 = stablehlo.reshape %130 : (tensor<32x4x8xf32>) -> tensor<1x32x2x2x8xf32>
    %s295 = stablehlo.dot_general %s294, %s291, batching_dims = [0, 2] x [0, 2], contracting_dims = [4] x [3], algorithm = <lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false> : (tensor<1x32x2x2x8xf32>, tensor<1x12x2x8xf32>) -> tensor<1x2x32x2x12xf32>
    %s296 = stablehlo.constant dense<0.35355338> : tensor<f32>
    %s297 = stablehlo.broadcast_in_dim %s296, dims = [] : (tensor<f32>) -> tensor<1x2x32x2x12xf32>
    %s298 = stablehlo.multiply %s295, %s297 : tensor<1x2x32x2x12xf32>
    %s299 = stablehlo.iota dim = 4 : tensor<1x2x32x2x12xi32>
    %s300 = stablehlo.reshape %58 : (tensor<32xi32>) -> tensor<1x32xi32>
    %s301 = stablehlo.broadcast_in_dim %s300, dims = [0, 2] : (tensor<1x32xi32>) -> tensor<1x2x32x2x12xi32>
    %s305 = stablehlo.constant dense<11> : tensor<i32>
    %s306 = stablehlo.broadcast_in_dim %s305, dims = [] : (tensor<i32>) -> tensor<1x2x32x2x12xi32>
    %s302 = stablehlo.add %s301, %s306 : tensor<1x2x32x2x12xi32>
    %s303 = stablehlo.subtract %s302, %s299 : tensor<1x2x32x2x12xi32>
    %s307 = stablehlo.constant dense<12> : tensor<i32>
    %s308 = stablehlo.broadcast_in_dim %s307, dims = [] : (tensor<i32>) -> tensor<1x2x32x2x12xi32>
    %s304 = stablehlo.remainder %s303, %s308 : tensor<1x2x32x2x12xi32>
    %s309 = stablehlo.constant dense<8> : tensor<i32>
    %s310 = stablehlo.broadcast_in_dim %s309, dims = [] : (tensor<i32>) -> tensor<1x2x32x2x12xi32>
    %s311 = stablehlo.compare LT, %s304, %s310, SIGNED : (tensor<1x2x32x2x12xi32>, tensor<1x2x32x2x12xi32>) -> tensor<1x2x32x2x12xi1>
    %s312 = stablehlo.compare LT, %s304, %s301, SIGNED : (tensor<1x2x32x2x12xi32>, tensor<1x2x32x2x12xi32>) -> tensor<1x2x32x2x12xi1>
    %s313 = stablehlo.and %s311, %s312 : tensor<1x2x32x2x12xi1>
    %s314 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s315 = stablehlo.broadcast_in_dim %s314, dims = [] : (tensor<f32>) -> tensor<1x2x32x2x12xf32>
    %s316 = stablehlo.select %s313, %s298, %s315 : tensor<1x2x32x2x12xi1>, tensor<1x2x32x2x12xf32>
    %s317 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s318 = stablehlo.reduce(%s316 init: %s317) applies stablehlo.maximum across dimensions = [4] : (tensor<1x2x32x2x12xf32>, tensor<f32>) -> tensor<1x2x32x2xf32>
    %s319 = stablehlo.broadcast_in_dim %s318, dims = [0, 1, 2, 3] : (tensor<1x2x32x2xf32>) -> tensor<1x2x32x2x12xf32>
    %s320 = stablehlo.subtract %s316, %s319 : tensor<1x2x32x2x12xf32>
    %s321 = stablehlo.exponential %s320 : tensor<1x2x32x2x12xf32>
    %s322 = stablehlo.constant dense<0.0> : tensor<f32>
    %s323 = stablehlo.reduce(%s321 init: %s322) applies stablehlo.add across dimensions = [4] : (tensor<1x2x32x2x12xf32>, tensor<f32>) -> tensor<1x2x32x2xf32>
    %s324 = stablehlo.broadcast_in_dim %s323, dims = [0, 1, 2, 3] : (tensor<1x2x32x2xf32>) -> tensor<1x2x32x2x12xf32>
    %s325 = stablehlo.divide %s321, %s324 : tensor<1x2x32x2x12xf32>
    %s326 = stablehlo.dot_general %s325, %s293, batching_dims = [0, 1] x [0, 2], contracting_dims = [4] x [1], algorithm = <lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false> : (tensor<1x2x32x2x12xf32>, tensor<1x12x2x8xf32>) -> tensor<1x2x32x2x8xf32>
    %s327 = stablehlo.transpose %s326, dims = [0, 2, 1, 3, 4] : (tensor<1x2x32x2x8xf32>) -> tensor<1x32x2x2x8xf32>
    %140 = stablehlo.reshape %s327 : (tensor<1x32x2x2x8xf32>) -> tensor<32x4x8xf32>
    %141 = stablehlo.reshape %140 : (tensor<32x4x8xf32>) -> tensor<32x32xf32>
    %142 = stablehlo.dot_general %141, %27, contracting_dims = [1] x [0] : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %143 = stablehlo.add %109, %142 : tensor<32x32xf32>
    %144 = stablehlo.multiply %143, %143 : tensor<32x32xf32>
    %s328 = stablehlo.constant dense<0.0> : tensor<f32>
    %s329 = stablehlo.reduce(%144 init: %s328) applies stablehlo.add across dimensions = [1] : (tensor<32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %s330 = stablehlo.constant dense<32.0> : tensor<32xf32>
    %s331 = stablehlo.divide %s329, %s330 : tensor<32xf32>
    %145 = stablehlo.broadcast_in_dim %s331, dims = [0] : (tensor<32xf32>) -> tensor<32x1xf32>
    %146 = stablehlo.add %145, %61 : tensor<32x1xf32>
    %147 = stablehlo.rsqrt %146 : tensor<32x1xf32>
    %s332 = stablehlo.broadcast_in_dim %147, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x32xf32>
    %148 = stablehlo.multiply %143, %s332 : tensor<32x32xf32>
    %149 = stablehlo.reshape %28 : (tensor<32xf32>) -> tensor<1x32xf32>
    %150 = stablehlo.broadcast_in_dim %149, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<32x32xf32>
    %151 = stablehlo.multiply %148, %150 : tensor<32x32xf32>
    %152 = stablehlo.dot_general %151, %29, contracting_dims = [1] x [0] : (tensor<32x32xf32>, tensor<32x48xf32>) -> tensor<32x48xf32>
    %153 = stablehlo.dot_general %151, %30, contracting_dims = [1] x [0] : (tensor<32x32xf32>, tensor<32x48xf32>) -> tensor<32x48xf32>
    %s333 = stablehlo.logistic %152 : tensor<32x48xf32>
    %154 = stablehlo.multiply %152, %s333 : tensor<32x48xf32>
    %155 = stablehlo.multiply %154, %153 : tensor<32x48xf32>
    %156 = stablehlo.dot_general %155, %31, contracting_dims = [1] x [0] : (tensor<32x48xf32>, tensor<48x32xf32>) -> tensor<32x32xf32>
    %157 = stablehlo.add %143, %156 : tensor<32x32xf32>
    %158 = stablehlo.multiply %157, %157 : tensor<32x32xf32>
    %s334 = stablehlo.constant dense<0.0> : tensor<f32>
    %s335 = stablehlo.reduce(%158 init: %s334) applies stablehlo.add across dimensions = [1] : (tensor<32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %s336 = stablehlo.constant dense<32.0> : tensor<32xf32>
    %s337 = stablehlo.divide %s335, %s336 : tensor<32xf32>
    %159 = stablehlo.broadcast_in_dim %s337, dims = [0] : (tensor<32xf32>) -> tensor<32x1xf32>
    %160 = stablehlo.add %159, %61 : tensor<32x1xf32>
    %161 = stablehlo.rsqrt %160 : tensor<32x1xf32>
    %s338 = stablehlo.broadcast_in_dim %161, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x32xf32>
    %162 = stablehlo.multiply %157, %s338 : tensor<32x32xf32>
    %163 = stablehlo.reshape %32 : (tensor<32xf32>) -> tensor<1x32xf32>
    %164 = stablehlo.broadcast_in_dim %163, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<32x32xf32>
    %165 = stablehlo.multiply %162, %164 : tensor<32x32xf32>
    %166 = stablehlo.dot_general %165, %33, contracting_dims = [1] x [0] : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %167 = stablehlo.dot_general %165, %34, contracting_dims = [1] x [0] : (tensor<32x32xf32>, tensor<32x16xf32>) -> tensor<32x16xf32>
    %168 = stablehlo.dot_general %165, %35, contracting_dims = [1] x [0] : (tensor<32x32xf32>, tensor<32x16xf32>) -> tensor<32x16xf32>
    %169 = stablehlo.reshape %166 : (tensor<32x32xf32>) -> tensor<32x4x8xf32>
    %170 = stablehlo.reshape %167 : (tensor<32x16xf32>) -> tensor<32x2x8xf32>
    %171 = stablehlo.reshape %168 : (tensor<32x16xf32>) -> tensor<32x2x8xf32>
    %172 = stablehlo.multiply %169, %49 : tensor<32x4x8xf32>
    %173 = stablehlo.slice %169 [0:32, 0:4, 4:8] : (tensor<32x4x8xf32>) -> tensor<32x4x4xf32>
    %174 = stablehlo.slice %169 [0:32, 0:4, 0:4] : (tensor<32x4x8xf32>) -> tensor<32x4x4xf32>
    %175 = stablehlo.negate %173 : tensor<32x4x4xf32>
    %176 = stablehlo.concatenate %175, %174, dim = 2 : (tensor<32x4x4xf32>, tensor<32x4x4xf32>) -> tensor<32x4x8xf32>
    %177 = stablehlo.multiply %176, %51 : tensor<32x4x8xf32>
    %178 = stablehlo.add %172, %177 : tensor<32x4x8xf32>
    %179 = stablehlo.multiply %170, %53 : tensor<32x2x8xf32>
    %180 = stablehlo.slice %170 [0:32, 0:2, 4:8] : (tensor<32x2x8xf32>) -> tensor<32x2x4xf32>
    %181 = stablehlo.slice %170 [0:32, 0:2, 0:4] : (tensor<32x2x8xf32>) -> tensor<32x2x4xf32>
    %182 = stablehlo.negate %180 : tensor<32x2x4xf32>
    %183 = stablehlo.concatenate %182, %181, dim = 2 : (tensor<32x2x4xf32>, tensor<32x2x4xf32>) -> tensor<32x2x8xf32>
    %184 = stablehlo.multiply %183, %55 : tensor<32x2x8xf32>
    %185 = stablehlo.add %179, %184 : tensor<32x2x8xf32>
    %s339 = stablehlo.reshape %11 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s340 = "stablehlo.scatter"(%s339, %4, %185) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s341: tensor<f32>, %s342: tensor<f32>):
       stablehlo.return %s342 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<32xi32>, tensor<32x2x8xf32>) -> tensor<260x2x8xf32>
    %186 = stablehlo.reshape %s340 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s343 = stablehlo.reshape %12 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s344 = "stablehlo.scatter"(%s343, %4, %171) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s345: tensor<f32>, %s346: tensor<f32>):
       stablehlo.return %s346 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<32xi32>, tensor<32x2x8xf32>) -> tensor<260x2x8xf32>
    %187 = stablehlo.reshape %s344 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s347 = "stablehlo.gather"(%186, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<1x8xi32>) -> tensor<1x8x4x2x8xf32>
    %s348 = stablehlo.reshape %s347 : (tensor<1x8x4x2x8xf32>) -> tensor<1x32x2x8xf32>
    %s349 = "stablehlo.gather"(%187, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<1x8xi32>) -> tensor<1x8x4x2x8xf32>
    %s350 = stablehlo.reshape %s349 : (tensor<1x8x4x2x8xf32>) -> tensor<1x32x2x8xf32>
    %s351 = stablehlo.reshape %178 : (tensor<32x4x8xf32>) -> tensor<1x32x2x2x8xf32>
    %s352 = stablehlo.dot_general %s351, %s348, batching_dims = [0, 2] x [0, 2], contracting_dims = [4] x [3], algorithm = <lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false> : (tensor<1x32x2x2x8xf32>, tensor<1x32x2x8xf32>) -> tensor<1x2x32x2x32xf32>
    %s353 = stablehlo.constant dense<0.35355338> : tensor<f32>
    %s354 = stablehlo.broadcast_in_dim %s353, dims = [] : (tensor<f32>) -> tensor<1x2x32x2x32xf32>
    %s355 = stablehlo.multiply %s352, %s354 : tensor<1x2x32x2x32xf32>
    %s356 = stablehlo.iota dim = 4 : tensor<1x2x32x2x32xi32>
    %s357 = stablehlo.reshape %58 : (tensor<32xi32>) -> tensor<1x32xi32>
    %s358 = stablehlo.broadcast_in_dim %s357, dims = [0, 2] : (tensor<1x32xi32>) -> tensor<1x2x32x2x32xi32>
    %s359 = stablehlo.compare LT, %s356, %s358, SIGNED : (tensor<1x2x32x2x32xi32>, tensor<1x2x32x2x32xi32>) -> tensor<1x2x32x2x32xi1>
    %s360 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s361 = stablehlo.broadcast_in_dim %s360, dims = [] : (tensor<f32>) -> tensor<1x2x32x2x32xf32>
    %s362 = stablehlo.select %s359, %s355, %s361 : tensor<1x2x32x2x32xi1>, tensor<1x2x32x2x32xf32>
    %s363 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s364 = stablehlo.reduce(%s362 init: %s363) applies stablehlo.maximum across dimensions = [4] : (tensor<1x2x32x2x32xf32>, tensor<f32>) -> tensor<1x2x32x2xf32>
    %s365 = stablehlo.broadcast_in_dim %s364, dims = [0, 1, 2, 3] : (tensor<1x2x32x2xf32>) -> tensor<1x2x32x2x32xf32>
    %s366 = stablehlo.subtract %s362, %s365 : tensor<1x2x32x2x32xf32>
    %s367 = stablehlo.exponential %s366 : tensor<1x2x32x2x32xf32>
    %s368 = stablehlo.constant dense<0.0> : tensor<f32>
    %s369 = stablehlo.reduce(%s367 init: %s368) applies stablehlo.add across dimensions = [4] : (tensor<1x2x32x2x32xf32>, tensor<f32>) -> tensor<1x2x32x2xf32>
    %s370 = stablehlo.broadcast_in_dim %s369, dims = [0, 1, 2, 3] : (tensor<1x2x32x2xf32>) -> tensor<1x2x32x2x32xf32>
    %s371 = stablehlo.divide %s367, %s370 : tensor<1x2x32x2x32xf32>
    %s372 = stablehlo.dot_general %s371, %s350, batching_dims = [0, 1] x [0, 2], contracting_dims = [4] x [1], algorithm = <lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false> : (tensor<1x2x32x2x32xf32>, tensor<1x32x2x8xf32>) -> tensor<1x2x32x2x8xf32>
    %s373 = stablehlo.transpose %s372, dims = [0, 2, 1, 3, 4] : (tensor<1x2x32x2x8xf32>) -> tensor<1x32x2x2x8xf32>
    %188 = stablehlo.reshape %s373 : (tensor<1x32x2x2x8xf32>) -> tensor<32x4x8xf32>
    %189 = stablehlo.reshape %188 : (tensor<32x4x8xf32>) -> tensor<32x32xf32>
    %190 = stablehlo.dot_general %189, %36, contracting_dims = [1] x [0] : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %191 = stablehlo.add %157, %190 : tensor<32x32xf32>
    %192 = stablehlo.multiply %191, %191 : tensor<32x32xf32>
    %s374 = stablehlo.constant dense<0.0> : tensor<f32>
    %s375 = stablehlo.reduce(%192 init: %s374) applies stablehlo.add across dimensions = [1] : (tensor<32x32xf32>, tensor<f32>) -> tensor<32xf32>
    %s376 = stablehlo.constant dense<32.0> : tensor<32xf32>
    %s377 = stablehlo.divide %s375, %s376 : tensor<32xf32>
    %193 = stablehlo.broadcast_in_dim %s377, dims = [0] : (tensor<32xf32>) -> tensor<32x1xf32>
    %194 = stablehlo.add %193, %61 : tensor<32x1xf32>
    %195 = stablehlo.rsqrt %194 : tensor<32x1xf32>
    %s378 = stablehlo.broadcast_in_dim %195, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x32xf32>
    %196 = stablehlo.multiply %191, %s378 : tensor<32x32xf32>
    %197 = stablehlo.reshape %37 : (tensor<32xf32>) -> tensor<1x32xf32>
    %198 = stablehlo.broadcast_in_dim %197, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<32x32xf32>
    %199 = stablehlo.multiply %196, %198 : tensor<32x32xf32>
    %200 = stablehlo.dot_general %199, %38, contracting_dims = [1] x [0] : (tensor<32x32xf32>, tensor<32x48xf32>) -> tensor<32x48xf32>
    %201 = stablehlo.dot_general %199, %39, contracting_dims = [1] x [0] : (tensor<32x32xf32>, tensor<32x48xf32>) -> tensor<32x48xf32>
    %s379 = stablehlo.logistic %200 : tensor<32x48xf32>
    %202 = stablehlo.multiply %200, %s379 : tensor<32x48xf32>
    %203 = stablehlo.multiply %202, %201 : tensor<32x48xf32>
    %204 = stablehlo.dot_general %203, %40, contracting_dims = [1] x [0] : (tensor<32x48xf32>, tensor<48x32xf32>) -> tensor<32x32xf32>
    %205 = stablehlo.add %191, %204 : tensor<32x32xf32>
    %206 = stablehlo.reshape %205 : (tensor<32x32xf32>) -> tensor<1x32x32xf32>
    %207 = stablehlo.slice %206 [0:1, 31:32, 0:32] : (tensor<1x32x32xf32>) -> tensor<1x1x32xf32>
    %208 = stablehlo.reshape %207 : (tensor<1x1x32xf32>) -> tensor<1x32xf32>
    %209 = stablehlo.constant dense<[[1.0E-6]]> : tensor<1x1xf32>
    %210 = stablehlo.multiply %208, %208 : tensor<1x32xf32>
    %s380 = stablehlo.constant dense<0.0> : tensor<f32>
    %s381 = stablehlo.reduce(%210 init: %s380) applies stablehlo.add across dimensions = [1] : (tensor<1x32xf32>, tensor<f32>) -> tensor<1xf32>
    %s382 = stablehlo.constant dense<32.0> : tensor<1xf32>
    %s383 = stablehlo.divide %s381, %s382 : tensor<1xf32>
    %211 = stablehlo.broadcast_in_dim %s383, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %212 = stablehlo.add %211, %209 : tensor<1x1xf32>
    %213 = stablehlo.rsqrt %212 : tensor<1x1xf32>
    %s384 = stablehlo.broadcast_in_dim %213, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x32xf32>
    %214 = stablehlo.multiply %208, %s384 : tensor<1x32xf32>
    %215 = stablehlo.reshape %41 : (tensor<32xf32>) -> tensor<1x32xf32>
    %216 = stablehlo.broadcast_in_dim %215, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32xf32>
    %217 = stablehlo.multiply %214, %216 : tensor<1x32xf32>
    %218 = stablehlo.dot_general %217, %42, contracting_dims = [1] x [0] : (tensor<1x32xf32>, tensor<32x64xf32>) -> tensor<1x64xf32>
    %219 = stablehlo.reshape %218 : (tensor<1x64xf32>) -> tensor<1x1x64xf32>
    return %219, %90, %91, %138, %139, %186, %187 : tensor<1x1x64xf32>, tensor<13x4x2x8xf32>, tensor<13x4x2x8xf32>, tensor<13x4x2x8xf32>, tensor<13x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>
  }
}
