module {
  func.func @main(%0: tensor<2x1xi32>, %1: tensor<2x1xi32>, %2: tensor<2x8xi32>, %3: tensor<2xi32>, %4: tensor<2xi32>, %5: tensor<65x4x2x8xf32> {tf.aliasing_output = 1 : i32}, %6: tensor<65x4x2x8xf32> {tf.aliasing_output = 2 : i32}, %7: tensor<65x4x2x8xf32> {tf.aliasing_output = 3 : i32}, %8: tensor<65x4x2x8xf32> {tf.aliasing_output = 4 : i32}, %9: tensor<65x4x2x8xf32> {tf.aliasing_output = 5 : i32}, %10: tensor<65x4x2x8xf32> {tf.aliasing_output = 6 : i32}, %11: tensor<64x32xf32>, %12: tensor<32xf32>, %13: tensor<32x32xf32>, %14: tensor<32x16xf32>, %15: tensor<32x16xf32>, %16: tensor<32x32xf32>, %17: tensor<32xf32>, %18: tensor<32x48xf32>, %19: tensor<32x48xf32>, %20: tensor<48x32xf32>, %21: tensor<32xf32>, %22: tensor<32x32xf32>, %23: tensor<32x16xf32>, %24: tensor<32x16xf32>, %25: tensor<32x32xf32>, %26: tensor<32xf32>, %27: tensor<32x48xf32>, %28: tensor<32x48xf32>, %29: tensor<48x32xf32>, %30: tensor<32xf32>, %31: tensor<32x32xf32>, %32: tensor<32x16xf32>, %33: tensor<32x16xf32>, %34: tensor<32x32xf32>, %35: tensor<32xf32>, %36: tensor<32x48xf32>, %37: tensor<32x48xf32>, %38: tensor<48x32xf32>, %39: tensor<32xf32>, %40: tensor<32x64xf32>) -> (tensor<2x1x64xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>) {
    %41 = stablehlo.reshape %1 : (tensor<2x1xi32>) -> tensor<2xi32>
    %42 = stablehlo.constant dense<[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.5403023, 0.9950042, 0.99995, 0.9999995, 0.5403023, 0.9950042, 0.99995, 0.9999995], [-0.41614684, 0.9800666, 0.9998, 0.999998, -0.41614684, 0.9800666, 0.9998, 0.999998], [-0.9899925, 0.9553365, 0.99955004, 0.9999955, -0.9899925, 0.9553365, 0.99955004, 0.9999955], [-0.6536436, 0.921061, 0.9992001, 0.999992, -0.6536436, 0.921061, 0.9992001, 0.999992], [0.2836622, 0.87758255, 0.99875027, 0.9999875, 0.2836622, 0.87758255, 0.99875027, 0.9999875], [0.96017027, 0.8253356, 0.99820054, 0.999982, 0.96017027, 0.8253356, 0.99820054, 0.999982], [0.75390226, 0.7648422, 0.997551, 0.9999755, 0.75390226, 0.7648422, 0.997551, 0.9999755], [-0.14550003, 0.6967067, 0.99680173, 0.999968, -0.14550003, 0.6967067, 0.99680173, 0.999968], [-0.91113025, 0.62161, 0.9959527, 0.9999595, -0.91113025, 0.62161, 0.9959527, 0.9999595], [-0.8390715, 0.5403023, 0.9950042, 0.99995, -0.8390715, 0.5403023, 0.9950042, 0.99995], [0.004425698, 0.45359612, 0.9939561, 0.9999395, 0.004425698, 0.45359612, 0.9939561, 0.9999395], [0.84385395, 0.36235777, 0.99280864, 0.999928, 0.84385395, 0.36235777, 0.99280864, 0.999928], [0.9074468, 0.26749882, 0.9915619, 0.9999155, 0.9074468, 0.26749882, 0.9915619, 0.9999155], [0.13673721, 0.16996714, 0.990216, 0.999902, 0.13673721, 0.16996714, 0.990216, 0.999902], [-0.7596879, 0.0707372, 0.9887711, 0.9998875, -0.7596879, 0.0707372, 0.9887711, 0.9998875], [-0.9576595, -0.029199522, 0.98722726, 0.999872, -0.9576595, -0.029199522, 0.98722726, 0.999872], [-0.27516335, -0.1288445, 0.9855848, 0.9998555, -0.27516335, -0.1288445, 0.9855848, 0.9998555], [0.6603167, -0.22720209, 0.9838437, 0.999838, 0.6603167, -0.22720209, 0.9838437, 0.999838], [0.9887046, -0.32328957, 0.9820042, 0.9998195, 0.9887046, -0.32328957, 0.9820042, 0.9998195], [0.40808207, -0.41614684, 0.9800666, 0.9998, 0.40808207, -0.41614684, 0.9800666, 0.9998], [-0.54772925, -0.5048461, 0.9780309, 0.9997795, -0.54772925, -0.5048461, 0.9780309, 0.9997795], [-0.99996084, -0.5885011, 0.97589743, 0.999758, -0.99996084, -0.5885011, 0.97589743, 0.999758], [-0.53283304, -0.66627604, 0.97366637, 0.99973553, -0.53283304, -0.66627604, 0.97366637, 0.99973553], [0.42417902, -0.73739374, 0.971338, 0.999712, 0.42417902, -0.73739374, 0.971338, 0.999712], [0.99120283, -0.8011436, 0.9689124, 0.9996875, 0.99120283, -0.8011436, 0.9689124, 0.9996875], [0.6469193, -0.8568888, 0.96638995, 0.99966204, 0.6469193, -0.8568888, 0.96638995, 0.99966204], [-0.29213881, -0.90407217, 0.9637709, 0.9996355, -0.29213881, -0.90407217, 0.9637709, 0.9996355], [-0.9626059, -0.94222236, 0.96105546, 0.99960804, -0.9626059, -0.94222236, 0.96105546, 0.99960804], [-0.74805754, -0.9709582, 0.95824385, 0.99957955, -0.74805754, -0.9709582, 0.95824385, 0.99957955], [0.15425146, -0.9899925, 0.9553365, 0.99955004, 0.15425146, -0.9899925, 0.9553365, 0.99955004], [0.91474235, -0.99913514, 0.95233357, 0.9995195, 0.91474235, -0.99913514, 0.95233357, 0.9995195]]> : tensor<32x8xf32>
    %43 = stablehlo.constant dense<[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.84147096, 0.099833414, 0.009999833, 9.999998E-4, 0.84147096, 0.099833414, 0.009999833, 9.999998E-4], [0.9092974, 0.19866933, 0.019998666, 0.0019999987, 0.9092974, 0.19866933, 0.019998666, 0.0019999987], [0.14112, 0.29552022, 0.029995501, 0.0029999956, 0.14112, 0.29552022, 0.029995501, 0.0029999956], [-0.7568025, 0.38941833, 0.039989334, 0.0039999895, -0.7568025, 0.38941833, 0.039989334, 0.0039999895], [-0.9589243, 0.47942555, 0.04997917, 0.0049999794, -0.9589243, 0.47942555, 0.04997917, 0.0049999794], [-0.2794155, 0.5646425, 0.059964005, 0.005999964, -0.2794155, 0.5646425, 0.059964005, 0.005999964], [0.6569866, 0.64421767, 0.06994285, 0.006999943, 0.6569866, 0.64421767, 0.06994285, 0.006999943], [0.98935825, 0.7173561, 0.0799147, 0.007999915, 0.98935825, 0.7173561, 0.0799147, 0.007999915], [0.4121185, 0.7833269, 0.08987855, 0.008999879, 0.4121185, 0.7833269, 0.08987855, 0.008999879], [-0.5440211, 0.84147096, 0.099833414, 0.009999833, -0.5440211, 0.84147096, 0.099833414, 0.009999833], [-0.9999902, 0.89120734, 0.1097783, 0.010999778, -0.9999902, 0.89120734, 0.1097783, 0.010999778], [-0.53657293, 0.9320391, 0.119712204, 0.011999712, -0.53657293, 0.9320391, 0.119712204, 0.011999712], [0.42016703, 0.9635582, 0.12963414, 0.012999634, 0.42016703, 0.9635582, 0.12963414, 0.012999634], [0.9906074, 0.98544973, 0.13954312, 0.013999542, 0.9906074, 0.98544973, 0.13954312, 0.013999542], [0.65028787, 0.997495, 0.14943813, 0.014999437, 0.65028787, 0.997495, 0.14943813, 0.014999437], [-0.2879033, 0.9995736, 0.15931821, 0.015999317, -0.2879033, 0.9995736, 0.15931821, 0.015999317], [-0.96139747, 0.9916648, 0.16918235, 0.016999181, -0.96139747, 0.9916648, 0.16918235, 0.016999181], [-0.75098723, 0.9738476, 0.17902957, 0.017999029, -0.75098723, 0.9738476, 0.17902957, 0.017999029], [0.1498772, 0.9463001, 0.1888589, 0.018998858, 0.1498772, 0.9463001, 0.1888589, 0.018998858], [0.9129453, 0.9092974, 0.19866933, 0.019998666, 0.9129453, 0.9092974, 0.19866933, 0.019998666], [0.8366556, 0.86320937, 0.2084599, 0.020998457, 0.8366556, 0.86320937, 0.2084599, 0.020998457], [-0.008851309, 0.8084964, 0.21822962, 0.021998225, -0.008851309, 0.8084964, 0.21822962, 0.021998225], [-0.84622043, 0.7457052, 0.22797753, 0.022997972, -0.84622043, 0.7457052, 0.22797753, 0.022997972], [-0.9055784, 0.6754632, 0.23770262, 0.023997696, -0.9055784, 0.6754632, 0.23770262, 0.023997696], [-0.13235176, 0.5984721, 0.24740396, 0.024997396, -0.13235176, 0.5984721, 0.24740396, 0.024997396], [0.76255846, 0.5155014, 0.25708055, 0.02599707, 0.76255846, 0.5155014, 0.25708055, 0.02599707], [0.95637596, 0.42737988, 0.26673144, 0.026996719, 0.95637596, 0.42737988, 0.26673144, 0.026996719], [0.2709058, 0.33498815, 0.27635565, 0.02799634, 0.2709058, 0.33498815, 0.27635565, 0.02799634], [-0.6636339, 0.23924933, 0.2859522, 0.028995935, -0.6636339, 0.23924933, 0.2859522, 0.028995935], [-0.9880316, 0.14112, 0.29552022, 0.029995501, -0.9880316, 0.14112, 0.29552022, 0.029995501], [-0.40403765, 0.041580662, 0.30505863, 0.030995036, -0.40403765, 0.041580662, 0.30505863, 0.030995036]]> : tensor<32x8xf32>
    %44 = "stablehlo.gather"(%42, %41) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 8>}> : (tensor<32x8xf32>, tensor<2xi32>) -> tensor<2x8xf32>
    %45 = "stablehlo.gather"(%43, %41) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 8>}> : (tensor<32x8xf32>, tensor<2xi32>) -> tensor<2x8xf32>
    %46 = stablehlo.reshape %44 : (tensor<2x8xf32>) -> tensor<2x1x8xf32>
    %47 = stablehlo.broadcast_in_dim %46, dims = [0, 1, 2] : (tensor<2x1x8xf32>) -> tensor<2x4x8xf32>
    %48 = stablehlo.reshape %45 : (tensor<2x8xf32>) -> tensor<2x1x8xf32>
    %49 = stablehlo.broadcast_in_dim %48, dims = [0, 1, 2] : (tensor<2x1x8xf32>) -> tensor<2x4x8xf32>
    %50 = stablehlo.reshape %44 : (tensor<2x8xf32>) -> tensor<2x1x8xf32>
    %51 = stablehlo.broadcast_in_dim %50, dims = [0, 1, 2] : (tensor<2x1x8xf32>) -> tensor<2x2x8xf32>
    %52 = stablehlo.reshape %45 : (tensor<2x8xf32>) -> tensor<2x1x8xf32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [0, 1, 2] : (tensor<2x1x8xf32>) -> tensor<2x2x8xf32>
    %54 = "stablehlo.gather"(%11, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 32>}> : (tensor<64x32xf32>, tensor<2x1xi32>) -> tensor<2x1x32xf32>
    %55 = stablehlo.reshape %54 : (tensor<2x1x32xf32>) -> tensor<2x32xf32>
    %56 = stablehlo.constant dense<[[1.0E-6], [1.0E-6]]> : tensor<2x1xf32>
    %57 = stablehlo.multiply %55, %55 : tensor<2x32xf32>
    %s211 = stablehlo.constant dense<0.0> : tensor<f32>
    %s212 = stablehlo.reduce(%57 init: %s211) applies stablehlo.add across dimensions = [1] : (tensor<2x32xf32>, tensor<f32>) -> tensor<2xf32>
    %s213 = stablehlo.constant dense<32.0> : tensor<2xf32>
    %s214 = stablehlo.divide %s212, %s213 : tensor<2xf32>
    %58 = stablehlo.broadcast_in_dim %s214, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %59 = stablehlo.add %58, %56 : tensor<2x1xf32>
    %60 = stablehlo.rsqrt %59 : tensor<2x1xf32>
    %s215 = stablehlo.broadcast_in_dim %60, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x32xf32>
    %61 = stablehlo.multiply %55, %s215 : tensor<2x32xf32>
    %62 = stablehlo.reshape %12 : (tensor<32xf32>) -> tensor<1x32xf32>
    %63 = stablehlo.broadcast_in_dim %62, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<2x32xf32>
    %64 = stablehlo.multiply %61, %63 : tensor<2x32xf32>
    %65 = stablehlo.dot_general %64, %13, contracting_dims = [1] x [0] : (tensor<2x32xf32>, tensor<32x32xf32>) -> tensor<2x32xf32>
    %66 = stablehlo.dot_general %64, %14, contracting_dims = [1] x [0] : (tensor<2x32xf32>, tensor<32x16xf32>) -> tensor<2x16xf32>
    %67 = stablehlo.dot_general %64, %15, contracting_dims = [1] x [0] : (tensor<2x32xf32>, tensor<32x16xf32>) -> tensor<2x16xf32>
    %68 = stablehlo.reshape %65 : (tensor<2x32xf32>) -> tensor<2x4x8xf32>
    %69 = stablehlo.reshape %66 : (tensor<2x16xf32>) -> tensor<2x2x8xf32>
    %70 = stablehlo.reshape %67 : (tensor<2x16xf32>) -> tensor<2x2x8xf32>
    %71 = stablehlo.multiply %68, %47 : tensor<2x4x8xf32>
    %72 = stablehlo.slice %68 [0:2, 0:4, 4:8] : (tensor<2x4x8xf32>) -> tensor<2x4x4xf32>
    %73 = stablehlo.slice %68 [0:2, 0:4, 0:4] : (tensor<2x4x8xf32>) -> tensor<2x4x4xf32>
    %74 = stablehlo.negate %72 : tensor<2x4x4xf32>
    %75 = stablehlo.concatenate %74, %73, dim = 2 : (tensor<2x4x4xf32>, tensor<2x4x4xf32>) -> tensor<2x4x8xf32>
    %76 = stablehlo.multiply %75, %49 : tensor<2x4x8xf32>
    %77 = stablehlo.add %71, %76 : tensor<2x4x8xf32>
    %78 = stablehlo.multiply %69, %51 : tensor<2x2x8xf32>
    %79 = stablehlo.slice %69 [0:2, 0:2, 4:8] : (tensor<2x2x8xf32>) -> tensor<2x2x4xf32>
    %80 = stablehlo.slice %69 [0:2, 0:2, 0:4] : (tensor<2x2x8xf32>) -> tensor<2x2x4xf32>
    %81 = stablehlo.negate %79 : tensor<2x2x4xf32>
    %82 = stablehlo.concatenate %81, %80, dim = 2 : (tensor<2x2x4xf32>, tensor<2x2x4xf32>) -> tensor<2x2x8xf32>
    %83 = stablehlo.multiply %82, %53 : tensor<2x2x8xf32>
    %84 = stablehlo.add %78, %83 : tensor<2x2x8xf32>
    %s216 = stablehlo.reshape %5 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s217 = "stablehlo.scatter"(%s216, %4, %84) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s218: tensor<f32>, %s219: tensor<f32>):
       stablehlo.return %s219 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<2xi32>, tensor<2x2x8xf32>) -> tensor<260x2x8xf32>
    %85 = stablehlo.reshape %s217 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s220 = stablehlo.reshape %6 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s221 = "stablehlo.scatter"(%s220, %4, %70) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s222: tensor<f32>, %s223: tensor<f32>):
       stablehlo.return %s223 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<2xi32>, tensor<2x2x8xf32>) -> tensor<260x2x8xf32>
    %86 = stablehlo.reshape %s221 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s224 = "stablehlo.gather"(%85, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<2x8xi32>) -> tensor<2x8x4x2x8xf32>
    %s225 = stablehlo.reshape %s224 : (tensor<2x8x4x2x8xf32>) -> tensor<2x32x2x8xf32>
    %s226 = "stablehlo.gather"(%86, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<2x8xi32>) -> tensor<2x8x4x2x8xf32>
    %s227 = stablehlo.reshape %s226 : (tensor<2x8x4x2x8xf32>) -> tensor<2x32x2x8xf32>
    %s228 = stablehlo.reshape %77 : (tensor<2x4x8xf32>) -> tensor<2x2x2x8xf32>
    %s229 = stablehlo.dot_general %s228, %s225, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [3], precision = [HIGHEST, HIGHEST] : (tensor<2x2x2x8xf32>, tensor<2x32x2x8xf32>) -> tensor<2x2x2x32xf32>
    %s230 = stablehlo.constant dense<0.35355338> : tensor<f32>
    %s231 = stablehlo.broadcast_in_dim %s230, dims = [] : (tensor<f32>) -> tensor<2x2x2x32xf32>
    %s232 = stablehlo.multiply %s229, %s231 : tensor<2x2x2x32xf32>
    %s233 = stablehlo.iota dim = 3 : tensor<2x2x2x32xi32>
    %s234 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<2xi32>) -> tensor<2x2x2x32xi32>
    %s236 = stablehlo.compare LT, %s233, %s234, SIGNED : (tensor<2x2x2x32xi32>, tensor<2x2x2x32xi32>) -> tensor<2x2x2x32xi1>
    %s237 = stablehlo.constant dense<8> : tensor<i32>
    %s238 = stablehlo.broadcast_in_dim %s237, dims = [] : (tensor<i32>) -> tensor<2x2x2x32xi32>
    %s239 = stablehlo.subtract %s234, %s238 : tensor<2x2x2x32xi32>
    %s240 = stablehlo.compare GE, %s233, %s239, SIGNED : (tensor<2x2x2x32xi32>, tensor<2x2x2x32xi32>) -> tensor<2x2x2x32xi1>
    %s235 = stablehlo.and %s236, %s240 : tensor<2x2x2x32xi1>
    %s241 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s242 = stablehlo.broadcast_in_dim %s241, dims = [] : (tensor<f32>) -> tensor<2x2x2x32xf32>
    %s243 = stablehlo.select %s235, %s232, %s242 : tensor<2x2x2x32xi1>, tensor<2x2x2x32xf32>
    %s244 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s245 = stablehlo.reduce(%s243 init: %s244) applies stablehlo.maximum across dimensions = [3] : (tensor<2x2x2x32xf32>, tensor<f32>) -> tensor<2x2x2xf32>
    %s246 = stablehlo.broadcast_in_dim %s245, dims = [0, 1, 2] : (tensor<2x2x2xf32>) -> tensor<2x2x2x32xf32>
    %s247 = stablehlo.subtract %s243, %s246 : tensor<2x2x2x32xf32>
    %s248 = stablehlo.exponential %s247 : tensor<2x2x2x32xf32>
    %s249 = stablehlo.constant dense<0.0> : tensor<f32>
    %s250 = stablehlo.reduce(%s248 init: %s249) applies stablehlo.add across dimensions = [3] : (tensor<2x2x2x32xf32>, tensor<f32>) -> tensor<2x2x2xf32>
    %s251 = stablehlo.broadcast_in_dim %s250, dims = [0, 1, 2] : (tensor<2x2x2xf32>) -> tensor<2x2x2x32xf32>
    %s252 = stablehlo.divide %s248, %s251 : tensor<2x2x2x32xf32>
    %s253 = stablehlo.dot_general %s252, %s227, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [HIGHEST, HIGHEST] : (tensor<2x2x2x32xf32>, tensor<2x32x2x8xf32>) -> tensor<2x2x2x8xf32>
    %87 = stablehlo.reshape %s253 : (tensor<2x2x2x8xf32>) -> tensor<2x4x8xf32>
    %88 = stablehlo.reshape %87 : (tensor<2x4x8xf32>) -> tensor<2x32xf32>
    %89 = stablehlo.dot_general %88, %16, contracting_dims = [1] x [0] : (tensor<2x32xf32>, tensor<32x32xf32>) -> tensor<2x32xf32>
    %90 = stablehlo.add %55, %89 : tensor<2x32xf32>
    %91 = stablehlo.multiply %90, %90 : tensor<2x32xf32>
    %s254 = stablehlo.constant dense<0.0> : tensor<f32>
    %s255 = stablehlo.reduce(%91 init: %s254) applies stablehlo.add across dimensions = [1] : (tensor<2x32xf32>, tensor<f32>) -> tensor<2xf32>
    %s256 = stablehlo.constant dense<32.0> : tensor<2xf32>
    %s257 = stablehlo.divide %s255, %s256 : tensor<2xf32>
    %92 = stablehlo.broadcast_in_dim %s257, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %93 = stablehlo.add %92, %56 : tensor<2x1xf32>
    %94 = stablehlo.rsqrt %93 : tensor<2x1xf32>
    %s258 = stablehlo.broadcast_in_dim %94, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x32xf32>
    %95 = stablehlo.multiply %90, %s258 : tensor<2x32xf32>
    %96 = stablehlo.reshape %17 : (tensor<32xf32>) -> tensor<1x32xf32>
    %97 = stablehlo.broadcast_in_dim %96, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<2x32xf32>
    %98 = stablehlo.multiply %95, %97 : tensor<2x32xf32>
    %99 = stablehlo.dot_general %98, %18, contracting_dims = [1] x [0] : (tensor<2x32xf32>, tensor<32x48xf32>) -> tensor<2x48xf32>
    %100 = stablehlo.dot_general %98, %19, contracting_dims = [1] x [0] : (tensor<2x32xf32>, tensor<32x48xf32>) -> tensor<2x48xf32>
    %s259 = stablehlo.logistic %99 : tensor<2x48xf32>
    %101 = stablehlo.multiply %99, %s259 : tensor<2x48xf32>
    %102 = stablehlo.multiply %101, %100 : tensor<2x48xf32>
    %103 = stablehlo.dot_general %102, %20, contracting_dims = [1] x [0] : (tensor<2x48xf32>, tensor<48x32xf32>) -> tensor<2x32xf32>
    %104 = stablehlo.add %90, %103 : tensor<2x32xf32>
    %105 = stablehlo.multiply %104, %104 : tensor<2x32xf32>
    %s260 = stablehlo.constant dense<0.0> : tensor<f32>
    %s261 = stablehlo.reduce(%105 init: %s260) applies stablehlo.add across dimensions = [1] : (tensor<2x32xf32>, tensor<f32>) -> tensor<2xf32>
    %s262 = stablehlo.constant dense<32.0> : tensor<2xf32>
    %s263 = stablehlo.divide %s261, %s262 : tensor<2xf32>
    %106 = stablehlo.broadcast_in_dim %s263, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %107 = stablehlo.add %106, %56 : tensor<2x1xf32>
    %108 = stablehlo.rsqrt %107 : tensor<2x1xf32>
    %s264 = stablehlo.broadcast_in_dim %108, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x32xf32>
    %109 = stablehlo.multiply %104, %s264 : tensor<2x32xf32>
    %110 = stablehlo.reshape %21 : (tensor<32xf32>) -> tensor<1x32xf32>
    %111 = stablehlo.broadcast_in_dim %110, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<2x32xf32>
    %112 = stablehlo.multiply %109, %111 : tensor<2x32xf32>
    %113 = stablehlo.dot_general %112, %22, contracting_dims = [1] x [0] : (tensor<2x32xf32>, tensor<32x32xf32>) -> tensor<2x32xf32>
    %114 = stablehlo.dot_general %112, %23, contracting_dims = [1] x [0] : (tensor<2x32xf32>, tensor<32x16xf32>) -> tensor<2x16xf32>
    %115 = stablehlo.dot_general %112, %24, contracting_dims = [1] x [0] : (tensor<2x32xf32>, tensor<32x16xf32>) -> tensor<2x16xf32>
    %116 = stablehlo.reshape %113 : (tensor<2x32xf32>) -> tensor<2x4x8xf32>
    %117 = stablehlo.reshape %114 : (tensor<2x16xf32>) -> tensor<2x2x8xf32>
    %118 = stablehlo.reshape %115 : (tensor<2x16xf32>) -> tensor<2x2x8xf32>
    %119 = stablehlo.multiply %116, %47 : tensor<2x4x8xf32>
    %120 = stablehlo.slice %116 [0:2, 0:4, 4:8] : (tensor<2x4x8xf32>) -> tensor<2x4x4xf32>
    %121 = stablehlo.slice %116 [0:2, 0:4, 0:4] : (tensor<2x4x8xf32>) -> tensor<2x4x4xf32>
    %122 = stablehlo.negate %120 : tensor<2x4x4xf32>
    %123 = stablehlo.concatenate %122, %121, dim = 2 : (tensor<2x4x4xf32>, tensor<2x4x4xf32>) -> tensor<2x4x8xf32>
    %124 = stablehlo.multiply %123, %49 : tensor<2x4x8xf32>
    %125 = stablehlo.add %119, %124 : tensor<2x4x8xf32>
    %126 = stablehlo.multiply %117, %51 : tensor<2x2x8xf32>
    %127 = stablehlo.slice %117 [0:2, 0:2, 4:8] : (tensor<2x2x8xf32>) -> tensor<2x2x4xf32>
    %128 = stablehlo.slice %117 [0:2, 0:2, 0:4] : (tensor<2x2x8xf32>) -> tensor<2x2x4xf32>
    %129 = stablehlo.negate %127 : tensor<2x2x4xf32>
    %130 = stablehlo.concatenate %129, %128, dim = 2 : (tensor<2x2x4xf32>, tensor<2x2x4xf32>) -> tensor<2x2x8xf32>
    %131 = stablehlo.multiply %130, %53 : tensor<2x2x8xf32>
    %132 = stablehlo.add %126, %131 : tensor<2x2x8xf32>
    %s265 = stablehlo.reshape %7 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s266 = "stablehlo.scatter"(%s265, %4, %132) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s267: tensor<f32>, %s268: tensor<f32>):
       stablehlo.return %s268 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<2xi32>, tensor<2x2x8xf32>) -> tensor<260x2x8xf32>
    %133 = stablehlo.reshape %s266 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s269 = stablehlo.reshape %8 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s270 = "stablehlo.scatter"(%s269, %4, %118) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s271: tensor<f32>, %s272: tensor<f32>):
       stablehlo.return %s272 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<2xi32>, tensor<2x2x8xf32>) -> tensor<260x2x8xf32>
    %134 = stablehlo.reshape %s270 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s273 = "stablehlo.gather"(%133, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<2x8xi32>) -> tensor<2x8x4x2x8xf32>
    %s274 = stablehlo.reshape %s273 : (tensor<2x8x4x2x8xf32>) -> tensor<2x32x2x8xf32>
    %s275 = "stablehlo.gather"(%134, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<2x8xi32>) -> tensor<2x8x4x2x8xf32>
    %s276 = stablehlo.reshape %s275 : (tensor<2x8x4x2x8xf32>) -> tensor<2x32x2x8xf32>
    %s277 = stablehlo.reshape %125 : (tensor<2x4x8xf32>) -> tensor<2x2x2x8xf32>
    %s278 = stablehlo.dot_general %s277, %s274, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [3], precision = [HIGHEST, HIGHEST] : (tensor<2x2x2x8xf32>, tensor<2x32x2x8xf32>) -> tensor<2x2x2x32xf32>
    %s279 = stablehlo.constant dense<0.35355338> : tensor<f32>
    %s280 = stablehlo.broadcast_in_dim %s279, dims = [] : (tensor<f32>) -> tensor<2x2x2x32xf32>
    %s281 = stablehlo.multiply %s278, %s280 : tensor<2x2x2x32xf32>
    %s282 = stablehlo.iota dim = 3 : tensor<2x2x2x32xi32>
    %s283 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<2xi32>) -> tensor<2x2x2x32xi32>
    %s285 = stablehlo.compare LT, %s282, %s283, SIGNED : (tensor<2x2x2x32xi32>, tensor<2x2x2x32xi32>) -> tensor<2x2x2x32xi1>
    %s286 = stablehlo.constant dense<8> : tensor<i32>
    %s287 = stablehlo.broadcast_in_dim %s286, dims = [] : (tensor<i32>) -> tensor<2x2x2x32xi32>
    %s288 = stablehlo.subtract %s283, %s287 : tensor<2x2x2x32xi32>
    %s289 = stablehlo.compare GE, %s282, %s288, SIGNED : (tensor<2x2x2x32xi32>, tensor<2x2x2x32xi32>) -> tensor<2x2x2x32xi1>
    %s284 = stablehlo.and %s285, %s289 : tensor<2x2x2x32xi1>
    %s290 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s291 = stablehlo.broadcast_in_dim %s290, dims = [] : (tensor<f32>) -> tensor<2x2x2x32xf32>
    %s292 = stablehlo.select %s284, %s281, %s291 : tensor<2x2x2x32xi1>, tensor<2x2x2x32xf32>
    %s293 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s294 = stablehlo.reduce(%s292 init: %s293) applies stablehlo.maximum across dimensions = [3] : (tensor<2x2x2x32xf32>, tensor<f32>) -> tensor<2x2x2xf32>
    %s295 = stablehlo.broadcast_in_dim %s294, dims = [0, 1, 2] : (tensor<2x2x2xf32>) -> tensor<2x2x2x32xf32>
    %s296 = stablehlo.subtract %s292, %s295 : tensor<2x2x2x32xf32>
    %s297 = stablehlo.exponential %s296 : tensor<2x2x2x32xf32>
    %s298 = stablehlo.constant dense<0.0> : tensor<f32>
    %s299 = stablehlo.reduce(%s297 init: %s298) applies stablehlo.add across dimensions = [3] : (tensor<2x2x2x32xf32>, tensor<f32>) -> tensor<2x2x2xf32>
    %s300 = stablehlo.broadcast_in_dim %s299, dims = [0, 1, 2] : (tensor<2x2x2xf32>) -> tensor<2x2x2x32xf32>
    %s301 = stablehlo.divide %s297, %s300 : tensor<2x2x2x32xf32>
    %s302 = stablehlo.dot_general %s301, %s276, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [HIGHEST, HIGHEST] : (tensor<2x2x2x32xf32>, tensor<2x32x2x8xf32>) -> tensor<2x2x2x8xf32>
    %135 = stablehlo.reshape %s302 : (tensor<2x2x2x8xf32>) -> tensor<2x4x8xf32>
    %136 = stablehlo.reshape %135 : (tensor<2x4x8xf32>) -> tensor<2x32xf32>
    %137 = stablehlo.dot_general %136, %25, contracting_dims = [1] x [0] : (tensor<2x32xf32>, tensor<32x32xf32>) -> tensor<2x32xf32>
    %138 = stablehlo.add %104, %137 : tensor<2x32xf32>
    %139 = stablehlo.multiply %138, %138 : tensor<2x32xf32>
    %s303 = stablehlo.constant dense<0.0> : tensor<f32>
    %s304 = stablehlo.reduce(%139 init: %s303) applies stablehlo.add across dimensions = [1] : (tensor<2x32xf32>, tensor<f32>) -> tensor<2xf32>
    %s305 = stablehlo.constant dense<32.0> : tensor<2xf32>
    %s306 = stablehlo.divide %s304, %s305 : tensor<2xf32>
    %140 = stablehlo.broadcast_in_dim %s306, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %141 = stablehlo.add %140, %56 : tensor<2x1xf32>
    %142 = stablehlo.rsqrt %141 : tensor<2x1xf32>
    %s307 = stablehlo.broadcast_in_dim %142, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x32xf32>
    %143 = stablehlo.multiply %138, %s307 : tensor<2x32xf32>
    %144 = stablehlo.reshape %26 : (tensor<32xf32>) -> tensor<1x32xf32>
    %145 = stablehlo.broadcast_in_dim %144, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<2x32xf32>
    %146 = stablehlo.multiply %143, %145 : tensor<2x32xf32>
    %147 = stablehlo.dot_general %146, %27, contracting_dims = [1] x [0] : (tensor<2x32xf32>, tensor<32x48xf32>) -> tensor<2x48xf32>
    %148 = stablehlo.dot_general %146, %28, contracting_dims = [1] x [0] : (tensor<2x32xf32>, tensor<32x48xf32>) -> tensor<2x48xf32>
    %s308 = stablehlo.logistic %147 : tensor<2x48xf32>
    %149 = stablehlo.multiply %147, %s308 : tensor<2x48xf32>
    %150 = stablehlo.multiply %149, %148 : tensor<2x48xf32>
    %151 = stablehlo.dot_general %150, %29, contracting_dims = [1] x [0] : (tensor<2x48xf32>, tensor<48x32xf32>) -> tensor<2x32xf32>
    %152 = stablehlo.add %138, %151 : tensor<2x32xf32>
    %153 = stablehlo.multiply %152, %152 : tensor<2x32xf32>
    %s309 = stablehlo.constant dense<0.0> : tensor<f32>
    %s310 = stablehlo.reduce(%153 init: %s309) applies stablehlo.add across dimensions = [1] : (tensor<2x32xf32>, tensor<f32>) -> tensor<2xf32>
    %s311 = stablehlo.constant dense<32.0> : tensor<2xf32>
    %s312 = stablehlo.divide %s310, %s311 : tensor<2xf32>
    %154 = stablehlo.broadcast_in_dim %s312, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %155 = stablehlo.add %154, %56 : tensor<2x1xf32>
    %156 = stablehlo.rsqrt %155 : tensor<2x1xf32>
    %s313 = stablehlo.broadcast_in_dim %156, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x32xf32>
    %157 = stablehlo.multiply %152, %s313 : tensor<2x32xf32>
    %158 = stablehlo.reshape %30 : (tensor<32xf32>) -> tensor<1x32xf32>
    %159 = stablehlo.broadcast_in_dim %158, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<2x32xf32>
    %160 = stablehlo.multiply %157, %159 : tensor<2x32xf32>
    %161 = stablehlo.dot_general %160, %31, contracting_dims = [1] x [0] : (tensor<2x32xf32>, tensor<32x32xf32>) -> tensor<2x32xf32>
    %162 = stablehlo.dot_general %160, %32, contracting_dims = [1] x [0] : (tensor<2x32xf32>, tensor<32x16xf32>) -> tensor<2x16xf32>
    %163 = stablehlo.dot_general %160, %33, contracting_dims = [1] x [0] : (tensor<2x32xf32>, tensor<32x16xf32>) -> tensor<2x16xf32>
    %164 = stablehlo.reshape %161 : (tensor<2x32xf32>) -> tensor<2x4x8xf32>
    %165 = stablehlo.reshape %162 : (tensor<2x16xf32>) -> tensor<2x2x8xf32>
    %166 = stablehlo.reshape %163 : (tensor<2x16xf32>) -> tensor<2x2x8xf32>
    %167 = stablehlo.multiply %164, %47 : tensor<2x4x8xf32>
    %168 = stablehlo.slice %164 [0:2, 0:4, 4:8] : (tensor<2x4x8xf32>) -> tensor<2x4x4xf32>
    %169 = stablehlo.slice %164 [0:2, 0:4, 0:4] : (tensor<2x4x8xf32>) -> tensor<2x4x4xf32>
    %170 = stablehlo.negate %168 : tensor<2x4x4xf32>
    %171 = stablehlo.concatenate %170, %169, dim = 2 : (tensor<2x4x4xf32>, tensor<2x4x4xf32>) -> tensor<2x4x8xf32>
    %172 = stablehlo.multiply %171, %49 : tensor<2x4x8xf32>
    %173 = stablehlo.add %167, %172 : tensor<2x4x8xf32>
    %174 = stablehlo.multiply %165, %51 : tensor<2x2x8xf32>
    %175 = stablehlo.slice %165 [0:2, 0:2, 4:8] : (tensor<2x2x8xf32>) -> tensor<2x2x4xf32>
    %176 = stablehlo.slice %165 [0:2, 0:2, 0:4] : (tensor<2x2x8xf32>) -> tensor<2x2x4xf32>
    %177 = stablehlo.negate %175 : tensor<2x2x4xf32>
    %178 = stablehlo.concatenate %177, %176, dim = 2 : (tensor<2x2x4xf32>, tensor<2x2x4xf32>) -> tensor<2x2x8xf32>
    %179 = stablehlo.multiply %178, %53 : tensor<2x2x8xf32>
    %180 = stablehlo.add %174, %179 : tensor<2x2x8xf32>
    %s314 = stablehlo.reshape %9 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s315 = "stablehlo.scatter"(%s314, %4, %180) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s316: tensor<f32>, %s317: tensor<f32>):
       stablehlo.return %s317 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<2xi32>, tensor<2x2x8xf32>) -> tensor<260x2x8xf32>
    %181 = stablehlo.reshape %s315 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s318 = stablehlo.reshape %10 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s319 = "stablehlo.scatter"(%s318, %4, %166) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s320: tensor<f32>, %s321: tensor<f32>):
       stablehlo.return %s321 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<2xi32>, tensor<2x2x8xf32>) -> tensor<260x2x8xf32>
    %182 = stablehlo.reshape %s319 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s322 = "stablehlo.gather"(%181, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<2x8xi32>) -> tensor<2x8x4x2x8xf32>
    %s323 = stablehlo.reshape %s322 : (tensor<2x8x4x2x8xf32>) -> tensor<2x32x2x8xf32>
    %s324 = "stablehlo.gather"(%182, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<2x8xi32>) -> tensor<2x8x4x2x8xf32>
    %s325 = stablehlo.reshape %s324 : (tensor<2x8x4x2x8xf32>) -> tensor<2x32x2x8xf32>
    %s326 = stablehlo.reshape %173 : (tensor<2x4x8xf32>) -> tensor<2x2x2x8xf32>
    %s327 = stablehlo.dot_general %s326, %s323, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [3], precision = [HIGHEST, HIGHEST] : (tensor<2x2x2x8xf32>, tensor<2x32x2x8xf32>) -> tensor<2x2x2x32xf32>
    %s328 = stablehlo.constant dense<0.35355338> : tensor<f32>
    %s329 = stablehlo.broadcast_in_dim %s328, dims = [] : (tensor<f32>) -> tensor<2x2x2x32xf32>
    %s330 = stablehlo.multiply %s327, %s329 : tensor<2x2x2x32xf32>
    %s331 = stablehlo.iota dim = 3 : tensor<2x2x2x32xi32>
    %s332 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<2xi32>) -> tensor<2x2x2x32xi32>
    %s333 = stablehlo.compare LT, %s331, %s332, SIGNED : (tensor<2x2x2x32xi32>, tensor<2x2x2x32xi32>) -> tensor<2x2x2x32xi1>
    %s334 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s335 = stablehlo.broadcast_in_dim %s334, dims = [] : (tensor<f32>) -> tensor<2x2x2x32xf32>
    %s336 = stablehlo.select %s333, %s330, %s335 : tensor<2x2x2x32xi1>, tensor<2x2x2x32xf32>
    %s337 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s338 = stablehlo.reduce(%s336 init: %s337) applies stablehlo.maximum across dimensions = [3] : (tensor<2x2x2x32xf32>, tensor<f32>) -> tensor<2x2x2xf32>
    %s339 = stablehlo.broadcast_in_dim %s338, dims = [0, 1, 2] : (tensor<2x2x2xf32>) -> tensor<2x2x2x32xf32>
    %s340 = stablehlo.subtract %s336, %s339 : tensor<2x2x2x32xf32>
    %s341 = stablehlo.exponential %s340 : tensor<2x2x2x32xf32>
    %s342 = stablehlo.constant dense<0.0> : tensor<f32>
    %s343 = stablehlo.reduce(%s341 init: %s342) applies stablehlo.add across dimensions = [3] : (tensor<2x2x2x32xf32>, tensor<f32>) -> tensor<2x2x2xf32>
    %s344 = stablehlo.broadcast_in_dim %s343, dims = [0, 1, 2] : (tensor<2x2x2xf32>) -> tensor<2x2x2x32xf32>
    %s345 = stablehlo.divide %s341, %s344 : tensor<2x2x2x32xf32>
    %s346 = stablehlo.dot_general %s345, %s325, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [HIGHEST, HIGHEST] : (tensor<2x2x2x32xf32>, tensor<2x32x2x8xf32>) -> tensor<2x2x2x8xf32>
    %183 = stablehlo.reshape %s346 : (tensor<2x2x2x8xf32>) -> tensor<2x4x8xf32>
    %184 = stablehlo.reshape %183 : (tensor<2x4x8xf32>) -> tensor<2x32xf32>
    %185 = stablehlo.dot_general %184, %34, contracting_dims = [1] x [0] : (tensor<2x32xf32>, tensor<32x32xf32>) -> tensor<2x32xf32>
    %186 = stablehlo.add %152, %185 : tensor<2x32xf32>
    %187 = stablehlo.multiply %186, %186 : tensor<2x32xf32>
    %s347 = stablehlo.constant dense<0.0> : tensor<f32>
    %s348 = stablehlo.reduce(%187 init: %s347) applies stablehlo.add across dimensions = [1] : (tensor<2x32xf32>, tensor<f32>) -> tensor<2xf32>
    %s349 = stablehlo.constant dense<32.0> : tensor<2xf32>
    %s350 = stablehlo.divide %s348, %s349 : tensor<2xf32>
    %188 = stablehlo.broadcast_in_dim %s350, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %189 = stablehlo.add %188, %56 : tensor<2x1xf32>
    %190 = stablehlo.rsqrt %189 : tensor<2x1xf32>
    %s351 = stablehlo.broadcast_in_dim %190, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x32xf32>
    %191 = stablehlo.multiply %186, %s351 : tensor<2x32xf32>
    %192 = stablehlo.reshape %35 : (tensor<32xf32>) -> tensor<1x32xf32>
    %193 = stablehlo.broadcast_in_dim %192, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<2x32xf32>
    %194 = stablehlo.multiply %191, %193 : tensor<2x32xf32>
    %195 = stablehlo.dot_general %194, %36, contracting_dims = [1] x [0] : (tensor<2x32xf32>, tensor<32x48xf32>) -> tensor<2x48xf32>
    %196 = stablehlo.dot_general %194, %37, contracting_dims = [1] x [0] : (tensor<2x32xf32>, tensor<32x48xf32>) -> tensor<2x48xf32>
    %s352 = stablehlo.logistic %195 : tensor<2x48xf32>
    %197 = stablehlo.multiply %195, %s352 : tensor<2x48xf32>
    %198 = stablehlo.multiply %197, %196 : tensor<2x48xf32>
    %199 = stablehlo.dot_general %198, %38, contracting_dims = [1] x [0] : (tensor<2x48xf32>, tensor<48x32xf32>) -> tensor<2x32xf32>
    %200 = stablehlo.add %186, %199 : tensor<2x32xf32>
    %201 = stablehlo.multiply %200, %200 : tensor<2x32xf32>
    %s353 = stablehlo.constant dense<0.0> : tensor<f32>
    %s354 = stablehlo.reduce(%201 init: %s353) applies stablehlo.add across dimensions = [1] : (tensor<2x32xf32>, tensor<f32>) -> tensor<2xf32>
    %s355 = stablehlo.constant dense<32.0> : tensor<2xf32>
    %s356 = stablehlo.divide %s354, %s355 : tensor<2xf32>
    %202 = stablehlo.broadcast_in_dim %s356, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %203 = stablehlo.add %202, %56 : tensor<2x1xf32>
    %204 = stablehlo.rsqrt %203 : tensor<2x1xf32>
    %s357 = stablehlo.broadcast_in_dim %204, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x32xf32>
    %205 = stablehlo.multiply %200, %s357 : tensor<2x32xf32>
    %206 = stablehlo.reshape %39 : (tensor<32xf32>) -> tensor<1x32xf32>
    %207 = stablehlo.broadcast_in_dim %206, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<2x32xf32>
    %208 = stablehlo.multiply %205, %207 : tensor<2x32xf32>
    %209 = stablehlo.dot_general %208, %40, contracting_dims = [1] x [0] : (tensor<2x32xf32>, tensor<32x64xf32>) -> tensor<2x64xf32>
    %210 = stablehlo.reshape %209 : (tensor<2x64xf32>) -> tensor<2x1x64xf32>
    return %210, %85, %86, %133, %134, %181, %182 : tensor<2x1x64xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>
  }
}
