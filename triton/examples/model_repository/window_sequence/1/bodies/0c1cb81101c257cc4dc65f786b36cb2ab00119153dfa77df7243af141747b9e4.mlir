module {
  func.func @main(%0: tensor<2x32xi32>, %1: tensor<2x32xi32>, %2: tensor<2x8xi32>, %3: tensor<2xi32>, %4: tensor<64xi32>, %5: tensor<2x8xi32>, %6: tensor<64xi32>, %7: tensor<13x4x2x8xf32> {tf.aliasing_output = 1 : i32}, %8: tensor<13x4x2x8xf32> {tf.aliasing_output = 2 : i32}, %9: tensor<13x4x2x8xf32> {tf.aliasing_output = 3 : i32}, %10: tensor<13x4x2x8xf32> {tf.aliasing_output = 4 : i32}, %11: tensor<65x4x2x8xf32> {tf.aliasing_output = 5 : i32}, %12: tensor<65x4x2x8xf32> {tf.aliasing_output = 6 : i32}, %13: tensor<64x32xf32>, %14: tensor<32xf32>, %15: tensor<32x32xf32>, %16: tensor<32x16xf32>, %17: tensor<32x16xf32>, %18: tensor<32x32xf32>, %19: tensor<32xf32>, %20: tensor<32x48xf32>, %21: tensor<32x48xf32>, %22: tensor<48x32xf32>, %23: tensor<32xf32>, %24: tensor<32x32xf32>, %25: tensor<32x16xf32>, %26: tensor<32x16xf32>, %27: tensor<32x32xf32>, %28: tensor<32xf32>, %29: tensor<32x48xf32>, %30: tensor<32x48xf32>, %31: tensor<48x32xf32>, %32: tensor<32xf32>, %33: tensor<32x32xf32>, %34: tensor<32x16xf32>, %35: tensor<32x16xf32>, %36: tensor<32x32xf32>, %37: tensor<32xf32>, %38: tensor<32x48xf32>, %39: tensor<32x48xf32>, %40: tensor<48x32xf32>, %41: tensor<32xf32>, %42: tensor<32x64xf32>) -> (tensor<2x1x64xf32>, tensor<13x4x2x8xf32>, tensor<13x4x2x8xf32>, tensor<13x4x2x8xf32>, tensor<13x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>) {
    %43 = stablehlo.reshape %1 : (tensor<2x32xi32>) -> tensor<64xi32>
    %44 = stablehlo.constant dense<[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.5403023, 0.9950042, 0.99995, 0.9999995, 0.5403023, 0.9950042, 0.99995, 0.9999995], [-0.41614684, 0.9800666, 0.9998, 0.999998, -0.41614684, 0.9800666, 0.9998, 0.999998], [-0.9899925, 0.9553365, 0.99955004, 0.9999955, -0.9899925, 0.9553365, 0.99955004, 0.9999955], [-0.6536436, 0.921061, 0.9992001, 0.999992, -0.6536436, 0.921061, 0.9992001, 0.999992], [0.2836622, 0.87758255, 0.99875027, 0.9999875, 0.2836622, 0.87758255, 0.99875027, 0.9999875], [0.96017027, 0.8253356, 0.99820054, 0.999982, 0.96017027, 0.8253356, 0.99820054, 0.999982], [0.75390226, 0.7648422, 0.997551, 0.9999755, 0.75390226, 0.7648422, 0.997551, 0.9999755], [-0.14550003, 0.6967067, 0.99680173, 0.999968, -0.14550003, 0.6967067, 0.99680173, 0.999968], [-0.91113025, 0.62161, 0.9959527, 0.9999595, -0.91113025, 0.62161, 0.9959527, 0.9999595], [-0.8390715, 0.5403023, 0.9950042, 0.99995, -0.8390715, 0.5403023, 0.9950042, 0.99995], [0.004425698, 0.45359612, 0.9939561, 0.9999395, 0.004425698, 0.45359612, 0.9939561, 0.9999395], [0.84385395, 0.36235777, 0.99280864, 0.999928, 0.84385395, 0.36235777, 0.99280864, 0.999928], [0.9074468, 0.26749882, 0.9915619, 0.9999155, 0.9074468, 0.26749882, 0.9915619, 0.9999155], [0.13673721, 0.16996714, 0.990216, 0.999902, 0.13673721, 0.16996714, 0.990216, 0.999902], [-0.7596879, 0.0707372, 0.9887711, 0.9998875, -0.7596879, 0.0707372, 0.9887711, 0.9998875], [-0.9576595, -0.029199522, 0.98722726, 0.999872, -0.9576595, -0.029199522, 0.98722726, 0.999872], [-0.27516335, -0.1288445, 0.9855848, 0.9998555, -0.27516335, -0.1288445, 0.9855848, 0.9998555], [0.6603167, -0.22720209, 0.9838437, 0.999838, 0.6603167, -0.22720209, 0.9838437, 0.999838], [0.9887046, -0.32328957, 0.9820042, 0.9998195, 0.9887046, -0.32328957, 0.9820042, 0.9998195], [0.40808207, -0.41614684, 0.9800666, 0.9998, 0.40808207, -0.41614684, 0.9800666, 0.9998], [-0.54772925, -0.5048461, 0.9780309, 0.9997795, -0.54772925, -0.5048461, 0.9780309, 0.9997795], [-0.99996084, -0.5885011, 0.97589743, 0.999758, -0.99996084, -0.5885011, 0.97589743, 0.999758], [-0.53283304, -0.66627604, 0.97366637, 0.99973553, -0.53283304, -0.66627604, 0.97366637, 0.99973553], [0.42417902, -0.73739374, 0.971338, 0.999712, 0.42417902, -0.73739374, 0.971338, 0.999712], [0.99120283, -0.8011436, 0.9689124, 0.9996875, 0.99120283, -0.8011436, 0.9689124, 0.9996875], [0.6469193, -0.8568888, 0.96638995, 0.99966204, 0.6469193, -0.8568888, 0.96638995, 0.99966204], [-0.29213881, -0.90407217, 0.9637709, 0.9996355, -0.29213881, -0.90407217, 0.9637709, 0.9996355], [-0.9626059, -0.94222236, 0.96105546, 0.99960804, -0.9626059, -0.94222236, 0.96105546, 0.99960804], [-0.74805754, -0.9709582, 0.95824385, 0.99957955, -0.74805754, -0.9709582, 0.95824385, 0.99957955], [0.15425146, -0.9899925, 0.9553365, 0.99955004, 0.15425146, -0.9899925, 0.9553365, 0.99955004], [0.91474235, -0.99913514, 0.95233357, 0.9995195, 0.91474235, -0.99913514, 0.95233357, 0.9995195]]> : tensor<32x8xf32>
    %45 = stablehlo.constant dense<[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.84147096, 0.099833414, 0.009999833, 9.999998E-4, 0.84147096, 0.099833414, 0.009999833, 9.999998E-4], [0.9092974, 0.19866933, 0.019998666, 0.0019999987, 0.9092974, 0.19866933, 0.019998666, 0.0019999987], [0.14112, 0.29552022, 0.029995501, 0.0029999956, 0.14112, 0.29552022, 0.029995501, 0.0029999956], [-0.7568025, 0.38941833, 0.039989334, 0.0039999895, -0.7568025, 0.38941833, 0.039989334, 0.0039999895], [-0.9589243, 0.47942555, 0.04997917, 0.0049999794, -0.9589243, 0.47942555, 0.04997917, 0.0049999794], [-0.2794155, 0.5646425, 0.059964005, 0.005999964, -0.2794155, 0.5646425, 0.059964005, 0.005999964], [0.6569866, 0.64421767, 0.06994285, 0.006999943, 0.6569866, 0.64421767, 0.06994285, 0.006999943], [0.98935825, 0.7173561, 0.0799147, 0.007999915, 0.98935825, 0.7173561, 0.0799147, 0.007999915], [0.4121185, 0.7833269, 0.08987855, 0.008999879, 0.4121185, 0.7833269, 0.08987855, 0.008999879], [-0.5440211, 0.84147096, 0.099833414, 0.009999833, -0.5440211, 0.84147096, 0.099833414, 0.009999833], [-0.9999902, 0.89120734, 0.1097783, 0.010999778, -0.9999902, 0.89120734, 0.1097783, 0.010999778], [-0.53657293, 0.9320391, 0.119712204, 0.011999712, -0.53657293, 0.9320391, 0.119712204, 0.011999712], [0.42016703, 0.9635582, 0.12963414, 0.012999634, 0.42016703, 0.9635582, 0.12963414, 0.012999634], [0.9906074, 0.98544973, 0.13954312, 0.013999542, 0.9906074, 0.98544973, 0.13954312, 0.013999542], [0.65028787, 0.997495, 0.14943813, 0.014999437, 0.65028787, 0.997495, 0.14943813, 0.014999437], [-0.2879033, 0.9995736, 0.15931821, 0.015999317, -0.2879033, 0.9995736, 0.15931821, 0.015999317], [-0.96139747, 0.9916648, 0.16918235, 0.016999181, -0.96139747, 0.9916648, 0.16918235, 0.016999181], [-0.75098723, 0.9738476, 0.17902957, 0.017999029, -0.75098723, 0.9738476, 0.17902957, 0.017999029], [0.1498772, 0.9463001, 0.1888589, 0.018998858, 0.1498772, 0.9463001, 0.1888589, 0.018998858], [0.9129453, 0.9092974, 0.19866933, 0.019998666, 0.9129453, 0.9092974, 0.19866933, 0.019998666], [0.8366556, 0.86320937, 0.2084599, 0.020998457, 0.8366556, 0.86320937, 0.2084599, 0.020998457], [-0.008851309, 0.8084964, 0.21822962, 0.021998225, -0.008851309, 0.8084964, 0.21822962, 0.021998225], [-0.84622043, 0.7457052, 0.22797753, 0.022997972, -0.84622043, 0.7457052, 0.22797753, 0.022997972], [-0.9055784, 0.6754632, 0.23770262, 0.023997696, -0.9055784, 0.6754632, 0.23770262, 0.023997696], [-0.13235176, 0.5984721, 0.24740396, 0.024997396, -0.13235176, 0.5984721, 0.24740396, 0.024997396], [0.76255846, 0.5155014, 0.25708055, 0.02599707, 0.76255846, 0.5155014, 0.25708055, 0.02599707], [0.95637596, 0.42737988, 0.26673144, 0.026996719, 0.95637596, 0.42737988, 0.26673144, 0.026996719], [0.2709058, 0.33498815, 0.27635565, 0.02799634, 0.2709058, 0.33498815, 0.27635565, 0.02799634], [-0.6636339, 0.23924933, 0.2859522, 0.028995935, -0.6636339, 0.23924933, 0.2859522, 0.028995935], [-0.9880316, 0.14112, 0.29552022, 0.029995501, -0.9880316, 0.14112, 0.29552022, 0.029995501], [-0.40403765, 0.041580662, 0.30505863, 0.030995036, -0.40403765, 0.041580662, 0.30505863, 0.030995036]]> : tensor<32x8xf32>
    %46 = "stablehlo.gather"(%44, %43) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 8>}> : (tensor<32x8xf32>, tensor<64xi32>) -> tensor<64x8xf32>
    %47 = "stablehlo.gather"(%45, %43) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 8>}> : (tensor<32x8xf32>, tensor<64xi32>) -> tensor<64x8xf32>
    %48 = stablehlo.reshape %46 : (tensor<64x8xf32>) -> tensor<64x1x8xf32>
    %49 = stablehlo.broadcast_in_dim %48, dims = [0, 1, 2] : (tensor<64x1x8xf32>) -> tensor<64x4x8xf32>
    %50 = stablehlo.reshape %47 : (tensor<64x8xf32>) -> tensor<64x1x8xf32>
    %51 = stablehlo.broadcast_in_dim %50, dims = [0, 1, 2] : (tensor<64x1x8xf32>) -> tensor<64x4x8xf32>
    %52 = stablehlo.reshape %46 : (tensor<64x8xf32>) -> tensor<64x1x8xf32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [0, 1, 2] : (tensor<64x1x8xf32>) -> tensor<64x2x8xf32>
    %54 = stablehlo.reshape %47 : (tensor<64x8xf32>) -> tensor<64x1x8xf32>
    %55 = stablehlo.broadcast_in_dim %54, dims = [0, 1, 2] : (tensor<64x1x8xf32>) -> tensor<64x2x8xf32>
    %56 = stablehlo.reshape %2 : (tensor<2x8xi32>) -> tensor<2x1x8xi32>
    %57 = stablehlo.broadcast_in_dim %56, dims = [0, 1, 2] : (tensor<2x1x8xi32>) -> tensor<2x32x8xi32>
    %58 = stablehlo.reshape %57 : (tensor<2x32x8xi32>) -> tensor<64x8xi32>
    %59 = stablehlo.reshape %5 : (tensor<2x8xi32>) -> tensor<2x1x8xi32>
    %60 = stablehlo.broadcast_in_dim %59, dims = [0, 1, 2] : (tensor<2x1x8xi32>) -> tensor<2x32x8xi32>
    %61 = stablehlo.reshape %60 : (tensor<2x32x8xi32>) -> tensor<64x8xi32>
    %62 = stablehlo.constant dense<1> : tensor<64xi32>
    %63 = stablehlo.add %43, %62 : tensor<64xi32>
    %64 = "stablehlo.gather"(%13, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 32>}> : (tensor<64x32xf32>, tensor<2x32xi32>) -> tensor<2x32x32xf32>
    %65 = stablehlo.reshape %64 : (tensor<2x32x32xf32>) -> tensor<64x32xf32>
    %66 = stablehlo.constant dense<[[1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6]]> : tensor<64x1xf32>
    %67 = stablehlo.multiply %65, %65 : tensor<64x32xf32>
    %s225 = stablehlo.constant dense<0.0> : tensor<f32>
    %s226 = stablehlo.reduce(%67 init: %s225) applies stablehlo.add across dimensions = [1] : (tensor<64x32xf32>, tensor<f32>) -> tensor<64xf32>
    %s227 = stablehlo.constant dense<32.0> : tensor<64xf32>
    %s228 = stablehlo.divide %s226, %s227 : tensor<64xf32>
    %68 = stablehlo.broadcast_in_dim %s228, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %69 = stablehlo.add %68, %66 : tensor<64x1xf32>
    %70 = stablehlo.rsqrt %69 : tensor<64x1xf32>
    %s229 = stablehlo.broadcast_in_dim %70, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x32xf32>
    %71 = stablehlo.multiply %65, %s229 : tensor<64x32xf32>
    %72 = stablehlo.reshape %14 : (tensor<32xf32>) -> tensor<1x32xf32>
    %73 = stablehlo.broadcast_in_dim %72, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<64x32xf32>
    %74 = stablehlo.multiply %71, %73 : tensor<64x32xf32>
    %75 = stablehlo.dot_general %74, %15, contracting_dims = [1] x [0] : (tensor<64x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    %76 = stablehlo.dot_general %74, %16, contracting_dims = [1] x [0] : (tensor<64x32xf32>, tensor<32x16xf32>) -> tensor<64x16xf32>
    %77 = stablehlo.dot_general %74, %17, contracting_dims = [1] x [0] : (tensor<64x32xf32>, tensor<32x16xf32>) -> tensor<64x16xf32>
    %78 = stablehlo.reshape %75 : (tensor<64x32xf32>) -> tensor<64x4x8xf32>
    %79 = stablehlo.reshape %76 : (tensor<64x16xf32>) -> tensor<64x2x8xf32>
    %80 = stablehlo.reshape %77 : (tensor<64x16xf32>) -> tensor<64x2x8xf32>
    %81 = stablehlo.multiply %78, %49 : tensor<64x4x8xf32>
    %82 = stablehlo.slice %78 [0:64, 0:4, 4:8] : (tensor<64x4x8xf32>) -> tensor<64x4x4xf32>
    %83 = stablehlo.slice %78 [0:64, 0:4, 0:4] : (tensor<64x4x8xf32>) -> tensor<64x4x4xf32>
    %84 = stablehlo.negate %82 : tensor<64x4x4xf32>
    %85 = stablehlo.concatenate %84, %83, dim = 2 : (tensor<64x4x4xf32>, tensor<64x4x4xf32>) -> tensor<64x4x8xf32>
    %86 = stablehlo.multiply %85, %51 : tensor<64x4x8xf32>
    %87 = stablehlo.add %81, %86 : tensor<64x4x8xf32>
    %88 = stablehlo.multiply %79, %53 : tensor<64x2x8xf32>
    %89 = stablehlo.slice %79 [0:64, 0:2, 4:8] : (tensor<64x2x8xf32>) -> tensor<64x2x4xf32>
    %90 = stablehlo.slice %79 [0:64, 0:2, 0:4] : (tensor<64x2x8xf32>) -> tensor<64x2x4xf32>
    %91 = stablehlo.negate %89 : tensor<64x2x4xf32>
    %92 = stablehlo.concatenate %91, %90, dim = 2 : (tensor<64x2x4xf32>, tensor<64x2x4xf32>) -> tensor<64x2x8xf32>
    %93 = stablehlo.multiply %92, %55 : tensor<64x2x8xf32>
    %94 = stablehlo.add %88, %93 : tensor<64x2x8xf32>
    %s230 = stablehlo.reshape %7 : (tensor<13x4x2x8xf32>) -> tensor<52x2x8xf32>
    %s231 = "stablehlo.scatter"(%s230, %6, %94) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s232: tensor<f32>, %s233: tensor<f32>):
       stablehlo.return %s233 : tensor<f32>
     }) : (tensor<52x2x8xf32>, tensor<64xi32>, tensor<64x2x8xf32>) -> tensor<52x2x8xf32>
    %95 = stablehlo.reshape %s231 : (tensor<52x2x8xf32>) -> tensor<13x4x2x8xf32>
    %s234 = stablehlo.reshape %8 : (tensor<13x4x2x8xf32>) -> tensor<52x2x8xf32>
    %s235 = "stablehlo.scatter"(%s234, %6, %80) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s236: tensor<f32>, %s237: tensor<f32>):
       stablehlo.return %s237 : tensor<f32>
     }) : (tensor<52x2x8xf32>, tensor<64xi32>, tensor<64x2x8xf32>) -> tensor<52x2x8xf32>
    %96 = stablehlo.reshape %s235 : (tensor<52x2x8xf32>) -> tensor<13x4x2x8xf32>
    %s238 = "stablehlo.gather"(%95, %61) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<13x4x2x8xf32>, tensor<64x8xi32>) -> tensor<64x8x4x2x8xf32>
    %s239 = stablehlo.reshape %s238 : (tensor<64x8x4x2x8xf32>) -> tensor<64x32x2x8xf32>
    %s240 = "stablehlo.gather"(%96, %61) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<13x4x2x8xf32>, tensor<64x8xi32>) -> tensor<64x8x4x2x8xf32>
    %s241 = stablehlo.reshape %s240 : (tensor<64x8x4x2x8xf32>) -> tensor<64x32x2x8xf32>
    %s242 = stablehlo.reshape %87 : (tensor<64x4x8xf32>) -> tensor<64x2x2x8xf32>
    %s243 = stablehlo.dot_general %s242, %s239, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [3], precision = [HIGHEST, HIGHEST] : (tensor<64x2x2x8xf32>, tensor<64x32x2x8xf32>) -> tensor<64x2x2x32xf32>
    %s244 = stablehlo.constant dense<0.35355338> : tensor<f32>
    %s245 = stablehlo.broadcast_in_dim %s244, dims = [] : (tensor<f32>) -> tensor<64x2x2x32xf32>
    %s246 = stablehlo.multiply %s243, %s245 : tensor<64x2x2x32xf32>
    %s247 = stablehlo.iota dim = 3 : tensor<64x2x2x32xi32>
    %s248 = stablehlo.broadcast_in_dim %63, dims = [0] : (tensor<64xi32>) -> tensor<64x2x2x32xi32>
    %s250 = stablehlo.compare LT, %s247, %s248, SIGNED : (tensor<64x2x2x32xi32>, tensor<64x2x2x32xi32>) -> tensor<64x2x2x32xi1>
    %s251 = stablehlo.constant dense<8> : tensor<i32>
    %s252 = stablehlo.broadcast_in_dim %s251, dims = [] : (tensor<i32>) -> tensor<64x2x2x32xi32>
    %s253 = stablehlo.subtract %s248, %s252 : tensor<64x2x2x32xi32>
    %s254 = stablehlo.compare GE, %s247, %s253, SIGNED : (tensor<64x2x2x32xi32>, tensor<64x2x2x32xi32>) -> tensor<64x2x2x32xi1>
    %s249 = stablehlo.and %s250, %s254 : tensor<64x2x2x32xi1>
    %s255 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s256 = stablehlo.broadcast_in_dim %s255, dims = [] : (tensor<f32>) -> tensor<64x2x2x32xf32>
    %s257 = stablehlo.select %s249, %s246, %s256 : tensor<64x2x2x32xi1>, tensor<64x2x2x32xf32>
    %s258 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s259 = stablehlo.reduce(%s257 init: %s258) applies stablehlo.maximum across dimensions = [3] : (tensor<64x2x2x32xf32>, tensor<f32>) -> tensor<64x2x2xf32>
    %s260 = stablehlo.broadcast_in_dim %s259, dims = [0, 1, 2] : (tensor<64x2x2xf32>) -> tensor<64x2x2x32xf32>
    %s261 = stablehlo.subtract %s257, %s260 : tensor<64x2x2x32xf32>
    %s262 = stablehlo.exponential %s261 : tensor<64x2x2x32xf32>
    %s263 = stablehlo.constant dense<0.0> : tensor<f32>
    %s264 = stablehlo.reduce(%s262 init: %s263) applies stablehlo.add across dimensions = [3] : (tensor<64x2x2x32xf32>, tensor<f32>) -> tensor<64x2x2xf32>
    %s265 = stablehlo.broadcast_in_dim %s264, dims = [0, 1, 2] : (tensor<64x2x2xf32>) -> tensor<64x2x2x32xf32>
    %s266 = stablehlo.divide %s262, %s265 : tensor<64x2x2x32xf32>
    %s267 = stablehlo.dot_general %s266, %s241, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [HIGHEST, HIGHEST] : (tensor<64x2x2x32xf32>, tensor<64x32x2x8xf32>) -> tensor<64x2x2x8xf32>
    %97 = stablehlo.reshape %s267 : (tensor<64x2x2x8xf32>) -> tensor<64x4x8xf32>
    %98 = stablehlo.reshape %97 : (tensor<64x4x8xf32>) -> tensor<64x32xf32>
    %99 = stablehlo.dot_general %98, %18, contracting_dims = [1] x [0] : (tensor<64x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    %100 = stablehlo.add %65, %99 : tensor<64x32xf32>
    %101 = stablehlo.multiply %100, %100 : tensor<64x32xf32>
    %s268 = stablehlo.constant dense<0.0> : tensor<f32>
    %s269 = stablehlo.reduce(%101 init: %s268) applies stablehlo.add across dimensions = [1] : (tensor<64x32xf32>, tensor<f32>) -> tensor<64xf32>
    %s270 = stablehlo.constant dense<32.0> : tensor<64xf32>
    %s271 = stablehlo.divide %s269, %s270 : tensor<64xf32>
    %102 = stablehlo.broadcast_in_dim %s271, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %103 = stablehlo.add %102, %66 : tensor<64x1xf32>
    %104 = stablehlo.rsqrt %103 : tensor<64x1xf32>
    %s272 = stablehlo.broadcast_in_dim %104, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x32xf32>
    %105 = stablehlo.multiply %100, %s272 : tensor<64x32xf32>
    %106 = stablehlo.reshape %19 : (tensor<32xf32>) -> tensor<1x32xf32>
    %107 = stablehlo.broadcast_in_dim %106, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<64x32xf32>
    %108 = stablehlo.multiply %105, %107 : tensor<64x32xf32>
    %109 = stablehlo.dot_general %108, %20, contracting_dims = [1] x [0] : (tensor<64x32xf32>, tensor<32x48xf32>) -> tensor<64x48xf32>
    %110 = stablehlo.dot_general %108, %21, contracting_dims = [1] x [0] : (tensor<64x32xf32>, tensor<32x48xf32>) -> tensor<64x48xf32>
    %s273 = stablehlo.logistic %109 : tensor<64x48xf32>
    %111 = stablehlo.multiply %109, %s273 : tensor<64x48xf32>
    %112 = stablehlo.multiply %111, %110 : tensor<64x48xf32>
    %113 = stablehlo.dot_general %112, %22, contracting_dims = [1] x [0] : (tensor<64x48xf32>, tensor<48x32xf32>) -> tensor<64x32xf32>
    %114 = stablehlo.add %100, %113 : tensor<64x32xf32>
    %115 = stablehlo.multiply %114, %114 : tensor<64x32xf32>
    %s274 = stablehlo.constant dense<0.0> : tensor<f32>
    %s275 = stablehlo.reduce(%115 init: %s274) applies stablehlo.add across dimensions = [1] : (tensor<64x32xf32>, tensor<f32>) -> tensor<64xf32>
    %s276 = stablehlo.constant dense<32.0> : tensor<64xf32>
    %s277 = stablehlo.divide %s275, %s276 : tensor<64xf32>
    %116 = stablehlo.broadcast_in_dim %s277, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %117 = stablehlo.add %116, %66 : tensor<64x1xf32>
    %118 = stablehlo.rsqrt %117 : tensor<64x1xf32>
    %s278 = stablehlo.broadcast_in_dim %118, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x32xf32>
    %119 = stablehlo.multiply %114, %s278 : tensor<64x32xf32>
    %120 = stablehlo.reshape %23 : (tensor<32xf32>) -> tensor<1x32xf32>
    %121 = stablehlo.broadcast_in_dim %120, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<64x32xf32>
    %122 = stablehlo.multiply %119, %121 : tensor<64x32xf32>
    %123 = stablehlo.dot_general %122, %24, contracting_dims = [1] x [0] : (tensor<64x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    %124 = stablehlo.dot_general %122, %25, contracting_dims = [1] x [0] : (tensor<64x32xf32>, tensor<32x16xf32>) -> tensor<64x16xf32>
    %125 = stablehlo.dot_general %122, %26, contracting_dims = [1] x [0] : (tensor<64x32xf32>, tensor<32x16xf32>) -> tensor<64x16xf32>
    %126 = stablehlo.reshape %123 : (tensor<64x32xf32>) -> tensor<64x4x8xf32>
    %127 = stablehlo.reshape %124 : (tensor<64x16xf32>) -> tensor<64x2x8xf32>
    %128 = stablehlo.reshape %125 : (tensor<64x16xf32>) -> tensor<64x2x8xf32>
    %129 = stablehlo.multiply %126, %49 : tensor<64x4x8xf32>
    %130 = stablehlo.slice %126 [0:64, 0:4, 4:8] : (tensor<64x4x8xf32>) -> tensor<64x4x4xf32>
    %131 = stablehlo.slice %126 [0:64, 0:4, 0:4] : (tensor<64x4x8xf32>) -> tensor<64x4x4xf32>
    %132 = stablehlo.negate %130 : tensor<64x4x4xf32>
    %133 = stablehlo.concatenate %132, %131, dim = 2 : (tensor<64x4x4xf32>, tensor<64x4x4xf32>) -> tensor<64x4x8xf32>
    %134 = stablehlo.multiply %133, %51 : tensor<64x4x8xf32>
    %135 = stablehlo.add %129, %134 : tensor<64x4x8xf32>
    %136 = stablehlo.multiply %127, %53 : tensor<64x2x8xf32>
    %137 = stablehlo.slice %127 [0:64, 0:2, 4:8] : (tensor<64x2x8xf32>) -> tensor<64x2x4xf32>
    %138 = stablehlo.slice %127 [0:64, 0:2, 0:4] : (tensor<64x2x8xf32>) -> tensor<64x2x4xf32>
    %139 = stablehlo.negate %137 : tensor<64x2x4xf32>
    %140 = stablehlo.concatenate %139, %138, dim = 2 : (tensor<64x2x4xf32>, tensor<64x2x4xf32>) -> tensor<64x2x8xf32>
    %141 = stablehlo.multiply %140, %55 : tensor<64x2x8xf32>
    %142 = stablehlo.add %136, %141 : tensor<64x2x8xf32>
    %s279 = stablehlo.reshape %9 : (tensor<13x4x2x8xf32>) -> tensor<52x2x8xf32>
    %s280 = "stablehlo.scatter"(%s279, %6, %142) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s281: tensor<f32>, %s282: tensor<f32>):
       stablehlo.return %s282 : tensor<f32>
     }) : (tensor<52x2x8xf32>, tensor<64xi32>, tensor<64x2x8xf32>) -> tensor<52x2x8xf32>
    %143 = stablehlo.reshape %s280 : (tensor<52x2x8xf32>) -> tensor<13x4x2x8xf32>
    %s283 = stablehlo.reshape %10 : (tensor<13x4x2x8xf32>) -> tensor<52x2x8xf32>
    %s284 = "stablehlo.scatter"(%s283, %6, %128) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s285: tensor<f32>, %s286: tensor<f32>):
       stablehlo.return %s286 : tensor<f32>
     }) : (tensor<52x2x8xf32>, tensor<64xi32>, tensor<64x2x8xf32>) -> tensor<52x2x8xf32>
    %144 = stablehlo.reshape %s284 : (tensor<52x2x8xf32>) -> tensor<13x4x2x8xf32>
    %s287 = "stablehlo.gather"(%143, %61) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<13x4x2x8xf32>, tensor<64x8xi32>) -> tensor<64x8x4x2x8xf32>
    %s288 = stablehlo.reshape %s287 : (tensor<64x8x4x2x8xf32>) -> tensor<64x32x2x8xf32>
    %s289 = "stablehlo.gather"(%144, %61) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<13x4x2x8xf32>, tensor<64x8xi32>) -> tensor<64x8x4x2x8xf32>
    %s290 = stablehlo.reshape %s289 : (tensor<64x8x4x2x8xf32>) -> tensor<64x32x2x8xf32>
    %s291 = stablehlo.reshape %135 : (tensor<64x4x8xf32>) -> tensor<64x2x2x8xf32>
    %s292 = stablehlo.dot_general %s291, %s288, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [3], precision = [HIGHEST, HIGHEST] : (tensor<64x2x2x8xf32>, tensor<64x32x2x8xf32>) -> tensor<64x2x2x32xf32>
    %s293 = stablehlo.constant dense<0.35355338> : tensor<f32>
    %s294 = stablehlo.broadcast_in_dim %s293, dims = [] : (tensor<f32>) -> tensor<64x2x2x32xf32>
    %s295 = stablehlo.multiply %s292, %s294 : tensor<64x2x2x32xf32>
    %s296 = stablehlo.iota dim = 3 : tensor<64x2x2x32xi32>
    %s297 = stablehlo.broadcast_in_dim %63, dims = [0] : (tensor<64xi32>) -> tensor<64x2x2x32xi32>
    %s299 = stablehlo.compare LT, %s296, %s297, SIGNED : (tensor<64x2x2x32xi32>, tensor<64x2x2x32xi32>) -> tensor<64x2x2x32xi1>
    %s300 = stablehlo.constant dense<8> : tensor<i32>
    %s301 = stablehlo.broadcast_in_dim %s300, dims = [] : (tensor<i32>) -> tensor<64x2x2x32xi32>
    %s302 = stablehlo.subtract %s297, %s301 : tensor<64x2x2x32xi32>
    %s303 = stablehlo.compare GE, %s296, %s302, SIGNED : (tensor<64x2x2x32xi32>, tensor<64x2x2x32xi32>) -> tensor<64x2x2x32xi1>
    %s298 = stablehlo.and %s299, %s303 : tensor<64x2x2x32xi1>
    %s304 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s305 = stablehlo.broadcast_in_dim %s304, dims = [] : (tensor<f32>) -> tensor<64x2x2x32xf32>
    %s306 = stablehlo.select %s298, %s295, %s305 : tensor<64x2x2x32xi1>, tensor<64x2x2x32xf32>
    %s307 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s308 = stablehlo.reduce(%s306 init: %s307) applies stablehlo.maximum across dimensions = [3] : (tensor<64x2x2x32xf32>, tensor<f32>) -> tensor<64x2x2xf32>
    %s309 = stablehlo.broadcast_in_dim %s308, dims = [0, 1, 2] : (tensor<64x2x2xf32>) -> tensor<64x2x2x32xf32>
    %s310 = stablehlo.subtract %s306, %s309 : tensor<64x2x2x32xf32>
    %s311 = stablehlo.exponential %s310 : tensor<64x2x2x32xf32>
    %s312 = stablehlo.constant dense<0.0> : tensor<f32>
    %s313 = stablehlo.reduce(%s311 init: %s312) applies stablehlo.add across dimensions = [3] : (tensor<64x2x2x32xf32>, tensor<f32>) -> tensor<64x2x2xf32>
    %s314 = stablehlo.broadcast_in_dim %s313, dims = [0, 1, 2] : (tensor<64x2x2xf32>) -> tensor<64x2x2x32xf32>
    %s315 = stablehlo.divide %s311, %s314 : tensor<64x2x2x32xf32>
    %s316 = stablehlo.dot_general %s315, %s290, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [HIGHEST, HIGHEST] : (tensor<64x2x2x32xf32>, tensor<64x32x2x8xf32>) -> tensor<64x2x2x8xf32>
    %145 = stablehlo.reshape %s316 : (tensor<64x2x2x8xf32>) -> tensor<64x4x8xf32>
    %146 = stablehlo.reshape %145 : (tensor<64x4x8xf32>) -> tensor<64x32xf32>
    %147 = stablehlo.dot_general %146, %27, contracting_dims = [1] x [0] : (tensor<64x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    %148 = stablehlo.add %114, %147 : tensor<64x32xf32>
    %149 = stablehlo.multiply %148, %148 : tensor<64x32xf32>
    %s317 = stablehlo.constant dense<0.0> : tensor<f32>
    %s318 = stablehlo.reduce(%149 init: %s317) applies stablehlo.add across dimensions = [1] : (tensor<64x32xf32>, tensor<f32>) -> tensor<64xf32>
    %s319 = stablehlo.constant dense<32.0> : tensor<64xf32>
    %s320 = stablehlo.divide %s318, %s319 : tensor<64xf32>
    %150 = stablehlo.broadcast_in_dim %s320, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %151 = stablehlo.add %150, %66 : tensor<64x1xf32>
    %152 = stablehlo.rsqrt %151 : tensor<64x1xf32>
    %s321 = stablehlo.broadcast_in_dim %152, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x32xf32>
    %153 = stablehlo.multiply %148, %s321 : tensor<64x32xf32>
    %154 = stablehlo.reshape %28 : (tensor<32xf32>) -> tensor<1x32xf32>
    %155 = stablehlo.broadcast_in_dim %154, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<64x32xf32>
    %156 = stablehlo.multiply %153, %155 : tensor<64x32xf32>
    %157 = stablehlo.dot_general %156, %29, contracting_dims = [1] x [0] : (tensor<64x32xf32>, tensor<32x48xf32>) -> tensor<64x48xf32>
    %158 = stablehlo.dot_general %156, %30, contracting_dims = [1] x [0] : (tensor<64x32xf32>, tensor<32x48xf32>) -> tensor<64x48xf32>
    %s322 = stablehlo.logistic %157 : tensor<64x48xf32>
    %159 = stablehlo.multiply %157, %s322 : tensor<64x48xf32>
    %160 = stablehlo.multiply %159, %158 : tensor<64x48xf32>
    %161 = stablehlo.dot_general %160, %31, contracting_dims = [1] x [0] : (tensor<64x48xf32>, tensor<48x32xf32>) -> tensor<64x32xf32>
    %162 = stablehlo.add %148, %161 : tensor<64x32xf32>
    %163 = stablehlo.multiply %162, %162 : tensor<64x32xf32>
    %s323 = stablehlo.constant dense<0.0> : tensor<f32>
    %s324 = stablehlo.reduce(%163 init: %s323) applies stablehlo.add across dimensions = [1] : (tensor<64x32xf32>, tensor<f32>) -> tensor<64xf32>
    %s325 = stablehlo.constant dense<32.0> : tensor<64xf32>
    %s326 = stablehlo.divide %s324, %s325 : tensor<64xf32>
    %164 = stablehlo.broadcast_in_dim %s326, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %165 = stablehlo.add %164, %66 : tensor<64x1xf32>
    %166 = stablehlo.rsqrt %165 : tensor<64x1xf32>
    %s327 = stablehlo.broadcast_in_dim %166, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x32xf32>
    %167 = stablehlo.multiply %162, %s327 : tensor<64x32xf32>
    %168 = stablehlo.reshape %32 : (tensor<32xf32>) -> tensor<1x32xf32>
    %169 = stablehlo.broadcast_in_dim %168, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<64x32xf32>
    %170 = stablehlo.multiply %167, %169 : tensor<64x32xf32>
    %171 = stablehlo.dot_general %170, %33, contracting_dims = [1] x [0] : (tensor<64x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    %172 = stablehlo.dot_general %170, %34, contracting_dims = [1] x [0] : (tensor<64x32xf32>, tensor<32x16xf32>) -> tensor<64x16xf32>
    %173 = stablehlo.dot_general %170, %35, contracting_dims = [1] x [0] : (tensor<64x32xf32>, tensor<32x16xf32>) -> tensor<64x16xf32>
    %174 = stablehlo.reshape %171 : (tensor<64x32xf32>) -> tensor<64x4x8xf32>
    %175 = stablehlo.reshape %172 : (tensor<64x16xf32>) -> tensor<64x2x8xf32>
    %176 = stablehlo.reshape %173 : (tensor<64x16xf32>) -> tensor<64x2x8xf32>
    %177 = stablehlo.multiply %174, %49 : tensor<64x4x8xf32>
    %178 = stablehlo.slice %174 [0:64, 0:4, 4:8] : (tensor<64x4x8xf32>) -> tensor<64x4x4xf32>
    %179 = stablehlo.slice %174 [0:64, 0:4, 0:4] : (tensor<64x4x8xf32>) -> tensor<64x4x4xf32>
    %180 = stablehlo.negate %178 : tensor<64x4x4xf32>
    %181 = stablehlo.concatenate %180, %179, dim = 2 : (tensor<64x4x4xf32>, tensor<64x4x4xf32>) -> tensor<64x4x8xf32>
    %182 = stablehlo.multiply %181, %51 : tensor<64x4x8xf32>
    %183 = stablehlo.add %177, %182 : tensor<64x4x8xf32>
    %184 = stablehlo.multiply %175, %53 : tensor<64x2x8xf32>
    %185 = stablehlo.slice %175 [0:64, 0:2, 4:8] : (tensor<64x2x8xf32>) -> tensor<64x2x4xf32>
    %186 = stablehlo.slice %175 [0:64, 0:2, 0:4] : (tensor<64x2x8xf32>) -> tensor<64x2x4xf32>
    %187 = stablehlo.negate %185 : tensor<64x2x4xf32>
    %188 = stablehlo.concatenate %187, %186, dim = 2 : (tensor<64x2x4xf32>, tensor<64x2x4xf32>) -> tensor<64x2x8xf32>
    %189 = stablehlo.multiply %188, %55 : tensor<64x2x8xf32>
    %190 = stablehlo.add %184, %189 : tensor<64x2x8xf32>
    %s328 = stablehlo.reshape %11 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s329 = "stablehlo.scatter"(%s328, %4, %190) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s330: tensor<f32>, %s331: tensor<f32>):
       stablehlo.return %s331 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<64xi32>, tensor<64x2x8xf32>) -> tensor<260x2x8xf32>
    %191 = stablehlo.reshape %s329 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s332 = stablehlo.reshape %12 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s333 = "stablehlo.scatter"(%s332, %4, %176) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s334: tensor<f32>, %s335: tensor<f32>):
       stablehlo.return %s335 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<64xi32>, tensor<64x2x8xf32>) -> tensor<260x2x8xf32>
    %192 = stablehlo.reshape %s333 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s336 = "stablehlo.gather"(%191, %58) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<64x8xi32>) -> tensor<64x8x4x2x8xf32>
    %s337 = stablehlo.reshape %s336 : (tensor<64x8x4x2x8xf32>) -> tensor<64x32x2x8xf32>
    %s338 = "stablehlo.gather"(%192, %58) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<64x8xi32>) -> tensor<64x8x4x2x8xf32>
    %s339 = stablehlo.reshape %s338 : (tensor<64x8x4x2x8xf32>) -> tensor<64x32x2x8xf32>
    %s340 = stablehlo.reshape %183 : (tensor<64x4x8xf32>) -> tensor<64x2x2x8xf32>
    %s341 = stablehlo.dot_general %s340, %s337, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [3], precision = [HIGHEST, HIGHEST] : (tensor<64x2x2x8xf32>, tensor<64x32x2x8xf32>) -> tensor<64x2x2x32xf32>
    %s342 = stablehlo.constant dense<0.35355338> : tensor<f32>
    %s343 = stablehlo.broadcast_in_dim %s342, dims = [] : (tensor<f32>) -> tensor<64x2x2x32xf32>
    %s344 = stablehlo.multiply %s341, %s343 : tensor<64x2x2x32xf32>
    %s345 = stablehlo.iota dim = 3 : tensor<64x2x2x32xi32>
    %s346 = stablehlo.broadcast_in_dim %63, dims = [0] : (tensor<64xi32>) -> tensor<64x2x2x32xi32>
    %s347 = stablehlo.compare LT, %s345, %s346, SIGNED : (tensor<64x2x2x32xi32>, tensor<64x2x2x32xi32>) -> tensor<64x2x2x32xi1>
    %s348 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s349 = stablehlo.broadcast_in_dim %s348, dims = [] : (tensor<f32>) -> tensor<64x2x2x32xf32>
    %s350 = stablehlo.select %s347, %s344, %s349 : tensor<64x2x2x32xi1>, tensor<64x2x2x32xf32>
    %s351 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s352 = stablehlo.reduce(%s350 init: %s351) applies stablehlo.maximum across dimensions = [3] : (tensor<64x2x2x32xf32>, tensor<f32>) -> tensor<64x2x2xf32>
    %s353 = stablehlo.broadcast_in_dim %s352, dims = [0, 1, 2] : (tensor<64x2x2xf32>) -> tensor<64x2x2x32xf32>
    %s354 = stablehlo.subtract %s350, %s353 : tensor<64x2x2x32xf32>
    %s355 = stablehlo.exponential %s354 : tensor<64x2x2x32xf32>
    %s356 = stablehlo.constant dense<0.0> : tensor<f32>
    %s357 = stablehlo.reduce(%s355 init: %s356) applies stablehlo.add across dimensions = [3] : (tensor<64x2x2x32xf32>, tensor<f32>) -> tensor<64x2x2xf32>
    %s358 = stablehlo.broadcast_in_dim %s357, dims = [0, 1, 2] : (tensor<64x2x2xf32>) -> tensor<64x2x2x32xf32>
    %s359 = stablehlo.divide %s355, %s358 : tensor<64x2x2x32xf32>
    %s360 = stablehlo.dot_general %s359, %s339, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [HIGHEST, HIGHEST] : (tensor<64x2x2x32xf32>, tensor<64x32x2x8xf32>) -> tensor<64x2x2x8xf32>
    %193 = stablehlo.reshape %s360 : (tensor<64x2x2x8xf32>) -> tensor<64x4x8xf32>
    %194 = stablehlo.reshape %193 : (tensor<64x4x8xf32>) -> tensor<64x32xf32>
    %195 = stablehlo.dot_general %194, %36, contracting_dims = [1] x [0] : (tensor<64x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
    %196 = stablehlo.add %162, %195 : tensor<64x32xf32>
    %197 = stablehlo.multiply %196, %196 : tensor<64x32xf32>
    %s361 = stablehlo.constant dense<0.0> : tensor<f32>
    %s362 = stablehlo.reduce(%197 init: %s361) applies stablehlo.add across dimensions = [1] : (tensor<64x32xf32>, tensor<f32>) -> tensor<64xf32>
    %s363 = stablehlo.constant dense<32.0> : tensor<64xf32>
    %s364 = stablehlo.divide %s362, %s363 : tensor<64xf32>
    %198 = stablehlo.broadcast_in_dim %s364, dims = [0] : (tensor<64xf32>) -> tensor<64x1xf32>
    %199 = stablehlo.add %198, %66 : tensor<64x1xf32>
    %200 = stablehlo.rsqrt %199 : tensor<64x1xf32>
    %s365 = stablehlo.broadcast_in_dim %200, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x32xf32>
    %201 = stablehlo.multiply %196, %s365 : tensor<64x32xf32>
    %202 = stablehlo.reshape %37 : (tensor<32xf32>) -> tensor<1x32xf32>
    %203 = stablehlo.broadcast_in_dim %202, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<64x32xf32>
    %204 = stablehlo.multiply %201, %203 : tensor<64x32xf32>
    %205 = stablehlo.dot_general %204, %38, contracting_dims = [1] x [0] : (tensor<64x32xf32>, tensor<32x48xf32>) -> tensor<64x48xf32>
    %206 = stablehlo.dot_general %204, %39, contracting_dims = [1] x [0] : (tensor<64x32xf32>, tensor<32x48xf32>) -> tensor<64x48xf32>
    %s366 = stablehlo.logistic %205 : tensor<64x48xf32>
    %207 = stablehlo.multiply %205, %s366 : tensor<64x48xf32>
    %208 = stablehlo.multiply %207, %206 : tensor<64x48xf32>
    %209 = stablehlo.dot_general %208, %40, contracting_dims = [1] x [0] : (tensor<64x48xf32>, tensor<48x32xf32>) -> tensor<64x32xf32>
    %210 = stablehlo.add %196, %209 : tensor<64x32xf32>
    %211 = stablehlo.reshape %210 : (tensor<64x32xf32>) -> tensor<2x32x32xf32>
    %212 = stablehlo.slice %211 [0:2, 31:32, 0:32] : (tensor<2x32x32xf32>) -> tensor<2x1x32xf32>
    %213 = stablehlo.reshape %212 : (tensor<2x1x32xf32>) -> tensor<2x32xf32>
    %214 = stablehlo.constant dense<[[1.0E-6], [1.0E-6]]> : tensor<2x1xf32>
    %215 = stablehlo.multiply %213, %213 : tensor<2x32xf32>
    %s367 = stablehlo.constant dense<0.0> : tensor<f32>
    %s368 = stablehlo.reduce(%215 init: %s367) applies stablehlo.add across dimensions = [1] : (tensor<2x32xf32>, tensor<f32>) -> tensor<2xf32>
    %s369 = stablehlo.constant dense<32.0> : tensor<2xf32>
    %s370 = stablehlo.divide %s368, %s369 : tensor<2xf32>
    %216 = stablehlo.broadcast_in_dim %s370, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %217 = stablehlo.add %216, %214 : tensor<2x1xf32>
    %218 = stablehlo.rsqrt %217 : tensor<2x1xf32>
    %s371 = stablehlo.broadcast_in_dim %218, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x32xf32>
    %219 = stablehlo.multiply %213, %s371 : tensor<2x32xf32>
    %220 = stablehlo.reshape %41 : (tensor<32xf32>) -> tensor<1x32xf32>
    %221 = stablehlo.broadcast_in_dim %220, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<2x32xf32>
    %222 = stablehlo.multiply %219, %221 : tensor<2x32xf32>
    %223 = stablehlo.dot_general %222, %42, contracting_dims = [1] x [0] : (tensor<2x32xf32>, tensor<32x64xf32>) -> tensor<2x64xf32>
    %224 = stablehlo.reshape %223 : (tensor<2x64xf32>) -> tensor<2x1x64xf32>
    return %224, %95, %96, %143, %144, %191, %192 : tensor<2x1x64xf32>, tensor<13x4x2x8xf32>, tensor<13x4x2x8xf32>, tensor<13x4x2x8xf32>, tensor<13x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>
  }
}
