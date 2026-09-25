module {
  func.func @main(%0: tensor<1x1xi32>, %1: tensor<1x1xi32>, %2: tensor<1x16xi32>, %3: tensor<1xi32>, %4: tensor<1xi32>, %5: tensor<1x16xi32>, %6: tensor<1xi32>, %7: tensor<17x4x2x8xf32> {tf.aliasing_output = 1 : i32}, %8: tensor<17x4x2x8xf32> {tf.aliasing_output = 2 : i32}, %9: tensor<17x4x2x8xf32> {tf.aliasing_output = 3 : i32}, %10: tensor<17x4x2x8xf32> {tf.aliasing_output = 4 : i32}, %11: tensor<65x4x2x8xf32> {tf.aliasing_output = 5 : i32}, %12: tensor<65x4x2x8xf32> {tf.aliasing_output = 6 : i32}, %13: tensor<64x32xf32>, %14: tensor<32xf32>, %15: tensor<32x32xf32>, %16: tensor<32x16xf32>, %17: tensor<32x16xf32>, %18: tensor<32x32xf32>, %19: tensor<32xf32>, %20: tensor<32x48xf32>, %21: tensor<32x48xf32>, %22: tensor<48x32xf32>, %23: tensor<32xf32>, %24: tensor<32x32xf32>, %25: tensor<32x16xf32>, %26: tensor<32x16xf32>, %27: tensor<32x32xf32>, %28: tensor<32xf32>, %29: tensor<32x48xf32>, %30: tensor<32x48xf32>, %31: tensor<48x32xf32>, %32: tensor<32xf32>, %33: tensor<32x32xf32>, %34: tensor<32x16xf32>, %35: tensor<32x16xf32>, %36: tensor<32x32xf32>, %37: tensor<32xf32>, %38: tensor<32x48xf32>, %39: tensor<32x48xf32>, %40: tensor<48x32xf32>, %41: tensor<32xf32>, %42: tensor<32x64xf32>) -> (tensor<1x1x64xf32>, tensor<17x4x2x8xf32>, tensor<17x4x2x8xf32>, tensor<17x4x2x8xf32>, tensor<17x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>) {
    %43 = stablehlo.reshape %1 : (tensor<1x1xi32>) -> tensor<1xi32>
    %44 = stablehlo.constant dense<[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.5403023, 0.9950042, 0.99995, 0.9999995, 0.5403023, 0.9950042, 0.99995, 0.9999995], [-0.41614684, 0.9800666, 0.9998, 0.999998, -0.41614684, 0.9800666, 0.9998, 0.999998], [-0.9899925, 0.9553365, 0.99955004, 0.9999955, -0.9899925, 0.9553365, 0.99955004, 0.9999955], [-0.6536436, 0.921061, 0.9992001, 0.999992, -0.6536436, 0.921061, 0.9992001, 0.999992], [0.2836622, 0.87758255, 0.99875027, 0.9999875, 0.2836622, 0.87758255, 0.99875027, 0.9999875], [0.96017027, 0.8253356, 0.99820054, 0.999982, 0.96017027, 0.8253356, 0.99820054, 0.999982], [0.75390226, 0.7648422, 0.997551, 0.9999755, 0.75390226, 0.7648422, 0.997551, 0.9999755], [-0.14550003, 0.6967067, 0.99680173, 0.999968, -0.14550003, 0.6967067, 0.99680173, 0.999968], [-0.91113025, 0.62161, 0.9959527, 0.9999595, -0.91113025, 0.62161, 0.9959527, 0.9999595], [-0.8390715, 0.5403023, 0.9950042, 0.99995, -0.8390715, 0.5403023, 0.9950042, 0.99995], [0.004425698, 0.45359612, 0.9939561, 0.9999395, 0.004425698, 0.45359612, 0.9939561, 0.9999395], [0.84385395, 0.36235777, 0.99280864, 0.999928, 0.84385395, 0.36235777, 0.99280864, 0.999928], [0.9074468, 0.26749882, 0.9915619, 0.9999155, 0.9074468, 0.26749882, 0.9915619, 0.9999155], [0.13673721, 0.16996714, 0.990216, 0.999902, 0.13673721, 0.16996714, 0.990216, 0.999902], [-0.7596879, 0.0707372, 0.9887711, 0.9998875, -0.7596879, 0.0707372, 0.9887711, 0.9998875], [-0.9576595, -0.029199522, 0.98722726, 0.999872, -0.9576595, -0.029199522, 0.98722726, 0.999872], [-0.27516335, -0.1288445, 0.9855848, 0.9998555, -0.27516335, -0.1288445, 0.9855848, 0.9998555], [0.6603167, -0.22720209, 0.9838437, 0.999838, 0.6603167, -0.22720209, 0.9838437, 0.999838], [0.9887046, -0.32328957, 0.9820042, 0.9998195, 0.9887046, -0.32328957, 0.9820042, 0.9998195], [0.40808207, -0.41614684, 0.9800666, 0.9998, 0.40808207, -0.41614684, 0.9800666, 0.9998], [-0.54772925, -0.5048461, 0.9780309, 0.9997795, -0.54772925, -0.5048461, 0.9780309, 0.9997795], [-0.99996084, -0.5885011, 0.97589743, 0.999758, -0.99996084, -0.5885011, 0.97589743, 0.999758], [-0.53283304, -0.66627604, 0.97366637, 0.99973553, -0.53283304, -0.66627604, 0.97366637, 0.99973553], [0.42417902, -0.73739374, 0.971338, 0.999712, 0.42417902, -0.73739374, 0.971338, 0.999712], [0.99120283, -0.8011436, 0.9689124, 0.9996875, 0.99120283, -0.8011436, 0.9689124, 0.9996875], [0.6469193, -0.8568888, 0.96638995, 0.99966204, 0.6469193, -0.8568888, 0.96638995, 0.99966204], [-0.29213881, -0.90407217, 0.9637709, 0.9996355, -0.29213881, -0.90407217, 0.9637709, 0.9996355], [-0.9626059, -0.94222236, 0.96105546, 0.99960804, -0.9626059, -0.94222236, 0.96105546, 0.99960804], [-0.74805754, -0.9709582, 0.95824385, 0.99957955, -0.74805754, -0.9709582, 0.95824385, 0.99957955], [0.15425146, -0.9899925, 0.9553365, 0.99955004, 0.15425146, -0.9899925, 0.9553365, 0.99955004], [0.91474235, -0.99913514, 0.95233357, 0.9995195, 0.91474235, -0.99913514, 0.95233357, 0.9995195], [0.8342234, -0.9982948, 0.94923544, 0.99948806, 0.8342234, -0.9982948, 0.94923544, 0.99948806], [-0.013276747, -0.98747975, 0.94604236, 0.9994556, -0.013276747, -0.98747975, 0.94604236, 0.9994556], [-0.8485703, -0.9667982, 0.9427547, 0.9994221, -0.8485703, -0.9667982, 0.9427547, 0.9994221], [-0.9036922, -0.9364567, 0.9393727, 0.99938756, -0.9036922, -0.9364567, 0.9393727, 0.99938756], [-0.12796369, -0.89675844, 0.9358968, 0.9993521, -0.12796369, -0.89675844, 0.9358968, 0.9993521], [0.76541406, -0.8481, 0.93232733, 0.99931556, 0.76541406, -0.8481, 0.93232733, 0.99931556], [0.95507365, -0.7909677, 0.9286646, 0.99927807, 0.95507365, -0.7909677, 0.9286646, 0.99927807], [0.26664293, -0.7259323, 0.92490906, 0.9992396, 0.26664293, -0.7259323, 0.92490906, 0.9992396], [-0.66693807, -0.6536436, 0.921061, 0.9992001, -0.66693807, -0.6536436, 0.921061, 0.9992001], [-0.98733926, -0.574824, 0.9171208, 0.99915963, -0.98733926, -0.574824, 0.9171208, 0.99915963], [-0.3999853, -0.4902608, 0.9130889, 0.99911815, -0.3999853, -0.4902608, 0.9130889, 0.99911815], [0.5551133, -0.40079919, 0.90896577, 0.99907565, 0.5551133, -0.40079919, 0.90896577, 0.99907565], [0.9998433, -0.30733287, 0.90475166, 0.99903214, 0.9998433, -0.30733287, 0.90475166, 0.99903214], [0.52532196, -0.2107958, 0.90044713, 0.9989877, 0.52532196, -0.2107958, 0.90044713, 0.9989877], [-0.43217793, -0.112152524, 0.8960525, 0.9989422, -0.43217793, -0.112152524, 0.8960525, 0.9989422], [-0.9923355, -0.012388663, 0.8915683, 0.9988957, -0.9923355, -0.012388663, 0.8915683, 0.9988957], [-0.64014435, 0.087498985, 0.8869949, 0.9988482, -0.64014435, 0.087498985, 0.8869949, 0.9988482], [0.30059254, 0.18651237, 0.88233286, 0.99879974, 0.30059254, 0.18651237, 0.88233286, 0.99879974], [0.964966, 0.2836622, 0.87758255, 0.99875027, 0.964966, 0.2836622, 0.87758255, 0.99875027], [0.7421542, 0.37797773, 0.8727445, 0.9986998, 0.7421542, 0.37797773, 0.8727445, 0.9986998], [-0.16299078, 0.46851668, 0.8678192, 0.9986483, -0.16299078, 0.46851668, 0.8678192, 0.9986483], [-0.9182828, 0.55437434, 0.8628071, 0.99859583, -0.9182828, 0.55437434, 0.8628071, 0.99859583], [-0.8293098, 0.63469285, 0.8577087, 0.99854237, -0.8293098, 0.63469285, 0.8577087, 0.99854237], [0.022126757, 0.7086698, 0.8525245, 0.9984879, 0.022126757, 0.7086698, 0.8525245, 0.9984879], [0.8532201, 0.77556586, 0.8472551, 0.9984324, 0.8532201, 0.77556586, 0.8472551, 0.9984324], [0.8998668, 0.8347128, 0.841901, 0.99837595, 0.8998668, 0.8347128, 0.841901, 0.99837595], [0.119180135, 0.8855195, 0.8364627, 0.9983185, 0.119180135, 0.8855195, 0.8364627, 0.9983185], [-0.7710802, 0.92747843, 0.83094066, 0.99826, -0.7710802, 0.92747843, 0.83094066, 0.99826], [-0.95241296, 0.96017027, 0.8253356, 0.99820054, -0.95241296, 0.96017027, 0.8253356, 0.99820054], [-0.25810164, 0.98326844, 0.819648, 0.9981401, -0.25810164, 0.98326844, 0.819648, 0.9981401], [0.67350715, 0.9965421, 0.8138785, 0.99807864, 0.67350715, 0.9965421, 0.8138785, 0.99807864], [0.9858966, 0.9998586, 0.8080275, 0.9980162, 0.9858966, 0.9998586, 0.8080275, 0.9980162]]> : tensor<64x8xf32>
    %45 = stablehlo.constant dense<[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.84147096, 0.099833414, 0.009999833, 9.999998E-4, 0.84147096, 0.099833414, 0.009999833, 9.999998E-4], [0.9092974, 0.19866933, 0.019998666, 0.0019999987, 0.9092974, 0.19866933, 0.019998666, 0.0019999987], [0.14112, 0.29552022, 0.029995501, 0.0029999956, 0.14112, 0.29552022, 0.029995501, 0.0029999956], [-0.7568025, 0.38941833, 0.039989334, 0.0039999895, -0.7568025, 0.38941833, 0.039989334, 0.0039999895], [-0.9589243, 0.47942555, 0.04997917, 0.0049999794, -0.9589243, 0.47942555, 0.04997917, 0.0049999794], [-0.2794155, 0.5646425, 0.059964005, 0.005999964, -0.2794155, 0.5646425, 0.059964005, 0.005999964], [0.6569866, 0.64421767, 0.06994285, 0.006999943, 0.6569866, 0.64421767, 0.06994285, 0.006999943], [0.98935825, 0.7173561, 0.0799147, 0.007999915, 0.98935825, 0.7173561, 0.0799147, 0.007999915], [0.4121185, 0.7833269, 0.08987855, 0.008999879, 0.4121185, 0.7833269, 0.08987855, 0.008999879], [-0.5440211, 0.84147096, 0.099833414, 0.009999833, -0.5440211, 0.84147096, 0.099833414, 0.009999833], [-0.9999902, 0.89120734, 0.1097783, 0.010999778, -0.9999902, 0.89120734, 0.1097783, 0.010999778], [-0.53657293, 0.9320391, 0.119712204, 0.011999712, -0.53657293, 0.9320391, 0.119712204, 0.011999712], [0.42016703, 0.9635582, 0.12963414, 0.012999634, 0.42016703, 0.9635582, 0.12963414, 0.012999634], [0.9906074, 0.98544973, 0.13954312, 0.013999542, 0.9906074, 0.98544973, 0.13954312, 0.013999542], [0.65028787, 0.997495, 0.14943813, 0.014999437, 0.65028787, 0.997495, 0.14943813, 0.014999437], [-0.2879033, 0.9995736, 0.15931821, 0.015999317, -0.2879033, 0.9995736, 0.15931821, 0.015999317], [-0.96139747, 0.9916648, 0.16918235, 0.016999181, -0.96139747, 0.9916648, 0.16918235, 0.016999181], [-0.75098723, 0.9738476, 0.17902957, 0.017999029, -0.75098723, 0.9738476, 0.17902957, 0.017999029], [0.1498772, 0.9463001, 0.1888589, 0.018998858, 0.1498772, 0.9463001, 0.1888589, 0.018998858], [0.9129453, 0.9092974, 0.19866933, 0.019998666, 0.9129453, 0.9092974, 0.19866933, 0.019998666], [0.8366556, 0.86320937, 0.2084599, 0.020998457, 0.8366556, 0.86320937, 0.2084599, 0.020998457], [-0.008851309, 0.8084964, 0.21822962, 0.021998225, -0.008851309, 0.8084964, 0.21822962, 0.021998225], [-0.84622043, 0.7457052, 0.22797753, 0.022997972, -0.84622043, 0.7457052, 0.22797753, 0.022997972], [-0.9055784, 0.6754632, 0.23770262, 0.023997696, -0.9055784, 0.6754632, 0.23770262, 0.023997696], [-0.13235176, 0.5984721, 0.24740396, 0.024997396, -0.13235176, 0.5984721, 0.24740396, 0.024997396], [0.76255846, 0.5155014, 0.25708055, 0.02599707, 0.76255846, 0.5155014, 0.25708055, 0.02599707], [0.95637596, 0.42737988, 0.26673144, 0.026996719, 0.95637596, 0.42737988, 0.26673144, 0.026996719], [0.2709058, 0.33498815, 0.27635565, 0.02799634, 0.2709058, 0.33498815, 0.27635565, 0.02799634], [-0.6636339, 0.23924933, 0.2859522, 0.028995935, -0.6636339, 0.23924933, 0.2859522, 0.028995935], [-0.9880316, 0.14112, 0.29552022, 0.029995501, -0.9880316, 0.14112, 0.29552022, 0.029995501], [-0.40403765, 0.041580662, 0.30505863, 0.030995036, -0.40403765, 0.041580662, 0.30505863, 0.030995036], [0.5514267, -0.058374144, 0.31456655, 0.03199454, 0.5514267, -0.058374144, 0.31456655, 0.03199454], [0.99991184, -0.15774569, 0.32404304, 0.03299401, 0.99991184, -0.15774569, 0.32404304, 0.03299401], [0.5290827, -0.25554112, 0.3334871, 0.03399345, 0.5290827, -0.25554112, 0.3334871, 0.03399345], [-0.42818266, -0.35078323, 0.3428978, 0.034992855, -0.42818266, -0.35078323, 0.3428978, 0.034992855], [-0.99177885, -0.44252044, 0.35227424, 0.035992224, -0.99177885, -0.44252044, 0.35227424, 0.035992224], [-0.6435381, -0.5298361, 0.36161542, 0.03699156, -0.6435381, -0.5298361, 0.36161542, 0.03699156], [0.29636857, -0.6118579, 0.37092048, 0.037990857, 0.29636857, -0.6118579, 0.37092048, 0.037990857], [0.96379536, -0.68776613, 0.3801884, 0.038990114, 0.96379536, -0.68776613, 0.3801884, 0.038990114], [0.74511313, -0.7568025, 0.38941833, 0.039989334, 0.74511313, -0.7568025, 0.38941833, 0.039989334], [-0.15862267, -0.8182771, 0.39860934, 0.040988512, -0.15862267, -0.8182771, 0.39860934, 0.040988512], [-0.91652155, -0.8715758, 0.40776044, 0.041987654, -0.91652155, -0.8715758, 0.40776044, 0.041987654], [-0.8317748, -0.91616595, 0.4168708, 0.04298675, -0.8317748, -0.91616595, 0.4168708, 0.04298675], [0.017701926, -0.9516021, 0.42593947, 0.043985803, 0.017701926, -0.9516021, 0.42593947, 0.043985803], [0.8509035, -0.9775301, 0.43496552, 0.044984814, 0.8509035, -0.9775301, 0.43496552, 0.044984814], [0.90178835, -0.993691, 0.44394812, 0.04598378, 0.90178835, -0.993691, 0.44394812, 0.04598378], [0.123573124, -0.9999232, 0.45288628, 0.0469827, 0.123573124, -0.9999232, 0.45288628, 0.0469827], [-0.76825464, -0.9961646, 0.46177918, 0.04798157, -0.76825464, -0.9961646, 0.46177918, 0.04798157], [-0.95375264, -0.98245263, 0.47062588, 0.048980393, -0.95375264, -0.98245263, 0.47062588, 0.048980393], [-0.26237485, -0.9589243, 0.47942555, 0.04997917, -0.26237485, -0.9589243, 0.47942555, 0.04997917], [0.6702292, -0.9258147, 0.48817724, 0.050977893, 0.6702292, -0.9258147, 0.48817724, 0.050977893], [0.9866276, -0.8834547, 0.49688014, 0.05197657, 0.9866276, -0.8834547, 0.49688014, 0.05197657], [0.39592516, -0.83226746, 0.50553334, 0.05297519, 0.39592516, -0.83226746, 0.50553334, 0.05297519], [-0.5587891, -0.7727645, 0.514136, 0.05397376, -0.5587891, -0.7727645, 0.514136, 0.05397376], [-0.99975514, -0.7055403, 0.52268726, 0.054972276, -0.99975514, -0.7055403, 0.52268726, 0.054972276], [-0.521551, -0.63126665, 0.5311862, 0.055970736, -0.521551, -0.63126665, 0.5311862, 0.055970736], [0.43616477, -0.5506855, 0.539632, 0.05696914, 0.43616477, -0.5506855, 0.539632, 0.05696914], [0.99287266, -0.46460217, 0.54802394, 0.057967488, 0.99287266, -0.46460217, 0.54802394, 0.057967488], [0.636738, -0.37387666, 0.556361, 0.058965776, 0.636738, -0.37387666, 0.556361, 0.058965776], [-0.3048106, -0.2794155, 0.5646425, 0.059964005, -0.3048106, -0.2794155, 0.5646425, 0.059964005], [-0.9661178, -0.18216251, 0.57286745, 0.060962178, -0.9661178, -0.18216251, 0.57286745, 0.060962178], [-0.7391807, -0.083089404, 0.58103514, 0.061960287, -0.7391807, -0.083089404, 0.58103514, 0.061960287], [0.1673557, 0.0168139, 0.58914477, 0.06295834, 0.1673557, 0.0168139, 0.58914477, 0.06295834]]> : tensor<64x8xf32>
    %46 = "stablehlo.gather"(%44, %43) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 8>}> : (tensor<64x8xf32>, tensor<1xi32>) -> tensor<1x8xf32>
    %47 = "stablehlo.gather"(%45, %43) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 8>}> : (tensor<64x8xf32>, tensor<1xi32>) -> tensor<1x8xf32>
    %48 = stablehlo.reshape %46 : (tensor<1x8xf32>) -> tensor<1x1x8xf32>
    %49 = stablehlo.broadcast_in_dim %48, dims = [0, 1, 2] : (tensor<1x1x8xf32>) -> tensor<1x4x8xf32>
    %50 = stablehlo.reshape %47 : (tensor<1x8xf32>) -> tensor<1x1x8xf32>
    %51 = stablehlo.broadcast_in_dim %50, dims = [0, 1, 2] : (tensor<1x1x8xf32>) -> tensor<1x4x8xf32>
    %52 = stablehlo.reshape %46 : (tensor<1x8xf32>) -> tensor<1x1x8xf32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [0, 1, 2] : (tensor<1x1x8xf32>) -> tensor<1x2x8xf32>
    %54 = stablehlo.reshape %47 : (tensor<1x8xf32>) -> tensor<1x1x8xf32>
    %55 = stablehlo.broadcast_in_dim %54, dims = [0, 1, 2] : (tensor<1x1x8xf32>) -> tensor<1x2x8xf32>
    %56 = "stablehlo.gather"(%13, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 32>}> : (tensor<64x32xf32>, tensor<1x1xi32>) -> tensor<1x1x32xf32>
    %57 = stablehlo.reshape %56 : (tensor<1x1x32xf32>) -> tensor<1x32xf32>
    %58 = stablehlo.constant dense<[[1.0E-6]]> : tensor<1x1xf32>
    %59 = stablehlo.multiply %57, %57 : tensor<1x32xf32>
    %s213 = stablehlo.constant dense<0.0> : tensor<f32>
    %s214 = stablehlo.reduce(%59 init: %s213) applies stablehlo.add across dimensions = [1] : (tensor<1x32xf32>, tensor<f32>) -> tensor<1xf32>
    %s215 = stablehlo.constant dense<32.0> : tensor<1xf32>
    %s216 = stablehlo.divide %s214, %s215 : tensor<1xf32>
    %60 = stablehlo.broadcast_in_dim %s216, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %61 = stablehlo.add %60, %58 : tensor<1x1xf32>
    %62 = stablehlo.rsqrt %61 : tensor<1x1xf32>
    %s217 = stablehlo.broadcast_in_dim %62, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x32xf32>
    %63 = stablehlo.multiply %57, %s217 : tensor<1x32xf32>
    %64 = stablehlo.reshape %14 : (tensor<32xf32>) -> tensor<1x32xf32>
    %65 = stablehlo.broadcast_in_dim %64, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32xf32>
    %66 = stablehlo.multiply %63, %65 : tensor<1x32xf32>
    %67 = stablehlo.dot_general %66, %15, contracting_dims = [1] x [0] : (tensor<1x32xf32>, tensor<32x32xf32>) -> tensor<1x32xf32>
    %68 = stablehlo.dot_general %66, %16, contracting_dims = [1] x [0] : (tensor<1x32xf32>, tensor<32x16xf32>) -> tensor<1x16xf32>
    %69 = stablehlo.dot_general %66, %17, contracting_dims = [1] x [0] : (tensor<1x32xf32>, tensor<32x16xf32>) -> tensor<1x16xf32>
    %70 = stablehlo.reshape %67 : (tensor<1x32xf32>) -> tensor<1x4x8xf32>
    %71 = stablehlo.reshape %68 : (tensor<1x16xf32>) -> tensor<1x2x8xf32>
    %72 = stablehlo.reshape %69 : (tensor<1x16xf32>) -> tensor<1x2x8xf32>
    %73 = stablehlo.multiply %70, %49 : tensor<1x4x8xf32>
    %74 = stablehlo.slice %70 [0:1, 0:4, 4:8] : (tensor<1x4x8xf32>) -> tensor<1x4x4xf32>
    %75 = stablehlo.slice %70 [0:1, 0:4, 0:4] : (tensor<1x4x8xf32>) -> tensor<1x4x4xf32>
    %76 = stablehlo.negate %74 : tensor<1x4x4xf32>
    %77 = stablehlo.concatenate %76, %75, dim = 2 : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x8xf32>
    %78 = stablehlo.multiply %77, %51 : tensor<1x4x8xf32>
    %79 = stablehlo.add %73, %78 : tensor<1x4x8xf32>
    %80 = stablehlo.multiply %71, %53 : tensor<1x2x8xf32>
    %81 = stablehlo.slice %71 [0:1, 0:2, 4:8] : (tensor<1x2x8xf32>) -> tensor<1x2x4xf32>
    %82 = stablehlo.slice %71 [0:1, 0:2, 0:4] : (tensor<1x2x8xf32>) -> tensor<1x2x4xf32>
    %83 = stablehlo.negate %81 : tensor<1x2x4xf32>
    %84 = stablehlo.concatenate %83, %82, dim = 2 : (tensor<1x2x4xf32>, tensor<1x2x4xf32>) -> tensor<1x2x8xf32>
    %85 = stablehlo.multiply %84, %55 : tensor<1x2x8xf32>
    %86 = stablehlo.add %80, %85 : tensor<1x2x8xf32>
    %s218 = stablehlo.reshape %7 : (tensor<17x4x2x8xf32>) -> tensor<68x2x8xf32>
    %s219 = "stablehlo.scatter"(%s218, %6, %86) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s220: tensor<f32>, %s221: tensor<f32>):
       stablehlo.return %s221 : tensor<f32>
     }) : (tensor<68x2x8xf32>, tensor<1xi32>, tensor<1x2x8xf32>) -> tensor<68x2x8xf32>
    %87 = stablehlo.reshape %s219 : (tensor<68x2x8xf32>) -> tensor<17x4x2x8xf32>
    %s222 = stablehlo.reshape %8 : (tensor<17x4x2x8xf32>) -> tensor<68x2x8xf32>
    %s223 = "stablehlo.scatter"(%s222, %6, %72) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s224: tensor<f32>, %s225: tensor<f32>):
       stablehlo.return %s225 : tensor<f32>
     }) : (tensor<68x2x8xf32>, tensor<1xi32>, tensor<1x2x8xf32>) -> tensor<68x2x8xf32>
    %88 = stablehlo.reshape %s223 : (tensor<68x2x8xf32>) -> tensor<17x4x2x8xf32>
    %s226 = "stablehlo.gather"(%87, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<17x4x2x8xf32>, tensor<1x16xi32>) -> tensor<1x16x4x2x8xf32>
    %s227 = stablehlo.reshape %s226 : (tensor<1x16x4x2x8xf32>) -> tensor<1x64x2x8xf32>
    %s228 = "stablehlo.gather"(%88, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<17x4x2x8xf32>, tensor<1x16xi32>) -> tensor<1x16x4x2x8xf32>
    %s229 = stablehlo.reshape %s228 : (tensor<1x16x4x2x8xf32>) -> tensor<1x64x2x8xf32>
    %s230 = stablehlo.reshape %79 : (tensor<1x4x8xf32>) -> tensor<1x2x2x8xf32>
    %s231 = stablehlo.dot_general %s230, %s227, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [3], precision = [HIGHEST, HIGHEST] : (tensor<1x2x2x8xf32>, tensor<1x64x2x8xf32>) -> tensor<1x2x2x64xf32>
    %s232 = stablehlo.constant dense<0.35355338> : tensor<f32>
    %s233 = stablehlo.broadcast_in_dim %s232, dims = [] : (tensor<f32>) -> tensor<1x2x2x64xf32>
    %s234 = stablehlo.multiply %s231, %s233 : tensor<1x2x2x64xf32>
    %s235 = stablehlo.iota dim = 3 : tensor<1x2x2x64xi32>
    %s236 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<1xi32>) -> tensor<1x2x2x64xi32>
    %s238 = stablehlo.compare LT, %s235, %s236, SIGNED : (tensor<1x2x2x64xi32>, tensor<1x2x2x64xi32>) -> tensor<1x2x2x64xi1>
    %s239 = stablehlo.constant dense<8> : tensor<i32>
    %s240 = stablehlo.broadcast_in_dim %s239, dims = [] : (tensor<i32>) -> tensor<1x2x2x64xi32>
    %s241 = stablehlo.subtract %s236, %s240 : tensor<1x2x2x64xi32>
    %s242 = stablehlo.compare GE, %s235, %s241, SIGNED : (tensor<1x2x2x64xi32>, tensor<1x2x2x64xi32>) -> tensor<1x2x2x64xi1>
    %s237 = stablehlo.and %s238, %s242 : tensor<1x2x2x64xi1>
    %s243 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s244 = stablehlo.broadcast_in_dim %s243, dims = [] : (tensor<f32>) -> tensor<1x2x2x64xf32>
    %s245 = stablehlo.select %s237, %s234, %s244 : tensor<1x2x2x64xi1>, tensor<1x2x2x64xf32>
    %s246 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s247 = stablehlo.reduce(%s245 init: %s246) applies stablehlo.maximum across dimensions = [3] : (tensor<1x2x2x64xf32>, tensor<f32>) -> tensor<1x2x2xf32>
    %s248 = stablehlo.broadcast_in_dim %s247, dims = [0, 1, 2] : (tensor<1x2x2xf32>) -> tensor<1x2x2x64xf32>
    %s249 = stablehlo.subtract %s245, %s248 : tensor<1x2x2x64xf32>
    %s250 = stablehlo.exponential %s249 : tensor<1x2x2x64xf32>
    %s251 = stablehlo.constant dense<0.0> : tensor<f32>
    %s252 = stablehlo.reduce(%s250 init: %s251) applies stablehlo.add across dimensions = [3] : (tensor<1x2x2x64xf32>, tensor<f32>) -> tensor<1x2x2xf32>
    %s253 = stablehlo.broadcast_in_dim %s252, dims = [0, 1, 2] : (tensor<1x2x2xf32>) -> tensor<1x2x2x64xf32>
    %s254 = stablehlo.divide %s250, %s253 : tensor<1x2x2x64xf32>
    %s255 = stablehlo.dot_general %s254, %s229, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [HIGHEST, HIGHEST] : (tensor<1x2x2x64xf32>, tensor<1x64x2x8xf32>) -> tensor<1x2x2x8xf32>
    %89 = stablehlo.reshape %s255 : (tensor<1x2x2x8xf32>) -> tensor<1x4x8xf32>
    %90 = stablehlo.reshape %89 : (tensor<1x4x8xf32>) -> tensor<1x32xf32>
    %91 = stablehlo.dot_general %90, %18, contracting_dims = [1] x [0] : (tensor<1x32xf32>, tensor<32x32xf32>) -> tensor<1x32xf32>
    %92 = stablehlo.add %57, %91 : tensor<1x32xf32>
    %93 = stablehlo.multiply %92, %92 : tensor<1x32xf32>
    %s256 = stablehlo.constant dense<0.0> : tensor<f32>
    %s257 = stablehlo.reduce(%93 init: %s256) applies stablehlo.add across dimensions = [1] : (tensor<1x32xf32>, tensor<f32>) -> tensor<1xf32>
    %s258 = stablehlo.constant dense<32.0> : tensor<1xf32>
    %s259 = stablehlo.divide %s257, %s258 : tensor<1xf32>
    %94 = stablehlo.broadcast_in_dim %s259, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %95 = stablehlo.add %94, %58 : tensor<1x1xf32>
    %96 = stablehlo.rsqrt %95 : tensor<1x1xf32>
    %s260 = stablehlo.broadcast_in_dim %96, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x32xf32>
    %97 = stablehlo.multiply %92, %s260 : tensor<1x32xf32>
    %98 = stablehlo.reshape %19 : (tensor<32xf32>) -> tensor<1x32xf32>
    %99 = stablehlo.broadcast_in_dim %98, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32xf32>
    %100 = stablehlo.multiply %97, %99 : tensor<1x32xf32>
    %101 = stablehlo.dot_general %100, %20, contracting_dims = [1] x [0] : (tensor<1x32xf32>, tensor<32x48xf32>) -> tensor<1x48xf32>
    %102 = stablehlo.dot_general %100, %21, contracting_dims = [1] x [0] : (tensor<1x32xf32>, tensor<32x48xf32>) -> tensor<1x48xf32>
    %s261 = stablehlo.logistic %101 : tensor<1x48xf32>
    %103 = stablehlo.multiply %101, %s261 : tensor<1x48xf32>
    %104 = stablehlo.multiply %103, %102 : tensor<1x48xf32>
    %105 = stablehlo.dot_general %104, %22, contracting_dims = [1] x [0] : (tensor<1x48xf32>, tensor<48x32xf32>) -> tensor<1x32xf32>
    %106 = stablehlo.add %92, %105 : tensor<1x32xf32>
    %107 = stablehlo.multiply %106, %106 : tensor<1x32xf32>
    %s262 = stablehlo.constant dense<0.0> : tensor<f32>
    %s263 = stablehlo.reduce(%107 init: %s262) applies stablehlo.add across dimensions = [1] : (tensor<1x32xf32>, tensor<f32>) -> tensor<1xf32>
    %s264 = stablehlo.constant dense<32.0> : tensor<1xf32>
    %s265 = stablehlo.divide %s263, %s264 : tensor<1xf32>
    %108 = stablehlo.broadcast_in_dim %s265, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %109 = stablehlo.add %108, %58 : tensor<1x1xf32>
    %110 = stablehlo.rsqrt %109 : tensor<1x1xf32>
    %s266 = stablehlo.broadcast_in_dim %110, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x32xf32>
    %111 = stablehlo.multiply %106, %s266 : tensor<1x32xf32>
    %112 = stablehlo.reshape %23 : (tensor<32xf32>) -> tensor<1x32xf32>
    %113 = stablehlo.broadcast_in_dim %112, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32xf32>
    %114 = stablehlo.multiply %111, %113 : tensor<1x32xf32>
    %115 = stablehlo.dot_general %114, %24, contracting_dims = [1] x [0] : (tensor<1x32xf32>, tensor<32x32xf32>) -> tensor<1x32xf32>
    %116 = stablehlo.dot_general %114, %25, contracting_dims = [1] x [0] : (tensor<1x32xf32>, tensor<32x16xf32>) -> tensor<1x16xf32>
    %117 = stablehlo.dot_general %114, %26, contracting_dims = [1] x [0] : (tensor<1x32xf32>, tensor<32x16xf32>) -> tensor<1x16xf32>
    %118 = stablehlo.reshape %115 : (tensor<1x32xf32>) -> tensor<1x4x8xf32>
    %119 = stablehlo.reshape %116 : (tensor<1x16xf32>) -> tensor<1x2x8xf32>
    %120 = stablehlo.reshape %117 : (tensor<1x16xf32>) -> tensor<1x2x8xf32>
    %121 = stablehlo.multiply %118, %49 : tensor<1x4x8xf32>
    %122 = stablehlo.slice %118 [0:1, 0:4, 4:8] : (tensor<1x4x8xf32>) -> tensor<1x4x4xf32>
    %123 = stablehlo.slice %118 [0:1, 0:4, 0:4] : (tensor<1x4x8xf32>) -> tensor<1x4x4xf32>
    %124 = stablehlo.negate %122 : tensor<1x4x4xf32>
    %125 = stablehlo.concatenate %124, %123, dim = 2 : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x8xf32>
    %126 = stablehlo.multiply %125, %51 : tensor<1x4x8xf32>
    %127 = stablehlo.add %121, %126 : tensor<1x4x8xf32>
    %128 = stablehlo.multiply %119, %53 : tensor<1x2x8xf32>
    %129 = stablehlo.slice %119 [0:1, 0:2, 4:8] : (tensor<1x2x8xf32>) -> tensor<1x2x4xf32>
    %130 = stablehlo.slice %119 [0:1, 0:2, 0:4] : (tensor<1x2x8xf32>) -> tensor<1x2x4xf32>
    %131 = stablehlo.negate %129 : tensor<1x2x4xf32>
    %132 = stablehlo.concatenate %131, %130, dim = 2 : (tensor<1x2x4xf32>, tensor<1x2x4xf32>) -> tensor<1x2x8xf32>
    %133 = stablehlo.multiply %132, %55 : tensor<1x2x8xf32>
    %134 = stablehlo.add %128, %133 : tensor<1x2x8xf32>
    %s267 = stablehlo.reshape %9 : (tensor<17x4x2x8xf32>) -> tensor<68x2x8xf32>
    %s268 = "stablehlo.scatter"(%s267, %6, %134) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s269: tensor<f32>, %s270: tensor<f32>):
       stablehlo.return %s270 : tensor<f32>
     }) : (tensor<68x2x8xf32>, tensor<1xi32>, tensor<1x2x8xf32>) -> tensor<68x2x8xf32>
    %135 = stablehlo.reshape %s268 : (tensor<68x2x8xf32>) -> tensor<17x4x2x8xf32>
    %s271 = stablehlo.reshape %10 : (tensor<17x4x2x8xf32>) -> tensor<68x2x8xf32>
    %s272 = "stablehlo.scatter"(%s271, %6, %120) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s273: tensor<f32>, %s274: tensor<f32>):
       stablehlo.return %s274 : tensor<f32>
     }) : (tensor<68x2x8xf32>, tensor<1xi32>, tensor<1x2x8xf32>) -> tensor<68x2x8xf32>
    %136 = stablehlo.reshape %s272 : (tensor<68x2x8xf32>) -> tensor<17x4x2x8xf32>
    %s275 = "stablehlo.gather"(%135, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<17x4x2x8xf32>, tensor<1x16xi32>) -> tensor<1x16x4x2x8xf32>
    %s276 = stablehlo.reshape %s275 : (tensor<1x16x4x2x8xf32>) -> tensor<1x64x2x8xf32>
    %s277 = "stablehlo.gather"(%136, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<17x4x2x8xf32>, tensor<1x16xi32>) -> tensor<1x16x4x2x8xf32>
    %s278 = stablehlo.reshape %s277 : (tensor<1x16x4x2x8xf32>) -> tensor<1x64x2x8xf32>
    %s279 = stablehlo.reshape %127 : (tensor<1x4x8xf32>) -> tensor<1x2x2x8xf32>
    %s280 = stablehlo.dot_general %s279, %s276, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [3], precision = [HIGHEST, HIGHEST] : (tensor<1x2x2x8xf32>, tensor<1x64x2x8xf32>) -> tensor<1x2x2x64xf32>
    %s281 = stablehlo.constant dense<0.35355338> : tensor<f32>
    %s282 = stablehlo.broadcast_in_dim %s281, dims = [] : (tensor<f32>) -> tensor<1x2x2x64xf32>
    %s283 = stablehlo.multiply %s280, %s282 : tensor<1x2x2x64xf32>
    %s284 = stablehlo.iota dim = 3 : tensor<1x2x2x64xi32>
    %s285 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<1xi32>) -> tensor<1x2x2x64xi32>
    %s287 = stablehlo.compare LT, %s284, %s285, SIGNED : (tensor<1x2x2x64xi32>, tensor<1x2x2x64xi32>) -> tensor<1x2x2x64xi1>
    %s288 = stablehlo.constant dense<8> : tensor<i32>
    %s289 = stablehlo.broadcast_in_dim %s288, dims = [] : (tensor<i32>) -> tensor<1x2x2x64xi32>
    %s290 = stablehlo.subtract %s285, %s289 : tensor<1x2x2x64xi32>
    %s291 = stablehlo.compare GE, %s284, %s290, SIGNED : (tensor<1x2x2x64xi32>, tensor<1x2x2x64xi32>) -> tensor<1x2x2x64xi1>
    %s286 = stablehlo.and %s287, %s291 : tensor<1x2x2x64xi1>
    %s292 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s293 = stablehlo.broadcast_in_dim %s292, dims = [] : (tensor<f32>) -> tensor<1x2x2x64xf32>
    %s294 = stablehlo.select %s286, %s283, %s293 : tensor<1x2x2x64xi1>, tensor<1x2x2x64xf32>
    %s295 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s296 = stablehlo.reduce(%s294 init: %s295) applies stablehlo.maximum across dimensions = [3] : (tensor<1x2x2x64xf32>, tensor<f32>) -> tensor<1x2x2xf32>
    %s297 = stablehlo.broadcast_in_dim %s296, dims = [0, 1, 2] : (tensor<1x2x2xf32>) -> tensor<1x2x2x64xf32>
    %s298 = stablehlo.subtract %s294, %s297 : tensor<1x2x2x64xf32>
    %s299 = stablehlo.exponential %s298 : tensor<1x2x2x64xf32>
    %s300 = stablehlo.constant dense<0.0> : tensor<f32>
    %s301 = stablehlo.reduce(%s299 init: %s300) applies stablehlo.add across dimensions = [3] : (tensor<1x2x2x64xf32>, tensor<f32>) -> tensor<1x2x2xf32>
    %s302 = stablehlo.broadcast_in_dim %s301, dims = [0, 1, 2] : (tensor<1x2x2xf32>) -> tensor<1x2x2x64xf32>
    %s303 = stablehlo.divide %s299, %s302 : tensor<1x2x2x64xf32>
    %s304 = stablehlo.dot_general %s303, %s278, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [HIGHEST, HIGHEST] : (tensor<1x2x2x64xf32>, tensor<1x64x2x8xf32>) -> tensor<1x2x2x8xf32>
    %137 = stablehlo.reshape %s304 : (tensor<1x2x2x8xf32>) -> tensor<1x4x8xf32>
    %138 = stablehlo.reshape %137 : (tensor<1x4x8xf32>) -> tensor<1x32xf32>
    %139 = stablehlo.dot_general %138, %27, contracting_dims = [1] x [0] : (tensor<1x32xf32>, tensor<32x32xf32>) -> tensor<1x32xf32>
    %140 = stablehlo.add %106, %139 : tensor<1x32xf32>
    %141 = stablehlo.multiply %140, %140 : tensor<1x32xf32>
    %s305 = stablehlo.constant dense<0.0> : tensor<f32>
    %s306 = stablehlo.reduce(%141 init: %s305) applies stablehlo.add across dimensions = [1] : (tensor<1x32xf32>, tensor<f32>) -> tensor<1xf32>
    %s307 = stablehlo.constant dense<32.0> : tensor<1xf32>
    %s308 = stablehlo.divide %s306, %s307 : tensor<1xf32>
    %142 = stablehlo.broadcast_in_dim %s308, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %143 = stablehlo.add %142, %58 : tensor<1x1xf32>
    %144 = stablehlo.rsqrt %143 : tensor<1x1xf32>
    %s309 = stablehlo.broadcast_in_dim %144, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x32xf32>
    %145 = stablehlo.multiply %140, %s309 : tensor<1x32xf32>
    %146 = stablehlo.reshape %28 : (tensor<32xf32>) -> tensor<1x32xf32>
    %147 = stablehlo.broadcast_in_dim %146, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32xf32>
    %148 = stablehlo.multiply %145, %147 : tensor<1x32xf32>
    %149 = stablehlo.dot_general %148, %29, contracting_dims = [1] x [0] : (tensor<1x32xf32>, tensor<32x48xf32>) -> tensor<1x48xf32>
    %150 = stablehlo.dot_general %148, %30, contracting_dims = [1] x [0] : (tensor<1x32xf32>, tensor<32x48xf32>) -> tensor<1x48xf32>
    %s310 = stablehlo.logistic %149 : tensor<1x48xf32>
    %151 = stablehlo.multiply %149, %s310 : tensor<1x48xf32>
    %152 = stablehlo.multiply %151, %150 : tensor<1x48xf32>
    %153 = stablehlo.dot_general %152, %31, contracting_dims = [1] x [0] : (tensor<1x48xf32>, tensor<48x32xf32>) -> tensor<1x32xf32>
    %154 = stablehlo.add %140, %153 : tensor<1x32xf32>
    %155 = stablehlo.multiply %154, %154 : tensor<1x32xf32>
    %s311 = stablehlo.constant dense<0.0> : tensor<f32>
    %s312 = stablehlo.reduce(%155 init: %s311) applies stablehlo.add across dimensions = [1] : (tensor<1x32xf32>, tensor<f32>) -> tensor<1xf32>
    %s313 = stablehlo.constant dense<32.0> : tensor<1xf32>
    %s314 = stablehlo.divide %s312, %s313 : tensor<1xf32>
    %156 = stablehlo.broadcast_in_dim %s314, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %157 = stablehlo.add %156, %58 : tensor<1x1xf32>
    %158 = stablehlo.rsqrt %157 : tensor<1x1xf32>
    %s315 = stablehlo.broadcast_in_dim %158, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x32xf32>
    %159 = stablehlo.multiply %154, %s315 : tensor<1x32xf32>
    %160 = stablehlo.reshape %32 : (tensor<32xf32>) -> tensor<1x32xf32>
    %161 = stablehlo.broadcast_in_dim %160, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32xf32>
    %162 = stablehlo.multiply %159, %161 : tensor<1x32xf32>
    %163 = stablehlo.dot_general %162, %33, contracting_dims = [1] x [0] : (tensor<1x32xf32>, tensor<32x32xf32>) -> tensor<1x32xf32>
    %164 = stablehlo.dot_general %162, %34, contracting_dims = [1] x [0] : (tensor<1x32xf32>, tensor<32x16xf32>) -> tensor<1x16xf32>
    %165 = stablehlo.dot_general %162, %35, contracting_dims = [1] x [0] : (tensor<1x32xf32>, tensor<32x16xf32>) -> tensor<1x16xf32>
    %166 = stablehlo.reshape %163 : (tensor<1x32xf32>) -> tensor<1x4x8xf32>
    %167 = stablehlo.reshape %164 : (tensor<1x16xf32>) -> tensor<1x2x8xf32>
    %168 = stablehlo.reshape %165 : (tensor<1x16xf32>) -> tensor<1x2x8xf32>
    %169 = stablehlo.multiply %166, %49 : tensor<1x4x8xf32>
    %170 = stablehlo.slice %166 [0:1, 0:4, 4:8] : (tensor<1x4x8xf32>) -> tensor<1x4x4xf32>
    %171 = stablehlo.slice %166 [0:1, 0:4, 0:4] : (tensor<1x4x8xf32>) -> tensor<1x4x4xf32>
    %172 = stablehlo.negate %170 : tensor<1x4x4xf32>
    %173 = stablehlo.concatenate %172, %171, dim = 2 : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x8xf32>
    %174 = stablehlo.multiply %173, %51 : tensor<1x4x8xf32>
    %175 = stablehlo.add %169, %174 : tensor<1x4x8xf32>
    %176 = stablehlo.multiply %167, %53 : tensor<1x2x8xf32>
    %177 = stablehlo.slice %167 [0:1, 0:2, 4:8] : (tensor<1x2x8xf32>) -> tensor<1x2x4xf32>
    %178 = stablehlo.slice %167 [0:1, 0:2, 0:4] : (tensor<1x2x8xf32>) -> tensor<1x2x4xf32>
    %179 = stablehlo.negate %177 : tensor<1x2x4xf32>
    %180 = stablehlo.concatenate %179, %178, dim = 2 : (tensor<1x2x4xf32>, tensor<1x2x4xf32>) -> tensor<1x2x8xf32>
    %181 = stablehlo.multiply %180, %55 : tensor<1x2x8xf32>
    %182 = stablehlo.add %176, %181 : tensor<1x2x8xf32>
    %s316 = stablehlo.reshape %11 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s317 = "stablehlo.scatter"(%s316, %4, %182) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s318: tensor<f32>, %s319: tensor<f32>):
       stablehlo.return %s319 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<1xi32>, tensor<1x2x8xf32>) -> tensor<260x2x8xf32>
    %183 = stablehlo.reshape %s317 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s320 = stablehlo.reshape %12 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s321 = "stablehlo.scatter"(%s320, %4, %168) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s322: tensor<f32>, %s323: tensor<f32>):
       stablehlo.return %s323 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<1xi32>, tensor<1x2x8xf32>) -> tensor<260x2x8xf32>
    %184 = stablehlo.reshape %s321 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s324 = "stablehlo.gather"(%183, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<1x16xi32>) -> tensor<1x16x4x2x8xf32>
    %s325 = stablehlo.reshape %s324 : (tensor<1x16x4x2x8xf32>) -> tensor<1x64x2x8xf32>
    %s326 = "stablehlo.gather"(%184, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<1x16xi32>) -> tensor<1x16x4x2x8xf32>
    %s327 = stablehlo.reshape %s326 : (tensor<1x16x4x2x8xf32>) -> tensor<1x64x2x8xf32>
    %s328 = stablehlo.reshape %175 : (tensor<1x4x8xf32>) -> tensor<1x2x2x8xf32>
    %s329 = stablehlo.dot_general %s328, %s325, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [3], precision = [HIGHEST, HIGHEST] : (tensor<1x2x2x8xf32>, tensor<1x64x2x8xf32>) -> tensor<1x2x2x64xf32>
    %s330 = stablehlo.constant dense<0.35355338> : tensor<f32>
    %s331 = stablehlo.broadcast_in_dim %s330, dims = [] : (tensor<f32>) -> tensor<1x2x2x64xf32>
    %s332 = stablehlo.multiply %s329, %s331 : tensor<1x2x2x64xf32>
    %s333 = stablehlo.iota dim = 3 : tensor<1x2x2x64xi32>
    %s334 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<1xi32>) -> tensor<1x2x2x64xi32>
    %s335 = stablehlo.compare LT, %s333, %s334, SIGNED : (tensor<1x2x2x64xi32>, tensor<1x2x2x64xi32>) -> tensor<1x2x2x64xi1>
    %s336 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s337 = stablehlo.broadcast_in_dim %s336, dims = [] : (tensor<f32>) -> tensor<1x2x2x64xf32>
    %s338 = stablehlo.select %s335, %s332, %s337 : tensor<1x2x2x64xi1>, tensor<1x2x2x64xf32>
    %s339 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s340 = stablehlo.reduce(%s338 init: %s339) applies stablehlo.maximum across dimensions = [3] : (tensor<1x2x2x64xf32>, tensor<f32>) -> tensor<1x2x2xf32>
    %s341 = stablehlo.broadcast_in_dim %s340, dims = [0, 1, 2] : (tensor<1x2x2xf32>) -> tensor<1x2x2x64xf32>
    %s342 = stablehlo.subtract %s338, %s341 : tensor<1x2x2x64xf32>
    %s343 = stablehlo.exponential %s342 : tensor<1x2x2x64xf32>
    %s344 = stablehlo.constant dense<0.0> : tensor<f32>
    %s345 = stablehlo.reduce(%s343 init: %s344) applies stablehlo.add across dimensions = [3] : (tensor<1x2x2x64xf32>, tensor<f32>) -> tensor<1x2x2xf32>
    %s346 = stablehlo.broadcast_in_dim %s345, dims = [0, 1, 2] : (tensor<1x2x2xf32>) -> tensor<1x2x2x64xf32>
    %s347 = stablehlo.divide %s343, %s346 : tensor<1x2x2x64xf32>
    %s348 = stablehlo.dot_general %s347, %s327, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [HIGHEST, HIGHEST] : (tensor<1x2x2x64xf32>, tensor<1x64x2x8xf32>) -> tensor<1x2x2x8xf32>
    %185 = stablehlo.reshape %s348 : (tensor<1x2x2x8xf32>) -> tensor<1x4x8xf32>
    %186 = stablehlo.reshape %185 : (tensor<1x4x8xf32>) -> tensor<1x32xf32>
    %187 = stablehlo.dot_general %186, %36, contracting_dims = [1] x [0] : (tensor<1x32xf32>, tensor<32x32xf32>) -> tensor<1x32xf32>
    %188 = stablehlo.add %154, %187 : tensor<1x32xf32>
    %189 = stablehlo.multiply %188, %188 : tensor<1x32xf32>
    %s349 = stablehlo.constant dense<0.0> : tensor<f32>
    %s350 = stablehlo.reduce(%189 init: %s349) applies stablehlo.add across dimensions = [1] : (tensor<1x32xf32>, tensor<f32>) -> tensor<1xf32>
    %s351 = stablehlo.constant dense<32.0> : tensor<1xf32>
    %s352 = stablehlo.divide %s350, %s351 : tensor<1xf32>
    %190 = stablehlo.broadcast_in_dim %s352, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %191 = stablehlo.add %190, %58 : tensor<1x1xf32>
    %192 = stablehlo.rsqrt %191 : tensor<1x1xf32>
    %s353 = stablehlo.broadcast_in_dim %192, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x32xf32>
    %193 = stablehlo.multiply %188, %s353 : tensor<1x32xf32>
    %194 = stablehlo.reshape %37 : (tensor<32xf32>) -> tensor<1x32xf32>
    %195 = stablehlo.broadcast_in_dim %194, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32xf32>
    %196 = stablehlo.multiply %193, %195 : tensor<1x32xf32>
    %197 = stablehlo.dot_general %196, %38, contracting_dims = [1] x [0] : (tensor<1x32xf32>, tensor<32x48xf32>) -> tensor<1x48xf32>
    %198 = stablehlo.dot_general %196, %39, contracting_dims = [1] x [0] : (tensor<1x32xf32>, tensor<32x48xf32>) -> tensor<1x48xf32>
    %s354 = stablehlo.logistic %197 : tensor<1x48xf32>
    %199 = stablehlo.multiply %197, %s354 : tensor<1x48xf32>
    %200 = stablehlo.multiply %199, %198 : tensor<1x48xf32>
    %201 = stablehlo.dot_general %200, %40, contracting_dims = [1] x [0] : (tensor<1x48xf32>, tensor<48x32xf32>) -> tensor<1x32xf32>
    %202 = stablehlo.add %188, %201 : tensor<1x32xf32>
    %203 = stablehlo.multiply %202, %202 : tensor<1x32xf32>
    %s355 = stablehlo.constant dense<0.0> : tensor<f32>
    %s356 = stablehlo.reduce(%203 init: %s355) applies stablehlo.add across dimensions = [1] : (tensor<1x32xf32>, tensor<f32>) -> tensor<1xf32>
    %s357 = stablehlo.constant dense<32.0> : tensor<1xf32>
    %s358 = stablehlo.divide %s356, %s357 : tensor<1xf32>
    %204 = stablehlo.broadcast_in_dim %s358, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %205 = stablehlo.add %204, %58 : tensor<1x1xf32>
    %206 = stablehlo.rsqrt %205 : tensor<1x1xf32>
    %s359 = stablehlo.broadcast_in_dim %206, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x32xf32>
    %207 = stablehlo.multiply %202, %s359 : tensor<1x32xf32>
    %208 = stablehlo.reshape %41 : (tensor<32xf32>) -> tensor<1x32xf32>
    %209 = stablehlo.broadcast_in_dim %208, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<1x32xf32>
    %210 = stablehlo.multiply %207, %209 : tensor<1x32xf32>
    %211 = stablehlo.dot_general %210, %42, contracting_dims = [1] x [0] : (tensor<1x32xf32>, tensor<32x64xf32>) -> tensor<1x64xf32>
    %212 = stablehlo.reshape %211 : (tensor<1x64xf32>) -> tensor<1x1x64xf32>
    return %212, %87, %88, %135, %136, %183, %184 : tensor<1x1x64xf32>, tensor<17x4x2x8xf32>, tensor<17x4x2x8xf32>, tensor<17x4x2x8xf32>, tensor<17x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>
  }
}
