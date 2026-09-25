module {
  func.func @main(%0: tensor<2x64xi32>, %1: tensor<2x64xi32>, %2: tensor<2x16xi32>, %3: tensor<2xi32>, %4: tensor<128xi32>, %5: tensor<65x4x2x8xf32> {tf.aliasing_output = 1 : i32}, %6: tensor<65x4x2x8xf32> {tf.aliasing_output = 2 : i32}, %7: tensor<65x4x2x8xf32> {tf.aliasing_output = 3 : i32}, %8: tensor<65x4x2x8xf32> {tf.aliasing_output = 4 : i32}, %9: tensor<65x4x2x8xf32> {tf.aliasing_output = 5 : i32}, %10: tensor<65x4x2x8xf32> {tf.aliasing_output = 6 : i32}, %11: tensor<64x32xf32>, %12: tensor<32xf32>, %13: tensor<32x32xf32>, %14: tensor<32x16xf32>, %15: tensor<32x16xf32>, %16: tensor<32x32xf32>, %17: tensor<32xf32>, %18: tensor<32x48xf32>, %19: tensor<32x48xf32>, %20: tensor<48x32xf32>, %21: tensor<32xf32>, %22: tensor<32x32xf32>, %23: tensor<32x16xf32>, %24: tensor<32x16xf32>, %25: tensor<32x32xf32>, %26: tensor<32xf32>, %27: tensor<32x48xf32>, %28: tensor<32x48xf32>, %29: tensor<48x32xf32>, %30: tensor<32xf32>, %31: tensor<32x32xf32>, %32: tensor<32x16xf32>, %33: tensor<32x16xf32>, %34: tensor<32x32xf32>, %35: tensor<32xf32>, %36: tensor<32x48xf32>, %37: tensor<32x48xf32>, %38: tensor<48x32xf32>, %39: tensor<32xf32>, %40: tensor<32x64xf32>) -> (tensor<2x1x64xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>) {
    %41 = stablehlo.reshape %1 : (tensor<2x64xi32>) -> tensor<128xi32>
    %42 = stablehlo.constant dense<[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.5403023, 0.9950042, 0.99995, 0.9999995, 0.5403023, 0.9950042, 0.99995, 0.9999995], [-0.41614684, 0.9800666, 0.9998, 0.999998, -0.41614684, 0.9800666, 0.9998, 0.999998], [-0.9899925, 0.9553365, 0.99955004, 0.9999955, -0.9899925, 0.9553365, 0.99955004, 0.9999955], [-0.6536436, 0.921061, 0.9992001, 0.999992, -0.6536436, 0.921061, 0.9992001, 0.999992], [0.2836622, 0.87758255, 0.99875027, 0.9999875, 0.2836622, 0.87758255, 0.99875027, 0.9999875], [0.96017027, 0.8253356, 0.99820054, 0.999982, 0.96017027, 0.8253356, 0.99820054, 0.999982], [0.75390226, 0.7648422, 0.997551, 0.9999755, 0.75390226, 0.7648422, 0.997551, 0.9999755], [-0.14550003, 0.6967067, 0.99680173, 0.999968, -0.14550003, 0.6967067, 0.99680173, 0.999968], [-0.91113025, 0.62161, 0.9959527, 0.9999595, -0.91113025, 0.62161, 0.9959527, 0.9999595], [-0.8390715, 0.5403023, 0.9950042, 0.99995, -0.8390715, 0.5403023, 0.9950042, 0.99995], [0.004425698, 0.45359612, 0.9939561, 0.9999395, 0.004425698, 0.45359612, 0.9939561, 0.9999395], [0.84385395, 0.36235777, 0.99280864, 0.999928, 0.84385395, 0.36235777, 0.99280864, 0.999928], [0.9074468, 0.26749882, 0.9915619, 0.9999155, 0.9074468, 0.26749882, 0.9915619, 0.9999155], [0.13673721, 0.16996714, 0.990216, 0.999902, 0.13673721, 0.16996714, 0.990216, 0.999902], [-0.7596879, 0.0707372, 0.9887711, 0.9998875, -0.7596879, 0.0707372, 0.9887711, 0.9998875], [-0.9576595, -0.029199522, 0.98722726, 0.999872, -0.9576595, -0.029199522, 0.98722726, 0.999872], [-0.27516335, -0.1288445, 0.9855848, 0.9998555, -0.27516335, -0.1288445, 0.9855848, 0.9998555], [0.6603167, -0.22720209, 0.9838437, 0.999838, 0.6603167, -0.22720209, 0.9838437, 0.999838], [0.9887046, -0.32328957, 0.9820042, 0.9998195, 0.9887046, -0.32328957, 0.9820042, 0.9998195], [0.40808207, -0.41614684, 0.9800666, 0.9998, 0.40808207, -0.41614684, 0.9800666, 0.9998], [-0.54772925, -0.5048461, 0.9780309, 0.9997795, -0.54772925, -0.5048461, 0.9780309, 0.9997795], [-0.99996084, -0.5885011, 0.97589743, 0.999758, -0.99996084, -0.5885011, 0.97589743, 0.999758], [-0.53283304, -0.66627604, 0.97366637, 0.99973553, -0.53283304, -0.66627604, 0.97366637, 0.99973553], [0.42417902, -0.73739374, 0.971338, 0.999712, 0.42417902, -0.73739374, 0.971338, 0.999712], [0.99120283, -0.8011436, 0.9689124, 0.9996875, 0.99120283, -0.8011436, 0.9689124, 0.9996875], [0.6469193, -0.8568888, 0.96638995, 0.99966204, 0.6469193, -0.8568888, 0.96638995, 0.99966204], [-0.29213881, -0.90407217, 0.9637709, 0.9996355, -0.29213881, -0.90407217, 0.9637709, 0.9996355], [-0.9626059, -0.94222236, 0.96105546, 0.99960804, -0.9626059, -0.94222236, 0.96105546, 0.99960804], [-0.74805754, -0.9709582, 0.95824385, 0.99957955, -0.74805754, -0.9709582, 0.95824385, 0.99957955], [0.15425146, -0.9899925, 0.9553365, 0.99955004, 0.15425146, -0.9899925, 0.9553365, 0.99955004], [0.91474235, -0.99913514, 0.95233357, 0.9995195, 0.91474235, -0.99913514, 0.95233357, 0.9995195], [0.8342234, -0.9982948, 0.94923544, 0.99948806, 0.8342234, -0.9982948, 0.94923544, 0.99948806], [-0.013276747, -0.98747975, 0.94604236, 0.9994556, -0.013276747, -0.98747975, 0.94604236, 0.9994556], [-0.8485703, -0.9667982, 0.9427547, 0.9994221, -0.8485703, -0.9667982, 0.9427547, 0.9994221], [-0.9036922, -0.9364567, 0.9393727, 0.99938756, -0.9036922, -0.9364567, 0.9393727, 0.99938756], [-0.12796369, -0.89675844, 0.9358968, 0.9993521, -0.12796369, -0.89675844, 0.9358968, 0.9993521], [0.76541406, -0.8481, 0.93232733, 0.99931556, 0.76541406, -0.8481, 0.93232733, 0.99931556], [0.95507365, -0.7909677, 0.9286646, 0.99927807, 0.95507365, -0.7909677, 0.9286646, 0.99927807], [0.26664293, -0.7259323, 0.92490906, 0.9992396, 0.26664293, -0.7259323, 0.92490906, 0.9992396], [-0.66693807, -0.6536436, 0.921061, 0.9992001, -0.66693807, -0.6536436, 0.921061, 0.9992001], [-0.98733926, -0.574824, 0.9171208, 0.99915963, -0.98733926, -0.574824, 0.9171208, 0.99915963], [-0.3999853, -0.4902608, 0.9130889, 0.99911815, -0.3999853, -0.4902608, 0.9130889, 0.99911815], [0.5551133, -0.40079919, 0.90896577, 0.99907565, 0.5551133, -0.40079919, 0.90896577, 0.99907565], [0.9998433, -0.30733287, 0.90475166, 0.99903214, 0.9998433, -0.30733287, 0.90475166, 0.99903214], [0.52532196, -0.2107958, 0.90044713, 0.9989877, 0.52532196, -0.2107958, 0.90044713, 0.9989877], [-0.43217793, -0.112152524, 0.8960525, 0.9989422, -0.43217793, -0.112152524, 0.8960525, 0.9989422], [-0.9923355, -0.012388663, 0.8915683, 0.9988957, -0.9923355, -0.012388663, 0.8915683, 0.9988957], [-0.64014435, 0.087498985, 0.8869949, 0.9988482, -0.64014435, 0.087498985, 0.8869949, 0.9988482], [0.30059254, 0.18651237, 0.88233286, 0.99879974, 0.30059254, 0.18651237, 0.88233286, 0.99879974], [0.964966, 0.2836622, 0.87758255, 0.99875027, 0.964966, 0.2836622, 0.87758255, 0.99875027], [0.7421542, 0.37797773, 0.8727445, 0.9986998, 0.7421542, 0.37797773, 0.8727445, 0.9986998], [-0.16299078, 0.46851668, 0.8678192, 0.9986483, -0.16299078, 0.46851668, 0.8678192, 0.9986483], [-0.9182828, 0.55437434, 0.8628071, 0.99859583, -0.9182828, 0.55437434, 0.8628071, 0.99859583], [-0.8293098, 0.63469285, 0.8577087, 0.99854237, -0.8293098, 0.63469285, 0.8577087, 0.99854237], [0.022126757, 0.7086698, 0.8525245, 0.9984879, 0.022126757, 0.7086698, 0.8525245, 0.9984879], [0.8532201, 0.77556586, 0.8472551, 0.9984324, 0.8532201, 0.77556586, 0.8472551, 0.9984324], [0.8998668, 0.8347128, 0.841901, 0.99837595, 0.8998668, 0.8347128, 0.841901, 0.99837595], [0.119180135, 0.8855195, 0.8364627, 0.9983185, 0.119180135, 0.8855195, 0.8364627, 0.9983185], [-0.7710802, 0.92747843, 0.83094066, 0.99826, -0.7710802, 0.92747843, 0.83094066, 0.99826], [-0.95241296, 0.96017027, 0.8253356, 0.99820054, -0.95241296, 0.96017027, 0.8253356, 0.99820054], [-0.25810164, 0.98326844, 0.819648, 0.9981401, -0.25810164, 0.98326844, 0.819648, 0.9981401], [0.67350715, 0.9965421, 0.8138785, 0.99807864, 0.67350715, 0.9965421, 0.8138785, 0.99807864], [0.9858966, 0.9998586, 0.8080275, 0.9980162, 0.9858966, 0.9998586, 0.8080275, 0.9980162]]> : tensor<64x8xf32>
    %43 = stablehlo.constant dense<[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.84147096, 0.099833414, 0.009999833, 9.999998E-4, 0.84147096, 0.099833414, 0.009999833, 9.999998E-4], [0.9092974, 0.19866933, 0.019998666, 0.0019999987, 0.9092974, 0.19866933, 0.019998666, 0.0019999987], [0.14112, 0.29552022, 0.029995501, 0.0029999956, 0.14112, 0.29552022, 0.029995501, 0.0029999956], [-0.7568025, 0.38941833, 0.039989334, 0.0039999895, -0.7568025, 0.38941833, 0.039989334, 0.0039999895], [-0.9589243, 0.47942555, 0.04997917, 0.0049999794, -0.9589243, 0.47942555, 0.04997917, 0.0049999794], [-0.2794155, 0.5646425, 0.059964005, 0.005999964, -0.2794155, 0.5646425, 0.059964005, 0.005999964], [0.6569866, 0.64421767, 0.06994285, 0.006999943, 0.6569866, 0.64421767, 0.06994285, 0.006999943], [0.98935825, 0.7173561, 0.0799147, 0.007999915, 0.98935825, 0.7173561, 0.0799147, 0.007999915], [0.4121185, 0.7833269, 0.08987855, 0.008999879, 0.4121185, 0.7833269, 0.08987855, 0.008999879], [-0.5440211, 0.84147096, 0.099833414, 0.009999833, -0.5440211, 0.84147096, 0.099833414, 0.009999833], [-0.9999902, 0.89120734, 0.1097783, 0.010999778, -0.9999902, 0.89120734, 0.1097783, 0.010999778], [-0.53657293, 0.9320391, 0.119712204, 0.011999712, -0.53657293, 0.9320391, 0.119712204, 0.011999712], [0.42016703, 0.9635582, 0.12963414, 0.012999634, 0.42016703, 0.9635582, 0.12963414, 0.012999634], [0.9906074, 0.98544973, 0.13954312, 0.013999542, 0.9906074, 0.98544973, 0.13954312, 0.013999542], [0.65028787, 0.997495, 0.14943813, 0.014999437, 0.65028787, 0.997495, 0.14943813, 0.014999437], [-0.2879033, 0.9995736, 0.15931821, 0.015999317, -0.2879033, 0.9995736, 0.15931821, 0.015999317], [-0.96139747, 0.9916648, 0.16918235, 0.016999181, -0.96139747, 0.9916648, 0.16918235, 0.016999181], [-0.75098723, 0.9738476, 0.17902957, 0.017999029, -0.75098723, 0.9738476, 0.17902957, 0.017999029], [0.1498772, 0.9463001, 0.1888589, 0.018998858, 0.1498772, 0.9463001, 0.1888589, 0.018998858], [0.9129453, 0.9092974, 0.19866933, 0.019998666, 0.9129453, 0.9092974, 0.19866933, 0.019998666], [0.8366556, 0.86320937, 0.2084599, 0.020998457, 0.8366556, 0.86320937, 0.2084599, 0.020998457], [-0.008851309, 0.8084964, 0.21822962, 0.021998225, -0.008851309, 0.8084964, 0.21822962, 0.021998225], [-0.84622043, 0.7457052, 0.22797753, 0.022997972, -0.84622043, 0.7457052, 0.22797753, 0.022997972], [-0.9055784, 0.6754632, 0.23770262, 0.023997696, -0.9055784, 0.6754632, 0.23770262, 0.023997696], [-0.13235176, 0.5984721, 0.24740396, 0.024997396, -0.13235176, 0.5984721, 0.24740396, 0.024997396], [0.76255846, 0.5155014, 0.25708055, 0.02599707, 0.76255846, 0.5155014, 0.25708055, 0.02599707], [0.95637596, 0.42737988, 0.26673144, 0.026996719, 0.95637596, 0.42737988, 0.26673144, 0.026996719], [0.2709058, 0.33498815, 0.27635565, 0.02799634, 0.2709058, 0.33498815, 0.27635565, 0.02799634], [-0.6636339, 0.23924933, 0.2859522, 0.028995935, -0.6636339, 0.23924933, 0.2859522, 0.028995935], [-0.9880316, 0.14112, 0.29552022, 0.029995501, -0.9880316, 0.14112, 0.29552022, 0.029995501], [-0.40403765, 0.041580662, 0.30505863, 0.030995036, -0.40403765, 0.041580662, 0.30505863, 0.030995036], [0.5514267, -0.058374144, 0.31456655, 0.03199454, 0.5514267, -0.058374144, 0.31456655, 0.03199454], [0.99991184, -0.15774569, 0.32404304, 0.03299401, 0.99991184, -0.15774569, 0.32404304, 0.03299401], [0.5290827, -0.25554112, 0.3334871, 0.03399345, 0.5290827, -0.25554112, 0.3334871, 0.03399345], [-0.42818266, -0.35078323, 0.3428978, 0.034992855, -0.42818266, -0.35078323, 0.3428978, 0.034992855], [-0.99177885, -0.44252044, 0.35227424, 0.035992224, -0.99177885, -0.44252044, 0.35227424, 0.035992224], [-0.6435381, -0.5298361, 0.36161542, 0.03699156, -0.6435381, -0.5298361, 0.36161542, 0.03699156], [0.29636857, -0.6118579, 0.37092048, 0.037990857, 0.29636857, -0.6118579, 0.37092048, 0.037990857], [0.96379536, -0.68776613, 0.3801884, 0.038990114, 0.96379536, -0.68776613, 0.3801884, 0.038990114], [0.74511313, -0.7568025, 0.38941833, 0.039989334, 0.74511313, -0.7568025, 0.38941833, 0.039989334], [-0.15862267, -0.8182771, 0.39860934, 0.040988512, -0.15862267, -0.8182771, 0.39860934, 0.040988512], [-0.91652155, -0.8715758, 0.40776044, 0.041987654, -0.91652155, -0.8715758, 0.40776044, 0.041987654], [-0.8317748, -0.91616595, 0.4168708, 0.04298675, -0.8317748, -0.91616595, 0.4168708, 0.04298675], [0.017701926, -0.9516021, 0.42593947, 0.043985803, 0.017701926, -0.9516021, 0.42593947, 0.043985803], [0.8509035, -0.9775301, 0.43496552, 0.044984814, 0.8509035, -0.9775301, 0.43496552, 0.044984814], [0.90178835, -0.993691, 0.44394812, 0.04598378, 0.90178835, -0.993691, 0.44394812, 0.04598378], [0.123573124, -0.9999232, 0.45288628, 0.0469827, 0.123573124, -0.9999232, 0.45288628, 0.0469827], [-0.76825464, -0.9961646, 0.46177918, 0.04798157, -0.76825464, -0.9961646, 0.46177918, 0.04798157], [-0.95375264, -0.98245263, 0.47062588, 0.048980393, -0.95375264, -0.98245263, 0.47062588, 0.048980393], [-0.26237485, -0.9589243, 0.47942555, 0.04997917, -0.26237485, -0.9589243, 0.47942555, 0.04997917], [0.6702292, -0.9258147, 0.48817724, 0.050977893, 0.6702292, -0.9258147, 0.48817724, 0.050977893], [0.9866276, -0.8834547, 0.49688014, 0.05197657, 0.9866276, -0.8834547, 0.49688014, 0.05197657], [0.39592516, -0.83226746, 0.50553334, 0.05297519, 0.39592516, -0.83226746, 0.50553334, 0.05297519], [-0.5587891, -0.7727645, 0.514136, 0.05397376, -0.5587891, -0.7727645, 0.514136, 0.05397376], [-0.99975514, -0.7055403, 0.52268726, 0.054972276, -0.99975514, -0.7055403, 0.52268726, 0.054972276], [-0.521551, -0.63126665, 0.5311862, 0.055970736, -0.521551, -0.63126665, 0.5311862, 0.055970736], [0.43616477, -0.5506855, 0.539632, 0.05696914, 0.43616477, -0.5506855, 0.539632, 0.05696914], [0.99287266, -0.46460217, 0.54802394, 0.057967488, 0.99287266, -0.46460217, 0.54802394, 0.057967488], [0.636738, -0.37387666, 0.556361, 0.058965776, 0.636738, -0.37387666, 0.556361, 0.058965776], [-0.3048106, -0.2794155, 0.5646425, 0.059964005, -0.3048106, -0.2794155, 0.5646425, 0.059964005], [-0.9661178, -0.18216251, 0.57286745, 0.060962178, -0.9661178, -0.18216251, 0.57286745, 0.060962178], [-0.7391807, -0.083089404, 0.58103514, 0.061960287, -0.7391807, -0.083089404, 0.58103514, 0.061960287], [0.1673557, 0.0168139, 0.58914477, 0.06295834, 0.1673557, 0.0168139, 0.58914477, 0.06295834]]> : tensor<64x8xf32>
    %44 = "stablehlo.gather"(%42, %41) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 8>}> : (tensor<64x8xf32>, tensor<128xi32>) -> tensor<128x8xf32>
    %45 = "stablehlo.gather"(%43, %41) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 8>}> : (tensor<64x8xf32>, tensor<128xi32>) -> tensor<128x8xf32>
    %46 = stablehlo.reshape %44 : (tensor<128x8xf32>) -> tensor<128x1x8xf32>
    %47 = stablehlo.broadcast_in_dim %46, dims = [0, 1, 2] : (tensor<128x1x8xf32>) -> tensor<128x4x8xf32>
    %48 = stablehlo.reshape %45 : (tensor<128x8xf32>) -> tensor<128x1x8xf32>
    %49 = stablehlo.broadcast_in_dim %48, dims = [0, 1, 2] : (tensor<128x1x8xf32>) -> tensor<128x4x8xf32>
    %50 = stablehlo.reshape %44 : (tensor<128x8xf32>) -> tensor<128x1x8xf32>
    %51 = stablehlo.broadcast_in_dim %50, dims = [0, 1, 2] : (tensor<128x1x8xf32>) -> tensor<128x2x8xf32>
    %52 = stablehlo.reshape %45 : (tensor<128x8xf32>) -> tensor<128x1x8xf32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [0, 1, 2] : (tensor<128x1x8xf32>) -> tensor<128x2x8xf32>
    %54 = stablehlo.reshape %2 : (tensor<2x16xi32>) -> tensor<2x1x16xi32>
    %55 = stablehlo.broadcast_in_dim %54, dims = [0, 1, 2] : (tensor<2x1x16xi32>) -> tensor<2x64x16xi32>
    %56 = stablehlo.reshape %55 : (tensor<2x64x16xi32>) -> tensor<128x16xi32>
    %57 = stablehlo.constant dense<1> : tensor<128xi32>
    %58 = stablehlo.add %41, %57 : tensor<128xi32>
    %59 = "stablehlo.gather"(%11, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 32>}> : (tensor<64x32xf32>, tensor<2x64xi32>) -> tensor<2x64x32xf32>
    %60 = stablehlo.reshape %59 : (tensor<2x64x32xf32>) -> tensor<128x32xf32>
    %61 = stablehlo.constant dense<[[1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6], [1.0E-6]]> : tensor<128x1xf32>
    %62 = stablehlo.multiply %60, %60 : tensor<128x32xf32>
    %s220 = stablehlo.constant dense<0.0> : tensor<f32>
    %s221 = stablehlo.reduce(%62 init: %s220) applies stablehlo.add across dimensions = [1] : (tensor<128x32xf32>, tensor<f32>) -> tensor<128xf32>
    %s222 = stablehlo.constant dense<32.0> : tensor<128xf32>
    %s223 = stablehlo.divide %s221, %s222 : tensor<128xf32>
    %63 = stablehlo.broadcast_in_dim %s223, dims = [0] : (tensor<128xf32>) -> tensor<128x1xf32>
    %64 = stablehlo.add %63, %61 : tensor<128x1xf32>
    %65 = stablehlo.rsqrt %64 : tensor<128x1xf32>
    %s224 = stablehlo.broadcast_in_dim %65, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x32xf32>
    %66 = stablehlo.multiply %60, %s224 : tensor<128x32xf32>
    %67 = stablehlo.reshape %12 : (tensor<32xf32>) -> tensor<1x32xf32>
    %68 = stablehlo.broadcast_in_dim %67, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<128x32xf32>
    %69 = stablehlo.multiply %66, %68 : tensor<128x32xf32>
    %70 = stablehlo.dot_general %69, %13, contracting_dims = [1] x [0] : (tensor<128x32xf32>, tensor<32x32xf32>) -> tensor<128x32xf32>
    %71 = stablehlo.dot_general %69, %14, contracting_dims = [1] x [0] : (tensor<128x32xf32>, tensor<32x16xf32>) -> tensor<128x16xf32>
    %72 = stablehlo.dot_general %69, %15, contracting_dims = [1] x [0] : (tensor<128x32xf32>, tensor<32x16xf32>) -> tensor<128x16xf32>
    %73 = stablehlo.reshape %70 : (tensor<128x32xf32>) -> tensor<128x4x8xf32>
    %74 = stablehlo.reshape %71 : (tensor<128x16xf32>) -> tensor<128x2x8xf32>
    %75 = stablehlo.reshape %72 : (tensor<128x16xf32>) -> tensor<128x2x8xf32>
    %76 = stablehlo.multiply %73, %47 : tensor<128x4x8xf32>
    %77 = stablehlo.slice %73 [0:128, 0:4, 4:8] : (tensor<128x4x8xf32>) -> tensor<128x4x4xf32>
    %78 = stablehlo.slice %73 [0:128, 0:4, 0:4] : (tensor<128x4x8xf32>) -> tensor<128x4x4xf32>
    %79 = stablehlo.negate %77 : tensor<128x4x4xf32>
    %80 = stablehlo.concatenate %79, %78, dim = 2 : (tensor<128x4x4xf32>, tensor<128x4x4xf32>) -> tensor<128x4x8xf32>
    %81 = stablehlo.multiply %80, %49 : tensor<128x4x8xf32>
    %82 = stablehlo.add %76, %81 : tensor<128x4x8xf32>
    %83 = stablehlo.multiply %74, %51 : tensor<128x2x8xf32>
    %84 = stablehlo.slice %74 [0:128, 0:2, 4:8] : (tensor<128x2x8xf32>) -> tensor<128x2x4xf32>
    %85 = stablehlo.slice %74 [0:128, 0:2, 0:4] : (tensor<128x2x8xf32>) -> tensor<128x2x4xf32>
    %86 = stablehlo.negate %84 : tensor<128x2x4xf32>
    %87 = stablehlo.concatenate %86, %85, dim = 2 : (tensor<128x2x4xf32>, tensor<128x2x4xf32>) -> tensor<128x2x8xf32>
    %88 = stablehlo.multiply %87, %53 : tensor<128x2x8xf32>
    %89 = stablehlo.add %83, %88 : tensor<128x2x8xf32>
    %s225 = stablehlo.reshape %5 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s226 = "stablehlo.scatter"(%s225, %4, %89) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s227: tensor<f32>, %s228: tensor<f32>):
       stablehlo.return %s228 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<128xi32>, tensor<128x2x8xf32>) -> tensor<260x2x8xf32>
    %90 = stablehlo.reshape %s226 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s229 = stablehlo.reshape %6 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s230 = "stablehlo.scatter"(%s229, %4, %75) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s231: tensor<f32>, %s232: tensor<f32>):
       stablehlo.return %s232 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<128xi32>, tensor<128x2x8xf32>) -> tensor<260x2x8xf32>
    %91 = stablehlo.reshape %s230 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s233 = "stablehlo.gather"(%90, %56) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<128x16xi32>) -> tensor<128x16x4x2x8xf32>
    %s234 = stablehlo.reshape %s233 : (tensor<128x16x4x2x8xf32>) -> tensor<128x64x2x8xf32>
    %s235 = "stablehlo.gather"(%91, %56) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<128x16xi32>) -> tensor<128x16x4x2x8xf32>
    %s236 = stablehlo.reshape %s235 : (tensor<128x16x4x2x8xf32>) -> tensor<128x64x2x8xf32>
    %s237 = stablehlo.reshape %82 : (tensor<128x4x8xf32>) -> tensor<128x2x2x8xf32>
    %s238 = stablehlo.dot_general %s237, %s234, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [3], precision = [HIGHEST, HIGHEST] : (tensor<128x2x2x8xf32>, tensor<128x64x2x8xf32>) -> tensor<128x2x2x64xf32>
    %s239 = stablehlo.constant dense<0.35355338> : tensor<f32>
    %s240 = stablehlo.broadcast_in_dim %s239, dims = [] : (tensor<f32>) -> tensor<128x2x2x64xf32>
    %s241 = stablehlo.multiply %s238, %s240 : tensor<128x2x2x64xf32>
    %s242 = stablehlo.iota dim = 3 : tensor<128x2x2x64xi32>
    %s243 = stablehlo.broadcast_in_dim %58, dims = [0] : (tensor<128xi32>) -> tensor<128x2x2x64xi32>
    %s245 = stablehlo.compare LT, %s242, %s243, SIGNED : (tensor<128x2x2x64xi32>, tensor<128x2x2x64xi32>) -> tensor<128x2x2x64xi1>
    %s246 = stablehlo.constant dense<8> : tensor<i32>
    %s247 = stablehlo.broadcast_in_dim %s246, dims = [] : (tensor<i32>) -> tensor<128x2x2x64xi32>
    %s248 = stablehlo.subtract %s243, %s247 : tensor<128x2x2x64xi32>
    %s249 = stablehlo.compare GE, %s242, %s248, SIGNED : (tensor<128x2x2x64xi32>, tensor<128x2x2x64xi32>) -> tensor<128x2x2x64xi1>
    %s244 = stablehlo.and %s245, %s249 : tensor<128x2x2x64xi1>
    %s250 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s251 = stablehlo.broadcast_in_dim %s250, dims = [] : (tensor<f32>) -> tensor<128x2x2x64xf32>
    %s252 = stablehlo.select %s244, %s241, %s251 : tensor<128x2x2x64xi1>, tensor<128x2x2x64xf32>
    %s253 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s254 = stablehlo.reduce(%s252 init: %s253) applies stablehlo.maximum across dimensions = [3] : (tensor<128x2x2x64xf32>, tensor<f32>) -> tensor<128x2x2xf32>
    %s255 = stablehlo.broadcast_in_dim %s254, dims = [0, 1, 2] : (tensor<128x2x2xf32>) -> tensor<128x2x2x64xf32>
    %s256 = stablehlo.subtract %s252, %s255 : tensor<128x2x2x64xf32>
    %s257 = stablehlo.exponential %s256 : tensor<128x2x2x64xf32>
    %s258 = stablehlo.constant dense<0.0> : tensor<f32>
    %s259 = stablehlo.reduce(%s257 init: %s258) applies stablehlo.add across dimensions = [3] : (tensor<128x2x2x64xf32>, tensor<f32>) -> tensor<128x2x2xf32>
    %s260 = stablehlo.broadcast_in_dim %s259, dims = [0, 1, 2] : (tensor<128x2x2xf32>) -> tensor<128x2x2x64xf32>
    %s261 = stablehlo.divide %s257, %s260 : tensor<128x2x2x64xf32>
    %s262 = stablehlo.dot_general %s261, %s236, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [HIGHEST, HIGHEST] : (tensor<128x2x2x64xf32>, tensor<128x64x2x8xf32>) -> tensor<128x2x2x8xf32>
    %92 = stablehlo.reshape %s262 : (tensor<128x2x2x8xf32>) -> tensor<128x4x8xf32>
    %93 = stablehlo.reshape %92 : (tensor<128x4x8xf32>) -> tensor<128x32xf32>
    %94 = stablehlo.dot_general %93, %16, contracting_dims = [1] x [0] : (tensor<128x32xf32>, tensor<32x32xf32>) -> tensor<128x32xf32>
    %95 = stablehlo.add %60, %94 : tensor<128x32xf32>
    %96 = stablehlo.multiply %95, %95 : tensor<128x32xf32>
    %s263 = stablehlo.constant dense<0.0> : tensor<f32>
    %s264 = stablehlo.reduce(%96 init: %s263) applies stablehlo.add across dimensions = [1] : (tensor<128x32xf32>, tensor<f32>) -> tensor<128xf32>
    %s265 = stablehlo.constant dense<32.0> : tensor<128xf32>
    %s266 = stablehlo.divide %s264, %s265 : tensor<128xf32>
    %97 = stablehlo.broadcast_in_dim %s266, dims = [0] : (tensor<128xf32>) -> tensor<128x1xf32>
    %98 = stablehlo.add %97, %61 : tensor<128x1xf32>
    %99 = stablehlo.rsqrt %98 : tensor<128x1xf32>
    %s267 = stablehlo.broadcast_in_dim %99, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x32xf32>
    %100 = stablehlo.multiply %95, %s267 : tensor<128x32xf32>
    %101 = stablehlo.reshape %17 : (tensor<32xf32>) -> tensor<1x32xf32>
    %102 = stablehlo.broadcast_in_dim %101, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<128x32xf32>
    %103 = stablehlo.multiply %100, %102 : tensor<128x32xf32>
    %104 = stablehlo.dot_general %103, %18, contracting_dims = [1] x [0] : (tensor<128x32xf32>, tensor<32x48xf32>) -> tensor<128x48xf32>
    %105 = stablehlo.dot_general %103, %19, contracting_dims = [1] x [0] : (tensor<128x32xf32>, tensor<32x48xf32>) -> tensor<128x48xf32>
    %s268 = stablehlo.logistic %104 : tensor<128x48xf32>
    %106 = stablehlo.multiply %104, %s268 : tensor<128x48xf32>
    %107 = stablehlo.multiply %106, %105 : tensor<128x48xf32>
    %108 = stablehlo.dot_general %107, %20, contracting_dims = [1] x [0] : (tensor<128x48xf32>, tensor<48x32xf32>) -> tensor<128x32xf32>
    %109 = stablehlo.add %95, %108 : tensor<128x32xf32>
    %110 = stablehlo.multiply %109, %109 : tensor<128x32xf32>
    %s269 = stablehlo.constant dense<0.0> : tensor<f32>
    %s270 = stablehlo.reduce(%110 init: %s269) applies stablehlo.add across dimensions = [1] : (tensor<128x32xf32>, tensor<f32>) -> tensor<128xf32>
    %s271 = stablehlo.constant dense<32.0> : tensor<128xf32>
    %s272 = stablehlo.divide %s270, %s271 : tensor<128xf32>
    %111 = stablehlo.broadcast_in_dim %s272, dims = [0] : (tensor<128xf32>) -> tensor<128x1xf32>
    %112 = stablehlo.add %111, %61 : tensor<128x1xf32>
    %113 = stablehlo.rsqrt %112 : tensor<128x1xf32>
    %s273 = stablehlo.broadcast_in_dim %113, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x32xf32>
    %114 = stablehlo.multiply %109, %s273 : tensor<128x32xf32>
    %115 = stablehlo.reshape %21 : (tensor<32xf32>) -> tensor<1x32xf32>
    %116 = stablehlo.broadcast_in_dim %115, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<128x32xf32>
    %117 = stablehlo.multiply %114, %116 : tensor<128x32xf32>
    %118 = stablehlo.dot_general %117, %22, contracting_dims = [1] x [0] : (tensor<128x32xf32>, tensor<32x32xf32>) -> tensor<128x32xf32>
    %119 = stablehlo.dot_general %117, %23, contracting_dims = [1] x [0] : (tensor<128x32xf32>, tensor<32x16xf32>) -> tensor<128x16xf32>
    %120 = stablehlo.dot_general %117, %24, contracting_dims = [1] x [0] : (tensor<128x32xf32>, tensor<32x16xf32>) -> tensor<128x16xf32>
    %121 = stablehlo.reshape %118 : (tensor<128x32xf32>) -> tensor<128x4x8xf32>
    %122 = stablehlo.reshape %119 : (tensor<128x16xf32>) -> tensor<128x2x8xf32>
    %123 = stablehlo.reshape %120 : (tensor<128x16xf32>) -> tensor<128x2x8xf32>
    %124 = stablehlo.multiply %121, %47 : tensor<128x4x8xf32>
    %125 = stablehlo.slice %121 [0:128, 0:4, 4:8] : (tensor<128x4x8xf32>) -> tensor<128x4x4xf32>
    %126 = stablehlo.slice %121 [0:128, 0:4, 0:4] : (tensor<128x4x8xf32>) -> tensor<128x4x4xf32>
    %127 = stablehlo.negate %125 : tensor<128x4x4xf32>
    %128 = stablehlo.concatenate %127, %126, dim = 2 : (tensor<128x4x4xf32>, tensor<128x4x4xf32>) -> tensor<128x4x8xf32>
    %129 = stablehlo.multiply %128, %49 : tensor<128x4x8xf32>
    %130 = stablehlo.add %124, %129 : tensor<128x4x8xf32>
    %131 = stablehlo.multiply %122, %51 : tensor<128x2x8xf32>
    %132 = stablehlo.slice %122 [0:128, 0:2, 4:8] : (tensor<128x2x8xf32>) -> tensor<128x2x4xf32>
    %133 = stablehlo.slice %122 [0:128, 0:2, 0:4] : (tensor<128x2x8xf32>) -> tensor<128x2x4xf32>
    %134 = stablehlo.negate %132 : tensor<128x2x4xf32>
    %135 = stablehlo.concatenate %134, %133, dim = 2 : (tensor<128x2x4xf32>, tensor<128x2x4xf32>) -> tensor<128x2x8xf32>
    %136 = stablehlo.multiply %135, %53 : tensor<128x2x8xf32>
    %137 = stablehlo.add %131, %136 : tensor<128x2x8xf32>
    %s274 = stablehlo.reshape %7 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s275 = "stablehlo.scatter"(%s274, %4, %137) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s276: tensor<f32>, %s277: tensor<f32>):
       stablehlo.return %s277 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<128xi32>, tensor<128x2x8xf32>) -> tensor<260x2x8xf32>
    %138 = stablehlo.reshape %s275 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s278 = stablehlo.reshape %8 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s279 = "stablehlo.scatter"(%s278, %4, %123) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s280: tensor<f32>, %s281: tensor<f32>):
       stablehlo.return %s281 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<128xi32>, tensor<128x2x8xf32>) -> tensor<260x2x8xf32>
    %139 = stablehlo.reshape %s279 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s282 = "stablehlo.gather"(%138, %56) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<128x16xi32>) -> tensor<128x16x4x2x8xf32>
    %s283 = stablehlo.reshape %s282 : (tensor<128x16x4x2x8xf32>) -> tensor<128x64x2x8xf32>
    %s284 = "stablehlo.gather"(%139, %56) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<128x16xi32>) -> tensor<128x16x4x2x8xf32>
    %s285 = stablehlo.reshape %s284 : (tensor<128x16x4x2x8xf32>) -> tensor<128x64x2x8xf32>
    %s286 = stablehlo.reshape %130 : (tensor<128x4x8xf32>) -> tensor<128x2x2x8xf32>
    %s287 = stablehlo.dot_general %s286, %s283, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [3], precision = [HIGHEST, HIGHEST] : (tensor<128x2x2x8xf32>, tensor<128x64x2x8xf32>) -> tensor<128x2x2x64xf32>
    %s288 = stablehlo.constant dense<0.35355338> : tensor<f32>
    %s289 = stablehlo.broadcast_in_dim %s288, dims = [] : (tensor<f32>) -> tensor<128x2x2x64xf32>
    %s290 = stablehlo.multiply %s287, %s289 : tensor<128x2x2x64xf32>
    %s291 = stablehlo.iota dim = 3 : tensor<128x2x2x64xi32>
    %s292 = stablehlo.broadcast_in_dim %58, dims = [0] : (tensor<128xi32>) -> tensor<128x2x2x64xi32>
    %s294 = stablehlo.compare LT, %s291, %s292, SIGNED : (tensor<128x2x2x64xi32>, tensor<128x2x2x64xi32>) -> tensor<128x2x2x64xi1>
    %s295 = stablehlo.constant dense<8> : tensor<i32>
    %s296 = stablehlo.broadcast_in_dim %s295, dims = [] : (tensor<i32>) -> tensor<128x2x2x64xi32>
    %s297 = stablehlo.subtract %s292, %s296 : tensor<128x2x2x64xi32>
    %s298 = stablehlo.compare GE, %s291, %s297, SIGNED : (tensor<128x2x2x64xi32>, tensor<128x2x2x64xi32>) -> tensor<128x2x2x64xi1>
    %s293 = stablehlo.and %s294, %s298 : tensor<128x2x2x64xi1>
    %s299 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s300 = stablehlo.broadcast_in_dim %s299, dims = [] : (tensor<f32>) -> tensor<128x2x2x64xf32>
    %s301 = stablehlo.select %s293, %s290, %s300 : tensor<128x2x2x64xi1>, tensor<128x2x2x64xf32>
    %s302 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s303 = stablehlo.reduce(%s301 init: %s302) applies stablehlo.maximum across dimensions = [3] : (tensor<128x2x2x64xf32>, tensor<f32>) -> tensor<128x2x2xf32>
    %s304 = stablehlo.broadcast_in_dim %s303, dims = [0, 1, 2] : (tensor<128x2x2xf32>) -> tensor<128x2x2x64xf32>
    %s305 = stablehlo.subtract %s301, %s304 : tensor<128x2x2x64xf32>
    %s306 = stablehlo.exponential %s305 : tensor<128x2x2x64xf32>
    %s307 = stablehlo.constant dense<0.0> : tensor<f32>
    %s308 = stablehlo.reduce(%s306 init: %s307) applies stablehlo.add across dimensions = [3] : (tensor<128x2x2x64xf32>, tensor<f32>) -> tensor<128x2x2xf32>
    %s309 = stablehlo.broadcast_in_dim %s308, dims = [0, 1, 2] : (tensor<128x2x2xf32>) -> tensor<128x2x2x64xf32>
    %s310 = stablehlo.divide %s306, %s309 : tensor<128x2x2x64xf32>
    %s311 = stablehlo.dot_general %s310, %s285, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [HIGHEST, HIGHEST] : (tensor<128x2x2x64xf32>, tensor<128x64x2x8xf32>) -> tensor<128x2x2x8xf32>
    %140 = stablehlo.reshape %s311 : (tensor<128x2x2x8xf32>) -> tensor<128x4x8xf32>
    %141 = stablehlo.reshape %140 : (tensor<128x4x8xf32>) -> tensor<128x32xf32>
    %142 = stablehlo.dot_general %141, %25, contracting_dims = [1] x [0] : (tensor<128x32xf32>, tensor<32x32xf32>) -> tensor<128x32xf32>
    %143 = stablehlo.add %109, %142 : tensor<128x32xf32>
    %144 = stablehlo.multiply %143, %143 : tensor<128x32xf32>
    %s312 = stablehlo.constant dense<0.0> : tensor<f32>
    %s313 = stablehlo.reduce(%144 init: %s312) applies stablehlo.add across dimensions = [1] : (tensor<128x32xf32>, tensor<f32>) -> tensor<128xf32>
    %s314 = stablehlo.constant dense<32.0> : tensor<128xf32>
    %s315 = stablehlo.divide %s313, %s314 : tensor<128xf32>
    %145 = stablehlo.broadcast_in_dim %s315, dims = [0] : (tensor<128xf32>) -> tensor<128x1xf32>
    %146 = stablehlo.add %145, %61 : tensor<128x1xf32>
    %147 = stablehlo.rsqrt %146 : tensor<128x1xf32>
    %s316 = stablehlo.broadcast_in_dim %147, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x32xf32>
    %148 = stablehlo.multiply %143, %s316 : tensor<128x32xf32>
    %149 = stablehlo.reshape %26 : (tensor<32xf32>) -> tensor<1x32xf32>
    %150 = stablehlo.broadcast_in_dim %149, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<128x32xf32>
    %151 = stablehlo.multiply %148, %150 : tensor<128x32xf32>
    %152 = stablehlo.dot_general %151, %27, contracting_dims = [1] x [0] : (tensor<128x32xf32>, tensor<32x48xf32>) -> tensor<128x48xf32>
    %153 = stablehlo.dot_general %151, %28, contracting_dims = [1] x [0] : (tensor<128x32xf32>, tensor<32x48xf32>) -> tensor<128x48xf32>
    %s317 = stablehlo.logistic %152 : tensor<128x48xf32>
    %154 = stablehlo.multiply %152, %s317 : tensor<128x48xf32>
    %155 = stablehlo.multiply %154, %153 : tensor<128x48xf32>
    %156 = stablehlo.dot_general %155, %29, contracting_dims = [1] x [0] : (tensor<128x48xf32>, tensor<48x32xf32>) -> tensor<128x32xf32>
    %157 = stablehlo.add %143, %156 : tensor<128x32xf32>
    %158 = stablehlo.multiply %157, %157 : tensor<128x32xf32>
    %s318 = stablehlo.constant dense<0.0> : tensor<f32>
    %s319 = stablehlo.reduce(%158 init: %s318) applies stablehlo.add across dimensions = [1] : (tensor<128x32xf32>, tensor<f32>) -> tensor<128xf32>
    %s320 = stablehlo.constant dense<32.0> : tensor<128xf32>
    %s321 = stablehlo.divide %s319, %s320 : tensor<128xf32>
    %159 = stablehlo.broadcast_in_dim %s321, dims = [0] : (tensor<128xf32>) -> tensor<128x1xf32>
    %160 = stablehlo.add %159, %61 : tensor<128x1xf32>
    %161 = stablehlo.rsqrt %160 : tensor<128x1xf32>
    %s322 = stablehlo.broadcast_in_dim %161, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x32xf32>
    %162 = stablehlo.multiply %157, %s322 : tensor<128x32xf32>
    %163 = stablehlo.reshape %30 : (tensor<32xf32>) -> tensor<1x32xf32>
    %164 = stablehlo.broadcast_in_dim %163, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<128x32xf32>
    %165 = stablehlo.multiply %162, %164 : tensor<128x32xf32>
    %166 = stablehlo.dot_general %165, %31, contracting_dims = [1] x [0] : (tensor<128x32xf32>, tensor<32x32xf32>) -> tensor<128x32xf32>
    %167 = stablehlo.dot_general %165, %32, contracting_dims = [1] x [0] : (tensor<128x32xf32>, tensor<32x16xf32>) -> tensor<128x16xf32>
    %168 = stablehlo.dot_general %165, %33, contracting_dims = [1] x [0] : (tensor<128x32xf32>, tensor<32x16xf32>) -> tensor<128x16xf32>
    %169 = stablehlo.reshape %166 : (tensor<128x32xf32>) -> tensor<128x4x8xf32>
    %170 = stablehlo.reshape %167 : (tensor<128x16xf32>) -> tensor<128x2x8xf32>
    %171 = stablehlo.reshape %168 : (tensor<128x16xf32>) -> tensor<128x2x8xf32>
    %172 = stablehlo.multiply %169, %47 : tensor<128x4x8xf32>
    %173 = stablehlo.slice %169 [0:128, 0:4, 4:8] : (tensor<128x4x8xf32>) -> tensor<128x4x4xf32>
    %174 = stablehlo.slice %169 [0:128, 0:4, 0:4] : (tensor<128x4x8xf32>) -> tensor<128x4x4xf32>
    %175 = stablehlo.negate %173 : tensor<128x4x4xf32>
    %176 = stablehlo.concatenate %175, %174, dim = 2 : (tensor<128x4x4xf32>, tensor<128x4x4xf32>) -> tensor<128x4x8xf32>
    %177 = stablehlo.multiply %176, %49 : tensor<128x4x8xf32>
    %178 = stablehlo.add %172, %177 : tensor<128x4x8xf32>
    %179 = stablehlo.multiply %170, %51 : tensor<128x2x8xf32>
    %180 = stablehlo.slice %170 [0:128, 0:2, 4:8] : (tensor<128x2x8xf32>) -> tensor<128x2x4xf32>
    %181 = stablehlo.slice %170 [0:128, 0:2, 0:4] : (tensor<128x2x8xf32>) -> tensor<128x2x4xf32>
    %182 = stablehlo.negate %180 : tensor<128x2x4xf32>
    %183 = stablehlo.concatenate %182, %181, dim = 2 : (tensor<128x2x4xf32>, tensor<128x2x4xf32>) -> tensor<128x2x8xf32>
    %184 = stablehlo.multiply %183, %53 : tensor<128x2x8xf32>
    %185 = stablehlo.add %179, %184 : tensor<128x2x8xf32>
    %s323 = stablehlo.reshape %9 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s324 = "stablehlo.scatter"(%s323, %4, %185) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s325: tensor<f32>, %s326: tensor<f32>):
       stablehlo.return %s326 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<128xi32>, tensor<128x2x8xf32>) -> tensor<260x2x8xf32>
    %186 = stablehlo.reshape %s324 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s327 = stablehlo.reshape %10 : (tensor<65x4x2x8xf32>) -> tensor<260x2x8xf32>
    %s328 = "stablehlo.scatter"(%s327, %4, %171) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s329: tensor<f32>, %s330: tensor<f32>):
       stablehlo.return %s330 : tensor<f32>
     }) : (tensor<260x2x8xf32>, tensor<128xi32>, tensor<128x2x8xf32>) -> tensor<260x2x8xf32>
    %187 = stablehlo.reshape %s328 : (tensor<260x2x8xf32>) -> tensor<65x4x2x8xf32>
    %s331 = "stablehlo.gather"(%186, %56) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<128x16xi32>) -> tensor<128x16x4x2x8xf32>
    %s332 = stablehlo.reshape %s331 : (tensor<128x16x4x2x8xf32>) -> tensor<128x64x2x8xf32>
    %s333 = "stablehlo.gather"(%187, %56) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 4, 2, 8>}> : (tensor<65x4x2x8xf32>, tensor<128x16xi32>) -> tensor<128x16x4x2x8xf32>
    %s334 = stablehlo.reshape %s333 : (tensor<128x16x4x2x8xf32>) -> tensor<128x64x2x8xf32>
    %s335 = stablehlo.reshape %178 : (tensor<128x4x8xf32>) -> tensor<128x2x2x8xf32>
    %s336 = stablehlo.dot_general %s335, %s332, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [3], precision = [HIGHEST, HIGHEST] : (tensor<128x2x2x8xf32>, tensor<128x64x2x8xf32>) -> tensor<128x2x2x64xf32>
    %s337 = stablehlo.constant dense<0.35355338> : tensor<f32>
    %s338 = stablehlo.broadcast_in_dim %s337, dims = [] : (tensor<f32>) -> tensor<128x2x2x64xf32>
    %s339 = stablehlo.multiply %s336, %s338 : tensor<128x2x2x64xf32>
    %s340 = stablehlo.iota dim = 3 : tensor<128x2x2x64xi32>
    %s341 = stablehlo.broadcast_in_dim %58, dims = [0] : (tensor<128xi32>) -> tensor<128x2x2x64xi32>
    %s342 = stablehlo.compare LT, %s340, %s341, SIGNED : (tensor<128x2x2x64xi32>, tensor<128x2x2x64xi32>) -> tensor<128x2x2x64xi1>
    %s343 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s344 = stablehlo.broadcast_in_dim %s343, dims = [] : (tensor<f32>) -> tensor<128x2x2x64xf32>
    %s345 = stablehlo.select %s342, %s339, %s344 : tensor<128x2x2x64xi1>, tensor<128x2x2x64xf32>
    %s346 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s347 = stablehlo.reduce(%s345 init: %s346) applies stablehlo.maximum across dimensions = [3] : (tensor<128x2x2x64xf32>, tensor<f32>) -> tensor<128x2x2xf32>
    %s348 = stablehlo.broadcast_in_dim %s347, dims = [0, 1, 2] : (tensor<128x2x2xf32>) -> tensor<128x2x2x64xf32>
    %s349 = stablehlo.subtract %s345, %s348 : tensor<128x2x2x64xf32>
    %s350 = stablehlo.exponential %s349 : tensor<128x2x2x64xf32>
    %s351 = stablehlo.constant dense<0.0> : tensor<f32>
    %s352 = stablehlo.reduce(%s350 init: %s351) applies stablehlo.add across dimensions = [3] : (tensor<128x2x2x64xf32>, tensor<f32>) -> tensor<128x2x2xf32>
    %s353 = stablehlo.broadcast_in_dim %s352, dims = [0, 1, 2] : (tensor<128x2x2xf32>) -> tensor<128x2x2x64xf32>
    %s354 = stablehlo.divide %s350, %s353 : tensor<128x2x2x64xf32>
    %s355 = stablehlo.dot_general %s354, %s334, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1], precision = [HIGHEST, HIGHEST] : (tensor<128x2x2x64xf32>, tensor<128x64x2x8xf32>) -> tensor<128x2x2x8xf32>
    %188 = stablehlo.reshape %s355 : (tensor<128x2x2x8xf32>) -> tensor<128x4x8xf32>
    %189 = stablehlo.reshape %188 : (tensor<128x4x8xf32>) -> tensor<128x32xf32>
    %190 = stablehlo.dot_general %189, %34, contracting_dims = [1] x [0] : (tensor<128x32xf32>, tensor<32x32xf32>) -> tensor<128x32xf32>
    %191 = stablehlo.add %157, %190 : tensor<128x32xf32>
    %192 = stablehlo.multiply %191, %191 : tensor<128x32xf32>
    %s356 = stablehlo.constant dense<0.0> : tensor<f32>
    %s357 = stablehlo.reduce(%192 init: %s356) applies stablehlo.add across dimensions = [1] : (tensor<128x32xf32>, tensor<f32>) -> tensor<128xf32>
    %s358 = stablehlo.constant dense<32.0> : tensor<128xf32>
    %s359 = stablehlo.divide %s357, %s358 : tensor<128xf32>
    %193 = stablehlo.broadcast_in_dim %s359, dims = [0] : (tensor<128xf32>) -> tensor<128x1xf32>
    %194 = stablehlo.add %193, %61 : tensor<128x1xf32>
    %195 = stablehlo.rsqrt %194 : tensor<128x1xf32>
    %s360 = stablehlo.broadcast_in_dim %195, dims = [0, 1] : (tensor<128x1xf32>) -> tensor<128x32xf32>
    %196 = stablehlo.multiply %191, %s360 : tensor<128x32xf32>
    %197 = stablehlo.reshape %35 : (tensor<32xf32>) -> tensor<1x32xf32>
    %198 = stablehlo.broadcast_in_dim %197, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<128x32xf32>
    %199 = stablehlo.multiply %196, %198 : tensor<128x32xf32>
    %200 = stablehlo.dot_general %199, %36, contracting_dims = [1] x [0] : (tensor<128x32xf32>, tensor<32x48xf32>) -> tensor<128x48xf32>
    %201 = stablehlo.dot_general %199, %37, contracting_dims = [1] x [0] : (tensor<128x32xf32>, tensor<32x48xf32>) -> tensor<128x48xf32>
    %s361 = stablehlo.logistic %200 : tensor<128x48xf32>
    %202 = stablehlo.multiply %200, %s361 : tensor<128x48xf32>
    %203 = stablehlo.multiply %202, %201 : tensor<128x48xf32>
    %204 = stablehlo.dot_general %203, %38, contracting_dims = [1] x [0] : (tensor<128x48xf32>, tensor<48x32xf32>) -> tensor<128x32xf32>
    %205 = stablehlo.add %191, %204 : tensor<128x32xf32>
    %206 = stablehlo.reshape %205 : (tensor<128x32xf32>) -> tensor<2x64x32xf32>
    %207 = stablehlo.slice %206 [0:2, 63:64, 0:32] : (tensor<2x64x32xf32>) -> tensor<2x1x32xf32>
    %208 = stablehlo.reshape %207 : (tensor<2x1x32xf32>) -> tensor<2x32xf32>
    %209 = stablehlo.constant dense<[[1.0E-6], [1.0E-6]]> : tensor<2x1xf32>
    %210 = stablehlo.multiply %208, %208 : tensor<2x32xf32>
    %s362 = stablehlo.constant dense<0.0> : tensor<f32>
    %s363 = stablehlo.reduce(%210 init: %s362) applies stablehlo.add across dimensions = [1] : (tensor<2x32xf32>, tensor<f32>) -> tensor<2xf32>
    %s364 = stablehlo.constant dense<32.0> : tensor<2xf32>
    %s365 = stablehlo.divide %s363, %s364 : tensor<2xf32>
    %211 = stablehlo.broadcast_in_dim %s365, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %212 = stablehlo.add %211, %209 : tensor<2x1xf32>
    %213 = stablehlo.rsqrt %212 : tensor<2x1xf32>
    %s366 = stablehlo.broadcast_in_dim %213, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x32xf32>
    %214 = stablehlo.multiply %208, %s366 : tensor<2x32xf32>
    %215 = stablehlo.reshape %39 : (tensor<32xf32>) -> tensor<1x32xf32>
    %216 = stablehlo.broadcast_in_dim %215, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<2x32xf32>
    %217 = stablehlo.multiply %214, %216 : tensor<2x32xf32>
    %218 = stablehlo.dot_general %217, %40, contracting_dims = [1] x [0] : (tensor<2x32xf32>, tensor<32x64xf32>) -> tensor<2x64xf32>
    %219 = stablehlo.reshape %218 : (tensor<2x64xf32>) -> tensor<2x1x64xf32>
    return %219, %90, %91, %138, %139, %186, %187 : tensor<2x1x64xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>, tensor<65x4x2x8xf32>
  }
}
