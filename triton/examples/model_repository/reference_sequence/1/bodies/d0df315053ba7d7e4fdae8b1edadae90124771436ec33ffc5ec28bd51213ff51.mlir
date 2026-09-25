module {
  func.func @main(%0: tensor<2x1xi32>, %1: tensor<2x1xi32>, %2: tensor<2x1xi32>, %3: tensor<2xi32>, %4: tensor<2xi32>, %5: tensor<6x2x2x2xf32>, %6: tensor<6x2x2x2xf32>) -> (tensor<2x1x11xf32>, tensor<6x2x2x2xf32>, tensor<6x2x2x2xf32>) {
    %7 = stablehlo.constant dense<[[-0.2635445, -0.13072938, 0.004242003, 0.20488322, -0.4494564, -0.13048166, 0.27476293, 0.056188524], [-0.4835068, 0.13924599, -0.24954891, -0.077622235, 0.090690196, 0.3369336, -0.2649241, 0.48084593], [0.36088705, -0.17312455, 0.1826027, 0.031459093, -0.28405803, -0.3954658, -0.42945492, -0.18111932], [-0.37018496, 0.20724636, 0.08288705, -0.120687604, -0.23705202, 0.25090468, 0.4307009, -0.32272595], [-0.16584897, 0.4832415, -0.11141509, 0.048047125, -0.033876777, 0.4943393, 0.3750469, 0.21140814], [0.4577301, 0.47499222, -0.34504223, -0.12772477, 0.18616444, -0.33645535, -0.007799566, -0.32610577], [0.09211463, 0.37183136, -0.16936111, -0.48081094, 0.4510659, -0.26915663, 0.3315332, -0.47409946], [-0.116140485, -0.45325327, -0.15447628, -0.3688602, 0.30963463, -0.09212339, -0.35268813, 0.11098063], [0.29427803, 0.433334, -0.48883963, -0.46928447, 0.4975065, 0.3038078, -0.044661045, -0.10859859], [0.17170733, 0.46524984, 0.23364067, -0.01168102, -0.084338546, -0.35397476, 0.40209305, 0.24697125], [0.13627416, 0.078165114, 0.03899026, -0.4477538, -0.07856071, 0.06334549, 0.4463697, -0.16363168]]> : tensor<11x8xf32>
    %8 = "stablehlo.gather"(%7, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 8>}> : (tensor<11x8xf32>, tensor<2x1xi32>) -> tensor<2x1x8xf32>
    %9 = stablehlo.reshape %8 : (tensor<2x1x8xf32>) -> tensor<2x8xf32>
    %10 = stablehlo.constant dense<[[-0.26315695, -0.04002559, -0.3110497, 0.24189961, 0.21700984, -0.38277084, -0.29420006, -0.027658522], [0.025208354, 0.1966945, 0.20417678, -0.3928101, 0.067551434, 0.28875124, -0.03659743, -0.053058863], [-0.4859587, -0.15792423, 0.48118693, -0.08176017, 0.40702897, 0.16736305, 0.24077779, -0.10940993], [-0.25278723, -0.35094255, -0.31778592, -0.35361242, -0.427199, -0.13178313, 0.018304646, -0.16760367], [-0.20253658, 0.089634, 0.2859589, -0.029203832, -0.17730284, 0.31634068, 0.2734937, 0.35792547], [0.1902315, 0.3646205, 0.21454239, 0.42884398, -0.20990348, -0.3373466, 0.38918978, 0.41584122], [0.3899812, -0.27408135, 0.022959173, 0.37901515, -0.49849463, -0.48751938, 0.124275506, -0.06295395], [-0.15096933, 0.11128211, 0.15103048, 0.27629066, -0.043364227, 0.46657056, -0.353126, 0.26632184]]> : tensor<8x8xf32>
    %11 = stablehlo.dot_general %9, %10, contracting_dims = [1] x [0] : (tensor<2x8xf32>, tensor<8x8xf32>) -> tensor<2x8xf32>
    %12 = stablehlo.reshape %11 : (tensor<2x8xf32>) -> tensor<2x4x2xf32>
    %13 = stablehlo.constant dense<[[-0.2627694, 0.050678194, 0.37365854, 0.27891594], [-0.11652386, 0.36494005, 0.136837, -0.11150551], [-0.46607655, 0.254143, -0.34209758, 0.29200208], [0.044412673, 0.24056888, 0.19172925, 0.4130364], [-0.3328045, -0.14272392, -0.22022891, -0.19497949], [0.09811592, -0.2698081, -0.088989556, -0.037700474], [-0.1353895, 0.09086859, 0.28154105, 0.41346276], [0.382654, 0.48552912, -0.3940916, -0.012481332]]> : tensor<8x4xf32>
    %14 = stablehlo.dot_general %9, %13, contracting_dims = [1] x [0] : (tensor<2x8xf32>, tensor<8x4xf32>) -> tensor<2x4xf32>
    %15 = stablehlo.reshape %14 : (tensor<2x4xf32>) -> tensor<2x2x2xf32>
    %16 = stablehlo.constant dense<[[-0.26238185, 0.14138192, 0.058366835, 0.31593233], [-0.45005763, 0.11265093, -0.432126, -0.19535255], [0.0426386, 0.3115915, 0.111628056, -0.02318573], [0.021273851, 0.19238651, 0.42005593, -0.120868385], [-0.1796503, -0.1275236, 0.07835531, -0.30819875], [-0.21079707, 0.29302073, -0.41875684, 0.03400892], [-0.017991781, -0.46732032, -0.11913192, 0.18053794], [0.19250709, 0.10284132, 0.1935122, 0.14264095]]> : tensor<8x4xf32>
    %17 = stablehlo.dot_general %9, %16, contracting_dims = [1] x [0] : (tensor<2x8xf32>, tensor<8x4xf32>) -> tensor<2x4xf32>
    %18 = stablehlo.reshape %17 : (tensor<2x4xf32>) -> tensor<2x2x2xf32>
    %s26 = stablehlo.reshape %5 : (tensor<6x2x2x2xf32>) -> tensor<12x2x2xf32>
    %s27 = "stablehlo.scatter"(%s26, %4, %15) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s28: tensor<f32>, %s29: tensor<f32>):
       stablehlo.return %s29 : tensor<f32>
     }) : (tensor<12x2x2xf32>, tensor<2xi32>, tensor<2x2x2xf32>) -> tensor<12x2x2xf32>
    %19 = stablehlo.reshape %s27 : (tensor<12x2x2xf32>) -> tensor<6x2x2x2xf32>
    %s30 = stablehlo.reshape %6 : (tensor<6x2x2x2xf32>) -> tensor<12x2x2xf32>
    %s31 = "stablehlo.scatter"(%s30, %4, %18) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
     ^bb0(%s32: tensor<f32>, %s33: tensor<f32>):
       stablehlo.return %s33 : tensor<f32>
     }) : (tensor<12x2x2xf32>, tensor<2xi32>, tensor<2x2x2xf32>) -> tensor<12x2x2xf32>
    %20 = stablehlo.reshape %s31 : (tensor<12x2x2xf32>) -> tensor<6x2x2x2xf32>
    %s34 = "stablehlo.gather"(%19, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 2, 2, 2>}> : (tensor<6x2x2x2xf32>, tensor<2x1xi32>) -> tensor<2x1x2x2x2xf32>
    %s35 = stablehlo.reshape %s34 : (tensor<2x1x2x2x2xf32>) -> tensor<2x2x2x2xf32>
    %s36 = "stablehlo.gather"(%20, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3, 4], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 2, 2, 2>}> : (tensor<6x2x2x2xf32>, tensor<2x1xi32>) -> tensor<2x1x2x2x2xf32>
    %s37 = stablehlo.reshape %s36 : (tensor<2x1x2x2x2xf32>) -> tensor<2x2x2x2xf32>
    %s38 = stablehlo.reshape %12 : (tensor<2x4x2xf32>) -> tensor<2x2x2x2xf32>
    %s39 = stablehlo.dot_general %s38, %s35, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [3] : (tensor<2x2x2x2xf32>, tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32>
    %s40 = stablehlo.constant dense<0.70710677> : tensor<f32>
    %s41 = stablehlo.broadcast_in_dim %s40, dims = [] : (tensor<f32>) -> tensor<2x2x2x2xf32>
    %s42 = stablehlo.multiply %s39, %s41 : tensor<2x2x2x2xf32>
    %s43 = stablehlo.iota dim = 3 : tensor<2x2x2x2xi32>
    %s44 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<2xi32>) -> tensor<2x2x2x2xi32>
    %s45 = stablehlo.compare LT, %s43, %s44, SIGNED : (tensor<2x2x2x2xi32>, tensor<2x2x2x2xi32>) -> tensor<2x2x2x2xi1>
    %s46 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s47 = stablehlo.broadcast_in_dim %s46, dims = [] : (tensor<f32>) -> tensor<2x2x2x2xf32>
    %s48 = stablehlo.select %s45, %s42, %s47 : tensor<2x2x2x2xi1>, tensor<2x2x2x2xf32>
    %s49 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %s50 = stablehlo.reduce(%s48 init: %s49) applies stablehlo.maximum across dimensions = [3] : (tensor<2x2x2x2xf32>, tensor<f32>) -> tensor<2x2x2xf32>
    %s51 = stablehlo.broadcast_in_dim %s50, dims = [0, 1, 2] : (tensor<2x2x2xf32>) -> tensor<2x2x2x2xf32>
    %s52 = stablehlo.subtract %s48, %s51 : tensor<2x2x2x2xf32>
    %s53 = stablehlo.exponential %s52 : tensor<2x2x2x2xf32>
    %s54 = stablehlo.constant dense<0.0> : tensor<f32>
    %s55 = stablehlo.reduce(%s53 init: %s54) applies stablehlo.add across dimensions = [3] : (tensor<2x2x2x2xf32>, tensor<f32>) -> tensor<2x2x2xf32>
    %s56 = stablehlo.broadcast_in_dim %s55, dims = [0, 1, 2] : (tensor<2x2x2xf32>) -> tensor<2x2x2x2xf32>
    %s57 = stablehlo.divide %s53, %s56 : tensor<2x2x2x2xf32>
    %s58 = stablehlo.dot_general %s57, %s37, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1] : (tensor<2x2x2x2xf32>, tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32>
    %21 = stablehlo.reshape %s58 : (tensor<2x2x2x2xf32>) -> tensor<2x4x2xf32>
    %22 = stablehlo.reshape %21 : (tensor<2x4x2xf32>) -> tensor<2x8xf32>
    %23 = stablehlo.constant dense<[[-0.2619943, 0.2320857, -0.25692493, 0.35294873, 0.21640861, -0.13963819, -0.0010889173, -0.2791996, -0.44864625, 0.36904, -0.4346463], [-0.3383736, -0.0018649101, 0.14420414, -0.3516174, 0.34522682, -0.026496112, -0.112323284, 0.37693954, -0.421418, 0.48028994, -0.14415044], [0.2514758, 0.105718315, 0.099405944, -0.025509238, 0.48019505, -0.05238688, 0.0023601055, -0.2798465, -0.21888405, 0.2977633, -0.3125993], [-0.09118867, 0.4780808, -0.26095665, 0.39241916, -0.21765506, -0.031165957, -0.20252258, 0.38773566, 0.03350544, -0.10670382, 0.0985502], [-0.39810735, -0.34002054, -0.41984206, -0.35831785, 0.2835809, -0.21181941, -0.40008008, -0.04150653, -0.3471762, -0.14260751, -0.49749756], [0.17048252, -0.25545573, -0.19511163, 0.06755078, 0.21174306, -0.102360845, 0.14265233, -0.35443962, -0.26765454, -0.40731275, 0.066327095], [0.4179482, 0.4480073, -0.4085858, -0.01298666, -0.34580696, -0.04832232, -0.41895926, 0.12654543, 0.29417276, 0.252661, -0.22336113], [0.06686902, 0.4338206, -0.4410959, 0.18462032, 0.394145, -0.4577728, 0.06807357, 0.48048335, -0.17179483, 0.4883132, -0.21695411]]> : tensor<8x11xf32>
    %24 = stablehlo.dot_general %22, %23, contracting_dims = [1] x [0] : (tensor<2x8xf32>, tensor<8x11xf32>) -> tensor<2x11xf32>
    %25 = stablehlo.reshape %24 : (tensor<2x11xf32>) -> tensor<2x1x11xf32>
    return %25, %19, %20 : tensor<2x1x11xf32>, tensor<6x2x2x2xf32>, tensor<6x2x2x2xf32>
  }
}
