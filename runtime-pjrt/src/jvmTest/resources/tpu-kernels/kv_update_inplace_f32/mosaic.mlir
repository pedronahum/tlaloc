module {
  func.func @tlaloc_kv_update(%arg0: memref<16x128xf32, #tpu.memory_space<any>>, %arg1: memref<4x128xf32, #tpu.memory_space<vmem>>, %arg2: memref<16x128xf32, #tpu.memory_space<any>>, %arg3: memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>) attributes {dimension_semantics = [], scalar_prefetch = 0 : i64, scratch_operands = 1 : i64, tpu.core_type = #tpu.core_type<tc>} {
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tpu.memref_slice %arg2[%c8_i32, %c0_i32] : memref<16x128xf32, #tpu.memory_space<any>> -> memref<4x128xf32, #tpu.memory_space<any>>
    tpu.enqueue_dma source(%arg1 : memref<4x128xf32, #tpu.memory_space<vmem>>) target(%0 : memref<4x128xf32, #tpu.memory_space<any>>) target_semaphore(%arg3 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>)
    %c8_i32_0 = arith.constant 8 : i32
    %c0_i32_1 = arith.constant 0 : i32
    %1 = tpu.memref_slice %arg2[%c8_i32_0, %c0_i32_1] : memref<16x128xf32, #tpu.memory_space<any>> -> memref<4x128xf32, #tpu.memory_space<any>>
    tpu.wait_dma2 semaphore(%arg3 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>) src(%arg1 : memref<4x128xf32, #tpu.memory_space<vmem>>) dst(%1 : memref<4x128xf32, #tpu.memory_space<any>>)
    return
  }
}
