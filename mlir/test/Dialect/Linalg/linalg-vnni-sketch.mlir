func.func @pack_ref()->tensor<32x32x16x32x2xbf16>{
  %0 = tensor.empty() : tensor<1024x1024xbf16>
  %1 = tensor.empty() : tensor<32x32x32x32xbf16>
  %2 = tensor.pack %0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1 : tensor<1024x1024xbf16> -> tensor<32x32x32x32xbf16>
  %3 = tensor.empty() : tensor<32x32x16x32x2xbf16>
  %4 = tensor.pack %2 inner_dims_pos = [2] inner_tiles = [2] into %3 : tensor<32x32x32x32xbf16> -> tensor<32x32x16x32x2xbf16>
  return %4:tensor<32x32x16x32x2xbf16>
}

// -----

func.func @pack(%0: tensor<1023x1023xbf16>,
                %1: tensor<32x32x32x32xbf16>,
                %t3: tensor<32x32x16x32x2xbf16>) -> tensor<32x32x16x32x2xbf16> {
  %pad = arith.constant 0.0 : bf16
  // tensor.empty must bufferize inplace or be transformed to 
  // bufferization.alloc_tensor explicitly. 
  // To circumvent this, use a function argument that bufferizes across boundaries.
  // %0 = tensor.empty() : tensor<1023x1023xbf16>
  // %1 = tensor.empty() : tensor<32x32x32x32xbf16>
  // %t3 = tensor.empty() : tensor<32x32x16x32x2xbf16>
  %2 = tensor.pack %0 padding_value(%pad : bf16)
    outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] 
      into %1 
        : tensor<1023x1023xbf16> -> tensor<32x32x32x32xbf16>

  %expand = tensor.expand_shape %2 [[0], [1], [2, 3], [4]]
    : tensor<32x32x32x32xbf16> into tensor<32x32x2x16x32xbf16>

  %res = linalg.transpose ins(%expand : tensor<32x32x2x16x32xbf16>)
                   outs(%t3 : tensor<32x32x16x32x2xbf16>) 
                   permutation = [0, 1, 3, 4, 2]

  return %res : tensor<32x32x16x32x2xbf16>
}

transform.sequence failures(propagate) {
^bb1(%module: !transform.any_op):
  %transp = transform.structured.match ops{["linalg.transpose"]} in %module
    : (!transform.any_op) -> !transform.any_op
  %forall_op, %tiled_transp = transform.structured.tile_to_forall_op %transp tile_sizes [1, 1, 0, 0, 0]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // TODO: fusing the expand_shape generates the following IR.
  // %expanded_1 = tensor.expand_shape %pack [[0], [1], [2, 3], [4]] 
  //   : tensor<32x32x32x32xbf16> into tensor<32x32x2x16x32xbf16>
  // %extracted_slice = tensor.extract_slice %expanded_1[%arg3, %arg4, 0, 0, 0] [1, 1, 2, 16, 32] [1, 1, 1, 1, 1]
  //   : tensor<32x32x2x16x32xbf16> to tensor<1x1x2x16x32xbf16>
  // This requires a new pattern to rewrite extract_slice(expand_shape) into expand_shape(extract_slice).
  %reshape = transform.structured.match ops{["tensor.expand_shape"]} in %module
    : (!transform.any_op) -> !transform.any_op
  %fused_reshape, %forall_op_2 = transform.structured.fuse_into_containing_op %reshape into %forall_op
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  // TODO: fusing the pack will require a pattern to rewrite extract_slice(pack) into pack(extract_slice).
  %pack = transform.structured.match ops{["tensor.pack"]} in %module
    : (!transform.any_op) -> !transform.any_op
  %fused_pack, %forall_op_3 = transform.structured.fuse_into_containing_op %pack into %forall_op_2
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  
  %fused_pack_with_type = transform.cast %fused_pack : !transform.any_op to !transform.op<"tensor.pack"> 
  transform.structured.lower_pack %fused_pack_with_type : (!transform.op<"tensor.pack">)
    -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)

  %func = transform.structured.match ops{["func.func"]} in %module 
    : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %func {
    transform.apply_patterns.canonicalization
  } {apply_cse} : !transform.any_op

  // %m2 = transform.bufferization.one_shot_bufferize 
  //   layout{IdentityLayoutMap} %module 
  //     {bufferize_function_boundaries = true}
  //       : (!transform.any_op) -> !transform.any_op

  // // Post-bufferization cse/canonicalization triggers removal of self-copies.
  // %f2 = transform.structured.match ops{["func.func"]} in %m2 
  //   : (!transform.any_op) -> !transform.any_op
  // transform.apply_patterns to %f2 {
  //   transform.apply_patterns.canonicalization
  // } {apply_cse} : !transform.any_op
}


// ----- 

func.func @pack(%arg0: tensor<1023x1023xbf16>, %arg1: tensor<32x32x32x32xbf16>, %arg2: tensor<32x32x16x32x2xbf16>) -> tensor<32x32x16x32x2xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = scf.forall (%arg3, %arg4) in (32, 32) shared_outs(%arg5 = %arg2) -> (tensor<32x32x16x32x2xbf16>) {
    %padded = tensor.pad %arg0 low[0, 0] high[1, 1] {
    ^bb0(%arg6: index, %arg7: index):
      tensor.yield %cst : bf16
    } : tensor<1023x1023xbf16> to tensor<1024x1024xbf16>
    %expanded = tensor.expand_shape %padded [[0, 1], [2, 3]] : tensor<1024x1024xbf16> into tensor<32x32x32x32xbf16>
    %transposed = linalg.transpose ins(%expanded : tensor<32x32x32x32xbf16>) outs(%arg1 : tensor<32x32x32x32xbf16>) permutation = [2, 0, 1, 3] 
    %extracted_slice = tensor.extract_slice %transposed[%arg3, %arg4, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32x32x32xbf16> to tensor<1x1x32x32xbf16>
    %expanded_0 = tensor.expand_shape %extracted_slice [[0], [1], [2, 3], [4]] : tensor<1x1x32x32xbf16> into tensor<1x1x2x16x32xbf16>
    // %extracted_slice = tensor.extract_slice %expanded_0[%arg3, %arg4, 0, 0, 0] [1, 1, 2, 16, 32] [1, 1, 1, 1, 1] : tensor<32x32x2x16x32xbf16> to tensor<1x1x2x16x32xbf16>
    %extracted_slice_1 = tensor.extract_slice %arg5[%arg3, %arg4, 0, 0, 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1] : tensor<32x32x16x32x2xbf16> to tensor<1x1x16x32x2xbf16>
    %transposed_2 = linalg.transpose ins(%expanded_0 : tensor<1x1x2x16x32xbf16>) outs(%extracted_slice_1 : tensor<1x1x16x32x2xbf16>) permutation = [0, 1, 3, 4, 2] 
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %transposed_2 into %arg5[%arg3, %arg4, 0, 0, 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1] : tensor<1x1x16x32x2xbf16> into tensor<32x32x16x32x2xbf16>
    }
  }
  return %0 : tensor<32x32x16x32x2xbf16>
}

transform.sequence failures(propagate) {
^bb1(%module: !transform.any_op):

  %func = transform.structured.match ops{["func.func"]} in %module 
    : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %func {
    transform.apply_patterns.linalg.bubble_up_extract_slice
    transform.apply_patterns.canonicalization
  } {apply_cse} : !transform.any_op
}


// -----

func.func @pack(%arg0: tensor<1023x1023xbf16>, %arg1: tensor<32x32x32x32xbf16>, %arg2: tensor<32x32x16x32x2xbf16>) -> tensor<32x32x16x32x2xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = scf.forall (%arg3, %arg4) in (32, 32) shared_outs(%arg5 = %arg2) -> (tensor<32x32x16x32x2xbf16>) {
    %padded = tensor.pad %arg0 low[0, 0] high[1, 1] {
    ^bb0(%arg6: index, %arg7: index):
      tensor.yield %cst : bf16
    } : tensor<1023x1023xbf16> to tensor<1024x1024xbf16>
    
    //----------------------------------------------------------------------------------//
    // extract_slice(expand_shape) -> expand_shape(extract_slice) by hand
    //----------------------------------------------------------------------------------//
    %extracted_slice_of_pad = tensor.extract_slice %padded[%arg4, %arg3] [32, 32] [1, 1] 
      : tensor<1024x1024xbf16> to tensor<32x32xbf16>
    %extracted_slice = tensor.expand_shape %extracted_slice_of_pad [[0, 1], [2, 3]] 
      : tensor<32x32xbf16> into tensor<1x32x1x32xbf16>
    // %expanded = tensor.expand_shape %padded [[0, 1], [2, 3]] 
    //   : tensor<1024x1024xbf16> into tensor<32x32x32x32xbf16>
    // %extracted_slice = tensor.extract_slice %expanded[%arg4, 0, %arg3, 0] [1, 32, 1, 32] [1, 1, 1, 1] 
    //   : tensor<32x32x32x32xbf16> to tensor<1x32x1x32xbf16>
    //----------------------------------------------------------------------------------//


    %extracted_slice_0 = tensor.extract_slice %arg1[%arg3, %arg4, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] 
      : tensor<32x32x32x32xbf16> to tensor<1x1x32x32xbf16>
    %transposed = linalg.transpose ins(%extracted_slice : tensor<1x32x1x32xbf16>) 
                                  outs(%extracted_slice_0 : tensor<1x1x32x32xbf16>) permutation = [2, 0, 1, 3] 
    %expanded_1 = tensor.expand_shape %transposed [[0], [1], [2, 3], [4]] 
      : tensor<1x1x32x32xbf16> into tensor<1x1x2x16x32xbf16>
    %extracted_slice_2 = tensor.extract_slice %arg5[%arg3, %arg4, 0, 0, 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1] 
      : tensor<32x32x16x32x2xbf16> to tensor<1x1x16x32x2xbf16>
    %transposed_3 = linalg.transpose ins(%expanded_1 : tensor<1x1x2x16x32xbf16>) 
                                    outs(%extracted_slice_2 : tensor<1x1x16x32x2xbf16>) permutation = [0, 1, 3, 4, 2] 
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %transposed_3 into %arg5[%arg3, %arg4, 0, 0, 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1] 
        : tensor<1x1x16x32x2xbf16> into tensor<32x32x16x32x2xbf16>
    }
  }
  return %0 : tensor<32x32x16x32x2xbf16>
}

transform.sequence failures(propagate) {
^bb1(%module: !transform.any_op):

  %func = transform.structured.match ops{["func.func"]} in %module 
    : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %func {
    transform.apply_patterns.linalg.bubble_up_extract_slice
    transform.apply_patterns.canonicalization
  } {apply_cse} : !transform.any_op


  %m2 = transform.bufferization.one_shot_bufferize 
    layout{IdentityLayoutMap} %module 
      {bufferize_function_boundaries = true}
        : (!transform.any_op) -> !transform.any_op

  // Post-bufferization cse/canonicalization triggers removal of self-copies.
  %f2 = transform.structured.match ops{["func.func"]} in %m2 
    : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %f2 {
    transform.apply_patterns.canonicalization
  } {apply_cse} : !transform.any_op
}
