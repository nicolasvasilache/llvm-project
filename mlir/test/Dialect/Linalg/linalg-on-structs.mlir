// https://github.com/ggerganov/ggml/blob/abea4b7609c14b837015ab625e3ac36c4708dd03/src/ggml.c#L712
//                            d    m              qh            qs
!ggml_st_q5_1 = !llvm.struct<(f16, f16, vector<4xi8>, vector<16xi8>)>
!ggml_vt_q5_1 = vector<32xf32>

// https://github.com/ggerganov/ggml/blob/abea4b7609c14b837015ab625e3ac36c4708dd03/src/ggml.c#L1991
func.func @ggml_extract_d_q5_1(%S: !ggml_st_q5_1) -> (!ggml_vt_q5_1) {
    %d = llvm.extractvalue %S[0] : !ggml_st_q5_1
    %ed = arith.extf %d : f16 to f32
    %res = vector.broadcast %ed : f32 to !ggml_vt_q5_1
    return %res: !ggml_vt_q5_1
}

// https://github.com/ggerganov/ggml/blob/abea4b7609c14b837015ab625e3ac36c4708dd03/src/ggml.c#L1992
func.func @ggml_extract_m_q5_1(%S: !ggml_st_q5_1) -> (!ggml_vt_q5_1) {
    %m = llvm.extractvalue %S[1] : !ggml_st_q5_1
    %em = arith.extf %m : f16 to f32
    %res = vector.broadcast %em : f32 to !ggml_vt_q5_1
    return %res: !ggml_vt_q5_1
}

// https://github.com/ggerganov/ggml/blob/abea4b7609c14b837015ab625e3ac36c4708dd03/src/ggml.c#L2006
func.func @ggml_extract_qsqh_q5_1(%S: !ggml_st_q5_1) -> (!ggml_vt_q5_1) {
    %qh = llvm.extractvalue %S[2] : !ggml_st_q5_1
    %bqh = vector.bitcast %qh : vector<4xi8> to vector<32xi1>
    %beqh = arith.extui %bqh : vector<32xi1> to vector<32xi8>
    //
    %qs = llvm.extractvalue %S[3] : !ggml_st_q5_1
    %bqs = vector.bitcast %qs : vector<16xi8> to vector<32xi4>
    %beqs = arith.extui %bqs : vector<32xi4> to vector<32xi8>
    // Combine the i4 and i1 vectors into a single i8 vector
    %two = arith.constant dense<2> : vector<32xi8>
    // TODO: i8 MAC if possible, guaranteed to only use 5 bits and zext.
    %scaled = arith.muli %beqs, %two : vector<32xi8>
    %added = arith.addi %scaled, %beqh : vector<32xi8>
    // Cast to floating point.
    // TODO: may need to go through i32 depending on the target (e.g. to hit good lowering paths to f16).
    %res = arith.uitofp %added : vector<32xi8> to !ggml_vt_q5_1
    return %res: !ggml_vt_q5_1
}

// https://github.com/ggerganov/ggml/blob/abea4b7609c14b837015ab625e3ac36c4708dd03/src/ggml.c#L1999
func.func @ggml_dequantize_block_q5_1(%S : !ggml_st_q5_1) -> (!ggml_vt_q5_1) {
    %qsqh = func.call @ggml_extract_qsqh_q5_1(%S) : (!ggml_st_q5_1) -> (!ggml_vt_q5_1)
    %m = func.call @ggml_extract_m_q5_1(%S) : (!ggml_st_q5_1) -> (!ggml_vt_q5_1)
    %d = func.call @ggml_extract_d_q5_1(%S) : (!ggml_st_q5_1) -> (!ggml_vt_q5_1)
    %res = vector.fma %qsqh, %d, %m : !ggml_vt_q5_1
    return %res: !ggml_vt_q5_1
}

// https://github.com/ggerganov/ggml/blob/abea4b7609c14b837015ab625e3ac36c4708dd03/src/ggml.c#L1984
func.func @ggml_dequantize_row_q5_1(%A: tensor<?x!ggml_st_q5_1>, %B: tensor<?x!ggml_vt_q5_1>) -> (tensor<?x!ggml_vt_q5_1>) {

  %res = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], 
      iterator_types = ["parallel"]
    } 
     ins(%A : tensor<?x!ggml_st_q5_1>) 
    outs(%B : tensor<?x!ggml_vt_q5_1>) {
  ^bb0(%S: !ggml_st_q5_1, %out: !ggml_vt_q5_1):
    %res = func.call @ggml_dequantize_block_q5_1(%S) : (!ggml_st_q5_1) -> (!ggml_vt_q5_1)
    linalg.yield %res : !ggml_vt_q5_1
  } -> tensor<?x!ggml_vt_q5_1>

  return %res : tensor<?x!ggml_vt_q5_1>
}


func.func @ggml_quantize_block_q5_1(%min : f32, %max: f32, %x: !ggml_vt_q5_1) -> (!ggml_st_q5_1) {
  %r0 = llvm.mlir.undef : !ggml_st_q5_1

  // const float d  = (max - min) / ((1 << 5) - 1);
  // y[i].d = GGML_FP32_TO_FP16(d);
  %diff = arith.subf %max, %min : f32
  %c31 = arith.constant 31.0 : f32
  %df32 = arith.divf %diff, %c31 : f32
  %d = arith.truncf %df32 : f32 to f16
  %r1 = llvm.insertvalue %d, %r0[0] : !ggml_st_q5_1

  // y[i].m = GGML_FP32_TO_FP16(min);
  %m = arith.truncf %min : f32 to f16
  %r2 = llvm.insertvalue %m, %r1[1] : !ggml_st_q5_1

  // const float id = d ? 1.0f/d : 0.0f;
  // TODO: this is dangerous.
  %c0 = arith.constant 0.0 : f32
  %c1 = arith.constant 1.0 : f32
  %inv = arith.divf %c1, %df32: f32
  %cmpz = arith.cmpf oeq, %df32, %c0: f32
  %id = arith.select %cmpz, %c0, %inv: f32

  // (x - min) * id
  %vid = vector.broadcast %id: f32 to !ggml_vt_q5_1
  %vmin = vector.broadcast %min: f32 to !ggml_vt_q5_1
  %xvmin = arith.subf %x, %vmin: !ggml_vt_q5_1
  %xvminscaled = arith.mulf %xvmin, %vid: !ggml_vt_q5_1

  // uint8_t xi = (uint8_t)(x + 0.5f);
  %vchalf = arith.constant dense<0.5> : !ggml_vt_q5_1
  %xplushalf = arith.addf %xvminscaled, %vchalf : !ggml_vt_q5_1
  %xplushalfi32 = arith.fptoui %xplushalf : !ggml_vt_q5_1 to vector<32xi32>

  // qh |= ((xi0 & 0x10) >> 4) << (j + 0);
  %cfb5 = arith.constant dense<0x00000010> : vector<32xi32>
  %vb5i32 = arith.andi %xplushalfi32, %cfb5 : vector<32xi32>
  %c4 = arith.constant dense<4> : vector<32xi32>
  %vb5i1 = arith.shrui %vb5i32, %c4 : vector<32xi32>
  %vb5 = arith.trunci %vb5i1 : vector<32xi32> to vector<32xi1>
  %bit5 = vector.bitcast %vb5 : vector<32xi1> to vector<4xi8>
  %r3 = llvm.insertvalue %bit5, %r2[2] : !ggml_st_q5_1

  // y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
  %cfb04 = arith.constant dense<0x0000000f> : vector<32xi32>
  %vb04i32 = arith.andi %xplushalfi32, %cfb04 : vector<32xi32>
  %vb04i4 = arith.trunci %vb04i32 : vector<32xi32> to vector<32xi4>
  %bits04 = vector.bitcast %vb04i4 : vector<32xi4> to vector<16xi8>
  %r4 = llvm.insertvalue %bits04, %r3[3] : !ggml_st_q5_1

  return %r4: !ggml_st_q5_1
}


// func.func @ggml_quantize_block_q5_1(%min: f32, %max: f32, %x: !ggml_vt_q5_1) -> (!ggml_st_q5_1) {
//   %r0 = llvm.mlir.undef : !ggml_st_q5_1
//   %r1 = llvm.insertvalue %d, %r0[0] : !ggml_st_q5_1
//   %r2 = llvm.insertvalue %min, %r1[1] : !ggml_st_q5_1

//   %bd = vector.broadcast %id : f32 to !ggml_vt_q5_1
//   %bmin = vector.broadcast %min : f32 to !ggml_vt_q5_1
//   %x
  
//   %res = vector.broadcast %ed : vector<1xf32> to !ggml_vt_q5_1
//   return %res: !ggml_vt_q5_1
// }



// mlir-opt mlir/test/Dialect/Linalg/linalg-on-structs.mlir --inline -test-transform-dialect-interpreter -test-transform-dialect-erase-schedule | \
// mlir-opt -test-lower-to-llvm -canonicalize -cse  | \
// mlir-translate -mlir-to-llvmir | \
// llc -o - -mcpu=skylake-avx512
transform.sequence failures(propagate) {
^bb1(%module: !transform.any_op):
  %m2 = transform.bufferization.one_shot_bufferize layout{IdentityLayoutMap} %module {bufferize_function_boundaries = true}
    : (!transform.any_op) -> !transform.any_op

  %func = transform.structured.match ops{["func.func"]} in %m2 : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %func {
    transform.apply_patterns.vector.rewrite_narrow_types
  } : !transform.any_op
}
