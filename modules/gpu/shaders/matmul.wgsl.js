/**
 * WGSL Tiled Matrix Multiplication (GEMM) Compute Shader
 * 
 * Computes C = A * B, with optional on-the-fly transposition of A or B:
 * - Standard:    C = A * B         (A: [M x K], B: [K x N] -> C: [M x N])
 * - Transpose A: C = A^T * B       (A: [K x M], B: [K x N] -> C: [M x N])
 * - Transpose B: C = A * B^T       (A: [M x K], B: [N x K] -> C: [M x N])
 * 
 * Uses 16x16 workgroup shared memory tiles (var<workgroup>) to reduce
 * global memory bandwidth by ~16x compared to naive GEMM.
 */

export const MATMUL_WGSL = /* wgsl */ `
struct GemmParams {
  M: u32,             // Output rows
  N: u32,             // Output cols
  K: u32,             // Shared inner dimension
  transposeA: u32,    // 0 = Normal, 1 = Transpose A
  transposeB: u32,    // 0 = Normal, 1 = Transpose B
  _pad: u32,
  _pad2: u32,
  _pad3: u32,
};

@group(0) @binding(0) var<uniform> params: GemmParams;
@group(0) @binding(1) var<storage, read> matrixA: array<f32>;
@group(0) @binding(2) var<storage, read> matrixB: array<f32>;
@group(0) @binding(3) var<storage, read_write> matrixC: array<f32>;

const TILE_SIZE: u32 = 16u;

var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn matmul_tiled(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let row = global_id.y;
  let col = global_id.x;
  let localX = local_id.x;
  let localY = local_id.y;

  var acc: f32 = 0.0;
  let numTiles: u32 = (params.K + TILE_SIZE - 1u) / TILE_SIZE;

  for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
    // 1. Cooperatively load tile of Matrix A
    let aK = t * TILE_SIZE + localX;
    if (params.transposeA == 0u) {
      // Normal A: [M x K]
      if (row < params.M && aK < params.K) {
        tileA[localY][localX] = matrixA[row * params.K + aK];
      } else {
        tileA[localY][localX] = 0.0;
      }
    } else {
      // Transposed A: physical memory is [K x M]
      // We want A^T[row, aK] = A[aK, row]
      if (row < params.M && aK < params.K) {
        tileA[localY][localX] = matrixA[aK * params.M + row];
      } else {
        tileA[localY][localX] = 0.0;
      }
    }

    // 2. Cooperatively load tile of Matrix B
    let bK = t * TILE_SIZE + localY;
    if (params.transposeB == 0u) {
      // Normal B: [K x N]
      if (bK < params.K && col < params.N) {
        tileB[localY][localX] = matrixB[bK * params.N + col];
      } else {
        tileB[localY][localX] = 0.0;
      }
    } else {
      // Transposed B: physical memory is [N x K]
      // We want B^T[bK, col] = B[col, bK]
      if (bK < params.K && col < params.N) {
        tileB[localY][localX] = matrixB[col * params.K + bK];
      } else {
        tileB[localY][localX] = 0.0;
      }
    }

    // Synchronize to ensure all threads finished loading into workgroup cache
    workgroupBarrier();

    // 3. Compute dot product for current tile
    for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
      acc = acc + tileA[localY][k] * tileB[k][localX];
    }

    // Synchronize before next tile to avoid overwriting tile cache
    workgroupBarrier();
  }

  // 4. Write result to output matrix C
  if (row < params.M && col < params.N) {
    matrixC[row * params.N + col] = acc;
  }
}
`;
