/**
 * WGSL Column Sum Reduction Shader
 * 
 * Computes bias gradient by summing error gradients across the batch dimension:
 * gradBias[c] = sum_{r=0}^{batchSize-1} gradOutput[r, c]
 */

export const REDUCTION_WGSL = /* wgsl */ `
struct ReduceParams {
  batchSize: u32,
  cols: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: ReduceParams;
@group(0) @binding(1) var<storage, read> gradOutput: array<f32>;
@group(0) @binding(2) var<storage, read_write> gradBias: array<f32>;

@compute @workgroup_size(64)
fn sum_columns(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let col = global_id.x;
  if (col >= params.cols) {
    return;
  }

  var sum: f32 = 0.0;
  for (var r: u32 = 0u; r < params.batchSize; r = r + 1u) {
    sum = sum + gradOutput[r * params.cols + col];
  }

  gradBias[col] = sum;
}
`;
