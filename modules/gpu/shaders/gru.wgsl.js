/**
 * WGSL Recurrent Sequence Compute Shader (BiGRU)
 * 
 * Computes forward (t = 0 -> T-1) or backward (t = T-1 -> 0) recurrent transitions:
 *   g_h = W_hh * h_{prev} + b_hh
 *   r_t = sigmoid(u_t^r + g_h^r)
 *   z_t = sigmoid(u_t^z + g_h^z)
 *   n_t = tanh(u_t^n + r_t * g_h^n)
 *   h_t = (1 - z_t) * h_{prev} + z_t * n_t
 * 
 * Parallelized across hidden units j in [0, H-1] within workgroup shared memory.
 * Executes the entire temporal sequence in a single GPU dispatch per direction.
 */

export const GRU_SEQUENCE_WGSL = /* wgsl */ `
struct GruParams {
  T: u32,             // Sequence length
  H: u32,             // Hidden dimension H
  outD: u32,          // Output row stride (H or 2*H)
  outOffset: u32,     // Column offset in output (0 for fwd, H for bwd)
  direction: u32,     // 0 = forward (0 -> T-1), 1 = backward (T-1 -> 0)
  hasH0: u32,         // 1 if h0 buffer provided, 0 if initialize to zero
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: GruParams;
@group(0) @binding(1) var<storage, read> U: array<f32>;          // [T x 3*H]
@group(0) @binding(2) var<storage, read> W_hh: array<f32>;       // [3*H x H]
@group(0) @binding(3) var<storage, read> b_hh: array<f32>;       // [3*H]
@group(0) @binding(4) var<storage, read> h0: array<f32>;         // [H] (optional)
@group(0) @binding(5) var<storage, read_write> outH: array<f32>; // [T x outD]

const MAX_H: u32 = 256u;
var<workgroup> h_shared: array<f32, 256>;
var<workgroup> h_next: array<f32, 256>;

fn run_gru_steps(tid: u32, wg_x: u32) {
  let isBidi = params.direction == 2u;
  let dir = select(params.direction, wg_x, isBidi);
  let threeH = 3u * params.H;
  let uStride = select(threeH, 2u * threeH, isBidi);
  let uDirOffset = select(0u, dir * threeH, isBidi);
  let wDirOffset = select(0u, dir * threeH * params.H, isBidi);
  let bDirOffset = select(0u, dir * threeH, isBidi);
  let outColOffset = select(params.outOffset, dir * params.H, isBidi);

  // Initialize shared state h0 (at step 0 of forward pass, or 0.0)
  if (tid < params.H) {
    if (params.hasH0 == 1u && dir == 0u) {
      h_shared[tid] = h0[tid];
    } else {
      h_shared[tid] = 0.0;
    }
  }
  workgroupBarrier();

  // Sequential recurrent execution across all time steps
  for (var s: u32 = 0u; s < params.T; s = s + 1u) {
    var t: u32 = s;
    if (dir == 1u) {
      t = params.T - 1u - s; // Backward: T-1 down to 0
    }

    if (tid < params.H) {
      let uOffset = t * uStride + uDirOffset;

      // 1. Matrix-vector dot products: g_h = W_hh * h_prev + b_hh
      var sum_r: f32 = b_hh[bDirOffset + tid];
      var sum_z: f32 = b_hh[bDirOffset + params.H + tid];
      var sum_n: f32 = b_hh[bDirOffset + 2u * params.H + tid];

      let row_r = wDirOffset + tid * params.H;
      let row_z = wDirOffset + (params.H + tid) * params.H;
      let row_n = wDirOffset + (2u * params.H + tid) * params.H;

      for (var k: u32 = 0u; k < params.H; k = k + 1u) {
        let h_val = h_shared[k];
        sum_r = sum_r + W_hh[row_r + k] * h_val;
        sum_z = sum_z + W_hh[row_z + k] * h_val;
        sum_n = sum_n + W_hh[row_n + k] * h_val;
      }

      // 2. Pre-projected input components
      let u_r = U[uOffset + tid];
      let u_z = U[uOffset + params.H + tid];
      let u_n = U[uOffset + 2u * params.H + tid];

      // 3. Gate activations
      // Sigmoid: 1 / (1 + exp(-clamp(x, -85, 85)))
      let r = 1.0 / (1.0 + exp(-clamp(u_r + sum_r, -85.0, 85.0)));
      let z = 1.0 / (1.0 + exp(-clamp(u_z + sum_z, -85.0, 85.0)));
      let n = tanh(u_n + r * sum_n);

      // 4. Update hidden state
      let h_prev = h_shared[tid];
      let h_new = (1.0 - z) * h_prev + z * n;

      h_next[tid] = h_new;

      // 5. Store hidden state into global sequence buffer
      outH[t * params.outD + outColOffset + tid] = h_new;
    }

    workgroupBarrier();

    if (tid < params.H) {
      h_shared[tid] = h_next[tid];
    }

    workgroupBarrier();
  }
}

@compute @workgroup_size(32)
fn gru_sequence_32(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  run_gru_steps(local_id.x, workgroup_id.x);
}

@compute @workgroup_size(256)
fn gru_sequence(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  run_gru_steps(local_id.x, workgroup_id.x);
}
`;
