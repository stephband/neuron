/**
 * WGSL Optimizer Compute Shaders
 * 
 * Supports:
 * - SGD with Momentum
 * - Adam (Adaptive Moment Estimation)
 */

export const OPTIMIZER_WGSL = /* wgsl */ `
struct SgdParams {
  lr: f32,
  momentum: f32,
  count: u32,
  pad: u32,
};

@group(0) @binding(0) var<uniform> sgdParams: SgdParams;
@group(0) @binding(1) var<storage, read> sgdGrads: array<f32>;
@group(0) @binding(2) var<storage, read_write> sgdVelocities: array<f32>;
@group(0) @binding(3) var<storage, read_write> sgdParamsData: array<f32>;

@compute @workgroup_size(64)
fn sgd_step(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let i = global_id.x;
  if (i >= sgdParams.count) {
    return;
  }

  let g = sgdGrads[i];
  let v = sgdParams.momentum * sgdVelocities[i] + g;
  sgdVelocities[i] = v;
  sgdParamsData[i] = sgdParamsData[i] - sgdParams.lr * v;
}

struct AdamParams {
  lr: f32,
  beta1: f32,
  beta2: f32,
  eps: f32,
  bc1: f32,      // 1.0 - pow(beta1, t)
  bc2: f32,      // 1.0 - pow(beta2, t)
  count: u32,
  pad: u32,
};

@group(0) @binding(0) var<uniform> adamParams: AdamParams;
@group(0) @binding(1) var<storage, read> adamGrads: array<f32>;
@group(0) @binding(2) var<storage, read_write> adamM: array<f32>;
@group(0) @binding(3) var<storage, read_write> adamV: array<f32>;
@group(0) @binding(4) var<storage, read_write> adamParamsData: array<f32>;

@compute @workgroup_size(64)
fn adam_step(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let i = global_id.x;
  if (i >= adamParams.count) {
    return;
  }

  let g = adamGrads[i];
  let m_val = adamParams.beta1 * adamM[i] + (1.0 - adamParams.beta1) * g;
  let v_val = adamParams.beta2 * adamV[i] + (1.0 - adamParams.beta2) * g * g;

  adamM[i] = m_val;
  adamV[i] = v_val;

  let m_hat = m_val / adamParams.bc1;
  let v_hat = v_val / adamParams.bc2;

  adamParamsData[i] = adamParamsData[i] - (adamParams.lr * m_hat) / (sqrt(v_hat) + adamParams.eps);
}
`;
