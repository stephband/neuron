/**
 * WGSL Element-Wise Compute Shaders
 */

export const ADD_BIAS_WGSL = /* wgsl */ `
struct Dimensions {
  rows: u32,
  cols: u32,
};

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> inputX: array<f32>;
@group(0) @binding(2) var<storage, read> inputBias: array<f32>;
@group(0) @binding(3) var<storage, read_write> outputZ: array<f32>;

// Broadcast Bias Addition: Z[r, c] = X[r, c] + b[c]
@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let col = global_id.x;
  let row = global_id.y;

  if (row < dims.rows && col < dims.cols) {
    let index = row * dims.cols + col;
    outputZ[index] = inputX[index] + inputBias[col];
  }
}
`;

export const RELU_FORWARD_WGSL = /* wgsl */ `
struct Dimensions {
  rows: u32,
  cols: u32,
};

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> inputZ: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputA: array<f32>;

// ReLU Forward: A[i] = max(0.0, Z[i])
@compute @workgroup_size(64)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let index = global_id.x;
  let total = dims.rows * dims.cols;

  if (index < total) {
    outputA[index] = max(0.0, inputZ[index]);
  }
}
`;

export const RELU_BACKWARD_WGSL = /* wgsl */ `
struct Dimensions {
  rows: u32,
  cols: u32,
};

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> gradOut: array<f32>;
@group(0) @binding(2) var<storage, read> cachedIn: array<f32>;
@group(0) @binding(3) var<storage, read_write> gradIn: array<f32>;

// ReLU Backward: gradIn[i] = gradOut[i] * (cachedIn[i] > 0.0 ? 1.0 : 0.0)
@compute @workgroup_size(64)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let index = global_id.x;
  let total = dims.rows * dims.cols;

  if (index < total) {
    if (cachedIn[index] > 0.0) {
      gradIn[index] = gradOut[index];
    } else {
      gradIn[index] = 0.0;
    }
  }
}
`;

export const SOFTMAX_WGSL = /* wgsl */ `
struct Dimensions {
  rows: u32,
  cols: u32,
};

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> probabilities: array<f32>;

// Numerically Stable Softmax: P = softmax(logits)
@compute @workgroup_size(64)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let b = global_id.x;
  if (b >= dims.rows) {
    return;
  }
  let rowOffset = b * dims.cols;
  let K = dims.cols;

  // 1. Max subtraction
  var maxVal: f32 = -1e30;
  for (var c: u32 = 0u; c < K; c = c + 1u) {
    let v = logits[rowOffset + c];
    if (v > maxVal) {
      maxVal = v;
    }
  }

  // 2. Sum of exp
  var sumExp: f32 = 0.0;
  for (var c: u32 = 0u; c < K; c = c + 1u) {
    let expVal = exp(logits[rowOffset + c] - maxVal);
    probabilities[rowOffset + c] = expVal;
    sumExp = sumExp + expVal;
  }

  // 3. Normalize
  let invSum: f32 = 1.0 / (sumExp + 1e-9);
  for (var c: u32 = 0u; c < K; c = c + 1u) {
    probabilities[rowOffset + c] = probabilities[rowOffset + c] * invSum;
  }
}
`;
