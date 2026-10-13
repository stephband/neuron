/**
 * WGSL Softmax Cross-Entropy Loss & Gradient Compute Shader
 * 
 * Computes:
 * 1. Numerically stable Softmax probabilities: P = softmax(logits)
 * 2. Cross-entropy loss per sample: L_i = -sum(Y_i * log(P_i))
 * 3. Logit gradient for backpropagation: dZ = (P - Y) / batchSize
 */

export const LOSS_WGSL = /* wgsl */ `
struct LossParams {
  batchSize: u32,
  numClasses: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: LossParams;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<f32>;
@group(0) @binding(3) var<storage, read_write> gradLogits: array<f32>;
@group(0) @binding(4) var<storage, read_write> probabilities: array<f32>;
@group(0) @binding(5) var<storage, read_write> sampleLosses: array<f32>;

@compute @workgroup_size(64)
fn softmax_cross_entropy(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let b = global_id.x;
  if (b >= params.batchSize) {
    return;
  }

  let rowOffset = b * params.numClasses;
  let K = params.numClasses;

  // 1. Numerically stable max subtraction
  var maxVal: f32 = -1e30;
  for (var c: u32 = 0u; c < K; c = c + 1u) {
    let v = logits[rowOffset + c];
    if (v > maxVal) {
      maxVal = v;
    }
  }

  // 2. Compute sum of exp
  var sumExp: f32 = 0.0;
  for (var c: u32 = 0u; c < K; c = c + 1u) {
    let expVal = exp(logits[rowOffset + c] - maxVal);
    probabilities[rowOffset + c] = expVal;
    sumExp = sumExp + expVal;
  }

  // 3. Normalize probabilities, compute per-sample loss and logit gradient
  var sampleLoss: f32 = 0.0;
  let invBatch: f32 = 1.0 / f32(params.batchSize);

  for (var c: u32 = 0u; c < K; c = c + 1u) {
    let prob = probabilities[rowOffset + c] / sumExp;
    probabilities[rowOffset + c] = prob;

    let targetVal = targets[rowOffset + c];
    if (targetVal > 0.0) {
      sampleLoss = sampleLoss - targetVal * log(max(1e-7, prob));
    }

    // Analytic gradient: (Y_hat - Y) / batchSize
    gradLogits[rowOffset + c] = (prob - targetVal) * invBatch;
  }

  sampleLosses[b] = sampleLoss;
}
`;
