import { createUniformBuffer } from './buffer.js';
import { MATMUL_WGSL } from './shaders/matmul.wgsl.js';
import {
  ADD_BIAS_WGSL,
  RELU_FORWARD_WGSL,
  RELU_BACKWARD_WGSL,
  SOFTMAX_WGSL,
} from './shaders/elementwise.wgsl.js';
import { LOSS_WGSL } from './shaders/loss.wgsl.js';
import { REDUCTION_WGSL } from './shaders/reduction.wgsl.js';
import { OPTIMIZER_WGSL } from './shaders/optimizer.wgsl.js';
import { GRU_SEQUENCE_WGSL } from './shaders/gru.wgsl.js';

/**
 * GPU Compute Operations Engine
 * 
 * Manages compiled compute pipelines and dispatches shaders.
 */
export class GpuOps {
  constructor(device) {
    this.device = device;
    this._initPipelines();
  }

  _initPipelines() {
    // 1. Matrix Multiplication Pipeline
    const matmulModule = this.device.createShaderModule({
      label: 'matmul_module',
      code: MATMUL_WGSL,
    });
    this.matmulPipeline = this.device.createComputePipeline({
      label: 'matmul_pipeline',
      layout: 'auto',
      compute: { module: matmulModule, entryPoint: 'matmul_tiled' },
    });

    // 2. Broadcast Bias Addition Pipeline
    const addBiasModule = this.device.createShaderModule({
      label: 'add_bias_module',
      code: ADD_BIAS_WGSL,
    });
    this.addBiasPipeline = this.device.createComputePipeline({
      label: 'add_bias_pipeline',
      layout: 'auto',
      compute: { module: addBiasModule, entryPoint: 'main' },
    });

    // 3. ReLU Forward Pipeline
    const reluFwdModule = this.device.createShaderModule({
      label: 'relu_forward_module',
      code: RELU_FORWARD_WGSL,
    });
    this.reluForwardPipeline = this.device.createComputePipeline({
      label: 'relu_forward_pipeline',
      layout: 'auto',
      compute: { module: reluFwdModule, entryPoint: 'main' },
    });

    // 4. ReLU Backward Pipeline
    const reluBwdModule = this.device.createShaderModule({
      label: 'relu_backward_module',
      code: RELU_BACKWARD_WGSL,
    });
    this.reluBackwardPipeline = this.device.createComputePipeline({
      label: 'relu_backward_pipeline',
      layout: 'auto',
      compute: { module: reluBwdModule, entryPoint: 'main' },
    });

    // 5. Softmax Cross-Entropy Loss & Gradient Pipeline
    const lossModule = this.device.createShaderModule({
      label: 'loss_module',
      code: LOSS_WGSL,
    });
    this.lossPipeline = this.device.createComputePipeline({
      label: 'loss_pipeline',
      layout: 'auto',
      compute: { module: lossModule, entryPoint: 'softmax_cross_entropy' },
    });

    // 6. Column Reduction Pipeline (for bias gradients)
    const reductionModule = this.device.createShaderModule({
      label: 'reduction_module',
      code: REDUCTION_WGSL,
    });
    this.reduceColsPipeline = this.device.createComputePipeline({
      label: 'reduction_pipeline',
      layout: 'auto',
      compute: { module: reductionModule, entryPoint: 'sum_columns' },
    });

    // 7. Optimizer Pipelines (SGD & Adam)
    const optModule = this.device.createShaderModule({
      label: 'optimizer_module',
      code: OPTIMIZER_WGSL,
    });
    this.sgdPipeline = this.device.createComputePipeline({
      label: 'sgd_pipeline',
      layout: 'auto',
      compute: { module: optModule, entryPoint: 'sgd_step' },
    });
    this.adamPipeline = this.device.createComputePipeline({
      label: 'adam_pipeline',
      layout: 'auto',
      compute: { module: optModule, entryPoint: 'adam_step' },
    });

    // 8. Recurrent GRU Sequence Pipeline
    const gruModule = this.device.createShaderModule({
      label: 'gru_module',
      code: GRU_SEQUENCE_WGSL,
    });
    this.gruSequencePipeline = this.device.createComputePipeline({
      label: 'gru_sequence_pipeline',
      layout: 'auto',
      compute: { module: gruModule, entryPoint: 'gru_sequence' },
    });

    // 9. Softmax Forward Pipeline
    const softmaxModule = this.device.createShaderModule({
      label: 'softmax_module',
      code: SOFTMAX_WGSL,
    });
    this.softmaxPipeline = this.device.createComputePipeline({
      label: 'softmax_pipeline',
      layout: 'auto',
      compute: { module: softmaxModule, entryPoint: 'main' },
    });
  }

  /**
   * Dispatches Tiled GEMM: C = A * B (with optional transpose)
   */
  dispatchMatmul(bufA, bufB, bufC, M, N, K, transposeA = false, transposeB = false, encoder = null) {
    const isLocalEncoder = !encoder;
    const cmdEncoder = encoder || this.device.createCommandEncoder({ label: 'gemm_encoder' });

    const uniformData = new Uint32Array([
      M,
      N,
      K,
      transposeA ? 1 : 0,
      transposeB ? 1 : 0,
      0,
      0,
      0,
    ]);
    const uniformBuffer = createUniformBuffer(this.device, uniformData, 'gemm_uniforms');

    const bindGroup = this.device.createBindGroup({
      layout: this.matmulPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: bufA } },
        { binding: 2, resource: { buffer: bufB } },
        { binding: 3, resource: { buffer: bufC } },
      ],
    });

    const pass = cmdEncoder.beginComputePass({ label: 'gemm_pass' });
    pass.setPipeline(this.matmulPipeline);
    pass.setBindGroup(0, bindGroup);

    const workgroupsX = Math.ceil(N / 16);
    const workgroupsY = Math.ceil(M / 16);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
    pass.end();

    if (isLocalEncoder) {
      this.device.queue.submit([cmdEncoder.finish()]);
    }
  }

  /**
   * Dispatches Broadcast Bias Addition: Z = X + b
   */
  dispatchAddBias(bufX, bufB, bufZ, rows, cols, encoder = null) {
    const isLocalEncoder = !encoder;
    const cmdEncoder = encoder || this.device.createCommandEncoder({ label: 'bias_encoder' });

    const uniformBuffer = createUniformBuffer(this.device, new Uint32Array([rows, cols, 0, 0]));

    const bindGroup = this.device.createBindGroup({
      layout: this.addBiasPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: bufX } },
        { binding: 2, resource: { buffer: bufB } },
        { binding: 3, resource: { buffer: bufZ } },
      ],
    });

    const pass = cmdEncoder.beginComputePass({ label: 'add_bias_pass' });
    pass.setPipeline(this.addBiasPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(cols / 16), Math.ceil(rows / 16), 1);
    pass.end();

    if (isLocalEncoder) {
      this.device.queue.submit([cmdEncoder.finish()]);
    }
  }

  /**
   * Dispatches ReLU Forward: output = max(0, input)
   */
  dispatchReluForward(bufIn, bufOut, rows, cols, encoder = null) {
    const isLocalEncoder = !encoder;
    const cmdEncoder = encoder || this.device.createCommandEncoder({ label: 'relu_forward_encoder' });

    const uniformBuffer = createUniformBuffer(this.device, new Uint32Array([rows, cols, 0, 0]));

    const bindGroup = this.device.createBindGroup({
      layout: this.reluForwardPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: bufIn } },
        { binding: 2, resource: { buffer: bufOut } },
      ],
    });

    const pass = cmdEncoder.beginComputePass({ label: 'relu_forward_pass' });
    pass.setPipeline(this.reluForwardPipeline);
    pass.setBindGroup(0, bindGroup);

    const total = rows * cols;
    pass.dispatchWorkgroups(Math.ceil(total / 64), 1, 1);
    pass.end();

    if (isLocalEncoder) {
      this.device.queue.submit([cmdEncoder.finish()]);
    }
  }

  /**
   * Dispatches ReLU Backward: gradIn = gradOut * (cachedIn > 0 ? 1 : 0)
   */
  dispatchReluBackward(bufGradOut, bufCachedIn, bufGradIn, rows, cols, encoder = null) {
    const isLocalEncoder = !encoder;
    const cmdEncoder = encoder || this.device.createCommandEncoder({ label: 'relu_backward_encoder' });

    const uniformBuffer = createUniformBuffer(this.device, new Uint32Array([rows, cols, 0, 0]));

    const bindGroup = this.device.createBindGroup({
      layout: this.reluBackwardPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: bufGradOut } },
        { binding: 2, resource: { buffer: bufCachedIn } },
        { binding: 3, resource: { buffer: bufGradIn } },
      ],
    });

    const pass = cmdEncoder.beginComputePass({ label: 'relu_backward_pass' });
    pass.setPipeline(this.reluBackwardPipeline);
    pass.setBindGroup(0, bindGroup);

    const total = rows * cols;
    pass.dispatchWorkgroups(Math.ceil(total / 64), 1, 1);
    pass.end();

    if (isLocalEncoder) {
      this.device.queue.submit([cmdEncoder.finish()]);
    }
  }

  /**
   * Dispatches Softmax Cross-Entropy Loss & Logit Gradient
   */
  dispatchSoftmaxLoss(bufLogits, bufTargets, bufGradLogits, bufProbs, bufSampleLoss, batchSize, numClasses, encoder = null) {
    const isLocalEncoder = !encoder;
    const cmdEncoder = encoder || this.device.createCommandEncoder({ label: 'loss_encoder' });

    const uniformBuffer = createUniformBuffer(this.device, new Uint32Array([batchSize, numClasses, 0, 0]));

    const bindGroup = this.device.createBindGroup({
      layout: this.lossPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: bufLogits } },
        { binding: 2, resource: { buffer: bufTargets } },
        { binding: 3, resource: { buffer: bufGradLogits } },
        { binding: 4, resource: { buffer: bufProbs } },
        { binding: 5, resource: { buffer: bufSampleLoss } },
      ],
    });

    const pass = cmdEncoder.beginComputePass({ label: 'loss_pass' });
    pass.setPipeline(this.lossPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(batchSize / 64), 1, 1);
    pass.end();

    if (isLocalEncoder) {
      this.device.queue.submit([cmdEncoder.finish()]);
    }
  }

  /**
   * Dispatches Column Sum Reduction (Bias Gradient)
   */
  dispatchReduceCols(bufGradOut, bufGradBias, batchSize, cols, encoder = null) {
    const isLocalEncoder = !encoder;
    const cmdEncoder = encoder || this.device.createCommandEncoder({ label: 'reduce_cols_encoder' });

    const uniformBuffer = createUniformBuffer(this.device, new Uint32Array([batchSize, cols, 0, 0]));

    const bindGroup = this.device.createBindGroup({
      layout: this.reduceColsPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: bufGradOut } },
        { binding: 2, resource: { buffer: bufGradBias } },
      ],
    });

    const pass = cmdEncoder.beginComputePass({ label: 'reduce_cols_pass' });
    pass.setPipeline(this.reduceColsPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(cols / 64), 1, 1);
    pass.end();

    if (isLocalEncoder) {
      this.device.queue.submit([cmdEncoder.finish()]);
    }
  }

  /**
   * Dispatches SGD with Momentum Parameter Update
   */
  dispatchSgdStep(bufGrads, bufVelocities, bufParamsData, count, lr, momentum, encoder = null) {
    const isLocalEncoder = !encoder;
    const cmdEncoder = encoder || this.device.createCommandEncoder({ label: 'sgd_encoder' });

    const rawBuf = new ArrayBuffer(16);
    const f32 = new Float32Array(rawBuf);
    const u32 = new Uint32Array(rawBuf);
    f32[0] = lr;
    f32[1] = momentum;
    u32[2] = count;
    u32[3] = 0;

    const uniformBuffer = createUniformBuffer(this.device, rawBuf);

    const bindGroup = this.device.createBindGroup({
      layout: this.sgdPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: bufGrads } },
        { binding: 2, resource: { buffer: bufVelocities } },
        { binding: 3, resource: { buffer: bufParamsData } },
      ],
    });

    const pass = cmdEncoder.beginComputePass({ label: 'sgd_pass' });
    pass.setPipeline(this.sgdPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(count / 64), 1, 1);
    pass.end();

    if (isLocalEncoder) {
      this.device.queue.submit([cmdEncoder.finish()]);
    }
  }

  /**
   * Dispatches Adam Parameter Update
   */
  dispatchAdamStep(bufGrads, bufM, bufV, bufParamsData, count, lr, beta1, beta2, eps, stepCount, encoder = null) {
    const isLocalEncoder = !encoder;
    const cmdEncoder = encoder || this.device.createCommandEncoder({ label: 'adam_encoder' });

    const bc1 = 1.0 - Math.pow(beta1, stepCount);
    const bc2 = 1.0 - Math.pow(beta2, stepCount);

    const rawBuf = new ArrayBuffer(32);
    const f32 = new Float32Array(rawBuf);
    const u32 = new Uint32Array(rawBuf);
    f32[0] = lr;
    f32[1] = beta1;
    f32[2] = beta2;
    f32[3] = eps;
    f32[4] = bc1;
    f32[5] = bc2;
    u32[6] = count;
    u32[7] = 0;

    const uniformBuffer = createUniformBuffer(this.device, rawBuf);

    const bindGroup = this.device.createBindGroup({
      layout: this.adamPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: bufGrads } },
        { binding: 2, resource: { buffer: bufM } },
        { binding: 3, resource: { buffer: bufV } },
        { binding: 4, resource: { buffer: bufParamsData } },
      ],
    });

    const pass = cmdEncoder.beginComputePass({ label: 'adam_pass' });
    pass.setPipeline(this.adamPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(count / 64), 1, 1);
    pass.end();

    if (isLocalEncoder) {
      this.device.queue.submit([cmdEncoder.finish()]);
    }
  }

  /**
   * Dispatches Recurrent GRU Sequence compute shader (Forward or Backward)
   * 
   * @param {GPUBuffer} bufU - Pre-projected input sequence [T x 3*H]
   * @param {GPUBuffer} bufWhh - Recurrent weight matrix [3*H x H]
   * @param {GPUBuffer} bufBhh - Recurrent bias vector [3*H]
   * @param {GPUBuffer|null} bufH0 - Optional initial state [H]
   * @param {GPUBuffer} bufOutH - Output sequence buffer [T x outD]
   * @param {number} T - Sequence length
   * @param {number} H - Hidden dimension
   * @param {number} outD - Output stride (H or 2*H)
   * @param {number} outOffset - Output column offset (0 for forward, H for backward)
   * @param {number} direction - 0 = Forward (0 -> T-1), 1 = Backward (T-1 -> 0)
   * @param {GPUCommandEncoder|null} encoder - Optional existing command encoder
   */
  dispatchGruSequence(
    bufU,
    bufWhh,
    bufBhh,
    bufH0,
    bufOutH,
    T,
    H,
    outD,
    outOffset = 0,
    direction = 0,
    encoder = null
  ) {
    const isLocalEncoder = !encoder;
    const cmdEncoder = encoder || this.device.createCommandEncoder({ label: 'gru_sequence_encoder' });

    const hasH0 = bufH0 ? 1 : 0;
    const uniformBuffer = createUniformBuffer(
      this.device,
      new Uint32Array([T, H, outD, outOffset, direction, hasH0, 0, 0]),
      'gru_uniforms'
    );

    const h0Buffer = bufH0 || bufBhh; // Placeholder buffer when hasH0 == 0

    const bindGroup = this.device.createBindGroup({
      layout: this.gruSequencePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: bufU } },
        { binding: 2, resource: { buffer: bufWhh } },
        { binding: 3, resource: { buffer: bufBhh } },
        { binding: 4, resource: { buffer: h0Buffer } },
        { binding: 5, resource: { buffer: bufOutH } },
      ],
    });

    const pass = cmdEncoder.beginComputePass({
      label: `gru_${direction === 0 ? 'fwd' : 'bwd'}_pass`,
    });
    pass.setPipeline(this.gruSequencePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1, 1, 1);
    pass.end();

    if (isLocalEncoder) {
      this.device.queue.submit([cmdEncoder.finish()]);
    }
  }

  /**
   * Dispatches Softmax Forward: probabilities = softmax(logits)
   * 
   * @param {GPUBuffer} bufLogits - Input logits [rows x cols]
   * @param {GPUBuffer} bufProbs - Output probabilities [rows x cols]
   * @param {number} rows - Number of rows (batchSize / T)
   * @param {number} cols - Number of classes (numClasses)
   * @param {GPUCommandEncoder|null} encoder - Optional existing command encoder
   */
  dispatchSoftmax(bufLogits, bufProbs, rows, cols, encoder = null) {
    const isLocalEncoder = !encoder;
    const cmdEncoder = encoder || this.device.createCommandEncoder({ label: 'softmax_encoder' });

    const uniformBuffer = createUniformBuffer(this.device, new Uint32Array([rows, cols, 0, 0]));

    const bindGroup = this.device.createBindGroup({
      layout: this.softmaxPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: bufLogits } },
        { binding: 2, resource: { buffer: bufProbs } },
      ],
    });

    const pass = cmdEncoder.beginComputePass({ label: 'softmax_pass' });
    pass.setPipeline(this.softmaxPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(rows / 64), 1, 1);
    pass.end();

    if (isLocalEncoder) {
      this.device.queue.submit([cmdEncoder.finish()]);
    }
  }
}
