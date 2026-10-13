import { Matrix2D } from '../matrix.js';
import { createStorageBuffer } from '../gpu/buffer.js';

/**
 * GPU Dense (Fully Connected) Layer
 */
export class GpuDenseLayer {
  constructor(device, gpuOps, inDim, outDim) {
    this.device = device;
    this.gpuOps = gpuOps;
    this.inDim = inDim;
    this.outDim = outDim;

    // Parameter buffers (He initialization for weights, zeros for bias)
    const initW = Matrix2D.randomHe(inDim, outDim);
    this.weights = createStorageBuffer(device, initW.data, 'dense_weights');
    this.biases = createStorageBuffer(device, new Float32Array(outDim), 'dense_biases');

    // Gradient buffers
    this.gradW = createStorageBuffer(device, inDim * outDim * 4, 'dense_grad_w');
    this.gradB = createStorageBuffer(device, outDim * 4, 'dense_grad_b');

    // Optimizer momentum / state buffers (for Adam and SGD)
    this.mW = createStorageBuffer(device, inDim * outDim * 4, 'dense_m_w');
    this.vW = createStorageBuffer(device, inDim * outDim * 4, 'dense_v_w');
    this.mB = createStorageBuffer(device, outDim * 4, 'dense_m_b');
    this.vB = createStorageBuffer(device, outDim * 4, 'dense_v_b');

    // Intermediate activations
    this.cachedInput = null;
    this._buffers = new Map();
  }

  _getBuffer(name, byteSize) {
    let buf = this._buffers.get(name);
    if (!buf || buf.size < byteSize) {
      if (buf) buf.destroy();
      buf = createStorageBuffer(this.device, byteSize, `dense_${name}`);
      this._buffers.set(name, buf);
    }
    return buf;
  }

  forward(inputBuf, batchSize, encoder) {
    this.cachedInput = inputBuf;
    const zBuf = this._getBuffer('z', batchSize * this.outDim * 4);
    const outBuf = this._getBuffer('out', batchSize * this.outDim * 4);

    // 1. Z = input * W  ([B x inDim] * [inDim x outDim] = [B x outDim])
    this.gpuOps.dispatchMatmul(
      inputBuf,
      this.weights,
      zBuf,
      batchSize,
      this.outDim,
      this.inDim,
      false,
      false,
      encoder
    );

    // 2. out = Z + b (broadcast over batch)
    this.gpuOps.dispatchAddBias(zBuf, this.biases, outBuf, batchSize, this.outDim, encoder);
    return outBuf;
  }

  backward(gradOutputBuf, batchSize, encoder) {
    // 1. dW = input^T * gradOutput  ([inDim x B] * [B x outDim] = [inDim x outDim])
    this.gpuOps.dispatchMatmul(
      this.cachedInput,
      gradOutputBuf,
      this.gradW,
      this.inDim,
      this.outDim,
      batchSize,
      true,
      false,
      encoder
    );

    // 2. db = sum_rows(gradOutput)  ([1 x outDim])
    this.gpuOps.dispatchReduceCols(gradOutputBuf, this.gradB, batchSize, this.outDim, encoder);

    // 3. dX = gradOutput * W^T  ([B x outDim] * [outDim x inDim] = [B x inDim])
    const gradInputBuf = this._getBuffer('gradInput', batchSize * this.inDim * 4);
    this.gpuOps.dispatchMatmul(
      gradOutputBuf,
      this.weights,
      gradInputBuf,
      batchSize,
      this.inDim,
      this.outDim,
      false,
      true,
      encoder
    );

    return gradInputBuf;
  }
}

/**
 * GPU ReLU Activation Layer
 */
export class GpuReLULayer {
  constructor(device, gpuOps) {
    this.device = device;
    this.gpuOps = gpuOps;
    this.cachedInput = null;
    this._buffers = new Map();
  }

  _getBuffer(name, byteSize) {
    let buf = this._buffers.get(name);
    if (!buf || buf.size < byteSize) {
      if (buf) buf.destroy();
      buf = createStorageBuffer(this.device, byteSize, `relu_${name}`);
      this._buffers.set(name, buf);
    }
    return buf;
  }

  forward(inputBuf, batchSize, dim, encoder) {
    this.cachedInput = inputBuf;
    const outBuf = this._getBuffer('out', batchSize * dim * 4);
    this.gpuOps.dispatchReluForward(inputBuf, outBuf, batchSize, dim, encoder);
    return outBuf;
  }

  backward(gradOutputBuf, batchSize, dim, encoder) {
    const gradInBuf = this._getBuffer('gradIn', batchSize * dim * 4);
    this.gpuOps.dispatchReluBackward(gradOutputBuf, this.cachedInput, gradInBuf, batchSize, dim, encoder);
    return gradInBuf;
  }
}

/**
 * GPU Softmax Cross-Entropy Loss & Gradient
 */
export class GpuSoftmaxCrossEntropy {
  constructor(device, gpuOps) {
    this.device = device;
    this.gpuOps = gpuOps;
    this._buffers = new Map();
  }

  _getBuffer(name, byteSize) {
    let buf = this._buffers.get(name);
    if (!buf || buf.size < byteSize) {
      if (buf) buf.destroy();
      buf = createStorageBuffer(this.device, byteSize, `loss_${name}`);
      this._buffers.set(name, buf);
    }
    return buf;
  }

  forwardAndBackward(logitsBuf, targetsBuf, batchSize, numClasses, encoder) {
    const totalElements = batchSize * numClasses;
    const gradLogits = this._getBuffer('gradLogits', totalElements * 4);
    const probabilities = this._getBuffer('probabilities', totalElements * 4);
    const sampleLosses = this._getBuffer('sampleLosses', batchSize * 4);

    this.gpuOps.dispatchSoftmaxLoss(
      logitsBuf,
      targetsBuf,
      gradLogits,
      probabilities,
      sampleLosses,
      batchSize,
      numClasses,
      encoder
    );

    return {
      gradLogits,
      probabilities,
      sampleLosses,
    };
  }
}

/**
 * GPU Adam Optimizer
 */
export class GpuAdam {
  constructor(device, gpuOps, lr = 0.05, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
    this.device = device;
    this.gpuOps = gpuOps;
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
    this.stepCount = 0;
  }

  step(layers, encoder) {
    this.stepCount++;
    for (const layer of layers) {
      if (!layer.weights) continue; // Skip non-parametric layers

      // Update Weights
      const wCount = layer.inDim * layer.outDim;
      this.gpuOps.dispatchAdamStep(
        layer.gradW,
        layer.mW,
        layer.vW,
        layer.weights,
        wCount,
        this.lr,
        this.beta1,
        this.beta2,
        this.eps,
        this.stepCount,
        encoder
      );

      // Update Biases
      const bCount = layer.outDim;
      this.gpuOps.dispatchAdamStep(
        layer.gradB,
        layer.mB,
        layer.vB,
        layer.biases,
        bCount,
        this.lr,
        this.beta1,
        this.beta2,
        this.eps,
        this.stepCount,
        encoder
      );
    }
  }
}

/**
 * GPU Sequential Neural Network Container
 */
export class GpuSequentialNetwork {
  constructor(device, gpuOps, layers = []) {
    this.device = device;
    this.gpuOps = gpuOps;
    this.layers = layers;
  }

  add(layer) {
    this.layers.push(layer);
    return this;
  }

  /**
   * Executes a full forward pass on GPU
   */
  forward(inputBuf, batchSize, encoder) {
    let current = inputBuf;
    for (const layer of this.layers) {
      if (layer instanceof GpuDenseLayer) {
        current = layer.forward(current, batchSize, encoder);
      } else if (layer instanceof GpuReLULayer) {
        // Find previous layer's outDim
        const prevIdx = this.layers.indexOf(layer) - 1;
        const dim = prevIdx >= 0 ? this.layers[prevIdx].outDim : 0;
        current = layer.forward(current, batchSize, dim, encoder);
      }
    }
    return current;
  }

  /**
   * Executes a full backward pass on GPU
   */
  backward(gradOutputBuf, batchSize, encoder) {
    let current = gradOutputBuf;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      const layer = this.layers[i];
      if (layer instanceof GpuDenseLayer) {
        current = layer.backward(current, batchSize, encoder);
      } else if (layer instanceof GpuReLULayer) {
        const prevIdx = i - 1;
        const dim = prevIdx >= 0 ? this.layers[prevIdx].outDim : 0;
        current = layer.backward(current, batchSize, dim, encoder);
      }
    }
    return current;
  }

  /**
   * GPU-resident training step:
   * Encodes forward, loss, backward, and optimizer updates in a SINGLE command buffer.
   * Zero CPU synchronization stalls.
   */
  trainStep(bufInputs, bufTargets, batchSize, numClasses, lossFn, optimizer) {
    const encoder = this.device.createCommandEncoder({ label: 'train_step_encoder' });

    // 1. Forward Pass
    const logitsBuf = this.forward(bufInputs, batchSize, encoder);

    // 2. Loss & Logit Gradient Pass
    const { gradLogits, probabilities, sampleLosses } = lossFn.forwardAndBackward(
      logitsBuf,
      bufTargets,
      batchSize,
      numClasses,
      encoder
    );

    // 3. Backward Pass
    this.backward(gradLogits, batchSize, encoder);

    // 4. Optimizer Step
    optimizer.step(this.layers, encoder);

    // 5. Submit all GPU passes in one burst
    this.device.queue.submit([encoder.finish()]);

    return {
      probabilities,
      sampleLosses,
    };
  }
}
