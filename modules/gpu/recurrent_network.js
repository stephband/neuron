import { createStorageBuffer, uploadToBuffer, readFromBuffer } from './buffer.js';
import { GpuBidirectionalGru } from './bigru_layer.js';

/**
 * GpuRecurrentNetwork: End-to-End WebGPU Recurrent Neural Network
 * 
 * Executes multi-layer, multi-head recurrent neural networks entirely in GPU VRAM:
 * - Priming conditioning projection
 * - Stacked BiGRU / GRU recurrent layers
 * - Linear output head projections
 * - Softmax probabilities computed via GPU compute shaders
 */
export class GpuRecurrentNetwork {
  constructor(device, gpuOps, {
    inDim,
    layers = [{ hiddenDim: 48, bidirectional: true }],
    priming = null,
    heads = [{ name: 'output', layerIndex: -1, outDim: 1, priorBias: null }],
  }) {
    this.device = device;
    this.gpuOps = gpuOps;
    this.inDim = inDim;
    this.layerConfigs = layers;
    this.primingConfig = priming;
    this.headConfigs = heads;

    // 1. Recurrent Layers
    this.layers = [];
    let currentInDim = inDim;
    for (let l = 0; l < layers.length; l++) {
      const cfg = layers[l];
      const hDim = cfg.hiddenDim || 48;
      const bidi = cfg.bidirectional !== undefined ? cfg.bidirectional : true;
      const layer = new GpuBidirectionalGru(device, gpuOps, currentInDim, hDim, bidi);
      this.layers.push(layer);
      currentInDim = layer.outDim;
    }

    // 2. Priming Projection
    this.priming = null;
    if (priming) {
      const H0 = this.layers[0].hiddenDim;
      const pDim = priming.inDim;
      this.priming = {
        type: priming.type || 'categorical',
        inDim: pDim,
        bufW: createStorageBuffer(device, H0 * pDim * 4, 'gpu_priming_W'),
        bufB: createStorageBuffer(device, H0 * 4, 'gpu_priming_b'),
        cpuW: new Float32Array(H0 * pDim),
        cpuB: new Float32Array(H0),
      };
    }

    // 3. Output Heads
    this.heads = [];
    for (let h = 0; h < heads.length; h++) {
      const hCfg = heads[h];
      const name = hCfg.name || `head_${h}`;
      let layerIdx = hCfg.layerIndex !== undefined ? hCfg.layerIndex : -1;
      if (layerIdx < 0) {
        layerIdx = this.layers.length + layerIdx;
      }
      const layerOutDim = this.layers[layerIdx].outDim;
      const outDim = hCfg.outDim;

      // Prepare bias + prior
      const biasWithPrior = new Float32Array(outDim);
      if (hCfg.priorBias) {
        for (let i = 0; i < outDim; i++) biasWithPrior[i] = hCfg.priorBias[i];
      }

      this.heads.push({
        name,
        layerIndex: layerIdx,
        outDim,
        layerOutDim,
        priorBias: hCfg.priorBias ? new Float32Array(hCfg.priorBias) : null,
        bufW: createStorageBuffer(device, outDim * layerOutDim * 4, `gpu_head_${name}_W`),
        bufB: createStorageBuffer(device, biasWithPrior, `gpu_head_${name}_b`),
      });
    }
  }

  /**
   * Synchronize weights from a trained CPU RecurrentNetwork or SpellerNetwork
   */
  syncFromCpuNetwork(cpuNet) {
    // 1. Priming weights
    if (this.priming && cpuNet.priming) {
      this.priming.cpuW.set(cpuNet.priming.W.data);
      this.priming.cpuB.set(cpuNet.priming.b);
      uploadToBuffer(this.device, this.priming.bufW, cpuNet.priming.W.data);
      uploadToBuffer(this.device, this.priming.bufB, cpuNet.priming.b);
    }

    // 2. Recurrent layer weights
    for (let l = 0; l < this.layers.length; l++) {
      this.layers[l].syncFromCpuLayer(cpuNet.layers[l]);
    }

    // 3. Head weights
    for (let h = 0; h < this.heads.length; h++) {
      const gpuHead = this.heads[h];
      const cpuHead = cpuNet.heads[h];

      uploadToBuffer(this.device, gpuHead.bufW, cpuHead.W.data);

      const combinedB = new Float32Array(cpuHead.b);
      if (gpuHead.priorBias) {
        for (let i = 0; i < combinedB.length; i++) {
          combinedB[i] += gpuHead.priorBias[i];
        }
      }
      uploadToBuffer(this.device, gpuHead.bufB, combinedB);
    }
  }

  /**
   * Load weights from a serialized weight dictionary
   * 
   * @param {Object} exported - { weights: { [name]: Float32Array } }
   */
  loadWeights(exported) {
    const weights = exported.weights || exported;

    // 1. Priming weights
    if (this.priming && weights.priming_W) {
      this.priming.cpuW.set(weights.priming_W);
      this.priming.cpuB.set(weights.priming_b);
      uploadToBuffer(this.device, this.priming.bufW, weights.priming_W);
      uploadToBuffer(this.device, this.priming.bufB, weights.priming_b);
    }

    // 2. Layer weights
    for (let l = 0; l < this.layers.length; l++) {
      const layer = this.layers[l];
      const layerWeights = {
        Wih_fwd: weights[`layer${l}_Wih_fwd`],
        bih_fwd: weights[`layer${l}_bih_fwd`],
        Whh_fwd: weights[`layer${l}_Whh_fwd`],
        bhh_fwd: weights[`layer${l}_bhh_fwd`],
      };
      if (layer.bidirectional) {
        layerWeights.Wih_bwd = weights[`layer${l}_Wih_bwd`];
        layerWeights.bih_bwd = weights[`layer${l}_bih_bwd`];
        layerWeights.Whh_bwd = weights[`layer${l}_Whh_bwd`];
        layerWeights.bhh_bwd = weights[`layer${l}_bhh_bwd`];
      }
      layer.loadWeights(layerWeights);
    }

    // 3. Head weights
    for (const head of this.heads) {
      const wKey = `head_${head.name}_W`;
      const bKey = `head_${head.name}_b`;
      if (weights[wKey]) {
        uploadToBuffer(this.device, head.bufW, weights[wKey]);
      }
      if (weights[bKey]) {
        const combinedB = new Float32Array(weights[bKey]);
        if (head.priorBias) {
          for (let i = 0; i < combinedB.length; i++) {
            combinedB[i] += head.priorBias[i];
          }
        }
        uploadToBuffer(this.device, head.bufB, combinedB);
      }
    }
  }

  /**
   * Forward pass across sequence of length T purely on GPU
   * 
   * @param {GPUBuffer} bufX - Input features buffer [T x inDim]
   * @param {number} T - Sequence length
   * @param {Object} options - Optional controls:
   *   - condition: categorical integer or Float32Array
   *   - h0: GPUBuffer or Float32Array
   * @returns {Object} Head output buffers { [head.name]: { bufLogits, bufProbs } }
   */
  forward(bufX, T, options = {}, encoder = null) {
    const isLocalEncoder = !encoder;
    const cmdEncoder = encoder || this.device.createCommandEncoder({ label: 'gpu_net_forward' });

    // 1. Resolve Priming State h0 on GPU
    let bufH0 = null;
    if (options.condition !== undefined && this.priming) {
      const H0 = this.layers[0].hiddenDim;
      const h0_cpu = new Float32Array(H0);
      const wData = this.priming.cpuW;
      const bData = this.priming.cpuB;
      const pDim = this.priming.inDim;

      if (this.priming.type === 'categorical') {
        const c = options.condition;
        for (let j = 0; j < H0; j++) {
          h0_cpu[j] = Math.tanh(wData[j * pDim + c] + bData[j]);
        }
      } else {
        const v = options.condition;
        for (let j = 0; j < H0; j++) {
          let sum = bData[j];
          for (let k = 0; k < pDim; k++) sum += wData[j * pDim + k] * v[k];
          h0_cpu[j] = Math.tanh(sum);
        }
      }
      bufH0 = createStorageBuffer(this.device, h0_cpu, 'gpu_priming_h0');
    } else if (options.bufH0) {
      bufH0 = options.bufH0;
    } else if (options.h0 instanceof Float32Array) {
      bufH0 = createStorageBuffer(this.device, options.h0, 'gpu_h0');
    }

    // 2. Propagate through recurrent layers
    const layerOutputs = [];
    let currentBuf = bufX;

    for (let l = 0; l < this.layers.length; l++) {
      const layerH0 = (l === 0) ? bufH0 : null;
      const bufH_l = this.layers[l].forward(currentBuf, T, layerH0, cmdEncoder);
      layerOutputs.push(bufH_l);
      currentBuf = bufH_l;
    }

    // 3. Compute Head Projections on GPU
    const results = {};

    for (const head of this.heads) {
      const bufH_source = layerOutputs[head.layerIndex];
      const inDimHead = head.layerOutDim;
      const outDim = head.outDim;

      const bufZ_mat = createStorageBuffer(this.device, T * outDim * 4, `temp_head_${head.name}_Zmat`);
      const bufLogits = createStorageBuffer(this.device, T * outDim * 4, `head_${head.name}_logits`);

      // GEMM: Z_mat = H_source * W_head^T
      this.gpuOps.dispatchMatmul(bufH_source, head.bufW, bufZ_mat, T, outDim, inDimHead, false, true, cmdEncoder);
      // Bias addition: Logits = Z_mat + b_head (+ prior)
      this.gpuOps.dispatchAddBias(bufZ_mat, head.bufB, bufLogits, T, outDim, cmdEncoder);

      // Softmax on GPU
      const bufProbs = createStorageBuffer(this.device, T * outDim * 4, `head_${head.name}_probs`);
      this.gpuOps.dispatchSoftmax(bufLogits, bufProbs, T, outDim, cmdEncoder);

      results[head.name] = {
        bufLogits,
        bufProbs,
      };
    }

    if (isLocalEncoder) {
      this.device.queue.submit([cmdEncoder.finish()]);
    }

    return results;
  }

  /**
   * Complete inference: runs forward pass on GPU and reads back predictions
   * 
   * @param {Float32Array} X - Input features [T x inDim]
   * @param {number} T - Sequence length
   * @param {Object} options - Options (e.g. condition, h0)
   * @returns {Promise<Object>} Readback probabilities and argmax class predictions
   */
  async predict(X, T, options = {}) {
    const bufX = createStorageBuffer(this.device, X, 'gpu_predict_X');
    const gpuResults = this.forward(bufX, T, options);

    const outputs = {};
    for (const head of this.heads) {
      const outDim = head.outDim;
      const probs = await readFromBuffer(this.device, gpuResults[head.name].bufProbs, T * outDim * 4);

      const preds = new Uint32Array(T);
      for (let t = 0; t < T; t++) {
        const offset = t * outDim;
        let maxP = -Infinity;
        let bestIdx = 0;
        for (let o = 0; o < outDim; o++) {
          if (probs[offset + o] > maxP) {
            maxP = probs[offset + o];
            bestIdx = o;
          }
        }
        preds[t] = bestIdx;
      }

      outputs[head.name] = {
        probs,
        preds,
      };
    }

    return outputs;
  }
}
