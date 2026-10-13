import { Matrix2D } from './matrix.js';
import { BidirectionalGru } from './rnn/bigru.js';

/**
 * Numerically stable Softmax for a batch of rows [T x C]
 */
function softmax(logits, T, C) {
  const probs = new Float32Array(T * C);
  for (let t = 0; t < T; t++) {
    const rowOffset = t * C;
    let maxVal = -Infinity;
    for (let c = 0; c < C; c++) {
      const v = logits[rowOffset + c];
      if (v > maxVal) maxVal = v;
    }
    let sumExp = 0.0;
    for (let c = 0; c < C; c++) {
      const expVal = Math.exp(logits[rowOffset + c] - maxVal);
      probs[rowOffset + c] = expVal;
      sumExp += expVal;
    }
    const invSum = 1.0 / (sumExp + 1e-9);
    for (let c = 0; c < C; c++) {
      probs[rowOffset + c] *= invSum;
    }
  }
  return probs;
}

/**
 * RecurrentNetwork: Fully Configurable Sequence-to-Sequence RNN Engine
 * 
 * Configurable features:
 * - Configurable input dimensions: arbitrary inDim
 * - Configurable layers: arbitrary depth, hidden dimensions, unidirectional or bidirectional
 * - Configurable primable state: direct h0 state injection or learned condition projection
 * - Configurable output heads: multi-task heads attached to any layer with arbitrary output dimensions
 * - Integrated BPTT, multi-task gradient accumulation, gradient clipping, and Adam optimizer
 */
export class RecurrentNetwork {
  /**
   * @param {Object} options
   * @param {number} options.inDim - Input feature dimension
   * @param {Array<Object>} options.layers - Array of layer configs: [{ hiddenDim, bidirectional }]
   * @param {Object|null} options.priming - Optional condition priming: { inDim, type: 'categorical' | 'continuous' }
   * @param {Array<Object>} options.heads - Array of head configs: [{ name, layerIndex, outDim, lossWeight, priorBias }]
   */
  constructor({
    inDim,
    layers = [{ hiddenDim: 48, bidirectional: true }],
    priming = null,
    heads = [{ name: 'output', layerIndex: -1, outDim: 1, lossWeight: 1.0, priorBias: null }],
  }) {
    if (!inDim || inDim <= 0) {
      throw new Error(`Invalid inDim: ${inDim}`);
    }
    this.inDim = inDim;
    this.layerConfigs = layers;
    this.primingConfig = priming;
    this.headConfigs = heads;

    // 1. Build Recurrent Layers
    this.layers = [];
    let currentInDim = inDim;

    for (let l = 0; l < layers.length; l++) {
      const cfg = layers[l];
      const hiddenDim = cfg.hiddenDim || 48;
      const bidirectional = cfg.bidirectional !== undefined ? cfg.bidirectional : true;
      const layer = new BidirectionalGru(currentInDim, hiddenDim, bidirectional);
      this.layers.push(layer);
      currentInDim = layer.outDim;
    }

    // 2. Build Condition Priming Projection (Optional)
    this.priming = null;
    if (priming) {
      const H0 = this.layers[0].hiddenDim;
      const pDim = priming.inDim;
      const pType = priming.type || 'categorical';

      this.priming = {
        type: pType,
        inDim: pDim,
        W: Matrix2D.randomXavier(H0, pDim),
        b: new Float32Array(H0),
        gradW: Matrix2D.zeros(H0, pDim),
        gradB: new Float32Array(H0),
      };
    }

    // 3. Build Output Classification/Regression Heads
    this.heads = [];
    for (let h = 0; h < heads.length; h++) {
      const hCfg = heads[h];
      const name = hCfg.name || `head_${h}`;
      let layerIdx = hCfg.layerIndex !== undefined ? hCfg.layerIndex : -1;
      if (layerIdx < 0) {
        layerIdx = this.layers.length + layerIdx;
      }
      if (layerIdx < 0 || layerIdx >= this.layers.length) {
        throw new Error(`Head "${name}" layerIndex ${layerIdx} out of range [0, ${this.layers.length - 1}]`);
      }

      const layerOutDim = this.layers[layerIdx].outDim;
      const outDim = hCfg.outDim;
      const lossWeight = hCfg.lossWeight !== undefined ? hCfg.lossWeight : 1.0;
      const priorBias = hCfg.priorBias ? new Float32Array(hCfg.priorBias) : null;

      this.heads.push({
        name,
        layerIndex: layerIdx,
        outDim,
        lossWeight,
        priorBias,
        W: Matrix2D.randomXavier(outDim, layerOutDim),
        b: new Float32Array(outDim),
        gradW: Matrix2D.zeros(outDim, layerOutDim),
        gradB: new Float32Array(outDim),
      });
    }

    // 4. Register all parameters for Adam Optimizer
    this.stepCount = 0;
    this.parameters = [];

    // Priming parameters
    if (this.priming) {
      this.parameters.push(
        { name: 'priming_W', weight: this.priming.W.data, grad: this.priming.gradW.data },
        { name: 'priming_b', weight: this.priming.b, grad: this.priming.gradB },
      );
    }

    // Layer parameters
    for (let l = 0; l < this.layers.length; l++) {
      const layer = this.layers[l];
      this.parameters.push(
        { name: `layer${l}_Wih_fwd`, weight: layer.cellFwd.W_ih.data, grad: layer.cellFwd.grad_W_ih.data },
        { name: `layer${l}_Whh_fwd`, weight: layer.cellFwd.W_hh.data, grad: layer.cellFwd.grad_W_hh.data },
        { name: `layer${l}_bih_fwd`, weight: layer.cellFwd.b_ih, grad: layer.cellFwd.grad_b_ih },
        { name: `layer${l}_bhh_fwd`, weight: layer.cellFwd.b_hh, grad: layer.cellFwd.grad_b_hh },
      );
      if (layer.cellBwd) {
        this.parameters.push(
          { name: `layer${l}_Wih_bwd`, weight: layer.cellBwd.W_ih.data, grad: layer.cellBwd.grad_W_ih.data },
          { name: `layer${l}_Whh_bwd`, weight: layer.cellBwd.W_hh.data, grad: layer.cellBwd.grad_W_hh.data },
          { name: `layer${l}_bih_bwd`, weight: layer.cellBwd.b_ih, grad: layer.cellBwd.grad_b_ih },
          { name: `layer${l}_bhh_bwd`, weight: layer.cellBwd.b_hh, grad: layer.cellBwd.grad_b_hh },
        );
      }
    }

    // Head parameters
    for (const head of this.heads) {
      this.parameters.push(
        { name: `head_${head.name}_W`, weight: head.W.data, grad: head.gradW.data },
        { name: `head_${head.name}_b`, weight: head.b, grad: head.gradB },
      );
    }

    // Allocate Adam moments (m, v)
    for (const p of this.parameters) {
      p.m = new Float32Array(p.weight.length);
      p.v = new Float32Array(p.weight.length);
    }

    // Cache for forward/backward passes
    this.cachedT = 0;
    this.cachedH = [];
    this.cachedHeadOutputs = {};
    this.cachedCondition = null;
    this.cachedH0_fwd = null;
  }

  /**
   * Resets all accumulated gradients to zero
   */
  zeroGrad() {
    for (const p of this.parameters) {
      p.grad.fill(0);
    }
    for (const layer of this.layers) {
      layer.zeroGrad();
    }
  }

  /**
   * Forward pass across sequence of length T
   * 
   * @param {Float32Array} X - Input features [T x inDim]
   * @param {number} T - Sequence length
   * @param {Object} options - Optional execution controls:
   *   - h0: Float32Array (direct priming state for layer 0)
   *   - condition: number (categorical index) or Float32Array (continuous conditioning)
   * @returns {Object} Outputs keyed by head name + intermediate hidden states
   */
  forward(X, T, options = {}) {
    this.cachedT = T;
    const L = this.layers.length;
    this.cachedH = new Array(L);

    // 1. Resolve Priming State h0 for Layer 0
    let h0_layer0 = null;
    this.cachedCondition = null;
    this.cachedH0_fwd = null;

    if (options.condition !== undefined && this.priming) {
      const H0 = this.layers[0].hiddenDim;
      h0_layer0 = new Float32Array(H0);
      const wData = this.priming.W.data;
      const bData = this.priming.b;
      const pDim = this.priming.inDim;

      if (this.priming.type === 'categorical') {
        const c = options.condition;
        this.cachedCondition = c;
        for (let j = 0; j < H0; j++) {
          h0_layer0[j] = Math.tanh(wData[j * pDim + c] + bData[j]);
        }
      } else {
        const v = options.condition;
        this.cachedCondition = v;
        for (let j = 0; j < H0; j++) {
          let sum = bData[j];
          const wRow = j * pDim;
          for (let k = 0; k < pDim; k++) {
            sum += wData[wRow + k] * v[k];
          }
          h0_layer0[j] = Math.tanh(sum);
        }
      }
      this.cachedH0_fwd = h0_layer0;
    } else if (options.h0) {
      h0_layer0 = options.h0;
      this.cachedH0_fwd = h0_layer0;
    }

    // 2. Propagate through recurrent layers
    let currentX = X;
    for (let l = 0; l < L; l++) {
      const layerH0 = (l === 0) ? h0_layer0 : (Array.isArray(options.h0Layers) ? options.h0Layers[l] : null);
      const H_l = this.layers[l].forward(currentX, T, layerH0);
      this.cachedH[l] = H_l;
      currentX = H_l;
    }

    // 3. Compute Head Projections
    const headOutputs = {};

    for (const head of this.heads) {
      const H_source = this.cachedH[head.layerIndex];
      const inDimHead = this.layers[head.layerIndex].outDim;
      const outDim = head.outDim;

      const Z = new Float32Array(T * outDim);
      const wData = head.W.data;
      const bData = head.b;
      const prior = head.priorBias;

      for (let t = 0; t < T; t++) {
        const hOffset = t * inDimHead;
        const zOffset = t * outDim;

        for (let o = 0; o < outDim; o++) {
          let sum = bData[o];
          if (prior) sum += prior[o];

          const wRow = o * inDimHead;
          for (let j = 0; j < inDimHead; j++) {
            sum += wData[wRow + j] * H_source[hOffset + j];
          }
          Z[zOffset + o] = sum;
        }
      }

      const P = softmax(Z, T, outDim);
      const preds = new Uint32Array(T);

      for (let t = 0; t < T; t++) {
        const offset = t * outDim;
        let maxP = -Infinity;
        let bestIdx = 0;
        for (let o = 0; o < outDim; o++) {
          if (P[offset + o] > maxP) {
            maxP = P[offset + o];
            bestIdx = o;
          }
        }
        preds[t] = bestIdx;
      }

      headOutputs[head.name] = {
        logits: Z,
        probs: P,
        preds,
      };
    }

    this.cachedHeadOutputs = headOutputs;
    return {
      ...headOutputs,
      hiddenStates: this.cachedH,
    };
  }

  /**
   * Computes multi-task cross-entropy losses and accuracies
   * 
   * @param {Object} targets - Object mapping head name to target class array
   * @returns {Object} { totalLoss, heads: { [name]: { loss, acc } } }
   */
  computeLoss(targets) {
    const T = this.cachedT;
    let totalLoss = 0.0;
    const headMetrics = {};

    for (const head of this.heads) {
      const targetArray = targets[head.name];
      if (!targetArray) continue;

      const outDim = head.outDim;
      const probs = this.cachedHeadOutputs[head.name].probs;
      const preds = this.cachedHeadOutputs[head.name].preds;

      let lossSum = 0.0;
      let correct = 0;

      for (let t = 0; t < T; t++) {
        const y = targetArray[t];
        const p = Math.max(1e-7, probs[t * outDim + y]);
        lossSum -= Math.log(p);
        if (preds[t] === y) correct++;
      }

      const headLoss = lossSum / T;
      const headAcc = correct / T;
      totalLoss += head.lossWeight * headLoss;

      headMetrics[head.name] = {
        loss: headLoss,
        acc: headAcc,
      };
    }

    return {
      totalLoss,
      heads: headMetrics,
    };
  }

  /**
   * Backward pass (Multi-task BPTT)
   * 
   * @param {Object} targets - Target arrays keyed by head name
   */
  backward(targets) {
    const T = this.cachedT;
    const L = this.layers.length;

    // Allocate gradient buffers for hidden states of each layer: dH[l]
    const dH = new Array(L);
    for (let l = 0; l < L; l++) {
      dH[l] = new Float32Array(T * this.layers[l].outDim);
    }

    // 1. Backpropagate through Heads into their respective layer dH buffers
    for (const head of this.heads) {
      const targetArray = targets[head.name];
      if (!targetArray) continue;

      const lIdx = head.layerIndex;
      const inDimHead = this.layers[lIdx].outDim;
      const outDim = head.outDim;
      const probs = this.cachedHeadOutputs[head.name].probs;
      const H_source = this.cachedH[lIdx];
      const wData = head.W.data;
      const gradW = head.gradW.data;
      const dH_layer = dH[lIdx];
      const weight = head.lossWeight;

      for (let t = 0; t < T; t++) {
        const y = targetArray[t];
        const zOffset = t * outDim;
        const hOffset = t * inDimHead;

        for (let o = 0; o < outDim; o++) {
          const targetVal = o === y ? 1.0 : 0.0;
          const dz = weight * (probs[zOffset + o] - targetVal) / T;

          head.gradB[o] += dz;
          const wRow = o * inDimHead;
          for (let j = 0; j < inDimHead; j++) {
            gradW[wRow + j] += dz * H_source[hOffset + j];
            dH_layer[hOffset + j] += dz * wData[wRow + j];
          }
        }
      }
    }

    // 2. Backpropagate through Recurrent Layers from top (L-1) to bottom (0)
    let dh0_layer0 = null;

    for (let l = L - 1; l >= 0; l--) {
      const { dX: dX_layer, dh0_fwd } = this.layers[l].backward(dH[l]);

      if (l === 0) {
        dh0_layer0 = dh0_fwd;
      } else {
        const prev_dH = dH[l - 1];
        for (let i = 0; i < prev_dH.length; i++) {
          prev_dH[i] += dX_layer[i];
        }
      }
    }

    // 3. Backpropagate into Condition Priming Projection (if applicable)
    if (this.priming && dh0_layer0 && this.cachedH0_fwd) {
      const H0 = this.layers[0].hiddenDim;
      const h0 = this.cachedH0_fwd;
      const pDim = this.priming.inDim;
      const gradW = this.priming.gradW.data;
      const gradB = this.priming.gradB;

      if (this.priming.type === 'categorical') {
        const c = this.cachedCondition;
        for (let j = 0; j < H0; j++) {
          const da = dh0_layer0[j] * (1.0 - h0[j] * h0[j]);
          gradB[j] += da;
          gradW[j * pDim + c] += da;
        }
      } else {
        const v = this.cachedCondition;
        for (let j = 0; j < H0; j++) {
          const da = dh0_layer0[j] * (1.0 - h0[j] * h0[j]);
          gradB[j] += da;
          const wRow = j * pDim;
          for (let k = 0; k < pDim; k++) {
            gradW[wRow + k] += da * v[k];
          }
        }
      }
    }

    return { dh0_layer0 };
  }

  /**
   * Clips gradients across all network parameters by global L2 norm
   */
  clipGradients(maxNorm = 5.0) {
    let sumSq = 0.0;
    for (const p of this.parameters) {
      const g = p.grad;
      for (let i = 0; i < g.length; i++) {
        sumSq += g[i] * g[i];
      }
    }
    const norm = Math.sqrt(sumSq);
    if (norm > maxNorm) {
      const scale = maxNorm / (norm + 1e-7);
      for (const p of this.parameters) {
        const g = p.grad;
        for (let i = 0; i < g.length; i++) {
          g[i] *= scale;
        }
      }
    }
    return norm;
  }

  /**
   * Performs an Adam optimizer step across all network parameters
   */
  adamStep(lr = 0.005, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
    this.stepCount++;
    const bc1 = 1.0 - Math.pow(beta1, this.stepCount);
    const bc2 = 1.0 - Math.pow(beta2, this.stepCount);

    for (const p of this.parameters) {
      const w = p.weight;
      const g = p.grad;
      const m = p.m;
      const v = p.v;

      for (let i = 0; i < w.length; i++) {
        const gradVal = g[i];
        m[i] = beta1 * m[i] + (1.0 - beta1) * gradVal;
        v[i] = beta2 * v[i] + (1.0 - beta2) * gradVal * gradVal;

        const mHat = m[i] / bc1;
        const vHat = v[i] / bc2;

        w[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
      }
    }
  }

  /**
   * Complete training step: forward, compute loss, backward, clip gradients, Adam update
   */
  trainStep(X, T, targets, options = {}, lr = 0.005) {
    this.zeroGrad();
    const forwardResults = this.forward(X, T, options);
    const lossResults = this.computeLoss(targets);
    this.backward(targets);
    const gradNorm = this.clipGradients(5.0);
    this.adamStep(lr);

    return {
      ...lossResults,
      gradNorm,
      ...forwardResults,
    };
  }

  /**
   * Export all network parameters for serialization
   */
  exportWeights() {
    const weights = {};
    for (const p of this.parameters) {
      weights[p.name] = Array.from(p.weight);
    }
    return {
      inDim: this.inDim,
      layerConfigs: this.layerConfigs,
      primingConfig: this.primingConfig,
      headConfigs: this.headConfigs,
      weights,
    };
  }

  /**
   * Load network parameters from an exported object
   */
  loadWeights(exported) {
    const weights = exported.weights || exported;
    for (const p of this.parameters) {
      if (weights[p.name]) {
        p.weight.set(weights[p.name]);
      }
    }
  }
}
