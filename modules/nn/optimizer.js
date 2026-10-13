import { Matrix2D } from '../matrix.js';

/**
 * Stochastic Gradient Descent (SGD) with optional Momentum
 */
export class SGD {
  constructor(learningRate = 0.01, momentum = 0.0) {
    this.lr = learningRate;
    this.momentum = momentum;
    this.velocities = new Map();
  }

  step(layers) {
    for (const layer of layers) {
      if (!layer.weights) continue; // Skip activation layers

      // Initialize momentum buffers if needed
      if (this.momentum > 0 && !this.velocities.has(layer)) {
        this.velocities.set(layer, {
          vW: Matrix2D.zeros(layer.weights.rows, layer.weights.cols),
          vB: Matrix2D.zeros(layer.biases.rows, layer.biases.cols),
        });
      }

      if (this.momentum > 0) {
        const { vW, vB } = this.velocities.get(layer);
        // v = momentum * v + grad
        // param = param - lr * v
        for (let i = 0; i < layer.weights.data.length; i++) {
          vW.data[i] = this.momentum * vW.data[i] + layer.gradW.data[i];
          layer.weights.data[i] -= this.lr * vW.data[i];
        }
        for (let i = 0; i < layer.biases.data.length; i++) {
          vB.data[i] = this.momentum * vB.data[i] + layer.gradB.data[i];
          layer.biases.data[i] -= this.lr * vB.data[i];
        }
      } else {
        // Vanilla SGD: param = param - lr * grad
        for (let i = 0; i < layer.weights.data.length; i++) {
          layer.weights.data[i] -= this.lr * layer.gradW.data[i];
        }
        for (let i = 0; i < layer.biases.data.length; i++) {
          layer.biases.data[i] -= this.lr * layer.gradB.data[i];
        }
      }
    }
  }
}

/**
 * Adam Optimizer (Adaptive Moment Estimation)
 */
export class Adam {
  constructor(learningRate = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
    this.lr = learningRate;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
    this.t = 0; // Timestep
    this.states = new Map();
  }

  step(layers) {
    this.t += 1;
    // Bias correction factors
    const correct1 = 1 - Math.pow(this.beta1, this.t);
    const correct2 = 1 - Math.pow(this.beta2, this.t);

    for (const layer of layers) {
      if (!layer.weights) continue;

      if (!this.states.has(layer)) {
        this.states.set(layer, {
          mW: Matrix2D.zeros(layer.weights.rows, layer.weights.cols),
          vW: Matrix2D.zeros(layer.weights.rows, layer.weights.cols),
          mB: Matrix2D.zeros(layer.biases.rows, layer.biases.cols),
          vB: Matrix2D.zeros(layer.biases.rows, layer.biases.cols),
        });
      }

      const { mW, vW, mB, vB } = this.states.get(layer);

      // 1. Update Weights
      const wData = layer.weights.data;
      const gwData = layer.gradW.data;
      const mwData = mW.data;
      const vwData = vW.data;

      for (let i = 0; i < wData.length; i++) {
        const g = gwData[i];
        mwData[i] = this.beta1 * mwData[i] + (1 - this.beta1) * g;
        vwData[i] = this.beta2 * vwData[i] + (1 - this.beta2) * g * g;

        const mHat = mwData[i] / correct1;
        const vHat = vwData[i] / correct2;

        wData[i] -= (this.lr * mHat) / (Math.sqrt(vHat) + this.eps);
      }

      // 2. Update Biases
      const bData = layer.biases.data;
      const gbData = layer.gradB.data;
      const mbData = mB.data;
      const vbData = vB.data;

      for (let i = 0; i < bData.length; i++) {
        const g = gbData[i];
        mbData[i] = this.beta1 * mbData[i] + (1 - this.beta1) * g;
        vbData[i] = this.beta2 * vbData[i] + (1 - this.beta2) * g * g;

        const mHat = mbData[i] / correct1;
        const vHat = vbData[i] / correct2;

        bData[i] -= (this.lr * mHat) / (Math.sqrt(vHat) + this.eps);
      }
    }
  }
}
