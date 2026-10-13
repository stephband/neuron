import { Matrix2D } from '../matrix.js';

/**
 * Softmax Cross-Entropy Loss with Numerically Stable Logits
 * 
 * Combines Softmax activation and Categorical Cross-Entropy loss.
 * Pairing them yields the simplified gradient: dZ = (Y_hat - Y) / batchSize
 */
export class SoftmaxCrossEntropy {
  constructor() {
    this.probabilities = null;
    this.targets = null;
  }

  /**
   * Forward pass: computes probabilities and scalar loss
   * logits:  Matrix2D [batch x classes]
   * targets: Matrix2D [batch x classes] (one-hot encoded)
   */
  forward(logits, targets) {
    const batch = logits.rows;
    const classes = logits.cols;
    this.targets = targets;
    this.probabilities = new Matrix2D(batch, classes);

    const lData = logits.data;
    const pData = this.probabilities.data;
    const tData = targets.data;

    let totalLoss = 0;

    for (let i = 0; i < batch; i++) {
      const rowOffset = i * classes;

      // 1. Find max for numerical stability
      let maxLogit = -Infinity;
      for (let j = 0; j < classes; j++) {
        const val = lData[rowOffset + j];
        if (val > maxLogit) maxLogit = val;
      }

      // 2. Compute exp and sum
      let sumExp = 0;
      for (let j = 0; j < classes; j++) {
        const expVal = Math.exp(lData[rowOffset + j] - maxLogit);
        pData[rowOffset + j] = expVal;
        sumExp += expVal;
      }

      // 3. Normalize into probabilities and accumulate negative log-likelihood
      const invSum = 1.0 / (sumExp + 1e-15);
      for (let j = 0; j < classes; j++) {
        pData[rowOffset + j] *= invSum;

        const targetVal = tData[rowOffset + j];
        if (targetVal > 0) {
          totalLoss -= targetVal * Math.log(pData[rowOffset + j] + 1e-15);
        }
      }
    }

    return totalLoss / batch;
  }

  /**
   * Backward pass: computes gradient w.r.t logits
   * dZ = (Y_hat - Y) / batchSize
   */
  backward() {
    const batch = this.probabilities.rows;
    const classes = this.probabilities.cols;
    const gradLogits = new Matrix2D(batch, classes);

    const pData = this.probabilities.data;
    const tData = this.targets.data;
    const gData = gradLogits.data;

    const scale = 1.0 / batch;
    for (let i = 0; i < pData.length; i++) {
      gData[i] = (pData[i] - tData[i]) * scale;
    }

    return gradLogits;
  }

  /**
   * Computes categorical classification accuracy
   */
  accuracy(probabilities, targets) {
    const batch = probabilities.rows;
    const classes = probabilities.cols;
    let correct = 0;

    for (let i = 0; i < batch; i++) {
      const offset = i * classes;
      let predClass = 0;
      let maxProb = -Infinity;
      let trueClass = 0;
      let maxTarget = -Infinity;

      for (let j = 0; j < classes; j++) {
        const p = probabilities.data[offset + j];
        const t = targets.data[offset + j];

        if (p > maxProb) {
          maxProb = p;
          predClass = j;
        }
        if (t > maxTarget) {
          maxTarget = t;
          trueClass = j;
        }
      }

      if (predClass === trueClass) {
        correct++;
      }
    }

    return correct / batch;
  }
}
