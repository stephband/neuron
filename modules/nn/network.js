import { Matrix2D } from './matrix.js';

/**
 * Sequential Neural Network Container
 */
export class SequentialNetwork {
  constructor(layers = []) {
    this.layers = layers;
  }

  add(layer) {
    this.layers.push(layer);
    return this;
  }

  forward(input) {
    let current = input;
    for (const layer of this.layers) {
      current = layer.forward(current);
    }
    return current;
  }

  backward(gradOutput) {
    let currentGrad = gradOutput;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      currentGrad = this.layers[i].backward(currentGrad);
    }
    return currentGrad;
  }

  /**
   * Run a single training step (forward + loss + backward + optimizer step)
   */
  trainStep(inputs, targets, lossFn, optimizer) {
    // 1. Forward pass through layers to get logits
    const logits = this.forward(inputs);

    // 2. Compute loss and softmax probabilities
    const loss = lossFn.forward(logits, targets);
    const accuracy = lossFn.accuracy(lossFn.probabilities, targets);

    // 3. Compute loss gradient with respect to logits
    const gradLogits = lossFn.backward();

    // 4. Backpropagate error gradients through all layers
    this.backward(gradLogits);

    // 5. Update weights and biases via optimizer
    optimizer.step(this.layers);

    return {
      loss,
      accuracy,
      probabilities: lossFn.probabilities,
    };
  }

  /**
   * Inference only: returns raw logits or probabilities using lossFn
   */
  predict(inputs, lossFn = null) {
    const logits = this.forward(inputs);
    if (lossFn) {
      lossFn.forward(logits, Matrix2D.zeros(logits.rows, logits.cols));
      return lossFn.probabilities;
    }
    return logits;
  }
}
