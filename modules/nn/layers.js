import { Matrix2D } from '../matrix.js';

/**
 * Dense (Fully Connected) Layer
 * 
 * Forward:  Z = X * W + b
 * Backward: 
 *   dW = X^T * dZ
 *   db = sum_rows(dZ)
 *   dX = dZ * W^T
 */
export class DenseLayer {
  constructor(inDim, outDim) {
    this.inDim = inDim;
    this.outDim = outDim;

    // Weights initialized with He Normal
    this.weights = Matrix2D.randomHe(inDim, outDim);
    // Biases initialized with zeros
    this.biases = Matrix2D.zeros(1, outDim);

    // Gradients
    this.gradW = Matrix2D.zeros(inDim, outDim);
    this.gradB = Matrix2D.zeros(1, outDim);

    // Cached input for backprop
    this.cachedInput = null;
  }

  forward(input) {
    this.cachedInput = input;
    // Z = input * W
    const linear = Matrix2D.matmul(input, this.weights);
    // Z = linear + b
    return Matrix2D.addBias(linear, this.biases);
  }

  backward(gradOutput) {
    // gradOutput is dZ: [batch x outDim]
    // 1. dW = (cachedInput)^T * dZ: [inDim x batch] * [batch x outDim] = [inDim x outDim]
    const inputT = Matrix2D.transpose(this.cachedInput);
    this.gradW.fill(0);
    Matrix2D.matmul(inputT, gradOutput, this.gradW);

    // 2. db = sum_rows(dZ): [1 x outDim]
    Matrix2D.sumRows(gradOutput, this.gradB);

    // 3. dX = dZ * W^T: [batch x outDim] * [outDim x inDim] = [batch x inDim]
    const weightsT = Matrix2D.transpose(this.weights);
    const gradInput = Matrix2D.matmul(gradOutput, weightsT);

    return gradInput;
  }
}

/**
 * ReLU Activation Layer
 * 
 * Forward:  A = max(0, Z)
 * Backward: dZ = dA * (Z > 0 ? 1 : 0)
 */
export class ReLULayer {
  constructor() {
    this.cachedInput = null;
  }

  forward(input) {
    this.cachedInput = input;
    const output = new Matrix2D(input.rows, input.cols);
    for (let i = 0; i < input.data.length; i++) {
      output.data[i] = Math.max(0, input.data[i]);
    }
    return output;
  }

  backward(gradOutput) {
    const gradInput = new Matrix2D(gradOutput.rows, gradOutput.cols);
    const inData = this.cachedInput.data;
    const gradOutData = gradOutput.data;
    const gradInData = gradInput.data;

    for (let i = 0; i < inData.length; i++) {
      gradInData[i] = inData[i] > 0 ? gradOutData[i] : 0;
    }
    return gradInput;
  }
}
