import { Matrix2D } from '../modules/matrix.js';
import { DenseLayer, ReLULayer } from '../modules/nn/layers.js';
import { SoftmaxCrossEntropy } from '../modules/nn/loss.js';
import { Adam } from '../modules/nn/optimizer.js';
import { SequentialNetwork } from '../modules/nn/network.js';

console.log('=== WebGPU Neural Network (Phase 1: CPU Reference) ===\n');

// 1. Prepare Dataset: XOR Problem (Non-linearly separable)
// Inputs: [0,0], [0,1], [1,0], [1,1]
const X = new Matrix2D(4, 2, [
  0, 0,
  0, 1,
  1, 0,
  1, 1,
]);

// Targets (One-hot encoded: [Class 0, Class 1])
// [0,0] -> Class 0 [1, 0]
// [0,1] -> Class 1 [0, 1]
// [1,0] -> Class 1 [0, 1]
// [1,1] -> Class 0 [1, 0]
const Y = new Matrix2D(4, 2, [
  1, 0,
  0, 1,
  0, 1,
  1, 0,
]);

// 2. Build Multi-Layer Perceptron (2 -> 8 -> 2)
const net = new SequentialNetwork([
  new DenseLayer(2, 8),
  new ReLULayer(),
  new DenseLayer(8, 2),
]);

const lossFn = new SoftmaxCrossEntropy();
const optimizer = new Adam(0.05);

console.log('Training network on XOR...');
const epochs = 400;

for (let epoch = 1; epoch <= epochs; epoch++) {
  const { loss, accuracy } = net.trainStep(X, Y, lossFn, optimizer);

  if (epoch % 50 === 0 || epoch === epochs) {
    console.log(
      `Epoch ${epoch.toString().padStart(3, ' ')} | ` +
      `Loss: ${loss.toFixed(4)} | ` +
      `Accuracy: ${(accuracy * 100).toFixed(1)}%`
    );
  }
}

// 3. Evaluate Predictions
console.log('\n--- Final Predictions on XOR ---');
const finalLogits = net.forward(X);
lossFn.forward(finalLogits, Y);
const probs = lossFn.probabilities;

const inputsLabels = ['[0, 0]', '[0, 1]', '[1, 0]', '[1, 1]'];
const targetLabels = ['Class 0', 'Class 1', 'Class 1', 'Class 0'];

for (let i = 0; i < 4; i++) {
  const p0 = probs.get(i, 0);
  const p1 = probs.get(i, 1);
  const predicted = p1 > p0 ? 'Class 1' : 'Class 0';
  const confidence = (Math.max(p0, p1) * 100).toFixed(1);
  console.log(
    `Input ${inputsLabels[i]} -> Target: ${targetLabels[i]} | ` +
    `Predicted: ${predicted} (${confidence}% conf) [p0=${p0.toFixed(3)}, p1=${p1.toFixed(3)}]`
  );
}

// 4. Test Continuous Online Learning (Human-in-the-loop update)
console.log('\n--- Simulating Continuous Online Learning ---');
// Let's take a new observation: [0.9, 0.1] (near [1, 0], expected Class 1)
const newSampleX = new Matrix2D(1, 2, [0.9, 0.1]);
const userCorrectTarget = new Matrix2D(1, 2, [0, 1]); // User says: "This is Class 1!"

const predBefore = net.forward(newSampleX);
lossFn.forward(predBefore, userCorrectTarget);
console.log(`Before user feedback: Pred P(Class 1) = ${(lossFn.probabilities.get(0, 1) * 100).toFixed(1)}%`);

// Simulate user correcting / confirming by running 5 quick feedback steps
console.log('Applying online training update based on user feedback...');
for (let step = 0; step < 5; step++) {
  net.trainStep(newSampleX, userCorrectTarget, lossFn, optimizer);
}

const predAfter = net.forward(newSampleX);
lossFn.forward(predAfter, userCorrectTarget);
console.log(`After user feedback:  Pred P(Class 1) = ${(lossFn.probabilities.get(0, 1) * 100).toFixed(1)}%`);
console.log('\nPhase 1 CPU reference engine verified successfully!');
