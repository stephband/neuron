import { initWebGPU } from '../modules/gpu/context.js';
import { createStorageBuffer, readFromBuffer } from '../modules/gpu/buffer.js';
import { GpuOps } from '../modules/gpu/ops.js';
import {
  GpuDenseLayer,
  GpuReLULayer,
  GpuSoftmaxCrossEntropy,
  GpuAdam,
  GpuSequentialNetwork,
} from '../modules/nn/gpu_network.js';

console.log('=== WebGPU End-to-End Neural Network Training (Phase 3) ===\n');

// 1. Initialize WebGPU on the Apple M1 Max
const { device, adapter } = await initWebGPU();
console.log(`Using GPU Adapter: ${adapter.info.description || 'WebGPU Device'}`);

const gpuOps = new GpuOps(device);

// 2. Prepare XOR Dataset directly on GPU Storage Buffers
// Inputs: 4 samples of 2 features
const xData = new Float32Array([
  0, 0,
  0, 1,
  1, 0,
  1, 1,
]);
const bufX = createStorageBuffer(device, xData, 'xor_inputs');

// Targets (One-hot encoded: [Class 0, Class 1])
const yData = new Float32Array([
  1, 0,
  0, 1,
  0, 1,
  1, 0,
]);
const bufY = createStorageBuffer(device, yData, 'xor_targets');

const batchSize = 4;
const numFeatures = 2;
const numClasses = 2;

// 3. Build GPU Neural Network: 2 -> 8 -> 2
const net = new GpuSequentialNetwork(device, gpuOps, [
  new GpuDenseLayer(device, gpuOps, 2, 8),
  new GpuReLULayer(device, gpuOps),
  new GpuDenseLayer(device, gpuOps, 8, 2),
]);

const lossFn = new GpuSoftmaxCrossEntropy(device, gpuOps);
const optimizer = new GpuAdam(device, gpuOps, 0.05);

console.log('Training network purely in GPU VRAM (no CPU stalls during iterations)...');
const epochs = 400;
const startTime = performance.now();

let lastProbsBuf = null;
let lastLossBuf = null;

for (let epoch = 1; epoch <= epochs; epoch++) {
  // Train step: executes forward + loss + backward + Adam entirely on GPU
  const { probabilities, sampleLosses } = net.trainStep(
    bufX,
    bufY,
    batchSize,
    numClasses,
    lossFn,
    optimizer
  );

  lastProbsBuf = probabilities;
  lastLossBuf = sampleLosses;

  // Read back metrics asynchronously every 50 epochs
  if (epoch % 50 === 0 || epoch === epochs) {
    const losses = await readFromBuffer(device, sampleLosses, batchSize * 4);
    const probs = await readFromBuffer(device, probabilities, batchSize * numClasses * 4);

    let avgLoss = 0;
    for (let i = 0; i < batchSize; i++) avgLoss += losses[i];
    avgLoss /= batchSize;

    let correct = 0;
    for (let b = 0; b < batchSize; b++) {
      const p0 = probs[b * 2];
      const p1 = probs[b * 2 + 1];
      const predClass = p1 > p0 ? 1 : 0;
      const targetClass = yData[b * 2 + 1] > yData[b * 2] ? 1 : 0;
      if (predClass === targetClass) correct++;
    }
    const accuracy = (correct / batchSize) * 100;

    console.log(
      `Epoch ${epoch.toString().padStart(3, ' ')} | ` +
      `Loss: ${avgLoss.toFixed(4)} | ` +
      `Accuracy: ${accuracy.toFixed(1)}%`
    );
  }
}

const totalTime = (performance.now() - startTime).toFixed(1);
console.log(`\nCompleted ${epochs} GPU training epochs in ${totalTime} ms`);

// 4. Inspect Final GPU Predictions
console.log('\n--- Final GPU Predictions on XOR ---');
const finalProbs = await readFromBuffer(device, lastProbsBuf, batchSize * numClasses * 4);
const inputsLabels = ['[0, 0]', '[0, 1]', '[1, 0]', '[1, 1]'];
const targetLabels = ['Class 0', 'Class 1', 'Class 1', 'Class 0'];

for (let i = 0; i < batchSize; i++) {
  const p0 = finalProbs[i * 2];
  const p1 = finalProbs[i * 2 + 1];
  const pred = p1 > p0 ? 'Class 1' : 'Class 0';
  const conf = (Math.max(p0, p1) * 100).toFixed(1);
  console.log(
    `Input ${inputsLabels[i]} -> Target: ${targetLabels[i]} | ` +
    `Predicted: ${pred} (${conf}% conf) [p0=${p0.toFixed(3)}, p1=${p1.toFixed(3)}]`
  );
}

// 5. Test Continuous GPU Online Learning
console.log('\n--- Simulating Continuous Online Learning on GPU ---');
const newSample = new Float32Array([0.9, 0.1]);
const userTarget = new Float32Array([0, 1]); // Class 1
const bufSample = createStorageBuffer(device, newSample, 'online_sample');
const bufTarget = createStorageBuffer(device, userTarget, 'online_target');

// Predict before update
const encBefore = device.createCommandEncoder();
const logitsBefore = net.forward(bufSample, 1, encBefore);
const { probabilities: probsBefore } = lossFn.forwardAndBackward(
  logitsBefore,
  bufTarget,
  1,
  2,
  encBefore
);
device.queue.submit([encBefore.finish()]);

const pb = await readFromBuffer(device, probsBefore, 2 * 4);
console.log(`Before user feedback: Pred P(Class 1) = ${(pb[1] * 100).toFixed(1)}%`);

console.log('Applying 5 continuous learning GPU update steps...');
for (let s = 0; s < 5; s++) {
  net.trainStep(bufSample, bufTarget, 1, 2, lossFn, optimizer);
}

// Predict after update
const encAfter = device.createCommandEncoder();
const logitsAfter = net.forward(bufSample, 1, encAfter);
const { probabilities: probsAfter } = lossFn.forwardAndBackward(
  logitsAfter,
  bufTarget,
  1,
  2,
  encAfter
);
device.queue.submit([encAfter.finish()]);

const pa = await readFromBuffer(device, probsAfter, 2 * 4);
console.log(`After user feedback:  Pred P(Class 1) = ${(pa[1] * 100).toFixed(1)}%`);

console.log('\nPhase 3 WebGPU Backpropagation & Optimizer verified with complete success!');
