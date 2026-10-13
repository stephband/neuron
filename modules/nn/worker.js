import { initWebGPU } from '../gpu/context.js';
import {
  createStorageBuffer,
  uploadToBuffer,
  readFromBuffer,
} from '../gpu/buffer.js';
import { GpuOps } from '../gpu/ops.js';
import {
  GpuDenseLayer,
  GpuReLULayer,
  GpuSoftmaxCrossEntropy,
  GpuAdam,
  GpuSequentialNetwork,
} from '../gpu/network.js';
import { ReplayBuffer } from './replay_buffer.js';

let device = null;
let adapter = null;
let gpuOps = null;
let network = null;
let lossFn = null;
let optimizer = null;
let replayBuffer = null;

let inDim = 2;
let hiddenDim = 16;
let numClasses = 2;

// Reusable buffers
let bufInputs = null;
let bufTargets = null;
let bufInputsCapacity = 0;
let bufTargetsCapacity = 0;

function ensureInputBuffers(batchSize, inD, outD) {
  const inBytes = batchSize * inD * 4;
  const outBytes = batchSize * outD * 4;

  if (!bufInputs || bufInputsCapacity < inBytes) {
    if (bufInputs) bufInputs.destroy();
    bufInputs = createStorageBuffer(device, inBytes, 'worker_input_buf');
    bufInputsCapacity = inBytes;
  }

  if (!bufTargets || bufTargetsCapacity < outBytes) {
    if (bufTargets) bufTargets.destroy();
    bufTargets = createStorageBuffer(device, outBytes, 'worker_target_buf');
    bufTargetsCapacity = outBytes;
  }
}

self.onmessage = async (event) => {
  const { type, id, payload } = event.data;

  try {
    switch (type) {
      case 'INIT': {
        const res = await initWebGPU();
        device = res.device;
        adapter = res.adapter;
        gpuOps = new GpuOps(device);

        inDim = payload?.inDim || 2;
        hiddenDim = payload?.hiddenDim || 16;
        numClasses = payload?.outDim || 2;
        const lr = payload?.lr || 0.05;

        // Construct Network: inDim -> hiddenDim -> ReLU -> hiddenDim -> ReLU -> numClasses
        network = new GpuSequentialNetwork(device, gpuOps, [
          new GpuDenseLayer(device, gpuOps, inDim, hiddenDim),
          new GpuReLULayer(device, gpuOps),
          new GpuDenseLayer(device, gpuOps, hiddenDim, hiddenDim),
          new GpuReLULayer(device, gpuOps),
          new GpuDenseLayer(device, gpuOps, hiddenDim, numClasses),
        ]);

        lossFn = new GpuSoftmaxCrossEntropy(device, gpuOps);
        optimizer = new GpuAdam(device, gpuOps, lr);
        replayBuffer = new ReplayBuffer(inDim, numClasses, 1000);

        self.postMessage({
          type: 'INIT_SUCCESS',
          id,
          payload: {
            adapterDescription: adapter.info.description || 'WebGPU Device',
            inDim,
            hiddenDim,
            numClasses,
          },
        });
        break;
      }

      case 'TRAIN_BATCH': {
        const { x, y, batchSize, epochs = 100, notifyEvery = 20 } = payload;
        ensureInputBuffers(batchSize, inDim, numClasses);

        uploadToBuffer(device, bufInputs, x);
        uploadToBuffer(device, bufTargets, y);

        // Store into experience replay buffer
        replayBuffer.add(x, y, batchSize);

        let finalLoss = 0;
        let finalAcc = 0;

        for (let epoch = 1; epoch <= epochs; epoch++) {
          const { probabilities, sampleLosses } = network.trainStep(
            bufInputs,
            bufTargets,
            batchSize,
            numClasses,
            lossFn,
            optimizer
          );

          if (epoch % notifyEvery === 0 || epoch === epochs) {
            const losses = await readFromBuffer(device, sampleLosses, batchSize * 4);
            const probs = await readFromBuffer(device, probabilities, batchSize * numClasses * 4);

            let sumLoss = 0;
            for (let i = 0; i < batchSize; i++) sumLoss += losses[i];
            finalLoss = sumLoss / batchSize;

            let correct = 0;
            for (let b = 0; b < batchSize; b++) {
              let predClass = 0, predMax = -Infinity;
              let trueClass = 0, trueMax = -Infinity;
              for (let c = 0; c < numClasses; c++) {
                if (probs[b * numClasses + c] > predMax) {
                  predMax = probs[b * numClasses + c];
                  predClass = c;
                }
                if (y[b * numClasses + c] > trueMax) {
                  trueMax = y[b * numClasses + c];
                  trueClass = c;
                }
              }
              if (predClass === trueClass) correct++;
            }
            finalAcc = correct / batchSize;

            self.postMessage({
              type: 'TRAIN_PROGRESS',
              id,
              payload: {
                epoch,
                epochs,
                loss: finalLoss,
                accuracy: finalAcc,
              },
            });
          }
        }

        self.postMessage({
          type: 'TRAIN_COMPLETE',
          id,
          payload: {
            loss: finalLoss,
            accuracy: finalAcc,
            replayBufferSize: replayBuffer.size,
          },
        });
        break;
      }

      case 'PREDICT': {
        const { x, batchSize } = payload;
        ensureInputBuffers(batchSize, inDim, numClasses);
        uploadToBuffer(device, bufInputs, x);

        const enc = device.createCommandEncoder({ label: 'worker_predict_encoder' });
        const logits = network.forward(bufInputs, batchSize, enc);
        const dummyTargets = createStorageBuffer(device, batchSize * numClasses * 4);
        const { probabilities } = lossFn.forwardAndBackward(
          logits,
          dummyTargets,
          batchSize,
          numClasses,
          enc
        );
        device.queue.submit([enc.finish()]);

        const probs = await readFromBuffer(device, probabilities, batchSize * numClasses * 4);
        dummyTargets.destroy();

        self.postMessage({
          type: 'PREDICT_RESULT',
          id,
          payload: { probs },
        });
        break;
      }

      case 'FEEDBACK': {
        // Continuous online learning: new sample + experience replay samples
        const { x, targetY, replayBatchSize = 16, steps = 5 } = payload;
        const miniBatch = replayBuffer.sampleWithFeedback(x, targetY, replayBatchSize);

        ensureInputBuffers(miniBatch.batchSize, inDim, numClasses);
        uploadToBuffer(device, bufInputs, miniBatch.x);
        uploadToBuffer(device, bufTargets, miniBatch.y);

        for (let s = 0; s < steps; s++) {
          network.trainStep(
            bufInputs,
            bufTargets,
            miniBatch.batchSize,
            numClasses,
            lossFn,
            optimizer
          );
        }

        // Run forward pass on the single feedback sample to return updated confidence
        ensureInputBuffers(1, inDim, numClasses);
        uploadToBuffer(device, bufInputs, x);
        const enc = device.createCommandEncoder();
        const logits = network.forward(bufInputs, 1, enc);
        const dummyTargets = createStorageBuffer(device, 1 * numClasses * 4);
        const { probabilities } = lossFn.forwardAndBackward(
          logits,
          dummyTargets,
          1,
          numClasses,
          enc
        );
        device.queue.submit([enc.finish()]);

        const newProbs = await readFromBuffer(device, probabilities, numClasses * 4);
        dummyTargets.destroy();

        self.postMessage({
          type: 'FEEDBACK_SUCCESS',
          id,
          payload: {
            newProbs,
            replayBufferSize: replayBuffer.size,
          },
        });
        break;
      }

      case 'DECISION_GRID': {
        const { gridPoints, batchSize } = payload;
        ensureInputBuffers(batchSize, inDim, numClasses);
        uploadToBuffer(device, bufInputs, gridPoints);

        const enc = device.createCommandEncoder({ label: 'grid_predict_encoder' });
        const logits = network.forward(bufInputs, batchSize, enc);
        const dummyTargets = createStorageBuffer(device, batchSize * numClasses * 4);
        const { probabilities } = lossFn.forwardAndBackward(
          logits,
          dummyTargets,
          batchSize,
          numClasses,
          enc
        );
        device.queue.submit([enc.finish()]);

        const probs = await readFromBuffer(device, probabilities, batchSize * numClasses * 4);
        dummyTargets.destroy();

        self.postMessage({
          type: 'DECISION_GRID_RESULT',
          id,
          payload: { probs },
        });
        break;
      }

      case 'RESET': {
        if (replayBuffer) replayBuffer.clear();
        // Re-initialize network weights
        network = new GpuSequentialNetwork(device, gpuOps, [
          new GpuDenseLayer(device, gpuOps, inDim, hiddenDim),
          new GpuReLULayer(device, gpuOps),
          new GpuDenseLayer(device, gpuOps, hiddenDim, hiddenDim),
          new GpuReLULayer(device, gpuOps),
          new GpuDenseLayer(device, gpuOps, hiddenDim, numClasses),
        ]);
        self.postMessage({ type: 'RESET_SUCCESS', id });
        break;
      }

      default:
        throw new Error(`Unknown message type: ${type}`);
    }
  } catch (err) {
    self.postMessage({
      type: 'ERROR',
      id,
      error: err.message,
    });
  }
};
