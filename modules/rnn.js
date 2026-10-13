// WebGPU Recurrent Neural Network Runtime
// Provides everything needed to run recurrent models on the GPU.
export { default as default, RNN, RNN as GpuRecurrentNetwork } from './rnn/network-gpu.js';
export { deserializeWeights } from './deserialize.js';
export { initWebGPU } from './gpu/context.js';
export { GpuOps } from './gpu/ops.js';
