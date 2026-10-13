
import RNN            from './rnn/network-gpu.js';
import { initWebGPU } from './gpu/context.js';
import { GpuOps }     from './gpu/ops.js';

/**
 * Asynchronously creates an RNN instance with initialized WebGPU resources,
 * validates that the weights byte length matches the architecture, and loads the weights.
 *
 * @param {Object} definition - Architecture definition (inDim, layers, priming, heads)
 * @param {Float32Array|ArrayBuffer|Uint8Array} weights - Model parameter weights
 * @param {GPUDevice} [device] - Optional WebGPU device
 * @param {GpuOps} [gpuOps] - Optional GpuOps instance
 * @returns {Promise<RNN>} Loaded RNN instance
 */

RNN.from = async function(definition, weights, device = null, gpuOps = null) {
    if (!device) {
        const gpu = await initWebGPU();
        device = gpu.device;
    }

    if (!gpuOps) {
        gpuOps = new GpuOps(device);
    }

    if (weights.byteLength % 4 !== 0) {
        throw new Error(`Invalid weights byte length ${ weights.byteLength } not a multiple of 4`);
    }

    const rnn = new this(device, gpuOps, definition);
    return rnn.loadWeights(weights);
}

export { RNN as default, initWebGPU, GpuOps };
