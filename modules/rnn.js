import RNN from "./rnn/network-gpu.js";
import { initWebGPU } from "./gpu/context.js";

/**
 * Asynchronously creates an RNN instance with initialized WebGPU resources,
 * validates that the weights byte length matches the architecture, and loads the weights.
 *
 * @param {Object} definition - Architecture definition (inDim, layers, priming, heads)
 * @param {Float32Array|ArrayBuffer|Uint8Array} weights - Model parameter weights
 * @param {GPUDevice} [device] - Optional WebGPU device
 * @returns {Promise<RNN>} Loaded RNN instance
 */

RNN.from = async function (definition, weights, device = null) {
    if (!device) {
        const gpu = await initWebGPU();
        device = gpu.device;
    }

    return new this(device, definition, weights);
};

export default RNN;
