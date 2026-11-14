import {
    createStorageBuffer,
    uploadToBuffer,
} from "../gpu/buffer.js";
import GPUOps from "../gpu/ops.js";
import { normalizeWeights } from "../deserialize.js";
import { resolveHeadMeta, resolveLayerConcat } from "./concat.js";
import { GpuBidirectionalGru } from "./gpu-bigru.js";

/**
 * RNN: End-to-End WebGPU Recurrent Neural Network
 *
 * Executes multi-layer, multi-head recurrent neural networks entirely in GPU VRAM:
 * - Priming conditioning projection
 * - Stacked BiGRU / GRU recurrent layers
 * - Linear output head projections
 * - Softmax probabilities computed via GPU compute shaders
 */

export default class RNN {
    constructor(device, {
        inDim,
        layers = [{ hiddenDim: 48, bidirectional: true }],
        priming = null,
        heads = [{
            name: "output",
            layerIndex: -1,
            outDim: 1,
            priorBias: null,
        }],
    }, weights = null) {
        this.device = device;
        this.gpuOps = GPUOps.from(device);
        this.inDim = inDim;
        this.layerConfigs = layers;
        this.primingConfig = priming;
        this.headConfigs = heads;

        // 1. Recurrent Layers
        const headMetaByName = resolveHeadMeta(heads, layers.length);
        this.layers = [];
        this.layerConcatSources = [];

        for (let l = 0; l < layers.length; l++) {
            const cfg = layers[l];
            const baseInDim = l === 0 ? inDim : this.layers[l - 1].outDim;
            const concatSources = resolveLayerConcat(
                cfg.concat,
                l,
                inDim,
                headMetaByName,
            );
            this.layerConcatSources.push(concatSources);

            const layerInDim = baseInDim +
                concatSources.reduce((sum, src) => sum + src.dim, 0);
            const hDim = cfg.hiddenDim || 48;
            const bidi = cfg.bidirectional !== undefined
                ? cfg.bidirectional
                : true;
            const layer = new GpuBidirectionalGru(
                device,
                layerInDim,
                hDim,
                bidi,
            );
            this.layers.push(layer);
        }

        // 2. Priming Projection
        this.priming = null;
        if (priming) {
            const H0 = this.layers[0].hiddenDim;
            const pDim = priming.inDim;
            this.priming = {
                type: priming.type || "categorical",
                inDim: pDim,
                bufW: createStorageBuffer(
                    device,
                    H0 * pDim * 4,
                    "gpu_priming_W",
                ),
                bufB: createStorageBuffer(device, H0 * 4, "gpu_priming_b"),
                bufH0: createStorageBuffer(device, H0 * 4, "gpu_priming_h0"),
                h0Cpu: new Float32Array(H0),
                cpuW: new Float32Array(H0 * pDim),
                cpuB: new Float32Array(H0),
            };
        }
        this.bufCustomH0 = null;

        // 3. Output Heads
        this.heads = [];
        for (let h = 0; h < heads.length; h++) {
            const hCfg = heads[h];
            const name = hCfg.name || `head_${h}`;
            let layerIdx = hCfg.layerIndex !== undefined ? hCfg.layerIndex : -1;
            if (layerIdx < 0) {
                layerIdx = this.layers.length + layerIdx;
            }
            const layerOutDim = this.layers[layerIdx].outDim;
            const outDim = hCfg.outDim;

            // Prepare bias + prior
            const biasWithPrior = new Float32Array(outDim);
            if (hCfg.priorBias) {
                for (let i = 0; i < outDim; i++) {
                    biasWithPrior[i] = hCfg.priorBias[i];
                }
            }

            this.heads.push({
                name,
                layerIndex: layerIdx,
                outDim,
                layerOutDim,
                priorBias: hCfg.priorBias
                    ? new Float32Array(hCfg.priorBias)
                    : null,
                bufW: createStorageBuffer(
                    device,
                    outDim * layerOutDim * 4,
                    `gpu_head_${name}_W`,
                ),
                bufB: createStorageBuffer(
                    device,
                    biasWithPrior,
                    `gpu_head_${name}_b`,
                ),
                bufLogits: null,
                bufProbs: null,
            });
        }

        // Reusable sequence scratch buffers (sized for capT)
        this.capT = 0;
        this.bufPredictX = null;
        this.layerConcatBufs = new Array(this.layers.length).fill(null);
        this.stagingBuffer = null;
        this._predictQueue = Promise.resolve();

        if (weights) {
            this.loadWeights(weights);
        }
    }

    /**
     * Ensures reusable sequence buffers (inputs, concat, head logits/probs, readback staging)
     * are allocated for at least T timesteps.
     */
    ensureScratchBuffers(T) {
        if (T <= this.capT) return;
        if (this.bufPredictX) this.bufPredictX.destroy();
        for (let l = 0; l < this.layerConcatBufs.length; l++) {
            if (this.layerConcatBufs[l]) this.layerConcatBufs[l].destroy();
        }
        for (const head of this.heads) {
            if (head.bufLogits) head.bufLogits.destroy();
            if (head.bufProbs) head.bufProbs.destroy();
        }
        if (this.stagingBuffer) this.stagingBuffer.destroy();

        this.capT = T;
        this.bufPredictX = createStorageBuffer(
            this.device,
            T * this.inDim * 4,
            "gpu_predict_X",
        );

        for (let l = 0; l < this.layers.length; l++) {
            if (this.layerConcatSources[l].length) {
                this.layerConcatBufs[l] = createStorageBuffer(
                    this.device,
                    T * this.layers[l].inDim * 4,
                    `gpu_layer_${l}_concat_in`,
                );
            }
        }

        let totalHeadOutFloats = 0;
        for (const head of this.heads) {
            const bytes = T * head.outDim * 4;
            head.bufLogits = createStorageBuffer(
                this.device,
                bytes,
                `head_${head.name}_logits`,
            );
            head.bufProbs = createStorageBuffer(
                this.device,
                bytes,
                `head_${head.name}_probs`,
            );
            totalHeadOutFloats += T * head.outDim;
        }

        const stagingBytes = Math.max(16, (totalHeadOutFloats * 4 + 3) & ~3);
        this.stagingBuffer = this.device.createBuffer({
            label: "gpu_predict_staging",
            size: stagingBytes,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
    }

    /**
     * Releases all GPU buffers owned by this network.
     */
    destroy() {
        for (const layer of this.layers) {
            layer.destroy();
        }
        if (this.priming) {
            this.priming.bufW.destroy();
            this.priming.bufB.destroy();
            this.priming.bufH0.destroy();
        }
        if (this.bufCustomH0) this.bufCustomH0.destroy();
        for (const head of this.heads) {
            head.bufW.destroy();
            head.bufB.destroy();
            if (head.bufLogits) head.bufLogits.destroy();
            if (head.bufProbs) head.bufProbs.destroy();
        }
        if (this.bufPredictX) this.bufPredictX.destroy();
        for (const buf of this.layerConcatBufs) {
            if (buf) buf.destroy();
        }
        if (this.stagingBuffer) this.stagingBuffer.destroy();
        this.capT = 0;
    }

    /**
     * Total number of float parameters expected by this network architecture.
     */
    get parameterCount() {
        let count = 0;
        if (this.priming) {
            count += this.priming.cpuW.length + this.priming.cpuB.length;
        }
        for (const layer of this.layers) {
            count += layer.parameterCount;
        }
        for (const head of this.heads) {
            count += head.outDim * head.layerOutDim + head.outDim;
        }
        return count;
    }

    /**
     * Load weights from a flat Float32Array or weight dictionary
     *
     * @param {Float32Array|ArrayBuffer|Object} weights - Flat weights or dictionary
     */
    loadWeights(weights) {
        const flat = normalizeWeights(weights, this.parameterCount);
        let offset = 0;

        // 1. Priming projection
        if (this.priming) {
            const sizeW = this.priming.cpuW.length;
            this.priming.cpuW.set(flat.subarray(offset, offset + sizeW));
            uploadToBuffer(this.device, this.priming.bufW, this.priming.cpuW);
            offset += sizeW;

            const sizeB = this.priming.cpuB.length;
            this.priming.cpuB.set(flat.subarray(offset, offset + sizeB));
            uploadToBuffer(this.device, this.priming.bufB, this.priming.cpuB);
            offset += sizeB;
        }

        // 2. Recurrent Layers
        for (const layer of this.layers) {
            offset += layer.loadWeights(flat, offset);
        }

        // 3. Linear Output Heads
        for (const head of this.heads) {
            const sizeW = head.outDim * head.layerOutDim;
            const wSlice = flat.subarray(offset, offset + sizeW);
            uploadToBuffer(this.device, head.bufW, wSlice);
            offset += sizeW;

            const sizeB = head.outDim;
            const bSlice = flat.subarray(offset, offset + sizeB);
            const combinedB = new Float32Array(bSlice);
            if (head.priorBias) {
                for (let i = 0; i < combinedB.length; i++) {
                    combinedB[i] += head.priorBias[i];
                }
            }
            uploadToBuffer(this.device, head.bufB, combinedB);
            offset += sizeB;
        }

        return this;
    }

    /**
     * Forward pass across sequence of length T purely on GPU
     *
     * @param {GPUBuffer} bufX - Input features buffer [T x inDim]
     * @param {number} T - Sequence length
     * @param {Object} options - Optional controls:
     *   - condition: categorical integer or Float32Array
     *   - h0: GPUBuffer or Float32Array
     * @returns {Object} Head output buffers { [head.name]: { bufLogits, bufProbs } }
     */
    forward(bufX, T, options = {}, encoder = null) {
        const isLocalEncoder = !encoder;
        const cmdEncoder = encoder ||
            this.device.createCommandEncoder({ label: "gpu_net_forward" });

        this.ensureScratchBuffers(T);

        // 1. Resolve Priming State h0 on GPU
        let bufH0 = null;
        if (options.condition !== undefined && this.priming) {
            const H0 = this.layers[0].hiddenDim;
            const h0_cpu = this.priming.h0Cpu;
            const wData = this.priming.cpuW;
            const bData = this.priming.cpuB;
            const pDim = this.priming.inDim;

            if (this.priming.type === "categorical") {
                const c = Number(options.condition);
                if (c >= 0 && c < pDim) {
                    for (let j = 0; j < H0; j++) {
                        h0_cpu[j] = Math.tanh(wData[j * pDim + c] + bData[j]);
                    }
                } else {
                    for (let j = 0; j < H0; j++) {
                        h0_cpu[j] = Math.tanh(bData[j]);
                    }
                }
            } else {
                const v = options.condition;
                for (let j = 0; j < H0; j++) {
                    let sum = bData[j];
                    for (let k = 0; k < pDim; k++) {
                        sum += wData[j * pDim + k] * v[k];
                    }
                    h0_cpu[j] = Math.tanh(sum);
                }
            }
            uploadToBuffer(this.device, this.priming.bufH0, h0_cpu);
            bufH0 = this.priming.bufH0;
        } else if (options.bufH0) {
            bufH0 = options.bufH0;
        } else if (options.h0 instanceof Float32Array) {
            if (!this.bufCustomH0) {
                this.bufCustomH0 = createStorageBuffer(
                    this.device,
                    this.layers[0].hiddenDim * 4,
                    "gpu_h0",
                );
            }
            uploadToBuffer(this.device, this.bufCustomH0, options.h0);
            bufH0 = this.bufCustomH0;
        }

        // 2. Propagate through recurrent layers and interleaved output heads
        const layerOutputs = [];
        const results = {};

        for (let l = 0; l < this.layers.length; l++) {
            const layer = this.layers[l];
            const layerH0 = (l === 0) ? bufH0 : null;
            const baseBuf = l === 0 ? bufX : layerOutputs[l - 1];
            const baseDim = l === 0 ? this.inDim : this.layers[l - 1].outDim;
            const concatSources = this.layerConcatSources[l];

            let layerInBuf = baseBuf;
            if (concatSources.length) {
                layerInBuf = this.layerConcatBufs[l];
                this.gpuOps.dispatchCopyCols(
                    baseBuf,
                    layerInBuf,
                    T,
                    baseDim,
                    layer.inDim,
                    0,
                    cmdEncoder,
                );
                let colOffset = baseDim;
                for (const src of concatSources) {
                    const srcBuf = src.type === "input"
                        ? bufX
                        : results[src.name].bufProbs;
                    this.gpuOps.dispatchCopyCols(
                        srcBuf,
                        layerInBuf,
                        T,
                        src.dim,
                        layer.inDim,
                        colOffset,
                        cmdEncoder,
                    );
                    colOffset += src.dim;
                }
            }

            const bufH_l = layer.forward(
                layerInBuf,
                T,
                layerH0,
                cmdEncoder,
            );
            layerOutputs.push(bufH_l);

            for (const head of this.heads) {
                if (head.layerIndex !== l) continue;

                const inDimHead = head.layerOutDim;
                const outDim = head.outDim;
                const bufLogits = head.bufLogits;
                const bufProbs = head.bufProbs;

                // Fused GEMM + Bias: Logits = H_source * W_head^T + b_head (+ prior)
                this.gpuOps.dispatchMatmul(
                    bufH_l,
                    head.bufW,
                    bufLogits,
                    T,
                    outDim,
                    inDimHead,
                    false,
                    true,
                    cmdEncoder,
                    head.bufB,
                );

                // Softmax on GPU
                this.gpuOps.dispatchSoftmax(
                    bufLogits,
                    bufProbs,
                    T,
                    outDim,
                    cmdEncoder,
                );

                results[head.name] = {
                    bufLogits,
                    bufProbs,
                };
            }
        }

        if (isLocalEncoder) {
            this.device.queue.submit([cmdEncoder.finish()]);
        }

        return results;
    }

    /**
     * Complete inference: runs forward pass on GPU and reads back predictions
     * in a single command submission and a single staging buffer map.
     *
     * @param {Float32Array} X - Input features [T x inDim]
     * @param {number} T - Sequence length
     * @param {Object} options - Options (e.g. condition, h0)
     * @returns {Promise<Object>} Readback probabilities and argmax class predictions
     */
    predict(X, T, options = {}) {
        const next = this._predictQueue.then(() =>
            this._predictNow(X, T, options)
        );
        this._predictQueue = next.catch(() => {});
        return next;
    }

    async _predictNow(X, T, options = {}) {
        this.ensureScratchBuffers(T);
        const inputData = X instanceof Float32Array ? X : new Float32Array(X);
        this.device.queue.writeBuffer(
            this.bufPredictX,
            0,
            inputData.buffer,
            inputData.byteOffset,
            T * this.inDim * 4,
        );

        const cmdEncoder = this.device.createCommandEncoder({
            label: "gpu_predict_encoder",
        });
        const gpuResults = this.forward(
            this.bufPredictX,
            T,
            options,
            cmdEncoder,
        );

        let byteOffset = 0;
        for (const head of this.heads) {
            const byteLen = T * head.outDim * 4;
            cmdEncoder.copyBufferToBuffer(
                gpuResults[head.name].bufProbs,
                0,
                this.stagingBuffer,
                byteOffset,
                byteLen,
            );
            byteOffset += byteLen;
        }

        this.device.queue.submit([cmdEncoder.finish()]);

        await this.stagingBuffer.mapAsync(GPUMapMode.READ, 0, byteOffset);
        const mappedRange = this.stagingBuffer.getMappedRange(0, byteOffset);
        const allProbs = new Float32Array(mappedRange.slice(0));
        this.stagingBuffer.unmap();

        const outputs = {};
        let floatOffset = 0;
        for (const head of this.heads) {
            const outDim = head.outDim;
            const len = T * outDim;
            const probs = allProbs.subarray(floatOffset, floatOffset + len);
            floatOffset += len;

            const preds = new Uint32Array(T);
            for (let t = 0; t < T; t++) {
                const offset = t * outDim;
                let maxP = -Infinity;
                let bestIdx = 0;
                for (let o = 0; o < outDim; o++) {
                    if (probs[offset + o] > maxP) {
                        maxP = probs[offset + o];
                        bestIdx = o;
                    }
                }
                preds[t] = bestIdx;
            }

            outputs[head.name] = {
                probs,
                preds,
            };
        }

        return outputs;
    }
}
