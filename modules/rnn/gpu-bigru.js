import { createStorageBuffer } from "../gpu/buffer.js";
import GPUOps from "../gpu/ops.js";

/**
 * GpuBidirectionalGru: WebGPU-accelerated Recurrent Layer
 *
 * Executes full sequential bidirectional GRU inference natively on GPU VRAM:
 * 1. Parallel Pre-projection GEMM across all time steps: U = X * W_ih^T + b_ih
 * 2. Sequential recurrent forward pass (t = 0 -> T-1)
 * 3. Sequential recurrent backward pass (t = T-1 -> 0)
 * 4. Output concatenation H_t = [h_fwd_t ; h_bwd_t]
 */
export class GpuBidirectionalGru {
    constructor(device, inDim, hiddenDim, bidirectional = true) {
        if (!hiddenDim || hiddenDim <= 0 || hiddenDim > 256) {
            throw new Error(
                `GpuBidirectionalGru hiddenDim ${hiddenDim} out of supported range [1, 256]`,
            );
        }
        this.device = device;
        this.gpuOps = GPUOps.from(device);
        this.inDim = inDim;
        this.hiddenDim = hiddenDim;
        this.bidirectional = bidirectional;
        this.outDim = bidirectional ? 2 * hiddenDim : hiddenDim;

        const H = hiddenDim;
        const dirCount = bidirectional ? 2 : 1;

        // Contiguous GPU Buffers for [Fwd (, Bwd)] Cell Weights
        this.bufW_ih = createStorageBuffer(
            device,
            dirCount * 3 * H * inDim * 4,
            "gru_Wih",
        );
        this.bufB_ih = createStorageBuffer(
            device,
            dirCount * 3 * H * 4,
            "gru_bih",
        );
        this.bufW_hh = createStorageBuffer(
            device,
            dirCount * 3 * H * H * 4,
            "gru_Whh",
        );
        this.bufB_hh = createStorageBuffer(
            device,
            dirCount * 3 * H * 4,
            "gru_bhh",
        );

        // Reusable sequence scratch buffers (sized for capT)
        this.capT = 0;
        this.bufU = null;
        this.bufOutH = null;
    }

    /**
     * Ensures reusable sequence scratch buffers can hold at least T steps.
     */
    ensureScratchBuffers(T) {
        if (T <= this.capT) return;
        if (this.bufU) this.bufU.destroy();
        if (this.bufOutH) this.bufOutH.destroy();

        const H = this.hiddenDim;
        const outD = this.outDim;
        const dirCount = this.bidirectional ? 2 : 1;
        this.capT = T;
        this.bufU = createStorageBuffer(
            this.device,
            T * dirCount * 3 * H * 4,
            "temp_U",
        );
        this.bufOutH = createStorageBuffer(
            this.device,
            T * outD * 4,
            "gru_outH",
        );
    }

    /**
     * Releases all GPU buffers owned by this layer.
     */
    destroy() {
        this.bufW_ih.destroy();
        this.bufB_ih.destroy();
        this.bufW_hh.destroy();
        this.bufB_hh.destroy();
        if (this.bufU) this.bufU.destroy();
        if (this.bufOutH) this.bufOutH.destroy();
        this.capT = 0;
    }

    /**
     * Total number of float parameters in this recurrent layer
     */
    get parameterCount() {
        const H = this.hiddenDim;
        const singleDirection = 3 * H * this.inDim + 3 * H * H + 6 * H;
        return this.bidirectional ? 2 * singleDirection : singleDirection;
    }

    /**
     * Upload weights from a flat Float32Array or ArrayBuffer slice
     *
     * @param {Float32Array|ArrayBuffer} weights - Flat weights
     * @param {number} [offset=0] - Starting float offset
     * @returns {number} Number of floats consumed
     */
    loadWeights(weights, offset = 0) {
        const queue = this.device.queue;
        const flat = weights instanceof Float32Array
            ? weights
            : new Float32Array(weights);
        const H = this.hiddenDim;
        const inDim = this.inDim;

        const size_Wih = 3 * H * inDim;
        const size_Whh = 3 * H * H;
        const size_b = 3 * H;

        let cur = offset;
        queue.writeBuffer(
            this.bufW_ih,
            0,
            flat.buffer,
            flat.byteOffset + cur * 4,
            size_Wih * 4,
        );
        cur += size_Wih;
        queue.writeBuffer(
            this.bufW_hh,
            0,
            flat.buffer,
            flat.byteOffset + cur * 4,
            size_Whh * 4,
        );
        cur += size_Whh;
        queue.writeBuffer(
            this.bufB_ih,
            0,
            flat.buffer,
            flat.byteOffset + cur * 4,
            size_b * 4,
        );
        cur += size_b;
        queue.writeBuffer(
            this.bufB_hh,
            0,
            flat.buffer,
            flat.byteOffset + cur * 4,
            size_b * 4,
        );
        cur += size_b;

        if (this.bidirectional) {
            queue.writeBuffer(
                this.bufW_ih,
                size_Wih * 4,
                flat.buffer,
                flat.byteOffset + cur * 4,
                size_Wih * 4,
            );
            cur += size_Wih;
            queue.writeBuffer(
                this.bufW_hh,
                size_Whh * 4,
                flat.buffer,
                flat.byteOffset + cur * 4,
                size_Whh * 4,
            );
            cur += size_Whh;
            queue.writeBuffer(
                this.bufB_ih,
                size_b * 4,
                flat.buffer,
                flat.byteOffset + cur * 4,
                size_b * 4,
            );
            cur += size_b;
            queue.writeBuffer(
                this.bufB_hh,
                size_b * 4,
                flat.buffer,
                flat.byteOffset + cur * 4,
                size_b * 4,
            );
            cur += size_b;
        }

        return cur - offset;
    }

    /**
     * Forward pass across sequence of length T on GPU
     *
     * @param {GPUBuffer} bufX - Input features buffer [T x inDim]
     * @param {number} T - Sequence length
     * @param {GPUBuffer|null} bufH0 - Optional initial state [H]
     * @param {GPUCommandEncoder|null} encoder - Optional command encoder
     * @returns {GPUBuffer} bufOutH - Output hidden states [T x outDim]
     */
    forward(bufX, T, bufH0 = null, encoder = null) {
        const isLocalEncoder = !encoder;
        const cmdEncoder = encoder ||
            this.device.createCommandEncoder({
                label: "gpu_bigru_forward_encoder",
            });

        const H = this.hiddenDim;
        const inDim = this.inDim;
        const outD = this.outDim;
        const dirCount = this.bidirectional ? 2 : 1;

        this.ensureScratchBuffers(T);
        const bufU = this.bufU;
        const bufOutH = this.bufOutH;

        // 1. Combined Input Pre-projection (Fwd + Bwd in single fused GEMM + Bias)
        this.gpuOps.dispatchMatmul(
            bufX,
            this.bufW_ih,
            bufU,
            T,
            dirCount * 3 * H,
            inDim,
            false,
            true,
            cmdEncoder,
            this.bufB_ih,
        );

        // 2. Recurrent Sequence Execution (Fwd and Bwd run in parallel across 2 workgroups)
        this.gpuOps.dispatchGruSequence(
            bufU,
            this.bufW_hh,
            this.bufB_hh,
            bufH0,
            bufOutH,
            T,
            H,
            outD,
            0,
            this.bidirectional ? 2 : 0,
            cmdEncoder,
        );

        if (isLocalEncoder) {
            this.device.queue.submit([cmdEncoder.finish()]);
        }

        return bufOutH;
    }
}
