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
        this.device = device;
        this.gpuOps = GPUOps.from(device);
        this.inDim = inDim;
        this.hiddenDim = hiddenDim;
        this.bidirectional = bidirectional;
        this.outDim = bidirectional ? 2 * hiddenDim : hiddenDim;

        const H = hiddenDim;

        // GPU Buffers for Forward Cell Weights
        this.bufW_ih_fwd = createStorageBuffer(
            device,
            3 * H * inDim * 4,
            "gru_Wih_fwd",
        );
        this.bufB_ih_fwd = createStorageBuffer(
            device,
            3 * H * 4,
            "gru_bih_fwd",
        );
        this.bufW_hh_fwd = createStorageBuffer(
            device,
            3 * H * H * 4,
            "gru_Whh_fwd",
        );
        this.bufB_hh_fwd = createStorageBuffer(
            device,
            3 * H * 4,
            "gru_bhh_fwd",
        );

        // GPU Buffers for Backward Cell Weights (if bidirectional)
        if (bidirectional) {
            this.bufW_ih_bwd = createStorageBuffer(
                device,
                3 * H * inDim * 4,
                "gru_Wih_bwd",
            );
            this.bufB_ih_bwd = createStorageBuffer(
                device,
                3 * H * 4,
                "gru_bih_bwd",
            );
            this.bufW_hh_bwd = createStorageBuffer(
                device,
                3 * H * H * 4,
                "gru_Whh_bwd",
            );
            this.bufB_hh_bwd = createStorageBuffer(
                device,
                3 * H * 4,
                "gru_bhh_bwd",
            );
        }
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
            this.bufW_ih_fwd,
            0,
            flat.buffer,
            flat.byteOffset + cur * 4,
            size_Wih * 4,
        );
        cur += size_Wih;
        queue.writeBuffer(
            this.bufW_hh_fwd,
            0,
            flat.buffer,
            flat.byteOffset + cur * 4,
            size_Whh * 4,
        );
        cur += size_Whh;
        queue.writeBuffer(
            this.bufB_ih_fwd,
            0,
            flat.buffer,
            flat.byteOffset + cur * 4,
            size_b * 4,
        );
        cur += size_b;
        queue.writeBuffer(
            this.bufB_hh_fwd,
            0,
            flat.buffer,
            flat.byteOffset + cur * 4,
            size_b * 4,
        );
        cur += size_b;

        if (this.bidirectional) {
            queue.writeBuffer(
                this.bufW_ih_bwd,
                0,
                flat.buffer,
                flat.byteOffset + cur * 4,
                size_Wih * 4,
            );
            cur += size_Wih;
            queue.writeBuffer(
                this.bufW_hh_bwd,
                0,
                flat.buffer,
                flat.byteOffset + cur * 4,
                size_Whh * 4,
            );
            cur += size_Whh;
            queue.writeBuffer(
                this.bufB_ih_bwd,
                0,
                flat.buffer,
                flat.byteOffset + cur * 4,
                size_b * 4,
            );
            cur += size_b;
            queue.writeBuffer(
                this.bufB_hh_bwd,
                0,
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

        // Temporary intermediate buffers
        const bufU_mat_fwd = createStorageBuffer(
            this.device,
            T * 3 * H * 4,
            "temp_U_mat_fwd",
        );
        const bufU_fwd = createStorageBuffer(
            this.device,
            T * 3 * H * 4,
            "temp_U_fwd",
        );
        const bufOutH = createStorageBuffer(
            this.device,
            T * outD * 4,
            "gru_outH",
        );

        // 1. Forward Input Pre-projection: U_fwd = X * W_ih_fwd^T + b_ih_fwd
        this.gpuOps.dispatchMatmul(
            bufX,
            this.bufW_ih_fwd,
            bufU_mat_fwd,
            T,
            3 * H,
            inDim,
            false,
            true,
            cmdEncoder,
        );
        this.gpuOps.dispatchAddBias(
            bufU_mat_fwd,
            this.bufB_ih_fwd,
            bufU_fwd,
            T,
            3 * H,
            cmdEncoder,
        );

        // 2. Forward Recurrent Sequence Execution (t = 0 -> T-1)
        this.gpuOps.dispatchGruSequence(
            bufU_fwd,
            this.bufW_hh_fwd,
            this.bufB_hh_fwd,
            bufH0,
            bufOutH,
            T,
            H,
            outD,
            0, // offset 0
            0, // direction forward
            cmdEncoder,
        );

        // 3. Backward Recurrent Sequence Execution (if bidirectional)
        if (this.bidirectional) {
            const bufU_mat_bwd = createStorageBuffer(
                this.device,
                T * 3 * H * 4,
                "temp_U_mat_bwd",
            );
            const bufU_bwd = createStorageBuffer(
                this.device,
                T * 3 * H * 4,
                "temp_U_bwd",
            );

            this.gpuOps.dispatchMatmul(
                bufX,
                this.bufW_ih_bwd,
                bufU_mat_bwd,
                T,
                3 * H,
                inDim,
                false,
                true,
                cmdEncoder,
            );
            this.gpuOps.dispatchAddBias(
                bufU_mat_bwd,
                this.bufB_ih_bwd,
                bufU_bwd,
                T,
                3 * H,
                cmdEncoder,
            );

            this.gpuOps.dispatchGruSequence(
                bufU_bwd,
                this.bufW_hh_bwd,
                this.bufB_hh_bwd,
                null, // Backward pass starts at zero
                bufOutH,
                T,
                H,
                outD,
                H, // offset H
                1, // direction backward
                cmdEncoder,
            );
        }

        if (isLocalEncoder) {
            this.device.queue.submit([cmdEncoder.finish()]);
        }

        return bufOutH;
    }
}
