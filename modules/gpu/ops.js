import weakCache from "fn/weak-cache.js";
import { createUniformBuffer } from "./buffer.js";
import { MATMUL_WGSL } from "./shaders/matmul.wgsl.js";
import { ADD_BIAS_WGSL, SOFTMAX_WGSL } from "./shaders/elementwise.wgsl.js";
import { GRU_SEQUENCE_WGSL } from "./shaders/gru.wgsl.js";

/**
 * GPU Compute Operations Engine (Inference)
 *
 * Manages compiled inference compute pipelines and dispatches shaders.
 */
export default class GPUOps {
    static from = weakCache((device) => new GPUOps(device));

    constructor(device) {
        this.device = device;

        // 1. Matrix Multiplication Pipeline
        const matmulModule = this.device.createShaderModule({
            label: "matmul_module",
            code: MATMUL_WGSL,
        });
        this.matmulPipeline = this.device.createComputePipeline({
            label: "matmul_pipeline",
            layout: "auto",
            compute: { module: matmulModule, entryPoint: "matmul_tiled" },
        });

        // 2. Broadcast Bias Addition Pipeline
        const addBiasModule = this.device.createShaderModule({
            label: "add_bias_module",
            code: ADD_BIAS_WGSL,
        });
        this.addBiasPipeline = this.device.createComputePipeline({
            label: "add_bias_pipeline",
            layout: "auto",
            compute: { module: addBiasModule, entryPoint: "main" },
        });

        // 3. Recurrent GRU Sequence Pipeline
        const gruModule = this.device.createShaderModule({
            label: "gru_module",
            code: GRU_SEQUENCE_WGSL,
        });
        this.gruSequencePipeline = this.device.createComputePipeline({
            label: "gru_sequence_pipeline",
            layout: "auto",
            compute: { module: gruModule, entryPoint: "gru_sequence" },
        });

        // 4. Softmax Forward Pipeline
        const softmaxModule = this.device.createShaderModule({
            label: "softmax_module",
            code: SOFTMAX_WGSL,
        });
        this.softmaxPipeline = this.device.createComputePipeline({
            label: "softmax_pipeline",
            layout: "auto",
            compute: { module: softmaxModule, entryPoint: "main" },
        });
    }

    /**
     * Dispatches Tiled GEMM: C = A * B (with optional transpose)
     */
    dispatchMatmul(
        bufA,
        bufB,
        bufC,
        M,
        N,
        K,
        transposeA = false,
        transposeB = false,
        encoder = null,
    ) {
        const isLocalEncoder = !encoder;
        const cmdEncoder = encoder ||
            this.device.createCommandEncoder({ label: "gemm_encoder" });

        const uniformData = new Uint32Array([
            M,
            N,
            K,
            transposeA ? 1 : 0,
            transposeB ? 1 : 0,
            0,
            0,
            0,
        ]);
        const uniformBuffer = createUniformBuffer(
            this.device,
            uniformData,
            "gemm_uniforms",
        );

        const bindGroup = this.device.createBindGroup({
            layout: this.matmulPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: bufA } },
                { binding: 2, resource: { buffer: bufB } },
                { binding: 3, resource: { buffer: bufC } },
            ],
        });

        const pass = cmdEncoder.beginComputePass({ label: "gemm_pass" });
        pass.setPipeline(this.matmulPipeline);
        pass.setBindGroup(0, bindGroup);

        const workgroupsX = Math.ceil(N / 16);
        const workgroupsY = Math.ceil(M / 16);
        pass.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
        pass.end();

        if (isLocalEncoder) {
            this.device.queue.submit([cmdEncoder.finish()]);
        }
    }

    /**
     * Dispatches Broadcast Bias Addition: Z = X + b
     */
    dispatchAddBias(bufX, bufB, bufZ, rows, cols, encoder = null) {
        const isLocalEncoder = !encoder;
        const cmdEncoder = encoder ||
            this.device.createCommandEncoder({ label: "bias_encoder" });

        const uniformBuffer = createUniformBuffer(
            this.device,
            new Uint32Array([rows, cols, 0, 0]),
        );

        const bindGroup = this.device.createBindGroup({
            layout: this.addBiasPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: bufX } },
                { binding: 2, resource: { buffer: bufB } },
                { binding: 3, resource: { buffer: bufZ } },
            ],
        });

        const pass = cmdEncoder.beginComputePass({ label: "add_bias_pass" });
        pass.setPipeline(this.addBiasPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(cols / 16), Math.ceil(rows / 16), 1);
        pass.end();

        if (isLocalEncoder) {
            this.device.queue.submit([cmdEncoder.finish()]);
        }
    }

    /**
     * Dispatches Recurrent GRU Sequence compute shader (Forward or Backward)
     *
     * @param {GPUBuffer} bufU - Pre-projected input sequence [T x 3*H]
     * @param {GPUBuffer} bufWhh - Recurrent weight matrix [3*H x H]
     * @param {GPUBuffer} bufBhh - Recurrent bias vector [3*H]
     * @param {GPUBuffer|null} bufH0 - Optional initial state [H]
     * @param {GPUBuffer} bufOutH - Output sequence buffer [T x outD]
     * @param {number} T - Sequence length
     * @param {number} H - Hidden dimension
     * @param {number} outD - Output stride (H or 2*H)
     * @param {number} outOffset - Output column offset (0 for forward, H for backward)
     * @param {number} direction - 0 = Forward (0 -> T-1), 1 = Backward (T-1 -> 0)
     * @param {GPUCommandEncoder|null} encoder - Optional existing command encoder
     */
    dispatchGruSequence(
        bufU,
        bufWhh,
        bufBhh,
        bufH0,
        bufOutH,
        T,
        H,
        outD,
        outOffset = 0,
        direction = 0,
        encoder = null,
    ) {
        const isLocalEncoder = !encoder;
        const cmdEncoder = encoder ||
            this.device.createCommandEncoder({ label: "gru_sequence_encoder" });

        const hasH0 = bufH0 ? 1 : 0;
        const uniformBuffer = createUniformBuffer(
            this.device,
            new Uint32Array([T, H, outD, outOffset, direction, hasH0, 0, 0]),
            "gru_uniforms",
        );

        const h0Buffer = bufH0 || bufBhh; // Placeholder buffer when hasH0 == 0

        const bindGroup = this.device.createBindGroup({
            layout: this.gruSequencePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: bufU } },
                { binding: 2, resource: { buffer: bufWhh } },
                { binding: 3, resource: { buffer: bufBhh } },
                { binding: 4, resource: { buffer: h0Buffer } },
                { binding: 5, resource: { buffer: bufOutH } },
            ],
        });

        const pass = cmdEncoder.beginComputePass({
            label: `gru_${direction === 0 ? "fwd" : "bwd"}_pass`,
        });
        pass.setPipeline(this.gruSequencePipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(1, 1, 1);
        pass.end();

        if (isLocalEncoder) {
            this.device.queue.submit([cmdEncoder.finish()]);
        }
    }

    /**
     * Dispatches Softmax Forward: probabilities = softmax(logits)
     *
     * @param {GPUBuffer} bufLogits - Input logits [rows x cols]
     * @param {GPUBuffer} bufProbs - Output probabilities [rows x cols]
     * @param {number} rows - Number of rows (batchSize / T)
     * @param {number} cols - Number of classes (numClasses)
     * @param {GPUCommandEncoder|null} encoder - Optional existing command encoder
     */
    dispatchSoftmax(bufLogits, bufProbs, rows, cols, encoder = null) {
        const isLocalEncoder = !encoder;
        const cmdEncoder = encoder ||
            this.device.createCommandEncoder({ label: "softmax_encoder" });

        const uniformBuffer = createUniformBuffer(
            this.device,
            new Uint32Array([rows, cols, 0, 0]),
        );

        const bindGroup = this.device.createBindGroup({
            layout: this.softmaxPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: bufLogits } },
                { binding: 2, resource: { buffer: bufProbs } },
            ],
        });

        const pass = cmdEncoder.beginComputePass({ label: "softmax_pass" });
        pass.setPipeline(this.softmaxPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(rows / 64), 1, 1);
        pass.end();

        if (isLocalEncoder) {
            this.device.queue.submit([cmdEncoder.finish()]);
        }
    }
}
