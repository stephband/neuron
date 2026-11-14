import weakCache from "fn/weak-cache.js";
import { createStorageBuffer, createUniformBuffer } from "./buffer.js";
import { MATMUL_WGSL } from "./shaders/matmul.wgsl.js";
import { ADD_BIAS_WGSL, COPY_COLS_WGSL, SOFTMAX_WGSL } from "./shaders/elementwise.wgsl.js";
import { GRU_SEQUENCE_WGSL } from "./shaders/gru.wgsl.js";

const UNIFORM_RING_SIZE = 64;

/**
 * GPU Compute Operations Engine (Inference)
 *
 * Manages compiled inference compute pipelines and dispatches shaders.
 */
export default class GPUOps {
    static from = weakCache((device) => new GPUOps(device));

    constructor(device) {
        this.device = device;
        this.dummyStorageBuffer = createStorageBuffer(device, 16, "ops_dummy_storage");
        this.scratchU32 = new Uint32Array(8);
        this.uniformRing = new Array(UNIFORM_RING_SIZE);
        for (let i = 0; i < UNIFORM_RING_SIZE; i++) {
            this.uniformRing[i] = createUniformBuffer(device, this.scratchU32, `ops_uniform_${i}`);
        }
        this.uniformCursor = 0;

        // 1. Matrix Multiplication Pipeline (with optional fused bias)
        const matmulModule = this.device.createShaderModule({
            label: "matmul_module",
            code: MATMUL_WGSL,
        });
        this.matmulPipeline = this.device.createComputePipeline({
            label: "matmul_pipeline",
            layout: "auto",
            compute: { module: matmulModule, entryPoint: "matmul_tiled" },
        });
        this.matmulLayout = this.matmulPipeline.getBindGroupLayout(0);

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
        this.addBiasLayout = this.addBiasPipeline.getBindGroupLayout(0);

        // 3. Recurrent GRU Sequence Pipelines (32-thread warp fast path + 256-thread full path)
        const gruModule = this.device.createShaderModule({
            label: "gru_module",
            code: GRU_SEQUENCE_WGSL,
        });
        this.gruSequence32Pipeline = this.device.createComputePipeline({
            label: "gru_sequence_32_pipeline",
            layout: "auto",
            compute: { module: gruModule, entryPoint: "gru_sequence_32" },
        });
        this.gruSequence32Layout = this.gruSequence32Pipeline.getBindGroupLayout(0);
        this.gruSequencePipeline = this.device.createComputePipeline({
            label: "gru_sequence_pipeline",
            layout: "auto",
            compute: { module: gruModule, entryPoint: "gru_sequence" },
        });
        this.gruSequenceLayout = this.gruSequencePipeline.getBindGroupLayout(0);

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
        this.softmaxLayout = this.softmaxPipeline.getBindGroupLayout(0);

        // 5. Column-Slice Copy Pipeline (for layer input concatenation)
        const copyColsModule = this.device.createShaderModule({
            label: "copy_cols_module",
            code: COPY_COLS_WGSL,
        });
        this.copyColsPipeline = this.device.createComputePipeline({
            label: "copy_cols_pipeline",
            layout: "auto",
            compute: { module: copyColsModule, entryPoint: "main" },
        });
        this.copyColsLayout = this.copyColsPipeline.getBindGroupLayout(0);
    }

    /**
     * Writes uniform u32 parameters into the next ring uniform buffer without allocating.
     */
    acquireUniformBuffer(u0 = 0, u1 = 0, u2 = 0, u3 = 0, u4 = 0, u5 = 0, byteLength = 16) {
        const buf = this.uniformRing[this.uniformCursor];
        this.uniformCursor = (this.uniformCursor + 1) & (UNIFORM_RING_SIZE - 1);
        const s = this.scratchU32;
        s[0] = u0;
        s[1] = u1;
        s[2] = u2;
        s[3] = u3;
        s[4] = u4;
        s[5] = u5;
        s[6] = 0;
        s[7] = 0;
        this.device.queue.writeBuffer(buf, 0, s.buffer, 0, byteLength);
        return buf;
    }

    /**
     * Dispatches Tiled GEMM: C = A * B (+ optional bias)
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
        bufBias = null,
    ) {
        const isLocalEncoder = !encoder;
        const cmdEncoder = encoder ||
            this.device.createCommandEncoder({ label: "gemm_encoder" });

        const uniformBuffer = this.acquireUniformBuffer(
            M,
            N,
            K,
            transposeA ? 1 : 0,
            transposeB ? 1 : 0,
            bufBias ? 1 : 0,
            32,
        );

        const bindGroup = this.device.createBindGroup({
            layout: this.matmulLayout,
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: bufA } },
                { binding: 2, resource: { buffer: bufB } },
                { binding: 3, resource: { buffer: bufC } },
                { binding: 4, resource: { buffer: bufBias || this.dummyStorageBuffer } },
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

        const uniformBuffer = this.acquireUniformBuffer(rows, cols, 0, 0, 0, 0, 16);

        const bindGroup = this.device.createBindGroup({
            layout: this.addBiasLayout,
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
        if (H > 256) {
            throw new Error(`GRU hiddenDim ${H} exceeds maximum supported workgroup size 256`);
        }
        const isLocalEncoder = !encoder;
        const cmdEncoder = encoder ||
            this.device.createCommandEncoder({ label: "gru_sequence_encoder" });

        const hasH0 = bufH0 ? 1 : 0;
        const uniformBuffer = this.acquireUniformBuffer(
            T,
            H,
            outD,
            outOffset,
            direction,
            hasH0,
            32,
        );

        const h0Buffer = bufH0 || this.dummyStorageBuffer;
        const pipeline = H <= 32 ? this.gruSequence32Pipeline : this.gruSequencePipeline;
        const layout = H <= 32 ? this.gruSequence32Layout : this.gruSequenceLayout;

        const bindGroup = this.device.createBindGroup({
            layout,
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
            label: `gru_${direction === 0 ? "fwd" : (direction === 1 ? "bwd" : "bidi")}_pass`,
        });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(direction === 2 ? 2 : 1, 1, 1);
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

        const uniformBuffer = this.acquireUniformBuffer(rows, cols, 0, 0, 0, 0, 16);

        const bindGroup = this.device.createBindGroup({
            layout: this.softmaxLayout,
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

    /**
     * Dispatches Column-Slice Copy: dst[r, dstColOffset + c] = src[r, c]
     *
     * @param {GPUBuffer} bufSrc - Source matrix [rows x srcCols]
     * @param {GPUBuffer} bufDst - Destination matrix [rows x dstCols]
     * @param {number} rows - Number of rows (T)
     * @param {number} srcCols - Source column width
     * @param {number} dstCols - Destination column width
     * @param {number} dstColOffset - Column offset in destination buffer
     * @param {GPUCommandEncoder|null} encoder - Optional existing command encoder
     */
    dispatchCopyCols(
        bufSrc,
        bufDst,
        rows,
        srcCols,
        dstCols,
        dstColOffset = 0,
        encoder = null,
    ) {
        const isLocalEncoder = !encoder;
        const cmdEncoder = encoder ||
            this.device.createCommandEncoder({ label: "copy_cols_encoder" });

        const uniformBuffer = this.acquireUniformBuffer(rows, srcCols, dstCols, dstColOffset, 0, 0, 16);

        const bindGroup = this.device.createBindGroup({
            layout: this.copyColsLayout,
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: bufSrc } },
                { binding: 2, resource: { buffer: bufDst } },
            ],
        });

        const pass = cmdEncoder.beginComputePass({ label: "copy_cols_pass" });
        pass.setPipeline(this.copyColsPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(
            Math.ceil(srcCols / 16),
            Math.ceil(rows / 16),
            1,
        );
        pass.end();

        if (isLocalEncoder) {
            this.device.queue.submit([cmdEncoder.finish()]);
        }
    }
}
