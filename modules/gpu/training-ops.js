import weakCache from "fn/weak-cache.js";
import { createUniformBuffer } from "./buffer.js";
import GPUOps from "./ops.js";
import {
    RELU_BACKWARD_WGSL,
    RELU_FORWARD_WGSL,
} from "./shaders/elementwise.wgsl.js";
import { LOSS_WGSL } from "./shaders/loss.wgsl.js";
import { REDUCTION_WGSL } from "./shaders/reduction.wgsl.js";
import { OPTIMIZER_WGSL } from "./shaders/optimizer.wgsl.js";

/**
 * GPU Compute Operations Engine (Training + Inference)
 *
 * Extends GPUOps with forward/backward activation, loss, reduction, and
 * optimizer compute pipelines.
 */
export default class TrainingGPUOps extends GPUOps {
    static from = weakCache((device) => new TrainingGPUOps(device));

    constructor(device) {
        super(device);

        // 1. ReLU Forward Pipeline
        const reluFwdModule = this.device.createShaderModule({
            label: "relu_forward_module",
            code: RELU_FORWARD_WGSL,
        });
        this.reluForwardPipeline = this.device.createComputePipeline({
            label: "relu_forward_pipeline",
            layout: "auto",
            compute: { module: reluFwdModule, entryPoint: "main" },
        });

        // 2. ReLU Backward Pipeline
        const reluBwdModule = this.device.createShaderModule({
            label: "relu_backward_module",
            code: RELU_BACKWARD_WGSL,
        });
        this.reluBackwardPipeline = this.device.createComputePipeline({
            label: "relu_backward_pipeline",
            layout: "auto",
            compute: { module: reluBwdModule, entryPoint: "main" },
        });

        // 3. Softmax Cross-Entropy Loss & Gradient Pipeline
        const lossModule = this.device.createShaderModule({
            label: "loss_module",
            code: LOSS_WGSL,
        });
        this.lossPipeline = this.device.createComputePipeline({
            label: "loss_pipeline",
            layout: "auto",
            compute: {
                module: lossModule,
                entryPoint: "softmax_cross_entropy",
            },
        });

        // 4. Column Reduction Pipeline (for bias gradients)
        const reductionModule = this.device.createShaderModule({
            label: "reduction_module",
            code: REDUCTION_WGSL,
        });
        this.reduceColsPipeline = this.device.createComputePipeline({
            label: "reduction_pipeline",
            layout: "auto",
            compute: { module: reductionModule, entryPoint: "sum_columns" },
        });

        // 5. Optimizer Pipelines (SGD & Adam)
        const optModule = this.device.createShaderModule({
            label: "optimizer_module",
            code: OPTIMIZER_WGSL,
        });
        this.sgdPipeline = this.device.createComputePipeline({
            label: "sgd_pipeline",
            layout: "auto",
            compute: { module: optModule, entryPoint: "sgd_step" },
        });
        this.adamPipeline = this.device.createComputePipeline({
            label: "adam_pipeline",
            layout: "auto",
            compute: { module: optModule, entryPoint: "adam_step" },
        });
    }

    /**
     * Dispatches ReLU Forward: output = max(0, input)
     */
    dispatchReluForward(bufIn, bufOut, rows, cols, encoder = null) {
        const isLocalEncoder = !encoder;
        const cmdEncoder = encoder ||
            this.device.createCommandEncoder({ label: "relu_forward_encoder" });

        const uniformBuffer = createUniformBuffer(
            this.device,
            new Uint32Array([rows, cols, 0, 0]),
        );

        const bindGroup = this.device.createBindGroup({
            layout: this.reluForwardPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: bufIn } },
                { binding: 2, resource: { buffer: bufOut } },
            ],
        });

        const pass = cmdEncoder.beginComputePass({
            label: "relu_forward_pass",
        });
        pass.setPipeline(this.reluForwardPipeline);
        pass.setBindGroup(0, bindGroup);

        const total = rows * cols;
        pass.dispatchWorkgroups(Math.ceil(total / 64), 1, 1);
        pass.end();

        if (isLocalEncoder) {
            this.device.queue.submit([cmdEncoder.finish()]);
        }
    }

    /**
     * Dispatches ReLU Backward: gradIn = gradOut * (cachedIn > 0 ? 1 : 0)
     */
    dispatchReluBackward(
        bufGradOut,
        bufCachedIn,
        bufGradIn,
        rows,
        cols,
        encoder = null,
    ) {
        const isLocalEncoder = !encoder;
        const cmdEncoder = encoder ||
            this.device.createCommandEncoder({
                label: "relu_backward_encoder",
            });

        const uniformBuffer = createUniformBuffer(
            this.device,
            new Uint32Array([rows, cols, 0, 0]),
        );

        const bindGroup = this.device.createBindGroup({
            layout: this.reluBackwardPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: bufGradOut } },
                { binding: 2, resource: { buffer: bufCachedIn } },
                { binding: 3, resource: { buffer: bufGradIn } },
            ],
        });

        const pass = cmdEncoder.beginComputePass({
            label: "relu_backward_pass",
        });
        pass.setPipeline(this.reluBackwardPipeline);
        pass.setBindGroup(0, bindGroup);

        const total = rows * cols;
        pass.dispatchWorkgroups(Math.ceil(total / 64), 1, 1);
        pass.end();

        if (isLocalEncoder) {
            this.device.queue.submit([cmdEncoder.finish()]);
        }
    }

    /**
     * Dispatches Softmax Cross-Entropy Loss & Logit Gradient
     */
    dispatchSoftmaxLoss(
        bufLogits,
        bufTargets,
        bufGradLogits,
        bufProbs,
        bufSampleLoss,
        batchSize,
        numClasses,
        encoder = null,
    ) {
        const isLocalEncoder = !encoder;
        const cmdEncoder = encoder ||
            this.device.createCommandEncoder({ label: "loss_encoder" });

        const uniformBuffer = createUniformBuffer(
            this.device,
            new Uint32Array([batchSize, numClasses, 0, 0]),
        );

        const bindGroup = this.device.createBindGroup({
            layout: this.lossPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: bufLogits } },
                { binding: 2, resource: { buffer: bufTargets } },
                { binding: 3, resource: { buffer: bufGradLogits } },
                { binding: 4, resource: { buffer: bufProbs } },
                { binding: 5, resource: { buffer: bufSampleLoss } },
            ],
        });

        const pass = cmdEncoder.beginComputePass({ label: "loss_pass" });
        pass.setPipeline(this.lossPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(batchSize / 64), 1, 1);
        pass.end();

        if (isLocalEncoder) {
            this.device.queue.submit([cmdEncoder.finish()]);
        }
    }

    /**
     * Dispatches Column Sum Reduction (Bias Gradient)
     */
    dispatchReduceCols(
        bufGradOut,
        bufGradBias,
        batchSize,
        cols,
        encoder = null,
    ) {
        const isLocalEncoder = !encoder;
        const cmdEncoder = encoder ||
            this.device.createCommandEncoder({ label: "reduce_cols_encoder" });

        const uniformBuffer = createUniformBuffer(
            this.device,
            new Uint32Array([batchSize, cols, 0, 0]),
        );

        const bindGroup = this.device.createBindGroup({
            layout: this.reduceColsPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: bufGradOut } },
                { binding: 2, resource: { buffer: bufGradBias } },
            ],
        });

        const pass = cmdEncoder.beginComputePass({ label: "reduce_cols_pass" });
        pass.setPipeline(this.reduceColsPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(cols / 64), 1, 1);
        pass.end();

        if (isLocalEncoder) {
            this.device.queue.submit([cmdEncoder.finish()]);
        }
    }

    /**
     * Dispatches SGD with Momentum Parameter Update
     */
    dispatchSgdStep(
        bufGrads,
        bufVelocities,
        bufParamsData,
        count,
        lr,
        momentum,
        encoder = null,
    ) {
        const isLocalEncoder = !encoder;
        const cmdEncoder = encoder ||
            this.device.createCommandEncoder({ label: "sgd_encoder" });

        const rawBuf = new ArrayBuffer(16);
        const f32 = new Float32Array(rawBuf);
        const u32 = new Uint32Array(rawBuf);
        f32[0] = lr;
        f32[1] = momentum;
        u32[2] = count;
        u32[3] = 0;

        const uniformBuffer = createUniformBuffer(this.device, rawBuf);

        const bindGroup = this.device.createBindGroup({
            layout: this.sgdPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: bufGrads } },
                { binding: 2, resource: { buffer: bufVelocities } },
                { binding: 3, resource: { buffer: bufParamsData } },
            ],
        });

        const pass = cmdEncoder.beginComputePass({ label: "sgd_pass" });
        pass.setPipeline(this.sgdPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(count / 64), 1, 1);
        pass.end();

        if (isLocalEncoder) {
            this.device.queue.submit([cmdEncoder.finish()]);
        }
    }

    /**
     * Dispatches Adam Parameter Update
     */
    dispatchAdamStep(
        bufGrads,
        bufM,
        bufV,
        bufParamsData,
        count,
        lr,
        beta1,
        beta2,
        eps,
        stepCount,
        encoder = null,
    ) {
        const isLocalEncoder = !encoder;
        const cmdEncoder = encoder ||
            this.device.createCommandEncoder({ label: "adam_encoder" });

        const bc1 = 1.0 - Math.pow(beta1, stepCount);
        const bc2 = 1.0 - Math.pow(beta2, stepCount);

        const rawBuf = new ArrayBuffer(32);
        const f32 = new Float32Array(rawBuf);
        const u32 = new Uint32Array(rawBuf);
        f32[0] = lr;
        f32[1] = beta1;
        f32[2] = beta2;
        f32[3] = eps;
        f32[4] = bc1;
        f32[5] = bc2;
        u32[6] = count;
        u32[7] = 0;

        const uniformBuffer = createUniformBuffer(this.device, rawBuf);

        const bindGroup = this.device.createBindGroup({
            layout: this.adamPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: bufGrads } },
                { binding: 2, resource: { buffer: bufM } },
                { binding: 3, resource: { buffer: bufV } },
                { binding: 4, resource: { buffer: bufParamsData } },
            ],
        });

        const pass = cmdEncoder.beginComputePass({ label: "adam_pass" });
        pass.setPipeline(this.adamPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(count / 64), 1, 1);
        pass.end();

        if (isLocalEncoder) {
            this.device.queue.submit([cmdEncoder.finish()]);
        }
    }
}
