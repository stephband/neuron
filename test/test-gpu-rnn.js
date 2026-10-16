import { initWebGPU } from "../modules/gpu/context.js";
import { GpuBidirectionalGru } from "../modules/rnn/gpu-bigru.js";
import { createStorageBuffer, readFromBuffer } from "../modules/gpu/buffer.js";
import { BidirectionalGru } from "../modules/rnn/bigru.js";
import RNN from "../modules/rnn.js";
import RecurrentNetwork from "../modules/rnn/network-cpu.js";

console.log("=== WebGPU Recurrent Neural Network Verification ===\n");

// 1. Initialize WebGPU
const { device, adapter } = await initWebGPU();
console.log(
    `1. Initialized WebGPU Device: ${
        adapter.info.description || "Generic Adapter"
    }`,
);

// Helper to check numerical equivalence
function assertClose(actual, expected, tolerance = 1e-4, testName = "Test") {
    if (actual.length !== expected.length) {
        throw new Error(
            `${testName} failed: Length mismatch (${actual.length} vs ${expected.length})`,
        );
    }
    let maxDiff = 0;
    for (let i = 0; i < actual.length; i++) {
        const diff = Math.abs(actual[i] - expected[i]);
        if (diff > maxDiff) maxDiff = diff;
        if (diff > tolerance) {
            throw new Error(
                `${testName} failed at index ${i}: actual=${
                    actual[i]
                }, expected=${expected[i]}, diff=${diff}`,
            );
        }
    }
    console.log(
        `   ✓ ${testName} passed (max numerical difference: ${
            maxDiff.toExponential(3)
        })`,
    );
}

// 2. Verify Recurrent Layer: GpuBidirectionalGru vs CPU BidirectionalGru
console.log(
    "\n2. Verifying GpuBidirectionalGru recurrent transitions against CPU reference...",
);

const T = 7;
const inDim = 16;
const H = 48;

const cpuBigru = new BidirectionalGru(inDim, H);
const gpuBigru = new GpuBidirectionalGru(device, inDim, H, true);
gpuBigru.syncFromCpuLayer(cpuBigru);

const X = new Float32Array(T * inDim);
for (let i = 0; i < X.length; i++) X[i] = (Math.random() - 0.5) * 2;

const h0 = new Float32Array(H);
for (let i = 0; i < h0.length; i++) h0[i] = Math.random() - 0.5;

// CPU Forward
const cpuOutH = cpuBigru.forward(X, T, h0);

// GPU Forward
const bufX = createStorageBuffer(device, X, "test_bufX");
const bufH0 = createStorageBuffer(device, h0, "test_bufH0");
const bufGpuOutH = gpuBigru.forward(bufX, T, bufH0);
const gpuOutH = await readFromBuffer(device, bufGpuOutH, T * 2 * H * 4);

assertClose(gpuOutH, cpuOutH, 1e-4, `BiGRU [T=${T}, Din=${inDim}, H=${H}]`);

// 3. Verify Unidirectional Mode on GPU
console.log("\n3. Verifying Unidirectional GRU Mode on GPU...");
const cpuUniGru = new BidirectionalGru(inDim, H, false);
const gpuUniGru = new GpuBidirectionalGru(device, inDim, H, false);
gpuUniGru.syncFromCpuLayer(cpuUniGru);

const cpuUniOut = cpuUniGru.forward(X, T, h0);
const bufUniOut = gpuUniGru.forward(bufX, T, bufH0);
const gpuUniOut = await readFromBuffer(device, bufUniOut, T * H * 4);

assertClose(
    gpuUniOut,
    cpuUniOut,
    1e-4,
    `Unidirectional GRU [T=${T}, Din=${inDim}, H=${H}]`,
);

// 4. Verify End-to-End Hierarchical Multi-Head RNN on GPU against CPU
console.log(
    "\n4. Verifying End-to-End Generic RecurrentNetwork on GPU vs CPU...",
);

const rnnDefinition = {
    inDim: 16,
    layers: [
        { hiddenDim: 32, bidirectional: true },
        { hiddenDim: 32, bidirectional: true },
    ],
    priming: {
        inDim: 8,
        type: "categorical",
    },
    heads: [
        { name: "head_alpha", layerIndex: 0, outDim: 6 },
        { name: "head_beta", layerIndex: 1, outDim: 10 },
    ],
};

const cpuNet = new RecurrentNetwork(rnnDefinition);
const gpuNet = new RNN(device, rnnDefinition);
gpuNet.syncFromCpuNetwork(cpuNet);

const seqT = 5;
const seqX = new Float32Array(seqT * rnnDefinition.inDim);
for (let i = 0; i < seqX.length; i++) seqX[i] = (Math.random() - 0.5) * 2;

const conditionClass = 3;
const cpuOut = cpuNet.forward(seqX, seqT, { condition: conditionClass });
const gpuOut = await gpuNet.predict(seqX, seqT, { condition: conditionClass });

assertClose(
    gpuOut.head_alpha.probs,
    cpuOut.head_alpha.probs,
    1e-4,
    "Head Alpha Probs",
);
assertClose(
    gpuOut.head_beta.probs,
    cpuOut.head_beta.probs,
    1e-4,
    "Head Beta Probs",
);

for (let t = 0; t < seqT; t++) {
    if (gpuOut.head_alpha.preds[t] !== cpuOut.head_alpha.preds[t]) {
        throw new Error(
            `Head Alpha prediction mismatch at step ${t}: GPU=${
                gpuOut.head_alpha.preds[t]
            }, CPU=${cpuOut.head_alpha.preds[t]}`,
        );
    }
    if (gpuOut.head_beta.preds[t] !== cpuOut.head_beta.preds[t]) {
        throw new Error(
            `Head Beta prediction mismatch at step ${t}: GPU=${
                gpuOut.head_beta.preds[t]
            }, CPU=${cpuOut.head_beta.preds[t]}`,
        );
    }
}
console.log("   ✓ All head predictions match 100% between CPU and GPU!");

console.log(
    "\nAll WebGPU WGSL recurrent kernels and GPU networks verified with 100% precision!",
);
