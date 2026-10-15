import { Matrix2D } from "../modules/matrix.js";
import { initWebGPU } from "../modules/gpu/context.js";
import { createStorageBuffer, readFromBuffer } from "../modules/gpu/buffer.js";
import GPUOps from "../modules/gpu/ops.js";

console.log("=== WebGPU Compute Pipeline Verification (Phase 2) ===\n");

// 1. Initialize WebGPU
const { device, adapter } = await initWebGPU();
console.log(
    `Initialized WebGPU Adapter: ${
        adapter.info.description || "Generic Adapter"
    }`,
);
const gpuOps = GPUOps.from(device);

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
        `  ✓ ${testName} passed (max difference: ${maxDiff.toExponential(3)})`,
    );
}

// 2. Test Tiled Matrix Multiplication (GEMM)
console.log("\n--- 1. Testing Tiled GEMM (C = A * B) ---");
const M = 64, K = 32, N = 48; // Non-square dimensions spanning multiple workgroups
const matA = Matrix2D.randomHe(M, K);
const matB = Matrix2D.randomHe(K, N);

// CPU Ground Truth
const matCcpu = Matrix2D.matmul(matA, matB);

// GPU Execution
const bufA = createStorageBuffer(device, matA.data, "buf_A");
const bufB = createStorageBuffer(device, matB.data, "buf_B");
const bufC = createStorageBuffer(device, M * N * 4, "buf_C");

gpuOps.dispatchMatmul(bufA, bufB, bufC, M, N, K);
const gpuC = await readFromBuffer(device, bufC, M * N * 4);
assertClose(gpuC, matCcpu.data, 1e-4, `Tiled GEMM [${M}x${K}] * [${K}x${N}]`);

// 3. Test Transposed Matrix Multiplications
console.log("\n--- 2. Testing Transposed GEMMs (Backprop operations) ---");
// Test A^T * B: A is [K x M], so A^T is [M x K], B is [K x N] -> C: [M x N]
const matA_for_transpose = Matrix2D.randomHe(K, M);
const cpuAt_B = Matrix2D.matmul(Matrix2D.transpose(matA_for_transpose), matB);
const bufA_trans = createStorageBuffer(
    device,
    matA_for_transpose.data,
    "buf_A_trans",
);
gpuOps.dispatchMatmul(bufA_trans, bufB, bufC, M, N, K, true, false);
const gpuAt_B = await readFromBuffer(device, bufC, M * N * 4);
assertClose(gpuAt_B, cpuAt_B.data, 1e-4, "Transpose A: C = A^T * B");

// Test A * B^T: A is [M x K], B_for_trans is [N x K], so B^T is [K x N] -> C: [M x N]
const matB_for_transpose = Matrix2D.randomHe(N, K);
const cpuA_Bt = Matrix2D.matmul(matA, Matrix2D.transpose(matB_for_transpose));
const bufB_trans = createStorageBuffer(
    device,
    matB_for_transpose.data,
    "buf_B_trans",
);
gpuOps.dispatchMatmul(bufA, bufB_trans, bufC, M, N, K, false, true);
const gpuA_Bt = await readFromBuffer(device, bufC, M * N * 4);
assertClose(gpuA_Bt, cpuA_Bt.data, 1e-4, "Transpose B: C = A * B^T");

// 4. Test Broadcast Bias Addition
console.log("\n--- 3. Testing Broadcast Bias Addition (Z = X + b) ---");
const matX = Matrix2D.randomHe(M, N);
const matBias = Matrix2D.randomHe(1, N);
const cpuZ = Matrix2D.addBias(matX, matBias);

const bufX = createStorageBuffer(device, matX.data, "buf_X");
const bufBias = createStorageBuffer(device, matBias.data, "buf_Bias");
const bufZ = createStorageBuffer(device, M * N * 4, "buf_Z");

gpuOps.dispatchAddBias(bufX, bufBias, bufZ, M, N);
const gpuZ = await readFromBuffer(device, bufZ, M * N * 4);
assertClose(gpuZ, cpuZ.data, 1e-5, "Broadcast Bias Addition: Z = X + b");

// 5. Test ReLU Forward and Backward
console.log("\n--- 4. Testing ReLU Forward & Backward ---");
const matReluIn = new Matrix2D(4, 4, [
    -1.5,
    2.0,
    -0.3,
    0.0,
    3.5,
    -2.1,
    0.8,
    -4.0,
    -0.1,
    1.2,
    -0.9,
    5.0,
    0.4,
    -0.4,
    2.2,
    -1.1,
]);
const cpuReluOut = new Float32Array(matReluIn.data.map((v) => Math.max(0, v)));

const bufReluIn = createStorageBuffer(device, matReluIn.data, "buf_relu_in");
const bufReluOut = createStorageBuffer(device, 16 * 4, "buf_relu_out");
gpuOps.dispatchReluForward(bufReluIn, bufReluOut, 4, 4);
const gpuReluOut = await readFromBuffer(device, bufReluOut, 16 * 4);
assertClose(gpuReluOut, cpuReluOut, 1e-5, "ReLU Forward");

// Backward
const matGradOut = new Matrix2D(4, 4, new Float32Array(16).fill(1.5));
const cpuGradIn = new Float32Array(16);
for (let i = 0; i < 16; i++) {
    cpuGradIn[i] = matReluIn.data[i] > 0 ? 1.5 : 0;
}

const bufGradOut = createStorageBuffer(device, matGradOut.data, "buf_grad_out");
const bufGradIn = createStorageBuffer(device, 16 * 4, "buf_grad_in");
gpuOps.dispatchReluBackward(bufGradOut, bufReluIn, bufGradIn, 4, 4);
const gpuGradIn = await readFromBuffer(device, bufGradIn, 16 * 4);
assertClose(gpuGradIn, cpuGradIn, 1e-5, "ReLU Backward");

console.log(
    "\nAll Phase 2 WebGPU Compute Shaders verified with 100% precision!",
);
